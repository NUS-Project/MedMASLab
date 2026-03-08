# multi_agent_colacare_full_log.py
"""
medagentboard/medqa/multi_agent_colacare.py

"""

from openai import OpenAI
import os
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import time
import argparse
from tqdm import tqdm
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods.ColaCare.medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from methods.ColaCare.medagentboard.utils.encode_image import encode_image
from methods.ColaCare.medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string
from methods.ColaCare.medagentboard.utils.keu import KEU
from methods.ColaCare.medagentboard.utils.analysishelper import AnalysisHelperLLM


class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    INTERNAL_MEDICINE = "Internal Medicine"
    SURGERY = "Surgery"
    RADIOLOGY = "Radiology"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    AUDITOR = "Auditor"

class BaseAgent:
    """Base class for all agents."""

    def __init__(self,
                 agent_id: str,
                 agent_type: AgentType,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor or Coordinator)
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]

    def call_llm(self,
                 system_message: Dict[str, str],
                 user_message: Dict[str, Any],
                 max_retries: int = 3) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing question and optional image
            max_retries: Maximum number of retry attempts

        Returns:
            A tuple containing:
            - LLM response text
            - The system message sent to the LLM
            - The user message sent to the LLM
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:50]}...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                    stream=True,
                )
                response_chunks = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)

                response = "".join(response_chunks)
                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response, system_message, user_message
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    error_message = f"LLM API call failed after {max_retries} attempts: {e}"
                    return error_message, system_message, user_message
                time.sleep(1)


class DoctorAgent(BaseAgent):
    """Doctor agent with a medical specialty."""

    def __init__(self,
                 agent_id: str,
                 specialty: MedicalSpecialty,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize a doctor agent.
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key)
        self.specialty = specialty
        print(f"Initializing {specialty.value} doctor agent, ID: {agent_id}, Model: {model_key}")

    def analyze_case(self,
                     question: str,
                     options: Optional[Dict[str, str]] = None,
                     image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a medical case.
        Returns:
            A dictionary containing full log of the analysis.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) analyzing case with model: {self.model_key}")

        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}. "
                       f"Analyze the medical case and provide your professional opinion on the question. "
                       f"Your output should be in JSON format, including 'explanation' (detailed reasoning) and "
                       f"'answer' (clear conclusion) fields."
        }

        if options:
            system_message["content"] += (
                f" For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                f"that best matches your conclusion. Be specific about which option you are selecting."
            )
        system_message["content"] += (
            f"Your output must be a JSON object with three fields: 'explanation' (your detailed reasoning), 'answer' (your final conclusion), "
            f"and 'keus' (a list of key evidential units). Each KEU in the list should be a string representing a single, verifiable piece of evidence "
            f"from the case (e.g., 'A 2cm nodule is visible in the upper left lung lobe.', 'The patient's white blood cell count is 15,000/µL.')."
        )

        user_content = []

        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        text_content = {
            "type": "text",
            "text": f"{question_with_options}\n\nProvide your analysis in JSON format, including 'explanation' and 'answer' fields."
        }
        user_content.append(text_content)

        user_message = {
            "role": "user",
            "content": user_content,
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1, # Each round includes analysis and review, hence this calculation
                "content": result
            })
            analysis_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
            }
            return analysis_log
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            analysis_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
            }
            return analysis_log


    def review_synthesis(self,
                         question: str,
                         synthesis: Dict[str, Any],
                         audit_trail: Dict[str, Any],
                         ccp_text: str = "",
                         options: Optional[Dict[str, str]] = None,
                         image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis.
        Returns:
            A dictionary containing full log of the review.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing synthesis with model: {self.model_key}")

        current_round = len(self.memory) // 2 + 1
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break

        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}, participating in round {current_round} of a multidisciplinary team consultation. "
                    f"Review the synthesis of multiple doctors' opinions and determine if you agree with the conclusion. "
                    f"Consider your previous analysis and the MetaAgent's synthesized opinion to decide whether to agree or provide a different perspective. "
                    f"Your output must be a JSON object, including:"
                    f"1. 'agree': boolean (true/false)."
                    f"2. 'current_viewpoint': Your current final answer after this review (e.g., 'A', 'B')."
                    f"3. 'viewpoint_changed': boolean, true if your 'current_viewpoint' is different from your initial analysis's answer."
                    f"4. 'justification_type': A string, must be one of ['evidence_based', 'consensus_based']. Choose 'evidence_based' if your decision is primarily driven by specific KEU facts. Choose 'consensus_based' if your decision is primarily to align with the synthesized opinion or majority view."
                    f"5. 'cited_references': A list of strings containing the KEU-IDs or Agent-IDs that influenced your decision."
                    f"6. 'reason': Your detailed textual explanation for your decision."
        }

        user_content = []

        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nAnswer: {own_analysis.get('answer', '')}\n\n"

        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested answer: {synthesis.get('answer', '')}"

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        text_content = {
            "type": "text",
            "text": f"Original question: {question_with_options}\n\n"
                    f"{own_analysis_text}"
                    f"Synthesized Opinion for Review:\n{synthesis_text}\n\n"
                    f"Available Key Evidential Units (KEUs):\n{keu_list_text}\n\n"
                    f"Available Critical Consensus Points (CCPs):\n{ccp_text}\n\n"
                    f"Pay attention to the potential conflicts (CCPs) listed above, as addressing them in your 'reason' field will strengthen your argument. "
                    f"Please provide your comprehensive review.\nYour 'reason' field MUST reference the KEU-IDs that support your decision. \n"
                    f"Your response MUST be a single JSON object, strictly adhering to the 6-field structure defined in your system instructions. "
                    f"Pay close attention to correctly populating 'viewpoints_changed', 'justification_type', and 'cited_references'."
        }
        user_content.append(text_content)

        user_message = {
            "role": "user",
            "content": user_content,
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} review successfully parsed")
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]
        except json.JSONDecodeError:
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}
            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "answer" not in result:
                if own_analysis and "answer" in own_analysis:
                    result["answer"] = own_analysis["answer"]
                else:
                    result["answer"] = synthesis.get("answer", "No answer provided")

        self.memory.append({
            "type": "review",
            "round": current_round,
            "content": result
        })

        review_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return review_log


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-max-latest"):
        """
        Initialize a meta agent.
        """
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                            question: str,
                            doctor_opinions: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            audit_trail: Dict[str, Any],
                            ccp_text: str = "",
                            current_round: int = 1,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Synthesize multiple doctors' opinions.
        Returns:
            A dictionary containing full log of the synthesis.
        """
        print(f"Meta agent synthesizing round {current_round} opinions with model: {self.model_key}")

        system_message = {
            "role": "system",
            "content": f"You are a medical consensus coordinator facilitating round {current_round} of a multidisciplinary team consultation. "
                       "Synthesize the opinions of multiple specialist doctors into a coherent analysis and conclusion. "
                       "Consider each doctor's expertise and perspective, and weigh their opinions accordingly. "
                       "Your output should be in JSON format, including 'explanation' (synthesized reasoning) and "
                       "'answer' (consensus conclusion) fields."
        }

        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                "that best represents the consensus view. Be specific about which option you are selecting."
            )

        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Answer: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)
        opinions_text = "\n".join(formatted_opinions)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        user_message = {
            "role": "user",
            "content": f"Question: {question_with_options}\n\n"
                       f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
                       f"Available KEUs (Key Evidential Units):\n{keu_list_text}\n"
                       f"Available CCPs (Critical Consensus Points):\n{ccp_text}\n\n"
                       f"Note that potential conflicts (CCPs) have been identified; a robust synthesis must acknowledge or resolve these points.\n\n"
                       f"**CRITICAL INSTRUCTION:** Your task is to synthesize these diverse opinions into a single, coherent analysis. In your 'explanation', you **MUST selectively cite only the most important KEU-IDs** that support your synthesized view. **DO NOT simply list all available KEUs.** Your goal is to demonstrate a deep understanding by building a new, consolidated argument from the strongest evidence (e.g., 'Synthesizing the specialists' views, the consensus leans towards X, primarily supported by the crucial findings in KEU-2 and KEU-5...').\n\n"
                       f"Provide your synthesis in JSON format, including 'explanation' (comprehensive reasoning) and 'answer' (clear conclusion) fields."
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent synthesis successfully parsed")
        except json.JSONDecodeError:
            print("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

        self.memory.append({
            "type": "synthesis",
            "round": current_round,
            "content": result
        })

        synthesis_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return synthesis_log

    def make_final_decision(self,
                            question: str,
                            doctor_reviews: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            current_synthesis: Dict[str, Any],
                            current_round: int,
                            max_rounds: int,
                            audit_trail: Dict[str, Any],
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a final decision based on doctor reviews.
        Returns:
            A dictionary containing full log of the decision.
        """
        print(f"Meta agent making round {current_round} decision with model: {self.model_key}")

        all_agree = all(review.get('agree', False) for review in doctor_reviews)
        reached_max_rounds = current_round >= max_rounds

        system_message = {
            "role": "system",
            "content": "You are a medical consensus coordinator making a final decision. "
        }
        if all_agree:
            system_message["content"] += "All doctors agree with your synthesis, generate a final report."
        elif reached_max_rounds:
            system_message["content"] += (
                f"Maximum number of discussion rounds ({max_rounds}) reached without full consensus. "
                f"Make a final decision using majority opinion approach."
            )
        else:
            system_message["content"] += (
                "Not all doctors agree with your synthesis, but a decision for the current round is needed."
            )

        system_message["content"] += (
            " Your output should be in JSON format, including 'explanation' (final reasoning) and "
            "'answer' (final conclusion) fields."
        )

        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                "that represents the final decision. Be specific about which option you are selecting."
            )

        formatted_reviews = []
        for i, (review, specialty) in enumerate(zip(doctor_reviews, doctor_specialties)):
            formatted_review = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_review += f"Agree: {'Yes' if review.get('agree', False) else 'No'}\n"
            formatted_review += f"Reason: {review.get('reason', '')}\n"
            formatted_review += f"Answer: {review.get('current_viewpoint', '')}\n"
            formatted_reviews.append(formatted_review)

        reviews_text = "\n".join(formatted_reviews)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        current_synthesis_text = (
            f"Current synthesized explanation: {current_synthesis.get('explanation', '')}\n"
            f"Current suggested answer: {current_synthesis.get('answer', '')}"
        )

        decision_type = "final" if all_agree or reached_max_rounds else "current round"

        previous_syntheses = []
        for i, mem in enumerate(self.memory):
            if mem["type"] == "synthesis" and mem["round"] < current_round:
                prev = f"Round {mem['round']} synthesis:\n"
                prev += f"Explanation: {mem['content'].get('explanation', '')}\n"
                prev += f"Answer: {mem['content'].get('answer', '')}"
                previous_syntheses.append(prev)

        previous_syntheses_text = "\n\n".join(previous_syntheses) if previous_syntheses else "No previous syntheses available."

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"
            
        user_message = {
            "role": "user",
            "content": f"Question: {question_with_options}\n\n"
                       f"{current_synthesis_text}\n\n"
                       f"Doctor Reviews on this synthesis:\n{reviews_text}\n\n"
                       f"{keu_list_text}\n\n"
                       f"**CRITICAL INSTRUCTION:** Your reasoning in the 'explanation' field must demonstrate synthesis, not just summarization. To do this, you **MUST selectively cite only the most pivotal KEU-IDs** that form the core basis of your conclusion. **DO NOT list or repeat all available KEUs.** Your task is to build an argument using the strongest evidence. (e.g., 'Based on the critical findings in KEU-1 and KEU-3, I conclude...').\n\n"
                       f"History of Previous Rounds:\n{previous_syntheses_text}\n\n"
                       f"Based on ALL available information presented above, provide your {decision_type} decision. Your explanation should be grounded in the evidence and reasoning from the synthesis and reviews. "
                       f"Your response must be in JSON format, including 'explanation' and 'answer' fields."
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent final decision successfully parsed")
        except json.JSONDecodeError:
            print("Meta agent final decision is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

        self.memory.append({
            "type": "decision",
            "round": current_round,
            "final": all_agree or reached_max_rounds,
            "content": result
        })

        decision_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return decision_log

class AuditorAgent(BaseAgent):
    def __init__(self, agent_id: str = "auditor", model_key: str = "gemini-2.5-pro"):
        super().__init__(agent_id, AgentType.AUDITOR, model_key) # Can reuse AUDITOR type
    def audit_domain_agent_contribution(self, question: str, agent_id: str, specialty: MedicalSpecialty, explanation: str) -> Dict[str, Any]:
        """
        Audits the contribution of a domain agent for specialty knowledge consistency and expertise relevance after it has spoken.
        """
        print(f"Auditor Agent: Auditing Domain Agent Contribution for {agent_id}...")
        system_message = {
            "role": "system",
            "content": """You are an expert in medical epistemology and collaborative intelligence. Your task is to analyze an argument from a specialist AI doctor and assess two key dimensions of their contribution.

You MUST provide a JSON object with two classifications:

1.  **`specialized_insight_emergence`**: Classify the degree to which the argument demonstrates the emergence of insights unique to the agent's assigned specialty, beyond general medical knowledge.
    - **"High"**: The reasoning presents a perspective, interpretation, or piece of knowledge that is highly specific to the assigned role and would likely not be offered by other specialists. It represents a unique, valuable contribution.
    - **"Medium"**: The reasoning contains some specialty-specific elements but is largely grounded in shared or overlapping medical knowledge.
    - **"Low"**: The reasoning is generic, lacks a distinct specialty perspective, and could have been generated by a generalist agent. No unique insight has emerged.
2.  **`expertise_relevance_category`**: Classify the relevance of this agent's specialty to the overall question.
    - **"Core"**: The specialty is central to diagnosing the problem.
    - **"Relevant"**: The specialty provides important, but not central, insights.
    - **"Ancillary"**: The specialty is only tangentially related.

Provide a concise `auditor_reasoning` explaining your choices.
"""

        }
        specialty_name = specialty.value if hasattr(specialty, 'value') else specialty

        user_message = {
            "role": "user",
            "content": f"Medical Question: \"{question}\"\n\n"
                       f"Agent: {agent_id} (Specialty: {specialty_name})\n"
                       f"Argument/Explanation:\n\"{explanation}\"\n\n"
                       f"Please provide your audit in the specified JSON format."
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def audit_risk_and_quality(self, agent_id: str, explanation: str) -> Dict[str, Any]:
        """
        Audits the risk-aversion category of an argument after any agent has spoken.
        """
        print(f"Auditor Agent: Auditing Risk and Quality for {agent_id}'s argument...")
        system_message = {
            "role": "system",
            "content": """You are a senior attending physician specializing in emergency medicine and patient triage. Your task is to analyze a medical argument and classify its implied **Diagnostic Urgency Level**.

This level reflects how quickly the argument suggests action should be taken, especially when faced with potentially high-risk diagnoses.

You MUST provide a JSON object with one classification:
- **`diagnostic_urgency_level`**:
  - **"Immediate (STAT)"**: The argument demands immediate, urgent action to investigate or rule out a high-risk, time-sensitive condition. This aligns with the 'worst-case-first' principle. (e.g., "The possibility of aortic dissection requires a STAT CT angiogram now.")
  - **"Standard (Routine)"**: The argument proposes a standard, routine diagnostic workup based on the most probable causes. It is diligent but not urgent. (e.g., "Let's order routine cardiac enzymes and an EKG.")
  - **"Delayed (Deferrable)"**: The argument suggests a passive or delayed course of action, such as "watchful waiting" or follow-up at a later date, downplaying the need for immediate investigation. (e.g., "Since this is likely musculoskeletal, let's have the patient follow up with their primary care physician next week.")

Provide a concise `auditor_reasoning` for your choice.
"""
        }
        user_message = {
            "role": "user",
            "content": f"Argument from Agent {agent_id}:\n\"{explanation}\"\n\nPlease provide your risk audit in the specified JSON format."
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def audit_overall_quality_for_decision(self, question: str, doctor_reviews: List[Dict[str, Any]], specialties: List[MedicalSpecialty]) -> List[Dict[str, Any]]:
        """
        Performs a comprehensive quality assessment of each domain agent's current argument before the meta agent makes a decision.
        """
        print("Auditor Agent: Auditing overall argument quality for decision-making...")
        system_message = {
            "role": "system",
            "content": """You are a lead physician and medical logician. Your task is to provide an **Overall Quality Category** for several arguments, to inform a final decision.

The Overall Quality considers all factors: logical soundness, evidence support, expertise relevance, and clinical safety.
- **"High"**: A very strong, reliable argument. It is logical, evidence-based, safe, and comes from a relevant perspective.
- **"Medium"**: A decent argument with some strengths but also notable weaknesses (e.g., logical gaps, ignores some risks).
- **"Low"**: A weak or dangerous argument that should be treated with caution.

For each doctor, you MUST provide a JSON object with:
1.  **`agent_id`**: The doctor's ID.
2.  **`overall_quality_category`**: "High", "Medium", or "Low".
3.  **`auditor_reasoning`**: A concise justification.

Your final output MUST be a JSON list of these objects.
"""
        }
        arguments_text = ""
        for i, review in enumerate(doctor_reviews):
            # 1. Safely retrieve specialty information
            specialty_name = "N/A"
            if i < len(specialties) and specialties[i]:
                specialty = specialties[i]
                specialty_name = specialty.value if hasattr(specialty, 'value') else specialty
            
            # 2. Directly get agent_id from review data instead of constructing it
            agent_id = review.get('agent_id', f'agent_{i+1}') # Provide a fallback ID

            supported_answer = review.get('current_viewpoint', review.get('answer', 'N/A'))

            reasoning = review.get('reason', review.get('reasoning', 'N/A'))
            
            arguments_text += f"\n---\nAgent ID: {agent_id} (Specialty: {specialty_name}):\n"
            arguments_text += f"Supported Answer: {supported_answer}\n"
            arguments_text += f"Reasoning: {reasoning}\n"
        
        user_message = { "role": "user", "content": f"Medical Question: {question}\n\nArguments:\n{arguments_text}\n\nPlease provide the overall quality audit as a JSON list." }
        
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return []

    def audit_single_argument_quality(self, question: str, explanation: str) -> Dict[str, Any]:
          """
          Performs a comprehensive quality assessment on a single argument, particularly for the meta agent's final decision.
          """
          print("Auditor Agent: Auditing single argument's overall quality...")
          system_message = {
              "role": "system",
              "content": """You are a lead physician and medical logician. Your task is to provide an **Overall Quality Category** for a given medical argument.

  The Overall Quality considers all factors: logical soundness, evidence support, and clinical safety.
  - **"High"**: A very strong, reliable argument. It is logical, evidence-based, safe, and provides a comprehensive justification.
  - **"Medium"**: A decent argument with some strengths but also notable weaknesses (e.g., logical gaps, ignores some risks, superficial reasoning).
  - **"Low"**: A weak or dangerous argument that should be treated with caution.

  You MUST provide a JSON object with:
  1.  **`overall_quality_category`**: "High", "Medium", or "Low".
  2.  **`auditor_reasoning`**: A concise justification for your rating.
  """
          }
          user_message = {
              "role": "user",
              "content": f"Medical Question: \"{question}\"\n\n"
                         f"Argument to Evaluate:\n\"{explanation}\"\n\n"
                         f"Please provide the overall quality audit as a JSON object."
          }
          response_text, _, _ = self.call_llm(system_message, user_message)
          try:
              return json.loads(preprocess_response_string(response_text))
          except (json.JSONDecodeError, TypeError):
              return {}

    def identify_critical_conflicts(self,
                                    contributions: List[Dict[str, Any]],
                                    context_description: str) -> List[Dict[str, Any]]:
        """
        Identifies critical conflict points from a series of text contributions.
        
        Args:
            contributions: A list of dictionaries, each containing 'agent_id', 'specialty', and 'text'.
            context_description: A string to describe the context in the prompt (e.g., "doctors' initial analyses", "doctors' review reasons").
        """
        print(f"Auditor Agent: Identifying critical conflict points (CCPs) from {context_description}...")

        # Filter out contributions with empty text
        valid_contributions = [c for c in contributions if c.get("text", "").strip()]
        
        # If there are no valid contributions, return an empty list directly.
        if not valid_contributions:
            return []

        system_message = {
            "role": "system",
            "content": """You are a meticulous and logical medical debate moderator. Your sole task is to read the provided arguments and identify direct, substantive contradictions about verifiable facts or core interpretations.

    You MUST ignore minor differences in phrasing. Focus only on clear conflicts (e.g., Feature A is present vs. Feature A is absent; Diagnosis X is likely vs. Diagnosis X is unlikely).

    Your final output MUST be a single JSON object containing a single key: "conflicts". The value of "conflicts" must be a list of conflict objects. 
    Each conflict object must have the following structure:
    - "conflicting_agents": A list of the agent_ids involved in this specific conflict.
    - "conflict_summary": A brief, one-sentence summary of the core disagreement.
    - "conflicting_statements": A list of objects, each detailing the specific statement, with keys "agent_id" and "statement_content".

    If there are no conflicts, return a JSON object with an empty list: {"conflicts": []}.
    """
        }

        context_text = f"Please analyze the following {context_description} for conflicts:\n\n"
        for contrib in valid_contributions:
            context_text += f"--- Argument from {contrib['agent_id']} ({contrib.get('specialty', 'N/A')}) ---\n"
            context_text += f"{contrib['text']}\n\n"

        user_message = { "role": "user", "content": context_text }

        response_text, _, _ = self.call_llm(system_message, user_message)
        
        try:
            parsed_response = json.loads(preprocess_response_string(response_text))
            conflicts = parsed_response.get("conflicts", [])
            print(f"Auditor Agent: Found {len(conflicts)} critical conflict point(s).")
            return conflicts
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing CCP response from LLM.")
            return []        
    def identify_key_evidential_units(self,
                                      question: str,
                                      doctor_opinions: List[Dict[str, Any]],
                                      doctor_agents: List[DoctorAgent],
                                      all_keus: List[Dict]) -> Dict[str, bool]:
        """
        Determines which of the proposed KEUs are "key" evidence by evaluating them against the doctors' initial analyses.

        Args:
            question: The main medical question.
            doctor_opinions: The list of parsed opinion outputs from each doctor.
            doctor_agents: The list of doctor agent instances to get their specialties.
            all_keus: A list of dictionaries, each with 'keu_id' and 'content'.

        Returns:
            A dictionary mapping keu_id to a boolean (True if it's key, False otherwise).
        """
        print("Auditor Agent: Identifying KEY evidential units with full context...")
        
        system_message = {
            "role": "system",
            "content": """You are a senior medical expert with exceptional diagnostic acumen. Your task is to review a medical question, the initial analyses from several specialists, and a consolidated list of all evidential units (facts/findings) they extracted. 

Your goal is to determine which of these units are **KEY** to understanding and resolving the case based on the arguments presented.

A **KEY** evidential unit is one that is:
- Directly foundational to a doctor's primary conclusion.
- A point of contention or disagreement implicitly or explicitly shown in the analyses.
- Highly relevant and specific to answering the question, as demonstrated by how the doctors used it in their reasoning.
- Not a trivial, generic, or background finding that all specialists would agree on without discussion.

Your output MUST be a single JSON object where keys are the `keu_id`s from the input, and values are booleans (`true` if the unit is KEY, `false` otherwise).
Example: {"KEU-0": true, "KEU-1": false, "KEU-2": true}
"""
        }

        # Build context from doctors' analyses
        opinions_context = "Here are the initial analyses from the specialists:\n\n"
        for i, opinion in enumerate(doctor_opinions):
            agent = doctor_agents[i]
            opinions_context += f"--- Analysis from {agent.agent_id} ({agent.specialty.value}) ---\n"
            opinions_context += f"Explanation: {opinion.get('explanation', 'N/A')}\n"
            opinions_context += f"Answer: {opinion.get('answer', 'N/A')}\n\n"

        keu_list_text = "\n".join([f"- {keu['keu_id']}: \"{keu['content']}\"" for keu in all_keus])

        user_message = {
            "role": "user",
            "content": f"**Medical Question:**\n \"{question}\"\n\n"
                       f"**Doctors' Analyses:**\n{opinions_context}"
                       f"**Consolidated List of All Evidential Units to Evaluate:**\n{keu_list_text}\n\n"
                       f"Based on the doctors' analyses, please provide your judgment on which of these are KEY units in the specified JSON format."
        }

        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            key_status_map = json.loads(preprocess_response_string(response_text))
            # Ensure all KEUs have a boolean value
            for keu in all_keus:
                if keu['keu_id'] not in key_status_map:
                    key_status_map[keu['keu_id']] = False
            return key_status_map
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing KEU key status response. Defaulting all to not key.")
            return {keu['keu_id']: False for keu in all_keus}        
class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                 max_rounds: int = 3,
                 doctor_configs: List[Dict] = None,
                 meta_model_key: str = "qwen-max-latest",
                 auditor_model_key: str = "gemini-2.5-pro",
                 conflict_analysis_model_key: str = "gemini-2.5-pro"):
        """
        Initialize MDT consultation.
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.SURGERY, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.RADIOLOGY, "model_key": "qwen-vl-max"},
        ]
        self.meta_model_key = meta_model_key

        self.doctor_agents = []
        for idx, config in enumerate(self.doctor_configs, 1):
            agent_id = f"doctor_{idx}"
            specialty = config["specialty"]
            model_key = config.get("model_key", "qwen-vl-max")
            doctor_agent = DoctorAgent(agent_id, specialty, model_key)
            self.doctor_agents.append(doctor_agent)

        self.meta_agent = MetaAgent("meta", meta_model_key)
        self.auditor_agent = AuditorAgent("auditor", auditor_model_key)

        # Initialize AnalysisHelperLLM to be available throughout the consultation process.
        self.analysis_llm = AnalysisHelperLLM(model_key=conflict_analysis_model_key)

        self.doctor_specialties = [doctor.specialty for doctor in self.doctor_agents]

        doctor_info_parts = []
        for config in self.doctor_configs:
            model_name = config.get('model_key', 'default')
            specialty_name = config['specialty'].value
            doctor_info_parts.append(f"{specialty_name} ({model_name})")
        doctor_info = ", ".join(doctor_info_parts)
        print(f"Initialized MDT consultation, max_rounds={max_rounds}, doctors: [{doctor_info}], meta_model={meta_model_key}")

    def run_consultation(self,
                         qid: str,
                         question: str,
                         options: Optional[Dict[str, str]] = None,
                         image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the MDT consultation process.
        Returns:
            A dictionary containing the complete case history and final result.
        """
        start_time = time.time()
        print(f"Starting MDT consultation for case {qid}")
        print(f"Question: {question}")
        if options:
            print(f"Options: {options}")

        case_history = {"rounds": []}
        current_round = 0
        final_decision_log = None
        consensus_reached = False
        decision_log = None # Initialize decision_log to handle cases with no rounds    
        audit_trail = {
            "keus": {},  # Dict[str, KEU]
            "viewpoints": {doc.agent_id: [] for doc in self.doctor_agents},
            "collaboration_audits": {}, # New, more structured audit records
            "ccps": {} # critical conflict points
        }
        all_unresolved_ccps = []
        ccp_counter = 0
        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": [], "decision": None} # Added a 'decision' field

            # Step 1: Each doctor analyzes the case
            doctor_opinion_parsed_outputs = []

            # Initialize the audit trail for Key Evidential Units (KEUs).
            if current_round == 1:
                keu_counter = 0

            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                opinion_log = doctor.analyze_case(question, options, image_path)
                parsed_output = opinion_log["parsed_output"]
                explanation = parsed_output.get("explanation", "")

                # Audit the domain agent's specialty-specific knowledge activation and risk aversion levels.
                contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, doctor.agent_id, doctor.specialty, explanation)
                risk_audit = self.auditor_agent.audit_risk_and_quality(doctor.agent_id, explanation)
                step_id = f"round_1_analysis_{doctor.agent_id}"
                audit_trail["collaboration_audits"][step_id] = {**contribution_audit, **risk_audit}
                doctor_opinion_parsed_outputs.append(parsed_output)

                # Record the initial viewpoint immediately after analyze_case.
                if current_round == 1:
                    initial_viewpoint_entry = {
                        "step": f"round_{current_round}_analysis",
                        "viewpoint": parsed_output.get("answer"),
                        "viewpoint_changed": False,
                        "justification_type": "initial_analysis",
                        # In the initial analysis, references are to the KEUs the agent found itself
                        "cited_references": [f"KEU-{idx}" for idx, content in enumerate(parsed_output.get("keus", []))]
                    }
                    audit_trail["viewpoints"][doctor.agent_id].append(initial_viewpoint_entry)
                # Extract Key Evidential Units.
                if current_round == 1 and "keus" in parsed_output:
                    for keu_content in parsed_output["keus"]:
                        keu_id = f"KEU-{keu_counter}"
                        # Register a KEU object for each piece of evidence.
                        new_keu = KEU(
                            keu_id=keu_id,
                            content=keu_content,
                            source_agent=doctor.agent_id,
                            round_introduced=current_round
                        )
                        audit_trail["keus"][keu_id] = new_keu
                        keu_counter += 1
                    # Call the auditor agent to identify which KEUs are key.
                    all_keus_for_audit = [{"keu_id": k, "content": v.content} for k, v in audit_trail["keus"].items()]
                    if all_keus_for_audit:
                        # Pass the doctors' opinions (doctor_opinion_parsed_outputs) as additional context.
                        key_status_map = self.auditor_agent.identify_key_evidential_units(
                            question,
                            doctor_opinion_parsed_outputs,
                            self.doctor_agents,
                            all_keus_for_audit
                        )
                        for keu_id, is_key in key_status_map.items():
                            if keu_id in audit_trail["keus"]:
                                audit_trail["keus"][keu_id].is_key = is_key

                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": opinion_log
                })
                print(f"Doctor {i+1} opinion: {opinion_log['parsed_output'].get('answer', '')}")
            
            # Identify critical conflict points after the initial analysis.
            if current_round == 1: # Initial conflicts are typically identified only after the first round.

                initial_contributions = []
                for i, opinion in enumerate(doctor_opinion_parsed_outputs):
                    agent = self.doctor_agents[i]
                    # Combine 'explanation' and 'keus' to form the complete viewpoint text.
                    full_text = opinion.get('explanation', '')
                    keus = opinion.get('keus', [])
                    if keus:
                        full_text += "\nKey Evidential Units:\n- " + "\n- ".join(keus)
                    
                    initial_contributions.append({
                        'agent_id': agent.agent_id,
                        'specialty': agent.specialty.value,
                        'text': full_text
                    })

                initial_ccps = self.auditor_agent.identify_critical_conflicts(
                    initial_contributions,
                    context_description="doctors' initial analyses"
                )
                audit_trail["ccps"][current_round] = []
                for ccp in initial_ccps:
                    ccp['ccp_id'] = f"CCP-{ccp_counter}" # Assign a unique ID.
                    ccp['round_identified'] = 1
                    ccp['status'] = 'unresolved'
                    ccp['round_resolved'] = None
                    audit_trail["ccps"][current_round].append(ccp)
                    ccp_counter += 1
                all_unresolved_ccps.extend(audit_trail["ccps"][current_round])

            ccp_text_for_prompt = ""
            if all_unresolved_ccps:
                ccp_text_for_prompt += "\n\n[ATTENTION] The following Critical Conflict Points (CCPs) from previous rounds remain UNRESOLVED and MUST be addressed:\n"
                for ccp in all_unresolved_ccps:
                    ccp_text_for_prompt += f"- CCP ID: {ccp['ccp_id']} (Identified in Round {ccp['round_identified']})\n"
                    ccp_text_for_prompt += f"  Conflict: {ccp['conflict_summary']}\n"
                    ccp_text_for_prompt += f"  Involved Agents: {', '.join(ccp['conflicting_agents'])}\n"


            # Step 2: Meta agent synthesizes opinions
            print("Meta agent synthesizing opinions")
            synthesis_log = self.meta_agent.synthesize_opinions(
                question, doctor_opinion_parsed_outputs, self.doctor_specialties,
                audit_trail=audit_trail, ccp_text=ccp_text_for_prompt, current_round=current_round, options=options
            )
            round_data["synthesis"] = synthesis_log
            synthesis_parsed_output = synthesis_log["parsed_output"]
            synthesis_explanation = synthesis_parsed_output.get("explanation", "")
            print(f"Meta agent synthesis: {synthesis_parsed_output.get('answer', '')}")

            # Audit the meta agent's risk aversion level.
            synthesis_risk_audit = self.auditor_agent.audit_risk_and_quality(self.meta_agent.agent_id, synthesis_explanation)
            step_id = f"round_{current_round}_synthesis"
            audit_trail["collaboration_audits"][step_id] = synthesis_risk_audit


            # Track the presence and citation of KEUs in the synthesis.
            for keu_id, keu in audit_trail["keus"].items():
                if keu_id in synthesis_explanation or keu.content in synthesis_explanation:
                    keu.present_in_synthesis[current_round] = True
                    keu.cited_by.append({
                        "agent_id": self.meta_agent.agent_id,
                        "round": current_round,
                        "action": "synthesis"
                    })
                else:
                    keu.present_in_synthesis[current_round] = False
            # Step 3: Doctors review synthesis
            doctor_review_parsed_outputs = []
            # Initialize the viewpoint tracker to ensure every doctor's viewpoint is recorded.
            if "viewpoints" not in audit_trail:
                audit_trail["viewpoints"] = {doc.agent_id: [] for doc in self.doctor_agents}
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review_log = doctor.review_synthesis(question, synthesis_parsed_output, audit_trail=audit_trail, ccp_text=ccp_text_for_prompt, options=options, image_path=image_path)
                review_parsed_output = review_log["parsed_output"]
                review_reason = review_parsed_output.get("reason", "")
                cited_refs = review_parsed_output.get("cited_references", [])

                # Audit the domain agent's expertise relevance, specialty knowledge activation, and risk aversion.
                contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, doctor.agent_id, doctor.specialty, review_reason)
                risk_audit = self.auditor_agent.audit_risk_and_quality(doctor.agent_id, review_reason)
                
                step_id = f"round_{current_round}_review_{doctor.agent_id}"
                audit_trail["collaboration_audits"][step_id] = {**contribution_audit, **risk_audit}

                # Check for citations and rebuttals of KEUs during the review process.
                for keu_id, keu in audit_trail["keus"].items():
                    # Check if cited.
                    if keu_id in cited_refs or keu_id in review_reason or keu.content in review_reason:
                        keu.cited_by.append({
                            "agent_id": doctor.agent_id,
                            "round": current_round,
                            "action": "review"
                        })
                    # Check if rebutted (using a simple keyword heuristic here; precise analysis is done in a later stage).
                    # The main rebuttal logic is placed in analyze_failures.py.
                    if not review_parsed_output.get("agree", True):
                        # If a doctor disagrees with the synthesis and mentions a KEU in their reasoning, we preliminarily mark it as a potential rebuttal.
                        if keu_id in review_reason or keu.content in review_reason:
                             keu.rebuttals.append({
                                 "agent_id": doctor.agent_id,
                                 "round": current_round,
                                 "reason": review_reason
                             })

                doctor_review_parsed_outputs.append(review_parsed_output)
                round_data["reviews"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": review_log
                })
                # Record viewpoint changes after the review.
                review_viewpoint_entry = {
                    "step": f"round_{current_round}_review",
                    "viewpoint": review_parsed_output.get("current_viewpoint"),
                    "viewpoint_changed": review_parsed_output.get("viewpoint_changed", False),
                    "justification_type": review_parsed_output.get("justification_type", "unknown"),
                    "cited_references": review_parsed_output.get("cited_references", []),
                }
                audit_trail["viewpoints"][doctor.agent_id].append(review_viewpoint_entry)
                agrees = review_parsed_output.get('agree', False)
                all_agree = all_agree and agrees
                print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            
            # Before the meta agent's decision, audit the overall quality of the arguments.
            overall_quality_audit = self.auditor_agent.audit_overall_quality_for_decision(
                question, doctor_review_parsed_outputs, self.doctor_specialties
            )
            audit_trail["collaboration_audits"][f"round_{current_round}_pre_decision_quality"] = overall_quality_audit


            # After each review round, update and identify conflicts.
            # Collect all discussion text from this round.
            round_discussion_text = round_data["synthesis"]["parsed_output"]["explanation"]
            for review in round_data["reviews"]:
                round_discussion_text += "\n" + review["log"]["parsed_output"].get("reason", "")
            
            # Determine if outstanding conflicts have been resolved.
            resolved_this_round_log = []
            still_unresolved = []
            for ccp in all_unresolved_ccps:
                was_addressed, resolution_reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, round_discussion_text)
                if was_addressed:
                    ccp['status'] = 'resolved'
                    ccp['round_resolved'] = current_round
                    ccp['resolution_reasoning'] = resolution_reasoning  # Store the reasoning
                    resolved_this_round_log.append(ccp)
                else:
                    still_unresolved.append(ccp)
            all_unresolved_ccps = still_unresolved
            print(f"Round {current_round}: {len(resolved_this_round_log)} CCP(s) were resolved.")
            # Identify new conflicts based on this round's review texts.
            # Note: The input here should be the review texts, as new conflicts often arise from disagreements with the synthesis.
            review_contributions = []
            for review_log in round_data["reviews"]:
                review_contributions.append({
                    'agent_id': review_log["doctor_id"],
                    'specialty': review_log["specialty"],
                    'text': review_log["log"]["parsed_output"].get("reason", "")
                })

            # Unconditionally call conflict identification.
            new_ccps = self.auditor_agent.identify_critical_conflicts(
                review_contributions,
                context_description="doctors' review reasons"
            )

            # 4. Add newly identified conflicts to the tracking list.
            if current_round not in audit_trail["ccps"]:
                audit_trail["ccps"][current_round] = []
                
            for ccp in new_ccps:
                ccp['ccp_id'] = f"CCP-{ccp_counter}"
                ccp['round_identified'] = current_round
                ccp['status'] = 'unresolved'
                ccp['round_resolved'] = None
                audit_trail["ccps"][current_round].append(ccp)
                ccp_counter += 1

            all_unresolved_ccps.extend(new_ccps)

            # Step 4: Meta agent makes decision based on reviews
            decision_log = self.meta_agent.make_final_decision(
                question, doctor_review_parsed_outputs, self.doctor_specialties,
                synthesis_parsed_output, current_round, self.max_rounds, audit_trail, options=options
            )
            
            round_data["decision"] = decision_log # Store the decision log for this round
            decision_explanation = decision_log.get("parsed_output", {}).get("explanation", "")
            
            # Audit the quality of the meta agent's decision argument.
            decision_quality_audit = self.auditor_agent.audit_single_argument_quality(question, decision_explanation)
            # Audit the risk aversion category of the meta agent's decision.
            decision_risk_audit = self.auditor_agent.audit_risk_and_quality(self.meta_agent.agent_id, decision_explanation)
            step_id = f"round_{current_round}_decision"
            audit_trail["collaboration_audits"][step_id] = {**decision_risk_audit, **decision_quality_audit}

            case_history["rounds"].append(round_data)

            if all_agree:
                consensus_reached = True
                final_decision_log = decision_log
                print("Consensus reached")
            else:
                print("No consensus reached, continuing to next round")
                if current_round == self.max_rounds:
                    final_decision_log = decision_log

        if not final_decision_log:
            final_decision_log = decision_log

        # Track the presence of KEUs in the final decision.
        if final_decision_log:
            final_explanation = final_decision_log.get("parsed_output", {}).get("explanation", "")
            for keu_id, keu in audit_trail["keus"].items():
                if keu_id in final_explanation or keu.content in final_explanation:
                    keu.present_in_final_decision = True
                    
        final_decision_parsed = final_decision_log['parsed_output'] if final_decision_log else {}
        print(f"Final decision: {final_decision_parsed.get('answer', 'N/A')}")

        processing_time = time.time() - start_time
        if "keus" in audit_trail and audit_trail["keus"]:
            serializable_keus = {keu_id: keu.to_dict()
                                 for keu_id, keu in audit_trail["keus"].items()}
            audit_trail["keus"] = serializable_keus
        case_history["final_decision_log"] = final_decision_log
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time
        case_history['audit_trail'] = audit_trail

        agent_final_states = {
            "meta_agent": {
                "id": self.meta_agent.agent_id,
                "memory": self.meta_agent.memory
            },
            "doctor_agents": [
                {
                    "id": doc.agent_id,
                    "specialty": doc.specialty.value,
                    "memory": doc.memory
                } for doc in self.doctor_agents
            ]
        }
        case_history["agent_final_states"] = agent_final_states

        return case_history


def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    Parse LLM response to extract structured output as a fallback.
    """
    try:
        parsed = json.loads(preprocess_response_string(response_text))
        return parsed
    except json.JSONDecodeError:
        lines = response_text.strip().split('\n')
        result = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace("'", "").replace('"', '')
                value = value.strip()
                result[key] = value

        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result


def process_input(item, doctor_configs=None, meta_model_key="qwen-max-latest",auditor_model_key="gemini-2.5-pro",conflict_analysis_model_key="deepseek-reasoner"):
    """
    Process a single input data item.
    """
    qid = item.get("qid")
    question = item.get("question")
    options = item.get("options")
    image_path = item.get("image_path")

    mdt = MDTConsultation(
        max_rounds=3,
        doctor_configs=doctor_configs,
        meta_model_key=meta_model_key,
        auditor_model_key=auditor_model_key,
        conflict_analysis_model_key = conflict_analysis_model_key
    )

    result_history = mdt.run_consultation(
        qid=qid,
        question=question,
        options=options,
        image_path=image_path,
    )
    return result_history


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name,like PathVQA,VQA-RAD")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], default="mc", help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--meta_model", type=str, required=True, help="deepseek-v3-official")
    parser.add_argument("--doctor_models", nargs='+', required=True, help="deepseek-v3-official if not processing image, or gemini-2.5-flash")
    parser.add_argument("--start", type=int, required=True, help="Start index for processing")
    parser.add_argument("--end", type=int, required=True, help="End index for processing")
    parser.add_argument("--auditor_model", type=str, required=True, help="gemini-2.5-pro")
    parser.add_argument("--conflict_model", type=str, required=True,default="deepseek-reasoner", help="Model for conflict analysis (AnalysisHelperLLM).")
    args = parser.parse_args()

    # Using a timestamped method name for unique log directories
    method = f"ColaCare_full_log_{time.strftime('%Y%m%d_%H%M%S')}"

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")
    qa_type = args.qa_type
    print(f"QA Format: {qa_type}")

    qa_type_folder = "multiple_choice" if qa_type == "mc" else "free-form"
    logs_dir = os.path.join("logs", "observation",'ColaCare', dataset_name)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")

    data_path = f"./my_datasets/processed/medqa/{dataset_name}/medqa_{qa_type}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    doctor_specialties = [
        MedicalSpecialty.INTERNAL_MEDICINE,
        MedicalSpecialty.SURGERY,
        MedicalSpecialty.RADIOLOGY
    ]

    if len(args.doctor_models) > len(doctor_specialties):
        print(f"Warning: More doctor models ({len(args.doctor_models)}) provided than specialties ({len(doctor_specialties)}). Extra models will not be used.")

    doctor_configs = []
    num_doctors_to_configure = min(len(args.doctor_models), len(doctor_specialties))
    for i in range(num_doctors_to_configure):
        doctor_configs.append({
            "specialty": doctor_specialties[i],
            "model_key": args.doctor_models[i]
        })

    doctor_model_names = [config["model_key"] for config in doctor_configs]
    print(f"Configuring {len(doctor_configs)} doctors with models: {doctor_model_names}")

    for item in tqdm(data[args.start:args.end], desc=f"Running MDT consultation on {dataset_name}"):
        qid = item["qid"]
        log_file_path = os.path.join(logs_dir, f"{qid}-result.json")

        if os.path.exists(log_file_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            full_case_history = process_input(
                item,
                doctor_configs=doctor_configs,
                meta_model_key=args.meta_model,
                auditor_model_key=args.auditor_model,
                conflict_analysis_model_key=args.conflict_model
            )

            final_decision_log = full_case_history.get("final_decision_log", {})
            final_decision_parsed = final_decision_log.get("parsed_output", {})
            predicted_answer = final_decision_parsed.get("answer", "Error: No answer found")
            print(f"Predicted answer for {qid}: {predicted_answer}")

            item_result = {
                "qid": qid,
                "timestamp": int(time.time()),
                "question": item["question"],
                "options": item.get("options"),
                "image_path": item.get("image_path"),
                "ground_truth": item.get("answer"),
                "predicted_answer": predicted_answer,
                "case_history": full_case_history, # This now contains the full, detailed log
            }

            save_json(item_result, log_file_path)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            # Optionally, save an error log
            error_log = {
                "qid": qid,
                "error": str(e)
            }
            save_json(error_log, os.path.join(logs_dir, f"{qid}-error.json"))


if __name__ == "__main__":
    main()