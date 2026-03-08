# multi_agent_colacare_full_log.py
"""
medagentboard/medqa/multi_agent_colacare.py

This version has been modified to include extensive logging for detailed analysis
of the multi-agent collaboration process.
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
from methods.ColaCare.medagentboard.utils.encode_image import encode_image, encode_media_to_content_parts, is_video_file
from methods.ColaCare.medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string
from methods.ColaCare.medagentboard.utils.keu import KEU


class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    INTERNAL_MEDICINE = "Internal Medicine"
    SURGERY = "Surgery"
    RADIOLOGY = "Radiology"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"


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
        self.token_stats = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]

    # MODIFICATION START: Adjusted return type to include prompts for logging.
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
    # MODIFICATION END
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM (model: {self.model_name}), system message: {system_message['content'][:50]}...")
                print(f"Agent {self.agent_id} API base_url: {self.client.base_url}")
                
                # Try non-streaming first for better compatibility with vLLM
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[system_message, user_message],
                        max_tokens=2048,
                        stream=False,
                    )
                    if not completion.choices or not completion.choices[0].message:
                        raise ValueError("Empty choices from LLM API")
                    response = completion.choices[0].message.content or ""
                    usage = getattr(completion, "usage", None)
                    self.token_stats["num_llm_calls"] += 1
                    self.token_stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.token_stats["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
                    print(f"Agent {self.agent_id} received response (non-streaming): {response[:100] if response else 'empty'}...")
                    return response, system_message, user_message
                except Exception as e1:
                    print(f"Agent {self.agent_id} non-streaming failed: {e1}, trying streaming...")
                    # Fallback to streaming
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[system_message, user_message],
                        stream=True,
                        max_tokens=2048,
                    )
                    
                    response_chunks = []
                    for chunk in completion:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content is not None:
                                response_chunks.append(delta.content)

                    response = "".join(response_chunks)
                    self.token_stats["num_llm_calls"] += 1
                    print(f"Agent {self.agent_id} received response (streaming): {response[:100] if response else 'empty'}...")
                    return response, system_message, user_message
                    
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                if retries >= max_retries:
                    # MODIFICATION START: Return error information for logging.
                    error_message = f"LLM API call failed after {max_retries} attempts: {e}"
                    return error_message, system_message, user_message
                    # MODIFICATION END
                time.sleep(2)


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

    # MODIFICATION START: Changed return type to a dictionary for comprehensive logging.
    def analyze_case(self,
                     question: str,
                     options: Optional[Dict[str, str]] = None,
                     image_path: Optional[str] = None) -> Dict[str, Any]:
    # MODIFICATION END
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

        user_content = []

        if image_path:
            user_content.extend(encode_media_to_content_parts(image_path))

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
                "round": len(self.memory) // 2 + 1, # Each round includes an analysis and a review, hence the division by 2.
                "content": result
            })
            analysis_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
            }
            return analysis_log # logging, returning the complete system and user messages.
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
            return analysis_log # logging, returning the complete system and user messages.


    # MODIFICATION START: Changed return type to a dictionary for comprehensive logging.
    def review_synthesis(self,
                         question: str,
                         synthesis: Dict[str, Any],
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
                       f"Your output should be in JSON format, including 'agree' (boolean or 'yes'/'no'), 'reason' (rationale for your decision), "
                       f"and 'answer' (your suggested answer if you disagree; if you agree, you can repeat the synthesized answer) fields."
        }

        user_content = []

        if image_path:
            user_content.extend(encode_media_to_content_parts(image_path))

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

        text_content = {
            "type": "text",
            "text": f"Original question: {question_with_options}\n\n"
                    f"{own_analysis_text}"
                    f"{synthesis_text}\n\n"
                    f"Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
                    f"1. 'agree': 'yes'/'no'\n"
                    f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                    f"3. 'answer': Your supported answer (can be the synthesized answer if you agree, or your own suggested answer if you disagree)"
        }
        user_content.append(text_content)

        user_message = {
            "role": "user",
            "content": user_content,
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message) # logging, capturing the complete system and user messages

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
        return review_log # logging, returning the complete system and user messages


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-max-latest"):
        """
        Initialize a meta agent.
        """
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    # MODIFICATION START: Changed return type to a dictionary for comprehensive logging.
    def synthesize_opinions(self,
                            question: str,
                            doctor_opinions: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            current_round: int = 1,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    # MODIFICATION END
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

        user_message = {
            "role": "user",
            "content": f"Question: {question_with_options}\n\n"
                       f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
                       f"Please synthesize these opinions into a consensus view. Provide your synthesis in JSON format, including "
                       f"'explanation' (comprehensive reasoning) and 'answer' (clear conclusion) fields."
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
        return synthesis_log # logging, returning the complete system and user messages.

    def make_final_decision(self,
                            question: str,
                            doctor_reviews: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            current_synthesis: Dict[str, Any],
                            current_round: int,
                            max_rounds: int,
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
            formatted_review += f"Answer: {review.get('answer', '')}\n"
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

        user_message = {
            "role": "user",
            "content": f"Question: {question_with_options}\n\n"
                       f"{current_synthesis_text}\n\n"
                       f"Doctor Reviews on this synthesis:\n{reviews_text}\n\n"
                       f"History of Previous Rounds:\n{previous_syntheses_text}\n\n"
                       f"Based on all the information, please provide your {decision_type} decision. "
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
        return decision_log # logging, returning the complete system and user messages.


class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                 max_rounds: int = 3,
                 doctor_configs: List[Dict] = None,
                 meta_model_key: str = "qwen-max-latest"):
        """
        Initialize MDT consultation.
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.SURGERY, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.RADIOLOGY, "model_key": "qwen-vl-max"},
        ]
        # self.meta_model_key = meta_model_key

        self.doctor_agents = []
        for idx, config in enumerate(self.doctor_configs, 1):
            agent_id = f"doctor_{idx}"
            specialty = config["specialty"]
            model_key = config.get("model_key", "qwen-vl-max")
            doctor_agent = DoctorAgent(agent_id, specialty, model_key)
            self.doctor_agents.append(doctor_agent)

        self.meta_agent = MetaAgent("meta", meta_model_key)
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

        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": [], "decision": None} # Added a 'decision' field

            # Step 1: Each doctor analyzes the case
            doctor_opinion_parsed_outputs = []
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                opinion_log = doctor.analyze_case(question, options, image_path)
                doctor_opinion_parsed_outputs.append(opinion_log["parsed_output"])
                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": opinion_log # logging, store the entire log
                })
                print(f"Doctor {i+1} opinion: {opinion_log['parsed_output'].get('answer', '')}")

            # Step 2: Meta agent synthesizes opinions
            print("Meta agent synthesizing opinions")
            synthesis_log = self.meta_agent.synthesize_opinions(
                question, doctor_opinion_parsed_outputs, self.doctor_specialties,
                current_round, options
            )
            round_data["synthesis"] = synthesis_log # Store the entire log
            synthesis_parsed_output = synthesis_log["parsed_output"]
            print(f"Meta agent synthesis: {synthesis_parsed_output.get('answer', '')}")

            # Step 3: Doctors review synthesis
            doctor_review_parsed_outputs = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review_log = doctor.review_synthesis(question, synthesis_parsed_output, options, image_path)
                review_parsed_output = review_log["parsed_output"]
                doctor_review_parsed_outputs.append(review_parsed_output)
                round_data["reviews"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": review_log # Store the entire log
                })

                agrees = review_parsed_output.get('agree', False)
                all_agree = all_agree and agrees
                print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            # Step 4: Meta agent makes decision based on reviews
            decision_log = self.meta_agent.make_final_decision(
                question, doctor_review_parsed_outputs, self.doctor_specialties,
                synthesis_parsed_output, current_round, self.max_rounds, options
            )
            round_data["decision"] = decision_log # Store the decision log for this round

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
        
        final_decision_parsed = final_decision_log['parsed_output'] if final_decision_log else {}
        print(f"Final decision: {final_decision_parsed.get('answer', 'N/A')}")

        processing_time = time.time() - start_time

        case_history["final_decision_log"] = final_decision_log
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time

        # MODIFICATION START: Add final state of all agent memories to the log.
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
        # MODIFICATION END

        agg = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for agent in self.doctor_agents:
            for k in agg:
                agg[k] += agent.token_stats.get(k, 0)
        for k in agg:
            agg[k] += self.meta_agent.token_stats.get(k, 0)
        case_history["token_stats"] = agg

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


def process_input(item, doctor_configs=None, meta_model_key="qwen-max-latest"):
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
    parser.add_argument("--meta_model", type=str, required=True, help="Model used for meta agent")
    parser.add_argument("--doctor_models", nargs='+', required=True, help="Models used for doctor agents. Provide one model name per doctor.")
    parser.add_argument("--num", type=int, required=True, help="vqa=50 ,qa=100")
    args = parser.parse_args()

    # Using a timestamped method name for unique log directories
    method = f"ColaCare_full_log_{time.strftime('%Y%m%d_%H%M%S')}"

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")
    qa_type = args.qa_type
    print(f"QA Format: {qa_type}")

    qa_type_folder = "multiple_choice" if qa_type == "mc" else "free-form"
    logs_dir = os.path.join("logs", "medqa", dataset_name, qa_type_folder, method)
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

    for item in tqdm(data[:args.num], desc=f"Running MDT consultation on {dataset_name}"):
        qid = item["qid"]
        log_file_path = os.path.join(logs_dir, f"{qid}-result.json")

        if os.path.exists(log_file_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            full_case_history = process_input(
                item,
                doctor_configs=doctor_configs,
                meta_model_key=args.meta_model
            )

            final_decision_log = full_case_history.get("final_decision_log", {})
            print("Final decision log:", final_decision_log)
            final_decision_parsed = final_decision_log.get("parsed_output", {})
            print("Final decision parsed:", final_decision_parsed)
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
                "case_history": full_case_history # This now contains the full, detailed log
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