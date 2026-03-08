"""
medagentboard/medqa/multi_agent_healthcareagent.py

This file implements the HealthcareAgent framework as a standalone, end-to-end baseline.
It is inspired by the paper "Healthcare agent: eliciting the power of large language models for medical consultation".
The framework processes a single medical query through a multi-step pipeline involving planning,
preliminary analysis, internal safety review ("discuss"), and final response modification.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from openai import OpenAI
from tqdm import tqdm

# --- START: Added imports for observation mechanisms ---
# Ensure project root is in path to import shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods.ColaCare.medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from methods.ColaCare.medagentboard.utils.encode_image import encode_image
from methods.ColaCare.medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string
from methods.ColaCare.medagentboard.utils.keu import KEU # Mechanism 1: KEU Class
from methods.ColaCare.medagentboard.utils.analysishelper import AnalysisHelperLLM # Mechanism 4: Conflict Resolution Helper
# --- END: Added imports ---


# --- START: Classes adapted for observation mechanisms ---
class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    # Used by AuditorAgent
    GENERAL_MEDICINE = "General Medicine"
    SAFETY_ETHICS = "Safety and Ethics"
    EMERGENCY_MEDICINE = "Emergency Medicine"
    FACTUAL_ACCURACY = "Factual Accuracy"


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
                 model_key: str):
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
        """Call the LLM and return the full log."""
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False}
                )
                if not completion.choices or not completion.choices[0].message:
                    raise ValueError("Empty choices from LLM API")
                response = completion.choices[0].message.content or ""
                print(f"Agent {self.agent_id} received response snippet: {response[:80]}...")
                return response, system_message, user_message
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    error_message = f"LLM API call failed after {max_retries} attempts: {e}"
                    return error_message, system_message, user_message
                time.sleep(1)
        return "", system_message, user_message


class AuditorAgent(BaseAgent):
    """Auditor agent to implement observation mechanisms."""
    def __init__(self, agent_id: str = "auditor", model_key: str = "gemini-2.5-pro"):
        super().__init__(agent_id, AgentType.AUDITOR, model_key)

    def audit_domain_agent_contribution(self, question: str, agent_id: str, specialty: MedicalSpecialty, explanation: str) -> Dict[str, Any]:
        print(f"Auditor Agent: Auditing Domain Agent Contribution for {agent_id}...")
        system_message = {
            "role": "system",
            "content": """
You are an expert in medical epistemology and collaborative intelligence. Your task is to analyze an argument from a specialist AI doctor and assess two key dimensions of their contribution.

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
        user_message = {
            "role": "user",
            "content": f"Medical Question: \"{question}\"\n\nAgent: {agent_id} (Specialty: {specialty.value})\nArgument/Explanation:\n\"{explanation}\"\n\nPlease provide your audit in the specified JSON format."
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def audit_risk_and_quality(self, agent_id: str, explanation: str) -> Dict[str, Any]:
        print(f"Auditor Agent: Auditing Risk and Quality for {agent_id}'s argument...")
        system_message = {
            "role": "system",
            "content": """
You are a senior attending physician specializing in emergency medicine and patient triage. Your task is to analyze a medical argument and classify its implied **Diagnostic Urgency Level**.

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

    def audit_single_argument_quality(self, question: str, explanation: str) -> Dict[str, Any]:
        print("Auditor Agent: Auditing single argument's overall quality...")
        system_message = {
            "role": "system",
            "content": """
You are a lead physician and medical logician. Your task is to provide an **Overall Quality Category** for a given medical argument.

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
            "content": f"Medical Question: \"{question}\"\n\nArgument to Evaluate:\n\"{explanation}\"\n\nPlease provide the overall quality audit as a JSON object."
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def identify_critical_conflicts(self, contributions: List[Dict[str, Any]], context_description: str) -> List[Dict[str, Any]]:
        print(f"Auditor Agent: Identifying critical conflict points (CCPs) from {context_description}...")
        valid_contributions = [c for c in contributions if c.get("text", "").strip()]
        if not valid_contributions or len(valid_contributions) < 2:
            return []
        system_message = {
            "role": "system",
            "content": """
You are a meticulous and logical medical debate moderator. Your sole task is to read the provided arguments and identify direct, substantive contradictions about verifiable facts or core interpretations.

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
        user_message = {"role": "user", "content": context_text}
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            parsed_response = json.loads(preprocess_response_string(response_text))
            conflicts = parsed_response.get("conflicts", [])
            print(f"Auditor Agent: Found {len(conflicts)} critical conflict point(s).")
            return conflicts
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing CCP response from LLM.")
            return []

    def identify_key_evidential_units(self, question: str, all_keus: List[Dict]) -> Dict[str, bool]:
        print("Auditor Agent: Identifying KEY evidential units...")
        system_message = {
            "role": "system",
            "content": """
You are a senior medical expert with exceptional diagnostic acumen. Your task is to review a medical question and a list of evidential units (facts/findings) extracted from a case. You must determine which of these units are **KEY** to reaching a correct and well-supported conclusion.

A **KEY** evidential unit is one that is:
- Highly relevant and specific to the question.
- Likely to significantly influence the final diagnosis or answer.
- Not a trivial, generic, or background finding.

Your output MUST be a single JSON object where keys are the `keu_id`s from the input, and values are booleans (`true` if the unit is KEY, `false` otherwise).
Example: {"KEU-0": true, "KEU-1": false, "KEU-2": true}
"""
        }
        keu_list_text = "\n".join([f"- {keu['keu_id']}: \"{keu['content']}\"" for keu in all_keus])
        user_message = {
            "role": "user",
            "content": f"Medical Question: \"{question}\"\n\nList of Evidential Units:\n{keu_list_text}\n\nPlease provide your judgment on which of these are KEY units in the specified JSON format."
        }
        self.client.response_format = {"type": "json_object"}
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            key_status_map = json.loads(preprocess_response_string(response_text))
            return key_status_map
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing KEU key status response. Defaulting all to not key.")
            return {keu['keu_id']: False for keu in all_keus}
# --- END: Adapted classes ---


# --- Prompts adapted from the "Healthcare agent" paper's logic ---

PLANNER_PROMPT_TEMPLATE = """
Based on the provided medical query, determine the best initial course of action.
- If the query is ambiguous, lacks critical details for a safe conclusion, or would benefit from further clarification, choose 'INQUIRY'.
- If you have sufficient information to provide a confident and safe diagnosis or answer, choose 'DIAGOOSE'.

Medical Query:
Question: {question}
{options_text}
{image_text}

Respond with a single word: DIAGNOSE or INQUIRY.
"""

INQUIRY_PROMPT_TEMPLATE = """
You are a medical doctor analyzing a case. To form an accurate and safe conclusion for the query below, you need more information.
Generate a list of the top 3 most critical follow-up questions you would ask to better understand the situation.

Medical Query:
Question: {question}
{options_text}
{image_text}

Return a JSON object with a single key "questions" containing a list of strings.
Example: {{"questions": ["How long have you experienced this symptom?", "Is there any associated pain?"]}}
"""

# MODIFICATION START: Mechanism 1 is injected here.
# The prompt now requires a 'keus' field.
PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE = """
As a medical doctor, provide a preliminary analysis of the following case based on the available information.
{inquiry_context}

Your output MUST be a JSON object with three keys:
1. "explanation": Your detailed reasoning and diagnostic process.
2. "answer": Your conclusion. For multiple-choice questions, this must be ONLY the option letter (e.g., 'A', 'B').
3. "keus": A list of key evidential units. Each KEU in the list should be a string representing a single, verifiable piece of evidence from the case (e.g., 'A 2cm nodule is visible in the upper left lung lobe.', 'The patient's white blood cell count is 15,000/µL.').

Medical Query:
Question: {question}
{options_text}
{image_text}
"""
# MODIFICATION END

# --- Safety Module Prompts (The "Discuss" Phase) ---

# MODIFICATION START: Mechanisms 1 & 4 are injected.
# The prompts now receive KEU and CCP context.
SAFETY_ETHICS_PROMPT = """
As a safety supervisor specializing in medical ethics, review the following AI doctor's preliminary analysis.
Your primary task is to critique it for necessary disclaimers about its AI nature and the potential risks of its advice.

First, review the available evidence and identified conflicts:
---
Available Key Evidential Units (KEUs):
{keu_list_text}
---
Active Critical Conflict Points (CCPs) to consider:
{ccp_text}
---

Now, review the AI's preliminary response:
---
{preliminary_response}
---

Based on ALL the information above, provide your concise feedback for improvement if the response is lacking in ethical considerations or disclaimers. If it is adequate, state that.
Your feedback should be a JSON object with a single key "feedback".
"""

SAFETY_EMERGENCY_PROMPT = """
As a safety supervisor specializing in emergency medicine, review the following AI doctor's preliminary analysis.
Your primary task is to determine if the case involves any potentially serious or life-threatening symptoms that warrant an immediate warning.

First, review the available evidence and identified conflicts:
---
Available Key Evidential Units (KEUs):
{keu_list_text}
---
Active Critical Conflict Points (CCPs) to consider:
{ccp_text}
---

Now, review the AI's preliminary response:
---
{preliminary_response}
---

Based on ALL the information above, if you identify high-risk elements (e.g., KEU-3 suggests a critical condition), highlight them and strongly suggest adding a clear warning to seek immediate medical attention. If no such emergencies are apparent, state that.
Your feedback should be a JSON object with a single key "feedback".
"""

SAFETY_ERROR_PROMPT = """
As a safety supervisor specializing in factual accuracy, review the following AI doctor's preliminary analysis.
Your primary task is to identify potential factual errors, misinterpretations of the evidence (image/text), or logical contradictions.

First, review the available evidence and identified conflicts:
---
Available Key Evidential Units (KEUs):
{keu_list_text}
---
Active Critical Conflict Points (CCPs) to consider:
{ccp_text}
---

Now, review the AI's preliminary response:
---
{preliminary_response}
---

Based on ALL the information above, point out any potential errors by referencing specific KEUs or CCPs, and suggest corrections. If no errors are found, state that.
Your feedback should be a JSON object with a single key "feedback".
"""
# MODIFICATION END


# --- Final Modification Prompt (The "Modify" Phase) ---
# MODIFICATION START: Mechanisms 1 & 4 are injected.
# The final prompt now demands KEU citation and CCP resolution.
FINAL_MODIFICATION_PROMPT_TEMPLATE = """
You are a senior medical supervisor tasked with creating the final, definitive response.
Revise the preliminary analysis by thoughtfully incorporating the feedback from the internal safety review.

**1. Original Medical Query:**
Question: {question}
{options_text}
{image_text}

**2. Available Evidence and Conflicts:**
---
Available Key Evidential Units (KEUs):
{keu_list_text}
---
[ATTENTION] The following Critical Conflict Points (CCPs) were identified and MUST be addressed or resolved in your final explanation:
{ccp_text}
---

**3. Preliminary Analysis (Draft):**
{preliminary_response}

**4. Internal Safety Review Feedback:**
- Ethics & Disclaimer Feedback: {ethics_feedback}
- Emergency Situation Feedback: {emergency_feedback}
- Factual Error Feedback: {error_feedback}

**CRITICAL INSTRUCTION:**
Your task is to integrate the feedback to create a final, safe, and accurate response.
In your 'explanation', you **MUST selectively cite only the most pivotal KEU-IDs** that form the core basis of your conclusion (e.g., 'Based on the findings in KEU-1 and KEU-4...').
Your explanation **MUST** also explicitly acknowledge and resolve the identified CCPs.

The final output must be a single, polished JSON object with "explanation" and "answer" keys.
For multiple-choice questions, the 'answer' field must contain ONLY the option letter.

**Final Revised JSON Output:**
"""
# MODIFICATION END


class HealthcareAgentFramework:
    """
    A standalone framework that implements the HealthcareAgent methodology,
    enhanced with quantitative observation mechanisms.
    """

    def __init__(self, model_key: str, auditor_model_key: str, conflict_model_key: str):
        self.model_key = model_key

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
        
        # --- START: Initialize observer agents ---
        self.auditor_agent = AuditorAgent(model_key=auditor_model_key)
        self.analysis_llm = AnalysisHelperLLM(model_key=conflict_model_key)
        # --- END: Initialize observer agents ---

        print(f"Initialized HealthcareAgentFramework with model: {self.model_name}")

    def _call_llm(self,
                  prompt: str,
                  agent_id_for_log: str,
                  image_path: Optional[str] = None,
                  expect_json: bool = True,
                  max_retries: int = 3) -> Tuple[str, Dict]:
        """
        A helper function to call the LLM, returning a log object.
        """
        system_message = {"role": "system", "content": "You are a highly capable and meticulous medical AI assistant."}
        user_content = [{"type": "text", "text": prompt}]

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            base64_image = encode_image(image_path)
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        user_message = {"role": "user", "content": user_content}

        messages = [system_message, user_message]
        response_format = {"type": "json_object"} if expect_json else None

        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent '{agent_id_for_log}' calling LLM (JSON: {expect_json})...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format
                )
                if not completion.choices or not completion.choices[0].message:
                    raise ValueError("Empty choices from LLM API")
                response = completion.choices[0].message.content or ""
                print(f"LLM call successful. Response snippet: {response[:80]}...")
                
                log_data = {
                    "parsed_output_str": response,
                    "llm_input": {
                        "system_message": system_message,
                        "user_message": user_message
                    }
                }
                return response, log_data
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    error_msg = f"LLM API call failed after {max_retries} attempts."
                    error_log = {"error": error_msg, "llm_input": {"system_message": system_message, "user_message": user_message}}
                    return error_msg, error_log
                time.sleep(1)
        return "", {}

    def run_query(self, data_item: Dict) -> Dict:
        """
        Processes a single medical query through the full HealthcareAgent pipeline.
        """
        qid = data_item["qid"]
        question = data_item["question"]
        options = data_item.get("options")
        image_path = data_item.get("image_path")
        ground_truth = data_item.get("answer")

        print(f"\n{'='*20} Processing QID: {qid} with HealthcareAgentFramework {'='*20}")
        start_time = time.time()
        
        # --- START: Initialize audit trail for observation mechanisms ---
        audit_trail = {
            "keus": {},
            "collaboration_audits": {},
            "ccps": {} 
        }
        keu_counter = 0
        ccp_counter = 0
        all_unresolved_ccps = []
        # --- END: Initialize audit trail ---

        case_history = {
            "steps": [],
            "audit_trail": audit_trail
        }

        options_text = ""
        if options:
            options_text = "Options:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
        image_text = "An image is provided for context." if image_path else ""

        try:
            # === STEP 1: Planner Module ===
            planner_prompt = PLANNER_PROMPT_TEMPLATE.format(question=question, options_text=options_text, image_text=image_text)
            action, planner_log = self._call_llm(planner_prompt, "Planner", image_path, expect_json=False)
            action = action.strip().upper()
            case_history["steps"].append({"step": "1_Planner", "log": planner_log, "decision": action})

            # === STEP 2: Inquiry Module (Optional) ===
            inquiry_context = ""
            if "INQUIRY" in action:
                inquiry_prompt = INQUIRY_PROMPT_TEMPLATE.format(question=question, options_text=options_text, image_text=image_text)
                inquiry_response_str, inquiry_log = self._call_llm(inquiry_prompt, "InquiryAgent", image_path, expect_json=True)
                inquiry_result = json.loads(preprocess_response_string(inquiry_response_str))
                questions = inquiry_result.get("questions", [])
                case_history["steps"].append({"step": "2_Inquiry", "log": inquiry_log, "generated_questions": questions})
                if questions:
                    inquiry_context = "To provide a robust answer, the following questions should be considered:\n- " + "\n- ".join(questions)
                    inquiry_context += "\n\nGiven this, here is a preliminary analysis based on the limited information:"
            else:
                 case_history["steps"].append({"step": "2_Inquiry", "generated_questions": "Skipped as per planner's decision."})

            # === STEP 3: Preliminary Analysis (Domain Agent) ===
            analysis_prompt = PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE.format(inquiry_context=inquiry_context, question=question, options_text=options_text, image_text=image_text)
            preliminary_response_str, analysis_log = self._call_llm(analysis_prompt, "PreliminaryAnalyzer", image_path, expect_json=True)
            preliminary_result = json.loads(preprocess_response_string(preliminary_response_str))
            
            # --- START: Mechanism 1 & 3 Auditing ---
            # Mechanism 1: Extract and register KEUs
            for keu_content in preliminary_result.get("keus", []):
                keu_id = f"KEU-{keu_counter}"
                new_keu = KEU(keu_id=keu_id, content=keu_content, source_agent="PreliminaryAnalyzer", round_introduced=1)
                audit_trail["keus"][keu_id] = new_keu
                keu_counter += 1
            
            all_keus_for_audit = [{"keu_id": k, "content": v.content} for k,v in audit_trail["keus"].items()]
            if all_keus_for_audit:
                key_status_map = self.auditor_agent.identify_key_evidential_units(question, all_keus_for_audit)
                for keu_id, is_key in key_status_map.items():
                    if keu_id in audit_trail["keus"]:
                        audit_trail["keus"][keu_id].is_key = is_key

            # Mechanism 3: Audit the contribution
            prelim_explanation = preliminary_result.get("explanation", "")
            contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, "PreliminaryAnalyzer", MedicalSpecialty.GENERAL_MEDICINE, prelim_explanation)
            risk_audit = self.auditor_agent.audit_risk_and_quality("PreliminaryAnalyzer", prelim_explanation)
            audit_trail["collaboration_audits"]["preliminary_analysis"] = {**contribution_audit, **risk_audit}
            # --- END: Mechanism 1 & 3 Auditing ---
            
            analysis_log['parsed_output'] = preliminary_result
            case_history["steps"].append({"step": "3_Preliminary_Analysis", "log": analysis_log})

            # === STEP 4: Safety Module ("Discuss" Phase as Reviewers) ===
            keu_list_text = "\n".join([f"- {k}: '{v.content}'" for k, v in audit_trail["keus"].items()]) or "No KEUs were extracted."
            ccp_text = "No conflicts identified yet."
            
            # -- Ethics Review --
            ethics_prompt = SAFETY_ETHICS_PROMPT.format(preliminary_response=preliminary_response_str, keu_list_text=keu_list_text, ccp_text=ccp_text)
            ethics_feedback_str, ethics_log = self._call_llm(ethics_prompt, "SafetyEthicsAgent", expect_json=True)
            ethics_feedback = json.loads(preprocess_response_string(ethics_feedback_str)).get("feedback", "")
            risk_audit_ethics = self.auditor_agent.audit_risk_and_quality("SafetyEthicsAgent", ethics_feedback)
            audit_trail["collaboration_audits"]["ethics_review"] = risk_audit_ethics

            # -- Emergency Review --
            emergency_prompt = SAFETY_EMERGENCY_PROMPT.format(preliminary_response=preliminary_response_str, keu_list_text=keu_list_text, ccp_text=ccp_text)
            emergency_feedback_str, emergency_log = self._call_llm(emergency_prompt, "SafetyEmergencyAgent", expect_json=True)
            emergency_feedback = json.loads(preprocess_response_string(emergency_feedback_str)).get("feedback", "")
            risk_audit_emergency = self.auditor_agent.audit_risk_and_quality("SafetyEmergencyAgent", emergency_feedback)
            audit_trail["collaboration_audits"]["emergency_review"] = risk_audit_emergency

            # -- Error Review --
            error_prompt = SAFETY_ERROR_PROMPT.format(preliminary_response=preliminary_response_str, keu_list_text=keu_list_text, ccp_text=ccp_text)
            error_feedback_str, error_log = self._call_llm(error_prompt, "SafetyErrorAgent", expect_json=True)
            error_feedback = json.loads(preprocess_response_string(error_feedback_str)).get("feedback", "")
            risk_audit_error = self.auditor_agent.audit_risk_and_quality("SafetyErrorAgent", error_feedback)
            audit_trail["collaboration_audits"]["error_review"] = risk_audit_error
            
            case_history["steps"].append({
                "step": "4_Safety_Review",
                "ethics_feedback": {"log": ethics_log, "feedback": ethics_feedback},
                "emergency_feedback": {"log": emergency_log, "feedback": emergency_feedback},
                "error_feedback": {"log": error_log, "feedback": error_feedback}
            })
            
            # --- START: Mechanism 4: Identify conflicts after review ---
            review_contributions = [
                {'agent_id': 'PreliminaryAnalyzer', 'specialty': MedicalSpecialty.GENERAL_MEDICINE.value, 'text': prelim_explanation},
                {'agent_id': 'SafetyEthicsAgent', 'specialty': MedicalSpecialty.SAFETY_ETHICS.value, 'text': ethics_feedback},
                {'agent_id': 'SafetyEmergencyAgent', 'specialty': MedicalSpecialty.EMERGENCY_MEDICINE.value, 'text': emergency_feedback},
                {'agent_id': 'SafetyErrorAgent', 'specialty': MedicalSpecialty.FACTUAL_ACCURACY.value, 'text': error_feedback}
            ]
            new_ccps = self.auditor_agent.identify_critical_conflicts(review_contributions, "preliminary analysis vs. safety reviews")
            audit_trail["ccps"]["round_1"] = []
            for ccp in new_ccps:
                ccp['ccp_id'] = f"CCP-{ccp_counter}"
                ccp['status'] = 'unresolved'
                audit_trail["ccps"]["round_1"].append(ccp)
                ccp_counter += 1
            all_unresolved_ccps.extend(audit_trail["ccps"]["round_1"])
            # --- END: Mechanism 4 ---

            # === STEP 5: Final Modification ("Modify" Phase as Meta Agent) ===
            ccp_text_for_prompt = "\n".join([f"- {c['ccp_id']}: {c['conflict_summary']}" for c in all_unresolved_ccps]) or "No critical conflicts identified."
            
            final_prompt = FINAL_MODIFICATION_PROMPT_TEMPLATE.format(
                question=question, options_text=options_text, image_text=image_text,
                preliminary_response=preliminary_response_str,
                ethics_feedback=ethics_feedback,
                emergency_feedback=emergency_feedback,
                error_feedback=error_feedback,
                keu_list_text=keu_list_text,
                ccp_text=ccp_text_for_prompt
            )
            final_response_str, final_log = self._call_llm(final_prompt, "FinalModifier", image_path, expect_json=True)
            
            # --- START: Mechanism 1, 3, 4 Auditing on Final Step ---
            final_result_json = json.loads(preprocess_response_string(final_response_str))
            final_explanation = final_result_json.get("explanation", "")
            
            # Mechanism 1: Track KEU presence in final decision
            for keu_id, keu in audit_trail["keus"].items():
                if keu_id in final_explanation or keu.content in final_explanation:
                    keu.present_in_final_decision = True
            
            # Mechanism 3: Audit final decision quality and risk
            quality_audit_final = self.auditor_agent.audit_single_argument_quality(question, final_explanation)
            risk_audit_final = self.auditor_agent.audit_risk_and_quality("FinalModifier", final_explanation)
            audit_trail["collaboration_audits"]["final_decision"] = {**quality_audit_final, **risk_audit_final}

            # Mechanism 4: Check if conflicts were resolved
            for ccp in all_unresolved_ccps:
                was_addressed, reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, final_explanation)
                if was_addressed:
                    ccp['status'] = 'resolved'
                    ccp['resolution_reasoning'] = reasoning
            # --- END: Auditing Final Step ---
            
            final_log['parsed_output'] = final_result_json
            case_history["steps"].append({"step": "5_Final_Modification", "log": final_log})

            # === STEP 6: Parse Final Result ===
            predicted_answer = final_result_json.get("answer", "Parsing Error")
            explanation = final_result_json.get("explanation", "Parsing Error")

        except Exception as e:
            print(f"FATAL ERROR during query processing for QID {qid}: {e}")
            predicted_answer = "Framework Error"
            explanation = str(e)
            case_history["error"] = str(e)

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s. Final Answer: {predicted_answer}")

        # Serialize KEU objects before saving
        if "keus" in audit_trail and audit_trail["keus"]:
            audit_trail["keus"] = {k: v.to_dict() for k, v in audit_trail["keus"].items()}

        final_output = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": question,
            "options": options,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "case_history": case_history,
            "processing_time": processing_time
        }
        return final_output

def main():
    parser = argparse.ArgumentParser(description="Run HealthcareAgent Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True, help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--start", type=int, required=True, help="Number of initial samples to start from")
    parser.add_argument("--end", type=int, required=True, help="Number of initial samples to end at")

    parser.add_argument("--model", type=str, default="qwen-vl-max", help="Model key to use for all agent steps")
    # --- START: Added arguments for observer models ---
    parser.add_argument("--auditor_model", type=str, required=True, help="Model for AuditorAgent (e.g., gemini-2.5-pro).")
    parser.add_argument("--conflict_model", type=str, required=True, help="Model for conflict analysis (e.g., deepseek-reasoner).")
    # --- END: Added arguments ---
    args = parser.parse_args()

    method_name = f"HealthcareAgent_full_log_{time.strftime('%Y%m%d_%H%M%S')}"

    logs_dir = os.path.join("logs", "observation", "HealthcareAgent", args.dataset)
    os.makedirs(logs_dir, exist_ok=True)
    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    framework = HealthcareAgentFramework(
        model_key=args.model,
        auditor_model_key=args.auditor_model,
        conflict_model_key=args.conflict_model
    )

    for item in tqdm(data[args.start:args.end], desc=f"Running HealthcareAgent on {args.dataset}"):
        qid = item["qid"]
        result_path = os.path.join(logs_dir, f"{qid}-result.json")

        if os.path.exists(result_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            result = framework.run_query(item)
            save_json(result, result_path)
        except Exception as e:
            print(f"CRITICAL MAIN LOOP ERROR processing item {qid}: {e}")
            error_result = {"qid": qid, "error": str(e), "timestamp": int(time.time())}
            save_json(error_result, result_path)

if __name__ == "__main__":
    main()