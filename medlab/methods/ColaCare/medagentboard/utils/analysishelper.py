import os
import json
from glob import glob
from typing import Tuple, Dict

from tqdm import tqdm
from openai import OpenAI
# Assume your LLM configurations and preprocessing functions are in an importable utils file.
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import preprocess_response_string

class AnalysisHelperLLM:
    """
    A helper class that uses an LLM to perform complex analysis tasks,
    such as determining if a conflict has been resolved.
    """
    def __init__(self, model_key: str):
        """
        Initializes the analysis helper.

        Args:
            model_key: The key for the LLM model to use.
        """
        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
        print(f"Initialized AnalysisHelperLLM with model: {self.model_name}")

    def check_if_rebutted(self, keu_to_check: Dict, case_history: Dict) -> bool:
        """
        Determines if a Key Evidential Unit (KEU) was effectively rebutted during the discussion.

        Args:
            keu_to_check: The KEU dictionary from the audit_trail.
            case_history: The full case history containing the discussion.

        Returns:
            True if the KEU was effectively rebutted, False otherwise.
        """
        system_message = {
            "role": "system",
            "content": """You are an expert debate judge and a sharp medical logician. Your task is to determine if a specific 'Claim' (a Key Evidential Unit from a minority opinion) was *effectively rebutted* during a medical discussion.

Definition of 'Effectively Rebutted':
- A simple disagreement ('I don't agree with KEU-5') is NOT a rebuttal.
- A rebuttal requires providing specific counter-evidence or a logical argument that directly invalidates or seriously challenges the claim.
- Example:
  - Claim: 'The image shows a clear fracture in the tibia.'
  - Effective Rebuttal: 'The line identified as a fracture is actually a nutrient canal, which is a normal anatomical feature. This is confirmed by its oblique orientation and sclerotic borders, unlike a typical fracture line.'
  - Ineffective Rebuttal: 'I have checked the image and I do not see a fracture.'

You MUST respond with a single JSON object with one key: `is_rebutted` (boolean: true or false).
"""
        }

        # Construct the full discussion context
        discussion_text = ""
        for i, round_data in enumerate(case_history.get("rounds", [])):
            discussion_text += f"--- ROUND {i+1} ---\n"
            synthesis = round_data.get("synthesis", {}).get("log", {}).get("parsed_output", {})
            if synthesis:
                discussion_text += f"MetaAgent Synthesis: {synthesis.get('explanation', 'N/A')}\n\n"

            for review in round_data.get("reviews", []):
                doctor_id = review.get("doctor_id")
                review_output = review.get("log", {}).get("parsed_output", {})
                if review_output:
                    discussion_text += f"{doctor_id} Review: {review_output.get('reason', 'N/A')}\n"
            discussion_text += "\n"

        user_message = {
            "role": "user",
            "content": f"Here is the claim from a minority doctor:\n"
                       f"Claim (from {keu_to_check['source_agent']}): \"{keu_to_check['content']}\"\n\n"
                       f"Here is the full discussion that followed:\n"
                       f"--- DISCUSSION START ---\n"
                       f"{discussion_text}"
                       f"--- DISCUSSION END ---\n\n"
                       f"Was this specific claim effectively rebutted during the discussion? Provide your judgment as a JSON object."
        }

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[system_message, user_message],
                response_format={"type": "json_object"},
            )
            if not completion.choices or not completion.choices[0].message:
                raise ValueError("Empty choices from LLM API")
            response_text = completion.choices[0].message.content or ""
            parsed_response = json.loads(preprocess_response_string(response_text))
            return parsed_response.get("is_rebutted", False)
        except Exception as e:
            print(f"LLM call for rebuttal check failed: {e}")
            # Conservatively default to the KEU not being rebutted if the check fails.
            return False

    def check_if_conflict_was_addressed(self, ccp_to_check: Dict, discussion_text: str) -> Tuple[bool, str]:
        """
        [Modified] Determines if a Critical Conflict Point (CCP) has been substantively resolved
        in a discussion. This version handles multi-party conflicts, uses stricter judgment
        criteria, and returns the reasoning for the decision.

        Args:
            ccp_to_check: The CCP dictionary from the audit log.
            discussion_text: The subsequent discussion text (e.g., synthesis or review reasons).

        Returns:
            A tuple (was_addressed, reasoning):
            - was_addressed (bool): True if the conflict was resolved, otherwise False.
            - reasoning (str): The LLM's justification for its decision.
        """
        system_message = {
            "role": "system",
            "content": """You are a highly critical and logical medical debate moderator. Your task is to determine if a specific 'Point of Conflict' was genuinely **RESOLVED**, not just mentioned, in the provided 'Discussion Text'.

**Your judgment must be strict. Default to `false` unless resolution criteria are explicitly met.**

### Definition of 'Substantively Resolved' (MUST MEET AT LEAST ONE):
1.  **New Decisive Evidence:** The discussion introduces a new piece of evidence or a clinical finding that directly supports one viewpoint and invalidates the others.
2.  **Logical Refutation:** The discussion provides a compelling logical argument that dismantles the reasoning of one or more conflicting statements, explaining *why* they are incorrect or less relevant in this specific context.
3.  **Hierarchical Prioritization:** The discussion establishes a clear clinical priority (e.g., "Address life-threatening risk A before diagnostic step B") with a strong justification that all parties would have to accept.

### What is NOT a Resolution:
-   **Simply Repeating/Acknowledging:** "Dr. A thinks X, Dr. B thinks Y." This is not a resolution.
-   **Choosing a Side without Justification:** "After reviewing the options, we will proceed with Dr. A's suggestion." This is an arbitrary choice, not a resolution. The *reasoning* for the choice is what matters.
-   **Compromise without Logic:** "Let's do a bit of both." This might be a valid clinical path, but it doesn't resolve the underlying logical conflict of which is *most* appropriate.
-   **Ignoring the Conflict:** The discussion proceeds without mentioning the core disagreement at all.

You MUST respond with a single JSON object with two keys:
1.  `"was_addressed"`: boolean (true only if the strict resolution criteria are met).
2.  `"resolution_reasoning"`: A brief string explaining *how* the conflict was resolved, or *why* it was not. If `false`, explain what was missing.
"""
        }

        # Dynamically build the list of conflicting statements to support multiple parties.
        conflicting_statements_text = ""
        for i, statement_data in enumerate(ccp_to_check['conflicting_statements']):
            agent_id = statement_data.get('agent_id', 'Unknown Agent')
            statement_content = statement_data.get('statement_content', 'No content provided.')
            conflicting_statements_text += f"{i+1}. From {agent_id}: \"{statement_content}\"\n"

        user_message = {
            "role": "user",
            "content": f"Here is the specific Point of Conflict you need to track:\n"
                       f"**Conflict Summary:** \"{ccp_to_check['conflict_summary']}\"\n\n"
                       f"**Conflicting Statements:**\n{conflicting_statements_text}\n"
                       f"--- \n"
                       f"Here is the Discussion Text that followed. Based on your strict criteria, did this text substantively **RESOLVE** the conflict?\n\n"
                       f"**--- DISCUSSION TEXT START ---**\n"
                       f"{discussion_text}\n"
                       f"**--- DISCUSSION TEXT END ---**\n\n"
                       f"Provide your judgment as a JSON object with `was_addressed` and `resolution_reasoning` keys."
        }

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[system_message, user_message],
                response_format={"type": "json_object"},
            )
            if not completion.choices or not completion.choices[0].message:
                raise ValueError("Empty choices from LLM API")
            response_text = completion.choices[0].message.content or ""
            parsed_response = json.loads(preprocess_response_string(response_text))

            reasoning = parsed_response.get("resolution_reasoning", "No reasoning provided by LLM.")
            decision = parsed_response.get("was_addressed", False)
            print(f"CCP Check for '{ccp_to_check.get('ccp_id', 'N/A')}': Addressed = {decision}. Reason: {reasoning}")

            return decision, reasoning
        except Exception as e:
            error_reason = f"LLM call for conflict address check failed: {e}"
            print(f"{error_reason} for '{ccp_to_check.get('ccp_id', 'N/A')}'")
            # Conservatively default to unresolved if the check fails, and return the error as the reason.
            return False, error_reason