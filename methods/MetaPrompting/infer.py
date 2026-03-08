from __future__ import annotations

import io
import json
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from methods.utils import get_apikey_and_baseurl_from_configs


# def _ensure_openai_env(root_path: str, model_name: str) -> None:
#     if not os.environ.get("OPENAI_API_KEY"):
#         api_key, base_url = get_apikey_and_baseurl_from_configs(root_path, model_name)
#         os.environ["OPENAI_API_KEY"] = api_key
#         if base_url and not os.environ.get("OPENAI_BASE_URL"):
#             os.environ["OPENAI_BASE_URL"] = base_url
#     else:
#         # Even if API key is already set, we may still want a base URL (for OpenAI-compatible providers).
#         if not os.environ.get("OPENAI_BASE_URL"):
#             _, base_url = get_apikey_and_baseurl_from_configs(root_path, model_name)
#             if base_url:
#                 os.environ["OPENAI_BASE_URL"] = base_url


def _extract_final_answer(text: str, is_vqa: bool = False, is_multi_select: bool = False) -> str:
    if not text:
        return ""
    m = re.search(r">>\s*FINAL\s+ANSWER:\s*\"\"\"\s*(.*?)\s*\"\"\"", text, re.IGNORECASE | re.DOTALL)
    raw = m.group(1).strip() if m else text.strip()

    if is_vqa:
        ans_match = re.search(r'[Aa]nswer\s*:\s*(.+)', raw)
        if ans_match:
            return ans_match.group(1).strip().strip("'\"`").strip()
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        return lines[-1] if lines else raw

    if is_multi_select:
        found = re.findall(r'[A-Fa-f]', raw)
        if found:
            unique = list(dict.fromkeys(c.upper() for c in found))
            return ", ".join(unique)
        return raw

    letter_match = re.search(r'^\s*\(?([A-Ea-e])\)?\s*[.)\]:]?\s', raw)
    if letter_match:
        return raw
    ans_match = re.search(r'[Aa]nswer\s*:\s*\(?([A-Ea-e])\)?', raw)
    if ans_match:
        return raw
    return raw


def metaprompting_infer(
    question: str,
    root_path: str,
    model_info: str,
    img_path: Optional[Any],
    api_key: str,
    base_url: str,
):
    """Thin adapter for upstream meta-prompting scaffolding.

    Returns:
        final_answer, token_stats
    """

    # _ensure_openai_env(root_path, model_info)

    base_dir = Path(__file__).resolve().parent
    meta_config_path = base_dir / "prompts" / "meta-v0-2023-08-14-baseline.json"
    instruction_path = base_dir / "prompts" / "meta-prompting-instruction.txt"

    meta_prompt_config_dict = json.loads(meta_config_path.read_text(encoding="utf-8"))
    question_prefix = instruction_path.read_text(encoding="utf-8")

    is_vqa = not bool(re.search(r'Options:|^\s*\([A-E]\)', question, re.MULTILINE))
    is_multi_select = bool(re.search(r'select\s+all\s+that\s+apply', question, re.IGNORECASE))

    if is_vqa:
        question_suffix = (
            "\n\nConsult relevant medical experts and solve this step by step. "
            "Provide a direct, concise answer. For yes/no questions, answer ONLY 'Yes' or 'No'. "
            "For other questions, answer with a brief specific phrase."
        )
        intermediate_feedback = (
            "Based on the expert's response, what is the answer? If confident, present your FINAL ANSWER now. "
            "Give a direct concise answer (e.g. 'Yes', 'No', 'Lung', 'CT scan'). Do NOT use option letters."
        )
    elif is_multi_select:
        question_suffix = (
            "\n\nConsult relevant medical experts and solve this step by step. "
            "This is a MULTI-SELECT question: there may be MORE THAN ONE correct answer. "
            "You MUST select ALL correct options from the given choices. "
            "End with your final answer listing ALL correct option letters separated by commas (e.g. 'A, C, D')."
        )
        intermediate_feedback = (
            "Based on the expert's response, what is the answer? If confident, present your FINAL ANSWER now. "
            "Remember this is a multi-select question. You MUST list ALL correct option letters separated by commas (e.g. 'A, C, D')."
        )
    else:
        question_suffix = (
            "\n\nConsult relevant medical experts and solve this step by step. "
            "You MUST select exactly one of the given options (A-E). "
            "Do NOT answer 'None of the above' or refuse to choose. "
            "End with your final answer as ONLY the option letter (e.g. '(A)')."
        )
        intermediate_feedback = (
            "Based on the expert's response, what is the answer? If confident, present your FINAL ANSWER now. "
            "You MUST choose one of the given options. "
            "For multiple-choice, present ONLY the option letter and text (e.g. '(A) Option text')."
        )
    expert_python_message = (
        'You are an expert in Python and can generate Python code. To execute the code and display its output in the terminal using print statements, '
        'please make sure to include "Please run this code!" after the code block (i.e., after the closing code blocks)'
    )

    # Lazy imports to keep `python -m main -h` safe.
    from .utils.language_model import OpenAI_LanguageModel
    from .utils.meta_scaffolding import MetaPromptingScaffolding

    model = OpenAI_LanguageModel(
        model_name=model_info,
        api_key=api_key,
        api_base=base_url,
    )

    meta_model = MetaPromptingScaffolding(
        language_model=model,
        fresh_eyes=True,
        generator_settings=meta_prompt_config_dict["generator"],
        verifier_settings=meta_prompt_config_dict["verifier"],
        summarizer_settings=meta_prompt_config_dict["summarizer"],
        error_message=meta_prompt_config_dict["meta-model"]["error-message"],
        final_answer_indicator=meta_prompt_config_dict["meta-model"]["final-answer-indicator"],
        include_expert_name_in_instruction=False,
        extract_output=False,
        expert_python_message=expert_python_message,
        intermediate_feedback=intermediate_feedback,
        use_zero_shot_cot_in_expert_messages=False,
    )

    meta_model_message_list = meta_prompt_config_dict["meta-model"]["message-list"]
    meta_model_settings = meta_prompt_config_dict["meta-model"]["parameters"]

    if img_path:
        model.set_images(img_path if isinstance(img_path, (list, tuple)) else [img_path])

    messages = [m.copy() for m in meta_model_message_list]
    messages.append(
        {
            "role": "user",
            "content": f"{question_prefix}Question: {question}{question_suffix}",
        }
    )

    # Upstream prints a lot; keep benchmark output clean.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        message_log = meta_model.meta_model_generate(
            prompt_or_messages=messages,
            max_tokens=meta_model_settings["max_tokens"],
            temperature=meta_model_settings["temperature"],
            top_p=meta_model_settings["top_p"],
            num_return_sequences=meta_model_settings["num_return_sequences"],
            counter=0,
            original_question=question,
        )

    final_text = ""
    try:
        if isinstance(message_log, list) and message_log:
            for msg in reversed(message_log):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content") or "")
                    break
            else:
                final_text = str(message_log[-1].get("content") or "")
        else:
            final_text = str(message_log)
    except Exception:
        final_text = str(message_log)

    final_answer = _extract_final_answer(final_text, is_vqa=is_vqa, is_multi_select=is_multi_select)
    token_stats = {
        model_info: {
            "num_llm_calls": int(getattr(model, "num_llm_calls", 0)),
            "prompt_tokens": int(getattr(model, "prompt_tokens", 0)),
            "completion_tokens": int(getattr(model, "completion_tokens", 0)),
        }
    }
    current_config = {"current_num_agents": 4, "round": 1}
    return final_answer, token_stats,current_config
