from __future__ import annotations

import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_OPTION_SPLIT_RE = re.compile(r"\s*\(([A-F])\)\s*", re.IGNORECASE)


def _parse_formatted_question(question: str) -> tuple[str, Optional[Dict[str, str]]]:
    """Parse this repo's formatted question into (stem, options_dict|None)."""

    text = str(question)
    if " Options: " not in text:
        return text, None

    stem, options_part = text.split(" Options: ", 1)

    parts = _OPTION_SPLIT_RE.split(options_part)
    options: Dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i].upper().strip()
        content = parts[i + 1].strip()
        if letter:
            options[letter] = content

    return stem.strip(), (options or None)


def _ensure_model_key_in_settings(root_path: str, model_key: str, api_key: str, base_url: str) -> None:
    """Inject `model_key` into ColaCare's LLM_MODELS_SETTINGS using repo config."""

    # cfg_path = Path(root_path).expanduser().resolve() / "model_api_configs" / "model_api_config.json"
    # if not cfg_path.exists():
    #     raise RuntimeError(f"Missing model API config file: {cfg_path}")

    # with open(cfg_path, "r", encoding="utf-8") as f:
    #     cfg = json.load(f)

    # entry = cfg.get(model_key)
    # if not isinstance(entry, dict):
    #     raise RuntimeError(
    #         f"Missing model '{model_key}' in {cfg_path}. "
    #         "Expected a mapping like {\"gpt-4o-mini\": {\"api_key\": ..., \"base_url\": ...}}"
    #     )

    # api_key = entry.get("api_key")
    # base_url = entry.get("base_url") or entry.get("model_url")
    # Support custom model_name from config (useful for vllm server)
    model_name = model_key
    # if not api_key:
    #     raise RuntimeError(f"Missing 'api_key' for model '{model_key}' in {cfg_path}")
    # if not base_url:
    #     raise RuntimeError(f"Missing 'base_url' (or 'model_url') for model '{model_key}' in {cfg_path}")

    # Ensure medagentboard package can be imported as a top-level module.
    # colacare_root = Path(root_path).resolve() / "methods" / "ColaCare"
    # if str(colacare_root) not in sys.path:
    #     sys.path.insert(0, str(colacare_root))

    from methods.ColaCare.medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS

    if model_key not in LLM_MODELS_SETTINGS:
        LLM_MODELS_SETTINGS[model_key] = {
            "api_key": str(api_key),
            "base_url": str(base_url),
            "model_name": str(model_name),
            "comment": "Injected from visual-tool-lab model_api_configs",
            "reasoning": False,
        }
    else:
        LLM_MODELS_SETTINGS[model_key].setdefault("api_key", str(api_key))
        LLM_MODELS_SETTINGS[model_key].setdefault("base_url", str(base_url))
        LLM_MODELS_SETTINGS[model_key].setdefault("model_name", str(model_name))


def colacare_infer(
        question: str,
        root_path: str,
        model_info: str = "gpt-4o-mini",
        img_path: Optional[Any] = None,
        api_key: str = None,
        base_url: str = None,

):
    """ColaCare entrypoint for this repo.

    Uses upstream workflow in:
    medagentboard/medqa/multi_agent_colacare_full_log.py

    Returns the final predicted answer (for MC tasks: option letter).
    """

    doctor_model_key = os.getenv("COLACARE_DOCTOR_MODEL", model_info)
    meta_model_key = os.getenv("COLACARE_META_MODEL", model_info)

    for key in {doctor_model_key, meta_model_key}:
        _ensure_model_key_in_settings(root_path, key, api_key, base_url)

    # Lazy import (keeps `python -m main -h` safe).
    from methods.ColaCare.medagentboard.medqa.multi_agent_colacare_full_log import (
        MedicalSpecialty,
        parse_structured_output,
        process_input,
    )

    stem, options = _parse_formatted_question(question)
    # img_path 可能是 Path 对象列表或字符串，提取第一个图片路径
    resolved_img_path = None
    if img_path:
        if isinstance(img_path, (list, tuple)) and len(img_path) > 0:
            resolved_img_path = str(img_path[0])
        else:
            resolved_img_path = str(img_path)
    item = {
        "qid": "0",
        "question": stem,
        "options": options,
        "image_path": resolved_img_path,
    }

    doctor_configs = [
        {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "model_key": doctor_model_key},
        {"specialty": MedicalSpecialty.SURGERY, "model_key": doctor_model_key},
        {"specialty": MedicalSpecialty.RADIOLOGY, "model_key": doctor_model_key},
    ]

    # NOTE: Removed output suppression for debugging
    full_case_history = process_input(
        item,
        doctor_configs=doctor_configs,
        meta_model_key=meta_model_key,
    )

    final_decision_log = (full_case_history or {}).get("final_decision_log", {})
    parsed = final_decision_log.get("parsed_output")

    if not isinstance(parsed, dict):
        raw_output = final_decision_log.get("raw_output") or ""
        parsed = parse_structured_output(str(raw_output))

    predicted_answer = (parsed or {}).get("answer")
    if not predicted_answer:
        predicted_answer = "No answer found"

    agg = (full_case_history or {}).get("token_stats", {})
    token_stats = {model_info: {
        "num_llm_calls": int(agg.get("num_llm_calls", 0)),
        "prompt_tokens": int(agg.get("prompt_tokens", 0)),
        "completion_tokens": int(agg.get("completion_tokens", 0)),
    }}
    total_rounds = int((full_case_history or {}).get("total_rounds", 1))
    current_config = {"current_num_agents": 3, "round": total_rounds}
    return str(predicted_answer).strip(), token_stats, current_config
