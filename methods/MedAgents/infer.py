from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import Dict, Tuple

from methods.utils import get_apikey_and_baseurl_from_configs
from .api_utils import api_handler
from .utils import fully_decode
from pathlib import Path
import yaml

def load_config(file_path: str) -> dict:
    """
    Load YAML configuration from a file.

    Args:
    file_path (str): Path to the YAML configuration file.

    Returns:
    dict: Dictionary of configuration parameters.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def _inject_openai_compatible_env(root_path: str, model_info: str) -> Tuple[str, str]:
    """Backfill OPENAI_* env vars from repo config if missing."""

    api_key, base_url = get_apikey_and_baseurl_from_configs(root_path, model_info)
    os.environ.setdefault("OPENAI_API_KEY", str(api_key))
    os.environ.setdefault("OPENAI_BASE_URL", str(base_url))
    os.environ.setdefault("OPENAI_API_BASE", str(base_url))
    return str(api_key), str(base_url)


def _split_question_options(question: str) -> Tuple[str, str]:
    text = (question or "").strip()
    if not text:
        return "", ""

    marker = "Options:"
    if marker in text:
        q, opts = text.split(marker, 1)
        return q.strip(), opts.strip()

    return text, ""


def _wrap_counting_handler(handler, counters: Dict[str, int]):
    original = handler.get_output_multiagent

    def _wrapped(*args, **kwargs):
        counters["num_llm_calls"] += 1
        out = original(*args, **kwargs)
        usage = getattr(handler, "_last_usage", None)
        if usage is not None:
            counters["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            counters["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return out

    handler.get_output_multiagent = _wrapped
    return handler


def medagents_infer(question: str, root_path: str, model_info: str, img_path=None,batch_manager=None):
    config_path = str(Path(root_path) / 'methods' / 'MedAgents' / 'configs' / 'config_main.yaml')
    config = load_config(config_path)
    num_qd = config.get('num_qd', 5)
    num_od = config.get('num_od', 2) #
    max_round = config.get('max_round', 3)
    role_mode = config.get('role_mode', "dynamic")

    # _inject_openai_compatible_env(root_path, model_info)

    q_text, options_text = _split_question_options(question)

    is_vqa = not options_text or not options_text.strip()
    is_mca = bool(re.search(r'select\s+all\s+that\s+apply', question, re.IGNORECASE))
    effective_max_round = max_round if max_round is not None else 3
    args = SimpleNamespace(method="syn_verif", max_attempt_vote=effective_max_round)

    handler = api_handler(img_paths=img_path,model_info=model_info,batch_manager=batch_manager)
    handler.engine = model_info

    counters = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    handler = _wrap_counting_handler(handler, counters)

    data_info = fully_decode(
        qid=0,
        realqid=0,
        question=q_text,
        options=options_text,
        gold_answer="",
        handler=handler,
        args=args,
        dataobj=None,
        num_qd=num_qd,
        num_od=num_od,
        is_mca=is_mca,
        role_mode=role_mode,
    )

    final = (data_info.get("pred_answer") or "").strip()
    if is_mca:
        # For MCA, extract all option letters and sort them
        valid = re.findall(r'\(([A-Z])\)', options_text) if options_text else []
        valid_set = set(v.upper() for v in valid) if valid else set('ABCDEF')
        found = re.findall(r'[A-Z]', final.upper())
        unique = sorted(set(f for f in found if f in valid_set))
        final = ", ".join(unique) if unique else final
    elif options_text and final and len(final) > 2:
        # Single-choice: extract one option letter
        valid = re.findall(r'\(([A-Z])\)', options_text)
        pattern = '|'.join(valid) if valid else 'A|B|C|D'
        m = re.search(r'\b(' + pattern + r')\b', final.upper())
        if m:
            final = m.group(1)
    current_num_agents= data_info.get("current_num_agents")
    token_stats = {model_info: dict(counters)}
    print(f"\nCurrent number of agents: {current_num_agents}")
    current_config = {"current_num_agents": current_num_agents, "round": 1}
    return final, token_stats, current_config
