from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _load_model_api_config(root_path: str) -> Dict[str, Any]:
    cfg_path = Path(root_path).expanduser().resolve() / "model_api_configs" / "model_api_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_general_config(
    root_path: str,
    model_info: str,
    *,
    model_temperature: float = 0.2,
    model_max_tokens: int = 2048,
    model_timeout: int = 600,
) -> Dict[str, Any]:
    """Build MASLab-style general_config from this repo's model_api_config.json.

    This repo format (current):
      {"gpt-4o-mini": {"model_url": "...", "api_key": "..."}}

    MASLab expects:
      {"model_api_config": {"gpt-4o-mini": {"model_list": [{...}]}}}
    """

    cfg = _load_model_api_config(root_path)
    entry = cfg.get(model_info)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Missing model '{model_info}' in model_api_configs/model_api_config.json")

    api_key = entry.get("api_key")
    base_url = entry.get("model_url") or entry.get("base_url")
    if not api_key or not base_url:
        raise RuntimeError(
            f"Model '{model_info}' must define 'api_key' and 'model_url' (or 'base_url') in model_api_configs/model_api_config.json"
        )

    model_api_config = {
        model_info: {
            "model_list": [
                {
                    "model_name": model_info,
                    "model_url": base_url,
                    "api_key": api_key,
                }
            ]
        }
    }

    return {
        "model_api_config": model_api_config,
        "model_name": model_info,
        "model_temperature": model_temperature,
        "model_max_tokens": model_max_tokens,
        "model_timeout": model_timeout,
    }
