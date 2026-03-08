from __future__ import annotations

from typing import Any, Dict, List, Optional

from .medqa import load_medqa_test_split, format_medqa_question,format_vqa_question,format_video_question


def load_test_split(dataset_dir: str, dataset_name: str) -> List[Dict[str, Any]]:
    # dataset_name = (dataset_name or "").lower()
    # if dataset_name == "medqa":
    return load_medqa_test_split(dataset_dir)
    # raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def format_question(sample: Dict[str, Any], dataset_name: str):
    # dataset_name = (dataset_name or "")
    if dataset_name in ['VQA_RAD','MedCXR','MedXpertQA_MM','slake','M3CoTBench']:
        return format_vqa_question(sample)
    elif dataset_name=='MedVidQA':
        return format_video_question(sample)
    else:
        return format_medqa_question(sample)
    # raise ValueError(f"Unsupported dataset_name: {dataset_name}")



# def extract_choice(model_output: str, dataset_name: str) -> Optional[str]:
#     dataset_name = (dataset_name or "").lower()
#     # if dataset_name == "medqa":
#     return extract_medqa_choice(model_output)
    # return None
