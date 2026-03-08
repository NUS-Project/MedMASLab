from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

def load_medqa_test_split(dataset_dir: str) -> List[Dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    test_path = dataset_path / "test.jsonl"

    test_qa: List[Dict[str, Any]] = []
    with open(test_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            test_qa.append(json.loads(line))
    return test_qa


def format_medqa_question(sample: Dict[str, Any]):
    options = sample.get("options", None)
    if options is not None:
        question = str(sample.get("question", "")) + " Options: "
    else:
        question = str(sample.get("question", ""))
    answer_type='closed'

    if options is not None and isinstance(options, dict):
        rendered = [f"({k}) {v}" for k, v in options.items()]
        question += " ".join(rendered)
        return question,None,answer_type

    return str(sample.get("question", "")),None,answer_type

def format_vqa_question(sample: Dict[str, Any]):
    options = sample.get("options", None) #
    visual_description = sample.get("visual_description", None)
    if options is not None:
        if visual_description is not None:
            question = str(sample.get("question", "")) +"The description of this image is provided here:"+str(visual_description) +" Options: "
        else:
            question = str(sample.get("question", "")) + " Options: "
    else:
        question = str(sample.get("question", ""))
    
    img_path=sample.get("img_path", {})
    answer_type=sample.get("answer_type", "closed")

    if options is not None and isinstance(options, dict):
        rendered = [f"({k}) {v}" for k, v in options.items()]
        question += " ".join(rendered)
        if answer_type.lower()=="mca":
            return question,img_path,answer_type
        else:
            return question,img_path,answer_type

    return str(sample.get("question", "")),img_path,answer_type

def format_video_question(sample: Dict[str, Any]):
    options = sample.get("options", None) #
   
    if options is not None:
        question = str(sample.get("question", "")) + " Options: "
    else:
        question = str(sample.get("question", ""))
    
    img_path=str(sample.get("id"))+".mp4"
    answer_type=sample.get("answer_type", "closed")

    if options is not None and isinstance(options, dict):
        rendered = [f"({k}) {v}" for k, v in options.items()]
        question += " ".join(rendered)
        return question,img_path,answer_type

    return str(sample.get("question", "")),img_path,answer_type


def num_repetition(result_path,num_llm_calls_total,Prompt_Tokens_total,completion_tokens_total,processed_count,time_cost,test_qa):
   
    with open(result_path, 'r', encoding='utf-8') as infile:
        if infile is None:
            return 0,0,0,0,0,0,test_qa
        correct_count = 0
        processed_ids = set()  # 使用集合存储已处理的ID（查找更快）
        
        for line in infile:
            result = json.loads(line.strip())
            # 累计统计信息
            processed_count += 1
            
            # 获取样本ID
            sample_id = result.get("id")
            
            processed_ids.add(sample_id)  # 记录已处理的ID
            
            # 统计正确数
            is_correct = str(result.get("is_correct", "")).lower()
            if is_correct == 'true':
                correct_count += 1
            
            # 累计时间和token统计
            time_cost += result.get("time_cost", 0)
            num_llm_calls_total += result.get("Num_Calls", 0)
            Prompt_Tokens_total += result.get("Prompt_Tokens", 0)
            completion_tokens_total += result.get("Completion_tokens", 0)
        # 过滤掉已处理的样本
        new_test_qa = [sample for sample in test_qa if sample.get('id') not in processed_ids]

        # 打印过滤信息
        print(f"\n已经过滤掉{len(test_qa) - len(new_test_qa)}个QA!")
        return num_llm_calls_total,Prompt_Tokens_total,completion_tokens_total,processed_count,time_cost,correct_count,new_test_qa

    # # If no match is found, return False and None values
    # return False, None, None, None, None, None

def check_repetition(sample: Dict[str, Any], result_path: str):
    question_id = str(sample.get("id", ""))
    if not os.path.isfile(result_path) or os.stat(result_path).st_size == 0:
        return False, None, None, None, None, None  # Return if the file doesn't exist or is empty

    # Open the JSONL file and iterate through each line
    with open(result_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            result = json.loads(line.strip())
            result_id = str(result.get("id", ""))

            # Check if the question_id matches the result_id
            if question_id == result_id:
                is_correct = result.get("is_correct", "")
                time_cost = result.get("time_cost", "")
                Num_Calls = result.get("Num_Calls", "")
                Prompt_Tokens = result.get("Prompt_Tokens", "")
                Completion_tokens = result.get("Completion_tokens", "")
                return True, is_correct, time_cost, Num_Calls, Prompt_Tokens, Completion_tokens

    # If no match is found, return False and None values
    return False, None, None, None, None, None

# _CHOICE_PATTERNS = [
#     re.compile(r"\(([A-E])\)", re.IGNORECASE),
#     re.compile(r"(?:the\s+answer\s+is\s*[:：]?\s*)\(?\s*([A-F])\s*\)?", re.IGNORECASE),
# ]


# def extract_medqa_choice(model_output: str) -> Optional[str]:
#     if not model_output:
#         return None

#     text = str(model_output).strip()
#     last_match: Optional[str] = None

#     for pattern in _CHOICE_PATTERNS:
#         for match in pattern.finditer(text):
#             last_match = match.group(1).upper()

#     return last_match
