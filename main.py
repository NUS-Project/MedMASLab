import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re
import threading
from methods.MetaPrompting import metaprompting_infer
from methods.MedAgents import medagents_infer
from methods.MDAgents.medagents import MDAgents_test
from methods.autogen import autogen_infer_medqa
from methods.dylan import dylan_infer_medqa
from methods.Cot import Cot_test
from methods.ColaCare import colacare_infer
from methods.Discussion import discussion_infer
from methods.SC import SelfConsistency_test
from methods.Reconcile.reconcile_test import Reconcile_test
from methods.general_model import test_BaseLine
from methods.debate import Debate_test
from methods.MDTeamGPT.MDTeamGPT_test import MDTeamGPT_test
from dataset_utils import load_test_split, format_question
from dataset_utils.medqa import num_repetition
from llm_evaluate import QwenVL_JudgeModel
import time
import concurrent.futures
from methods.thread import BatchInferenceManager,Video_BatchInferenceManager,LLava_BatchInferenceManager,Video_LLava_BatchInferenceManager
from methods.vllm_thread import VLLMBatchInferenceManager
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
run_started_at = datetime.now(timezone.utc)
run_started_perf = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--general_model_path', type=str, default=str(PROJECT_ROOT / 'models'))
parser.add_argument('--dataset_path', type=str, default=str(PROJECT_ROOT / 'data'))
parser.add_argument('--dataset_name', type=str, default='medqa',
                    choices=['medqa', 'pubmedqa', 'medbullets', 'MMLU', 'dxbench', 'VQA_RAD',
                             'MedCXR', 'MedXpertQA_MM', 'slake', 'MedVidQA','M3CoTBench'])
parser.add_argument('--judge_model', type=str, default='Qwen2.5-VL-32B-Instruct',
                    choices=['gpt-4o-mini', 'Qwen2.5-VL-32B-Instruct'])
parser.add_argument(
    '--model',
    type=str,
    default='MDAgents',
    choices=['Qwen2.5-VL-72B-Instruct','Qwen2.5-VL-3B-Instruct', 'Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-7B-Instruct','LLaVA-v1.6-mistral-7b-hf','LLaVA-v1.6-mistral-7b-hf', 'MDAgents', 'autogen', 'dylan', 
             'Discussion','Cot', 'SelfConsistency', 'MDTeamGPT', 'Debate', 'Reconcile', 'MedAgents', 'MetaPrompting', 'ColaCare','gpt-4o-mini'],
)
parser.add_argument('--base_model', type=str, default='Qwen2.5-VL-7B-Instruct',
                    choices=['gpt-4o-mini', 'Qwen2.5-VL-7B-Instruct','LLaVA-v1.6-mistral-7b-hf','Qwen2.5-VL-3B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct'],
                    help='Base model id for methods that call OpenAI-compatible APIs (e.g., gpt-4o-mini, gemini-2.5-pro)')
parser.add_argument('--root_path', type=str, default=str(PROJECT_ROOT))
parser.add_argument("--device", type=str, default="cuda", help='Device / device_map, e.g. "cuda:3" or "auto"')
parser.add_argument('--num_samples', type=int, default=99999)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for parallel inference')
parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads')
parser.add_argument('--judge_batch_size', type=int, default=2, help='Batch size for parallel judge inference')
# parser.add_argument('--video_batch_size', type=int, default=3, help='Batch size for parallel judge inference')
# parser.add_argument('--use_vllm', type=bool, default=True, help='whether to use vllm')
parser.add_argument('--base_vllm_url', type=str, default='https://yinli.one/v1', help='base_vllm url')# http://localhost:8006/v1
parser.add_argument('--judge_vllm_url', type=str, default='https://yinli.one/v1', help='judge_vllm url') # http://localhost:8007/v1
args = parser.parse_args()

args.root_path = str(Path(args.root_path).expanduser().resolve())
args.dataset_path = str(Path(args.dataset_path).expanduser().resolve())
dataset_path_name = str(Path(args.dataset_path) / args.dataset_name)
test_qa = load_test_split(dataset_path_name, args.dataset_name)

os.environ['DECORD_REWIND_RETRY_MAX'] = '32'
Judge_Model=None
QwenVL_model=None
batch_manager=None
Judge_batch_manager=None
video_batch_manager=None

out_path = Path(args.root_path) / 'output'
if not os.path.exists(out_path):
    os.makedirs(out_path)

output_file_json_name = f"{args.model}_{args.base_model}_{args.dataset_name}.jsonl"
output_file_json_path = Path(args.root_path) / 'output' / output_file_json_name
if not output_file_json_path.exists():
    output_file_json_path.touch()

batch_manager = VLLMBatchInferenceManager(
            model=args.base_model,
            root_path=args.root_path,
            batch_size=args.batch_size,
            timeout=0.5,
            vllm_url=args.base_vllm_url
        )
Judge_batch_manager = VLLMBatchInferenceManager(
                model=args.judge_model,
                root_path=args.root_path,
                batch_size=args.judge_batch_size,
                timeout=0.5,
                vllm_url=args.judge_vllm_url
            )
batch_manager.start()
Judge_batch_manager.start()
Judge_Model = QwenVL_JudgeModel(args.judge_model, args.root_path, args.device)

correct_count = 0
num_total = len(test_qa)
num_to_run = min(args.num_samples, num_total) if args.num_samples else num_total
processed_count = 0
time_cost = 0
num_llm_calls_total = 0
Prompt_Tokens_total = 0
completion_tokens_total = 0
current_config = None
token_stats = None
curr_processed_count = 0
################# 检查之前的评测历史
num_llm_calls_total, Prompt_Tokens_total, completion_tokens_total, processed_count, time_cost, correct_count,test_qa = num_repetition(
    output_file_json_path, num_llm_calls_total, Prompt_Tokens_total, completion_tokens_total, processed_count,
    time_cost,test_qa)
#################
Retention_interval = min((num_to_run - processed_count) ,args.save_interval)  
results_buffer = []  #
def process_sample(sample, args,batch_manager=None,Judge_batch_manager=None,Judge_Model=None):
   
    question, img_path, answer_type= format_question(sample, args.dataset_name)

    if img_path is not None:
        if isinstance(img_path, str):
            img_path = Path(args.root_path) / 'data' / args.dataset_name / 'imgs' / img_path
            img_path = [img_path]
        else:
            full_img_paths = []
            for img in img_path:
                full_img_paths.append(Path(args.root_path) / 'data' / args.dataset_name / 'imgs' / img)
            img_path = full_img_paths

    question_id = sample['id']
    correct_answer_idx = sample.get('answer_idx', "")
    correct_answer_content = sample.get('answer')

    print(f"\n[Thread {threading.current_thread().name}] \nProcessing question: {question_id}")
    start_time = time.time()                                       
    if args.model == "MetaPrompting":
        final_decision, token_stats, current_config = metaprompting_infer(question, args.root_path, args.base_model, img_path)
    elif args.model == "MedAgents":
        final_decision, token_stats, current_config = medagents_infer(question, args.root_path, args.base_model, img_path,
                                   num_qd=args.medagents_num_qd, num_od=args.medagents_num_od,
                                   max_round=args.medagents_max_round,
                                   role_mode=args.medagents_role_mode or "dynamic")
    elif args.model == "MDAgents":
        final_decision, token_stats, current_config = MDAgents_test(question, args.root_path, args.base_model, img_path,
                                                                    batch_manager)
    elif args.model == "autogen":
        final_decision, token_stats, current_config = autogen_infer_medqa(question, args.root_path, args.base_model, img_paths=img_path)
    elif args.model == "dylan":
        final_decision, token_stats, current_config = dylan_infer_medqa(question, args.root_path, args.base_model, img_paths=img_path)
    elif args.model == "Discussion":
        final_decision, token_stats, current_config = discussion_infer(question, args.root_path, args.base_model,
                                                                           img_path, batch_manager)
    elif args.model == "Cot":
        final_decision, token_stats = Cot_test(args.root_path, args.base_model, question, img_path)
    elif args.model == "SelfConsistency":
        final_decision, token_stats = SelfConsistency_test(args.root_path, args.base_model, question, img_path)
    elif args.model == "MDTeamGPT":
        final_decision, token_stats, current_config = MDTeamGPT_test(question, img_path, args.root_path, batch_manager)
    elif args.model == "Debate":
        final_decision, token_stats, current_config = Debate_test(question, args.root_path, args.base_model, img_path,
                                                                  batch_manager)
    elif args.model == "Reconcile":
        final_decision, token_stats, current_config = Reconcile_test(question, args.root_path, args.base_model,
                                                                     img_path,batch_manager)
    elif args.model == "ColaCare":
        final_decision, token_stats, current_config = colacare_infer(question, args.root_path, args.base_model, img_path)
    else:
        final_decision, token_stats, current_config = test_BaseLine(question, img_path, batch_manager)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 评估结果
    final_decision_orgin = final_decision
    Judge_result = Judge_Model.chat(question, final_decision, correct_answer_idx, correct_answer_content, img_path,
                                    answer_type,Judge_batch_manager)
    cleaned_result = re.sub(r'[^a-zA-Z]', '', Judge_result).lower()  # Clean the result
    if cleaned_result[:5] == 'wrong':
        is_correct = False
    else:
        is_correct = True
    print(f"\nfinal_decision_content:{final_decision_orgin}\nAnswer:{correct_answer_content}\nis_correct:{is_correct}\nanswer_idx{correct_answer_idx} ")
    # 统计token使用
    for model_name, stats in token_stats.items():
        num_llm_calls = stats['num_llm_calls']
        Prompt_Tokens = stats['prompt_tokens']
        completion_tokens = stats['completion_tokens']

    result = {
        'id': question_id,
        # 'question': question,
        'final_decision': final_decision_orgin,
        'answer': correct_answer_content ,
        'right_option': correct_answer_idx,
        'clean_result': final_decision,
        'answer_type':answer_type,
        'is_correct': is_correct,
        'time_cost': elapsed_time,
        'Num_Calls': num_llm_calls,
        'Prompt_Tokens': Prompt_Tokens ,
        "Judge_result": Judge_result,
        'Completion_tokens': completion_tokens,
        'current_config': current_config,
    }

    return result, is_correct, current_config,elapsed_time,num_llm_calls,Prompt_Tokens,completion_tokens



results_buffer = []
correct_count_lock = threading.Lock()
file_lock = threading.Lock()


def save_results_buffer(buffer, output_path):
    """线程安全地保存结果"""
    with file_lock:
        with open(output_path, 'a', encoding='utf-8') as file:
            for result in buffer:
                json.dump(result, file, ensure_ascii=False)
                file.write("\n")


# 使用线程池处理样本
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    # concurrent.futures.ThreadPoolExecutor: Python标准库的线程池
    # max_workers=args.num_workers: 线程池中的线程数量（例如4个）
    # with语句：确保线程池在代码块结束后自动关闭，释放资源
    # executor: 线程池管理器对象，用于提交和管理任务
    futures = []

    for sample in test_qa[:num_to_run]:
        future = executor.submit(
            process_sample, sample,args,batch_manager, Judge_batch_manager,Judge_Model
        )
        futures.append(future)

    # 收集结果
    for future in tqdm(concurrent.futures.as_completed(futures),
                       total=len(futures),
                       desc=f"Infer {args.model}",
                       unit="Sample"):
        try:
            result, is_correct, current_config,elapsed_time,num_llm_calls,Prompt_Tokens,completion_tokens = future.result()

            with correct_count_lock:
                processed_count += 1
                curr_processed_count += 1
                if is_correct:
                    correct_count += 1

                num_llm_calls_total += num_llm_calls
                Prompt_Tokens_total += Prompt_Tokens
                completion_tokens_total += completion_tokens
                time_cost += elapsed_time

            results_buffer.append(result)

            # 定期保存
            if curr_processed_count >= Retention_interval:
                curr_processed_count=0
                save_results_buffer(results_buffer, output_file_json_path)
                print(f"\n已保存 {len(results_buffer)} 条结果到文件")
                results_buffer.clear()
                # curr_processed_count = 0

        except Exception as e:
            print(f"\n处理样本时出错: {e}")
            import traceback

            traceback.print_exc()

# 保存剩余结果
if results_buffer:
    save_results_buffer(results_buffer, output_file_json_path)
    print(f"\n最终保存 {processed_count} 条结果到文件")


batch_manager.stop()
Judge_batch_manager.stop()


# 输出统计信息
print(f"[INFO] Accuracy: {(correct_count / processed_count) * 100:.2f}%")
print(f"\n[INFO] Average Calls: {num_llm_calls_total / processed_count:.2f}")
print(f"[INFO] Average Prompt Tokens: {Prompt_Tokens_total / processed_count:.2f}")
print(f"[INFO] Average Completion Tokens: {completion_tokens_total / processed_count:.2f}")
print(f"[INFO] Average time cost: {time_cost / num_to_run:.2f}s")
