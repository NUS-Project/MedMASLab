import os
import re
import ast
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

random.seed(1234)
datasets = ["SQA", "GSM8k", "ECQA", "Aqua"]

def prepare_context(test_sample, convincing_samples=None, intervene=False, dataset="SQA"):
    assert dataset in datasets
    context = []
    if convincing_samples:
        for cs in convincing_samples:
            context.append(f"Q: {cs['train_sample']['question']}\nA:" + str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']}))
     
    if intervene:
        context.append("Q: " + test_sample['question'] + "\nAnswer the question given the fact that " + test_sample['gold_explanation'])
    else:
        context.append("Q: " + test_sample['question'])

    if dataset in ["ECQA", "Aqua", "ScienceQA"]:
        context.append("The options are: {}. Please select an option as your answer.".format(" ".join(test_sample["options"])))
        
    context.append("Please answer the question with step-by-step reasoning.")
    context.append("\nAlso, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right.")
    context.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
    if dataset == "SQA":
        context.append("Only answer yes or no in the \"answer\" field.")
    elif dataset == "GSM8k":
        context.append("Only place a single numeric value in the \"answer\" field.")
    elif dataset == "ECQA":
        context.append("Only place 1,2,3,4,5 representing your choice in the \"answer\" field.")
    elif dataset == "Aqua":
        context.append("Only place A,B,C,D,E representing your choice in the \"answer\" field.")  
    context.append("Do not output irrelevant content.")
    return "\n".join(context)
    
def prepare_context_for_chat_assistant(sample, convincing_samples=None, intervene=False, dataset="SQA"):
    assert dataset in datasets
    contexts = []
    if convincing_samples:
        for cs in convincing_samples:
            contexts.append({"role": "user", "content": f"Q: {cs['train_sample']['question']}"})
            contexts.append({"role": "assistant", "content": str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']})})

    if intervene:
        contexts.append({"role": "user", "content": f"Q: {sample['question']}" + "\nAnswer the question given the fact that " + sample['gold_explanation']})  
    else:
        contexts.append({"role": "user", "content": f"Q: {sample['question']}"})
        
    if dataset in ["ECQA", "Aqua"]:
        contexts[-1]["content"] += "The options are: {}. Please select an option as your answer.".format(" ".join(sample["options"]))
        
    contexts[-1]["content"] += " Please answer the question with step-by-step reasoning."
    contexts[-1]["content"] += " Also, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right."
    contexts[-1]["content"] += " Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format."
    
    if dataset == "SQA":
        contexts[-1]["content"] += " Only answer yes or no in the \"answer\" field."
    elif dataset == "GSM8k":
        contexts[-1]["content"] += " Only place a single numeric value in the \"answer\" field."
    elif dataset == "ECQA":
        contexts[-1]["content"] += " Only place 1,2,3,4,5 representing your choice in the \"answer\" field."
    elif dataset == "Aqua":
        contexts[-1]["content"] += " Only place A,B,C,D,E representing your choice in the \"answer\" field."    
    contexts[-1]["content"] += " Do not output irrelevant content."
    return contexts

def prepare_context_for_bard(test_sample, convincing_samples=None, intervene=False, dataset="SQA"):
    assert dataset in datasets
    context, convincing_icx, unhelpful_icx = [], [], []
    if convincing_samples:
        for cs in convincing_samples:
            convincing_icx.append((f"Q: {cs['train_sample']['question']}", str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']})))
    
    if intervene:
        context.append("Q: " + test_sample['question'] + "\nAnswer the question given the fact that " + test_sample['gold_explanation'])
    else:
        context.append("Q: " + test_sample['question'])
        
    if dataset in ["ECQA", "Aqua"]:
        context.append("The options are: {}. Please select an option as your answer.".format(" ".join(test_sample["options"])))  
        
    context.append("Please answer the question with step-by-step reasoning.")
    context.append("Also, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right.")
    context.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
    
    if dataset == "SQA":
        context.append("Only answer yes or no in the \"answer\" field. Do not output irrelevant content.")
    elif dataset == "GSM8k":
        context.append("Only place a single numeric value in the \"answer\" field. Do not output irrelevant content.")
    elif dataset == "ECQA":
        context.append("Only place 1,2,3,4,5 representing your choice in the \"answer\" field.")
    elif dataset == "Aqua":
        context.append("Only place A,B,C,D,E representing your choice in the \"answer\" field.")
    context.append("Do not output irrelevant content.")
    return "\n".join(context), convincing_icx, unhelpful_icx

def clean_model_output_by_colon(model_output):
    
    # 按冒号分割
    model_output = model_output.replace("\n", " ")
    colon_positions = [i for i, char in enumerate(model_output) if char == ':']
    
    # 检查冒号数量是否足够（至少需要3个）
    if len(colon_positions) < 3:
        # raise ValueError(f"冒号数量不足,需要至少3个冒号,但只找到{model_output}")
        # result={"reasoning": second_part,"answer": third_part_cleaned,"confidence_level": fourth_part}
        return {"reasoning": model_output,"answer": "","confidence_level": 0.0}
    
    # 获取第一个位置A，倒数第二个位置B，倒数第一个位置C
    position_A = colon_positions[0]      # 第一个冒号
    position_B = colon_positions[-2]     # 倒数第二个冒号
    position_C = colon_positions[-1] 
    parts=[model_output[position_A+1:position_B],model_output[position_B+1:position_C] ,model_output[position_C+1:]]
    # parts = model_output.split(':', maxsplit=3)  # 最多分4段
    
    if len(parts) < 2:
        # print(f"⚠️ 未找到冒号，返回原文")
        return model_output
    
    
    # 处理第二段
    if len(parts) >= 2:
        second_part = parts[1]
        first_letter_idx = -1
        for i, char in enumerate(second_part):
            if char.isalpha():
                first_letter_idx = i
                break
        
        if first_letter_idx > 0:
            second_part = ' ' * first_letter_idx + second_part[first_letter_idx:]
        
        if len(second_part) > 10:
            second_part = second_part[:-10]
        
        # ← 新增：将引号和逗号替换为空格
        second_part = second_part.replace('"', ' ').replace(',', ' ').replace("'", ' ')
        
        # print(f"✅ 第二段: {second_part}")
    
    # 处理第三段
    if len(parts) >= 3:
        third_part = parts[2]
        position_A = -1
        for i, char in enumerate(third_part):
            if char.isalpha():
                position_A = i
                break
        
        if position_A == -1:
            third_part_cleaned = third_part
        else:
            target_start_idx = -1
            
            # 从position_A开始查找
            search_text = third_part[position_A:]
            
            # 使用正则或字符串查找
            import re
            match = re.search(r'confidence_level', search_text, re.IGNORECASE)
            
            if match:
                # confidence_level 在 search_text 中的起始位置
                target_start_idx = position_A + match.start()
                
                # 只保留 position_A 到 confidence_level 第一个字之间的内容
                third_part_cleaned = third_part[position_A:target_start_idx]
                
            else:
                third_part_cleaned = third_part[position_A:]
        
        # ← 新增：将引号和逗号替换为空格
        third_part_cleaned = third_part_cleaned.replace('"', '').replace(',', '').replace("'", '')
        
        # print(f"✂️ 第三段: {third_part_cleaned}")
    else:
        third_part_cleaned = ""
    stripped = third_part_cleaned.replace('\n', '').replace(' ', '')
    if len(stripped)<=10:
        third_part_cleaned = stripped
    
    
    # 处理第四段
    if len(parts) >= 4:
        fourth_part = parts[3]
        
        # ← 新增：只保留阿拉伯数字和小数点
        import re
        numbers_only = re.findall(r'[\d.]+', fourth_part)
        
        if numbers_only:
            # 取第一个数字串并转为 float
            try:
                fourth_part = float(numbers_only[0])
                # print(f"📝 第四段（原始）: {fourth_part}")
                # print(f"📝 第四段（提取数字）: {numbers_only[0]}")
                # print(f"📝 第四段（转为float）: {fourth_part_float}")
                # fourth_part = str(fourth_part_float)
            except ValueError:
                # print(f"⚠️ 无法转换为 float: {numbers_only[0]}")
                fourth_part = numbers_only[0]
        else:
            # print(f"⚠️ 第四段未找到数字: {fourth_part}")
            fourth_part = 0.0
    else:
        fourth_part = ""
    
    # 重新组合（去掉第一段）
    result={"reasoning": second_part,"answer": third_part_cleaned,"confidence_level": fourth_part}
    return result

def parse_json(model_output):
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    output=model_output
    try:
        model_output = model_output.replace("\n", " ")
        model_output = re.search('({.+})', model_output).group(0)
        model_output = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", model_output)
        result = ast.literal_eval(model_output)
    except (SyntaxError, NameError, AttributeError):
# try:
        result=clean_model_output_by_colon(output)
        # model_output = model_output.replace("\n", " ")
        # if model_output[-2] == '"':
        #     model_output = model_output[:-2] + model_output[-1]
        # model_output = re.search('({.+})', model_output).group(0)
        # model_output = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", model_output)
        # result = ast.literal_eval(model_output)
    # except (SyntaxError, NameError, AttributeError):
    #     return "ERR_SYNTAX"
    return result

def find_idx_by_element(input_list, element):
    return [i for i, a in enumerate(input_list) if a == element]

def find_element_by_indices(input_list, index_list):
    return [b for i, b in enumerate(input_list) for k in index_list if i == k]

def trans_confidence(x):
    if x is None or x == '':
        return 0.0
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1
    else:
        return 0.0

# def parse_output(all_results, rounds):
#     c, g, b = "claude", "gpt3", "bard"
#     r = "_output_"+str(rounds)
    
#     for i in all_results:
#         certainty_vote = {}
            
#         for o in [c, g, b]:
#             if o+r in i:
#                 i[o+"_pred_"+str(rounds)] = i[o+r]["answer"]
#                 i[o+"_exp_"+str(rounds)] = f"I think the answer is {i[o+r]['answer']} because {i[o+r]['reasoning']} My confidence level is {i[o+r]['confidence_level']}." 
#                 if i[o+r]["answer"] not in certainty_vote:
#                     certainty_vote[i[o+r]["answer"]] = trans_confidence(i[o+r]["confidence_level"]) + 1e-5
#                 else:
#                     certainty_vote[i[o+r]["answer"]] += trans_confidence(i[o+r]["confidence_level"])
#         if c+r in i and g+r in i and b+r in i:
#             i["vote_"+str(rounds)] = [i["claude_pred_"+str(rounds)], i["gpt3_pred_"+str(rounds)], i["bard_pred_"+str(rounds)]]
#             i["exps_"+str(rounds)] = [i["claude_exp_"+str(rounds)], i["gpt3_exp_"+str(rounds)], i["bard_exp_"+str(rounds)]]
#             i["weighted_vote_"+str(rounds)] = certainty_vote
#             i["weighted_max_"+str(rounds)] = max(certainty_vote, key=certainty_vote.get)

#             i["debate_prompt_"+str(rounds)] = ""
#             vote = Counter(i["vote_"+str(rounds)]).most_common(2)
#             i["majority_ans_"+str(rounds)] = vote[0][0]
#             if len(vote) > 1: # not all the agents give the same answer
#                 for v in vote:
#                     i["debate_prompt_"+str(rounds)] += f"There are {v[1]} agents think the answer is {v[0]}. "
#                     exp_index = find_idx_by_element(i["vote_"+str(rounds)], v[0])
#                     group_exp = find_element_by_indices(i["exps_"+str(rounds)], exp_index)
#                     exp = "\n".join(["One agent solution: " + g for g in group_exp])
#                     i["debate_prompt_"+str(rounds)] += exp + "\n\n"
                    
#     return all_results


def model_parse_output(all_results, rounds):
    c, g, b = "Gemini", "gpt", "bard"
    r = "_output_" + str(rounds)
    # print(f"\nall_results_context:{all_results}")

    for i in all_results:
        certainty_vote = {}
        # print(f"\nI_context:{i}")

        for o in [c, g, b]:
            if o + r in i:
                if isinstance(i[o + r]["answer"], list):
                    if len(i[o + r]["answer"]) == 0:
                        i[o + r]["answer"] = "Unknown"
                    else:
                        i[o + r]["answer"] = ",".join(map(str, i[o + r]["answer"]))
                    print(f"\ni[o + r]['answer']:{i[o + r]['answer']}")
                # if isinstance(i[o + r]['answer'], list):
                #     i[o + r]['answer']=",".join(map(str, i[o + r]['answer']))
                #     # raise ValueError(f"\ni[o + r]['answer'] is a list:{i[o + r]['answer']}")
                #     print(f"\ni[o + r]['answer']:{i[o + r]['answer']}")
                i[o + "_pred_" + str(rounds)] = i[o + r]["answer"]
                i[o + "_exp_" + str(
                    rounds)] = f"I think the answer is {i[o + r]['answer']} because {i[o + r]['reasoning']} My confidence level is {i[o + r]['confidence_level']}."
                if i[o + r]["answer"] not in certainty_vote:
                    certainty_vote[i[o + r]["answer"]] = trans_confidence(i[o + r]["confidence_level"]) + 1e-5
                else:
                    certainty_vote[i[o + r]["answer"]] += trans_confidence(i[o + r]["confidence_level"])
        if c + r in i and g + r in i and b + r in i:
            i["vote_" + str(rounds)] = [i["Gemini_pred_" + str(rounds)], i["gpt_pred_" + str(rounds)],
                                        i["bard_pred_" + str(rounds)]]
            i["exps_" + str(rounds)] = [i["Gemini_exp_" + str(rounds)], i["gpt_exp_" + str(rounds)],
                                        i["bard_exp_" + str(rounds)]]
            i["weighted_vote_" + str(rounds)] = certainty_vote
            i["weighted_max_" + str(rounds)] = max(certainty_vote, key=certainty_vote.get)

            i["debate_prompt_" + str(rounds)] = ""
            vote = Counter(i["vote_" + str(rounds)]).most_common(2)
            i["majority_ans_" + str(rounds)] = vote[0][0]
            if len(vote) > 1:  # not all the agents give the same answer
                for v in vote:
                    i["debate_prompt_" + str(rounds)] += f"There are {v[1]} agents think the answer is {v[0]}. "
                    exp_index = find_idx_by_element(i["vote_" + str(rounds)], v[0])
                    group_exp = find_element_by_indices(i["exps_" + str(rounds)], exp_index)
                    exp = "\n".join(["One agent solution: " + g for g in group_exp])
                    i["debate_prompt_" + str(rounds)] += exp + "\n\n"

    return all_results

# def evaluate_single_model(results):
#     num_correct = 0
#     for i in results:
#         if i["gold_answer"] == i["prediction"]["answer"]:
#             num_correct+=1
#     return num_correct / len(results)


def clean_model_output(all_results, rounds):
    co, go, bo = "Gemini_output_" + str(rounds), "gpt_output_" + str(rounds), "bard_output_" + str(rounds)
    Unified_content=[]
    Unified_new_content=[]
    for i in all_results: 
        for o in [co, go, bo]:
            if o in i:
                # print(f"\n每个answer:{i[o]['answer']}")
                # Unified_content.append(i[o]["answer"])
                if "reasoning" not in i[o]:
                    i[o]["reasoning"] = ""
                elif type(i[o]["reasoning"]) is list:
                    i[o]["reasoning"] = " ".join(i[o]["reasoning"])

                if "answer" not in i[o]:
                    raise ValueError("answer is not here!")

                if "confidence_level" not in i[o] or not i[o]["confidence_level"]:
                    i[o]["confidence_level"] = 0.0
                else:
                    if type(i[o]["confidence_level"]) is str and "%" in i[o]["confidence_level"]:
                        i[o]["confidence_level"] = float(i[o]["confidence_level"].replace("%", "")) / 100
                    else:
                        try:
                            i[o]["confidence_level"] = float(i[o]["confidence_level"])
                            # print(f"i[o]['confidence_level']:{i[o]['confidence_level']}")
                        except:
                            # print(i[o]["confidence_level"])
                            i[o]["confidence_level"] = 0.0

    # if need_judge:
    #     submit_msg=[{"role": "user", "content": str(Unified_content)+"""Standardize the format and content of the parts within this data that are very similar in content or have the same meaning.Your output should also include three elements.
    #                                                         Your output should follow this format strictly:
    #                                                         {<context_A>,<context_B>,<context_C>}
    #                                                         For example,you should Turn ["A","B","C"] into {"A","B","C"},Turn ["A","B","B"] into {"A","B","B"},
    #                                                         Turn ["A","A","A"] into {"A","A","A"},Turn ["B","B","B"] into {"B","B","B"},
    #                                                         Turn ["lung disease","Lung Disease","heart disease"] into {"lung disease","lung disease","heart disease"},Turn ["lung disease","lung cancer","heart disease"] into {"lung disease","lung disease","heart disease"},
    #                                                         Turn ["Yes","no","yes"] into {"yes","no","yes"},Turn ["Yes","no","maybe"] into {"Yes","no","maybe"},
    #                                                         Turn [] into {},
    #                                                         If there is content in the output, Your output should also include three elements.Your output should follow this format strictly, No other Comments! 
    #                                                     """}]
    #     msg,num_prompt,num_completion=batch_manager.submit_request(submit_msg)
    #     if "{" not in msg and "}" not in msg:
    #             # Bard sometimes doesn"t follow the instruction of generate a JSON format output
    #             print("解析失败")
    #             raise ValueError("cannot find { or } in the model output.")
    #     else:
    #         try:
    #             msg = msg.replace("\n", " ")
    #             msg = re.search('({.+})', msg).group(0)
    #             msg = ast.literal_eval(msg)  
    #         except (SyntaxError, ValueError) as e:
    #             print(f"解析失败: {e}")
    #             raise ValueError(f"\n输入解析内容:{str(Unified_content)},输出解析内容:{str(msg)}")
    #     if len(msg) == 1:
    #         msg = list(msg) * 3 
    #     elif len(msg) == 2:
    #         msg = list(msg)
    #         msg.append(msg[1])
    #     if len(msg)!=3:
    #         raise ValueError(f"\n输入解析内容:{str(Unified_content)},输出解析内容:{str(msg)}")
    #     for message in msg:
    #         Unified_new_content.append(message)

    #     for i in all_results:
    #         for idx, o in enumerate([co, go, bo]):
    #             if o in i:
    #                 i[o]["answer"]=str(Unified_new_content[idx])

        
    #     return all_results,num_prompt,num_completion,1
    # else:
    return all_results,0,0,0


