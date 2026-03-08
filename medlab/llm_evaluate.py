import os
from openai import OpenAI
from methods.utils import chat_content
import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoModelForCausalLM, AutoTokenizer

from qwen_vl_utils import process_vision_info
from methods.utils import qwen_vl_chat_content,setup_model,qwen_vl_generate_content


class JudgeModel:
    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def chat(self, query, response, right_label,right_context,img_path,answer_type):
        # 发起聊天请求
        if answer_type.lower() == "closed":
            answer_type='There is only one correct answer content and the corresponding correct answer label.'
            prompt = f"You are a diligent and precise medical assistant tasked with evaluating the correctness of responses. \nYou will receive a question, an output sentence from agent, the correct answer content, and the corresponding correct answer label.{answer_type} Your tasks are as follows:\n1.Determine if you can find the agent's answer in the output sentence from agent. If you can not find the agent's answer in the output sentence from agent, respond with [wrong].\n2.If you find the agent's answer, evaluate whether it accurately answers the question based on the provided right answer content and right answer label. Respond with either [right] or [wrong].\nREMEMBER: Your judgment result must be only [right], or [wrong]."
        else:
            answer_type = 'This question has more than one correct answer and corresponding labels for the correct answers.This reference answer is for reference only.'
            prompt = f"You are a diligent and precise medical assistant tasked with evaluating the correctness of responses. \nYou will receive a question, an output sentence from agent, the correct reference answer content, and the corresponding correct reference answer label.{answer_type} Your tasks are as follows:\n1.Determine if you can find the agent's answer in the output sentence from agent. If you can not find the agent's answer in the output sentence from agent, respond with [wrong].\n2.If you find the agent's answer, evaluate whether it accurately answers the question based on the provided right reference answer content, right reference answer label and your own thinking for the question. Respond with either [right] or [wrong].\nREMEMBER: Your judgment result must be only [right], or [wrong]."
        content = [{"role": "system", "content": prompt}]
        if img_path is not None:
            content.append(chat_content(img_path, format_prompt(query, response, right_label,right_context)))
        else:
            content.append({"role": "user", "content": format_prompt(query, response, right_label,right_context)})
        response = self.client.chat.completions.create(
            model= self.model_id,
            messages=content,
            stream=False
        )

        # 返回助手的响应
        return response.choices[0].message.content



class QwenVL_JudgeModel:
    def __init__(self, judge_model,root_path: str, device: str):
        # elif self.model_info =='Qwen2.5-VL-7B-Instruct':
        # self.model, self.processor = setup_model(judge_model,root_path)
        self.device = device
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        #     trust_remote_code=True,
        # )
        # # print(model_path)
        # self.processor = AutoProcessor.from_pretrained(model_path)

    def chat(self, query, response, right_label,right_context,img_path,answer_type,Judge_batch_manager):
        # 发起聊天请求
        if answer_type.lower() == "closed":
            answer_type='There is only one right answer content and the corresponding right answer label.'
            prompt = f"You are a diligent and precise medical assistant tasked with evaluating the correctness of responses. \nYou will receive a question, an output sentence from agent, the right answer content, and the corresponding right answer label.{answer_type} Your tasks are as follows:\n1.Determine if you can find the agent's answer in the output sentence from agent. If you can not find the agent's answer in the output sentence from agent, respond with [wrong].\n2.If you find the agent's answer, evaluate whether it accurately answers the question based on the provided right answer content and right answer label. Respond with either [right] or [wrong].\nREMEMBER: Your judgment result must be only [right] or [wrong].No other comments!"
        elif answer_type.lower() == "mca":
            answer_type = 'This question has more than one right answer and corresponding labels for the right answers.These answers are fixed.'
            prompt = f"You are a diligent and precise medical assistant tasked with evaluating the correctness of responses. \nYou will receive a question, an output sentence from agent, the right answers' content, and the corresponding right answer labels.{answer_type} Your tasks are as follows:\n1.Determine if you can find the agent's answer in the output sentence from agent. If you can not find the agent's answer in the output sentence from agent, respond with [wrong].\n2.If you find the agent's answer, evaluate whether it accurately answers the question based on the provided right answer contents and right answer labels.If any correct answer is missing from the agents'answer or any incorrect answer is included, respond with [wrong]. Respond with either [right] or [wrong].\nREMEMBER: Your judgment result must be only [right] or [wrong].No other comments!"
        else:
            answer_type = 'This question has more than one right answer and corresponding labels for the right answers.This reference answer is for reference only.'
            prompt = f"You are a diligent and precise medical assistant tasked with evaluating the correctness of responses. \nYou will receive a question, an output sentence from agent, the right reference answer content, and the corresponding right reference answer label.{answer_type} Your tasks are as follows:\n1.Determine if you can find the agent's answer in the output sentence from agent. If you can not find the agent's answer in the output sentence from agent, respond with [wrong].\n2.If you find the agent's answer, evaluate whether it accurately answers the question based on the provided right reference answer content, right reference answer label and your own thinking for the question. Respond with either [right] or [wrong].\nREMEMBER: Your judgment result must be only [right] or [wrong].No other comments!"
        # content = [{"role": "system", "content": prompt}]
        messages = [
            {
                "role": "system",
                 "content": [{"type": "text", "text": prompt}]
        
            }
        ]
        # if img_path:
        messages.append(qwen_vl_chat_content(img_path,format_prompt(query, response, right_label,right_context)))
        # else:
        #      messages.append({"role": "user", "content": format_prompt(query, response, right_label,right_context)})
        completion,prompt_tokens,completion_tokens=Judge_batch_manager.submit_request(messages)
        return completion



def format_prompt(query, response, right_label,right_context):
    prompt = f'''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, the correct answer content, and the corresponding correct answer label. Your tasks are as follows:\n1.Determine if the output sentence is complete. If it is incomplete, respond with [wrong].\n2.If the output sentence is complete, evaluate whether it accurately answers the question based on the provided right answer content and right answer label. Respond with either [right] or [wrong].\nREMEMBER: Your judgment result must be only [right] or [wrong].
-
Special considerations:

1. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered right if it clearly indicates the right option's code or the right option's content.

2. **No Explicit Options**: If the question does not provide explicit options, you need to actively judge whether this output is right based on the reference answer.
-

Question: {str(query)}

Output sentence of agents: {str(response)}
'''

    if right_label:
        prompt += f'right answer label: {str(right_label)}\n'

    prompt += f'right answer context: {str(right_context)}\n\nJudgement Result:'
    return prompt