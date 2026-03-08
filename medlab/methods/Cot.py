from openai import OpenAI
from methods.utils import get_apikey_and_baseurl_from_configs,chat_content

class CoT_model:
    def __init__(self, model_info, root_path):
        self.model_info = model_info
        api_key, base_url = get_apikey_and_baseurl_from_configs(root_path, model_info)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, message, image_path):
        cot_prompt = "let's think step by step."
        message = message+cot_prompt
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if image_path is not None:
            messages.append(chat_content(image_path, message))
        else:
            messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model_info,
            messages=messages,
            max_tokens=5120,
        )
        prompt_tokens, completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        token_stats = {
            "CoT_model": {
                "num_llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        }

        return response.choices[0].message.content,token_stats


# class Voting_model:
#     def __init__(self, model_info, root_path):
#         self.model_info = model_info
#         api_key, base_url = get_apikey_and_baseurl_from_configs(root_path, model_info)
#         self.client = OpenAI(api_key=api_key, base_url=base_url)
#
#     def chat(self, message, image_path):
#         cot_prompt = "let's think step by step and generate the answer."
#         # message = message+cot_prompt
#         messages = [{"role": "system", "content": "You are a helpful assistant."}]
#         if image_path is not None:
#             messages.append(chat_content(image_path, message+cot_prompt))
#         else:
#             messages.append({"role": "user", "content": message+cot_prompt})
#         token_stats = {
#             "CoT_SC_model": {
#                 "num_llm_calls": 0,
#                 "prompt_tokens": 0,
#                 "completion_tokens": 0
#             }
#         }
#         # responses = []
#         num_responses = 5  # 这里可以设置生成的回答数量
#         sc_prompt = f"You are a helpful assistant.Here are the thought processes and answers of {num_responses} agents regarding the same question. You need to identify their final answers and count the number of occurrences for each answer. Then, return the answer that appears the most frequently.If there are multiple answers of the same frequency, then randomly select one of them to return."
#         responses = [{"role": "system", "content": sc_prompt}]
#         for num in range(num_responses):
#             response = self.client.chat.completions.create(
#                 model=self.model_info,
#                 messages=messages,
#                 max_tokens=5120,
#                 temperature=0.8
#             )
#             token_stats["CoT_SC_model"]["num_llm_calls"]+=1
#             token_stats["CoT_SC_model"]["prompt_tokens"]+=response.usage.prompt_tokens
#             token_stats["CoT_SC_model"]["completion_tokens"]+=response.usage.completion_tokens
#             resp=f"This is Agent{num+1}'s thought process and answer:"+response.choices[0].message.content
#             if num==(num_responses-1):
#                 resp=resp+"You need to identify their final answers and count the number of occurrences for each answer. Then, return the answer that appears the most frequently.If there are multiple answers of the same frequency, then randomly select one of them to return."
#             responses.append({"role": "user", "content": resp})
#
#         response = self.client.chat.completions.create(
#             model=self.model_info,
#             messages=responses,
#             max_tokens=512,
#         )
#         token_stats["CoT_SC_model"]["num_llm_calls"] += 1
#         token_stats["CoT_SC_model"]["prompt_tokens"] += response.usage.prompt_tokens
#         token_stats["CoT_SC_model"]["completion_tokens"] += response.usage.completion_tokens
#
#         # responses.append()
#
#
#         # response = self.client.chat.completions.create(
#         #     model=self.model_info,
#         #     messages=messages,
#         #     max_tokens=5120,
#         # )
#         # prompt_tokens, completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
#         # token_stats = {
#         #     "CoT_model": {
#         #         "num_llm_calls": 1,
#         #         "prompt_tokens": prompt_tokens,
#         #         "completion_tokens": completion_tokens
#         #     }
#         # }
#
#         return response.choices[0].message.content,token_stats

def Cot_test(root_path,model_info,message, image_path):
    Cot_model_test=CoT_model(model_info, root_path)
    final_decision, token_stats=Cot_model_test.chat(message, image_path)
    return final_decision, token_stats




