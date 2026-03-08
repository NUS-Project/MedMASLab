from openai import OpenAI
from methods.utils import get_apikey_and_baseurl_from_configs,chat_content

class SelfConsistency_model:
    def __init__(self, model_info, root_path):
        self.model_info = model_info
        api_key, base_url = get_apikey_and_baseurl_from_configs(root_path, model_info)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, message, image_path):
        cot_prompt = "let's think step by step."
        question = message
        message = message+cot_prompt
        messages = [{"role": "system", "content": "You are a helpful medical assistant."}]
        if image_path is not None:
            messages.append(chat_content(image_path, message))
        else:
            messages.append({"role": "user", "content": message})

        num_responses = 5
        token_stats = {
            "SelfConsistency": {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }
        resp=None
        for num in range(num_responses):
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=messages,
                max_tokens=5120,
                temperature=0.8
            )
            token_stats["SelfConsistency"]["num_llm_calls"] += 1
            token_stats["SelfConsistency"]["prompt_tokens"] += response.usage.prompt_tokens
            token_stats["SelfConsistency"]["completion_tokens"] += response.usage.completion_tokens
            resp=f"This is Agent {num+1}'s thought process and answer:"+response.choices[0].message.content

        instructions=[{"role": "system", "content": "You are a helpful medical assistant."}]
        instruction = f"Here are the views and answers of {num_responses}agents regarding the same question:"+question+ resp+f"Now,Given you the question: \n{question} and all the above {num_responses} agents's views about this question, reason over them carefully and provide a final answer to the question."
        # instructions.append({"role": "user", "content": instruction})
        if image_path is not None:
            instructions.append(chat_content(image_path, instruction))
        else:
            instructions.append({"role": "user", "content": instruction})
        response = self.client.chat.completions.create(
            model=self.model_info,
            messages=instructions,
            max_tokens=5120,
        )
        prompt_tokens, completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        token_stats["SelfConsistency"]["num_llm_calls"] += 1
        token_stats["SelfConsistency"]["prompt_tokens"] += prompt_tokens
        token_stats["SelfConsistency"]["completion_tokens"] += completion_tokens

        return response.choices[0].message.content,token_stats



def SelfConsistency_test(root_path,model_info,message, image_path):
    SelfConsistency_test=SelfConsistency_model(model_info, root_path)
    final_decision, token_stats=SelfConsistency_test.chat(message, image_path)
    return final_decision, token_stats




