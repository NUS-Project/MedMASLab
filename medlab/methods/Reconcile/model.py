import os
import time
from tkinter import N
# from tqdm import tqdm
from methods.utils import get_apikey_and_baseurl_from_configs
from methods.Reconcile.utils import parse_json
from openai import OpenAI
from methods.utils import setup_model,qwen_vl_chat_content
from methods.thread import qwen_generate_content

# remember="Remember:Your output should follow this format strictly: {"reasoning":< your reasoning>, "answer":<your answer>, "confidence_level": < your confidence_level number>}."

def add_options():
    # if need_judge:
    return "Rmemeber:You should follow this format strictly.No other Comments!"
    # else:
    # return """Rmemeber: . If there are options for this question, your answer part should be A/B/C/D/E/F.\nFor Example: {"reasoning":<your reasoning>, "answer":"A", "confidence_level": 0.55} or {"reasoning":<your reasoning>, "answer":"C", "confidence_level": 0.76}.You should follow this format strictly.No other Comments!"""


class Reconcile_Model:
    def __init__(self, root_path,model_info,batch_manager):
        # self.client_gemini = genai.Client()
        # self.need_judge=need_judge
        self.batch_manager=batch_manager
        # self.video_batch_manager=video_batch_manager
        self.model_info=model_info
        # if self.model_info in ["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        #     self.gpt_model_info=model_info
        #     self.bard_model_info = "gemini-2.5-flash"
        #     self.gemini_model_info = "gemini-2.0-flash"
        #     gpt_api_key, gpt_base_url = get_apikey_and_baseurl_from_configs(root_path, self.gpt_model_info)
        #     bard_api_key, bard_base_url = get_apikey_and_baseurl_from_configs(root_path, self.bard_model_info)
        #     gemini_api_key, gemini_base_url =get_apikey_and_baseurl_from_configs(root_path, self.gemini_model_info)
        #     self.client_gpt = OpenAI(api_key=gpt_api_key, base_url=gpt_base_url)
        #     self.client_gemini = OpenAI(api_key=gemini_api_key, base_url=gemini_base_url)
        #     self.client_bard = OpenAI(api_key=bard_api_key, base_url=bard_base_url)

        self.token_stats = {
            "ReConcile": {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }

    def get_token_stats(self):
        return self.token_stats

    def Gemini_gen_ans(self, sample, convincing_samples=None, additional_instruc=None, intervene=False, dataset="SQA",img_path=None):
        # contexts = prepare_context(sample, convincing_samples, intervene, dataset)
        if convincing_samples is not None:
            contexts = convincing_samples + sample + """Please answer the question with step-by-step reasoning.Also, evaluate your confidence level (between 0.00 and 1.00) to indicate the possibility of your answer being right.You must be precise to two decimal places.
                                   Your output should follow this format strictly: 
                                   {"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}.
                                    .
                                """+add_options()
        else:
            contexts = sample + """Please answer the question with step-by-step reasoning.Also, evaluate your confidence level (between 0.00 and 1.00) to indicate the possibility of your answer being right.You must be precise to two decimal places.
                                   Your output should follow this format strictly: 
                                   {"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. 
                                    .
                                """+add_options()
        if additional_instruc:
            contexts += " ".join(additional_instruc)
        # print(f"\nGemini_gen_input: {contexts}")
        messages = [{"role": "system", "content": f"You are a helpful assistant.{add_options()}"}]
        # messages.append({"role": "user", "content": contexts})
        # if self.model_info in ["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        #     messages.append({"role": "user", "content": contexts})
        #     output = self.client_gemini.chat.completions.create(
        #         model=self.gemini_model_info,
        #         messages=messages,
        #         max_tokens=5120,
        #     )
            # prompt_tokens, completion_tokens = output.usage.prompt_tokens, output.usage.completion_tokens
            # output = output.choices[0].message.content
        # elif "Qwen" in self.model_info or "LLaVA" in self.model_info:
        messages.append(qwen_vl_chat_content(img_path,contexts)) 
        output,prompt_tokens,completion_tokens=qwen_generate_content(messages,self.batch_manager)

        self.token_stats["ReConcile"]["num_llm_calls"] += 1
        self.token_stats["ReConcile"]["prompt_tokens"] += prompt_tokens
        self.token_stats["ReConcile"]["completion_tokens"] += completion_tokens
        # print(f"\nGemini_gen_ans_output: {output}")

        # if output:
        result = parse_json(output)
        # print(f"\nGemini_gen_ans_result: {result}")
        # if result == "ERR_SYNTAX":
        #     # print("incomplete JSON format.")
        #     # print(output)
        #     # print("-----------------------")
        #     raise ValueError(f"incomplete JSON format.{output}")
        # # print(f"\nGemini_gen_ans: {result}")
        return result

    def Gemini_debate(self, test_samples, all_results, rounds, convincing_samples,img_path=None):
        r = "_" + str(rounds - 1)
        result = None
        for i, s in enumerate(all_results):
            # print(f"\ni:{i}")
            # print(f"\ns:{s}")
            # print("\nhere!")
            if "Gemini_output_" + str(rounds) not in s and "debate_prompt" + r in s and len(s["debate_prompt" + r]):
                additional_instruc = [
                    "\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question."]
                additional_instruc.append("Clearly states that which pointview do you agree or disagree and why.\n\n")
                additional_instruc.append(s["debate_prompt" + r])
                additional_instruc.append(
                    """ Output your answer in json format, with the format as follows:{"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. Please strictly output in JSON format.{add_options()}""")
                # print(f"\nadditional_instruc: {additional_instruc}\n")
                result = self.Gemini_gen_ans(sample=test_samples[i],
                                             convincing_samples=convincing_samples,
                                             additional_instruc=additional_instruc,
                                             intervene=False,img_path=img_path)
                if result is None:
                    raise ValueError("Gemini_gen_ans failed!")
        return all_results


    def gpt_gen_ans(self,sample, convincing_samples=None, additional_instruc=None, intervene=False, dataset="SQA",img_path=None):
        # contexts = prepare_context_for_chat_assistant(sample, convincing_samples, intervene, dataset)
        if convincing_samples is not None:
            contexts = convincing_samples+sample + """Please answer the question with step-by-step reasoning.Also, evaluate your confidence level (between 0.00 and 1.00) to indicate the possibility of your answer being right.You must be precise to two decimal places.
                                   Your output should follow this format strictly: 
                                   {"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. 
                                    .
                                """+add_options()
        else:
            contexts = sample + """Please answer the question with step-by-step reasoning.Also, evaluate your confidence level (between 0.00 and 1.00) to indicate the possibility of your answer being right.You must be precise to two decimal places.
                                   Your output should follow this format strictly: 
                                   {"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. 
                                    .
                                """+add_options()
        if additional_instruc:
            contexts += " ".join(additional_instruc)
        # print(f"\ncontexts: {contexts}")
        messages = [{"role": "system", "content": f"You are a helpful assistant.{add_options()}"}]
        # if self.model_info in ["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        #     messages.append({"role": "user", "content": contexts})
        #     output = self.client_gpt.chat.completions.create(
        #         model=self.gpt_model_info,
        #         messages=messages,
        #         max_tokens=5120,
        #     )
            # prompt_tokens, completion_tokens =output.usage.prompt_tokens,output.usage.completion_tokens
            # output=output.choices[0].message.content
        # elif "Qwen" in self.model_info or "LLaVA" in self.model_info:
        messages.append(qwen_vl_chat_content(img_path,contexts))
        output,prompt_tokens,completion_tokens=qwen_generate_content(messages,self.batch_manager)
        self.token_stats["ReConcile"]["num_llm_calls"] += 1
        self.token_stats["ReConcile"]["prompt_tokens"] += prompt_tokens
        self.token_stats["ReConcile"]["completion_tokens"] += completion_tokens
        #
        # if output:
        #     if "{" not in output or "}" not in output:
        #         raise ValueError("cannot find { or } in the model output.")
        result = parse_json(output)
            # print(f"\nresult here: {result}")
            # if result == "ERR_SYNTAX":
            #     raise ValueError("incomplete JSON format.")
        # print(f"\ngpt_gen_ans: {result}")
        return result


    def bard_gen_ans(self,sample, convincing_samples=None, additional_instruc=None, intervene=False, dataset="SQA",img_path=None):
       
        if convincing_samples is not None:
            msg = convincing_samples+sample +  """Please answer the question with step-by-step reasoning.Also, evaluate your confidence level (between 0.00 and 1.00) to indicate the possibility of your answer being right.You must be precise to two decimal places.
                                   Your output should follow this format strictly: 
                                   {"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. 
                                    .
                                """+add_options()
        else:
            msg = sample +  """Please answer the question with step-by-step reasoning.Also, evaluate your confidence level (between 0.00 and 1.00) to indicate the possibility of your answer being right.You must be precise to two decimal places.
                                   Your output should follow this format strictly: 
                                   {"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. 
                                    .
                                """+add_options()
        if additional_instruc:
            msg += " ".join(additional_instruc)
        # gen_plam = AI_model(model_info="gpt-4o-mini")
        # output = gen_plam.chat(msg)
        # print(f"\nbard_input:{msg}")
        messages = [{"role": "system", "content": f"You are a helpful assistant.{add_options()}"}]
        
        # if self.model_info in ["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        #     messages.append({"role": "user", "content": msg})
        #     output = self.client_bard.chat.completions.create(
        #         model=self.bard_model_info,
        #         messages=messages,
        #         max_tokens=5120,
        #     )
        #     prompt_tokens, completion_tokens = output.usage.prompt_tokens, output.usage.completion_tokens
        #     output = output.choices[0].message.content
        # elif "Qwen" in self.model_info or "LLaVA" in self.model_info:
        messages.append(qwen_vl_chat_content(img_path,msg))
        output,prompt_tokens,completion_tokens=qwen_generate_content(messages,self.batch_manager)
        self.token_stats["ReConcile"]["num_llm_calls"] += 1
        self.token_stats["ReConcile"]["prompt_tokens"] += prompt_tokens
        self.token_stats["ReConcile"]["completion_tokens"] += completion_tokens
        

        # if not output:
        #     raise ValueError
        # if "{" not in output and "}" not in output:
        #     # Bard sometimes doesn"t follow the instruction of generate a JSON format output
        #     print("parsing the output into json format using bard...")
        #     raise ValueError("cannot find { or } in the model output.")
        # else:
        result = parse_json(output)
            # if result == "ERR_SYNTAX":
            #     raise ValueError("incomplete JSON format.")
        # print(f"\nbard_gen_ans: {result}")
        return result


    def gpt_debate(self,test_samples, all_results, rounds, convincing_samples,img_path=None):
        r = "_" + str(rounds - 1)
        for i, s in enumerate(all_results):
            if "gpt_output_" + str(rounds) not in s and "debate_prompt" + r in s and len(s["debate_prompt" + r]):
                additional_instruc = [
                    "\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question."]
                additional_instruc.append("Clearly states that which pointview do you agree or disagree and why.\n\n")
                additional_instruc.append(s["debate_prompt" + r])
                additional_instruc.append(""" Output your answer in json format, with the format as follows:{"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. Please strictly output in JSON format.{add_options()}""")
                    # "Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
                result = self.gpt_gen_ans(sample=test_samples[i],
                                     convincing_samples=convincing_samples,
                                     additional_instruc=additional_instruc,
                                     intervene=False,img_path=img_path)
                s["gpt_output_" + str(rounds)] = result
        return all_results


    def bard_debate(self,test_samples, all_results, rounds, convincing_samples,img_path=None):
        r = "_" + str(rounds - 1)
        for i, s in enumerate(all_results):
            if "bard_output_" + str(rounds) not in s and "debate_prompt" + r in s and len(s["debate_prompt" + r]):
                additional_instruc = [
                    "\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question."]
                additional_instruc.append("Clearly states that which pointview do you agree or disagree and why.\n\n")
                additional_instruc.append(s["debate_prompt" + r])
                additional_instruc.append(""" Output your answer in json format, with the format as follows:{"reasoning":<your reasoning>, "answer": <your answer>, "confidence_level": <your confidence_level number >}. Please strictly output in JSON format.{add_options()}""")
                    # "Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")

                result = self.bard_gen_ans(test_samples[i],
                                      convincing_samples=convincing_samples,
                                      additional_instruc=additional_instruc,
                                      intervene=False,img_path=img_path)
                # except ValueError:
                #     print("cannot generate valid answer for this sample.")
                #     result = invalid_result(dataset)

                s["bard_output_" + str(rounds)] = result
                # time.sleep(1)
        return all_results

