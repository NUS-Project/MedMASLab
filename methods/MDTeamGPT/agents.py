from typing import List, Dict, Any, Callable
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from methods.MDTeamGPT.tools import MedicalTools
from methods.utils import get_apikey_and_baseurl_from_configs, encode_image, chat_content
from methods.utils import setup_model, qwen_vl_generate_content, qwen_vl_chat_content, is_options
from methods.thread import qwen_generate_content

SPECIALIST_POOL = [
    "General Internal Medicine Doctor",
    "General Surgeon",
    "Pediatrician",
    "Obstetrician and Gynecologist",
    "Radiologist",
    "Neurologist",
    "Pathologist",
    "Pharmacist"
]


class MDTAgents:
    def __init__(self, api_key_vl=None, base_url_vl=None, api_key_text=None, base_url_text=None, text_model=None,
                 vl_model=None, enable_tools=True, root_path=None, batch_manager=None):
        # self.vl_model=vl_model
        # self.text_model=text_model
        # self.video_batch_manager=video_batch_manager
        self.batch_manager = batch_manager
        # if api_key_text is not None and base_url_text is not None :
        #     self.llm = ChatOpenAI(
        #         model=text_model,
        #         api_key=api_key_text,
        #         base_url=base_url_text,
        #         temperature=0.7,
        #         streaming=True,
        #         stream_usage=True
        #     )
        #     self.critic_llm = ChatOpenAI(
        #         model=text_model,
        #         api_key=api_key_text,
        #         base_url=base_url_text,
        #         temperature=0.0,
        #         streaming=False
        #     )
        # if api_key_vl is not None and base_url_vl is not None :
        #     self.vl_llm = ChatOpenAI(
        #         model=vl_model,
        #         api_key=api_key_vl,
        #         base_url=base_url_vl,
        #         temperature=0.1,
        #         max_tokens=2048,
        #         streaming=True,
        #         stream_usage=True

        #     )

        self.tools = MedicalTools(enable=enable_tools)
        # Callbacks
        self.stream_callback = None
        self.tool_callback = None
        self.token_stats = {
            "MDTeamGPT": {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }

    def set_stream_callback(self, callback: Callable[[str, str], None]):
        self.stream_callback = callback

    def set_tool_callback(self, callback: Callable[[str, str, str], None]):
        self.tool_callback = callback

    def get_token_stats(self):
        return self.token_stats

    # 1. Primary Care (Triage)
    def primary_care_doctor(self, case_info: str) -> Dict[str, Any]:
        msg = f"""You are a Primary Care Doctor at the Triage Desk.
            Analyze the patient case and select the most appropriate specialists.

            Available Specialists:
            {SPECIALIST_POOL}

            Patient Case: {case_info}

            TASK:
            1. Explain your reasoning.
            2. Select AT LEAST 3 specialists.

            OUTPUT JSON FORMAT:
            {{
                "reasoning": "...",
                "selected_roles": ["Role A", "Role B", "Role C"...]
            }}
            """
        # if "Qwen" in self.text_model or "LLaVA" in self.text_model: #qwen_generate_content(messages,batch_manager,video_batch_manager)
        content, prompt_tokens, completion_tokens = qwen_generate_content(
            [qwen_vl_chat_content(image_path=None, message=msg)], self.batch_manager)
        self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        self.token_stats["MDTeamGPT"]["prompt_tokens"] += prompt_tokens
        self.token_stats["MDTeamGPT"]["completion_tokens"] += completion_tokens
        #  print(f"\nPrimary_care_doctor_content:\n{content}")
        #  print(f"\nResult: {content}")
        # else:
        #     prompt = ChatPromptTemplate.from_template(msg)
        #     chain = prompt | self.llm
        #     result = chain.invoke(prompt)
        #     self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        #     self.token_stats["MDTeamGPT"]["prompt_tokens"] += result.usage_metadata["input_tokens"]
        #     self.token_stats["MDTeamGPT"]["completion_tokens"] += result.usage_metadata["output_tokens"]
        #     # print(f"\nResult: {result}")
        #     content = result.content.strip()
        # print(f"\ntoken_count:{result.usage_metadata}")

        start_idx = content.find('{')
        end_idx = content.find('}', start_idx + 1)  # 只在 { 后面搜索 }
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]
        # if content.startswith("```json"): content = content[7:]
        # if content.endswith("```"): content = content[:-3]
        # print(f"\nCleaned Content: {content}")

        try:
            data = json.loads(content)
            selected = [s for s in data.get("selected_roles", []) if s in SPECIALIST_POOL]
            remaining = [s for s in SPECIALIST_POOL if s not in selected]
            while len(selected) < 3 and remaining:
                selected.append(remaining.pop(0))
            data["selected_roles"] = selected
            # print(f"\ndata: {data}")
            return data
        except:
            return {
                "reasoning": "Fallback selection.",
                "selected_roles": ["General Internal Medicine Doctor", "General Surgeon", "Radiologist"]
            }

    # 2. Specialists (Consultation)
    def specialist_consult(self, role: str, case_info: str, residual_context: str,
                           image_data=None, round_num=1):

        # Tool Usage Logic
        tool_context = ""
        if self.tools.enable:
            try:
                # if "Qwen" in self.text_model or "LLaVA" in self.text_model:
                msg = f"Extract 1 specific medical query string for {role} to research regarding: {case_info}. Return ONLY the query."
                # kw,prompt_tokens,completion_tokens=qwen_vl_generate_content(self.critic_model,self.critic_processor,[qwen_vl_chat_content(image_path=image_data,message=msg)])
                kw, prompt_tokens, completion_tokens = qwen_generate_content(
                    [qwen_vl_chat_content(image_path=image_data, message=msg)], self.batch_manager)

                self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
                self.token_stats["MDTeamGPT"]["prompt_tokens"] += prompt_tokens
                self.token_stats["MDTeamGPT"]["completion_tokens"] += completion_tokens
                # print(f"\nsearch_query:{kw}")
                # else:
                #     kw_prompt = ChatPromptTemplate.from_template(
                #         "Extract 1 specific medical query string for {role} to research regarding: {case}. Return ONLY the query.")
                #     kw_chain = kw_prompt | self.critic_llm
                #     kw = kw_chain.invoke({"case": case_info[:300], "role": role})
                #     self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
                #     self.token_stats["MDTeamGPT"]["prompt_tokens"] += kw.usage_metadata["input_tokens"]
                #     self.token_stats["MDTeamGPT"]["completion_tokens"] += kw.usage_metadata["output_tokens"]
                #     kw = kw.content
                # print(f"\nsearch_query:{kw}")

                if kw and "no query" not in kw.lower():
                    tool_res = self.tools.run_tools(kw)
                    if tool_res:
                        if self.tool_callback:
                            self.tool_callback(role, kw, tool_res)
                        tool_context = f"\n[External Tools Data]:\n{tool_res}\n"
            except Exception as e:
                print(f"Tool error: {e}")

        # Strict Reasoning Structure
        structure_instruction = """
        IMPORTANT INSTRUCTIONS:
        1. **Independence**: You are providing your opinion INDEPENDENTLY. You cannot see the opinions of other specialists in this current round. You can only see the summary of previous rounds (if any).
        2. **Blindness**: You do NOT have access to the ground truth or final correct diagnosis. Rely only on the case description and your knowledge.
        3. **Structure**: You must structure your response in exactly three sections:

           - **1. Context Summary**: 
             (If Round 1: Summarize "Prior Knowledge". If Round > 1: Summarize "Residual Context" from previous rounds.)

           - **2. Clinical Reasoning**: 
             (Analyze the case. If tool data exists, use it. If image exists, describe findings. Explain step-by-step.)

           - **3. Conclusion**: 
             (State your clear medical opinion or diagnosis.)
        """
        #         """
        # 重要说明：
        # 1. **独立性**：您正在自主发表自己的观点。在本次轮次中，您无法看到其他专家的意见。您只能看到之前轮次的总结（如果有）。
        # 2. **失明**：您无法获取真实情况或最终的正确诊断结果。请仅依据病例描述以及您的专业知识来进行判断。
        # 3. **结构要求**：您的回复必须严格按照以下三个部分进行组织：
        # - **1. 背景概述**：
        # （如果为第 1 轮：总结“已知信息”。如果轮数大于 1：总结前几轮的“剩余背景信息”。）
        # - **2. 临床推理**：
        # （分析病例。如果有工具数据，可加以利用。如果有影像资料，则描述其发现情况。请按步骤进行详细说明。）
        # - **3. 结论**：
        # （陈述您明确的医疗意见或诊断结果。）"""
        system_prompt = f"You are a {role}. Provide expert medical opinion.\n{structure_instruction}"

        user_text = f"Question: {case_info}\n{tool_context}\n"

        if round_num == 1:
            user_text += "\n[Status]: Round 1. Analyze independently."
            user_text += f"\n*** PRIOR KNOWLEDGE / CONTEXT ***\n{residual_context}\n"
            if image_data:
                user_text += " [Image Provided]. Describe findings and integrate with diagnosis."
            else:
                user_text += " No image provided."
        else:
            user_text += f"\n[Status]: Round {round_num}.\n"
            user_text += f"*** RESIDUAL CONTEXT (Previous Rounds) ***\n{residual_context}\n"
            user_text += "Review the summaries of previous rounds. Support, refute, or synthesize based on that history."

        # if "Qwen" in self.text_model or "LLaVA" in self.text_model:
        messages = [qwen_vl_chat_content(message=system_prompt)]
        # target_llm,target_llm_processor=self.llm_model, self.llm_processor
        if round_num == 1 and image_data:
            # target_llm,target_llm_processor=self.vl_model, self.vl_processor
            # content_payload=[qwen_vl_chat_content(image_data,user_text)]
            messages.append(qwen_vl_chat_content(image_data, user_text))
        else:
            messages.append(qwen_vl_chat_content(message=user_text))
        try:
            full_res = ""
            aggregate = None
            full_res, prompt_tokens, completion_tokens = qwen_generate_content(messages, self.batch_manager)
            self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
            self.token_stats["MDTeamGPT"]["prompt_tokens"] += prompt_tokens
            self.token_stats["MDTeamGPT"]["completion_tokens"] += completion_tokens
            # print(f"\nspecialist_consult_content:\n{full_res}")
            # print(f"\naggregate_token:{aggregate.usage_metadata}")
            if self.stream_callback:
                self.stream_callback(role, full_res)
            return full_res
        except Exception as e:
            return f"Error: {e}"

    # 3. Lead Physician
    def lead_physician_synthesis(self, round_dialogues: List[str], round_num: int):
        # Lead Physician DOES see all dialogues from the current round (to synthesize them),
        # but DOES NOT see Ground Truth.
        # print(f"\nround_dialogues:{round_dialogues}")
        msg = f"""You are the Lead Physician.
            Synthesize the specialists' discussions from Round {round_num} into a concise structured summary.

            Specialists' Output (Current Round):
            {round_dialogues}

            TASK:
            Create a JSON object containing EXACTLY these 6 fields:

            1. "Consistency": (Aggregates the parts of individual statements that are consistent across multiple agent statements).
            2. "Conflict": (Identifies conflicting points between statements; empty if none).
            3. "Independence": (Extracts unique viewpoints of each agent not mentioned by others).
            4. "Integration": (Synthesizes all statements into a cohesive summary).
            5. "Tools_Usage": (Summarize specific tools/searches used in this round).
            6. "Long_Term_Experience": (Extract and summarize any prior experience/knowledge referenced from the database).

            Return ONLY valid JSON.
            """
        # if "Qwen" in self.text_model or "LLaVA" in self.text_model:
        content, prompt_tokens, completion_tokens = qwen_generate_content(
            [qwen_vl_chat_content(image_path=None, message=msg)], self.batch_manager)
        self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        self.token_stats["MDTeamGPT"]["prompt_tokens"] += prompt_tokens
        self.token_stats["MDTeamGPT"]["completion_tokens"] += completion_tokens
        content = content.strip()
        # print(f"\nlead_physician_synthesis_content:\n{content}")
        # else:
        #     prompt = ChatPromptTemplate.from_template(msg)
        #     chain = prompt | self.llm
        #     res = chain.invoke(prompt)
        #     # print(f"res_token_count:{res.usage_metadata}")
        #     self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        #     self.token_stats["MDTeamGPT"]["prompt_tokens"] += res.usage_metadata["input_tokens"]
        #     self.token_stats["MDTeamGPT"]["completion_tokens"] += res.usage_metadata["output_tokens"]
        #     content = res.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        # print(f"\nlead_physician_synthesis_cleaned_content:\n{content}")
        return content

    # 4. Safety Reviewer
    def safety_reviewer(self, current_bullet: str, round_num: int, max_rounds: int, question: str):
        if round_num == (max_rounds - 1):
            msg = f"""You are the Safety and Ethics Reviewer.
            Summarize the last round's synthesis.

            Faced question:
            {question}

            Current round's Context:
            {current_bullet}

            TASK:
            Based on the medical diagnosis of these experts and question, output the answer that you think is most likely to be correct.
            OUTPUT FORMAT (Strict):
            STATUS: [CONVERGED]
            FINAL_ANSWER: [Your final answer]
            """
        else:
            msg = f"""You are the Safety and Ethics Reviewer.
                Review the current round's synthesis.

                Faced question:
                {question}

                current round's Context:
                {current_bullet}

                TASK:
                1.Determine if the medical diagnosis has converged to a solid, safe conclusion without major conflicts.
                2.If the medical diagnosis has converged to a solid, safe conclusion without major conflicts, you should summarize their answers into a concise final answer and your output format should be like this:
                STATUS: [CONVERGED]
                FINAL_ANSWER: [The final answer]
                3.If the medical diagnosis has converged,  your output format should be like this:
                OUTPUT FORMAT (Strict):
                STATUS: [DIVERGED]
                FINAL_ANSWER: [Continuing discussion]
                """
        # if "Qwen" in self.text_model or "LLaVA" in self.text_model:
        # print(f"\nsafety_reviewer_msg:\n{msg}")
        content, prompt_tokens, completion_tokens = qwen_generate_content(
            [qwen_vl_chat_content(image_path=None, message=msg)], self.batch_manager)
        self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        self.token_stats["MDTeamGPT"]["prompt_tokens"] += prompt_tokens
        self.token_stats["MDTeamGPT"]["completion_tokens"] += completion_tokens
        out = content.strip()
        # print(f"\nsafety_reviewer_content:\n{out}")
        # else:
        #     prompt = ChatPromptTemplate.from_template(msg)
        #     chain = prompt | self.critic_llm
        #     res = chain.invoke(prompt)
        #     self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        #     self.token_stats["MDTeamGPT"]["prompt_tokens"] += res.usage_metadata["input_tokens"]
        #     self.token_stats["MDTeamGPT"]["completion_tokens"] += res.usage_metadata["output_tokens"]
        #     out=res.content
        return out

    # 5. CoT Reviewer
    def cot_reviewer(self, case_info, final_answer, ground_truth):
        # Only this agent sees the Ground Truth
        msg = f"""You are the 'Chain-of-Thought Reviewer'.

            CASE: {case_info[:500]}
            MODEL ANSWER: {final_answer}
            GROUND TRUTH: {ground_truth}

            TASK:
            Step 1: Determine correctness (letters match for Choice, semantic match for Open).

            Step 2: Generate specific fields based on correctness.

            IF CORRECT:
               - "is_correct": true
               - "summary_s4": A concise summary of the final reasoning (S4_final).

            IF INCORRECT:
               - "is_correct": false
               - "initial_hypothesis": What was the likely first thought?
               - "analysis_process": Step-by-step breakdown of the failure.
               - "final_conclusion": The wrong conclusion reached.
               - "error_reflection": Why it was wrong and how to avoid it.

            OUTPUT JSON ONLY.
            """
        # 你就是“思维过程审查员”。
        # 案例：{案例}
        # 标准答案：{答案}
        # 真实情况：{实际情况}
        # 任務：
        # 步驟
        # 1：判斷正確性（選項中的字母需匹配，開放式問題需語義相符）
        # 第2步：根据正确性生成特定字段。
        # 如果正确：
        # - "is_correct": true
        # - "summary_s4"： 最终推理的简洁总结（S4_final）
        # 如果错误：
        # - "is_correct": false
        # - "initial_hypothesis"： 最初可能的想法是什么？
        # - "analysis_process"： 失败的分步分析过程。
        # - "final_conclusion"： 所得出的错误结论。
        # - "error_reflection"： 为何会错误以及如何避免此类错误。
        # 只输出
        # JSON
        # 格式的数据。
        # if "Qwen" in self.text_model or "LLaVA" in self.text_model:
        try:
            content, prompt_tokens, completion_tokens = qwen_vl_generate_content(self.critic_model,
                                                                                 self.critic_processor, [
                                                                                     qwen_vl_chat_content(
                                                                                         image_path=None, message=msg)])
            self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
            self.token_stats["MDTeamGPT"]["prompt_tokens"] += prompt_tokens
            self.token_stats["MDTeamGPT"]["completion_tokens"] += completion_tokens
            content = content.strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            return json.loads(content)
        except:
            return {"is_correct": False, "analysis_text": "Parse Error"}
        # else:
        #     prompt = ChatPromptTemplate.from_template(msg)
        #     chain = prompt | self.critic_llm
        #     try:
        #         res = chain.invoke(prompt)
        #         self.token_stats["MDTeamGPT"]["num_llm_calls"] += 1
        #         self.token_stats["MDTeamGPT"]["prompt_tokens"] += res.usage_metadata["input_tokens"]
        #         self.token_stats["MDTeamGPT"]["completion_tokens"] += res.usage_metadata["output_tokens"]
        #         content = res.content.strip()
        #         if content.startswith("```json"): content = content[7:]
        #         if content.endswith("```"): content = content[:-3]
        #         return json.loads(content)
        #     except:
        #         return {"is_correct": False, "analysis_text": "Parse Error"}