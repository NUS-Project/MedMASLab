import os
import json
import random
from prettytable import PrettyTable 
from termcolor import cprint
from openai import OpenAI
from pptree import *
import transformers
import torch
from typing import Any, Dict, List
from pathlib import Path
from methods.utils import get_apikey_and_baseurl_from_configs,chat_content
import yaml
from methods.utils import setup_model,qwen_vl_chat_content,qwen_vl_generate_content,is_options
from qwen_vl_utils import process_vision_info
from methods.thread import qwen_generate_content
import re


def load_config(file_path: str) -> dict:
    """
    Load YAML configuration from a file.

    Args:
    file_path (str): Path to the YAML configuration file.

    Returns:
    dict: Dictionary of configuration parameters.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



def parse_recruitment_json(recruited_text: str,recruit_example:str) -> List[Dict[str, Any]]:
    start_index = recruited_text.find('{')
    end_index = recruited_text.rfind('}')
    # 切割出有效的 JSON 字符串
    recruited_text = recruited_text[start_index:end_index + 1]
    # data = json.loads(recruited_text)
    # default_recruited_text=""" """
    # for i in range(4):
    try:
        data = json.loads(recruited_text)
        # break
    except json.JSONDecodeError as e:
        # print(f"第{i+1}次解析失败内容:{recruited_text}")
        # 如果还没有用完，尝试修正
        # if i < 2:
        #     # 如果字符串以双"}"结尾，则将最后两个"}"替换为"}"
        #     if recruited_text.endswith('}}'):
        #         recruited_text = recruited_text[:-2] + '}'
        #     else:
        #         break
        # else:
        data = json.loads(recruit_example)
                # 如果循环结束还没有成功，抛出异常 recruit_example
                # raise ValueError(f"多次尝试后无法解析JSON：{recruited_text}")


    groups_out = []
    for _, g in data.items():  # 第1个组=Group1，第2个组=Group2...
        g_items = list(g.items())
        group_goal = str(g_items[0][1]).strip() if g_items else ""

        members: List[Dict[str, str]] = []
        for _, mv in g_items[1:]:  # Member1/2/3... 依次
            # 允许 mv 是 list[dict] 或 dict
            if isinstance(mv, list) and mv and isinstance(mv[0], dict):
                mv = mv[0]
            if not isinstance(mv, dict):
                continue

            mv_items = list(mv.items())
            role = str(mv_items[0][1]).strip() if len(mv_items) >= 1 else ""
            desc = str(mv_items[1][1]).strip() if len(mv_items) >= 2 else ""

            if role:
                members.append({"role": role, "expertise_description": desc})

        groups_out.append({"group_goal": group_goal, "members": members})

    return groups_out

class Agent:
    def __init__(self, instruction, role,  model_info='gemini-2.5-flash',root_path=None,batch_manager=None):
        self.instruction = instruction
        self.role = role
        self.root_path=root_path
        self.model_info = model_info
        self.batch_manager=batch_manager
       
        # self.img_path = img_path
        self.token_stats = {
            self.model_info: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }
        # if self.model_info == 'gemini-2.5-flash':
        #     api_key,base_url=get_apikey_and_baseurl_from_configs(self.root_path,self.model_info)
        #     self.client = OpenAI(api_key=api_key, base_url=base_url)
        #     self.messages = [
        #         {"role": "system", "content": instruction},
        #     ]
        # elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
        #     api_key,base_url=get_apikey_and_baseurl_from_configs(self.root_path,self.model_info)
        #     self.client =  OpenAI(api_key=api_key, base_url=base_url)
        #     self.messages = [
        #         {"role": "system", "content": instruction},
        #     ]
                    
        # elif self.model_info =='Llama-3.3-70B-Instruct':
        #     self.pipeline = setup_model(model_info,root_path)
        #     self.messages = [
        #         {"role": "system", "content": instruction},
        #     ]
        # elif "Qwen" in self.model_info or "LLaVA" in self.model_info:
            # self.model, self.processor = setup_model(model_info,root_path)
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text":instruction}]},
        ]

    def chat(self, message, img_path=None, chat_mode=True):
        # if self.model_info == 'gemini-2.5-flash':
        #     if img_path:
        #         self.messages.append(chat_content(img_path,message))
        #     else:
        #         self.messages.append({"role": "user", "content": message})
        #     response = self.client.chat.completions.create(
        #         model=self.model_info,
        #         messages=self.messages,
        #         stream=False
        #     )
        #     num_prompt_tokens, num_completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        #     self.token_stats[self.model_info]["num_llm_calls"] += 1
        #     self.token_stats[self.model_info]["prompt_tokens"] += num_prompt_tokens
        #     self.token_stats[self.model_info]["completion_tokens"] += num_completion_tokens
        #     self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        #     return response.choices[0].message.content

        # elif "Qwen" in self.model_info or "LLaVA" in self.model_info:

        self.messages.append(qwen_vl_chat_content(img_path,message))
        output,num_prompt_tokens,num_completion_tokens=qwen_generate_content(self.messages,self.batch_manager)
        self.token_stats[self.model_info]["num_llm_calls"] += 1
        self.token_stats[self.model_info]["prompt_tokens"] += num_prompt_tokens
        self.token_stats[self.model_info]["completion_tokens"] += num_completion_tokens
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text":output}]})
        return output
        
        # elif self.model_info=='Llama-3.3-70B-Instruct':
        #     self.messages.append({"role": "user", "content": message})
        #     response = self.pipeline(
        #         self.messages,
        #         max_new_tokens=2560
        #     )
        #     return response[0]["generated_text"][-1]["content"]

        # elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
        #     if img_path:
        #         self.messages.append(chat_content(img_path, message))
        #     else:
        #         self.messages.append({"role": "user", "content": message})

        #     response = self.client.chat.completions.create(
        #         model=self.model_info,
        #         messages=self.messages
        #     )
        #     self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        #     num_prompt_tokens, num_completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        #     self.token_stats[self.model_info]["num_llm_calls"] += 1
        #     self.token_stats[self.model_info]["prompt_tokens"] += num_prompt_tokens
        #     self.token_stats[self.model_info]["completion_tokens"] += num_completion_tokens
        #     return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        # if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
        #     if img_path:
        #         self.messages.append(chat_content(img_path, message))
        #     else:
        #         self.messages.append({"role": "user", "content": message})
        #     response = self.client.chat.completions.create(
        #         model=self.model_info,
        #         messages=self.messages,
        #         # temperature=0.5,
        #     )
        #     # print(f"response: {response}")

        #     num_prompt_tokens, num_completion_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        #     self.token_stats[self.model_info]["num_llm_calls"] += 1
        #     self.token_stats[self.model_info]["prompt_tokens"] += num_prompt_tokens
        #     self.token_stats[self.model_info]["completion_tokens"] += num_completion_tokens
        #     responses = response.choices[0].message.content
        #     return responses

        # elif "Qwen" in self.model_info or "LLaVA" in self.model_info:
        self.messages.append(qwen_vl_chat_content(img_path,message))
        output,num_prompt_tokens,num_completion_tokens=qwen_generate_content(self.messages,self.batch_manager)
        self.token_stats[self.model_info]["num_llm_calls"] += 1
        self.token_stats[self.model_info]["prompt_tokens"] += num_prompt_tokens
        self.token_stats[self.model_info]["completion_tokens"] += num_completion_tokens
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text":output}]})
        return output
        
        # else:
        #     self.model_info = 'Llama-3.3-70B-Instruct'
        #     self.messages.append({"role": "user", "content": message})
        #     response = self.pipeline(
        #         self.messages,
        #         max_new_tokens=10240
        #     )
        #     return response[0]["generated_text"][-1]["content"]

    def get_token_stats(self):
        return self.token_stats

class Group:
    def __init__(self, goal, members, question,model_info,root_path,batch_manager):
        self.goal = goal
        self.root_path = root_path
        self.members = []
        self.batch_manager=batch_manager
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model_info,root_path=root_path,batch_manager=self.batch_manager)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.model_info =model_info
        self.group_token_stats = {
            self.model_info: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }
        # self.examplers = examplers

    def interact(self, comm_type,  img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which names {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt,img_path)
            except:
                delivery = assist_members[0].chat(delivery_prompt,img_path)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where names {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])
            investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt,img_path)
            num_llm_calls = 0
            Prompt_Tokens = 0
            completion_tokens = 0
            lead_member_stats =lead_member.get_token_stats()
            for model_name, stats in lead_member_stats.items():
                num_llm_calls = stats['num_llm_calls']
                Prompt_Tokens = stats['prompt_tokens']
                completion_tokens = stats['completion_tokens']

            for idx, agent in enumerate(assist_members):
                idx_stats = agent.get_token_stats()
                for model_name, stats in idx_stats.items():
                    num_llm_calls = stats['num_llm_calls'] + num_llm_calls
                    Prompt_Tokens = stats['prompt_tokens'] + Prompt_Tokens
                    completion_tokens = stats['completion_tokens'] + completion_tokens

            self.group_token_stats = {
                self.model_info: {"num_llm_calls": num_llm_calls, "prompt_tokens": Prompt_Tokens, "completion_tokens": completion_tokens}
            }

            return response

        elif comm_type == 'external':
            return

    def get_group_token_stats(self):
        return self.group_token_stats

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents


def MDAgents_test(question,root_path,model_info,img_path,batch_manager):
    config_path = str(Path(root_path) / 'methods' / 'MDAgents' / 'configs' / 'config_main.yaml')
    config = load_config(config_path)
    difficulty = config.get('difficulty', 'adaptive')
    num_teams = config.get('num_teams', 3)  # Default to 3 if not specified
    num_agents = config.get('num_agents', 3)  # Default to 3 if not specified
    intermediate_num_agents= config.get('intermediate_num_agents', 5)
    current_num_agents =0
    round =1
    difficulty, token_stats_deter = determine_difficulty(question, difficulty,model_info,root_path,img_path,batch_manager)
    num_llm_calls=0
    Prompt_Tokens=0
    completion_tokens=0
    if token_stats_deter is not None:
        current_num_agents +=1
        for model_name, stats in token_stats_deter.items():
            num_llm_calls = stats['num_llm_calls']
            Prompt_Tokens = stats['prompt_tokens']
            completion_tokens = stats['completion_tokens']

    prompt_file = str(Path(root_path) / 'methods' / 'MDAgents' / 'Recruit_prompt.txt')
    print(f"\ndifficulty: {difficulty}")

    if difficulty == 'basic':
        final_decision, token_stats = process_basic_query(question,  model_info,root_path,img_path,batch_manager)
        current_num_agents +=1
    elif difficulty == 'intermediate':
        num_rounds=1
        final_decision, token_stats, round = process_intermediate_query(question,  model_info,intermediate_num_agents,root_path,img_path,num_rounds,batch_manager)
        current_num_agents = current_num_agents+intermediate_num_agents+2
    else:
        final_decision, token_stats = process_advanced_query(question,prompt_file,model_info,num_teams,num_agents,root_path,img_path,batch_manager)
        current_num_agents = current_num_agents+num_teams*num_agents+2

    for model_name, stats in token_stats.items():
        num_llm_calls = stats['num_llm_calls'] + num_llm_calls
        Prompt_Tokens = stats['prompt_tokens'] + Prompt_Tokens
        completion_tokens = stats['completion_tokens'] + completion_tokens
    token_stats = {
        model_info: {"num_llm_calls": num_llm_calls, "prompt_tokens": Prompt_Tokens,
                     "completion_tokens": completion_tokens}
    }
    current_config = {"current_num_agents": current_num_agents, "round": round}
    print(f"\ncurrent_num_agents: {current_num_agents} \n round: {round}")
    return final_decision, token_stats,current_config


def determine_difficulty(question, difficulty,model_info,root_path,img_path,batch_manager):
    if difficulty != 'adaptive':
        return difficulty,None
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model_info,root_path=root_path,batch_manager=batch_manager)
    response = medical_agent.chat(difficulty_prompt,img_path=img_path)
    token_stats = medical_agent.get_token_stats()

    if response is None:
        return 'intermediate', token_stats

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic', token_stats
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate', token_stats
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced', token_stats
    else:
        return 'intermediate', token_stats

def process_basic_query(question, model_info,root_path,img_path,batch_manager):
    # if need_judge:
    msg='You are a helpful assistant that answers questions about medical knowledge.If you come across a question with no options, then simply answer the question directly.'
    # else:
    #     msg="""You are a helpful assistant that answers questions about medical knowledge.If you come across a multiple-choice question with options,then simply answer the letter corresponding to your answer choiceProvide only the letter corresponding to your answer choice (A/B/C/D/E/F),No other Comments! 
    #     Your answer should follow this format strictly: \nAnswer: <your answer>.\nFor Example:Answer: A or Answer: B or Answer: C or Answer: D or Answer: E or Answer: F.""" 
        
    single_agent = Agent(instruction=msg, role='medical expert',  model_info=model_info,root_path=root_path,batch_manager=batch_manager)
    final_decision= single_agent.temp_responses(f'''{msg}\nThe following is a question about medical knowledge.\n\n**Question:** {question}''',img_path=img_path)
    token_stats = single_agent.get_token_stats()
    return final_decision,token_stats


def parse_agents_from_recruitment(recruited, num_agents):
    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]
    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)
    print(f"\nagents_data:{agents_data}")
    if len(agents_data) > num_agents:
        agents_data = [agent for agent in agents_data if re.search(r'\d', str(agent))]
      
    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
        description = agent[0].split('-')[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
    
    return agents_data,  agent_list,agent_emoji, hierarchy_agents


def process_intermediate_query(question, model_info,intermediate_num_agents,root_path,img_path,num_rounds,batch_manager):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_info,root_path=root_path,batch_manager=batch_manager)
    # tmp_agent.chat(recruit_prompt)
    num_agents = intermediate_num_agents # You can adjust this number as needed
    recruited = tmp_agent.chat(f"{recruit_prompt} Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.",img_path=img_path)
    num_llm_calls = 0
    Prompt_Tokens = 0
    completion_tokens = 0
    token_stats_tmp_agent = tmp_agent.get_token_stats()
    for model_name, stats in token_stats_tmp_agent.items():
        num_llm_calls = stats['num_llm_calls']+num_llm_calls
        Prompt_Tokens = stats['prompt_tokens']+Prompt_Tokens
        completion_tokens = stats['completion_tokens']+completion_tokens
    print(f"\nrecruited:{recruited}")
    #######################################
    try:
        agents_data,  agent_list,agent_emoji, hierarchy_agents=parse_agents_from_recruitment(recruited,num_agents)
    except:
        default_recruited="1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent"
        agents_data,  agent_list,agent_emoji, hierarchy_agents=parse_agents_from_recruitment(default_recruited,num_agents)
#######################################
    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model_info,root_path=root_path,batch_manager=batch_manager)
        
        # _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    # num_rounds = num_rounds
    num_turns = num_agents
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    # initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(
            f'''Given the medical query below, please indicate the answer you believe is correct and provide your reasoning.\nQuestion: {question}Your response should follow this format:\n1. Your answer.\n2. Your reasoning.''',
            img_path=img_path)
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    # agent_rs = Agent(instruction=f"You are a medical assistant who are good at summarizing conversation records from various domain experts.There are {num_agents} medical experts discussing the issue here.You need to summarize the latest conversation. Before doing this, indicate which expert is speaking to which expert.Remember, do not change the original meaning of the conversation. You can only summarize the latest conversation.", role="medical assistant", model_info=model_info,root_path=root_path)
    # 你是一名医疗助理，擅长根据来自不同领域的多位专家的观点进行总结和综合分析。
    # agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
    assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[1].items())
    init_comment=assessment
    interaction_content=""
    new_assessment=""
    current_round=0
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        current_round=n
        print(f"num_turns:{num_turns}")
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")
            if turn_num > 0 or n >1:
                # interaction_content=agent_rs.chat(f"{interaction_content}\nThis is the latest conversation. Please summarize it.")
                print(f"\ninteraction_content:\n{interaction_content}\n")
                new_assessment = interaction_content
                interaction_content=""
            num_yes = 0
            if turn_num == 0 and n>1:
                new_assessment=f"Question: {question}"+new_assessment
            for idx, v in enumerate(medical_agents):
                if n==1 and turn_num==0:
                    message = f"{assessment}\nGiven the opinions from other medical experts in the team, please indicate whether you want to talk to any expert (yes/no)"
                else:
                    message = f"{new_assessment}\nGiven the opinions from other medical experts in the team, please indicate whether you want to talk to any expert (yes/no)"
                   
                # print(f"message:\n{message}")
                participate = v.chat(message,img_path=img_path)
                # 鉴于您团队中其他医学专家的意见，请告知您是否希望与任何专家进行交流（是 / 否）
                print(f"\nparticipate: {participate}")
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with expert 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    # 输入您想与哪位专家交谈的号码：
                    # 专家1：骨科医生 - 专注于肌肉骨骼疾病的外科治疗。
                    # 专家2：医学伦理学家 - 专注于医疗实践中的伦理问题，包括患者披露和知情同意。
                    # 专家3：风险管理专家 - 在医疗环境中致力于降低风险，并提供法律和合规意见。
                    # 专家4：患者倡导者 - 代表患者的利益和权利，确保医疗的透明度和沟通。
                    # 专家5：医疗法律专家 - 专注于影响医疗实践的法律和法规，包括医疗事故和知情同意
                    # 例如，如果您想与专家1交谈，请只返回1。如果您想与多个专家交谈，请返回1,2，并且不要返回理由。
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]
                    # print(f"chosen_experts: {chosen_experts}")

                    for ce in chosen_experts:
                        specific_question = v.chat(f"First, you should show your identity, and leave your question,opinion and reason to an expert you chose (medical expert {ce}. {medical_agents[ce-1].role}).")
                        # 请先展示您的专业知识，然后再将您的意见提交给您选定的专家。
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                        # print(f"specific_question: {specific_question}")
                        # print(f"interaction_log: {interaction_log}")
                        interaction_content=interaction_content+f"This is the content of what {v.role} wants to discuss with {medical_agents[ce-1].role}.\n The content is as follows: \n {specific_question}\n"
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response
            # print(response)
        
        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer
    if final_answer is None:
        final_answer=init_comment
    print(f"\ntmp_final_answer: {final_answer}")
    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)
    print("\n[DEBUG] Medical Agents Stats:")
    for idx, agent in enumerate(medical_agents):
        idx_stats = agent.get_token_stats()
        for model_name, stats in idx_stats.items():
            num_llm_calls = stats['num_llm_calls'] + num_llm_calls
            Prompt_Tokens = stats['prompt_tokens'] + Prompt_Tokens
            completion_tokens = stats['completion_tokens'] + completion_tokens

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])

    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model_info,root_path=root_path,batch_manager=batch_manager)
    # moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')

    # print(f"\nfinal_answer：{final_answer}")

    # _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
    # if need_judge:
    msg=f"Question:{question}\nEach agent's final answer{final_answer}\nGiven each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote."
    # else:
    #     # msg=f'You are a helpful assistant that answers questions about medical knowledge.If you come across a multiple-choice question with options,then simply answer the letter corresponding to your answer choice.{is_options(need_judge)}' 
    #     msg=f"Question:{question}\nEach agent's final answer{final_answer}\nGiven each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote.Remember:your final answer by taking majority vote should {is_options(need_judge)}"
    _decision = moderator.temp_responses(
        msg,
        img_path=img_path)

    # 在给出每个代理的最终答案后，请审查每个代理的意见，并通过多数票得出对该问题的最终答案。您的答案应采用以下格式：
    # 答案： C）第2个咽弓
    # {最终答案}
    # 问题： {问题}
    #Provide only the letter corresponding to your answer choice (A/B/C/D/E/F).
    token_stats_moderator = moderator.get_token_stats()
    for model_name, stats in token_stats_moderator.items():
        num_llm_calls = stats['num_llm_calls']+num_llm_calls
        Prompt_Tokens = stats['prompt_tokens']+Prompt_Tokens
        completion_tokens = stats['completion_tokens']+completion_tokens

    print("\U0001F468\u200D\u2696\uFE0F moderator's final decision (by majority vote):", _decision)
    print()
    token_stats = {
        model_info: {"num_llm_calls":num_llm_calls, "prompt_tokens": Prompt_Tokens, "completion_tokens": completion_tokens}
    }

    return _decision,token_stats,current_round

def process_advanced_query(question,  prompt_file,model_info,num_teams,num_agents,root_path,img_path,batch_manager):
    print("[STEP 1] Recruitment")
    group_instances = []

    # recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""
    #您是一位经验丰富的医学专家。面对如此复杂的医疗问题，您需要组建多学科团队（MDT），并让团队成员共同给出准确且可靠的解答。
    prompts_file = load_prompts_from_file(prompt_file)
    prompts_recruit = prompts_file["MEDICAL_ASSISTANT"]
    print("MEDICAL_ASSISTANT",prompts_recruit)
    print("MEDICAL_ASSISTANT",prompts_recruit)
    recruit_example = prompts_file["MEDICAL_RECRUIT"]
    tmp_agent = Agent(instruction=prompts_recruit, role='system', model_info=model_info,root_path=root_path,batch_manager=batch_manager)
    num_llm_calls=0
    Prompt_Tokens=0
    completion_tokens=0
    # recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.")
    recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question, please return your recruitment plan to better make an accurate answer.\n\nFor example:{recruit_example} When you return your answer, please strictly refer to the above format.",img_path=img_path)
    token_stats_tmp_agent= tmp_agent.get_token_stats()
    for model_name, stats in token_stats_tmp_agent.items():
        num_llm_calls = stats['num_llm_calls'] + num_llm_calls
        Prompt_Tokens = stats['prompt_tokens'] + Prompt_Tokens
        completion_tokens = stats['completion_tokens'] + completion_tokens
    #问题：{问题}\n\n您应当组建 {团队数量} 个具有不同专业特长或不同目的的医疗多学科团队（MDT），并且每个 MDT 都应配备 {医生数量} 名临床医生。考虑到医疗问题以及所提供的选项，请返回您的招募计划，以便更准确地给出答案。
    # \n\n例如，以下可以是一个示例答案：
    # \n小组 1 - 初步评估团队（IAT）
        # \n成员 1：耳鼻喉科医生（耳鼻喉外科医生）（负责人） - 专长于耳、鼻和喉手术，包括甲状腺切除术。由于其在手术干预和处理任何手术并发症（如神经损伤）方面起着关键作用，该成员担任组长。
        # \n成员 2：普通外科医生 - 提供额外的外科专业知识，并在甲状腺手术并发症的总体管理中提供支持。
        # \n成员 3：麻醉师 - 专注于围手术期护理、疼痛管理以及评估任何因麻醉可能影响声音和气道功能的并发症。
    # \n\n小组 2 - 诊断证据团队（DET）
        # \n成员 1：内分泌科医生（负责人） - 负责格雷夫斯病的长期管理，包括激素治疗和术后任何相关并发症的监测。
        # \“成员 2：言语语言病理学家——专长于嗓音和吞咽障碍的治疗，为神经受损后的患者提供康复服务，以改善其言语和嗓音质量。
        # \n成员 3：神经科医生——评估并提供有关神经损伤及潜在恢复策略的建议，为患者的治疗提供神经学方面的专业知识。
    # \n\n小组 3 - 患者病史团队（PHT）
        # \n成员1：精神科医生或心理学家（负责人）——处理慢性疾病及其治疗所带来的心理影响，包括与嗓音变化、自尊心以及应对策略相关的问题。
        # \n成员 2：物理治疗师——提供锻炼和策略，以维持身体健康，并通过整体健康状况间接支持嗓音功能的恢复。
        # \n成员 3：职业治疗师——帮助患者适应嗓音的变化，特别是如果他们的职业高度依赖于声音交流，帮助他们找到维持职业角色的策略。
    # 第 4 组 - 最终评审与决策团队（FRDT）
        # 成员 1：各专业领域的高级顾问（负责人） - 提供整体的专业知识和决策指导
        # 成员 2：临床决策专家 - 协调来自不同团队的各种建议，并制定综合治疗方案
        # 成员 3：高级诊断支持人员 - 利用先进的诊断工具和技术来确认神经损伤的确切程度和原因，从而辅助做出最终决策
        
    # 以上只是一个示例，因此您应自行组建独特的多学科团队（MDT），但您的招聘计划中应包含初始评估团队（IAT）和最终评审与决策团队（FRDT）。在提交答案时，请严格遵循上述格式。
    print("\n[DEBUG] ===== recruiter raw output =====")
    print(recruited)
    print("[DEBUG] ===== end recruiter raw output =====\n")
    # NEW: parse recruiter JSON
    try:
        groups_parsed = parse_recruitment_json(recruited,recruit_example)
    except json.JSONDecodeError as e:
        raise ValueError(f"Recruiter did not return valid JSON: {e}\nRaw:\n{recruited}")

    print(f"[DEBUG] parsed {len(groups_parsed)} groups from JSON")
    for g in groups_parsed:
        group_instances.append(Group(g["group_goal"], g["members"], question,model_info=model_info,root_path=root_path,batch_manager=batch_manager))
        
    # for i, g in enumerate(groups_parsed):
    #     print(f"Group {i+1} - {g['group_goal']}")
    #     for j, m in enumerate(g["members"]):
    #         print(f" Member {j+1} ({m['role']}): {m['expertise_description']}")

        # IMPORTANT: Group expects members as list of {role, expertise_description}
        # group_instances.append(Group(g["group_goal"], g["members"], question))
    # groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    # group_strings = ["Group " + group for group in groups]
     
    # print(f"[DEBUG] split into {len(group_strings)} group blocks")
    # print(f"group_strings_context: {group_strings}")
    # for i1, gs in enumerate(group_strings):
    #     print(f"\n[DEBUG] ----- Group block #{i1} -----")
    #     print(gs)
    #     print(f"[DEBUG] ----- end block #{i1} -----\n")
    #     res_gs = parse_group_info(gs)
    #     print("[DEBUG] parsed group_goal:", repr(res_gs["group_goal"]))
    #     print("[DEBUG] parsed members:", res_gs["members"])
    #     print(f"Group {i1+1} - {res_gs['group_goal']}")
    #     for i2, member in enumerate(res_gs['members']):
    #         print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
    #     print()

    #     group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
    #     group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    # for group_instance in group_instances:
    #     if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
    #         init_assessment = group_instance.interact(comm_type='internal')
    #         initial_assessments.append([group_instance.goal, init_assessment])
    if group_instances:  # 只取第一个
        init_assessment = group_instances[0].interact(comm_type='internal',img_path=img_path)
        initial_assessments.append([group_instances[0].goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    if len(group_instances) > 2:
        for group_instance in group_instances[1:-1]:
            # if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal',img_path=img_path)
            assessments.append([group_instance.goal, assessment])
    # for group_instance in group_instances:
    #     if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
    #         assessment = group_instance.interact(comm_type='internal')
    #         assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    # for group_instance in group_instances:
    #     if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
    #         decision = group_instance.interact(comm_type='internal')
    #         final_decisions.append([group_instance.goal, decision])
    if len(group_instances) > 1:
            group_instance = group_instances[-1]
            decision = group_instance.interact(comm_type='internal',img_path=img_path)
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # for model_name, stats in enumerate(group_instances):
    for idx, agent in enumerate(group_instances):
        idx_stats = agent.get_group_token_stats()
        for model_name, stats in idx_stats.items():
            num_llm_calls = stats['num_llm_calls'] + num_llm_calls
            Prompt_Tokens = stats['prompt_tokens'] + Prompt_Tokens
            completion_tokens = stats['completion_tokens'] + completion_tokens


    # STEP 3. Final Decision
    # if need_judge:
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query.If you come across a multiple-choice question with options,then simply answer the letter corresponding to your answer choice.If you come across a question with no options, then simply answer the question directly."""
    # else:
    #     decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query.{is_options(need_judge)}"""
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model_info,root_path=root_path,batch_manager=batch_manager)
    # tmp_agent.chat(decision_prompt)
    # if need_judge:
    msgs=f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}"""
    # else:
    #     msgs=f"""Investigation:\n{initial_assessment_report}\nRemember:{is_options(need_judge)}\n\nQuestion: {question}"""

    final_decision = tmp_agent.temp_responses(msgs, img_path=img_path)
    token_stats_tmp = tmp_agent.get_token_stats()
    for model_name, stats in token_stats_tmp.items():
        num_llm_calls = stats['num_llm_calls'] + num_llm_calls
        Prompt_Tokens = stats['prompt_tokens'] + Prompt_Tokens
        completion_tokens = stats['completion_tokens'] + completion_tokens

    token_stats = {
        model_info: {"num_llm_calls": num_llm_calls, "prompt_tokens": Prompt_Tokens,
                     "completion_tokens": completion_tokens}
    }

    return final_decision,token_stats


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load multiple prompts from a file.

    Args:
    file_path (str): Path to the file containing prompts.

    Returns:
    Dict[str, str]: A dictionary of prompt names and their content.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts = {}
    current_prompt = None
    current_content = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_prompt:
                    prompts[current_prompt] = "\n".join(current_content).strip()
                current_prompt = line[1:-1]
                current_content = []
            elif line:
                current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts
