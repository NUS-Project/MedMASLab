from openai import OpenAI
import yaml,time
from methods.thread import qwen_generate_content
from methods.utils import get_apikey_and_baseurl_from_configs
from methods.utils import setup_model,qwen_vl_generate_content,qwen_vl_chat_content,is_options

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user",
                "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(
        question)
    return {"role": "user", "content": prefix_string}


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


def construct_assistant_message(content):
    # content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


class debate:
    def __init__(self, model_info, batch_manager):
        self.model_info = model_info
        self.batch_manager=batch_manager
        self.token_stats = {
                "debate_llm": {
                    "num_llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
            }

    def chat(self, answer_context, if_last=False):
        # try:
        if if_last:
            answer_context.append(
                {"role": "user", "content": f"Based on all the above information,{is_options()}"})
        completion,prompt_tokens,completion_tokens=qwen_generate_content(answer_context,self.batch_manager)
            
        
        self.token_stats["debate_llm"]["num_llm_calls"] += 1
        self.token_stats["debate_llm"]["prompt_tokens"] += prompt_tokens
        self.token_stats["debate_llm"]["completion_tokens"] += completion_tokens

        return completion

    def get_token_stats(self):
        return self.token_stats

def Debate_test(question, root_path, model_info,img_path,batch_manager):
    agents = 3
    rounds = 2
    final_decision = ''
    # agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]
    agent_contexts = [[qwen_vl_chat_content(img_path,question)] for agent in range(agents)]

    if_last = False
    debate_agent = debate(model_info, batch_manager)

    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):

            if round != 0:
                # print(f"\nagent_contexts:{agent_contexts}")
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
                # print(f"\nagent_contexts_other:{agent_contexts_other}")
                message = construct_message(agent_contexts_other, question, 2 * round - 1)
                agent_context.append(message)

            if round == (rounds - 1) and i == (agents - 1):
                # print("\nThis is the last process!\n")
                if_last = True
                completion = debate_agent.chat(agent_context, if_last)
                final_decision = completion
                # print(f"\nThe answer is {completion.choices[0].message.content}\n")
            else:
                completion = debate_agent.chat(agent_context, if_last)

            # completion = generate_answer(agent_context)

            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
            # print(f"\nRound:{round},Context:{completion.choices[0].message.content}")

    # response_dict[question] = (agent_contexts, answer)
    token_stats = debate_agent.get_token_stats()
    current_config = {"current_num_agents": agents, "round": rounds}

    return final_decision, token_stats,current_config

