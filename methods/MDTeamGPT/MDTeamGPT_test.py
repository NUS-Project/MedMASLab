import base64
import json
from pathlib import Path
from methods.MDTeamGPT.agents import MDTAgents
from methods.MDTeamGPT.workflow import create_workflow
from methods.MDTeamGPT.knowledge_base import kb_system
import yaml
from methods.utils import get_apikey_and_baseurl_from_configs, encode_image


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


class TerminalUIHandler:
    def __init__(self):
        self.current_role = None
        self.full_text = ""

    def on_token(self, role, token):
        self.full_text += f"{role} is speaking... "+token
        print(self.full_text)  # Print token in real-time

    def finish_turn(self):
        if self.full_text:
            print(f"\n{self.full_text}")

    def on_tool_output(self, role, query, result):
        print(f"[Tool - {role}] Query: {query}, Result: {result[20:]}")


def MDTeamGPT_test(question, image_path, root_path,batch_manager=None):
    # Load the configuration
    config_path = str(Path(root_path) / 'methods' / 'MDTeamGPT' / 'configs' / 'config_main.yaml')
    config = load_config(config_path)
    # vl_model = config.get('vl_model', 'gpt-4o-mini')
    # text_model = config.get('text_model', 'gpt-4o-mini')
    enable_tools = config.get('enable_tools', True)
    # if vl_model in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini'] and text_model in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
    #     api_key_vl, base_url_vl = get_apikey_and_baseurl_from_configs(root_path, vl_model)
    #     api_key_text, base_url_text = get_apikey_and_baseurl_from_configs(root_path, text_model)
    #     agents = MDTAgents(api_key_vl, base_url_vl, api_key_text, base_url_text, text_model, vl_model,
    #                    enable_tools)
    # elif ("Qwen" in vl_model and "Qwen" in text_model) or ("LLaVA" in vl_model and "LLaVA" in text_model):
    agents = MDTAgents(enable_tools=enable_tools,root_path=root_path,batch_manager=batch_manager)
    
    app = create_workflow(agents)
    # Prepare initial state
    state = {
        "case_info": question,
        "image_base64": image_path,
        "ground_truth": "",
        "selected_roles": [],
        "triage_reason": "",
        "current_round": 1,
        "max_rounds": 6,
        "context_bullets": [],
        "final_answer": "",
        "is_converged": False,
        "kb_context_text": "",
        "kb_context_docs": [],
        "current_num_agents":0,
       
    }

    # Set callbacks for output handling
    ui = TerminalUIHandler()
    agents.set_stream_callback(ui.on_token)
    agents.set_tool_callback(ui.on_tool_output)
    for event in app.stream(state):
        # Handle events similarly as before
        if "triage" in event:
            data = event["triage"]
            # print(
            #     f"Triage Complete: {data['triage_reason']}, Selected Specialists: {', '.join(data['selected_roles'])}")

        if "consultation_layer" in event:
            data = event["consultation_layer"]
            current_round = data["current_round"]
            # print(f"Round {current_round} Consultation in progress...")

        if "safety_layer" in event:
            data = event["safety_layer"]
            if data["is_converged"]:
                final_answer = data["final_answer"]
                # print(f"Final Medical Conclusion: {final_answer}")
                token_stats = agents.get_token_stats()
                current_round=data["current_round"]
                # print(f"\n Round {current_round}")
                current_num_agents=data["current_num_agents"]
                # print(f"\nfinal Number of agents: {current_num_agents}")
                current_config = {"current_num_agents": current_num_agents, "round": current_round}
                return final_answer, token_stats,current_config
            else:
                print("⚠️ Divergence detected. Continuing...")


