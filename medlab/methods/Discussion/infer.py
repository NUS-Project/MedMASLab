from __future__ import annotations
from pathlib import Path
import io
import json
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from .multi_agent.discussion import LLM_Debate
from methods.utils import qwen_vl_chat_content


# def _ensure_openai_env(root_path: str, model_name: str) -> None:
#     if not os.environ.get("OPENAI_API_KEY"):
#         api_key, base_url = get_apikey_and_baseurl_from_configs(root_path, model_name)
#         os.environ["OPENAI_API_KEY"] = api_key
#         if base_url and not os.environ.get("OPENAI_BASE_URL"):
#             os.environ["OPENAI_BASE_URL"] = base_url
#     else:
#         if not os.environ.get("OPENAI_BASE_URL"):
#             _, base_url = get_apikey_and_baseurl_from_configs(root_path, model_name)
#             if base_url:
#                 os.environ["OPENAI_BASE_URL"] = base_url


def _load_agents_config(config_path: Path, model_name: str) -> list[dict[str, Any]]:
    agents_config = json.loads(config_path.read_text(encoding="utf-8"))
    for cfg in agents_config:
        if cfg.get("type") == "openai":
            cfg["model_name"] = model_name
    return agents_config


# def _extract_choice_letter(text: str) -> str:
#     # Try to extract a multiple-choice letter if present.
#     m = re.search(r"\b([A-E])\b", text.strip(), flags=re.IGNORECASE)
#     if m:
#         return m.group(1).upper()
#     return text.strip()


def discussion_infer(
        question,
        root_path,
        model_info,
        img_path=None,
        batch_manager=None
):
    """Thin adapter for upstream Discussion (multi-agent discussion).

    Note: upstream is designed for creativity benchmarks; here we preserve its
    discussion workflow and return the final-round response from one agent.
    """
    # base_dir = Path(__file__).resolve().parent
    multi_agent_dir = Path(root_path) / "methods" / "Discussion" / "multi_agent"
    config_path = multi_agent_dir / "config_role.json"
    agents_config = _load_agents_config(config_path, model_info)

    # Lazy import to keep `python -m main -h` safe if optional deps are absent.

    # Mirror upstream defaults
    rounds = 5
    prompt = 1

    # We avoid writing a full dataset file and instead run a single-question loop
    # that mirrors upstream LLM_Discussion_Scientific.run().
    # print("到达1")
    runner = LLM_Debate(
        agents_config=agents_config,
        dataset_file="",
        rounds=rounds,
        task="Scientific",
        prompt=prompt,
        root_path=root_path,
        batch_manager=batch_manager
    )
    # print("到达2")
    # stdout_buf = io.StringIO()
    # stderr_buf = io.StringIO()
    # with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
    chat_history = {agent.agent_name: [] for agent in runner.agents}
    # print(f"\nchat_history:{chat_history}")
    initial_prompt = (
            "Initiate a discussion with others to collectively complete the following task: "
            + question
            + runner.discussion_prompt
    )

    most_recent_responses: Dict[str, list[dict[str, Any]]] = {}
    last_round_responses: Dict[str, str] = {}
    round = 0
    idx = 0
    for round_idx in range(runner.rounds):
        # print("到达3")
        round += 1
        # print("here!")
        is_last_round = round_idx == runner.rounds - 1
        is_first_round = round_idx == 0
        round_responses: Dict[str, list[dict[str, Any]]] = {agent.agent_name: [] for agent in runner.agents}
        # idx=0
        for agent in runner.agents:
            # idx+=1
            if getattr(agent, "agent_role", "None") != "None":
                agent_role_prompt = (
                    f"You are a {agent.agent_role} whose specialty is {agent.agent_speciality}. "
                    f"{agent.agent_role_prompt} Remember to claim your role in the beginning of each conversation. "
                )
            else:
                agent_role_prompt = ""

            if is_first_round:
                formatted_initial_prompt = qwen_vl_chat_content(img_path, agent_role_prompt + initial_prompt)
                # formatted_initial_prompt = agent.construct_user_message(agent_role_prompt + initial_prompt)
                chat_history[agent.agent_name].append(formatted_initial_prompt)
                response = agent.generate_answer(chat_history[agent.agent_name])
            else:
                combined_prompt = runner.construct_response(
                    question,
                    most_recent_responses,
                    agent,
                    is_last_round,
                )
                formatted_combined_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                chat_history[agent.agent_name].append(formatted_combined_prompt)
                if is_last_round:
                    # print(f"\nchat_history[agent.agent_name]:{chat_history[agent.agent_name]}")
                    response = agent.generate_answer(chat_history[agent.agent_name], is_last_round=is_last_round)
                    # print(f"\nlast_round_response:{response}")
                    # break
                else:
                    response = agent.generate_answer(chat_history[agent.agent_name])

            if is_last_round:
                last_round_responses[agent.agent_name] = str(response)

            formatted_response = agent.construct_assistant_message(response)
            chat_history[agent.agent_name].append(formatted_response)
            round_responses[agent.agent_name].append(formatted_response)

        most_recent_responses = round_responses

    # Choose one agent's final-round response (first agent, deterministic).
    chosen_agent_name = runner.agents[0].agent_name if runner.agents else ""
    print(f"\n runner.agents[0].agent_name:{runner.agents[0].agent_name}")
    final_decision = last_round_responses.get(chosen_agent_name, "")
    # final_decision = _extract_choice_letter(final_text) if final_text else ""

    # Aggregate token usage from OpenAI agents (patched in agents.py).
    num_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    # current_num_agents=0
    for agent in runner.agents:
        # current_num_agents+=1
        # print(f"\ncurrent_num_agents:{len(runner.agents)}")
        num_calls += agent.num_llm_calls
        prompt_tokens += agent.prompt_tokens
        completion_tokens += agent.completion_tokens

    token_stats = {
        model_info: {
            "num_llm_calls": num_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    }
    current_config = {"current_num_agents": 4, "round": round}

    return final_decision, token_stats, current_config
