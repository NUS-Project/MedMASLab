from typing import TypedDict, List, Annotated, Any
import operator,re
from langgraph.graph import StateGraph, END
from methods.MDTeamGPT.knowledge_base import kb_system


class MDTState(TypedDict):
    case_info: str
    image_base64: List[str]
    ground_truth: str

    selected_roles: List[str]
    triage_reason: str

    current_round: int
    max_rounds: int

    context_bullets: Annotated[List[str], operator.add]
    final_answer: str
    is_converged: bool

    kb_context_text: str
    kb_context_docs: Any
    current_num_agents: int
    # need_judge:bool


def create_workflow(agents_instance):
    def node_triage(state: MDTState):
        if hasattr(agents_instance, 'llm'):
            api_key = getattr(agents_instance.llm, 'api_key', None)
            base_url = getattr(agents_instance.llm, 'base_url', None)
        else:
            # Qwen 本地模型，无需 API Key
            api_key = None
            base_url = None
        kb_system.init_embeddings(
            # api_key=agents_instance.llm.openai_api_key,
            # base_url=agents_instance.llm.openai_api_base
            api_key=api_key,
            base_url=base_url
        )

        retrieval_result = kb_system.retrieve_context_details(state["case_info"])
        triage_result = agents_instance.primary_care_doctor(state["case_info"])
        state["current_num_agents"]=state["current_num_agents"]+1+len(triage_result["selected_roles"])
        # print(f"\nCurrent number of agents: {state['current_num_agents']}")
        return {
            "selected_roles": triage_result["selected_roles"],
            "triage_reason": triage_result["reasoning"],
            "current_round": 0,
            "kb_context_text": retrieval_result["text"],
            "kb_context_docs": retrieval_result["docs"],
            "context_bullets": [],
            "current_num_agents": state["current_num_agents"]
        }

    def node_consultation_and_synthesis(state: MDTState):
        roles = state["selected_roles"]
        rnd = state["current_round"]
        bullets = state["context_bullets"]

        #  Logic Check: Residual Context
        # 1. This is calculated BEFORE the agent loop.
        # 2. It only contains info from PREVIOUS rounds (bullets).
        # 3. Therefore, agents in this round CANNOT see each other's current output.
        residual_context = ""
        if rnd == 1:
            residual_context = f"PRIOR KNOWLEDGE FROM DB:\n{state['kb_context_text']}"
        else:
            recent_bullets = bullets[-2:]
            for i, b in enumerate(recent_bullets):
                bullet_rnd = rnd - len(recent_bullets) + i
                residual_context += f"--- Round {bullet_rnd} Summary ---\n{b}\n"

        dialogues = []
        conversation=''
        for role in roles:
            img = state["image_base64"] if rnd == 1 else None

            # Logic Check: Independence & Blindness
            # 1. 'residual_context' is static for all agents in this loop.
            # 2. 'ground_truth' is NOT passed to the agent.
            res = agents_instance.specialist_consult(
                role, state["case_info"], residual_context, img, rnd
            )
            dialogues.append(f"**{role}**: {res}")
            conversation+=f"{role}"

        # Lead Physician synthesizes the accumulated dialogues
        summary_json = agents_instance.lead_physician_synthesis(dialogues, rnd)
        state["current_num_agents"] = state["current_num_agents"] + 1
        # print(f"\nCurrent number of agents: {state['current_num_agents']}")

        return {
            "context_bullets": [summary_json],
            "current_round": rnd,
            "current_num_agents": state["current_num_agents"]
        }

    def node_safety_check(state: MDTState):
        # print(f"\ncontext_bullets:{state['context_bullets']}")
        last_bullet = state["context_bullets"][-1]
        rnd = state["current_round"]

        # Safety Reviewer checks convergence based on the summary 
        review = agents_instance.safety_reviewer(last_bullet, rnd,state["max_rounds"],state["case_info"])
        state["current_num_agents"] = state["current_num_agents"] + 1
        # print(f"\nCurrent number of agents: {state['current_num_agents']}")
        review_cleaned = re.sub(r'[^a-zA-Z]', '', review)
        is_converged = "STATUSCONVERGED" in review_cleaned.upper()
        final_ans = ""

        if "FINAL_ANSWER:" in review.upper():
        # 在原始 review 中找到位置（大小写不敏感）
            review_upper = review.upper()
            final_answer_pos = review_upper.find("FINAL_ANSWER:")
            
            # 从原始 review 中按照位置提取内容
            if final_answer_pos != -1:
                # FINAL_ANSWER: 长度为 13
                final_ans = review[final_answer_pos + 13:].strip()
            else:
                final_ans = review

        # if "FINAL_ANSWER:" in review.upper():
        #     parts = review.upper().split("FINAL_ANSWER:")
        #     final_ans = parts[1].strip() if len(parts) > 1 else review

        if rnd >= (state["max_rounds"]-1):
            is_converged = True
            if not final_ans:
                final_ans = "Max rounds reached. Proceeding with latest hypothesis."

        return {
            "is_converged": is_converged,
            "final_answer": final_ans,
            "current_round": rnd + 1,
            "current_num_agents": state["current_num_agents"]
        }

    def router(state: MDTState):
        if state["is_converged"]:
            return "end"
        return "continue"

    workflow = StateGraph(MDTState)

    workflow.add_node("triage", node_triage)
    workflow.add_node("consultation_layer", node_consultation_and_synthesis)
    workflow.add_node("safety_layer", node_safety_check)

    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "consultation_layer")
    workflow.add_edge("consultation_layer", "safety_layer")

    workflow.add_conditional_edges("safety_layer", router, {"continue": "consultation_layer", "end": END})

    return workflow.compile()