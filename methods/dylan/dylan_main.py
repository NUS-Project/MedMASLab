from __future__ import annotations

import math
import random
import re

from methods.mas_base import MAS


class DyLAN_Main(MAS):
    """Dynamic LLM Agent Network (DyLAN) implementation."""

    def __init__(self,model_name,batch_manager):
        # method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(model_name, batch_manager)

        self.seed = self.method_config.get("random_seed", 0)
        self.num_agents = self.method_config.get("num_agents", 4)
        self.num_rounds = self.method_config.get("num_rounds", 3)
        self.activation = self._get_activation_map()[self.method_config.get("activation", "listwise")]
        self.qtype = "open-ended"
        base_roles = self.method_config.get("roles", ["Assistant"] * self.num_agents)
        self.roles = [base_roles[i % len(base_roles)] for i in range(self.num_agents)]

        self.cmp_res = lambda x, y: self._sentence_bleu(x, [y], lowercase=True) >= 0.9 * 100
        self.ans_parser = lambda x: x

        self.nodes = []
        self.edges = []
        self._init_network()

    def inference(self, sample):
        query = sample["query"]
        self._img_paths = sample.get("img_paths", None)
        random.seed(self.seed)

        self._zero_grad()
        self._set_allnodes_deactivated()
        assert self.num_rounds > 2

        loop_indices = list(range(self.num_agents))
        random.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self._activate_node(node_idx, query)
            activated_indices.append(node_idx)
            if idx >= math.floor(2 / 3 * self.num_agents):
                reached, reply = self._check_consensus(activated_indices, list(range(self.num_agents)))
                if reached:
                    # print("\n here! a")
                    current_config = {"current_num_agents": self.num_agents, "round": 0}
                    return {"response": reply}, current_config

        loop_indices = list(range(self.num_agents, self.num_agents * 2))
        random.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self._activate_node(node_idx, query)
            activated_indices.append(node_idx)
            if idx >= math.floor(2 / 3 * self.num_agents):
                reached, reply = self._check_consensus(activated_indices, list(range(self.num_agents)))
                if reached:
                    # print("\n here! b")
                    current_config = {"current_num_agents": self.num_agents, "round": 1}
                    return {"response": reply}, current_config

        idx_mask = list(range(self.num_agents))
        idxs = list(range(self.num_agents, self.num_agents * 2))
        round =0
        for rid in range(2, self.num_rounds):
            round=rid
            print("\nround:", round)
            if self.num_agents > 3:
                replies = [self.nodes[idx]["reply"] or "" for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]

                if self.activation == 0:
                    tops = self._listwise_ranker(shuffled_replies, query)
                    idx_mask = list(map(lambda x: idxs[indices[x]] % self.num_agents, tops))

            loop_indices = list(range(self.num_agents * rid, self.num_agents * (rid + 1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    self._activate_node(node_idx, query)
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2 / 3 * len(idx_mask)):
                        reached, reply = self._check_consensus(idxs, idx_mask)
                        if reached:
                            # print("\n here! c")
                            print(f"\nround:{round}")
                            current_config = {"current_num_agents": self.num_agents, "round": round}
                            return {"response": reply},current_config

        response = self._most_frequent([self.nodes[idx]["answer"] or "" for idx in idxs], self.cmp_res)[0]
        # print("\n here! d")
        print(f"\nround:{round}")
        current_config = {"current_num_agents": self.num_agents, "round": round}
        return {"response": response},current_config

    def _get_activation_map(self):
        return {"listwise": 0, "trueskill": 1, "window": 2, "none": -1}

    def _init_network(self):
        for idx in range(self.num_agents):
            self.nodes.append(
                {
                    "role": self.roles[idx],
                    "reply": None,
                    "answer": "",
                    "active": False,
                    "importance": 0,
                    "to_edges": [],
                    "from_edges": [],
                    "question": None,
                }
            )

        first_round_nodes = list(range(self.num_agents))

        for rid in range(1, self.num_rounds):
            start_idx = rid * self.num_agents
            for idx in range(self.num_agents):
                node_idx = start_idx + idx
                self.nodes.append(
                    {
                        "role": self.roles[idx],
                        "reply": None,
                        "answer": "",
                        "active": False,
                        "importance": 0,
                        "to_edges": [],
                        "from_edges": [],
                        "question": None,
                    }
                )
                for prev_idx in first_round_nodes:
                    edge = {"weight": 0, "from": prev_idx, "to": node_idx}
                    self.edges.append(edge)
                    self.nodes[prev_idx]["to_edges"].append(len(self.edges) - 1)
                    self.nodes[node_idx]["from_edges"].append(len(self.edges) - 1)
            first_round_nodes = list(range(start_idx, start_idx + self.num_agents))

    def _zero_grad(self):
        for edge in self.edges:
            edge["weight"] = 0

    def _deactivate_node(self, idx):
        self.nodes[idx]["active"] = False
        self.nodes[idx]["reply"] = None
        self.nodes[idx]["answer"] = ""
        self.nodes[idx]["question"] = None
        self.nodes[idx]["importance"] = 0

    def _set_allnodes_deactivated(self):
        for idx in range(len(self.nodes)):
            self._deactivate_node(idx)

    def _get_role_map(self):
        return {
            "Assistant": "You are a super-intelligent AI assistant and medical expert with deep clinical knowledge.",
            "Clinician": "You are an experienced clinician skilled in diagnosis, treatment planning, and evidence-based medicine.",
            "Pharmacologist": "You are a pharmacologist with expertise in drug mechanisms, interactions, and therapeutic protocols.",
            "Anatomist": "You are an anatomist with deep knowledge of human anatomy, physiology, and pathophysiology.",
            "Pathologist": "You are a pathologist skilled in disease mechanisms, laboratory diagnostics, and clinical correlations.",
            "Doctor": "You are a doctor and medical expert who provides evidence-based diagnoses and treatment plans.",
            "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist": "You are an economist. You are good at economics, finance, and business.",
            "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy.",
            "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
            "Programmer": "You are a programmer. You are good at computer science, engineering, and physics.",
            "Historian": "You are a historian. You research and analyze events in the past.",
        }

    def _construct_message(self, responses, question):
        if len(responses) == 0:
            return {"role": "user", "content": question}

        prefix_string = question + "\n\nThese are the responses from other agents: "
        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent response " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = (
            prefix_string
            + "\n\nUsing the answer from other agents as additional advice with critical thinking, can you give an updated answer? "
            "Examine your solution and that other agents step by step. Notice that their answers might be all wrong. "
            "Please answer the question in detail. Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. "
            "Put all {} scores in the form like [[1, 5, 2, ...]]. ".format(len(responses))
            + "End your response with your final answer choice in the format: Answer: X"
        )

        return {"role": "user", "content": prefix_string}

    def _construct_ranking_message(self, responses, question):
        prefix_string = question + "\n\nThese are the responses from other agents: "
        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent response " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 answers and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
        return {"role": "user", "content": prefix_string}

    def _parse_ranks(self, completion, max_num=4):
        if not completion:
            return random.sample(list(range(max_num)), 2)
        pattern = r"\[([1234567]),\s*([1234567])\]"
        matches = re.findall(pattern, completion)

        try:
            match = matches[-1]
            tops = [int(match[0]) - 1, int(match[1]) - 1]

            def clip(x):
                if x < 0:
                    return 0
                if x > max_num - 1:
                    return max_num - 1
                return x

            tops = [clip(x) for x in tops]
        except Exception:
            tops = random.sample(list(range(max_num)), 2)

        return tops

    def _find_array(self, text):
        if not text:
            return [0]
        matches = re.findall(r"\[\[(.*?)\]\]", text)
        if matches:
            last_match = matches[-1].replace(" ", "")

            def convert(x):
                try:
                    return int(x)
                except Exception:
                    return 0

            try:
                return list(map(convert, last_match.split(",")))
            except Exception:
                return [0]
        return [0]

    def _activate_node(self, idx, question):
        role_map = self._get_role_map()
        role = self.nodes[idx]["role"]
        role_prompt = role_map.get(role, role_map["Assistant"])

        responses = [self.nodes[_idx]["answer"] for _idx in range(self.num_agents) if self.nodes[_idx]["answer"]]
        message = self._construct_message(responses, question)
        # 首次激活（Round 0，无历史回复）时传入图片；后续轮次不再传图片以节省 token
        use_img = self._img_paths if not responses else None
        completion = self.call_llm(messages=[{"role": "system", "content": role_prompt}, message], img_paths=use_img)
        if not completion:
            completion = ""

        self.nodes[idx]["active"] = True
        self.nodes[idx]["reply"] = completion
        self.nodes[idx]["answer"] = completion
        self.nodes[idx]["question"] = question
        self.nodes[idx]["importance"] = 1

    def _check_consensus(self, activated_indices, idx_mask):
        answers = [self.nodes[idx]["answer"] for idx in activated_indices]
        consensus_answer, ca_cnt = self._most_frequent(answers, self.cmp_res)
        if ca_cnt > math.floor(2 / 3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def _listwise_ranker(self, responses, question):
        assert 2 < len(responses)
        message = self._construct_ranking_message(responses, question)
        completion = self.call_llm(messages=[message]) or ""
        return self._parse_ranks(completion, max_num=len(responses))

    def _most_frequent(self, clist, cmp_func):
        counter = 0
        num = clist[0]
        for i in clist:
            current_frequency = sum(cmp_func(i, item) for item in clist)
            if current_frequency > counter:
                counter = current_frequency
                num = i
        return num, counter

    def _sentence_bleu(self, reference, hypotheses, lowercase=False):
        if not reference or not hypotheses:
            return 0
        reference = str(reference)
        hypotheses = [str(h) for h in hypotheses if h]
        if not hypotheses:
            return 0
        if lowercase:
            reference = reference.lower()
            hypotheses = [h.lower() for h in hypotheses]

        if not hypotheses:
            return 0
        ref_tokens = reference.split()
        scores = []
        for hyp in hypotheses:
            hyp_tokens = hyp.split()
            matches = sum(1 for token in hyp_tokens if token in ref_tokens)
            precision = matches / len(hyp_tokens) if hyp_tokens else 0
            scores.append(precision * 100)
        return max(scores)
