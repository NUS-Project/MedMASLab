"""
Evaluate existing JSONL outputs with 5 evaluation protocols.

Reads pre-computed JSONL files (containing final_decision) and applies
5 evaluation protocols without re-running inference. Only needs a Judge
LLM (Qwen2.5-VL-32B-Instruct) for Protocols 1 & 2.

Usage:
  python eval_from_existing_jsonl.py \
    --input-dir Eval_yunhang \
    --data-dir data \
    --output-dir output_eval_yunhang \
    --stats-log-dir stats_logs_eval_yunhang \
    --methods Debate LLM-Discussion MDAgents MDTeamGPT Reconcile Qwen2.5-VL-7B-Instruct \
    --datasets dxbench pubmedqa VQA_RAD \
    --base-model Qwen2.5-VL-7B-Instruct \
    --judge-url http://localhost:8003/v1
"""

import argparse
import json
import os
import re
import glob
import time
from datetime import datetime
from openai import OpenAI


def load_jsonl(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def load_questions(data_dir, dataset):
    """Load test data and build id->question mapping."""
    test_path = os.path.join(data_dir, dataset, "test.jsonl")
    id2question = {}
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("id", "")
            q = d.get("question", "")
            opts = d.get("options", "")
            if opts:
                if isinstance(opts, list):
                    opts = "\n".join(opts)
                q = f"{q}\nOptions: {opts}"
            id2question[sid] = q
    return id2question


def find_input_file(input_dir, method, base_model, dataset):
    path = os.path.join(input_dir, f"{method}_{base_model}_{dataset}.jsonl")
    if os.path.isfile(path):
        return path
    return None


# ── Protocol 1: LLM-Semantic-Judge ──
def judge_prompt(question, response, right_label, right_context, answer_type):
    if answer_type and answer_type.lower() == "closed":
        at_desc = "There is only one correct answer content and the corresponding correct answer label."
    else:
        at_desc = "This question has more than one correct answer and corresponding labels for the correct answers. This reference answer is for reference only."

    system = (
        f"You are a diligent and precise medical assistant tasked with evaluating "
        f"the correctness of responses. \nYou will receive a question, an output "
        f"sentence from agent, the correct answer content, and the corresponding "
        f"correct answer label.{at_desc} Your tasks are as follows:\n"
        f"1.Determine if you can find the agent's answer in the output sentence "
        f"from agent. If you can not find the agent's answer in the output sentence "
        f"from agent, respond with [wrong].\n"
        f"2.If you find the agent's answer, evaluate whether it accurately answers "
        f"the question based on the provided right answer content and right answer "
        f"label. Respond with either [right] or [wrong].\n"
        f"REMEMBER: Your judgment result must be only [right], or [wrong]."
    )

    user = (
        f"Question: {question}\n\n"
        f"Output sentence: {response}\n"
    )
    if right_label:
        user += f"right answer label: {right_label}\n"
    user += f"right answer context: {right_context}\n\nJudgement Result:"

    return system, user


def eval_llm_semantic_judge(samples, id2question, llm_client, llm_model):
    correct = 0
    judge_results = []
    for s in samples:
        existing_jr = str(s.get("Judge_result", "")).strip()
        if existing_jr:
            cleaned = re.sub(r"[^a-zA-Z]", "", existing_jr).lower()
            is_correct = not cleaned.startswith("wrong") and bool(existing_jr)
            if is_correct:
                correct += 1
            judge_results.append(existing_jr)
            continue

        sid = s.get("id", "")
        question = id2question.get(sid, "")
        fd = str(s.get("final_decision", ""))
        right_label = str(s.get("right_option", ""))
        right_context = str(s.get("answer", ""))
        answer_type = str(s.get("answer_type", "closed"))

        system, user = judge_prompt(question, fd, right_label, right_context, answer_type)

        try:
            resp = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=20,
                temperature=0,
            )
            jr = resp.choices[0].message.content.strip()
        except Exception as e:
            jr = f"[error: {e}]"

        cleaned = re.sub(r"[^a-zA-Z]", "", jr).lower()
        is_correct = not cleaned.startswith("wrong") and bool(jr) and "error" not in jr
        if is_correct:
            correct += 1
        judge_results.append(jr)

    return correct, len(samples), judge_results


# ── Protocol 2: LLM-Extract-Compare ──
def eval_llm_extract_compare(samples, llm_client, llm_model):
    correct = 0
    for s in samples:
        fd = str(s.get("final_decision", ""))
        gt_label = str(s.get("right_option", "")).strip().upper()
        if not gt_label or not fd:
            continue

        prompt = (
            f"Given the following model response to a medical question, extract ONLY "
            f"the final answer option letter (e.g. A, B, C, D, E). "
            f"If no clear option letter is present, respond with 'NONE'.\n\n"
            f"Model response: {fd[:1500]}\n\n"
            f"Extracted answer letter:"
        )
        try:
            resp = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            extracted = resp.choices[0].message.content.strip().upper()
            letter_match = re.search(r"[A-E]", extracted)
            if letter_match and letter_match.group(0) == gt_label:
                correct += 1
        except Exception:
            pass
    return correct, len(samples)


# ── Protocol 3: Rule-Multi-Regex ──
def eval_rule_multi_regex(samples):
    correct = 0
    for s in samples:
        fd_str = str(s.get("final_decision", ""))
        gt = str(s.get("right_option", "")).strip().lower()
        if not gt:
            continue

        extracted = None
        opt_match = re.search(r"\(([A-Ea-e])\)", fd_str)
        if opt_match:
            extracted = opt_match.group(1).lower()
        else:
            lead = re.match(r"^\s*([A-Ea-e])(?:\s|[^a-zA-Z])", fd_str)
            if lead:
                extracted = lead.group(1).lower()
            else:
                cleaned = re.sub(r"[^a-zA-Z]", "", fd_str).lower()
                if len(cleaned) == 1 and cleaned in "abcde":
                    extracted = cleaned
                else:
                    m = re.search(r"answer\s*(?:is|:)?\s*(?:\(?([a-eA-E])\)?)", fd_str, re.IGNORECASE)
                    if m:
                        extracted = m.group(1).lower()
                    else:
                        m2 = re.search(r"option\s*(?:is|:)?\s*(?:\(?([a-eA-E])\)?)", fd_str, re.IGNORECASE)
                        if m2:
                            extracted = m2.group(1).lower()
        if extracted and extracted == gt:
            correct += 1
    return correct, len(samples)


# ── Protocol 4: Rule-First-Letter ──
def eval_rule_first_letter(samples):
    correct = 0
    for s in samples:
        fd_str = str(s.get("final_decision", ""))
        gt = str(s.get("right_option", "")).strip().upper()
        if not gt:
            continue
        m = re.search(r"[A-E]", fd_str.upper())
        if m and m.group(0) == gt:
            correct += 1
    return correct, len(samples)


# ── Protocol 5: Rule-Exact-Match ──
def eval_rule_exact_match(samples):
    correct = 0
    for s in samples:
        fd_str = str(s.get("final_decision", "")).strip()
        gt = str(s.get("right_option", "")).strip().lower()
        if not gt:
            continue
        fd_clean = fd_str.strip().lower()
        if fd_clean == gt:
            correct += 1
        elif len(fd_clean) >= 1 and fd_clean[0] == gt and (len(fd_clean) == 1 or not fd_clean[1].isalpha()):
            correct += 1
    return correct, len(samples)


PROTOCOLS = [
    "LLM-Semantic-Judge",
    "LLM-Extract-Compare",
    "Rule-Multi-Regex",
    "Rule-First-Letter",
    "Rule-Exact-Match",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing JSONL with 5 protocols")
    parser.add_argument("--input-dir", default="Eval_yunhang")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output_eval_yunhang")
    parser.add_argument("--stats-log-dir", default="stats_logs_eval_yunhang")
    parser.add_argument("--methods", nargs="+",
                        default=["Debate", "LLM-Discussion", "MDAgents", "MDTeamGPT",
                                 "Reconcile", "Qwen2.5-VL-7B-Instruct"])
    parser.add_argument("--datasets", nargs="+", default=["dxbench", "pubmedqa", "VQA_RAD"])
    parser.add_argument("--base-model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--judge-url", default=None)
    parser.add_argument("--judge-model", default="Qwen2.5-VL-32B-Instruct")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.stats_log_dir, exist_ok=True)

    llm_client = None
    if args.judge_url:
        llm_client = OpenAI(api_key="EMPTY", base_url=args.judge_url)

    for dataset in args.datasets:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*70}")

        id2question = load_questions(args.data_dir, dataset)
        print(f"  Loaded {len(id2question)} questions from {dataset}")

        all_results = {}

        for method in args.methods:
            fpath = find_input_file(args.input_dir, method, args.base_model, dataset)
            if not fpath:
                print(f"\n  [WARN] No file: {method}_{args.base_model}_{dataset}.jsonl")
                continue

            samples = load_jsonl(fpath)
            if not samples:
                print(f"\n  [WARN] Empty: {fpath}")
                continue

            total = len(samples)
            print(f"\n  {'='*60}")
            print(f"  Method: {method}  |  Samples: {total}")
            print(f"  {'='*60}")

            results = {}
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Protocol 1: LLM-Semantic-Judge
            if llm_client:
                t0 = time.time()
                c, t, judge_results_list = eval_llm_semantic_judge(
                    samples, id2question, llm_client, args.judge_model)
                elapsed = time.time() - t0
                results["LLM-Semantic-Judge"] = (c, t)
                print(f"    LLM-Semantic-Judge:   {c}/{t} = {c/t*100:.2f}%  ({elapsed:.1f}s)")

                for i, s in enumerate(samples):
                    if i < len(judge_results_list):
                        s["Judge_result_new"] = judge_results_list[i]
            else:
                c, t = 0, total
                for s in samples:
                    jr = str(s.get("Judge_result", "")).strip()
                    if jr:
                        cleaned = re.sub(r"[^a-zA-Z]", "", jr).lower()
                        if not cleaned.startswith("wrong"):
                            c += 1
                results["LLM-Semantic-Judge"] = (c, t)
                print(f"    LLM-Semantic-Judge:   {c}/{t} = {c/t*100:.2f}%  (from existing)")

            # Protocol 2: LLM-Extract-Compare
            if llm_client:
                t0 = time.time()
                c, t = eval_llm_extract_compare(samples, llm_client, args.judge_model)
                elapsed = time.time() - t0
                results["LLM-Extract-Compare"] = (c, t)
                print(f"    LLM-Extract-Compare:  {c}/{t} = {c/t*100:.2f}%  ({elapsed:.1f}s)")
            else:
                results["LLM-Extract-Compare"] = (None, total)
                print(f"    LLM-Extract-Compare:  [skipped - no --judge-url]")

            # Protocol 3-5
            c, t = eval_rule_multi_regex(samples)
            results["Rule-Multi-Regex"] = (c, t)
            print(f"    Rule-Multi-Regex:     {c}/{t} = {c/t*100:.2f}%")

            c, t = eval_rule_first_letter(samples)
            results["Rule-First-Letter"] = (c, t)
            print(f"    Rule-First-Letter:    {c}/{t} = {c/t*100:.2f}%")

            c, t = eval_rule_exact_match(samples)
            results["Rule-Exact-Match"] = (c, t)
            print(f"    Rule-Exact-Match:     {c}/{t} = {c/t*100:.2f}%")

            all_results[method] = results

            # Write output JSONL
            out_path = os.path.join(
                args.output_dir,
                f"{method}_{args.base_model}_{dataset}_eval5.jsonl"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                for s in samples:
                    for p in PROTOCOLS:
                        c_p, t_p = results.get(p, (None, 0))
                        if c_p is not None and t_p > 0:
                            s[f"eval_{p}_acc"] = f"{c_p/t_p*100:.2f}%"
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            # Write stats log
            log_path = os.path.join(
                args.stats_log_dir,
                f"{method}_{dataset}_{args.base_model}_{ts}.log"
            )
            with open(log_path, "w") as lf:
                lf.write(f"Task: {method} on {dataset} (re-evaluation from existing JSONL)\n")
                lf.write(f"Base Model: {args.base_model}\n")
                lf.write(f"Judge Model: {args.judge_model}\n")
                lf.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                lf.write(f"Input File: {fpath}\n")
                lf.write(f"{'='*50}\n")
                lf.write(f"[INFO] Total Samples: {total}\n")
                lf.write(f"[INFO] === Evaluation Protocol Comparison ===\n")
                for p in PROTOCOLS:
                    c_p, t_p = results.get(p, (None, 0))
                    if c_p is not None and t_p > 0:
                        lf.write(f"[INFO] {p}: {c_p/t_p*100:.2f}% ({c_p}/{t_p})\n")
                    else:
                        lf.write(f"[INFO] {p}: N/A\n")
            print(f"    Stats log: {log_path}")

        # Write CSV for this dataset
        csv_path = os.path.join(
            args.output_dir,
            f"eval_comparison_{dataset}_{args.base_model}.csv"
        )
        with open(csv_path, "w") as f:
            f.write("Method," + ",".join(PROTOCOLS) + "\n")
            for method in args.methods:
                if method not in all_results:
                    continue
                row = [method]
                for p in PROTOCOLS:
                    c, t = all_results[method].get(p, (None, 0))
                    if c is not None and t > 0:
                        row.append(f"{c/t*100:.2f}")
                    else:
                        row.append("N/A")
                f.write(",".join(row) + "\n")
        print(f"\n  CSV saved: {csv_path}")

        # Print summary table
        print(f"\n  {'='*80}")
        print(f"  SUMMARY: {dataset} + {args.base_model}")
        print(f"  {'='*80}")
        header = f"  {'Method':<25}"
        for p in PROTOCOLS:
            header += f" {p:<22}"
        print(header)
        print("  " + "-" * 78)
        for method in args.methods:
            if method not in all_results:
                continue
            row = f"  {method:<25}"
            for p in PROTOCOLS:
                c, t = all_results[method].get(p, (None, 0))
                if c is not None and t > 0:
                    row += f" {c/t*100:>6.2f}% ({c}/{t})     "
                else:
                    row += f" {'N/A':>6}               "
            print(row)


if __name__ == "__main__":
    main()
