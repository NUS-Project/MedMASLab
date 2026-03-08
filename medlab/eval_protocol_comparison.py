"""
Evaluation Protocol Comparison for MedMASLab.

Reads existing output JSONLs and re-evaluates each sample's `final_decision`
using 5 different evaluation protocols (2 VLM-based + 3 rule-based),
analogous to MASLab's evaluation comparison experiment.

Protocols:
  1. VLM-Semantic-Judge   (≈ MASLab VLM-xVerify)  - full semantic judge
  2. VLM-Extract-Compare  (≈ MASLab VLM-2Step)    - VLM extracts answer, then compare
  3. Rule-Multi-Regex     (≈ MASLab Rule-DyLAN)    - multi-pattern regex extraction
  4. Rule-First-Letter    (≈ MASLab Rule-HF)       - first valid option letter
  5. Rule-Exact-Match     (≈ MASLab Rule-Hendrycks) - strict exact string match

Usage:
  python eval_protocol_comparison.py \
    --output-dir output \
    --dataset pubmedqa \
    --methods ColaCare autogen MedAgents dylan MetaPrompting \
    --base-model Qwen2.5-VL-7B-Instruct \
    [--VLM-url http://localhost:8003/v1]  # for VLM-Extract-Compare protocol
"""

import argparse
import json
import os
import re
import glob
from collections import defaultdict
from openai import OpenAI


def find_output_file(output_dir, method, base_model, dataset):
    pattern = os.path.join(output_dir, f"{method}_{base_model}_{dataset}_*.jsonl")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


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


# ── Protocol 1: VLM-Semantic-Judge ──
# Already stored in the JSONL as `is_correct` and `Judge_result`
def eval_VLM_semantic_judge(samples):
    correct = 0
    for s in samples:
        judge = str(s.get("Judge_result", ""))
        cleaned = re.sub(r"[^a-zA-Z]", "", judge).lower()
        if not cleaned.startswith("wrong") and judge.strip():
            if s.get("is_correct", False):
                correct += 1
    return correct, len(samples)


# ── Protocol 2: VLM-Extract-Compare ──
# Two-step: VLM extracts the answer letter, then exact-match with ground truth
def eval_VLM_extract_compare(samples, VLM_client, VLM_model):
    correct = 0
    for s in samples:
        fd = str(s.get("final_decision", ""))
        question = str(s.get("question", s.get("id", "")))
        gt_label = str(s.get("right_option", "")).strip().upper()
        if not gt_label:
            gt_label = str(s.get("answer_idx", "")).strip().upper()
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
            resp = VLM_client.chat.completions.create(
                model=VLM_model,
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
# Multiple regex patterns to extract option letter (same as main_rulebased.py)
def eval_rule_multi_regex(samples):
    correct = 0
    for s in samples:
        fd_str = str(s.get("final_decision", ""))
        gt = str(s.get("right_option", "")).strip().lower()
        if not gt:
            gt = str(s.get("answer_idx", "")).strip().lower()
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
# Extract only the very first uppercase letter (A-E) found in the response
def eval_rule_first_letter(samples):
    correct = 0
    for s in samples:
        fd_str = str(s.get("final_decision", ""))
        gt = str(s.get("right_option", "")).strip().upper()
        if not gt:
            gt = str(s.get("answer_idx", "")).strip().upper()
        if not gt:
            continue

        m = re.search(r"[A-E]", fd_str.upper())
        if m and m.group(0) == gt:
            correct += 1
    return correct, len(samples)


# ── Protocol 5: Rule-Exact-Match ──
# Strict: strip and lowercase everything, check if response starts with or equals the letter
def eval_rule_exact_match(samples):
    correct = 0
    for s in samples:
        fd_str = str(s.get("final_decision", "")).strip()
        gt = str(s.get("right_option", "")).strip().lower()
        if not gt:
            gt = str(s.get("answer_idx", "")).strip().lower()
        if not gt:
            continue

        fd_clean = fd_str.strip().lower()
        if fd_clean == gt:
            correct += 1
        elif len(fd_clean) >= 1 and fd_clean[0] == gt and (len(fd_clean) == 1 or not fd_clean[1].isalpha()):
            correct += 1
    return correct, len(samples)


def main():
    parser = argparse.ArgumentParser(description="Evaluation Protocol Comparison")
    parser.add_argument("--output-dir", default="output", help="Directory with output JSONLs")
    parser.add_argument("--dataset", default="pubmedqa")
    parser.add_argument("--methods", nargs="+",
                        default=["ColaCare", "autogen", "MedAgents", "dylan", "MetaPrompting"])
    parser.add_argument("--base-model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--VLM-url", default="https://api.vectorengine.ai/v1", help="vVLM URL for VLM-Extract-Compare")
    parser.add_argument("--VLM-model", default="Qwen2.5-VL-7B-Instruct", help="Model for extraction")
    parser.add_argument("--result-file", default=None, help="Output result CSV path")
    parser.add_argument("--stats-log-dir", default=None,
                        help="If set, append 5-protocol accuracies to each method's stats log")
    args = parser.parse_args()

    VLM_client = None
    if args.VLM_url:
        VLM_client = OpenAI(api_key="EMPTY", base_url=args.VLM_url)

    protocols = [
        "VLM-Semantic-Judge",
        "VLM-Extract-Compare",
        "Rule-Multi-Regex",
        "Rule-First-Letter",
        "Rule-Exact-Match",
    ]

    all_results = {}

    for method in args.methods:
        fpath = find_output_file(args.output_dir, method, args.base_model, args.dataset)
        if not fpath:
            print(f"[WARN] No output file found for {method} + {args.base_model} + {args.dataset}")
            continue

        samples = load_jsonl(fpath)
        if not samples:
            print(f"[WARN] Empty file: {fpath}")
            continue

        total = len(samples)
        print(f"\n{'='*60}")
        print(f"Method: {method}  |  File: {os.path.basename(fpath)}  |  Samples: {total}")
        print(f"{'='*60}")

        results = {}

        # Protocol 1
        c, t = eval_VLM_semantic_judge(samples)
        results["VLM-Semantic-Judge"] = (c, t)
        print(f"  VLM-Semantic-Judge:   {c}/{t} = {c/t*100:.2f}%")

        # Protocol 2
        if VLM_client:
            c, t = eval_VLM_extract_compare(samples, VLM_client, args.VLM_model)
            results["VLM-Extract-Compare"] = (c, t)
            print(f"  VLM-Extract-Compare:  {c}/{t} = {c/t*100:.2f}%")
        else:
            results["VLM-Extract-Compare"] = (None, total)
            print(f"  VLM-Extract-Compare:  [skipped - no --VLM-url]")

        # Protocol 3
        c, t = eval_rule_multi_regex(samples)
        results["Rule-Multi-Regex"] = (c, t)
        print(f"  Rule-Multi-Regex:     {c}/{t} = {c/t*100:.2f}%")

        # Protocol 4
        c, t = eval_rule_first_letter(samples)
        results["Rule-First-Letter"] = (c, t)
        print(f"  Rule-First-Letter:    {c}/{t} = {c/t*100:.2f}%")

        # Protocol 5
        c, t = eval_rule_exact_match(samples)
        results["Rule-Exact-Match"] = (c, t)
        print(f"  Rule-Exact-Match:     {c}/{t} = {c/t*100:.2f}%")

        all_results[method] = results

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: {args.dataset} + {args.base_model}")
    print(f"{'='*80}")
    header = f"{'Method':<20}"
    for p in protocols:
        header += f" {p:<22}"
    print(header)
    print("-" * 80)
    for method in args.methods:
        if method not in all_results:
            continue
        row = f"{method:<20}"
        for p in protocols:
            c, t = all_results[method].get(p, (None, 0))
            if c is not None and t > 0:
                row += f" {c/t*100:>6.2f}% ({c}/{t})     "
            else:
                row += f" {'N/A':>6}               "
        print(row)

    # Save to CSV
    result_path = args.result_file or os.path.join(
        args.output_dir, f"eval_protocol_comparison_{args.dataset}_{args.base_model}.csv"
    )
    with open(result_path, "w") as f:
        f.write("Method," + ",".join(protocols) + "\n")
        for method in args.methods:
            if method not in all_results:
                continue
            row_vals = [method]
            for p in protocols:
                c, t = all_results[method].get(p, (None, 0))
                if c is not None and t > 0:
                    row_vals.append(f"{c/t*100:.2f}")
                else:
                    row_vals.append("N/A")
            f.write(",".join(row_vals) + "\n")
    print(f"\nResults saved to: {result_path}")

    # Append 5-protocol results to each method's stats log
    if args.stats_log_dir and os.path.isdir(args.stats_log_dir):
        for method in args.methods:
            if method not in all_results:
                continue
            pattern = os.path.join(
                args.stats_log_dir,
                f"{method}_{args.dataset}_{args.base_model}_*.log"
            )
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if not matches:
                print(f"[WARN] No stats log found for {method} (pattern: {pattern})")
                continue
            log_path = matches[0]
            with open(log_path, "a") as lf:
                lf.write("==================================================\n")
                lf.write("[INFO] === Evaluation Protocol Comparison ===\n")
                for p in protocols:
                    c, t = all_results[method].get(p, (None, 0))
                    if c is not None and t > 0:
                        lf.write(f"[INFO] {p}: {c/t*100:.2f}% ({c}/{t})\n")
                    else:
                        lf.write(f"[INFO] {p}: N/A\n")
            print(f"[INFO] Updated stats log: {log_path}")


if __name__ == "__main__":
    main()
