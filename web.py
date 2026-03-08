"""
MedMASLab Web Demo
======================
A user-friendly web interface for multi-agent medical QA evaluation.

Usage:
    python web_demo.py [--port 7890] [--share]
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Patch Gradio networking check for HPC environments where localhost
# is not directly reachable but the server still binds correctly.
import gradio.networking as _gnet

_gnet.url_ok = lambda *a, **kw: True

import gradio as gr  # noqa: E402

# Patch gradio_client JSON schema bug (bool schema not handled)
try:
    import gradio_client.utils as _gc_utils
    _orig_json_schema = _gc_utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_METHODS = ["ColaCare", "MetaPrompting"]
BATCH_METHODS = ["Debate", "Reconcile", "Discussion","MDAgents","MDTeamGPT", "autogen", "dylan","MedAgents"]
BUILTIN_METHODS = API_METHODS + BATCH_METHODS

AVAILABLE_DATASETS = [
    "medqa", "pubmedqa", "dxbench", "VQA_RAD",
    "MedCXR", "slake", "M3CoTBench", "MMLU",
]

METHOD_INFO: Dict[str, Dict[str, Any]] = {
    "ColaCare": {
        "desc": "Multi-specialty collaborative consultation: Internal Medicine, Surgery, and Radiology doctors discuss, followed by a Meta-doctor who synthesizes a final decision.",
        "agents": 3,
        "icon": "🏥",
    },
    "MedAgents": {
        "desc": "Domain-aware expert recruitment: automatically selects question-domain and option-domain medical experts, who independently analyze and then vote on the answer.",
        "agents": "5-12",
        "icon": "👨‍⚕️",
    },
    "MetaPrompting": {
        "desc": "Hierarchical meta-orchestration: a Meta-Expert delegates sub-problems to dynamically summoned specialist experts and synthesizes their outputs.",
        "agents": 4,
        "icon": "🧠",
    },
    "autogen": {
        "desc": "AutoGen-based structured multi-agent conversation with medical domain specialization.",
        "agents": 2,
        "icon": "🤖",
    },
    "dylan": {
        "desc": "Dynamic Agent Network: agents communicate through confidence-weighted edges, forming an adaptive reasoning graph.",
        "agents": 4,
        "icon": "🌐",
    },
    "Debate": {
        "desc": "Multi-agent debate: several agents propose answers, argue over disagreements, and converge to a consensus through structured debate rounds.",
        "agents": 3,
        "icon": "⚔️",
    },
    "Reconcile": {
        "desc": "Round-table reconciliation: agents provide initial answers, then iteratively discuss and reconcile differences until consensus.",
        "agents": 3,
        "icon": "🤝",
    },
    "Discussion": {
        "desc": "Open-ended multi-agent discussion: agents engage in free-form collaborative reasoning to reach a joint conclusion.",
        "agents": 3,
        "icon": "💬",
    },
    "MDTeamGPT":{
        "desc":"MDTeamGPT simulates MDT consultations for static medical cases. We overhaul its case-presentation module to systematically process temporal video frames and impose strict multimodal grounding rules on inter-agent peer-review protocols to guarantee standardized evaluations.",
        "agents": 6,
        "icon": "🤖",
    },
    "MDAgents":{
        "desc":"MDAgents provides an adaptive collaboration topology dynamically determined by medical query complexity. We re-architect its routing logic to concurrently evaluate multimodal spatial-temporal complexity, replacing its free-text output with a schema-driven aggregation module.",
        "agents": 6,
        "icon": "🤖",
    },
    "MedAgentAudit": {
        "desc": "Real-time multimodal consensus monitoring node that audits collaborative inference, actively detecting and intervening against spurious consensuses lacking video visual evidence.",
        "agents": "3+",
        "icon": "🔍",
        "category": "Medical MAS",
    },
    "MedLA": {
        "desc": "Syllogistic logic trees for complex medical reasoning: anchors deduction tree nodes to key feature frames in medical videos, ensuring semantically self-consistent and visually evidence-based inference.",
        "agents": 3,
        "icon": "🌳",
        "category": "Medical MAS",
    },
    "CXRAgent": {
        "desc": "Multi-stage reasoning pipeline with a Director agent and Evidence-Driven Validator (EDV) for continuous evaluation of dynamic temporal segments, anchoring diagnostics to video visual evidence.",
        "agents": "3-4",
        "icon": "🫁",
        "category": "Medical MAS",
    },
    "LINS": {
        "desc": "Multi-agent retrieval-augmented framework generating citation-grounded medical texts, cross-validating with dynamic video frames via the MAIRAG algorithm for hallucination-free results.",
        "agents": 3,
        "icon": "📚",
        "category": "Medical MAS",
    },
    "MedOrch": {
        "desc": "Orchestrates domain-specific medical tools and reasoning agents with spatiotemporal visual analysis tools, ensuring high traceability of diagnostic processes and deterministic structured outputs.",
        "agents": "3+",
        "icon": "🎼",
        "category": "Medical MAS",
    },
    "MoMA": {
        "desc": "Mixture of Multimodal Agents: assigns specialist agents to extract temporal pathological features from medical videos, with an Aggregator Agent forming clinical consensus via continuous vision-language alignment.",
        "agents": "3+",
        "icon": "🧩",
        "category": "Medical MAS",
    },"CoT": {
        "desc": "Vision-driven medical chain-of-thought: guides models to generate intermediate reasoning steps, explicitly citing anatomical changes and lesion features to reduce hallucination.",
        "agents": 1,
        "icon": "💭",
        "category": "Single Agent",
    },
    "VLM": {
        "desc": "Advanced multimodal foundation model (e.g., Qwen3-VL) with native long-context and video temporal understanding, directly parsing medical video streams for end-to-end clinical answers.",
        "agents": 1,
        "icon": "👁️",
        "category": "Single Agent",
    },
    # ── Multi-Agent Systems for General Tasks ──
    "Self-Consistency": {
        "desc": "Samples multiple reasoning paths and marginalizes to replace greedy decoding, extended to jointly evaluate medical semantic deduction with spatial-temporal visual features for cross-modal self-consistency.",
        "agents": 1,
        "icon": "🔄",
        "category": "General MAS",
    }
}

EXAMPLE_QUESTIONS = [
    [
        "A 55-year-old male presents with sudden onset of severe chest pain radiating to the back, associated with diaphoresis and hypertension. CT angiogram reveals a dissection flap in the descending aorta. What is the most likely diagnosis?",
        "(A) Acute Myocardial Infarction (B) Pulmonary Embolism (C) Aortic Dissection (D) Pneumothorax (E) Pericarditis",
    ],
    [
        "A 30-year-old woman presents with fatigue, weight gain, cold intolerance, and constipation. Lab results show elevated TSH and low free T4. What is the most appropriate initial treatment?",
        "(A) Levothyroxine (B) Methimazole (C) Propylthiouracil (D) Radioactive iodine (E) Thyroidectomy",
    ],
    [
        "Does metformin reduce the risk of cardiovascular events in patients with type 2 diabetes?",
        "(A) Yes (B) No (C) Maybe",
    ],
]

# ---------------------------------------------------------------------------
# 1. Global Configuration (in-memory, not persisted)
# ---------------------------------------------------------------------------

_global_config = {
    "base_url": None,
    "base_key": None,
    "base_model": None,
    "judge_url": None,
    "judge_key": None,
    "judge_model": None,
}

_config_lock = threading.Lock()


def _set_global_config(base_url: str, base_key: str, base_model: str,
                       judge_url: str = None, judge_key: str = None, judge_model: str = None):
    """Save configuration to global variables (not persisted to disk)"""
    global _global_config
    with _config_lock:
        _global_config = {
            "base_url": base_url,
            "base_key": base_key,
            "base_model": base_model,
            "judge_url": base_url,
            "judge_key": base_key,
            "judge_model": base_model,
        }


def _get_global_config():
    """Get global configuration"""
    with _config_lock:
        return _global_config.copy()


# ---------------------------------------------------------------------------
# 2. Test Connection (with conversation log display)
# ---------------------------------------------------------------------------

def test_connection(base_url: str, api_key: str, model_name: str):
    """
    Send 'hello!' message, display conversation, judge connection success.
    On success, save config to global variables.
    """
    if not base_url:
        return "⚠️ Please enter the API Base URL."
    if not model_name:
        return "⚠️ Please enter the Model Name."

    try:
        # ── Create client ──
        client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)

        # ── Build message ──
        messages = [{"role": "user", "content": "hello!"}]

        # ── Display request info ──
        conversation_log = (
            f"📤 **Sending Request**\n"
            f"```\n"  
            f"Base URL: {base_url}\n"  
            f"Model: {model_name}\n"  
            f"API Key: {api_key if api_key and api_key != 'EMPTY' else '(None - local mode)'}\n"  
            f"Message: hello!\n"  
            f"```\n\n"
        )

        # ── Call API ──
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )

        # ── Extract response ──
        answer = response.choices[0].message.content

        # ── Check if empty ──
        if not answer or answer.strip() == "":
            conversation_log += (
                f"📥 **Received Response**\n"
                f"❌ **Empty Response** - Connection failed!\n"
            )
            return conversation_log

        # ✨ Success: Save to global config ✨
        _set_global_config(base_url, api_key, model_name)
        print("✨ Success: Save to global config ✨")

        # ── Display full conversation ──
        conversation_log += (
            f"📥 **Received Response**\n"
            f"```\n"  
            f"{answer}\n"  
            f"```\n\n"
            f"✅ **Connection Successful!**\n"
            f"{'═' * 50}\n"
            f"✓ API Endpoint: Working\n"
            f"✓ Model: {model_name} Available\n"
            f"✓ Response: Non-empty and valid\n"
            f"✓ Configuration saved to memory (session only)\n"
        )
        from custom_mas_ui import set_api_config
        set_api_config(base_url, api_key, model_name)

        return conversation_log

    except Exception as e:
        # ── Exception handling ──
        conversation_log = (
            f"📤 **Sending Request**\n"
            f"```\n"  
            f"Base URL: {base_url}\n"  
            f"Model: {model_name}\n"  
            f"Message: hello!\n"  
            f"```\n\n"
            f"❌ **Connection Failed**\n"
            f"```\n"  
            f"Error: {type(e).__name__}\n"  
            f"Details: {str(e)}\n"  
            f"```\n"
        )
        return conversation_log


# ---------------------------------------------------------------------------
# 3. Method invocation helpers
# ---------------------------------------------------------------------------
#  method, full_q, root, base_model, img, api_key=base_key or "EMPTY", base_url=base_url
def _call_api_method(method: str, question: str, root_path: str,
                     model_name: str, img_path, api_key, base_url):
    """Call an API-based method and return (answer, token_stats, config)."""
    if method == "ColaCare":
        from methods.ColaCare import colacare_infer
        return colacare_infer(question, root_path, model_name, img_path,api_key, base_url)

    if method == "MetaPrompting":
        from methods.MetaPrompting import metaprompting_infer
        return metaprompting_infer(question, root_path, model_name, img_path,api_key, base_url)


    raise ValueError(f"Unknown API method: {method}")


def _call_batch_method(method: str, question: str, root_path: str,
                       model_name: str, img_path, batch_mgr):
    """Call a batch-manager method and return (answer, token_stats, config)."""
    if method == "Debate":
        from methods.debate import Debate_test
        return Debate_test(question, root_path, model_name,
                           img_path, batch_mgr)
    if method == "Reconcile":
        from methods.Reconcile.reconcile_test import Reconcile_test
        return Reconcile_test(question, root_path, model_name,
                              img_path, batch_mgr)
    if method == "Discussion":
        from methods.Discussion import discussion_infer
        return discussion_infer(question, root_path, model_name,
                                    img_path, batch_mgr)
    if method == "MDAgents":
        from methods.MDAgents.medagents import MDAgents_test
        return MDAgents_test(question, root_path, model_name, img_path,batch_mgr)
    if method == "MDTeamGPT":
        from methods.MDTeamGPT.MDTeamGPT_test import MDTeamGPT_test
        return MDTeamGPT_test(question, img_path, root_path, batch_mgr)

    if method == "autogen":
        from methods.autogen import autogen_infer_medqa
        return autogen_infer_medqa(question, root_path, model_name,
                                   img_paths=img_path,batch_manager=batch_mgr)

    if method == "dylan":
        from methods.dylan import dylan_infer_medqa
        return dylan_infer_medqa(
            question, root_path, model_name, img_paths=img_path,batch_manager=batch_mgr
        )

    if method == "MedAgents":
        from methods.MedAgents import medagents_infer
        return medagents_infer(
            question, root_path, model_name, img_path,batch_manager=batch_mgr
        )

    raise ValueError(f"Unknown batch method: {method}")


def _make_batch_manager(base_url: str, model_name: str, api_key: str = None):
    """
    Create batch manager with configuration parameters.
    Pass api_key to VLLMBatchInferenceManager.
    """
    from methods.vllm_thread import VLLMBatchInferenceManager
    mgr = VLLMBatchInferenceManager(
        model=model_name,
        root_path=str(PROJECT_ROOT),
        batch_size=4,
        timeout=0.5,
        vllm_url=base_url,
        api_key=api_key or "EMPTY",  # ✨ Pass API key
    )
    mgr.start()
    return mgr


def _load_custom_method(filepath: str):
    """Dynamically import a user-provided .py file and return its `infer` fn."""
    spec = importlib.util.spec_from_file_location("_user_custom_method", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "infer", None)
    if fn is None:
        raise ValueError(
            "Custom method file must define a function named `infer`.\n"
            "Signature: infer(question, client, model_name, **kwargs) -> str"
        )
    return fn


# ---------------------------------------------------------------------------
# 4. Single question inference
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════
# 修改 run_single_question（删除Custom分支）
# ═══════════════════════════════════════════
def run_single_question(
        method, question, options_text, image_path,
        progress=gr.Progress(),
):
    """Run one question through the selected method (built-in only)."""

    if not question or not question.strip():
        return "⚠️ Please enter a question.", "", ""


    config = _get_global_config()
    base_url = config["base_url"]
    base_key = config["base_key"]
    base_model = config["base_model"]

    if not base_url or not base_model:
        return (
            "⚠️ Please test connection first on the Setup tab.",
            "",
            ""
        )

    progress(0, desc="Preparing …")

    full_q = question.strip()
    if options_text and options_text.strip():
        full_q += " Options: " + options_text.strip()

    img = [image_path] if image_path else None
    root = str(PROJECT_ROOT)

    progress(0.15, desc=f"Running {method} …")
    t0 = time.time()
    batch_mgr = None

    try:
        # ✨ 注意：删除了 method == "Custom" 的分支
        if method in API_METHODS:
            answer, token_stats, config_result = _call_api_method(
                method, full_q, root, base_model, img, api_key=base_key or "EMPTY", base_url=base_url)
        elif method in BATCH_METHODS:
            batch_mgr = _make_batch_manager(base_url, base_model, api_key=base_key,)
            answer, token_stats, config_result = _call_batch_method(
                method, full_q, root, base_model, img, batch_mgr)
        else:
            return f"⚠️ Unknown method: {method}", "", ""

        elapsed = time.time() - t0
        progress(0.9, desc="Formatting results …")

        total_calls = total_pt = total_ct = 0
        if token_stats:
            for v in token_stats.values():
                total_calls += int(v.get("num_llm_calls", 0))
                total_pt += int(v.get("prompt_tokens", 0))
                total_ct += int(v.get("completion_tokens", 0))

        if not isinstance(config_result, dict):
            config_result = {}
        stats = (
            f"⏱ Time: {elapsed:.2f}s\n"
            f"📞 LLM/VLM Calls: {total_calls}\n"
            f"📝 Prompt Tokens: {total_pt:,}\n"
            f"💬 Completion Tokens: {total_ct:,}\n"
            f"👥 Agents: {config_result.get('current_num_agents', '-')}\n"
            f"🔄 Rounds: {config_result.get('round', '-')}"
        )
        progress(1.0, desc="Done!")
        return str(answer), stats, ""

    except Exception as exc:
        tb = traceback.format_exc()
        return f"❌ {exc}", "", tb
    finally:
        if batch_mgr is not None:
            try:
                batch_mgr.stop()
            except Exception:
                pass


# ═══════════════════════════════════════════
# 新增：运行 Custom MAS 的函数
# ═══════════════════════════════════════════
def run_custom_mas(
        question,
        options_text,
        image_path,
        progress=gr.Progress(),
):
    """Run the generated custom MAS test_sample function."""

    if not question or not question.strip():
        return "⚠️ Please enter a question.", "", ""
    try:
        from gradio_tmp.custom_mas import test_sample
    except ImportError:
        test_sample = None
    # 检查是否导入了 test_sample
    if test_sample is None:
        return (
            "❌ Custom MAS not found. Please generate it first in the Custom MAS tab.",
            "",
            "Error: test_sample module not imported"
        )

    progress(0, desc="Preparing …")

    # 组合完整问题
    full_q = question.strip()
    if options_text and options_text.strip():
        full_q += " Options: " + options_text.strip()

    progress(0.2, desc="Checking API configuration …")

    config = _get_global_config()
    base_url = config["base_url"]
    base_key = config["base_key"]
    base_model = config["base_model"]

    if not base_url or not base_model:
        return (
            "⚠️ Please test connection first on the Setup tab.",
            "",
            ""
        )

    progress(0.4, desc="Running Custom MAS …")

    t0 = time.time()

    try:
        # 调用生成的 test_sample 函数
        # test_sample 函数签名：test_sample(question: str) -> str
        answer = test_sample(full_q,image_path=image_path)
        answer = str(answer)

        elapsed = time.time() - t0

        progress(0.9, desc="Formatting results …")

        stats = (
            f"⏱ Time: {elapsed:.2f}s\n"
            f"🤖 Model: {base_model}\n"
            f"📍 Status: ✅ Success"
        )

        progress(1.0, desc="Done!")

        return answer, stats, ""

    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        return (
            f"❌ Error running custom MAS: {exc}",
            f"⏱ Time: {elapsed:.2f}s\n🔴 Status: Failed",
            tb
        )

# ═══════════════════════════════════════════
# 新增 run_custom_method 函数
# ═══════════════════════════════════════════
def run_custom_method(
        custom_file,
        question,
        options_text,
        image_path,
        progress=gr.Progress(),
):
    """Run custom method - standalone step between Quick Test and Batch Evaluation."""

    if not custom_file:
        return "⚠️ Please upload a custom method .py file.", "", ""

    if not question or not question.strip():
        return "⚠️ Please enter a question.", "", ""

    config = _get_global_config()
    base_url = config["base_url"]
    base_key = config["base_key"]
    base_model = config["base_model"]

    if not base_url or not base_model:
        return (
            "⚠️ Please test connection first on the Setup tab.",
            "",
            ""
        )

    progress(0, desc="Loading custom method …")

    full_q = question.strip()
    if options_text and options_text.strip():
        full_q += " Options: " + options_text.strip()

    t0 = time.time()

    try:
        # Load custom method
        fn = _load_custom_method(custom_file)
        client = OpenAI(api_key=base_key or "EMPTY", base_url=base_url)

        progress(0.5, desc="Executing custom method …")
        raw = fn(full_q, client, base_model)
        answer = str(raw)

        elapsed = time.time() - t0

        progress(0.9, desc="Formatting results …")

        stats = (
            f"⏱ Execution Time: {elapsed:.2f}s\n"
            f"🤖 Model: {base_model}\n"
            f"📝 Question Length: {len(full_q)} chars\n"
            f"✅ Status: Completed"
        )

        progress(1.0, desc="Done!")
        return str(answer), stats, ""

    except Exception as exc:
        tb = traceback.format_exc()
        return f"❌ Error: {exc}", "", tb


# ---------------------------------------------------------------------------
# 5. Batch evaluation
# ---------------------------------------------------------------------------

def run_batch(
    method, dataset_name, uploaded_file,
    num_samples, num_workers,
    # ma_nqd, ma_nod, ma_mr,
    # dy_na, dy_nr,
    custom_file,
    progress=gr.Progress(),
):
    """Run batch evaluation on a dataset."""

    # ✨ Get config from global variables ✨
    config = _get_global_config()
    base_url = config["base_url"]
    base_key = config["base_key"]
    base_model = config["base_model"]

    if not base_url or not base_model:
        return (
            "⚠️ Configure the Base Model API first by testing connection on Setup tab.",
            [],
            ""
        )

    progress(0.02, desc="Updating configuration …")

    # ── load samples ──
    progress(0.05, desc="Loading dataset …")
    if uploaded_file is not None:
        samples = []
        with open(uploaded_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        ds_tag = "custom"
    else:
        from dataset_utils import load_test_split
        ds_path = str(PROJECT_ROOT / "data" / dataset_name)
        samples = load_test_split(ds_path, dataset_name)
        ds_tag = dataset_name

    n = min(int(num_samples or 10), len(samples))
    samples = samples[:n]
    if not samples:
        return "⚠️ No samples found.", [], ""

    root = str(PROJECT_ROOT)
    batch_mgr = None
    if method in BATCH_METHODS:
        # ✨ Pass config parameters to batch manager ✨
        batch_mgr = _make_batch_manager(base_url, base_model, base_key)

    results: List[dict] = []
    correct = 0
    total_time = 0.0
    custom_fn = None
    if method == "Custom":
        if not custom_file:
            return "⚠️ Upload a custom method file.", [], ""
        custom_fn = _load_custom_method(custom_file)

    try:
        for idx, sample in enumerate(samples):
            progress((idx + 1) / n, desc=f"[{idx+1}/{n}] Processing …")

            # ── format question ──
            try:
                from dataset_utils import format_question as _fq
                fmt = _fq(sample, ds_tag if ds_tag != "custom" else "medqa")
                q_text = fmt[0] if isinstance(fmt, tuple) else str(fmt)
                img_raw = fmt[1] if isinstance(fmt, tuple) and len(fmt) > 1 else None
            except Exception:
                q_text = sample.get("question", "")
                opts = sample.get("options", "")
                if isinstance(opts, list):
                    opts = " ".join(opts)
                elif isinstance(opts, dict):
                    opts = " ".join(f"({k}) {v}" for k, v in opts.items())
                if opts:
                    q_text += " Options: " + opts
                img_raw = None

            img = None
            if img_raw and isinstance(img_raw, str):
                img = [PROJECT_ROOT / "data" / ds_tag / "imgs" / img_raw]
            elif img_raw and isinstance(img_raw, list):
                img = [PROJECT_ROOT / "data" / ds_tag / "imgs" / p for p in img_raw]

            t0 = time.time()
            try:
                if method == "Custom":
                    client = OpenAI(api_key=base_key or "EMPTY", base_url=base_url)
                    ans = str(custom_fn(q_text, client, base_model))
                    tok = {}
                    cfg = {}
                elif method in API_METHODS:
                    ans, tok, cfg = _call_api_method(
                        method, q_text, root, base_model, img,base_key, base_url)
                else:
                    ans, tok, cfg = _call_batch_method(
                        method, q_text, root, base_model, img, batch_mgr)
            except Exception as e:
                ans = f"Error: {e}"
                tok, cfg = {}, {}

            elapsed = time.time() - t0
            total_time += elapsed

            gt = str(sample.get("answer_idx", "")).strip().lower()
            is_correct = False
            if gt:
                fd = str(ans)
                m = re.search(r"\(([A-Ea-e])\)", fd)
                ext = m.group(1).lower() if m else None
                if ext is None:
                    m2 = re.match(r"^\s*([A-Ea-e])(?:\s|[^a-zA-Z])", fd)
                    ext = m2.group(1).lower() if m2 else None
                if ext and ext == gt:
                    is_correct = True

            if is_correct:
                correct += 1

            results.append({
                "id": sample.get("id", idx),
                "final_decision": str(ans),
                "answer": sample.get("answer", ""),
                "right_option": sample.get("answer_idx", ""),
                "is_correct": is_correct,
                "time_cost": round(elapsed, 2),
            })

    finally:
        if batch_mgr:
            try:
                batch_mgr.stop()
            except Exception:
                pass

    # ── save & summarize ──
    acc = correct / len(results) * 100 if results else 0
    avg_t = total_time / len(results) if results else 0

    out_dir = PROJECT_ROOT / "output_web_demo"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{method}_{base_model}_{ds_tag}_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = (
        f"📊  Evaluation Complete\n"
        f"{'═' * 44}\n"
        f"  Method:       {method}\n"
        f"  Dataset:      {ds_tag}\n"
        f"  Base Model:   {base_model}\n"
        f"{'─' * 44}\n"
        f"  Samples:      {len(results)}\n"
        f"  Correct:      {correct}\n"
        f"  Accuracy:     {acc:.2f}%\n"
        f"  Avg Time:     {avg_t:.2f}s / sample\n"
        f"{'═' * 44}\n"
        f"  Results saved: {out_path.name}\n"
    )

    rows = []
    for r in results[:100]:
        rows.append([
            str(r["id"]),
            str(r["final_decision"])[:120],
            str(r.get("right_option", "")),
            "✅" if r["is_correct"] else "❌",
            f"{r['time_cost']:.1f}s",
        ])

    return summary, rows, str(out_path)


# ---------------------------------------------------------------------------
# 6. Build the Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.header-banner {
    text-align: center;
    padding: 28px 20px 18px;
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #7c3aed 100%);
    border-radius: 14px;
    margin-bottom: 16px;
}
.header-banner h1 { color: #fff !important; font-size: 2.2em !important; margin: 0 0 6px !important; }
.header-banner p  { color: rgba(255,255,255,.88) !important; margin: 2px 0 !important; }
.method-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.method-card {
    border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;
    background: #f8fafc; transition: box-shadow .2s;
}
.method-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.08); }
.method-card h4 { margin: 0 0 4px; }
.method-card p  { margin: 0; font-size: .92em; color: #475569; }
footer { display: none !important; }
"""


def _build_method_cards_html():
    cards = []
    for name, info in METHOD_INFO.items():
        cards.append(
            f'<div class="method-card">'
            f'<h4>{info["icon"]} {name}</h4>'
            f'<p>{info["desc"]}</p>'
            f'<p style="margin-top:6px;font-size:.85em;color:#94a3b8;">'
            f'Agents: {info["agents"]}</p></div>'
        )
    return '<div class="method-grid">' + "".join(cards) + "</div>"


def create_app():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="violet",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="MedMASLab", theme=theme, css=CUSTOM_CSS) as app:

        # ── Header ──
        gr.HTML(
            '<div class="header-banner">'
            "<h1>🏥 MedMASLab</h1>"
            "<p>Multi-Agent System Evaluation Framework for Medical Benchmarks</p>"
            '<p style="font-size:.9em;opacity:.75;">Setup your API → Pick a method → Ask medical questions</p>'
            "</div>"
        )

        with gr.Tabs():
            # ═══════════════════════════════════════════
            #  TAB 1 – API Setup
            # ═══════════════════════════════════════════
            with gr.TabItem("⚙️  API Setup", id="tab_setup"):
                gr.Markdown(
                    "### Connect to your LLM/VLM endpoint\n"
                    "Any **OpenAI-compatible** API works: OpenAI, vLLM, Ollama, LM Studio, etc."
                )
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("#### 🤖 Base Model *(required)*")
                        inp_base_url = gr.Textbox(
                            label="API Base URL",
                            placeholder="http://localhost:8000/v1",
                            info="The /v1 endpoint of your LLM/VLM service",
                        )
                        inp_base_key = gr.Textbox(
                            label="API Key",
                            value="EMPTY",
                            type="password",
                            info="Use EMPTY for local endpoints",
                        )
                        inp_base_model = gr.Textbox(
                            label="Model Name",
                            placeholder="Qwen2.5-VL-7B-Instruct",
                            info="Must match a model served at the URL above",
                        )
                    with gr.Column():
                        gr.Markdown("#### 📝 Connection Flow")
                        gr.Markdown(
                            "1. **Enter** your LLM/VLM API details\n"
                            "2. **Click** 'Test Connection'\n"
                            "3. **View** the full conversation log below\n"
                            "4. **If successful**, config is saved to memory\n"
                            "5. **Go to** Quick Test tab to start asking questions"
                        )

                with gr.Row():
                    btn_test = gr.Button("🔗 Test Connection", variant="primary", size="lg")

                # ✨ Display conversation log ✨
                status_box = gr.Markdown("", label="Connection Test Result")

                btn_test.click(
                    fn=test_connection,
                    inputs=[inp_base_url, inp_base_key, inp_base_model],
                    outputs=[status_box],
                )

            # ═══════════════════════════════════════════
            #  TAB 2 – Quick Test (Single Question)
            # ═══════════════════════════════════════════
            with gr.TabItem("🔬 Quick Test", id="tab_single"):
                with gr.Row():
                    # ── Left: method selection ──
                    with gr.Column(scale=2):
                        gr.Markdown("### Pick a Method")
                        method_dd = gr.Dropdown(
                            choices=BUILTIN_METHODS,  # ✨ 删除了 + ["Custom"]
                            value="ColaCare",
                            label="Method",
                        )
                        method_desc_md = gr.Markdown(
                            f'*{METHOD_INFO["ColaCare"]["icon"]} '
                            f'{METHOD_INFO["ColaCare"]["desc"]}*'
                        )

                    # ── Right: question input ──
                    with gr.Column(scale=3):
                        gr.Markdown("### Enter Your Question")
                        q_input = gr.Textbox(
                            label="Medical Question",
                            placeholder=(
                                "e.g., A 45-year-old male presents with "
                                "acute chest pain radiating to the left arm …"
                            ),
                            lines=5,
                        )
                        opt_input = gr.Textbox(
                            label="Options (optional)",
                            placeholder="(A) … (B) … (C) … (D) …",
                            lines=2,
                        )
                        img_input = gr.Image(
                            label="Medical Image (optional)",
                            type="filepath",
                        )

                # examples
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=[q_input, opt_input],
                    label="💡 Example Questions (click to fill)",
                )

                btn_run = gr.Button("▶ Run Inference", variant="primary", size="lg")

                gr.Markdown("---\n### Results")
                with gr.Row():
                    out_answer = gr.Textbox(
                        label="🎯 Final Answer",
                        lines=4,
                        interactive=False
                    )
                    out_stats = gr.Textbox(
                        label="📊 Statistics",
                        lines=6,
                        interactive=False
                    )
                out_error = gr.Textbox(
                    label="Error Traceback (if any)",
                    lines=4,
                    interactive=False,
                    visible=True
                )

                def _on_method_change(m):
                    info = METHOD_INFO.get(m)
                    desc = (f'*{info["icon"]} {info["desc"]}*'
                            if info else "*Select a method*")
                    return desc

                method_dd.change(
                    _on_method_change,
                    inputs=[method_dd],
                    outputs=[method_desc_md],
                )

                btn_run.click(
                    fn=run_single_question,
                    inputs=[
                        method_dd, q_input, opt_input, img_input,
                    ],
                    outputs=[out_answer, out_stats, out_error],
                )

            # ═══════════════════════════════════════════
            #  TAB 3 – Custom MAS (New Step)
            # ═══════════════════════════════════════════
            with gr.TabItem("🎨 Custom MAS", id="tab_custom_mas"):
                gr.Markdown("""
                ### 🎨 Design Your Custom Multi-Agent System
                Create a medical multi-agent system visually, then generate and test it.
                """)

                # Import and create the designer UI
                from custom_mas_ui import create_custom_mas_ui

                # 创建可视化设计界面
                create_custom_mas_ui()

            # ═══════════════════════════════════════════
            #  TAB 4 – Test Custom MAS (NEW)
            # ═══════════════════════════════════════════
            with gr.TabItem("🧪 Test Custom MAS", id="tab_test_custom_mas"):
                gr.Markdown("""
                        ### 🧪 Test Your Generated Custom MAS

                        Test the multi-agent system you just designed and generated.
                        """)

                # Input section
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 📋 Enter Your Question")
                        test_q_input = gr.Textbox(
                            label="Medical Question",
                            placeholder="e.g., A 45-year-old male presents with acute chest pain...",
                            lines=5,
                        )
                        test_opt_input = gr.Textbox(
                            label="Options (optional)",
                            placeholder="(A) … (B) … (C) … (D) …",
                            lines=2,
                        )
                        test_img_input = gr.Image(
                            label="Medical Image (optional)",
                            type="filepath",
                        )

                # Examples section
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=[test_q_input, test_opt_input],
                    label="💡 Example Questions (click to fill)",
                )

                # Run button
                test_btn_run = gr.Button("▶ Run Custom MAS", variant="primary", size="lg")

                # Results section
                gr.Markdown("---\n### 📊 Results")
                with gr.Row():
                    test_out_answer = gr.Textbox(
                        label="🎯 Final Answer",
                        lines=6,
                        interactive=False
                    )
                    test_out_stats = gr.Textbox(
                        label="📈 Statistics",
                        lines=6,
                        interactive=False
                    )

                test_out_error = gr.Textbox(
                    label="⚠️ Error Traceback (if any)",
                    lines=4,
                    interactive=False,
                    visible=True
                )

                # ✨ 在按钮定义之后添加事件处理
                test_btn_run.click(
                    fn=run_custom_mas,
                    inputs=[
                        test_q_input,
                        test_opt_input,
                        test_img_input,
                    ],
                    outputs=[test_out_answer, test_out_stats, test_out_error],
                )

            # ═══════════════════════════════════════════
            #  TAB 3 – Batch Evaluation
            # ═══════════════════════════════════════════
            with gr.TabItem("📊  Batch Evaluation", id="tab_batch"):
                gr.Markdown("### Evaluate on a full dataset")
                with gr.Row():
                    with gr.Column():
                        b_method = gr.Dropdown(
                            choices=BUILTIN_METHODS + ["Custom"],
                            value="ColaCare",
                            label="Method",
                        )
                        b_dataset = gr.Dropdown(
                            choices=AVAILABLE_DATASETS,
                            value="pubmedqa",
                            label="Dataset (built-in)",
                        )
                        b_file = gr.File(
                            label="Or upload JSONL",
                            file_types=[".jsonl", ".json"],
                            type="filepath",
                        )
                    with gr.Column():
                        b_nsamples = gr.Number(label="Samples", value=10,
                                               minimum=1, precision=0)
                        b_nworkers = gr.Number(label="Concurrent Workers",
                                               value=4, minimum=1, precision=0)
                        b_custom = gr.File(
                            label="Custom Method (.py)",
                            file_types=[".py"],
                            type="filepath",
                            visible=False,
                        )

                b_run = gr.Button("▶  Start Evaluation", variant="primary",
                                  size="lg")

                b_summary = gr.Textbox(label="Summary", lines=12,
                                       interactive=False)
                b_table = gr.Dataframe(
                    headers=["ID", "Answer", "Correct", "Result", "Time"],
                    label="Detailed Results",
                    interactive=False,
                )
                b_path = gr.Textbox(label="Output file", interactive=False)

                def _toggle_b_custom(m):
                    return gr.update(visible=(m == "Custom"))

                b_method.change(_toggle_b_custom, [b_method], [b_custom])

                b_run.click(
                    fn=run_batch,
                    inputs=[
                        b_method, b_dataset, b_file,
                        b_nsamples, b_nworkers,
                        # bma_nqd, bma_nod, bma_mr,
                        # bdy_na, bdy_nr,
                        b_custom,
                    ],
                    outputs=[b_summary, b_table, b_path],
                )

            # ═══════════════════════════════════════════
            #  TAB 4 – Guide
            # ═══════════════════════════════════════════
            with gr.TabItem("📖  Guide", id="tab_guide"):
                gr.Markdown("## Supported Multi-Agent Methods")
                gr.HTML(_build_method_cards_html())

                gr.Markdown(
                    """

## 🚀 Quick Start Guide

### **Complete Workflow (5 Steps)**

| Step | Tab | Action | Description |
|:----:|:---:|--------|-------------|
| **1** | **⚙️ API Setup** | Enter your LLM/VLM API details | - **API Base URL**: `http://localhost:8000/v1` (or your endpoint)- **API Key**: `EMPTY` (for local) or your key- **Model Name**: `Qwen2.5-VL-7B-Instruct` (or your model) |
| **2** | **⚙️ API Setup** | Click **🔗 Test Connection** | Verifies API is working and saves config to memory |
| **3** | **🔬 Quick Test** | Select built-in method & ask question | 1. Pick a method (ColaCare, MedPAL, etc.). Enter medical question. (Optional) Add options or image. Click **▶ Run Inference**. View results, statistics, and token usage |
| **4** | **🎨 Custom MAS** | Design your multi-agent system | 1. Add agents from palette. Adjust positions with sliders. Connect agents (define data flow). View & modify prompts. Click **🚀 Generate System Snapshot**. LLM generates code automatically ✨ Code parser extracts & saves `custom_mas.py` |
| **5** | **🧪 Test Custom MAS** | Test your generated system | 1. Enter medical question. (Optional) Add options or image. Click **▶ Run Custom MAS**. View final answer & statistics |

---

""")
    return app

from openai import OpenAI

def infer(question: str, client: OpenAI, model_name: str, **kwargs) -> str:
    """
    Args:
        question   – The full question string (with options appended)
        client     – An OpenAI client connected to the configured API
        model_name – The model name from the API Setup tab

    Returns:
        The final answer as a plain string.
    """
    # Example: single-call baseline
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        temperature=0.1,
    )  # ✨ 这里的右括号很重要！
    return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="MedMASLab Web Demo")
    parser.add_argument("--port", type=int, default=7890)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
    help="Create a public Gradio link")
    cli = parser.parse_args()

    app = create_app()
    print(f"\n{'═' * 56}")
    print(f" 🏥 MedMASLab Web Demo")
    print(f" → http://{cli.host}:{cli.port}")
    print(f"{'═' * 56}\n")
    app.queue()
    app.launch(
    server_name=cli.host,
    server_port=cli.port,
    share=cli.share,
    show_error=True,
    )

if __name__ == "__main__":  # ✅ 正确：__name__ 是特殊变量
    main()