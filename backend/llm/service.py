from __future__ import annotations

import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

from backend.bootstrap import ensure_python_paths
from backend.llm.model_routing import resolve_model_identity
from backend.settings import (
    get_hf_local_files_only,
    get_hf_load_in_4bit,
    get_hf_load_in_8bit,
    get_hf_max_memory,
    get_hf_model_path,
    get_history_turns,
    get_llm_backend,
    get_lora_adapter_path,
    get_ollama_base_url,
    get_ollama_model,
    get_ollama_timeout,
    resolve_model_option,
)

ensure_python_paths()


def _load_tokenizer(model_path: str) -> Any:
    """加载 HuggingFace tokenizer；只有走 HF 本地模型备用路径时才会用到。"""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=get_hf_local_files_only(),
            trust_remote_code=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=get_hf_local_files_only(),
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _system_prompt() -> str:
    """构造系统提示词，统一约束 Agent/Qwen 的回答风格和研究方向。"""
    return (
        "你是一个本地知识库 AI 助手，服务于轨迹分析、轨迹压缩、强化学习和实验设计。"
        "你的核心任务是稳定连续对话、结合本地知识库证据、自然解释问题，并在证据不足时明确说明不确定性。"
        "优先承接用户上文中的“这个”“继续”“刚才那个”等追问，不要把每轮都当成孤立问题。"
        "如果给了检索证据，请优先基于证据回答；如果证据和问题不匹配，要说明证据不足，不要编造。"
        "回答应清楚、具体、像研究助手一样能解释取舍；只有在确实有帮助时才使用表格或分点。"
        "遇到强化学习状态转移链路时，请使用 LaTeX 公式，例如 \\[s_t \\xrightarrow{\\;\\pi_\\theta\\;} a_t \\xrightarrow{\\;\\mathcal{E}\\;} (r_{t+1}, s_{t+1})\\]，不要写成普通箭头文本或代码块。"
    )


def build_messages(query: str, history: list[dict[str, str]], context_block: str) -> list[dict[str, str]]:
    """把系统提示词、压缩上下文、历史消息和当前问题组装成模型输入。"""
    system = _system_prompt()
    if context_block.strip():
        system += f"\n\n以下是可用的本地证据，请优先参考：\n{context_block}"
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-get_history_turns():])
    messages.append({"role": "user", "content": query})
    return messages


# 核心1：Ollama 健康检查。前端状态面板判断 Qwen/Ollama 是否可用时会走这里。
def ollama_available() -> bool:
    request = urllib.request.Request(f"{get_ollama_base_url()}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5):
            return True
    except Exception:
        return False


# 核心1：Ollama/Qwen 的底层请求函数。所有走 Ollama 的回答最终都会 POST 到 `/api/chat`。
def _chat_ollama(messages: list[dict[str, str]], model_name: str | None = None) -> str:
    body = {
        "model": model_name or get_ollama_model(),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }
    request = urllib.request.Request(
        f"{get_ollama_base_url()}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=get_ollama_timeout()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"OLLAMA_UNREACHABLE: {error}") from error
    except TimeoutError as error:
        raise RuntimeError(f"OLLAMA_TIMEOUT: generation exceeded {get_ollama_timeout()} seconds") from error
    return (payload.get("message") or {}).get("content", "").strip()


@lru_cache(maxsize=4)
def _load_hf_runtime(model_path_override: str = "") -> tuple[Any, Any, str]:
    """加载 HuggingFace 本地模型运行时，作为 Ollama 不可用或显式选择 HF 时的备用路径。"""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_path = model_path_override or get_hf_model_path()
    if not model_path:
        raise RuntimeError("LOCAL_LLM_MODEL_PATH is not set.")

    tokenizer = _load_tokenizer(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": get_hf_local_files_only(),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if get_hf_load_in_4bit():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif get_hf_load_in_8bit():
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if get_hf_max_memory():
        kwargs["max_memory"] = {"cuda:0": get_hf_max_memory()}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    adapter_path = get_lora_adapter_path()
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=get_hf_local_files_only(),
        )
    if device != "cuda":
        model.to(device)
    model.eval()
    return tokenizer, model, device


def _chat_hf(messages: list[dict[str, str]], model_path: str | None = None) -> str:
    """使用 HuggingFace 本地模型生成回答，接口形状与 Ollama 路径保持一致。"""
    import torch

    tokenizer, model, device = _load_hf_runtime(model_path or "")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=320,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def _fallback_context_reply(query: str, context_block: str) -> str:
    """当模型不可用时生成降级提示，避免接口直接返回空内容。"""
    if not context_block.strip():
        return (
            "这次没有成功拿到模型生成结果。常见原因是本地模型服务未启动、生成超时，"
            "或者模型正在被其他任务占用。你可以稍后重试；如果反复出现，请检查 Ollama/HF 模型服务状态。"
        )
    return (
        "这次模型没有顺利完成生成，所以我先把已经检索到的相关内容整理出来。\n\n"
        f"你的问题：{query}\n\n"
        f"可用证据摘录：\n{context_block[:1200]}\n\n"
        "你可以直接重试一次，也可以继续让我基于这些证据做结构化整理。"
    )


def llm_status() -> dict[str, Any]:
    """返回当前 LLM 后端状态，供前端健康检查和模型选择面板展示。"""
    backend = get_llm_backend()
    if backend == "hf":
        model_path = get_hf_model_path()
        identity = resolve_model_identity(model_path or "hf-local-model", backend=backend)
        return {
            "backend": "hf",
            "ready": bool(model_path),
            "detail": f"{model_path} + LoRA {get_lora_adapter_path()}" if get_lora_adapter_path() else (model_path or "LOCAL_LLM_MODEL_PATH is not set."),
            "canonical_model": identity.canonical_model,
            "provider_kind": identity.provider_kind,
        }
    ready = ollama_available()
    identity = resolve_model_identity(get_ollama_model(), backend=backend)
    return {
        "backend": "ollama",
        "ready": ready,
        "detail": get_ollama_model() if ready else f"Ollama is not reachable at {get_ollama_base_url()}",
        "canonical_model": identity.canonical_model,
        "provider_kind": identity.provider_kind,
    }


def chat_reply(query: str, history: list[dict[str, str]], context_block: str, model_id: str | None = None) -> str:
    """普通聊天入口：先组装消息，再交给 run_messages 统一派发。"""
    messages = build_messages(query, history, context_block)
    return run_messages(messages, query=query, context_block=context_block, model_id=model_id)


# 核心1：统一模型派发入口。Agent、RAG、上下文压缩和普通聊天都会先进入这里，再转到 Ollama/Qwen 或 HF。
def run_messages(messages: list[dict[str, str]], query: str = "", context_block: str = "", model_id: str | None = None) -> str:
    option = resolve_model_option(model_id)
    backend = str(option.get("backend") or get_llm_backend())
    model_value = str(option.get("model") or "")
    if backend == "hf":
        return _chat_hf(messages, model_value)

    try:
        return _chat_ollama(messages, model_value or None)
    except RuntimeError as error:
        if get_hf_model_path():
            return _chat_hf(messages)
        message = str(error)
        if message.startswith("OLLAMA_UNREACHABLE"):
            return (
                "我当前没有连上 Ollama 服务。"
                f"请确认 {get_ollama_base_url()} 可以访问，并且模型 {model_value or get_ollama_model()} 已可用。"
            )
        if message.startswith("OLLAMA_TIMEOUT"):
            return _fallback_context_reply(query, context_block)
        return _fallback_context_reply(query, context_block)
