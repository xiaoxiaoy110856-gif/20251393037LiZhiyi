from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()


DEFAULT_SYSTEM_PROMPT = (
    "You are a local chat assistant focused on reinforcement learning research. "
    "Answer naturally, explain clearly, and avoid inventing evidence."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_example(example: dict[str, Any], system_prompt: str) -> tuple[list[dict[str, str]], str, str]:
    if "messages" in example:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": item["role"], "content": item["content"]} for item in example["messages"])
        reference = str(example.get("reference") or example.get("output") or example.get("answer") or "")
        prompt_text = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
        return messages, prompt_text, reference

    instruction = str(example.get("instruction") or example.get("query") or "")
    input_text = str(example.get("input") or example.get("context") or "")
    reference = str(example.get("output") or example.get("answer") or example.get("reference") or "")
    user_text = instruction if not input_text else f"{instruction}\n\nContext:\n{input_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return messages, user_text, reference


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def token_set(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    parts = re.findall(r"[\u4e00-\u9fff]{1,2}|[a-z0-9_]+", normalized)
    return {part for part in parts if part}


def char_f1(prediction: str, reference: str) -> float:
    pred_chars = list(normalize_text(prediction))
    ref_chars = list(normalize_text(reference))
    if not pred_chars and not ref_chars:
        return 1.0
    if not pred_chars or not ref_chars:
        return 0.0
    ref_pool = ref_chars.copy()
    overlap = 0
    for char in pred_chars:
        if char in ref_pool:
            ref_pool.remove(char)
            overlap += 1
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def overlap_score(prediction: str, reference: str) -> float:
    pred_tokens = token_set(prediction)
    ref_tokens = token_set(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def build_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_path: str, adapter_path: str, load_in_4bit: bool, load_in_8bit: bool, cpu: bool, max_memory: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    use_cuda = torch.cuda.is_available() and not cpu
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16 if use_cuda else torch.float32,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if use_cuda:
        kwargs["device_map"] = "auto"
    if max_memory:
        kwargs["max_memory"] = {"cuda:0": max_memory}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return model


def generate_answer(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(target_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=1.08,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local chat model without touching the training workflow.")
    parser.add_argument("--model", required=True, help="HF model directory.")
    parser.add_argument("--data", type=Path, required=True, help="Evaluation dataset in JSONL format.")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N samples when > 0.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-memory", default="")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    examples = read_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No evaluation samples found in {args.data}")

    tokenizer = build_tokenizer(args.model)
    model = build_model(args.model, args.adapter, args.load_in_4bit, args.load_in_8bit, args.cpu, args.max_memory)

    details: list[dict[str, Any]] = []
    em_total = 0.0
    overlap_total = 0.0
    char_f1_total = 0.0

    for index, example in enumerate(examples, start=1):
        messages, prompt_text, reference = normalize_example(example, args.system_prompt)
        prediction = generate_answer(model, tokenizer, messages, args.max_new_tokens, args.temperature, args.top_p)
        em = exact_match(prediction, reference) if reference else 0.0
        overlap = overlap_score(prediction, reference) if reference else 0.0
        f1 = char_f1(prediction, reference) if reference else 0.0
        em_total += em
        overlap_total += overlap
        char_f1_total += f1
        details.append(
            {
                "index": index,
                "prompt": prompt_text,
                "reference": reference,
                "prediction": prediction,
                "exact_match": round(em, 4),
                "token_overlap_recall": round(overlap, 4),
                "char_f1": round(f1, 4),
            }
        )

    count = len(details)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "data": str(args.data),
        "count": count,
        "metrics": {
            "exact_match": round(em_total / count, 4),
            "token_overlap_recall": round(overlap_total / count, 4),
            "char_f1": round(char_f1_total / count, 4),
        },
    }
    payload = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
