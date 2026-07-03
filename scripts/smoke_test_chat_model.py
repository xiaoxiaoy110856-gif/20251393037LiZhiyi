from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description="快速测试 HF 聊天模型或基座模型 + LoRA adapter 是否能生成回答。")
    parser.add_argument("--model", required=True, help="HF 模型目录。可以是基座模型，也可以是合并后的模型。")
    parser.add_argument("--adapter", default="", help="可选 LoRA adapter 目录。")
    parser.add_argument("--prompt", default="请用强化学习视角解释轨迹简化任务。")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--load-in-4bit", action="store_true", help="使用 bitsandbytes 4bit 量化，显存最低。")
    parser.add_argument("--load-in-8bit", action="store_true", help="使用 bitsandbytes 8bit 量化。")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU 运行。")
    parser.add_argument("--max-memory", default="", help="限制 GPU 显存，例如 18GiB。")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cuda = torch.cuda.is_available() and not args.cpu
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16 if use_cuda else torch.float32,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if use_cuda:
        model_kwargs["device_map"] = "auto"
    if args.max_memory:
        model_kwargs["max_memory"] = {"cuda:0": args.max_memory}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()

    messages = [
        {"role": "system", "content": "你是一个面向强化学习研究的本地聊天助手。"},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_device = "cuda" if use_cuda and not (args.load_in_4bit or args.load_in_8bit) else next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(target_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.08,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
