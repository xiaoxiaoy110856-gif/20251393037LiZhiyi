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
    parser = argparse.ArgumentParser(description="把 LoRA adapter 合并进基座模型，导出一个可直接聊天加载的 HF 模型目录。")
    parser.add_argument("--base-model", required=True, help="基座模型目录，例如 /home/xiaoy/project/models/gemma-4-E4B-it")
    parser.add_argument("--adapter", required=True, help="LoRA adapter 目录，例如 models/rl_assistant_lora")
    parser.add_argument("--output", required=True, help="合并后的模型输出目录，例如 models/rl_assistant_merged")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": torch.float16 if torch.cuda.is_available() else torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter, local_files_only=True)
    model = model.merge_and_unload()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=True)
    tokenizer.save_pretrained(str(output))
    print(f"Merged chat model saved to {output}")


if __name__ == "__main__":
    main()
