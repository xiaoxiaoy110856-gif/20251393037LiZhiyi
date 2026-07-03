from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.storage.db import save_training_run


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DPO LoRA adapter for the local RL assistant.")
    parser.add_argument("--model", required=True, help="Local HF base model directory.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "assistant_dpo.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "rl_assistant_dpo_lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = parser.parse_args()

    examples = read_jsonl(args.data)
    if not examples:
        raise SystemExit(f"No DPO samples found. Run scripts/build_dpo_pairs.py or create {args.data}.")

    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        import torch
    except Exception as error:
        raise SystemExit(
            "DPO training requires datasets, peft, transformers, torch, and the HuggingFace trl package. "
            f"Install missing dependencies first. Import error: {error}"
        )

    save_training_run(
        run_type="dpo_lora",
        model_path=args.model,
        data_path=str(args.data),
        output_path=str(args.output),
        status="started",
        notes=f"epochs={args.epochs}, beta={args.beta}, batch_size={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}",
        metrics={"train_examples": len(examples), "beta": args.beta},
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = prepare_model_for_kbit_training(model) if getattr(model, "is_loaded_in_8bit", False) else model
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora)

    dataset = Dataset.from_list(
        [
            {
                "prompt": str(row["prompt"]),
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
            }
            for row in examples
        ]
    )

    training_args = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    try:
        trainer = DPOTrainer(model=model, ref_model=None, args=training_args, train_dataset=dataset, processing_class=tokenizer)
    except TypeError:
        trainer = DPOTrainer(model=model, ref_model=None, args=training_args, train_dataset=dataset, tokenizer=tokenizer)

    try:
        trainer.train()
        trainer.save_model(str(args.output))
        tokenizer.save_pretrained(str(args.output))
        save_training_run(
            run_type="dpo_lora",
            model_path=args.model,
            data_path=str(args.data),
            output_path=str(args.output),
            status="completed",
            metrics={"train_examples": len(examples), "beta": args.beta},
        )
        print(json.dumps({"ok": True, "output": str(args.output), "train_examples": len(examples)}, ensure_ascii=False, indent=2))
    except Exception as error:
        save_training_run(
            run_type="dpo_lora",
            model_path=args.model,
            data_path=str(args.data),
            output_path=str(args.output),
            status="failed",
            notes=str(error),
            metrics={"train_examples": len(examples), "beta": args.beta},
        )
        raise


if __name__ == "__main__":
    main()
