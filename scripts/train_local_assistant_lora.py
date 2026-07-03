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


SYSTEM_PROMPT = (
    "You are a local AI assistant for reinforcement learning research. "
    "You should answer naturally about trajectories, path planning, reward design, "
    "policy optimization, PPO/DPO, and reinforcement learning. "
    "When evidence is insufficient, say so clearly and do not invent conclusions."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def load_conversation_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = []
    for session in payload.get("sessions", []):
        messages = session.get("messages", [])
        for index in range(0, len(messages) - 1, 2):
            user = messages[index]
            assistant = messages[index + 1]
            if user.get("role") == "user" and assistant.get("role") == "assistant":
                examples.append({"instruction": user.get("content", ""), "output": assistant.get("content", "")})
    return examples


def normalize_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in example:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": item["role"], "content": item["content"]} for item in example["messages"])
        return messages

    instruction = str(example.get("instruction") or example.get("query") or "")
    input_text = str(example.get("input") or example.get("context") or "")
    output = str(example.get("output") or example.get("answer") or "")
    user = instruction if not input_text else f"{instruction}\n\nAdditional context:\n{input_text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local LoRA/SFT adapter for the RL assistant.")
    parser.add_argument("--model", required=True, help="Local HF model directory.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "assistant_sft.jsonl")
    parser.add_argument("--include-conversations", action="store_true", help="Also include conversations/sessions.json as training samples.")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "rl_assistant_lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    raw_examples = read_jsonl(args.data)
    if args.include_conversations:
        raw_examples.extend(load_conversation_examples(ROOT / "conversations" / "sessions.json"))
    raw_examples = [item for item in raw_examples if item]
    if not raw_examples:
        raise SystemExit(f"No training samples found. Please create {args.data}.")

    save_training_run(
        run_type="sft_lora",
        model_path=args.model,
        data_path=str(args.data),
        output_path=str(args.output),
        status="started",
        notes=f"epochs={args.epochs}, batch_size={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = tokenizer.apply_chat_template(normalize_messages(example), tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, truncation=True, max_length=args.max_length)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(raw_examples).map(tokenize, remove_columns=list(raw_examples[0].keys()))
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
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    try:
        trainer.train()
        trainer.save_model(str(args.output))
        tokenizer.save_pretrained(str(args.output))
        save_training_run(
            run_type="sft_lora",
            model_path=args.model,
            data_path=str(args.data),
            output_path=str(args.output),
            status="completed",
            metrics={"train_examples": len(raw_examples)},
        )
        print(f"LoRA adapter saved to {args.output}")
    except Exception as error:
        save_training_run(
            run_type="sft_lora",
            model_path=args.model,
            data_path=str(args.data),
            output_path=str(args.output),
            status="failed",
            notes=str(error),
        )
        raise


if __name__ == "__main__":
    main()
