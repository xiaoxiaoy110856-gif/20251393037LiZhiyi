from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()


REQUIRED_ANY = [
    ("config", ["config.json"]),
    ("tokenizer", ["tokenizer.json", "tokenizer.model", "spiece.model"]),
    ("weights", ["model.safetensors", "pytorch_model.bin", "model-00001-of-00002.safetensors"]),
]


def inspect_model_dir(model_dir: Path, try_load: bool) -> dict:
    files = {path.name for path in model_dir.iterdir() if path.is_file()} if model_dir.exists() else set()
    checks = {}
    missing = []
    for group, names in REQUIRED_ANY:
        ok = any(name in files for name in names)
        checks[group] = ok
        if not ok:
            missing.append(f"{group}: one of {', '.join(names)}")

    shard_files = sorted(path.name for path in model_dir.glob("*.safetensors"))
    payload = {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "checks": checks,
        "safetensor_shards": shard_files,
        "missing": missing,
        "loadable": False,
        "detail": "",
    }

    if missing or not try_load:
        payload["detail"] = "目录文件检查完成；未执行 Transformers 加载。" if not try_load else "模型目录缺少必要文件。"
        return payload

    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        payload["loadable"] = True
        payload["detail"] = f"Transformers 可读取：model_type={getattr(config, 'model_type', '')}, vocab={len(tokenizer)}"
    except Exception as error:  # pragma: no cover - depends on server model files
        payload["detail"] = str(error)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地 HuggingFace/Gemma 模型目录是否可用于聊天服务。")
    parser.add_argument("model_dir", type=Path, help="完整 HF 模型目录，不是单个 safetensors 文件。")
    parser.add_argument("--try-load", action="store_true", help="尝试用 Transformers 读取 config/tokenizer。")
    args = parser.parse_args()
    print(json.dumps(inspect_model_dir(args.model_dir, args.try_load), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
