from __future__ import annotations

import os
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "kb"
RAW_KB_DIR = KB_DIR / "raw"
PARSED_KB_DIR = KB_DIR / "parsed"
INDEX_DIR = KB_DIR / "index"
KB_JSON_PATH = PARSED_KB_DIR / "knowledge_base.json"
CONVERSATIONS_DIR = ROOT / "conversations"
UI_DIR = ROOT / "ui"
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
REPOS_DIR = ROOT / "repos"
OUTPUTS_DIR = ROOT / "outputs"
GENERATED_IMAGES_DIR = OUTPUTS_DIR / "generated_images"
FILE_BACKUPS_DIR = OUTPUTS_DIR / "file_backups"
QWEN35_OLLAMA_DIR = Path(r"E:\Ollma\environment\Qwen3.5")
COMFYUI_DIR = Path(r"C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI")


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_local_env() -> None:
    _load_env_file(ROOT / ".env")
    _load_env_file(ROOT / ".env.local")


load_local_env()


def get_raw_kb_dir() -> Path:
    return Path(os.getenv("LOCAL_KB_RAW_DIR", RAW_KB_DIR))


def get_parsed_kb_dir() -> Path:
    return Path(os.getenv("LOCAL_KB_PARSED_DIR", PARSED_KB_DIR))


def get_index_dir() -> Path:
    return Path(os.getenv("LOCAL_KB_INDEX_DIR", INDEX_DIR))


def get_kb_json_path() -> Path:
    return Path(os.getenv("LOCAL_KB_JSON_PATH", KB_JSON_PATH))


def get_conversations_dir() -> Path:
    return Path(os.getenv("LOCAL_CONVERSATIONS_DIR", CONVERSATIONS_DIR))


def get_ui_dir() -> Path:
    override = os.getenv("LOCAL_UI_DIR", "").strip()
    if override:
        return Path(override)
    if FRONTEND_DIST_DIR.exists():
        return FRONTEND_DIST_DIR
    return UI_DIR


def get_repos_dir() -> Path:
    return Path(os.getenv("LOCAL_REPOS_DIR", REPOS_DIR))


def get_ui_mode() -> str:
    mode = os.getenv("LOCAL_UI_MODE", "assistant").strip().lower() or "assistant"
    return mode if mode in {"assistant", "workspace", "advanced"} else "assistant"


def advanced_ui_enabled() -> bool:
    return os.getenv("LOCAL_ENABLE_ADVANCED_UI", "1").strip().lower() not in {"0", "false", "no", "off"} or get_ui_mode() == "advanced"


def get_db_backend() -> str:
    return os.getenv("LOCAL_DB_BACKEND", "file").strip().lower() or "file"


def mysql_enabled() -> bool:
    return get_db_backend() == "mysql"


def get_mysql_host() -> str:
    return os.getenv("LOCAL_MYSQL_HOST", "127.0.0.1").strip() or "127.0.0.1"


def get_mysql_port() -> int:
    return int(os.getenv("LOCAL_MYSQL_PORT", "3306"))


def get_mysql_user() -> str:
    return os.getenv("LOCAL_MYSQL_USER", "root").strip() or "root"


def get_mysql_password() -> str:
    return os.getenv("LOCAL_MYSQL_PASSWORD", "").strip()


def get_mysql_database() -> str:
    return os.getenv("LOCAL_MYSQL_DATABASE", "trl_agent").strip() or "trl_agent"


# 核心1：LLM 后端路由配置。前端选择模型后，后端会根据这里的环境变量决定走 Ollama/Qwen 还是 HF 本地模型。
def get_llm_backend() -> str:
    return os.getenv("LOCAL_LLM_BACKEND", "ollama").strip().lower() or "ollama"


# 核心1：Ollama 中实际调用的 Qwen 模型名称，默认值用于本地 `ollama run qwen3.5:latest` 这类服务。
def get_ollama_model() -> str:
    return os.getenv("LOCAL_LLM_MODEL", "qwen3.5:latest").strip() or "qwen3.5:latest"


# 备用模型配置：如果前端选择 Gemma4，会从这里读取 Ollama 模型名。
def get_gemma_ollama_model() -> str:
    return os.getenv("LOCAL_GEMMA_OLLAMA_MODEL", "gemma4:latest").strip() or "gemma4:latest"


# 核心1：Ollama 服务地址。`llm_service._chat_ollama()` 会基于这个地址请求 `/api/chat`。
def get_ollama_base_url() -> str:
    return os.getenv("LOCAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/") or "http://127.0.0.1:11434"


# 核心1：Ollama 单次生成的超时时间，避免本地大模型卡住后端请求。
def get_ollama_timeout() -> int:
    return int(os.getenv("LOCAL_OLLAMA_TIMEOUT", "120"))


# HF 本地模型路径：作为 Ollama 不可用时的备用生成路径。
def get_hf_model_path() -> str:
    return os.getenv("LOCAL_LLM_MODEL_PATH", "").strip()


# 核心1：前端模型下拉框的数据源。Qwen/Ollama、Gemma/Ollama 等选项都从这里暴露给 UI。
def get_model_options() -> list[dict[str, str | bool]]:
    custom = os.getenv("LOCAL_MODEL_OPTIONS", "").strip()
    if custom:
        try:
            parsed = json.loads(custom)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    options: list[dict[str, str | bool]] = [
        {
            "id": "qwen35-ollama",
            "label": "Qwen3.5 (Ollama)",
            "backend": "ollama",
            "model": get_ollama_model(),
            "path": str(QWEN35_OLLAMA_DIR),
            "available": QWEN35_OLLAMA_DIR.exists(),
        }
    ]
    options.append(
        {
            "id": "gemma4-ollama",
            "label": "Gemma4 (Ollama)",
            "backend": "ollama",
            "model": get_gemma_ollama_model(),
            "path": "",
            "available": True,
        }
    )
    hf_model_path = get_hf_model_path()
    if hf_model_path:
        options.append(
            {
                "id": "gemma-hf",
                "label": "Gemma / HF Local",
                "backend": "hf",
                "model": hf_model_path,
                "path": hf_model_path,
                "available": Path(hf_model_path).exists(),
            }
        )
    else:
        options.append(
            {
                "id": "gemma-hf",
                "label": "Gemma / HF Local",
                "backend": "hf",
                "model": "",
                "path": "",
                "available": False,
            }
        )
    return options


# 核心1：把前端传入的 model_id 转换成后端可执行的模型配置。
def resolve_model_option(model_id: str | None = None) -> dict[str, str | bool]:
    options = get_model_options()
    requested = (model_id or os.getenv("LOCAL_DEFAULT_MODEL_ID", "")).strip()
    if requested:
        for option in options:
            if str(option.get("id", "")) == requested:
                return option
    current_backend = get_llm_backend()
    for option in options:
        if str(option.get("backend", "")) == current_backend and str(option.get("model", "")):
            return option
    return options[0]


def get_lora_adapter_path() -> str:
    return os.getenv("LOCAL_LORA_ADAPTER_PATH", "").strip()


def get_hf_local_files_only() -> bool:
    return os.getenv("LOCAL_HF_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no"}


def get_hf_load_in_4bit() -> bool:
    return os.getenv("LOCAL_HF_LOAD_IN_4BIT", "0").strip().lower() in {"1", "true", "yes", "on"}


def get_hf_load_in_8bit() -> bool:
    return os.getenv("LOCAL_HF_LOAD_IN_8BIT", "0").strip().lower() in {"1", "true", "yes", "on"}


def get_hf_max_memory() -> str:
    return os.getenv("LOCAL_HF_MAX_MEMORY", "").strip()


def get_embed_model() -> str:
    return os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5").strip() or "BAAI/bge-small-zh-v1.5"


def get_embed_device() -> str:
    return os.getenv("LOCAL_EMBED_DEVICE", "cpu").strip().lower() or "cpu"


def get_top_k() -> int:
    return int(os.getenv("LOCAL_TOP_K", "4"))


def get_history_turns() -> int:
    return int(os.getenv("LOCAL_HISTORY_TURNS", "8"))


def get_agent_max_turns() -> int:
    return int(os.getenv("LOCAL_AGENT_MAX_TURNS", "3"))


def get_context_char_limit() -> int:
    return int(os.getenv("LOCAL_CONTEXT_CHAR_LIMIT", "3600"))


def context_max_input_tokens() -> int:
    return int(os.getenv("CONTEXT_MAX_INPUT_TOKENS", "12000"))


def context_recent_message_count() -> int:
    return int(os.getenv("CONTEXT_RECENT_MESSAGE_COUNT", "12"))


def context_compression_trigger_tokens() -> int:
    return int(os.getenv("CONTEXT_COMPRESSION_TRIGGER_TOKENS", "8000"))


def context_segment_message_count() -> int:
    return int(os.getenv("CONTEXT_SEGMENT_MESSAGE_COUNT", "30"))


def context_summary_max_tokens() -> int:
    return int(os.getenv("CONTEXT_SUMMARY_MAX_TOKENS", "1500"))


def context_retrieval_max_items() -> int:
    return int(os.getenv("CONTEXT_RETRIEVAL_MAX_ITEMS", "8"))


def context_retrieval_max_tokens() -> int:
    return int(os.getenv("CONTEXT_RETRIEVAL_MAX_TOKENS", "2500"))


# 核心2：上下文压缩总开关。关闭后，后端只使用最近历史消息，不写入压缩摘要。
def context_compression_enabled() -> bool:
    return os.getenv("CONTEXT_ENABLE_COMPRESSION", "true").strip().lower() not in {"0", "false", "no", "off"}


def context_build_log_enabled() -> bool:
    return os.getenv("CONTEXT_ENABLE_BUILD_LOG", "true").strip().lower() not in {"0", "false", "no", "off"}


def get_research_focus() -> str:
    return os.getenv("LOCAL_RESEARCH_FOCUS", "reinforcement_learning").strip().lower() or "reinforcement_learning"


# 核心5：RAG 检索开关。关闭后，Agent 不再从本地知识库取证据。
def rag_enabled() -> bool:
    return os.getenv("LOCAL_ENABLE_RAG", "1").strip().lower() not in {"0", "false", "no", "off"}


# 核心5：强化学习检索策略开关。开启后，RAG 检索前会先由策略选择 retrieval action。
def retrieval_policy_enabled() -> bool:
    return os.getenv("LOCAL_ENABLE_RETRIEVAL_POLICY", "1").strip().lower() not in {"0", "false", "no", "off"}


def get_retrieval_policy_path() -> str:
    return os.getenv("LOCAL_RETRIEVAL_POLICY_PATH", "").strip()


# 核心1：Agent 编排开关。开启后聊天走工具调用循环，关闭后直接走普通 LLM/RAG 回答。
def agent_enabled() -> bool:
    return os.getenv("LOCAL_ENABLE_AGENT", "1").strip().lower() not in {"0", "false", "no", "off"}


def get_host() -> str:
    return os.getenv("LOCAL_ASSISTANT_HOST", "127.0.0.1").strip() or "127.0.0.1"


def get_port() -> int:
    return int(os.getenv("LOCAL_ASSISTANT_PORT", "8765"))


def get_generated_images_dir() -> Path:
    return Path(os.getenv("IMAGE_OUTPUT_DIR") or os.getenv("LOCAL_GENERATED_IMAGES_DIR", GENERATED_IMAGES_DIR))


def get_image_public_base_url() -> str:
    return os.getenv("IMAGE_PUBLIC_BASE_URL", "").strip().rstrip("/")


# 核心3：图片生成服务选择，支持 OpenAI、ComfyUI 或外部命令路径。
def get_image_provider() -> str:
    return os.getenv("IMAGE_PROVIDER", "comfyui").strip().lower() or "comfyui"


def get_image_model() -> str:
    return os.getenv("IMAGE_MODEL", "gpt-image-2").strip() or "gpt-image-2"


def get_image_default_size() -> str:
    return os.getenv("IMAGE_DEFAULT_SIZE", "1024x1024").strip() or "1024x1024"


def get_image_default_quality() -> str:
    return os.getenv("IMAGE_DEFAULT_QUALITY", "auto").strip() or "auto"


def get_image_default_format() -> str:
    return os.getenv("IMAGE_DEFAULT_FORMAT", "png").strip().lower() or "png"


def get_image_default_background() -> str:
    return os.getenv("IMAGE_DEFAULT_BACKGROUND", "auto").strip().lower() or "auto"


def get_image_generation_max_retries() -> int:
    return int(os.getenv("IMAGE_GENERATION_MAX_RETRIES", "2"))


def get_image_generation_default_batch_size() -> int:
    return max(1, min(4, int(os.getenv("IMAGE_GENERATION_DEFAULT_BATCH_SIZE", "4"))))


def get_image_quality_min_score() -> float:
    return float(os.getenv("IMAGE_QUALITY_MIN_SCORE", "0.75"))


def get_comfyui_timeout_seconds() -> int:
    return int(os.getenv("LOCAL_COMFYUI_TIMEOUT_SECONDS", os.getenv("COMFYUI_TIMEOUT_SECONDS", "900")))


def get_comfyui_default_steps() -> int:
    return int(os.getenv("LOCAL_COMFYUI_STEPS", os.getenv("COMFYUI_DEFAULT_STEPS", "20")))


def get_comfyui_default_cfg() -> float:
    return float(os.getenv("LOCAL_COMFYUI_CFG", os.getenv("COMFYUI_DEFAULT_CFG", "7")))


def get_comfyui_checkpoint() -> str:
    return os.getenv("LOCAL_COMFYUI_CHECKPOINT", os.getenv("COMFYUI_CHECKPOINT", "")).strip()


def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/") or "https://api.openai.com/v1"


def get_file_write_root() -> Path:
    return Path(os.getenv("LOCAL_FILE_WRITE_ROOT", ROOT))


def get_file_backups_dir() -> Path:
    return Path(os.getenv("LOCAL_FILE_BACKUPS_DIR", FILE_BACKUPS_DIR))


def get_workspace_root() -> Path:
    return Path(os.getenv("LOCAL_WORKSPACE_ROOT", ROOT))


def get_workspace_max_read_bytes() -> int:
    return int(os.getenv("LOCAL_WORKSPACE_MAX_READ_BYTES", "200000"))


def get_workspace_index_max_file_bytes() -> int:
    return int(os.getenv("LOCAL_WORKSPACE_INDEX_MAX_FILE_BYTES", "2000000"))


def get_image_generator_command() -> str:
    return os.getenv("LOCAL_IMAGE_GENERATOR_COMMAND", "").strip()


def get_comfyui_dir() -> Path:
    return Path(os.getenv("LOCAL_COMFYUI_DIR", COMFYUI_DIR))


def get_comfyui_url() -> str:
    return os.getenv("LOCAL_COMFYUI_URL", "http://127.0.0.1:8188").strip().rstrip("/") or "http://127.0.0.1:8188"


def comfyui_enabled() -> bool:
    return os.getenv("LOCAL_ENABLE_COMFYUI", "1").strip().lower() not in {"0", "false", "no", "off"}


def ensure_runtime_dirs() -> None:
    for path in [
        get_raw_kb_dir(),
        get_parsed_kb_dir(),
        get_index_dir(),
        get_conversations_dir(),
        get_ui_dir(),
        get_repos_dir(),
        get_generated_images_dir(),
        get_file_backups_dir(),
    ]:
        path.mkdir(parents=True, exist_ok=True)
