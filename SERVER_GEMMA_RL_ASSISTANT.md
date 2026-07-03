# Gemma / HuggingFace 本地助手部署说明

本文档用于在不走 Ollama、直接加载 HuggingFace 本地模型时参考。当前项目默认推荐 Ollama/Qwen；Gemma/HF 作为可选模型路径保留。

## 1. 模型目录要求

HuggingFace 本地模型目录应包含完整文件，例如：

```text
config.json
generation_config.json
tokenizer.json 或 tokenizer.model
tokenizer_config.json
special_tokens_map.json
model.safetensors.index.json
model-00001-of-000xx.safetensors
model-00002-of-000xx.safetensors
```

只有 `.safetensors` 分片通常不够，缺 tokenizer/config 会加载失败。

## 2. 后端 HF 配置

```powershell
$env:LOCAL_LLM_BACKEND="hf"
$env:LOCAL_LLM_MODEL_PATH="E:\path\to\gemma-model"
$env:LOCAL_HF_LOCAL_FILES_ONLY="1"
$env:LOCAL_HF_LOAD_IN_4BIT="0"
$env:LOCAL_HF_LOAD_IN_8BIT="0"
$env:LOCAL_HF_MAX_MEMORY="7GiB"
```

如果使用 LoRA：

```powershell
$env:LOCAL_LORA_ADAPTER_PATH="C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl\models\rl_assistant_lora"
```

## 3. 启动后端

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl"

$env:LOCAL_ENABLE_AGENT="1"
$env:LOCAL_ENABLE_RAG="1"
$env:LOCAL_ASSISTANT_HOST="127.0.0.1"
$env:LOCAL_ASSISTANT_PORT="8765"

python scripts\serve_local_assistant.py
```

## 4. GPU 注意事项

HF 路径会在 `backend/llm_service.py` 中检查：

```python
torch.cuda.is_available()
```

如果可用，会使用：

```python
device_map = "auto"
torch_dtype = torch.float16
```

8GB 显存下建议：

- 优先使用较小模型或量化模型。
- 控制上下文长度。
- 避免同时运行大图生成和大模型推理。

## 5. 当前 Agent 能力

不管模型是 Ollama 还是 HF，Agent 工具系统相同：

```text
backend/agent_loop.py
backend/tool_registry.py
```

支持：

- RAG 检索
- workspace 文件分析
- 长对话压缩
- ComfyUI 图片生成
- 文件编辑提案

## 6. 训练建议

训练不是当前第一优先级。建议先稳定：

```text
聊天质量
RAG 质量
上下文压缩
文件工具
图片生成
```

然后再考虑：

- SFT
- LoRA
- PPO/DPO

训练数据建议放在：

```text
training_data/
```

模型输出建议放在：

```text
models/
```

## 7. 相关核心文档

```text
PROJECT_CORE_GUIDE.md
DEPLOYMENT_GUIDE.md
docs/context-compression.md
docs/image-generation.md
docs/image-quality-pipeline.md
WORKSPACE_AGENT_TOOLS.md
```
