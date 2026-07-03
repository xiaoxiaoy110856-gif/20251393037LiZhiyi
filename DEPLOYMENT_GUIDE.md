# TRL 本地助手项目部署说明

本文档说明当前项目的数据库、后端、前端、Ollama/Qwen、ComfyUI 图片生成服务如何启动。核心代码说明见 `PROJECT_CORE_GUIDE.md`。

项目根目录：

```text
C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl
```

ComfyUI 目录：

```text
C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI
```

## 1. MySQL 数据库

推荐使用当前 MySQL 配置：

```powershell
$env:LOCAL_DB_BACKEND="mysql"
$env:LOCAL_MYSQL_HOST="127.0.0.1"
$env:LOCAL_MYSQL_PORT="3306"
$env:LOCAL_MYSQL_USER="root"
$env:LOCAL_MYSQL_PASSWORD="123456"
$env:LOCAL_MYSQL_DATABASE="trl_agent"
```

后端启动时会自动建库建表，主要包括：

```text
chat_sessions
chat_messages
conversation_summaries
conversation_states
context_build_logs
image_artifacts
eval_runs
training_runs
retrieval_rl_runs
retrieval_rl_episodes
trajectory_runs
```

手动初始化：

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl"
python scripts\init_mysql_storage.py
```

检查状态：

```powershell
python scripts\doctor_local_assistant.py
```

## 2. Ollama / Qwen

默认文本模型后端是 Ollama：

```powershell
$env:LOCAL_LLM_BACKEND="ollama"
$env:LOCAL_LLM_MODEL="qwen3.5:latest"
$env:LOCAL_OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:LOCAL_OLLAMA_TIMEOUT="120"
```

检查 Ollama：

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing
```

如果 `nvidia-smi` 中能看到 `ollama.exe`，说明 Ollama 已在使用 GPU。

## 3. ComfyUI 图片生成

当前推荐使用 GPU 启动 ComfyUI。你已经安装了 CUDA 版 PyTorch 时，不要加 `--cpu`。

检查 GPU：

```powershell
D:\Miniconda\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

启动 ComfyUI：

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI"
D:\Miniconda\python.exe main.py --listen 127.0.0.1 --port 8188
```

启动成功后应看到：

```text
Device: cuda:0 NVIDIA GeForce RTX 5060 Laptop GPU
To see the GUI go to: http://127.0.0.1:8188
```

如果只能 CPU 启动，才使用：

```powershell
D:\Miniconda\python.exe main.py --listen 127.0.0.1 --port 8188 --cpu
```

真实文生图需要 checkpoint 模型文件，放到：

```text
C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI\models\checkpoints
```

支持常见后缀：

```text
.safetensors
.ckpt
```

## 4. 后端启动

推荐环境变量：

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl"

$env:LOCAL_DB_BACKEND="mysql"
$env:LOCAL_MYSQL_HOST="127.0.0.1"
$env:LOCAL_MYSQL_PORT="3306"
$env:LOCAL_MYSQL_USER="root"
$env:LOCAL_MYSQL_PASSWORD="123456"
$env:LOCAL_MYSQL_DATABASE="trl_agent"

$env:LOCAL_LLM_BACKEND="ollama"
$env:LOCAL_LLM_MODEL="qwen3.5:latest"
$env:LOCAL_OLLAMA_BASE_URL="http://127.0.0.1:11434"

$env:LOCAL_ENABLE_AGENT="1"
$env:LOCAL_ENABLE_RAG="1"
$env:LOCAL_EMBED_DEVICE="cpu"

$env:LOCAL_ASSISTANT_HOST="127.0.0.1"
$env:LOCAL_ASSISTANT_PORT="8765"

$env:IMAGE_PROVIDER="comfyui"
$env:LOCAL_ENABLE_COMFYUI="1"
$env:LOCAL_COMFYUI_DIR="C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI"
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
$env:LOCAL_COMFYUI_TIMEOUT_SECONDS="900"
$env:IMAGE_GENERATION_DEFAULT_BATCH_SIZE="4"
$env:IMAGE_GENERATION_MAX_RETRIES="2"
$env:IMAGE_QUALITY_MIN_SCORE="0.75"

python scripts\serve_local_assistant.py
```

后端地址：

```text
http://127.0.0.1:8765
```

健康检查：

```text
http://127.0.0.1:8765/api/health
```

## 5. 前端启动

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl\frontend"
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

前端地址：

```text
http://127.0.0.1:5173
```

Vite 代理：

```text
/api -> http://127.0.0.1:8765
/generated-images -> http://127.0.0.1:8765
```

生产构建：

```powershell
npm.cmd run build
```

## 6. 图片生成当前架构

Agent 图片请求默认走增强工具：

```text
generate_image_advanced
```

核心代码：

```text
backend/image_quality.py
backend/comfyui_workflow.py
backend/tool_registry.py
backend/agent_loop.py
```

图片生成链路：

```text
用户 prompt
  -> ImagePromptRewriter
  -> ImageGenerationPlan
  -> negative prompt
  -> ComfyUI batch candidates
  -> ImageCritic score
  -> QualityController retry/select best
  -> /generated-images/<file>
  -> 前端对话框内显示图片
```

图片文件保存在：

```text
outputs/generated_images
```

MySQL 不保存图片二进制，只保存聊天文本、URL 引用和可扩展 metadata。

## 7. 长对话压缩当前架构

核心代码：

```text
backend/context_assembler.py
backend/conversation_compressor.py
backend/context_repositories.py
backend/context_state.py
backend/relevant_context.py
backend/context_utils.py
```

核心思路：

```text
MySQL 保存完整原始消息
  -> 旧消息生成 segment summary
  -> conversation_states 保存 rolling state
  -> 每次请求只装配 rolling state + 相关摘要 + 相关旧消息 + 最近消息
```

## 8. 推荐启动顺序

```text
1. 启动 MySQL
2. 启动 Ollama
3. 启动 ComfyUI
4. 启动 TRL 后端
5. 启动前端或访问后端托管页面
```

## 9. 常见问题

### 图片显示成链接或破图

确认前端开发服务器有 `/generated-images` 代理，并重启前端：

```powershell
npm.cmd run dev -- --host 127.0.0.1
```

### 只有占位图

检查：

```text
1. ComfyUI 是否启动
2. http://127.0.0.1:8188 是否能打开
3. models\checkpoints 是否有 .safetensors / .ckpt
4. 后端是否重启并检测到 ComfyUI
```

### ComfyUI 很慢

检查是否使用 GPU：

```text
Device: cuda
```

如果日志是 `Device: cpu` 或 `torch+cpu`，说明没有用上 CUDA。

### 测试

```powershell
python -m unittest tests.test_image_quality_pipeline tests.test_image_generation tests.test_workspace_tools tests.test_context_compression
```
