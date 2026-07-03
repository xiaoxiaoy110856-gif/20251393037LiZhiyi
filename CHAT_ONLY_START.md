# 最小聊天模式启动

这个模式用于先验证本地聊天助手能跑通，不强制构建知识库，也不强制启动 ComfyUI。

## 1. 适用场景

- 只想测试 Qwen/Gemma 文本对话。
- 暂时不需要 RAG。
- 暂时不需要图片生成。
- 想排查模型服务、后端、前端是否正常。

## 2. 启动 Ollama

确认 Ollama 已经运行：

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing
```

默认模型：

```powershell
$env:LOCAL_LLM_BACKEND="ollama"
$env:LOCAL_LLM_MODEL="qwen3.5:latest"
$env:LOCAL_OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

## 3. 启动后端

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl"

$env:LOCAL_DB_BACKEND="file"
$env:LOCAL_ENABLE_AGENT="1"
$env:LOCAL_ENABLE_RAG="0"
$env:LOCAL_ENABLE_COMFYUI="0"
$env:LOCAL_ASSISTANT_HOST="127.0.0.1"
$env:LOCAL_ASSISTANT_PORT="8765"

python scripts\serve_local_assistant.py
```

打开：

```text
http://127.0.0.1:8765
```

## 4. 启动前端开发模式

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl\frontend"
npm.cmd run dev -- --host 127.0.0.1
```

打开：

```text
http://127.0.0.1:5173
```

## 5. 后续开启完整能力

开启 MySQL：

```powershell
$env:LOCAL_DB_BACKEND="mysql"
$env:LOCAL_MYSQL_HOST="127.0.0.1"
$env:LOCAL_MYSQL_PORT="3306"
$env:LOCAL_MYSQL_USER="root"
$env:LOCAL_MYSQL_PASSWORD="123456"
$env:LOCAL_MYSQL_DATABASE="trl_agent"
```

开启 RAG：

```powershell
$env:LOCAL_ENABLE_RAG="1"
```

开启 ComfyUI 图片生成：

```powershell
$env:IMAGE_PROVIDER="comfyui"
$env:LOCAL_ENABLE_COMFYUI="1"
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
```

## 6. 当前核心代码位置

```text
backend/app.py
backend/agent_loop.py
backend/tool_registry.py
backend/llm_service.py
backend/memory_store.py
frontend/src/views/WorkspaceView.vue
frontend/src/stores/appStore.js
```
