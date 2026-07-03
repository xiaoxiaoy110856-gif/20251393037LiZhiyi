# TRL 本地 AI 助手用户手册

这份手册面向日常使用，不讲太多代码细节。代码结构和核心实现位置见 `PROJECT_CORE_GUIDE.md`。

## 1. 启动顺序

推荐按这个顺序启动：

```text
1. MySQL
2. Ollama
3. ComfyUI
4. TRL 后端
5. 前端页面
```

## 2. 启动 ComfyUI

如果你的 PyTorch 已经能识别 GPU，启动时不要加 `--cpu`：

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI"
D:\Miniconda\python.exe main.py --listen 127.0.0.1 --port 8188
```

看到类似下面内容说明 GPU 可用：

```text
Device: cuda:0 NVIDIA GeForce RTX 5060 Laptop GPU
To see the GUI go to: http://127.0.0.1:8188
```

如果只想临时用 CPU，才使用：

```powershell
D:\Miniconda\python.exe main.py --listen 127.0.0.1 --port 8188 --cpu
```

图片模型文件放在：

```text
C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI\models\checkpoints
```

需要 `.safetensors` 或 `.ckpt` 文件。

## 3. 启动后端

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
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
$env:LOCAL_COMFYUI_TIMEOUT_SECONDS="900"

python scripts\serve_local_assistant.py
```

后端地址：

```text
http://127.0.0.1:8765
```

## 4. 启动前端

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl\frontend"
npm.cmd run dev -- --host 127.0.0.1
```

前端地址：

```text
http://127.0.0.1:5173
```

## 5. 对话助手怎么用

打开前端后，进入对话页面。你可以直接问：

```text
帮我解释 PPO 为什么能稳定策略更新
```

或者：

```text
结合知识库解释轨迹压缩和强化学习有什么关系
```

如果开启了 RAG，右侧会显示检索证据。

## 6. 上传文件并分析

在输入框旁点击 `+`，选择本地文本文件，然后输入：

```text
帮我分析这个文件
```

发送后，文件名会显示在你的消息里，助手会基于上传内容回答。

当前更适合上传文本类文件。大型二进制文件、密钥文件、证书文件不建议上传。

## 7. 分析本地项目文件夹

Agent 已经具备受控 workspace 文件工具。你可以问：

```text
分析这个项目结构
```

或者：

```text
登录逻辑在哪里实现？
```

Agent 会先搜索/列文件，再读取相关片段。它不会默认访问 workspace 外的路径。

## 8. 生成图片

你可以直接在对话框里说：

```text
帮我生成一张红色跑车在夜晚湿润城市街道上的照片
```

系统会自动走高质量图片管线：

```text
重写英文 prompt
加入 negative prompt
生成候选图
评分
必要时重试
返回最佳图片
```

图片会直接显示在对话框里，不只是链接。

也可以点击输入框旁的图片工具按钮，打开图片生成面板。

## 9. 提升图片质量的写法

尽量使用英文、具体描述：

```text
A single red exotic supercar parked alone on an empty wet city street at night, low-angle front three-quarter view, closed doors, intact body panels, realistic wheels, cinematic lighting, glossy reflections, photorealistic, no people, no text, no watermark, no other vehicles.
```

不要只写：

```text
跑车
```

因为本地扩散模型对短中文 prompt 不稳定。

## 10. 模型选择

底部模型下拉框可以选择：

- Qwen3.5 Ollama
- Gemma4 Ollama
- 本地 HF/Gemma 模型

如果某个模型无法选中，通常是对应服务或模型路径没有配置好。

## 11. 长对话记忆

系统会把完整原始消息保存到 MySQL，同时用上下文压缩避免每次都把全部历史塞给模型。

这意味着：

- 历史不会因为压缩被删除。
- 长对话会更稳。
- 追问早期内容时，系统会检索相关摘要或旧消息。

## 12. 图片和数据库

图片文件保存在：

```text
outputs/generated_images
```

MySQL 不保存图片二进制，只保存聊天文本和图片 URL/引用。

## 13. 常见问题

### 图片变成占位图

检查：

```text
1. ComfyUI 是否启动
2. ComfyUI 是否有 checkpoint 模型
3. 后端是否重启
4. LOCAL_COMFYUI_URL 是否是 http://127.0.0.1:8188
```

### 图片生成很慢

检查 ComfyUI 日志是否是：

```text
Device: cuda
```

如果是 `Device: cpu`，说明没有用 GPU。

### 图片不像你要的东西

使用英文详细 prompt，并明确数量、主体、场景、视角、光照、排除项。

### 短问题回答慢

可能原因：

- Ollama 模型较大。
- ComfyUI 正在同时占 GPU。
- 上下文过长。
- `num_ctx` 太大。

建议避免同时生成图和问大模型。

## 14. 核心代码入口

日常维护最常看的文件：

```text
backend/app.py
backend/agent_loop.py
backend/tool_registry.py
backend/llm_service.py
backend/context_assembler.py
backend/conversation_compressor.py
backend/workspace_tools.py
backend/workspace_security.py
backend/image_quality.py
backend/comfyui_workflow.py
frontend/src/views/WorkspaceView.vue
frontend/src/stores/appStore.js
```

这些文件里已经标注了 `CORE ENTRY` 注释，方便你快速找到主链路入口。

## 15. 测试

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl"
python -m unittest tests.test_image_quality_pipeline tests.test_image_generation tests.test_workspace_tools tests.test_context_compression
```
