# TRL 本地 AI 助手项目总览

本文档汇总当前项目的主要信息：项目目标、技术栈、核心功能、目录结构、运行方式、数据流、配置项、测试方式和后续扩展方向。

## 1. 项目定位

本项目是一个面向“人工智能与决策 / 强化学习 / 轨迹分析”方向的本地 AI 助手系统。

它不是单纯的聊天页面，也不是单纯的论文展示系统，而是一个可以在本地运行的个人 AI 工作台：

- 能连续对话
- 能使用本地知识库
- 能保存会话记忆
- 能压缩长上下文
- 能读取和分析本地项目文件
- 能生成图片
- 能接入本地 Ollama / HuggingFace 模型
- 能使用 MySQL 保存结构化数据

一句话概括：

```text
Vue 前端 + Python 本地后端 + Ollama/HF 文本模型 + ComfyUI 图片生成 + MySQL 持久化 + RAG 知识库 + Agent 工具系统
```

## 2. 技术栈

前端：

- Vue 3
- Vite
- Element Plus
- ECharts
- Leaflet

后端：

- Python
- 原生 HTTP server 路由
- Ollama API
- HuggingFace Transformers 可选
- ComfyUI API
- MySQL / 文件存储 fallback

知识库：

- 本地 `kb/raw`
- 文档解析与 chunk
- 关键词检索
- 可选 LlamaIndex / HuggingFace embedding

图片生成：

- ComfyUI 本地生成
- OpenAI Image API provider 预留/支持
- 高质量图片生成管线

## 3. 主要目录

```text
backend/                 后端核心代码
frontend/                Vue 前端
scripts/                 启动、检查、构建、初始化脚本
docs/                    功能文档
kb/raw/                  原始知识库资料
kb/parsed/               解析后的知识库
kb/index/                索引目录
outputs/generated_images 图片生成结果
training_data/           训练/评估数据
models/                  本地模型、LoRA 等
tests/                   自动化测试
```

## 4. 核心后端文件

### API 与业务入口

```text
backend/app.py
```

主要负责：

- 健康检查 payload
- 聊天请求 payload
- 知识库请求 payload
- 图片生成 payload
- 文件分析和编辑 payload
- 轨迹/策略展示相关 payload

核心入口已经用 `CORE ENTRY` 注释标出。

### Agent 主循环

```text
backend/agent_loop.py
```

负责：

- 判断用户请求类型
- 图片生成请求直接走 `generate_image_advanced`
- 普通请求进入 JSON 工具调用循环
- 汇总工具结果
- 返回最终回答

### 工具注册中心

```text
backend/tool_registry.py
```

所有 Agent 工具集中注册在这里：

- `list_files`
- `read_file`
- `search_text`
- `get_file_metadata`
- `build_file_index`
- `search_project_docs`
- `summarize_experiment_text`
- `generate_image_advanced`
- `generate_image`

### 模型调用

```text
backend/llm_service.py
backend/model_routing.py
```

支持：

- Ollama
- HuggingFace 本地模型

默认走 Ollama。

### 配置

```text
backend/settings.py
```

负责读取环境变量和默认路径。

### 数据库

```text
backend/db.py
backend/memory_store.py
```

负责：

- MySQL 连接
- 建库建表
- 会话读写
- 消息持久化
- 文件存储 fallback

## 5. Agent 工具能力

### 本地知识库检索

```text
backend/knowledge_store.py
```

用于 RAG 检索，回答强化学习、轨迹、PPO、DQN、DPO、路径规划等相关问题。

### 文件夹级读取和分析

```text
backend/workspace_security.py
backend/workspace_tools.py
```

能力：

- 列出文件树
- 读取文件
- 按行号读取大文件片段
- 全文搜索
- 构建文件 manifest/index
- 获取文件 metadata

安全限制：

- 只能访问 workspace root 内部
- 阻止 `../` 目录穿越
- 阻止 symlink 逃逸
- 默认忽略 `.git`、`node_modules`、`dist`、`build`、`.venv` 等目录
- 二进制文件不硬读
- 敏感文件默认不读取

### 文件修改能力

```text
backend/tools.py
```

提供：

- 文件编辑提案
- diff
- hash 校验
- 应用编辑
- 自动备份

## 6. 长对话上下文压缩

核心文件：

```text
backend/context_assembler.py
backend/conversation_compressor.py
backend/context_repositories.py
backend/context_state.py
backend/context_utils.py
backend/relevant_context.py
```

核心思想：

```text
MySQL 保存完整原始消息
  -> 旧消息生成 segment summary
  -> conversation_states 保存 rolling state
  -> 每次请求动态装配上下文
```

模型实际看到的上下文包括：

1. rolling state
2. relevant summaries
3. relevant old messages
4. recent messages
5. current user message

原始消息不会因为压缩被删除。

相关 MySQL 表：

```text
chat_sessions
chat_messages
conversation_summaries
conversation_states
context_build_logs
```

## 7. 图片生成系统

### 基础图片服务

```text
backend/image_service.py
backend/image_generation_service.py
backend/openai_image_provider.py
backend/image_storage.py
```

支持：

- ComfyUI
- OpenAI Image API provider
- SVG fallback
- 静态图片访问 `/generated-images/...`

### 高质量图片生成管线

```text
backend/image_quality.py
backend/comfyui_workflow.py
```

核心链路：

```text
用户 prompt
  -> ImagePromptRewriter
  -> ImageGenerationPlan
  -> Positive prompt
  -> Negative prompt
  -> ComfyUI batch candidates
  -> ImageCritic scoring
  -> QualityController retry/select best
  -> 保存图片
  -> 返回 /generated-images/xxx.png
```

当前支持 preset：

- `automotive_photorealistic`
- `product_photography`
- `portrait_photorealistic`
- `cinematic_scene`
- `poster_design`
- `logo_draft`
- `interior_design`
- `general_photorealistic`

聊天中的图片请求默认调用：

```text
generate_image_advanced
```

图片文件保存在：

```text
outputs/generated_images
```

MySQL 不保存图片二进制，只保存聊天文本、URL 引用和可扩展 metadata。

## 8. 前端核心文件

```text
frontend/src/views/WorkspaceView.vue
frontend/src/stores/appStore.js
frontend/src/api.js
frontend/src/styles.css
frontend/vite.config.js
```

主要能力：

- 对话界面
- 文件上传显示
- 模型选择
- 图片生成工具栏
- 右侧证据/上下文/图片面板
- Markdown / 表格 / LaTeX / 图片渲染
- `/api` 和 `/generated-images` 代理

## 9. 主要数据流

### 普通聊天

```text
用户输入
  -> /api/chat
  -> chat_payload
  -> ContextAssembler
  -> agent_chat
  -> LLM / tools
  -> append_turn
  -> 前端显示
```

### 文件分析

```text
用户上传文件或要求分析 workspace
  -> Agent 判断工具需求
  -> list_files / search_text / read_file
  -> 读取相关片段
  -> 回答并引用路径/行号
```

### 图片生成

```text
用户要求生成图片
  -> agent_chat 判断 image intent
  -> generate_image_advanced
  -> image_quality.py
  -> comfyui_workflow.py
  -> ComfyUI
  -> outputs/generated_images
  -> Markdown image
  -> 前端显示图片
```

### 长对话

```text
完整消息保存 MySQL
  -> ConversationCompressor
  -> summaries/state
  -> ContextAssembler
  -> 精简上下文给模型
```

## 10. 关键环境变量

MySQL：

```powershell
$env:LOCAL_DB_BACKEND="mysql"
$env:LOCAL_MYSQL_HOST="127.0.0.1"
$env:LOCAL_MYSQL_PORT="3306"
$env:LOCAL_MYSQL_USER="root"
$env:LOCAL_MYSQL_PASSWORD="123456"
$env:LOCAL_MYSQL_DATABASE="trl_agent"
```

Ollama：

```powershell
$env:LOCAL_LLM_BACKEND="ollama"
$env:LOCAL_LLM_MODEL="qwen3.5:latest"
$env:LOCAL_OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

Agent / RAG：

```powershell
$env:LOCAL_ENABLE_AGENT="1"
$env:LOCAL_ENABLE_RAG="1"
$env:LOCAL_EMBED_DEVICE="cpu"
```

ComfyUI：

```powershell
$env:IMAGE_PROVIDER="comfyui"
$env:LOCAL_ENABLE_COMFYUI="1"
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
$env:LOCAL_COMFYUI_TIMEOUT_SECONDS="900"
```

图片质量：

```powershell
$env:IMAGE_GENERATION_DEFAULT_BATCH_SIZE="4"
$env:IMAGE_GENERATION_MAX_RETRIES="2"
$env:IMAGE_QUALITY_MIN_SCORE="0.75"
```

上下文压缩：

```powershell
$env:CONTEXT_MAX_INPUT_TOKENS="12000"
$env:CONTEXT_RECENT_MESSAGE_COUNT="12"
$env:CONTEXT_COMPRESSION_TRIGGER_TOKENS="8000"
$env:CONTEXT_ENABLE_COMPRESSION="true"
```

## 11. 启动方式

完整启动见：

```text
DEPLOYMENT_GUIDE.md
USER_MANUAL.md
```

简要顺序：

```text
1. 启动 MySQL
2. 启动 Ollama
3. 启动 ComfyUI
4. 启动后端
5. 启动前端
```

后端：

```powershell
python scripts\serve_local_assistant.py
```

前端：

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

## 12. 测试

当前核心测试：

```powershell
python -m unittest tests.test_image_quality_pipeline tests.test_image_generation tests.test_workspace_tools tests.test_context_compression
```

覆盖：

- 图片质量管线
- OpenAI/ComfyUI 图片接口
- workspace 文件工具
- 上下文压缩
- tool registry

## 13. 当前已完成能力

- Vue 聊天前端
- 本地 Python 后端
- Ollama 多模型选择
- MySQL 会话持久化
- 长对话上下文压缩
- RAG 知识库检索
- 文件夹级读取、搜索、分析
- 文件编辑提案和应用
- ComfyUI GPU 图片生成
- 图片质量优化管线
- 图片在对话框内直接显示
- 用户手册、部署文档、核心代码说明

## 14. 当前限制

- 图片质量评分目前是规则型，不是真正 VLM 视觉评审。
- High-res fix 已预留接口，但还没有接具体二阶段 workflow。
- 图片编辑、mask 编辑、ControlNet、IP-Adapter 未启用。
- LoRA 训练只预留结构，没有实现完整训练流水线。
- `app.py`、`db.py`、`knowledge_store.py` 仍然较大，后续可以继续拆分。

## 15. 后续扩展建议

优先级建议：

1. 给图片质量管线接入 VLM 评分。
2. 增加 ComfyUI high-res fix workflow。
3. 增加 ControlNet / IP-Adapter。
4. 把 `app.py` 拆成 chat/image/file/policy/trajectory payload 模块。
5. 把 `db.py` 拆成 schema、repositories、run storage。
6. 强化 RAG eval 和自动回归测试。
7. 最后再考虑 SFT / LoRA / PPO / DPO。

## 16. 必读文档

```text
USER_MANUAL.md
DEPLOYMENT_GUIDE.md
PROJECT_CORE_GUIDE.md
agent-core-explanation.md
LOCAL_ASSISTANT_RULES.md
WORKSPACE_AGENT_TOOLS.md
docs/context-compression.md
docs/image-generation.md
docs/image-quality-pipeline.md
```
