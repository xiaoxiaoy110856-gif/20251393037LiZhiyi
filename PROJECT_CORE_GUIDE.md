# 项目核心思路与代码位置

这份文档用于快速理解当前项目后端的核心链路。项目整体是 Vue 前端 + Python 本地后端，LLM 主要通过 Ollama/HuggingFace 调用，图片生成通过 ComfyUI/OpenAI provider，聊天、上下文和结果可持久化到 MySQL。

## 一、整体请求链路

前端入口：

- `frontend/src/views/WorkspaceView.vue`：主对话页、文件上传、图片工具栏、右侧上下文/证据/图片面板。
- `frontend/src/stores/appStore.js`：前端状态管理，负责发送聊天、图片生成、文件上传等请求。
- `frontend/src/api.js`：封装 `/api/chat`、`/api/images/generate` 等 HTTP 请求。

后端入口：

- `scripts/serve_local_assistant.py`：本地 HTTP 服务和路由分发。
- `backend/app.py`：API payload 层，把 HTTP 参数转成业务调用，例如聊天、知识库、图片、文件编辑、轨迹保存。

聊天主链路：

```text
frontend WorkspaceView
  -> /api/chat
  -> backend/app.py: chat_payload
  -> ContextAssembler 动态装配上下文
  -> agent_loop.py: agent_chat
  -> tool_registry.py 执行工具
  -> llm_service.py 调用 Ollama/HF
  -> memory_store.py 保存会话消息
```

## 二、Agent 与工具调用

核心文件：

- `backend/agent_loop.py`
- `backend/tool_registry.py`

`agent_loop.py` 负责判断用户请求是否需要工具。普通问题会进入 LLM 工具选择循环；明确图片生成请求会直接走 `generate_image_advanced`，避免模型只用文字描述。

`tool_registry.py` 是所有 Agent 工具的注册表，当前包括：

- `list_files`
- `read_file`
- `search_text`
- `get_file_metadata`
- `build_file_index`
- `search_project_docs`
- `summarize_experiment_text`
- `generate_image_advanced`
- `generate_image`

## 三、模型调用

核心文件：

- `backend/llm_service.py`
- `backend/model_routing.py`
- `backend/settings.py`

`llm_service.py` 支持两类后端：

- Ollama：默认走 `LOCAL_LLM_BACKEND=ollama`，通过 `LOCAL_OLLAMA_BASE_URL` 调用。
- HuggingFace：当选择 HF local model 时加载本地模型，可用 CUDA。

你的 Ollama 已经能被 `nvidia-smi` 看到，说明文字推理可以用 GPU；ComfyUI 也已通过 CUDA 版 PyTorch 使用 GPU。

## 四、长对话上下文压缩

核心文件：

- `backend/context_assembler.py`
- `backend/conversation_compressor.py`
- `backend/context_repositories.py`
- `backend/context_state.py`
- `backend/relevant_context.py`
- `backend/context_utils.py`

核心思路：

```text
MySQL 保存完整原始消息
  -> 长对话触发 ConversationCompressor
  -> 旧消息生成 segment summary
  -> conversation_states 保存滚动状态
  -> ContextAssembler 每次调用模型前只装配：
     rolling state + 相关摘要 + 相关旧消息片段 + 最近消息 + 当前问题
```

数据库表：

- `chat_sessions`
- `chat_messages`
- `conversation_summaries`
- `conversation_states`
- `context_build_logs`

## 五、文件夹级读取与分析

核心文件：

- `backend/workspace_security.py`
- `backend/workspace_tools.py`

`workspace_security.py` 负责 workspace 安全边界：

- 所有路径必须在 `LOCAL_WORKSPACE_ROOT` 内。
- 阻止 `../` 目录穿越。
- 默认不读二进制文件和敏感文件。

`workspace_tools.py` 提供 Agent 文件工具：

- `list_files`
- `read_file`
- `search_text`
- `get_file_metadata`
- `build_file_index`

## 六、知识库与 RAG

核心文件：

- `backend/knowledge_store.py`
- `backend/retrieval_rl_env.py`

`knowledge_store.py` 负责读取 `kb/raw` 文档、切分、索引、检索和生成证据片段。聊天时若启用 RAG，会把检索证据传给 Agent 或 LLM。

`retrieval_rl_env.py` 是检索策略/强化学习评估环境相关逻辑。

## 七、图片生成

基础图片服务：

- `backend/image_service.py`
- `backend/image_generation_service.py`
- `backend/openai_image_provider.py`
- `backend/image_storage.py`
- `backend/comfyui_workflow.py`

高质量图片管线：

- `backend/image_quality.py`

核心链路：

```text
用户自然语言图片需求
  -> ImagePromptRewriter
  -> ImageGenerationPlan
  -> Negative Prompt Builder
  -> ComfyUIWorkflowRunner
  -> batch 候选图
  -> ImageCritic 评分
  -> QualityController 选择最佳图
  -> 必要时修复 prompt/参数并重试
  -> 返回 /generated-images/... 给前端显示
```

`image_quality.py` 里包含：

- `ImageGenerationPlan`
- `IMAGE_PRESETS`
- `ImageNegativePromptBuilder`
- `ImagePromptRewriter`
- `ImageCritic`
- `ImageGenerationQualityController`
- `generate_image_advanced`

汽车类 prompt 会自动加入：

- one car only
- no other vehicles
- closed doors
- intact body panels
- realistic wheels
- malformed wheels / duplicate car / extra vehicles 等 negative prompt

## 八、数据库

核心文件：

- `backend/db.py`
- `backend/memory_store.py`

`db.py` 负责 MySQL 连接和表结构初始化。  
`memory_store.py` 负责会话和消息读写，MySQL 不可用时可退回文件存储。

重要表：

- `chat_sessions`
- `chat_messages`
- `conversation_summaries`
- `conversation_states`
- `context_build_logs`
- `image_artifacts`
- `eval_runs`
- `training_runs`
- `retrieval_rl_runs`
- `trajectory_runs`

图片本体不存 MySQL，而是保存在：

```text
outputs/generated_images
```

MySQL 只保存消息文本、图片 URL/引用、后续可扩展保存 artifact metadata。

## 九、配置入口

核心文件：

- `backend/settings.py`

常用环境变量：

```powershell
$env:LOCAL_DB_BACKEND="mysql"
$env:LOCAL_MYSQL_HOST="127.0.0.1"
$env:LOCAL_MYSQL_PORT="3306"
$env:LOCAL_MYSQL_USER="root"
$env:LOCAL_MYSQL_PASSWORD="123456"
$env:LOCAL_MYSQL_DATABASE="trl_agent"

$env:LOCAL_ENABLE_AGENT="1"
$env:LOCAL_ENABLE_RAG="1"
$env:LOCAL_LLM_BACKEND="ollama"
$env:LOCAL_OLLAMA_BASE_URL="http://127.0.0.1:11434"

$env:IMAGE_PROVIDER="comfyui"
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
$env:LOCAL_COMFYUI_TIMEOUT_SECONDS="900"
$env:IMAGE_GENERATION_DEFAULT_BATCH_SIZE="4"
$env:IMAGE_GENERATION_MAX_RETRIES="2"
$env:IMAGE_QUALITY_MIN_SCORE="0.75"
```

## 十、推荐维护顺序

如果后续继续整理，建议优先级：

1. `backend/app.py` 按 chat/image/file/policy/trajectory 拆成 payload service。
2. `backend/db.py` 把 schema 初始化和 run 保存逻辑拆开。
3. `backend/knowledge_store.py` 把文档解析、chunk、检索、rerank 分离。
4. 图片高级能力再接 VLM 评分、ControlNet、LoRA。

当前不要再把所有东西继续合到一个大文件里。现在的边界是比较合适的：Agent、上下文、图片、文件、知识库、数据库各自独立。
