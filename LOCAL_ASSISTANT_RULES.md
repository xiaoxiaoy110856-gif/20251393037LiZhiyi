# 本地知识库 AI 助手规则

## 1. 项目目标

本项目目标是构建一个可以在本地稳定运行的个人 AI 助手，核心能力包括：

- 连续对话
- 本地知识库检索
- MySQL 会话记忆
- 长对话上下文压缩
- 文件夹级读取、检索、分析
- 可控文件编辑
- ComfyUI 本地图片生成
- 多模型选择

项目优先级是：

```text
先系统稳定
再上下文和工具能力
再图片/文件等扩展能力
最后再考虑训练和个性化微调
```

## 2. 当前核心模块

### 本地模型

默认文本模型通过 Ollama 调用：

```text
backend/llm_service.py
backend/model_routing.py
backend/settings.py
```

### Agent 工具系统

```text
backend/agent_loop.py
backend/tool_registry.py
```

Agent 必须通过工具访问本地文件、知识库、图片生成能力，不应绕过 registry。

### 知识库 RAG

```text
backend/knowledge_store.py
backend/retrieval_rl_env.py
```

知识库负责事实，模型负责解释。

### 长对话压缩

```text
backend/context_assembler.py
backend/conversation_compressor.py
backend/context_repositories.py
backend/context_state.py
backend/context_utils.py
backend/relevant_context.py
```

原始消息完整保存到 MySQL，不因压缩删除。模型调用时使用 rolling state、相关摘要、相关旧消息和最近消息。

### 文件夹级分析

```text
backend/workspace_security.py
backend/workspace_tools.py
```

所有路径必须限制在 workspace root 内，不允许访问整个磁盘。

### 图片生成

```text
backend/image_quality.py
backend/comfyui_workflow.py
backend/image_service.py
backend/image_generation_service.py
backend/openai_image_provider.py
backend/image_storage.py
```

聊天里的图片请求默认走：

```text
generate_image_advanced
```

不要直接把用户原始 prompt 扔给 ComfyUI。

## 3. Agent 行为规则

Agent 应该：

1. 对文件内容不做假设，必须先用工具查看。
2. 分析项目结构时先 `build_file_index` 或 `list_files`。
3. 查找功能位置时先 `search_text`，再 `read_file` 相关片段。
4. 长文件只读取相关行号范围。
5. 回答文件问题时尽量引用路径和行号。
6. 遇到图片生成请求时调用 `generate_image_advanced`。
7. 讨论图片生成方法或 prompt 写法时，不调用图片生成工具。
8. 遇到用户本人图像生成时，如果没有参考图，先要求上传参考图。
9. 遇到图片编辑请求时，说明当前主要支持文生图，编辑接口为预留能力。
10. 不读取 `.env`、密钥、证书、token、private key，除非用户明确确认风险。

## 4. 图片生成规则

图片生成请求必须经过质量管线：

```text
用户 prompt
  -> ImagePromptRewriter
  -> ImageGenerationPlan
  -> positive prompt + negative prompt
  -> ComfyUI batch candidates
  -> ImageCritic scoring
  -> QualityController retry/select best
```

汽车、人物、机械结构等容易畸形的主体必须补充结构约束。例如汽车：

- one car only
- no other vehicles
- closed doors
- intact body panels
- realistic wheels
- no malformed wheels

当前评分器是规则型，未来可以接 VLM/object detection。

## 5. 文件访问规则

所有本地文件工具必须经过：

```text
backend/workspace_security.py
```

禁止：

- `../` 路径穿越
- workspace 外路径
- 直接读取二进制文件
- 默认读取敏感文件

默认忽略：

```text
.git
node_modules
dist
build
venv
.venv
__pycache__
.next
coverage
.cache
```

## 6. 上下文规则

长对话不能把全部历史消息塞给模型。必须使用：

```text
ContextAssembler
```

装配顺序：

```text
rolling state
relevant summaries
relevant old messages
recent messages
current user message
```

压缩失败不能阻塞正常聊天。

## 7. 训练规则

训练不是当前第一优先级。只有在系统稳定、评估标准明确之后，再考虑：

- SFT
- LoRA
- PPO
- DPO
- 图片 LoRA 训练

当前图片系统只预留 `TrainingPlan` 和 LoRA 字段，不默认训练。

## 8. 本地部署原则

优先本地可控：

- Ollama 本地模型
- MySQL 本地存储
- ComfyUI 本地图片生成
- 本地 workspace 文件工具

可以支持 OpenAI Image API，但不作为当前本地闭环的必需条件。

## 9. 后续开发默认顺序

1. 保证后端和前端稳定启动。
2. 保证 MySQL 会话和上下文压缩可用。
3. 保证 RAG 检索质量。
4. 保证 workspace 文件工具安全。
5. 优化图片生成质量。
6. 接 VLM 评分、ControlNet、IP-Adapter、LoRA。
7. 最后再考虑复杂训练。

## 10. 最终原则

任何实现都应该让项目更接近：

```text
一个能在本地稳定运行、能记住上下文、能利用知识库、能安全访问文件、能生成图片的个人 AI 助手。
```
