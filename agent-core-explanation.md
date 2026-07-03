# Agent 核心代码解析

本文档解释当前 TRL 本地助手的 Agent 核心代码。它已经不是 Rust CLI Agent，而是 Python 后端 + Vue 前端 + Ollama/ComfyUI/MySQL 的本地助手项目。

## 1. 核心链路

```text
用户在前端输入
  -> frontend/src/stores/appStore.js
  -> POST /api/chat
  -> backend/app.py: chat_payload
  -> ContextAssembler 动态装配上下文
  -> backend/agent_loop.py: agent_chat
  -> backend/tool_registry.py 执行工具
  -> backend/llm_service.py 调用 Ollama/HF
  -> backend/memory_store.py 保存会话
```

## 2. Agent 决策入口

核心文件：

```text
backend/agent_loop.py
```

`agent_chat()` 是 Agent 的主入口。它会先判断用户请求类型：

1. 明确图片生成请求：直接调用 `generate_image_advanced`。
2. 普通文本/检索/文件分析请求：进入 JSON 工具调用循环。
3. 图片编辑请求：当前提示“只支持文生图”。
4. 用户本人图像请求：要求先上传参考图。

图片请求不再让 LLM 自己决定是否调用工具，而是由代码直接路由，避免模型只给文字解释。

## 3. 工具注册

核心文件：

```text
backend/tool_registry.py
```

所有 Agent 工具都集中注册在 `TOOL_REGISTRY`：

```text
list_files
read_file
search_text
get_file_metadata
build_file_index
search_project_docs
summarize_experiment_text
generate_image_advanced
generate_image
```

执行工具统一走：

```python
execute_tool(name, tool_input)
```

这样工具名称、参数描述、handler 不会散落在业务代码里。

## 4. 普通工具调用循环

普通请求走这个协议：

```json
{"type":"tool_use","id":"tool_1","name":"search_project_docs","input":{"query":"..."}}
```

或者：

```json
{"type":"final","answer":"..."}
```

流程：

```text
Agent system prompt + history + tool catalog
  -> LLM 输出 JSON
  -> agent_loop 解析 JSON
  -> execute_tool
  -> tool result 写入上下文
  -> 继续下一轮
```

当前每轮最多工具调用次数由：

```text
LOCAL_AGENT_MAX_TURNS
```

控制。

## 5. 图片生成特殊路径

图片请求默认调用：

```text
generate_image_advanced
```

核心文件：

```text
backend/image_quality.py
backend/comfyui_workflow.py
```

链路：

```text
用户 prompt
  -> ImagePromptRewriter
  -> ImageGenerationPlan
  -> negative prompt
  -> ComfyUIWorkflowRunner
  -> 多张候选图
  -> ImageCritic 评分
  -> QualityController 选择最佳图
  -> 返回 Markdown 图片
```

返回给前端的内容里包含：

```markdown
![generated image](/generated-images/quality_xxx.png)
```

前端会直接在对话框里渲染图片。

## 6. 文件夹级分析

核心文件：

```text
backend/workspace_security.py
backend/workspace_tools.py
```

安全规则：

- 所有路径必须通过 `resolve_safe_path`。
- 禁止访问 workspace 外路径。
- 禁止 `../` 目录穿越。
- 默认忽略 `.git`、`node_modules`、`dist`、`build`、`.venv` 等目录。
- 二进制文件不硬读。
- 敏感文件默认不读取。

工具：

```text
list_files
read_file
search_text
get_file_metadata
build_file_index
```

## 7. 长对话上下文

核心文件：

```text
backend/context_assembler.py
backend/conversation_compressor.py
backend/context_repositories.py
backend/context_state.py
backend/context_utils.py
backend/relevant_context.py
```

核心策略：

```text
MySQL 保存完整原始消息
旧消息生成 segment summary
conversation_states 保存 rolling state
每次调用模型时动态装配上下文
```

上下文装配顺序：

```text
rolling state
相关摘要
相关旧消息
最近消息
当前用户输入
```

## 8. 模型调用

核心文件：

```text
backend/llm_service.py
backend/model_routing.py
backend/settings.py
```

支持：

- Ollama
- 本地 HuggingFace model

当前默认：

```text
LOCAL_LLM_BACKEND=ollama
LOCAL_LLM_MODEL=qwen3.5:latest
```

## 9. 持久化

核心文件：

```text
backend/db.py
backend/memory_store.py
```

MySQL 保存：

- 会话
- 消息
- 上下文摘要
- rolling state
- 图片 artifact metadata
- 训练/评估/轨迹记录

图片文件本体保存在：

```text
outputs/generated_images
```

## 10. Agent 本质

当前 Agent 的本质是：

```text
LLM 负责理解和表达
Agent loop 负责工具选择和执行闭环
Tool registry 负责能力边界
ContextAssembler 负责长上下文控制
Workspace tools 负责本地文件可控访问
Image quality pipeline 负责图片生成质量控制
Memory/MySQL 负责长期持久化
```

这也是后续维护时最重要的边界：不要把工具逻辑塞回 LLM prompt，也不要让 LLM 直接绕过安全层访问文件或图片路径。
