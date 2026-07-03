# 核心代码文件说明

这份文档用于说明本项目 Agent 的几组核心代码文件分别负责什么。整体流程可以理解为：

`前端输入 -> /api/chat -> Agent 编排 -> 工具/RAG/RL 检索策略 -> Qwen/Ollama 生成回答 -> 前端展示`

## 1. Qwen / Ollama 后端接口

### `backend/settings.py`

作用：集中管理环境变量和模型配置。

重点位置：

- `get_llm_backend()`：决定当前使用 `ollama` 还是 HuggingFace 本地模型。
- `get_ollama_model()`：读取 Qwen 模型名，默认是 `qwen3.5:latest`。
- `get_ollama_base_url()`：读取 Ollama 服务地址，默认是 `http://127.0.0.1:11434`。
- `get_model_options()`：给前端模型选择框提供可选模型。

### `backend/llm_service.py`

作用：统一封装 LLM 调用。

重点位置：

- `build_messages()`：把系统提示词、压缩上下文、历史对话、当前问题组装成模型消息。
- `_chat_ollama()`：真正向 Ollama `/api/chat` 发送请求。
- `_chat_hf()`：HuggingFace 本地模型备用路径。
- `run_messages()`：统一入口，Agent、RAG、上下文压缩都通过这里调用 Qwen。

### `backend/app.py`

作用：后端业务接口层。

重点位置：

- `chat_payload()`：聊天主入口，前端 `/api/chat` 请求进入这里。
- `policy_evaluation_payload()`：给前端强化学习效果图提供数据。
- `save_trajectory_payload()`：保存地图轨迹、DQN/PPO、S3/RLTS/Mlsimp 展示结果。

### `backend/agent_loop.py`

作用：Agent 的工具调用与回答生成主循环。

重点位置：

- `agent_chat()`：Agent 主入口。
- `_parse_model_action()`：解析模型输出，判断是调用工具还是最终回答。
- `_execute_tool()`：执行工具调用。
- `should_generate_image()`：判断用户是不是在要求生成图片。

## 2. 上下文压缩

### `backend/conversation_compressor.py`

作用：把长对话中的旧消息压缩成摘要，减小后续 prompt 长度。

重点位置：

- `maybe_compress()`：压缩入口，只有开启 MySQL 和压缩开关时才执行。
- `_compress()`：选择可压缩的历史消息并写入摘要。
- `_summarize_segment()`：调用 Qwen 生成结构化摘要。

### `backend/context_assembler.py`

作用：在每次生成回答前，把“滚动状态 + 相关摘要 + 最近消息”重新组装成上下文。

重点位置：

- `assemble()`：上下文组装入口。
- `_can_add()`：判断某段上下文是否还能放进 token 预算。
- `_trim_messages()`：在预算不足时保留最近消息。

## 3. 图片生成

### `backend/tool_registry.py`

作用：注册 Agent 可调用的工具。

重点位置：

- `generate_image_advanced`：Agent 生成图片时调用的高级工具。
- `read_file`、`search_text`：Agent 分析本地代码/文件时使用的工具。

### `backend/image_quality.py`

作用：高级图片生成质量控制。

重点位置：

- `ImagePromptRewriter`：改写用户 prompt。
- `ImageCritic`：对候选图进行质量评分。
- `ImageGenerationQualityController.generate()`：完整执行“改写 prompt -> 生成候选图 -> 评分 -> 必要时重试”。
- `generate_image_advanced()`：工具注册层调用的入口函数。

### `backend/image_service.py`

作用：简单图片生成路径，支持 OpenAI、ComfyUI 或外部命令。

重点位置：

- `generate_image()`：直接图片生成接口，主要给 `/api/images/generate` 使用。

## 4. 前端对话框文字读取

### `frontend/src/views/WorkspaceView.vue`

作用：聊天页面组件。

重点位置：

- `v-model="appState.queryInput"`：用户输入框绑定。
- `onSend()`：触发发送。
- `messageText()`：渲染消息前清理附件标记。

### `frontend/src/stores/appStore.js`

作用：前端状态管理和 API 调用。

重点位置：

- `sendCurrentMessage()`：读取输入框文字并发送到后端。
- `attachLocalFile()`：读取上传文件文本。
- `generateImageFromPrompt()` / `sendImageGenerationFromComposer()`：图片生成相关入口。

### `frontend/src/api.js`

作用：前端 API 封装。

重点位置：

- `api.chat()`：请求后端 `/api/chat`。
- `api.generateImage()`：请求后端 `/api/images/generate`。

## 5. 强化学习检索策略

### `backend/retrieval_rl_env.py`

作用：定义强化学习环境。

重点位置：

- `ACTIONS`：离散检索动作，如 `baseline`、`rl_focus`、`trajectory_focus`、`reward_focus`。
- `features_for_query()`：把问题转成策略状态向量。
- `source_hit()`：判断是否找到预期来源。
- `topic_hit()`：判断是否覆盖预期主题。
- `RetrievalRLEnv.step()`：执行一次检索动作并计算 reward。

### `backend/retrieval_policy.py`

作用：线上加载训练好的策略，并为每个问题选择检索动作。

重点位置：

- `_candidate_policy_paths()`：按优先级查找 LinUCB、Dueling DDQN、DPO、ORPO、PPO 等策略文件。
- `_load_policy()`：根据 checkpoint 类型加载不同策略。
- `choose_retrieval_action()`：线上真正选择 retrieval action 的入口。

### `backend/tools.py`

作用：Agent 工具实现。

重点位置：

- `search_project_docs()`：Agent 的 RAG 检索工具，会先调用 RL 策略选择检索动作。
- `read_local_file()` / `propose_file_edit()` / `apply_file_edit()`：本地文件读取和修改能力。

### 训练脚本

- `scripts/train_retrieval_policy_ppo.py`：PPO 检索策略训练。
- `scripts/train_retrieval_policy_dpo.py`：DPO 偏好训练。
- `scripts/train_retrieval_policy_orpo.py`：ORPO 对照训练。
- `scripts/train_retrieval_policy_linucb.py`：LinUCB 上下文 bandit 策略。
- `scripts/train_retrieval_policy_dueling_ddqn.py`：Dueling Double DQN 神经策略。

这些脚本都会输出 `evaluation.json` 和 `training_trace.json`，前端图表使用这些文件展示训练效果。

## 6. 地图 DQN / PPO 展示

### `frontend/src/views/PathDemoView.vue`

作用：地图轨迹演示页面。

重点位置：

- `fetchRoadRoute()`：从 OSRM 获取真实道路 baseline。
- `buildDqnDecisionPoints()`：把路口/转向点转换成 DQN 决策点。
- `buildPpoStages()`：把路径分段展示为 PPO 阶段策略。
- `renderDqn()`：绘制 DQN 决策链。
- `renderPpo()`：绘制 PPO 阶段链。

## 7. S3 / RLTS / MLsimp 展示

### `frontend/src/views/PathDemoView.vue`

作用：同一个地图页面中展示轨迹压缩算法。

重点位置：

- `compressionStrategies`：保存 S3、RLTS、Mlsimp 三种算法保留点。
- `buildCompressionStrategies()`：根据路线点和转向点生成三种压缩结果。
- `renderCompression()`：在地图上高亮压缩后保留的关键点。
- `compressionStats`：计算保留点数、总点数和压缩率。

### `backend/db.py`

作用：数据库存储。

重点位置：

- `trajectory_runs` 表：保存地图轨迹、RL 方法、压缩方法和路线几何。
- `save_trajectory_run()`：写入当前地图实验结果。
- `list_recent_trajectory_runs()`：读取最近地图实验记录。

## 8. 总结

本项目不是简单调用 LlamaIndex/LangChain，而是自己实现了：

- Qwen/Ollama 后端接口；
- Agent 工具调用循环；
- 本地文件读取与修改；
- 上下文压缩；
- RAG 知识库检索；
- PPO/DPO/ORPO/LinUCB/Dueling DDQN 检索策略；
- 地图轨迹算法可视化；
- 图片生成工具链。

也就是说，强化学习没有直接“改写 Qwen 大模型本体”，而是用于优化 Agent 在回答前的检索策略，使 Qwen 获得更准确的证据上下文。
