# PPT 核心代码展示

本文档用于放入 PPT。代码均为项目核心逻辑的精简展示版，保留关键路径和参数，并加入中文注释，方便答辩时解释。

整体流程：

```text
前端输入
  -> /api/chat
  -> Agent 主循环
  -> 工具调用 / RAG / 强化学习检索策略
  -> Ollama 调用 Qwen
  -> 返回回答 / 图片 / 地图展示 / 训练效果图
```

## 1. 核心功能代码展示

### 1.1 Ollama + Qwen 后端调用

来源文件：

- `backend/settings.py`
- `backend/llm_service.py`

```python
# backend/settings.py

# 核心1：读取当前 LLM 后端，默认使用 Ollama。
# 如果后续要换成本地 HF 模型，只需要改 LOCAL_LLM_BACKEND。
def get_llm_backend() -> str:
    return os.getenv("LOCAL_LLM_BACKEND", "ollama").strip().lower() or "ollama"


# 核心1：读取 Ollama 中实际运行的 Qwen 模型名称。
# 默认值 qwen3.5:latest 对应本地 Ollama 模型。
def get_ollama_model() -> str:
    return os.getenv("LOCAL_LLM_MODEL", "qwen3.5:latest").strip() or "qwen3.5:latest"


# 核心1：Ollama 服务地址，后端会请求 http://127.0.0.1:11434/api/chat。
def get_ollama_base_url() -> str:
    return os.getenv("LOCAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
```

```python
# backend/llm_service.py

def _chat_ollama(messages: list[dict[str, str]], model_name: str | None = None) -> str:
    """Ollama/Qwen 的底层请求函数。"""

    # 1. 构造 Ollama /api/chat 请求体
    body = {
        "model": model_name or get_ollama_model(),  # 默认调用 Qwen
        "messages": messages,                       # system/history/user 消息
        "stream": False,                            # 非流式，便于后端统一返回
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }

    # 2. 请求本地 Ollama 服务
    request = urllib.request.Request(
        f"{get_ollama_base_url()}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # 3. 读取 Qwen 返回结果
    with urllib.request.urlopen(request, timeout=get_ollama_timeout()) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return (payload.get("message") or {}).get("content", "").strip()


def run_messages(messages: list[dict[str, str]], query: str = "", context_block: str = "", model_id: str | None = None) -> str:
    """统一模型派发入口：Agent、RAG、上下文压缩都会通过这里调用 Qwen。"""

    option = resolve_model_option(model_id)
    backend = str(option.get("backend") or get_llm_backend())
    model_value = str(option.get("model") or "")

    if backend == "hf":
        return _chat_hf(messages, model_value)

    # 默认路径：Ollama + Qwen
    return _chat_ollama(messages, model_value or None)
```

说明：

- `settings.py` 负责模型配置；
- `llm_service.py` 负责真正调用 Qwen；
- 项目中 Agent、上下文压缩、普通聊天最终都会进入 `run_messages()`。

---

### 1.2 `/api/chat` 聊天主入口

来源文件：

- `backend/app.py`

```python
# backend/app.py

def chat_payload(
    query: str,
    session_id: str | None,
    top_k: int,
    attachment_name: str = "",
    attachment_text: str = "",
    model_id: str | None = None,
) -> dict[str, Any]:
    """聊天主入口：前端 /api/chat 请求进入这里。"""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # 1. 读取当前会话和历史消息
    session = get_session(session_id)
    history = session.get("messages", [])

    # 2. 上下文压缩：旧消息先压缩，再重新组装成较短上下文
    ConversationCompressor(model_id=model_id).maybe_compress(session["id"])
    assembled_context = ContextAssembler().assemble(session["id"], query, history_fallback=history)
    history = assembled_context["messages"]

    # 3. 文件附件：文件正文不直接塞进聊天框，而是作为隐藏上下文给模型
    file_context = ""
    visible_query = query
    if attachment_text.strip():
        label = attachment_name.strip() or "attached_file"
        visible_query = f"{query}\n\n[Attached file: {label}, {len(attachment_text.strip())} chars included for analysis]"
        file_context = f"\n\n[Attached file: {label}]\n{attachment_text.strip()}"

    # 4. 如果 Agent 开启，则走 Agent 工具调用；否则走普通 Qwen 回答
    if agent_enabled() and not file_context:
        agent_result = agent_chat(query, history=history, top_k=top_k, model_id=model_id)
        answer = agent_result["answer"]
        traces = agent_result.get("tool_traces", [])
    else:
        answer = chat_reply(query, history=history, context_block=file_context, model_id=model_id)
        traces = []

    # 5. 保存可见的用户问题和助手回答
    updated = append_turn(session["id"], visible_query, answer)

    return {
        "ok": True,
        "session": updated,
        "answer": answer,
        "toolTraces": traces,
    }
```

说明：

- 这个函数是聊天链路的总入口；
- 上下文压缩、文件读取、Agent/Qwen 调用都从这里串起来；
- 前端输入框最终会变成这里的 `query`。

---

### 1.3 上下文压缩

来源文件：

- `backend/conversation_compressor.py`
- `backend/context_assembler.py`

```python
# backend/conversation_compressor.py

class ConversationCompressor:
    """把旧对话压缩成摘要，避免每次都把完整历史发给 Qwen。"""

    def maybe_compress(self, conversation_id: str) -> dict[str, Any]:
        # 只有开启压缩，并且 MySQL 可用时才执行
        if not context_compression_enabled() or not self.repository.mysql_available():
            return {"compressed": False, "reason": "disabled_or_no_mysql"}

        return self._compress(conversation_id)

    def _summarize_segment(self, segment: list[dict[str, Any]]) -> dict[str, Any]:
        # 把一段旧消息整理成 transcript
        transcript = "\n".join(
            f"[message_id={message['id']} role={message['role']}]\n{message.get('content', '')}"
            for message in segment
        )

        # 调用 Qwen 生成结构化摘要
        raw = run_messages(
            [
                {"role": "system", "content": COMPRESSION_PROMPT},
                {"role": "user", "content": transcript},
            ],
            query="Compress this conversation segment.",
            context_block="",
            model_id=self.model_id,
        )

        return _extract_json(raw)
```

```python
# backend/context_assembler.py

class ContextAssembler:
    """把 rolling state、相关摘要、最近消息重新组装成当前轮 prompt。"""

    def assemble(
        self,
        conversation_id: str,
        current_user_message: str,
        history_fallback: list[dict[str, Any]] | None = None,
        max_input_tokens: int | None = None,
    ) -> dict[str, Any]:
        # 1. 如果压缩关闭，则只裁剪最近历史
        if not context_compression_enabled() or not self.repository.mysql_available():
            messages = self._trim_messages(history_fallback or [], context_max_input_tokens())
            return {"messages": messages, "metadata": {"mode": "fallback"}}

        # 2. 读取原始消息、摘要和 rolling state
        messages = self.repository.fetch_messages(conversation_id)
        summaries = self.repository.fetch_summaries(conversation_id)
        state = self.repository.fetch_state(conversation_id)

        # 3. 根据当前问题找相关摘要和相关旧消息
        relevant = self.retriever.retrieve(current_user_message, summaries, messages)

        # 4. 按优先级组装给 Qwen 的上下文
        assembled = []
        if state and state.get("state_json"):
            assembled.append({
                "role": "system",
                "content": "[Conversation rolling state]\n" + json.dumps(state["state_json"], ensure_ascii=False),
            })

        for summary in relevant["summaries"]:
            assembled.append({
                "role": "system",
                "content": f"[Relevant summary]\n{summary['content']}",
            })

        return {"messages": assembled, "metadata": {"mode": "compressed"}}
```

说明：

- `ConversationCompressor` 负责“压缩旧消息”；
- `ContextAssembler` 负责“把摘要重新拿回来给 Qwen 用”；
- 这样长对话不会无限增长。

---

### 1.4 Agent 工具调用与图片生成

来源文件：

- `backend/agent_loop.py`
- `backend/tool_registry.py`
- `backend/image_quality.py`

```python
# backend/agent_loop.py

def should_generate_image(query: str) -> bool:
    """判断用户是不是明确要求生成图片。"""
    text = query or ""
    return bool(IMAGE_ACTION_RE.search(text)) and not bool(IMAGE_DISCUSSION_RE.search(text))


def agent_chat(query: str, history: list[dict[str, str]] | None = None, top_k: int | None = None, model_id: str | None = None) -> dict[str, Any]:
    """Agent 主循环：决定直接生成图片、调用工具，或让 Qwen 输出最终答案。"""

    tool_traces: list[ToolTrace] = []

    # 1. 如果用户明确要求生成图片，直接调用高级图片工具
    if should_generate_image(query):
        tool_input = {
            "prompt": query,
            "quality_mode": "balanced",
            "allow_retry": True,
        }
        tool_output = execute_tool("generate_image_advanced", tool_input)

        # 2. 工具结果会进入 trace，方便前端展示和后端调试
        trace = ToolTrace(
            id="tool_1",
            name="generate_image_advanced",
            input=tool_input,
            output=tool_output,
        )
        tool_traces.append(trace)

        return {
            "answer": "已生成图片。",
            "sources": [],
            "tool_traces": [{"id": trace.id, "name": trace.name, "input": trace.input, "output": trace.output}],
        }

    # 3. 普通 Agent 流程：把工具目录和历史消息发给 Qwen，让 Qwen 决定是否调用工具
    runtime_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    runtime_messages.extend(history or [])
    runtime_messages.append({"role": "system", "content": _tool_catalog_block(top_k or get_top_k())})
    runtime_messages.append({"role": "user", "content": query})

    raw_response = run_messages(runtime_messages, query=query, context_block="", model_id=model_id)
    action_type, payload = _parse_model_action(raw_response)

    if action_type == "final":
        return {"answer": payload.get("answer", raw_response), "tool_traces": []}
```

```python
# backend/tool_registry.py

# Agent 能调用的工具都注册在这里
TOOL_REGISTRY: dict[str, ToolSpec] = {
    "search_project_docs": ToolSpec(
        name="search_project_docs",
        description="搜索本地 RL/轨迹知识库。",
        input_schema={"query": "string", "top_k": "integer"},
        handler=_search_project_docs,
    ),

    "read_file": ToolSpec(
        name="read_file",
        description="读取工作区文件。",
        input_schema={"path": "string", "start_line": "integer", "end_line": "integer"},
        handler=_read_file,
    ),

    "generate_image_advanced": ToolSpec(
        name="generate_image_advanced",
        description="高级图片生成：prompt 改写、候选图生成、质量评分、失败重试。",
        input_schema={"prompt": "string", "quality_mode": "fast | balanced | high"},
        handler=_generate_image_advanced,
    ),
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """所有 Agent 工具调用都经过这里，便于统一管理和审计。"""
    spec = TOOL_REGISTRY.get(name)
    if not spec:
        return {"tool": name, "error": f"Unsupported tool: {name}"}
    return spec.handler(**tool_input)
```

```python
# backend/image_quality.py

class ImageGenerationQualityController:
    """高级图片生成控制器：prompt 改写 -> 批量生成 -> 质量评分 -> 必要时重试。"""

    def generate(self, prompt: str, quality_mode: str = "balanced", allow_retry: bool = True, **kwargs) -> dict[str, Any]:
        # 1. 改写 prompt，生成正向提示词、负向提示词和采样参数
        plan = self.rewriter.rewrite(prompt, quality_mode=quality_mode)

        # 2. 生成一批候选图
        artifacts = self.runner.generate(plan)

        # 3. 对候选图进行质量评估
        reports = [self.critic.evaluate_image(artifact, plan) for artifact in artifacts]

        # 4. 选择分数最高的图片返回
        best_artifact, best_report = sorted(
            zip(artifacts, reports),
            key=lambda item: item[1].score,
            reverse=True,
        )[0]

        return {
            "ok": True,
            "type": "image_result",
            "final_image": {
                "url": best_artifact.get("url", ""),
                "score": best_report.score,
            },
            "quality_report": best_report.to_dict(),
        }
```

---

### 1.5 对话框文字读取与沙盒授权

来源文件：

- `frontend/src/views/WorkspaceView.vue`
- `frontend/src/stores/appStore.js`
- `backend/workspace_security.py`

```vue
<!-- frontend/src/views/WorkspaceView.vue -->

<!-- 核心4：主聊天输入框。
     用户输入的文字绑定到 appState.queryInput，
     sendCurrentMessage() 会读取这个值并发送给后端。 -->
<el-input
  v-model="appState.queryInput"
  type="textarea"
  :rows="4"
  resize="none"
  placeholder="请输入你的问题"
  @keydown.ctrl.enter.prevent="onSend"
/>
```

```javascript
// frontend/src/stores/appStore.js

export async function sendCurrentMessage() {
  // 1. 读取输入框文字
  const query = appState.queryInput.trim();
  if (!query || appState.sending) return;

  // 2. 确保当前有会话
  await ensureActiveSession();

  // 3. 构造发给后端 /api/chat 的 payload
  const payload = {
    query,
    session_id: appState.sessionId,
    top_k: appState.topK,
    model_id: appState.selectedModelId,
    attachment_name: appState.composerAttachment?.name || "",
    attachment_text: appState.composerAttachment?.text || "",
  };

  // 4. 清空输入框，并标记正在发送
  appState.queryInput = "";
  appState.sending = true;

  // 5. 请求后端 Agent/Qwen
  const data = await api.chat(payload);

  // 6. 后端返回新历史后，刷新当前会话
  upsertSession({
    ...data.session,
    messages: data.history || [],
  });
}
```

```javascript
// frontend/src/stores/appStore.js

async function ensureSandboxApproval(scope, description) {
  // 1. 同一个浏览器窗口里，相同 scope 只需要确认一次
  if (appState.sandboxApprovals[scope]) return true;

  // 2. 第一次遇到该操作时弹窗确认
  const ok = window.confirm(`${description}\n\nAllow this operation for this window?`);
  if (!ok) return false;

  // 3. 保存授权结果
  appState.sandboxApprovals[scope] = true;
  saveSandboxApprovals(appState.sandboxApprovals);
  return true;
}
```

```python
# backend/workspace_security.py

def resolve_safe_path(relative_path: str = ".") -> Path:
    """文件沙盒：所有文件读取/修改都必须先经过这里。"""

    root = workspace_root()
    candidate = Path(relative_path)

    # 1. 相对路径会被限制在工作区内
    if not candidate.is_absolute():
        candidate = root / candidate

    # 2. 解析真实路径，防止 ../ 和符号链接逃逸
    resolved = candidate.expanduser().resolve()

    # 3. 如果最终路径不在工作区内，直接拒绝
    if not (resolved == root or root in resolved.parents):
        raise ValueError(f"Path is outside workspace root: {relative_path}")

    return resolved
```

---

### 1.6 强化学习检索环境与奖励

来源文件：

- `backend/retrieval_rl_env.py`

```python
# backend/retrieval_rl_env.py

# 离散检索动作：RL 学的不是直接改写 Qwen，
# 而是“回答前应该如何检索资料”。
ACTIONS: tuple[RetrievalAction, ...] = (
    RetrievalAction("baseline", "使用原始 query 和默认 top-k。", top_k=4),
    RetrievalAction("rl_focus", "强化学习方向检索。", top_k=4, query_suffixes=("reinforcement learning policy optimization",)),
    RetrievalAction("trajectory_focus", "轨迹方向检索。", top_k=4, query_suffixes=("trajectory planning trajectory optimization",)),
    RetrievalAction("paper_focus", "论文和 benchmark 方向检索。", top_k=5, query_suffixes=("paper method benchmark ablation",)),
    RetrievalAction("reward_focus", "奖励设计和偏好优化方向检索。", top_k=5, query_suffixes=("reward design preference optimization",)),
    RetrievalAction("broad_search", "更宽泛检索，拿更多证据。", top_k=7),
)


def features_for_query(query: str) -> list[float]:
    """把用户问题转成 RL 状态向量。"""
    lowered = query.lower()
    return [
        1.0 if "ppo" in lowered else 0.0,
        1.0 if "dqn" in lowered else 0.0,
        1.0 if "trajectory" in lowered or "轨迹" in query else 0.0,
        1.0 if "reward" in lowered or "奖励" in query else 0.0,
        1.0 if "paper" in lowered or "论文" in query else 0.0,
        min(len(query) / 120.0, 1.0),
    ]


class RetrievalRLEnv:
    """一个问题就是一个 episode：策略选择检索动作，环境返回 reward。"""

    def step(self, action_index: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        # 1. 当前问题
        example = self.examples[self.current_index]
        query = str(example.get("query") or example.get("task") or "").strip()

        # 2. 执行动作，改写检索 query
        action = ACTIONS[action_index]
        retrieval_query = compose_retrieval_query(query, action)

        # 3. 查询知识库
        results = search_knowledge(retrieval_query, top_k=action.top_k)
        combined_snippets = " ".join(item.get("snippet", "") for item in results)

        # 4. 计算三类奖励指标
        source_score = source_hit(results, example.get("expected_sources", []))
        topic_score = topic_hit(results, example.get("expected_topics", []))
        point_score = point_recall(combined_snippets, example.get("expected_points", []))

        # 5. 总 reward：找对来源 + 覆盖主题 + 召回答案点，同时惩罚过大的 top-k
        reward = (
            self.reward_weights["source_hit"] * source_score
            + self.reward_weights["topic_hit"] * topic_score
            + self.reward_weights["point_recall"] * point_score
            - 0.015 * max(action.top_k - 4, 0)
        )

        return features_for_query(query), reward, True, {
            "action": action.name,
            "source_hit": round(source_score, 4),
            "topic_hit": round(topic_score, 4),
            "point_recall": round(point_score, 4),
            "reward": round(reward, 4),
        }
```

---

### 1.7 线上 RL 策略选择

来源文件：

- `backend/retrieval_policy.py`
- `backend/tools.py`

```python
# backend/retrieval_policy.py

def _candidate_policy_paths() -> list[Path]:
    """按优先级寻找已经训练好的检索策略。"""
    return [
        ROOT / "outputs" / "retrieval_policy_linucb" / "retrieval_policy_linucb.json",
        ROOT / "outputs" / "retrieval_policy_dueling_ddqn" / "retrieval_policy_dueling_ddqn.pt",
        ROOT / "outputs" / "retrieval_policy_dpo_torch" / "retrieval_policy_dpo.pt",
        ROOT / "outputs" / "retrieval_policy_orpo_torch" / "retrieval_policy_orpo.pt",
        ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_ppo.pt",
    ]


def choose_retrieval_action(query: str, requested_top_k: int = 4) -> dict[str, Any]:
    """线上检索动作选择入口。"""

    # 1. 如果没开启 RL 策略，则使用 baseline
    if not retrieval_policy_enabled():
        action = ACTIONS[0]
        return {
            "algorithm": "baseline",
            "action": action.name,
            "retrieval_query": compose_retrieval_query(query, action),
            "top_k": requested_top_k,
        }

    # 2. 加载可用策略，选择动作
    for path in _candidate_policy_paths():
        loaded = _load_policy(str(path))
        if not loaded:
            continue

        algorithm, policy = loaded
        state = features_for_query(query)
        action_index, raw_scores = policy.choose(state)
        action = ACTIONS[action_index]

        return {
            "algorithm": algorithm,
            "action": action.name,
            "retrieval_query": compose_retrieval_query(query, action),
            "top_k": action.top_k,
            "scores": raw_scores,
        }
```

```python
# backend/tools.py

def search_project_docs(query: str, top_k: int = 4) -> dict[str, Any]:
    """Agent 的 RAG 工具：先用 RL 策略选检索动作，再查知识库。"""

    # 1. RL 策略选择 retrieval action
    policy = choose_retrieval_action(query, requested_top_k=top_k)

    # 2. 根据策略改写后的 query 和 top_k 搜索知识库
    retrieval_query = str(policy.get("retrieval_query") or query)
    retrieval_top_k = int(policy.get("top_k") or top_k)
    results = search_knowledge(retrieval_query, top_k=retrieval_top_k)

    # 3. 返回结果时带上 policy，方便前端/日志解释 RL 做了什么
    return {
        "tool": "search_project_docs",
        "query": query,
        "retrieval_query": retrieval_query,
        "policy": policy,
        "count": len(results),
        "results": results,
    }
```

---

## 2. 强化学习训练参数代码

### 2.1 PPO 训练参数

来源文件：

- `scripts/train_retrieval_policy_ppo.py`

```python
def parse_args() -> argparse.Namespace:
    """PPO 检索策略训练参数。"""
    parser = argparse.ArgumentParser(description="Train a PPO retrieval policy for trajectory/RL RAG.")

    # 训练数据：每条样本包含 query、expected_sources、expected_topics、expected_points
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")

    # 输出目录：保存 retrieval_policy_ppo.pt、training_trace.json、evaluation.json
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo")

    # PPO 外层更新次数；项目里训练 60 Epoch 时使用 --epochs 60
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=0, help="Alias for --updates")

    # 每次更新采样多少个检索 episode
    parser.add_argument("--rollout-size", type=int, default=32)

    # 每批 rollout 内部重复优化次数
    parser.add_argument("--ppo-epochs", type=int, default=4)

    # PPO minibatch 大小
    parser.add_argument("--minibatch-size", type=int, default=16)

    # PPO clipping 参数，限制策略更新幅度
    parser.add_argument("--clip-range", type=float, default=0.2)

    # value loss 和 entropy loss 权重
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)

    # 学习率和随机种子
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)

    # reward 权重：Source Hit 最高，其次 Topic Hit，最后 Point Recall
    parser.add_argument("--source-weight", type=float, default=0.5)
    parser.add_argument("--topic-weight", type=float, default=0.3)
    parser.add_argument("--point-weight", type=float, default=0.2)

    return parser.parse_args()
```

PPO 训练命令示例：

```powershell
python scripts\train_retrieval_policy_ppo.py `
  --epochs 60 `
  --output outputs\retrieval_policy_ppo_torch_60
```

---

### 2.2 DPO 训练参数

来源文件：

- `scripts/train_retrieval_policy_dpo.py`

```python
def parse_args() -> argparse.Namespace:
    """DPO 偏好训练参数。"""
    parser = argparse.ArgumentParser(description="DPO-train a retrieval policy from action preference pairs.")

    # 训练数据
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")

    # DPO 从 PPO checkpoint 初始化，继续做偏好优化
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_ppo.pt")

    # 输出 DPO checkpoint 和 dpo_pairs.jsonl
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch")

    # DPO 训练轮数
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    # beta 控制偏好约束强度
    parser.add_argument("--beta", type=float, default=0.4)

    # margin 决定 reward 差距多大才构成 chosen/rejected pair
    parser.add_argument("--margin", type=float, default=0.02)

    # SFT 稳定项，防止策略偏离过猛
    parser.add_argument("--sft-coef", type=float, default=0.15)

    # reward 权重
    parser.add_argument("--source-weight", type=float, default=0.5)
    parser.add_argument("--topic-weight", type=float, default=0.3)
    parser.add_argument("--point-weight", type=float, default=0.2)

    return parser.parse_args()
```

DPO 核心训练目标：

```python
# chosen 是高 reward 动作，rejected 是低 reward 动作
preference_logit = (chosen_logp - rejected_logp) - (ref_chosen - ref_rejected)

# DPO loss：让当前策略相比参考策略更偏向 chosen
dpo_loss = -F.logsigmoid(beta * preference_logit).mean()

# SFT loss：让策略保持对 chosen action 的监督学习稳定性
sft_loss = F.cross_entropy(logits, chosen)

loss = dpo_loss + sft_coef * sft_loss
```

---

### 2.3 ORPO 训练参数

来源文件：

- `scripts/train_retrieval_policy_orpo.py`

```python
def parse_args() -> argparse.Namespace:
    """ORPO 对照训练参数。"""
    parser = argparse.ArgumentParser(description="ORPO-train a retrieval policy from action preference pairs.")

    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch" / "dpo_pairs.jsonl")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_ppo.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_orpo_torch")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    # ORPO 的 odds-ratio 惩罚权重
    parser.add_argument("--odds-coef", type=float, default=0.25)

    return parser.parse_args()
```

ORPO 核心目标：

```python
# chosen/rejected action 的 log-odds 差距
chosen_log_odds = _log_odds(log_probs, chosen)
rejected_log_odds = _log_odds(log_probs, rejected)
odds_margin = chosen_log_odds - rejected_log_odds

# chosen action 的监督学习损失
sft_loss = F.cross_entropy(logits, chosen)

# odds-ratio loss：鼓励 chosen 的 odds 高于 rejected
odds_loss = -F.logsigmoid(odds_margin).mean()

loss = sft_loss + odds_coef * odds_loss
```

---

### 2.4 LinUCB 训练参数

来源文件：

- `scripts/train_retrieval_policy_linucb.py`

```python
def parse_args() -> argparse.Namespace:
    """LinUCB 上下文 bandit 参数。"""
    parser = argparse.ArgumentParser(description="Train a LinUCB contextual-bandit retrieval policy.")

    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch" / "dpo_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_linucb")

    # online 模式更符合“单步检索策略选择”
    parser.add_argument("--mode", choices=["online", "offline"], default="online")

    # online 交互 episode 数
    parser.add_argument("--episodes", type=int, default=420)

    # offline 拟合轮数
    parser.add_argument("--epochs", type=int, default=20)

    # alpha 越大，探索越强
    parser.add_argument("--alpha", type=float, default=0.12)

    # epsilon 用于 online 模式的随机探索
    parser.add_argument("--epsilon", type=float, default=0.08)

    # L2 正则，避免线性模型不稳定
    parser.add_argument("--l2", type=float, default=1.0)

    return parser.parse_args()
```

LinUCB 动作选择：

```python
def scores(self, state: list[float]) -> list[float]:
    x = np.asarray(state, dtype=np.float64)
    values = []

    for action_index in range(self.theta.shape[0]):
        # 预测 reward
        mean = float(self.theta[action_index] @ x)

        # 不确定性 bonus，用来鼓励探索
        uncertainty = float(np.sqrt(max(x @ self.a_inv[action_index] @ x, 0.0)))

        # LinUCB 分数 = 预测收益 + alpha * 不确定性
        values.append(mean + self.alpha * uncertainty)

    return values
```

---

### 2.5 Dueling Double DQN 训练参数

来源文件：

- `scripts/train_retrieval_policy_dueling_ddqn.py`

```python
def parse_args() -> argparse.Namespace:
    """Dueling Double DQN 检索策略参数。"""
    parser = argparse.ArgumentParser(description="Train a Dueling Double DQN retrieval policy from offline rewards.")

    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch" / "dpo_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_dueling_ddqn")

    # online/offline 两种训练方式
    parser.add_argument("--mode", choices=["online", "offline"], default="online")

    # online 交互 episode 数
    parser.add_argument("--episodes", type=int, default=520)

    # offline 训练 epoch 数
    parser.add_argument("--epochs", type=int, default=240)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)

    # epsilon-greedy 探索参数
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-end", type=float, default=0.06)
    parser.add_argument("--epsilon-decay", type=float, default=0.988)

    return parser.parse_args()
```

Dueling Q 网络结构：

```python
class DuelingQNetwork(nn.Module):
    """把 Q 值拆成状态价值 value 和动作优势 advantage。"""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        # 公共特征提取层
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
        )

        # value 分支：估计当前问题状态本身有多好
        self.value = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))

        # advantage 分支：估计每个检索动作相对其他动作的优势
        self.advantage = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, action_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(states)
        value = self.value(hidden)
        advantage = self.advantage(hidden)

        # Q(s,a) = V(s) + A(s,a) - mean(A)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

---

## 3. 前端地图 API 实现

### 3.1 前端地图 API 封装

来源文件：

- `frontend/src/api.js`

```javascript
// frontend/src/api.js

export const api = {
  // 读取最近保存的地图轨迹实验
  trajectories: (limit = 20) =>
    fetchJson(`/api/trajectories?limit=${encodeURIComponent(limit)}`),

  // 保存当前地图实验，包括 baseline、DQN/PPO、S3/RLTS/Mlsimp
  saveTrajectory: (payload) =>
    fetchJson("/api/trajectories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
```

---

### 3.2 OSRM 地图路线获取

来源文件：

- `frontend/src/views/PathDemoView.vue`

```javascript
// OSRM 公开路线 API，用来获取真实道路网络上的 baseline
const OSRM_URL = "https://router.project-osrm.org/route/v1/driving";


async function fetchRoadRoute() {
  // 1. 读取当前起点和终点
  const { start, end } = effectiveEndpoints.value;

  // 2. OSRM 要求经纬度顺序是 lng,lat
  const startLngLat = `${start[1]},${start[0]}`;
  const endLngLat = `${end[1]},${end[0]}`;

  // 3. overview=full 返回完整路线点；steps=true 返回转向步骤，供 DQN/PPO 展示使用
  const url = `${OSRM_URL}/${startLngLat};${endLngLat}?overview=full&geometries=geojson&steps=true`;

  // 4. 请求真实道路路线
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`OSRM 请求失败：${response.status}`);

  const payload = await response.json();
  const route = payload.routes?.[0];
  if (!route?.geometry?.coordinates?.length) {
    throw new Error("没有拿到可用路线。");
  }

  return route;
}
```

---

### 3.3 生成地图轨迹：baseline + DQN/PPO + S3/RLTS/Mlsimp

来源文件：

- `frontend/src/views/PathDemoView.vue`

```javascript
async function generateTrajectory() {
  // 1. 请求 OSRM，拿到真实道路 baseline
  const route = await fetchRoadRoute();

  // 2. OSRM 返回 lng,lat；Leaflet 绘图使用 lat,lng，所以这里要反转
  routeGeometry.value = route.geometry.coordinates.map(([lng, lat]) => [lat, lng]);

  // 3. 保存路线距离和时长
  routeDistanceKm.value = Number((route.distance / 1000).toFixed(2));
  routeDurationMin.value = Number((route.duration / 60).toFixed(1));

  // 4. 保存 OSRM steps，后续用于构造 DQN 决策点和 PPO 阶段
  routeSteps.value = route.legs?.[0]?.steps || [];

  // 5. 构造 DQN/PPO 展示数据
  dqnDecisionPoints.value = buildDqnDecisionPoints(routeSteps.value);
  ppoStages.value = buildPpoStages(routeSteps.value);

  // 6. 构造 S3/RLTS/Mlsimp 压缩展示数据
  buildCompressionStrategies();

  // 7. 渲染地图
  routeReady.value = true;
  renderRouteEndpoints();
  renderRoute();

  // 8. 保存到数据库
  await saveCurrentTrajectory();
}
```

---

### 3.4 DQN / PPO 地图展示

来源文件：

- `frontend/src/views/PathDemoView.vue`

```javascript
function buildDqnDecisionPoints(steps) {
  // DQN：把 OSRM 的转向步骤转换成离散决策点
  const total = routeGeometry.value.length || 1;

  return steps
    .filter((step) => step.maneuver?.location)
    .map((step, index) => {
      const coords = [step.maneuver.location[1], step.maneuver.location[0]];
      return {
        coords,
        routeIndex: Math.min(total - 1, Math.round(((index + 1) / steps.length) * (total - 1))),
        note: `${step.maneuver.type || "continue"}，距离 ${(step.distance / 1000).toFixed(2)} km`,
      };
    });
}


function buildPpoStages(steps) {
  // PPO：把整条路线分成若干阶段，展示阶段策略
  const chunkSize = Math.max(1, Math.ceil(steps.length / 4));
  const groups = [];

  for (let index = 0; index < steps.length; index += chunkSize) {
    groups.push(steps.slice(index, index + chunkSize));
  }

  return groups.map((group, index) => {
    const last = group[group.length - 1];
    const distance = group.reduce((sum, item) => sum + (item.distance || 0), 0);

    return {
      coords: [last.maneuver.location[1], last.maneuver.location[0]],
      note: `阶段 ${index + 1}，包含 ${group.length} 个道路步骤，距离 ${(distance / 1000).toFixed(2)} km`,
    };
  });
}
```

---

### 3.5 S3 / RLTS / MLsimp 压缩展示

来源文件：

- `frontend/src/views/PathDemoView.vue`

```javascript
function buildCompressionStrategies() {
  // 输入：完整路线点
  const total = routeGeometry.value.length;
  if (!total) return;

  // 转向点通常是轨迹压缩中比较重要的点
  const turnIndices = dqnDecisionPoints.value.map((item) => item.routeIndex);

  // 去重 + 排序 + 去掉越界点
  const uniqueSorted = (values) =>
    Array.from(new Set(values.filter((value) => value >= 0 && value < total))).sort((a, b) => a - b);

  // S3：保留端点、转向点、少量均匀骨架点
  const s3 = uniqueSorted([
    0,
    total - 1,
    ...turnIndices,
    ...Array.from({ length: 8 }, (_, idx) => Math.round((idx * (total - 1)) / 7)),
  ]);

  // RLTS：更密集地保留路口附近点，强调局部行为变化
  const rlts = uniqueSorted([
    0,
    total - 1,
    ...turnIndices.flatMap((idx) => [idx - 1, idx, idx + 1]),
    ...Array.from({ length: 10 }, (_, idx) => Math.round((idx * (total - 1)) / 9)),
  ]);

  // MLsimp：保留端点、部分转向点和代表性长段变化点
  const mlsimp = uniqueSorted([
    0,
    total - 1,
    ...turnIndices.filter((_, index) => index % 2 === 0),
    ...Array.from({ length: 6 }, (_, idx) => Math.round((idx * (total - 1)) / 5)),
  ]);

  compressionStrategies.value = {
    s3: { indices: s3, summary: "S3 保留端点、明显转折点和少量均匀骨架点。" },
    rlts: { indices: rlts, summary: "RLTS 更关注路口附近的关键点和局部行为变化。" },
    mlsimp: { indices: mlsimp, summary: "MLsimp 更偏向端点和代表性长段变化点。" },
  };
}
```

---

### 3.6 保存地图实验到后端数据库

来源文件：

- `frontend/src/views/PathDemoView.vue`
- `backend/app.py`
- `backend/db.py`

```javascript
// frontend/src/views/PathDemoView.vue

async function saveCurrentTrajectory() {
  // 把当前地图实验结果发给后端
  const data = await api.saveTrajectory({
    trajectory_type: effectiveEndpoints.value.type,
    scenario_id: effectiveEndpoints.value.scenarioId,
    scenario_label: effectiveEndpoints.value.scenarioLabel,

    // 如果当前是强化学习模式，保存 DQN/PPO/baseline
    rl_method: methodCategory.value === "rl" ? selectedRlMethod.value : "",

    // 如果当前是压缩模式，保存 S3/RLTS/Mlsimp
    compression_method: methodCategory.value === "compression" ? selectedCompressionMethod.value : "",

    map_provider: "OpenStreetMap",
    route_provider: "OSRM",
    start: effectiveEndpoints.value.start,
    end: effectiveEndpoints.value.end,
    distance_km: routeDistanceKm.value,
    duration_min: routeDurationMin.value,
    route_geometry: routeGeometry.value,
    compression: compressionStrategies.value,
  });

  savedTrajectoryId.value = data.trajectoryId || null;
}
```

```python
# backend/app.py

def save_trajectory_payload(
    trajectory_type: str,
    scenario_id: str,
    scenario_label: str,
    rl_method: str,
    compression_method: str,
    map_provider: str,
    route_provider: str,
    start: list[float],
    end: list[float],
    distance_km: float,
    duration_min: float,
    route_geometry: list[list[float]],
    compression: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """地图实验保存入口。"""

    trajectory_id = save_trajectory_run(
        trajectory_type=trajectory_type,
        scenario_id=scenario_id,
        scenario_label=scenario_label,
        rl_method=rl_method,
        compression_method=compression_method,
        map_provider=map_provider,
        route_provider=route_provider,
        start_coords=start,
        end_coords=end,
        distance_km=distance_km,
        duration_min=duration_min,
        route_geometry=route_geometry,
        compression=compression or {},
        metadata=metadata or {},
    )

    return {"ok": True, "trajectoryId": trajectory_id}
```

```python
# backend/db.py

def save_trajectory_run(...):
    """把地图轨迹和算法展示结果写入 MySQL。"""

    cursor.execute(
        """
        INSERT INTO trajectory_runs (
            trajectory_type,
            scenario_id,
            scenario_label,
            rl_method,
            compression_method,
            map_provider,
            route_provider,
            start_lat,
            start_lng,
            end_lat,
            end_lng,
            distance_km,
            duration_min,
            route_geometry_json,
            compression_json,
            metadata_json,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            trajectory_type,
            scenario_id,
            scenario_label,
            rl_method,
            compression_method,
            map_provider,
            route_provider,
            float(start_coords[0]),
            float(start_coords[1]),
            float(end_coords[0]),
            float(end_coords[1]),
            float(distance_km),
            float(duration_min),
            json.dumps(route_geometry, ensure_ascii=False),
            json.dumps(compression or {}, ensure_ascii=False),
            json.dumps(metadata or {}, ensure_ascii=False),
            datetime.now(),
        ),
    )
```

## 4. PPT 中可直接讲的总结

```text
1. Qwen 不是孤立回答，而是被 Agent 包装在工具调用流程中。
2. 强化学习不是微调 Qwen 本体，而是训练“检索动作选择策略”。
3. PPO/DPO/ORPO/LinUCB/DDQN 的目标是让 RAG 更容易命中正确来源和正确主题。
4. 前端地图用 OSRM 获取真实路线，再叠加 DQN/PPO 决策点和 S3/RLTS/MLsimp 压缩点。
5. 所有结果可以写入数据库，便于展示最近实验记录。
```
