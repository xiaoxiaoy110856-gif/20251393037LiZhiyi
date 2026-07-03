# 本项目 RAG 与 LinUCB 详细分析和路线

## 1. 一句话结论

本项目中：

- **RAG** 负责“从本地论文知识库中查证据”。
- **LinUCB** 负责“在查证据之前，判断应该怎么查”。
- **Qwen/Ollama** 负责“读取用户问题和检索证据，生成最终回答”。

也就是说，最终系统不是让 Qwen 直接凭空回答，而是：

```text
用户问题
  -> LinUCB 选择检索策略
  -> RAG 搜索本地论文知识库
  -> Qwen 基于证据生成回答
```

## 2. 总体架构路线

```text
前端 WorkspaceView.vue
  -> appStore.js / api.js
  -> /api/chat
  -> backend/app.py
  -> backend/context/ 上下文压缩与装配
  -> backend/agent/ Agent 判断是否需要调用工具
  -> backend/agent/tools.py::search_project_docs()
  -> backend/retrieval/policy.py::choose_retrieval_action()
  -> LinUCB 选择 action
  -> backend/retrieval/knowledge_store.py::search_knowledge()
  -> RAG 返回论文证据
  -> backend/llm/service.py::run_messages()
  -> Ollama / Qwen 生成最终回答
```

更直观地说：

```text
用户问题
  ↓
Qwen 作为 Agent Planner 判断是否需要查资料
  ↓
search_project_docs 工具
  ↓
LinUCB 选择检索动作
  ↓
RAG 按优化后的 query/top_k 搜本地论文库
  ↓
Qwen 读取 evidence 后回答
```

## 3. RAG 在项目中用在哪里

### 3.1 RAG 的核心职责

RAG 的作用是：

```text
把用户问题变成知识库检索
  -> 找到相关论文、笔记、代码片段或知识块
  -> 整理成 evidence/context
  -> 交给 Qwen 生成回答
```

它解决的问题是：

- Qwen 本身不知道本地论文库里有哪些内容；
- 用户问到 PPO、DPO、ORPO、LinUCB、轨迹压缩、路径规划等项目资料时，需要有本地证据；
- 回答不能只靠模型记忆，而要尽量引用项目知识库里的论文和材料。

### 3.2 RAG 核心代码位置

RAG 核心代码在：

```text
backend/retrieval/knowledge_store.py
```

最核心函数：

```python
def search_knowledge(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    ...
```

作用：

- 接收用户问题或 LinUCB 改写后的检索 query；
- 使用知识库索引进行向量检索；
- 检索失败或禁用索引时，走 fallback 词法检索；
- 对结果进行 rerank；
- 返回 top-k 条证据。

RAG 结果会包含：

```text
title      # 文档或论文标题
topics     # 主题标签
path       # 来源路径
section    # 文档段落
doc_type   # 文档类型
snippet    # 证据片段
score      # 检索得分
```

### 3.3 RAG 检索流程

`search_knowledge()` 内部主要做这些事：

```text
query
  -> expand_query(query)
  -> 向量检索 get_cached_index().as_retriever()
  -> 合并重复 path/section
  -> _rerank_results()
  -> 如果向量索引不可用，走 _fallback_search()
  -> 返回 top-k evidence
```

其中 `_rerank_results()` 会综合多个加分项：

```text
baseScore       # 向量检索原始分
lexicalBoost    # 关键词命中加分
focusBoost      # 当前研究重点加分
docTypeBoost    # 文档类型加分
aliasBoost      # 论文/算法别名命中加分
```

所以本项目的 RAG 不是简单关键词搜索，而是：

```text
向量检索 + query rewrite + 词法增强 + 主题/文档类型/别名 rerank
```

### 3.4 RAG 如何变成 Qwen 的上下文

RAG 检索到的结果会被整理成 context block：

```text
backend/retrieval/knowledge_store.py::build_context_block()
```

格式类似：

```text
[Evidence 1]
Title: ...
Section: ...
Topics: ...
Path: ...
Excerpt: ...
```

然后这个 context block 会进入 Qwen 的系统提示或工具结果里，让 Qwen 基于证据回答。

### 3.5 RAG 的三条使用路径

#### 路径 A：Agent 工具调用路径，最重要

这是当前项目里最核心的 RAG 使用方式：

```text
用户问题
  -> /api/chat
  -> backend/app.py::chat_payload()
  -> backend/agent/loop.py::agent_chat()
  -> Qwen 判断需要查资料
  -> 调用 search_project_docs
  -> backend/agent/tools.py::search_project_docs()
  -> search_knowledge()
  -> 返回证据
  -> Qwen 最终回答
```

对应代码：

```text
backend/agent/tools.py::search_project_docs()
```

其中核心逻辑是：

```python
policy = choose_retrieval_action(query, requested_top_k=top_k)
retrieval_query = str(policy.get("retrieval_query") or query)
retrieval_top_k = int(policy.get("top_k") or top_k)
results = search_knowledge(retrieval_query, top_k=retrieval_top_k)
```

这说明：

- Agent 不是直接搜原始 query；
- 它会先让 LinUCB 选择检索动作；
- 然后 RAG 按 LinUCB 优化后的 query/top_k 搜知识库。

#### 路径 B：普通聊天 RAG 路径

如果 Agent 没开启，但是 RAG 开启，则聊天接口会直接检索知识库：

```text
backend/app.py::chat_payload()
```

对应逻辑：

```python
if rag_enabled() and not agent_enabled():
    results = search_knowledge(query, top_k=top_k)
    context_block = build_context_block(results) if results else ""
```

这条路径适合普通 RAG 问答。

#### 路径 C：调试检索接口

项目也提供了直接搜索知识库的接口：

```text
backend/app.py::search_payload()
```

作用：

- 用于调试 RAG 检索结果；
- 不一定走完整 Agent；
- 可以直接看某个 query 能搜出哪些 evidence。

## 4. LinUCB 在项目中用在哪里

### 4.1 LinUCB 的定位

LinUCB 位于 RAG 前面，属于：

```text
检索策略层
```

它不负责生成回答，也不直接修改 Qwen。

它负责：

```text
给定用户 query，选择最合适的 retrieval action。
```

换句话说：

```text
RAG 负责“查”；
LinUCB 负责“决定怎么查”。
```

### 4.2 LinUCB 为什么适合本项目

本项目的检索任务天然是单步决策：

```text
一个用户问题
  -> 选择一个检索动作
  -> 得到检索结果
  -> 计算 Source Hit / Topic Hit / Point Recall
```

这种任务非常适合 Contextual Bandit。

LinUCB 的优势是：

- 训练快；
- 小数据也能稳定；
- 可解释；
- 每个 action 都能看到分数；
- 适合“单步选择 retrieval action”。

### 4.3 LinUCB 核心代码位置

线上策略选择：

```text
backend/retrieval/policy.py
```

训练环境与动作定义：

```text
backend/retrieval/rl_env.py
```

训练脚本：

```text
scripts/train_retrieval_policy_linucb.py
```

训练输出：

```text
outputs/retrieval_policy_linucb/retrieval_policy_linucb.json
outputs/retrieval_policy_linucb/evaluation.json
outputs/retrieval_policy_linucb/training_trace.json
```

### 4.4 LinUCB 使用的 action

LinUCB 不是从无限动作里选，而是在项目预先定义的离散检索动作中选择。

动作定义在：

```text
backend/retrieval/rl_env.py::ACTIONS
```

当前包括：

| Action | 作用 | top_k |
|---|---|---|
| `baseline` | 原始 query，普通检索 | 4 |
| `rl_focus` | 偏强化学习术语 | 4 |
| `trajectory_focus` | 偏轨迹分析/轨迹优化 | 4 |
| `paper_focus` | 偏论文、方法、benchmark、实验 | 5 |
| `compression_focus` | 偏轨迹压缩/简化 | 5 |
| `planning_focus` | 偏路径规划、轨迹优化 | 5 |
| `similarity_focus` | 偏轨迹相似度、子轨迹搜索 | 5 |
| `reward_focus` | 偏奖励设计、偏好优化、逆强化学习 | 5 |
| `broad_search` | 扩大检索范围，拿更多证据 | 7 |

例如用户问：

```text
PPO 在 trajectory planning 中通常扮演什么角色？
```

LinUCB 可能选择：

```text
paper_focus
```

因为这个问题更像论文方法总结，需要找论文和 benchmark。

### 4.5 LinUCB 使用的 query 特征

LinUCB 会先把 query 转成状态特征：

```text
backend/retrieval/rl_env.py::features_for_query()
```

特征包括：

| 特征 | 含义 |
|---|---|
| 是否包含 `ppo` | 判断是否是 PPO 相关问题 |
| 是否包含 `dqn` | 判断是否是 DQN 相关问题 |
| 是否包含 `sac` | 判断是否是 SAC 相关问题 |
| 是否是压缩/简化问题 | `compression`、`simplification`、`压缩`、`简化` |
| 是否是轨迹问题 | `trajectory`、`轨迹` |
| 是否是规划/优化问题 | `planning`、`optimization`、`规划`、`优化` |
| 是否是相似度问题 | `similarity`、`subtrajectory`、`Frechet`、`Hausdorff`、`DTW` |
| 是否是奖励/偏好问题 | `reward`、`preference`、`奖励`、`偏好` |
| 是否是论文/实验问题 | `paper`、`survey`、`benchmark`、`experiment`、`论文`、`实验` |
| query 长度 | 归一化到 0-1 |
| 中文字符数量 | 归一化到 0-1 |
| 英文字符数量 | 归一化到 0-1 |

所以 LinUCB 的输入不是完整自然语言，而是一个紧凑的 query 特征向量。

### 4.6 LinUCB 的选择公式

LinUCB 每个 action 都有一个线性奖励预测器。

对某个 action 的分数是：

\[
\mathrm{score}(a) = \theta_a^\top x + \alpha \sqrt{x^\top A_a^{-1}x}
\]

含义：

- \(x\)：当前用户问题的特征向量；
- \(\theta_a^\top x\)：该 action 的预测收益；
- \(\sqrt{x^\top A_a^{-1}x}\)：不确定性；
- \(\alpha\)：探索强度；
- 最终选择 score 最高的 action。

对应代码：

```text
backend/retrieval/policy.py::_LinUCBPolicy.choose()
```

简化逻辑：

```python
mean = sum(weight * value for weight, value in zip(theta, state))
uncertainty = sqrt(state.T @ A_inv @ state)
score = mean + alpha * uncertainty
```

这就是 LinUCB 的核心：

```text
既利用已经学到的高收益动作，也给不确定动作保留探索机会。
```

## 5. LinUCB 的训练路线

### 5.1 训练数据

训练样本来自检索评估数据：

```text
training_data/retrieval_rl_eval_extended.jsonl
```

每条样本通常包含：

```text
query
expected_sources
expected_topics
expected_points
```

含义：

- `query`：用户问题；
- `expected_sources`：希望命中的论文或资料来源；
- `expected_topics`：希望覆盖的主题；
- `expected_points`：希望召回的答案要点。

### 5.2 训练环境

训练环境在：

```text
backend/retrieval/rl_env.py::RetrievalRLEnv
```

一个 episode 的过程：

```text
选择一个问题
  -> 提取 query 特征
  -> 策略选择 action
  -> 根据 action 改写 retrieval_query
  -> RAG 检索知识库
  -> 计算 Source Hit / Topic Hit / Point Recall
  -> 得到 reward
  -> 更新 LinUCB 参数
```

### 5.3 reward 设计

项目中的 reward 是：

\[
R = 0.5 \cdot \mathrm{SourceHit}
  + 0.3 \cdot \mathrm{TopicHit}
  + 0.2 \cdot \mathrm{PointRecall}
  - 0.015 \cdot \max(\mathrm{top\_k} - 4, 0)
\]

含义：

- Source Hit 权重最高：优先找对论文/来源；
- Topic Hit 次之：保证覆盖正确研究主题；
- Point Recall 再次：召回答案要点；
- 对过大的 top-k 做轻微惩罚：防止策略只靠“多搜”提升分数。

这套 reward 直接服务于你的答辩指标：

```text
Source Hit
Topic Hit
Point Recall
```

### 5.4 训练参数

当前 LinUCB 训练参数来自：

```text
scripts/train_retrieval_policy_linucb.py
```

主要参数：

```text
mode = online
episodes = 420
alpha = 0.12
epsilon = 0.08
l2 = 1.0
source_weight = 0.5
topic_weight = 0.3
point_weight = 0.2
```

解释：

- `online`：按 episode 交互式更新；
- `episodes=420`：训练 420 次单步检索决策；
- `alpha=0.12`：控制 UCB 探索强度；
- `epsilon=0.08`：保留少量随机探索；
- `l2=1.0`：正则化，避免线性模型不稳定。

### 5.5 训练输出

训练完成后输出：

```text
outputs/retrieval_policy_linucb/retrieval_policy_linucb.json
```

里面保存：

```text
theta       # 每个 action 的线性参数
a_inv       # 每个 action 的逆矩阵，用于不确定性计算
alpha       # 探索强度
actions     # 动作定义
reward_weights
mode
episodes
```

线上系统会读取这个 json 作为 LinUCB 策略。

## 6. LinUCB 的上线运行路线

### 6.1 策略加载

线上策略入口：

```text
backend/retrieval/policy.py::_candidate_policy_paths()
```

当前候选策略加载顺序中，LinUCB 排在最前：

```text
outputs/retrieval_policy_linucb/retrieval_policy_linucb.json
outputs/retrieval_policy_dueling_ddqn/retrieval_policy_dueling_ddqn.pt
outputs/retrieval_policy_dpo_torch/retrieval_policy_dpo.pt
outputs/retrieval_policy_orpo_torch/retrieval_policy_orpo.pt
outputs/retrieval_policy_ppo_torch_60/retrieval_policy_ppo.pt
...
```

这说明：

```text
只要 LinUCB checkpoint 存在且可加载，线上优先使用 LinUCB。
```

### 6.2 策略决策

线上决策入口：

```text
backend/retrieval/policy.py::choose_retrieval_action()
```

核心步骤：

```text
query
  -> features_for_query(query)
  -> 加载 LinUCB checkpoint
  -> policy.choose(state)
  -> 得到 action_index
  -> 取 ACTIONS[action_index]
  -> compose_retrieval_query(query, action)
  -> 返回 retrieval_query / top_k / scores
```

返回内容包含：

```text
algorithm
checkpoint
action
action_description
original_query
retrieval_query
top_k
scores
```

所以前端或日志可以看到：

```text
LinUCB 选择了哪个 action
每个 action 的分数是多少
最终用于 RAG 的 query 是什么
```

### 6.3 和 RAG 的结合点

LinUCB 和 RAG 真正结合在：

```text
backend/agent/tools.py::search_project_docs()
```

路线：

```text
search_project_docs(query)
  -> choose_retrieval_action(query)
  -> retrieval_query = policy["retrieval_query"]
  -> retrieval_top_k = policy["top_k"]
  -> search_knowledge(retrieval_query, top_k=retrieval_top_k)
```

这是本项目最关键的一段：

```text
LinUCB 不替代 RAG，而是优化 RAG 的输入。
```

## 7. 当前效果数据

当前评估文件：

```text
outputs/retrieval_policy_linucb/evaluation.json
```

结果：

| 指标 | Baseline | LinUCB | 提升 |
|---|---:|---:|---:|
| Average Reward | 0.5111 | 0.6963 | +0.1852 |
| Source Hit | 0.3056 | 0.6481 | +34.25pp |
| Topic Hit | 0.8426 | 0.9329 | +9.03pp |
| Point Recall | 0.5278 | 0.5370 | +0.92pp |

最能体现强化学习收益的是：

```text
Source Hit +34.25pp
Topic Hit +9.03pp
```

这说明 LinUCB 主要提升的是：

- 更容易找到正确论文/来源；
- 更容易覆盖正确研究主题；
- 让后续 Qwen 回答更有证据。

## 8. RAG 与 LinUCB 的关系

### 8.1 它们不是同一个东西

| 模块 | 解决什么问题 |
|---|---|
| RAG | 从知识库里找证据 |
| LinUCB | 选择最合适的检索动作 |
| Qwen | 基于问题和证据生成回答 |

RAG 是检索系统，LinUCB 是检索策略。

### 8.2 它们的关系

```text
LinUCB
  -> 输出 retrieval action
  -> 改写 retrieval_query / top_k
  -> 交给 RAG

RAG
  -> 根据 retrieval_query 搜知识库
  -> 返回 evidence
  -> 交给 Qwen
```

### 8.3 为什么不是直接让 Qwen 判断怎么搜

可以让 Qwen 判断，但不稳定：

- Qwen 可能每次选择不一致；
- 不容易量化检索策略是否提升；
- 不容易和 baseline 做公平对比；
- 不容易画出 Source Hit / Topic Hit 的提升曲线。

LinUCB 的优势是：

- 策略明确；
- 可评估；
- 可重复；
- 有训练数据和指标支撑；
- 可以和 baseline、PPO、DPO、ORPO、DDQN 做同一批问题上的对照。

## 9. 与 PPO/DPO/ORPO 的关系

在当前项目的检索策略实验里：

```text
LinUCB、DDQN、PPO、DPO、ORPO 都作为策略方法做对照。
```

但最终系统分工应该这样讲：

| 方法 | 更适合的位置 | 作用 |
|---|---|---|
| LinUCB | 检索策略层 | 给 query 选择 retrieval action |
| DDQN | 检索策略神经网络对照 | 学离散 action，但需要更多样本 |
| PPO | Agent 多步行为层 | 学什么时候搜、什么时候读文件、什么时候结束 |
| DPO | 回答偏好层 | 让回答更重证据、更少幻觉 |
| ORPO | 轻量偏好优化层 | 用较低成本做偏好优化对照 |

所以答辩时不要说：

```text
PPO、DPO、ORPO、LinUCB 完全平行。
```

更准确说：

```text
它们在实验中可以对照，但在系统中负责不同层级。
LinUCB 当前在检索策略任务上的综合效果最好，因此作为上线主策略。
```

## 10. 可以放 PPT 的核心表达

### 10.1 一句话版

> RAG 负责从本地论文知识库中检索证据，LinUCB 位于 RAG 前面，负责根据用户问题选择最合适的检索动作，从而提高 Source Hit 和 Topic Hit，最后 Qwen 基于检索证据生成回答。

### 10.2 稍微详细版

> 本项目将一次 RAG 检索建模为 contextual bandit 问题。用户问题先被转成 query 特征，LinUCB 在多个 retrieval action 中选择最优动作，例如偏强化学习、偏轨迹、偏论文或扩大检索范围。随后 RAG 使用优化后的 retrieval query 检索本地论文库，并将证据交给 Qwen 生成回答。评估结果表明，LinUCB 相比 baseline 在 Source Hit 上提升 34.25pp，在 Topic Hit 上提升 9.03pp。

### 10.3 流程图版

```text
用户问题
  ↓
features_for_query()
  ↓
LinUCB 计算每个 action 的 UCB 分数
  ↓
选择 action
  ↓
compose_retrieval_query()
  ↓
RAG search_knowledge()
  ↓
build_context_block()
  ↓
Qwen/Ollama 生成回答
```

## 11. 代码位置总览

| 功能 | 文件位置 |
|---|---|
| 聊天 API 入口 | `backend/app.py` |
| Agent 主循环 | `backend/agent/loop.py` |
| Agent RAG 工具 | `backend/agent/tools.py` |
| RAG 搜索实现 | `backend/retrieval/knowledge_store.py` |
| LinUCB 上线策略 | `backend/retrieval/policy.py` |
| LinUCB 动作/特征/reward 环境 | `backend/retrieval/rl_env.py` |
| LinUCB 训练脚本 | `scripts/train_retrieval_policy_linucb.py` |
| LinUCB 训练输出 | `outputs/retrieval_policy_linucb/` |
| Qwen/Ollama 调用 | `backend/llm/service.py` |
| 前端聊天页面 | `frontend/src/views/WorkspaceView.vue` |
| 前端训练效果图 | `frontend/src/views/TrainingView.vue` |

## 12. 最终答辩口径

可以这样讲：

> 本项目不是简单地把 Qwen 接到一个知识库上，而是在 RAG 前面增加了一个强化学习检索策略层。RAG 负责检索本地论文证据，LinUCB 负责根据用户问题选择最合适的检索动作。这样做的好处是，检索策略可以被训练和评估，而不是完全依赖模型临场判断。最终结果显示，LinUCB 在 Source Hit 和 Topic Hit 上都明显优于 baseline，因此它是当前系统推荐上线的检索策略。

再简短一点：

> RAG 让 Agent 能查资料，LinUCB 让 Agent 更会查资料。
