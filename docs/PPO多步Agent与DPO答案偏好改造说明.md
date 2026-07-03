# PPO 多步 Agent 与 DPO 答案偏好改造说明

## 1. 改造结论

本次改造把 PPO / DPO 从原来的“检索动作选择层”上移到更适合它们的位置：

```text
PPO：负责多步 Agent 决策流程
DPO：负责答案偏好质量训练
LinUCB：本次不改，继续作为检索动作选择层历史对照
```

这样划分后，PPO / DPO 的优势会更清楚：

```text
LinUCB 适合单步选择“怎么搜”；
PPO 适合多步选择“Agent 下一步做什么”；
DPO 适合偏好学习“哪个回答更可信”。
```

## 2. PPO 改成了什么

新增代码：

```text
backend/agent/rl_training.py
scripts/train_agent_policy_ppo.py
```

新的 PPO 不再只做：

```text
query -> retrieval action -> reward
```

而是训练完整 Agent 行为链：

```text
用户问题
  -> 读取文件 read_file
  -> RAG 检索 search_project_docs
  -> 证据重排 rerank_evidence
  -> 上下文压缩 compress_context
  -> 再次检索 second_search
  -> 生成回答 generate_answer
  -> 根据答案要点、工具使用、证据数量给 reward
```

PPO 动作集合：

```text
read_file
search_project_docs
rerank_evidence
compress_context
second_search
generate_answer
```

状态特征包含：

```text
问题是否涉及 PPO / DPO / 强化学习 / 轨迹
是否包含文件路径
当前步数
是否已经读文件
证据数量
是否已重排
是否已压缩
是否已二次检索
当前上下文要点召回率
预期工具命中率
重复动作次数
剩余步数
```

奖励由几部分组成：

```text
答案要点召回 answer_point_recall
预期工具命中 expected_tool_hit
证据数量 evidence_count
上下文压缩后的有效信息密度
二次检索带来的新增证据
重复动作和无效动作惩罚
```

## 3. PPO 为什么比原来合理

原来的 PPO 和 LinUCB 都在做“单步检索动作选择”，所以 PPO 的优势发挥不出来。

现在 PPO 面对的是多步决策：

```text
先读文件还是先检索？
检索后要不要重排？
上下文太长要不要压缩？
证据不足要不要二次检索？
什么时候停止并生成回答？
```

这才是 PPO 更擅长的强化学习场景。

## 4. PPO 训练命令

```bash
python scripts/train_agent_policy_ppo.py
```

默认参数：

```text
updates = 60
rollout_episodes = 12
max_steps = 6
ppo_epochs = 4
clip_range = 0.2
gamma = 0.95
lr = 3e-4
entropy_coef = 0.015
imitation_coef = 0.3
```

训练输出：

```text
outputs/agent_policy_ppo_multi_step/agent_policy_ppo.pt
outputs/agent_policy_ppo_multi_step/metrics.json
outputs/agent_policy_ppo_multi_step/evaluation.json
outputs/agent_policy_ppo_multi_step/training_trace.json
```

当前训练结果：

```text
Baseline average reward: 1.0239
PPO average reward:      1.7469
Reward gain:             +0.7230

Baseline answer point recall: 0.4167
PPO answer point recall:      0.5278
Recall gain:                  +0.1111

Expected tool hit:
Baseline = 1.0000
PPO      = 1.0000
```

工具命中没有提升，是因为 baseline 在当前 3 条评测样本上已经把预期工具都命中了；PPO 的提升主要体现在多步证据链、压缩、二次检索和最终 reward。

## 5. DPO 改成了什么

新增代码：

```text
backend/agent/answer_preference.py
scripts/train_answer_preference_dpo.py
```

新增/扩充数据：

```text
training_data/assistant_dpo.jsonl
```

新的 DPO 不再训练 retrieval action，而是训练答案偏好：

```text
prompt
  -> chosen_answer
  -> rejected_answer
  -> DPO 学习 chosen 相对 rejected 的偏好分差
```

DPO 偏好依据：

```text
证据是否充分
是否结合项目/知识库/文件来源
是否覆盖关键概念
是否减少泛化空话和无证据推断
结构是否清楚
领域术语是否准确
回答是否具体
```

## 6. DPO 训练命令

```bash
python scripts/train_answer_preference_dpo.py
```

默认参数：

```text
epochs = 120
batch_size = 8
lr = 5e-2
beta = 0.4
l2 = 0.001
```

训练输出：

```text
outputs/answer_preference_dpo/answer_preference_model.json
outputs/answer_preference_dpo/metrics.json
outputs/answer_preference_dpo/evaluation.json
outputs/answer_preference_dpo/training_trace.json
```

当前训练结果：

```text
Initial pairwise accuracy: 0.5000
Trained pairwise accuracy: 1.0000
Accuracy gain:             +0.5000

Initial average margin: 0.0000
Trained average margin: 6.5516
Margin gain:             +6.5516
```

训练后的偏好权重方向：

```text
证据语言 evidence_language:       正权重
结构清晰 structure_language:      正权重
领域术语 domain_term_density:     正权重
具体细节 concrete_detail:         正权重
泛化/无支撑 generic_or_unsupported: 负权重
```

这说明 DPO 已经学到：更应该偏好有证据、有结构、有领域细节的回答，而不是泛泛回答。

## 7. 当前项目口径

答辩时建议这样说：

```text
本项目没有把 PPO/DPO 继续局限在单步检索动作选择上。
LinUCB 仍然适合单步检索策略；
PPO 被提升到多步 Agent 行为策略，训练 Agent 在读取文件、检索、重排、压缩、二次检索和回答之间做决策；
DPO 被用于答案偏好优化，通过 chosen/rejected 数据让模型偏向证据充分、结构清楚、少幻觉的回答。
```

这样能解释为什么之前 LinUCB 在检索层表现好，同时也能体现 PPO/DPO 的真正价值。
