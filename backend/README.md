# Backend Layout

后端按职责拆成几个子包，根目录只保留启动入口、配置和路径初始化。

## 根目录

- `app.py`：HTTP/API 业务入口，负责组装 payload 和调用各业务模块。
- `settings.py`：环境变量、路径、模型、RAG、图片、数据库等配置。
- `bootstrap.py`：运行脚本时补齐 Python import 路径。

## 子包

- `agent/`：Agent 主循环、工具注册表、本地工具实现。
- `context/`：上下文压缩、摘要存储、rolling state、相关上下文召回。
- `image/`：ComfyUI/OpenAI 图片生成、质量评分、图片保存。
- `llm/`：Ollama/Qwen、HF 本地模型、模型路由。
- `retrieval/`：知识库构建与搜索、RAG 检索策略、RL 检索环境。
- `storage/`：MySQL/本地会话、训练记录、轨迹实验记录。
- `workspace/`：本地文件读取、搜索、编辑提案和沙盒路径校验。

## 调用方向

```text
app.py
  -> agent / llm / context / retrieval / image / storage / workspace
  -> settings.py
```

业务模块之间尽量保持单向调用：`agent` 可以调用 `retrieval`、`llm`、`workspace` 和 `image`；`context` 可以调用 `llm` 做摘要；`retrieval` 不反向依赖 `agent`。
