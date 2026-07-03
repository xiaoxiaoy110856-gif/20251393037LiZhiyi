# 20251393037LiZhiyi

面向轨迹压缩和轨迹特征表示的多智能体协作科研辅助系统。

本仓库为课程设计 GitHub 提交版，保留核心源码、前端页面、后端接口、训练策略脚本、测试文件和软件设计文档；不包含模型文件、依赖目录、缓存、完整训练输出、真实对话记录和大体积知识库索引。

## 核心功能

- 本地 Ollama/Qwen 科研问答
- RAG 知识库检索增强
- Agent 工具调用与仓库/文件分析
- LinUCB、DDQN、PPO、DPO、ORPO 等策略脚本与结果展示
- 轨迹规划与轨迹压缩演示
- 图片生成接口入口
- MySQL 或本地文件存储回退

## 启动方式

1. 启动 Ollama：

```powershell
ollama serve
```

2. 启动后端：

```powershell
cd "项目根目录"
python scripts\serve_local_assistant.py
```

后端默认地址：

```text
http://127.0.0.1:8765
```

3. 启动前端：

```powershell
cd frontend
npm install
npm run dev
```

前端默认地址：

```text
http://127.0.0.1:5173
```

## 目录说明

| 目录/文件 | 说明 |
| --- | --- |
| backend | 后端核心代码，包括对话、RAG、Agent、LLM、存储和安全模块 |
| frontend | Vue/Vite 前端源码，不包含 node_modules 和 dist |
| scripts | 后端启动脚本与训练策略脚本 |
| docs | 项目说明和结构文档 |
| tests | 测试代码 |
| kb | 知识库目录占位说明 |
| outputs | 输出目录占位说明 |
| repos | 仓库分析目录占位说明 |
| training_data | 训练数据目录占位说明 |
| *.docx | 软件设计文档最终版 |

## 说明

MySQL 不是本地运行的强制前提，项目可按配置使用文件存储回退。若 `/api/health` 中 `llmReady=false`，请先检查 Ollama 服务和模型名称。
