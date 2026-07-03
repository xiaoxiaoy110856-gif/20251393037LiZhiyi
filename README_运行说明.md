# 项目运行说明

本项目包含 Vue 前端、Python 后端、Ollama/Qwen 本地模型、RAG 知识库、PPO/DPO 训练脚本、地图轨迹展示、ComfyUI 图片生成和可选 MySQL 存储。

## 1. 前端启动

```powershell
cd frontend
npm install
npm run dev
```

默认地址：

```text
http://127.0.0.1:5173
```

前端页面主要包括：

```text
/workspace   聊天、文件读取、Agent、RAG、图片生成
/knowledge   知识库概览
/training    PPO/DPO/ORPO 等训练效果图表
/policy      强化学习策略解释
/pathdemo    地图轨迹、DQN/PPO、S3/RLTS/Mlsimp 展示
/status      系统状态检查
/repos       本地仓库和文件工具
```

## 2. 后端启动

建议先安装依赖：

```powershell
pip install -r requirements-trajectory-rag.txt
```

启动后端：

```powershell
python scripts/serve_local_assistant.py
```

默认地址：

```text
http://127.0.0.1:8765
```

## 3. Ollama / Qwen

后端默认通过 Ollama 调用 Qwen：

```text
LOCAL_LLM_BACKEND=ollama
LOCAL_OLLAMA_BASE_URL=http://127.0.0.1:11434
LOCAL_LLM_MODEL=qwen3.5:latest
```

启动 Ollama 后确认模型可用：

```powershell
ollama list
ollama run qwen3.5:latest
```

## 4. RAG 知识库

构建本地知识库：

```powershell
python scripts/build_rag_corpus.py
python scripts/build_rag_index.py
```

说明：

```text
kb/raw      放原始资料
kb/parsed   保存切分后的知识库 JSON
kb/index    保存向量索引
```

提交源码包时不放论文 PDF，避免体积过大；运行者可自行把论文放入 `kb/raw` 后重建索引。

## 5. MySQL 数据库

默认可以使用本地文件存储：

```text
LOCAL_DB_BACKEND=file
```

如果要连 MySQL：

```text
LOCAL_DB_BACKEND=mysql
LOCAL_MYSQL_HOST=127.0.0.1
LOCAL_MYSQL_PORT=3306
LOCAL_MYSQL_USER=root
LOCAL_MYSQL_PASSWORD=your_password
LOCAL_MYSQL_DATABASE=trl_agent
```

初始化数据库：

```powershell
python scripts/init_mysql_storage.py
```

## 6. ComfyUI 图片生成

先启动 ComfyUI，确认页面可访问：

```text
http://127.0.0.1:8188
```

项目配置：

```text
LOCAL_ENABLE_COMFYUI=1
LOCAL_COMFYUI_URL=http://127.0.0.1:8188
IMAGE_PROVIDER=comfyui
```

## 7. PPO / DPO 训练

PPO 多步 Agent 决策训练：

```powershell
python scripts/train_agent_policy_ppo.py --updates 60
```

DPO 答案偏好训练：

```powershell
python scripts/train_answer_preference_dpo.py
```

训练结果保存在：

```text
outputs/agent_policy_ppo_multi_step/
outputs/answer_preference_dpo/
```

## 8. 打包说明

源码包应保留：

```text
frontend/
backend/
scripts/
training_data/
tests/
docs/
requirements*.txt
README_运行说明.md
.env.example
PPT 文件
```

源码包应排除：

```text
node_modules/
dist/
build/
venv/
__pycache__/
.env
论文 PDF
模型权重
大图片/视频
日志文件
临时输出
```
