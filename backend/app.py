from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.agent.loop import agent_chat
from backend.bootstrap import ensure_python_paths
from backend.context.assembler import ContextAssembler
from backend.context.compressor import ConversationCompressor
from backend.storage.db import (
    database_status,
    get_latest_retrieval_rl_run,
    list_recent_trajectory_runs,
    save_trajectory_run,
)
from backend.retrieval.knowledge_store import build_context_block, build_index, build_kb, knowledge_overview, search_knowledge
from backend.image.generation_types import ImageGenerationError
from backend.image.quality import generate_image_advanced
from backend.image.service import comfyui_status, generate_image
from backend.llm.service import chat_reply, llm_status
from backend.llm.model_routing import resolve_model_identity
from backend.storage.memory_store import append_turn, clear_session, create_session, get_session, list_sessions
from backend.settings import (
    ROOT,
    advanced_ui_enabled,
    agent_enabled,
    get_embed_model,
    get_hf_model_path,
    get_image_provider,
    get_llm_backend,
    get_ollama_model,
    get_model_options,
    get_ui_mode,
    rag_enabled,
)
from backend.agent.tools import analyze_local_path, apply_file_edit, clone_github_repo, list_repo_directories, propose_file_edit
from backend.workspace.tools import read_file as read_workspace_file

ensure_python_paths()

logger = logging.getLogger(__name__)


def health_payload() -> dict[str, Any]:
    # 核心入口：前端状态面板调用这里，用一个 payload 返回模型、数据库、RAG、Agent、ComfyUI 是否可用。
    overview = (
        knowledge_overview()
        if rag_enabled()
        else {
            "title": "RAG disabled",
            "documentCount": 0,
            "rawDirectory": "",
            "researchFocus": "",
        }
    )
    llm = llm_status()
    model_identity = resolve_model_identity(
        get_hf_model_path() if get_llm_backend() == "hf" else get_ollama_model(),
        backend=get_llm_backend(),
    )
    db = database_status()
    return {
        "ok": True,
        "llmBackend": llm["backend"],
        "llmModel": get_hf_model_path() if get_llm_backend() == "hf" else get_ollama_model(),
        "llmReady": llm["ready"],
        "llmDetail": llm["detail"],
        "llmCanonicalModel": llm.get("canonical_model", model_identity.canonical_model),
        "llmProviderKind": llm.get("provider_kind", model_identity.provider_kind),
        "modelOptions": get_model_options(),
        "comfyui": comfyui_status(),
        "agentEnabled": agent_enabled(),
        "ragEnabled": rag_enabled(),
        "embeddingModel": get_embed_model(),
        "knowledgeTitle": overview["title"],
        "knowledgeDocuments": overview["documentCount"],
        "knowledgeRawDirectory": overview["rawDirectory"],
        "researchFocus": overview["researchFocus"],
        "sessions": len(list_sessions()),
        "uiMode": get_ui_mode(),
        "advancedUiEnabled": advanced_ui_enabled(),
        "assistantContract": {
            "localModel": True,
            "conversationMemory": True,
            "knowledgeRetrieval": rag_enabled(),
            "minimalChatFirst": True,
            "advancedTrainingOptional": True,
        },
        "dbBackend": db["backend"],
        "dbEnabled": db["enabled"],
        "dbReady": db["ready"],
        "dbDetail": db["detail"],
    }


def create_session_payload(title: str | None = None) -> dict[str, Any]:
    # 会话接口：创建一个新的聊天会话，并返回前端需要的 session 对象。
    session = create_session(title)
    return {"ok": True, "session": session}


def sessions_payload() -> dict[str, Any]:
    # 会话接口：列出所有聊天会话，供侧边栏展示历史对话。
    return {"ok": True, "sessions": list_sessions()}


def session_detail_payload(session_id: str) -> dict[str, Any]:
    # 会话接口：读取单个会话的完整消息历史。
    session = get_session(session_id)
    return {"ok": True, "session": session}


def knowledge_payload() -> dict[str, Any]:
    # 知识库接口：返回当前 RAG 知识库的文档数量、主题和样例标题。
    if not rag_enabled():
        return {"ok": True, "title": "RAG disabled", "documentCount": 0, "topics": {}, "sampleTitles": []}
    return {"ok": True, **knowledge_overview()}


def rebuild_knowledge_payload() -> dict[str, Any]:
    # 知识库接口：重新扫描/构建本地知识库索引。
    if not rag_enabled():
        return {"ok": False, "detail": "RAG is disabled. Set LOCAL_ENABLE_RAG=1 to rebuild the knowledge base."}
    kb = build_kb()
    index = build_index()
    return {"ok": True, "documents": kb["source"]["documentCount"], **index}


def search_payload(query: str, top_k: int) -> dict[str, Any]:
    # RAG 接口：直接搜索知识库，主要用于调试检索结果，不一定经过完整 Agent。
    if not rag_enabled():
        return {"ok": True, "results": []}
    results = search_knowledge(query, top_k=top_k)
    return {"ok": True, "results": results}


def chat_payload(
    query: str,
    session_id: str | None,
    top_k: int,
    attachment_name: str = "",
    attachment_text: str = "",
    model_id: str | None = None,
) -> dict[str, Any]:
    # 核心1/2/4：聊天主入口。这里读取会话历史、执行上下文压缩、转发给 Agent/Qwen，并保存本轮对话。
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    session = get_session(session_id)
    history = session.get("messages", [])
    assembled_context = {"messages": history, "metadata": {"mode": "legacy"}}
    try:
        # 核心2：长对话先压缩旧消息，再按当前问题重新组装成较短的 prompt 窗口。
        ConversationCompressor(model_id=model_id).maybe_compress(session["id"])
        assembled_context = ContextAssembler().assemble(session["id"], query, history_fallback=history)
        history = assembled_context["messages"]
    except Exception as error:
        logger.warning("context assembly failed session_id=%s error=%s", session.get("id"), error)
    results = []
    context_block = ""
    if rag_enabled() and not agent_enabled():
        results = search_knowledge(query, top_k=top_k)
        context_block = build_context_block(results) if results else ""
    traces: list[dict[str, Any]] = []
    file_context = ""
    visible_query = query
    if attachment_text.strip():
        # 核心4：附件文件正文作为隐藏上下文送给模型；聊天框里只显示轻量附件标记，避免界面被长文本刷屏。
        label = attachment_name.strip() or "attached_file"
        visible_query = (
            f"{query}\n\n"
            f"[Attached file: {label}, {len(attachment_text.strip())} chars included for analysis]"
        )
        file_context = f"\n\n[Attached file: {label}]\n{attachment_text.strip()}"

    if agent_enabled() and not file_context:
        agent_result = agent_chat(query, history=history, top_k=top_k, model_id=model_id)
        answer = agent_result["answer"]
        traces = agent_result.get("tool_traces", [])
        if agent_result.get("sources"):
            results = agent_result["sources"]
        if agent_result.get("context_preview"):
            context_block = agent_result["context_preview"]
    else:
        answer = chat_reply(query, history=history, context_block=f"{context_block}{file_context}", model_id=model_id)

    updated = append_turn(session["id"], visible_query, answer)
    try:
        ConversationCompressor(model_id=model_id).maybe_compress(session["id"])
    except Exception as error:
        logger.warning("post-chat compression failed session_id=%s error=%s", session.get("id"), error)
    return {
        "ok": True,
        "session": {
            "id": updated["id"],
            "title": updated["title"],
            "summary": updated["summary"],
            "updated_at": updated["updated_at"],
        },
        "answer": answer,
        "sources": results,
        "toolTraces": traces,
        "history": updated["messages"][-12:],
        "contextPreview": (f"{context_block}{file_context}\n\n[Context assembly]\n{json.dumps(assembled_context.get('metadata', {}), ensure_ascii=False)}")[:2000],
    }


def clear_session_payload(session_id: str) -> dict[str, Any]:
    session = clear_session(session_id)
    return {"ok": True, "session": session}


def retrieval_training_payload() -> dict[str, Any]:
    # 强化学习接口：返回最近一次检索策略训练/评测结果，供前端训练页展示。
    db_latest = get_latest_retrieval_rl_run()
    latest_file = _latest_file_retrieval_run()
    best_file = _best_file_retrieval_run()
    comparison_candidates = _comparison_file_retrieval_runs()
    comparison_best = _best_retrieval_run(comparison_candidates)
    best_available = comparison_best or _best_retrieval_run([item for item in [db_latest, latest_file, best_file] if item])
    # Prefer the strongest file artifact for the visual comparison. The
    # database can still contain older DQN runs, which are useful history but
    # misleading for the PPO/DPO/ORPO comparison panel.
    latest = best_available or latest_file or db_latest or best_file
    if not latest:
        return {"ok": True, "available": False}

    trace = latest.get("episodes", {}).get("training_trace", [])
    evaluation_series, representative, baseline_match = _evaluation_series_for_run(latest)
    comparison_runs = []
    for run in comparison_candidates:
        run_series, _, _ = _evaluation_series_for_run(run)
        run_metrics = run.get("metrics", {}) or {}
        comparison_runs.append(
            {
                "name": run["run_name"],
                "algorithm": str(run_metrics.get("algorithm") or "").upper(),
                "outputPath": run["output_path"],
                "metrics": run_metrics,
                "series": run_series,
                "improvement": {
                    "rewardGain": float(run_metrics.get("reward_gain_vs_baseline") or 0),
                    "sourceHitGain": float(run_metrics.get("source_hit_gain_vs_baseline") or 0),
                    "topicHitGain": float(run_metrics.get("topic_hit_gain_vs_baseline") or 0),
                    "pointRecallGain": float(run_metrics.get("point_recall_gain_vs_baseline") or 0),
                },
            }
        )

    return {
        "ok": True,
        "available": True,
        "run": {
            "id": latest["id"],
            "name": latest["run_name"],
            "status": latest["status"],
            "dataPath": latest["data_path"],
            "outputPath": latest["output_path"],
            "metrics": latest.get("metrics", {}),
            "algorithm": str((latest.get("metrics", {}) or {}).get("algorithm") or "dqn").upper(),
            "updatedAt": latest.get("updated_at", ""),
        },
        "bestRun": {
            "name": best_available.get("run_name", "") if best_available else "",
            "algorithm": str(((best_available or {}).get("metrics", {}) or {}).get("algorithm") or "").upper(),
            "rewardGain": float(((best_available or {}).get("metrics", {}) or {}).get("reward_gain_vs_baseline") or 0),
            "outputPath": str((best_available or {}).get("output_path") or ""),
        },
        "comparisonRuns": comparison_runs,
        "improvement": {
            "rewardGain": float((latest.get("metrics", {}) or {}).get("reward_gain_vs_baseline") or 0),
            "sourceHitGain": float((latest.get("metrics", {}) or {}).get("source_hit_gain_vs_baseline") or 0),
            "topicHitGain": float((latest.get("metrics", {}) or {}).get("topic_hit_gain_vs_baseline") or 0),
            "pointRecallGain": float((latest.get("metrics", {}) or {}).get("point_recall_gain_vs_baseline") or 0),
            "betterThanBaseline": float((latest.get("metrics", {}) or {}).get("reward_gain_vs_baseline") or 0) > 0,
        },
        "curve": trace,
        "evaluationSeries": evaluation_series,
        "episode": representative,
        "baselineEpisode": baseline_match,
    }


def _read_json(path: Path) -> dict[str, Any]:
    # 工具函数：安全读取 JSON 文件，文件不存在或格式错误时返回空 dict。
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # 工具函数：安全读取 jsonl 文件，跳过空行和解析失败的行。
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _evaluation_series_for_run(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    # 强化学习图表：把 evaluation.json 转成逐题折线图需要的 series。
    trained_episodes = run.get("episodes", {}).get("trained_episode", [])
    baseline_episodes = run.get("episodes", {}).get("baseline_episode", [])

    representative = trained_episodes[0] if trained_episodes else None
    if trained_episodes:
        representative = max(trained_episodes, key=lambda item: float(item.get("reward", 0.0)))

    baseline_match = None
    if representative and baseline_episodes:
        rep_query = representative.get("query")
        baseline_match = next((item for item in baseline_episodes if item.get("query") == rep_query), baseline_episodes[0])

    baseline_by_query = {str(item.get("query") or ""): item for item in baseline_episodes}
    evaluation_series: list[dict[str, Any]] = []
    for index, trained in enumerate(trained_episodes, start=1):
        query = str(trained.get("query") or "")
        baseline = baseline_by_query.get(query) or (baseline_episodes[index - 1] if index - 1 < len(baseline_episodes) else {})
        trained_reward = float(trained.get("reward") or 0)
        baseline_reward = float(baseline.get("reward") or 0)
        evaluation_series.append(
            {
                "index": index,
                "query": query,
                "label": f"Q{index}",
                "baselineReward": round(baseline_reward, 4),
                "trainedReward": round(trained_reward, 4),
                "rewardGain": round(trained_reward - baseline_reward, 4),
                "baselineSourceHit": float(baseline.get("source_hit") or 0),
                "trainedSourceHit": float(trained.get("source_hit") or 0),
                "sourceHitGain": round(float(trained.get("source_hit") or 0) - float(baseline.get("source_hit") or 0), 4),
                "baselineTopicHit": float(baseline.get("topic_hit") or 0),
                "trainedTopicHit": float(trained.get("topic_hit") or 0),
                "topicHitGain": round(float(trained.get("topic_hit") or 0) - float(baseline.get("topic_hit") or 0), 4),
                "baselinePointRecall": float(baseline.get("point_recall") or 0),
                "trainedPointRecall": float(trained.get("point_recall") or 0),
                "trainedAction": str(trained.get("chosen_action") or ""),
                "baselineAction": str(baseline.get("chosen_action") or ""),
            }
        )
    return evaluation_series, representative, baseline_match


def _metrics_from_evaluation(evaluation: dict[str, Any], algorithm: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    # 强化学习图表：从 evaluation.json 中提取 baseline、trained 和 gain 指标。
    trained = evaluation.get("trained_policy", {}) if isinstance(evaluation.get("trained_policy"), dict) else {}
    baseline = evaluation.get("baseline_policy", {}) if isinstance(evaluation.get("baseline_policy"), dict) else {}
    metrics = {
        "algorithm": algorithm,
        "trained_average_reward": float(trained.get("average_reward") or 0),
        "trained_average_source_hit": float(trained.get("average_source_hit") or 0),
        "trained_average_topic_hit": float(trained.get("average_topic_hit") or 0),
        "trained_average_point_recall": float(trained.get("average_point_recall") or 0),
        "baseline_average_reward": float(baseline.get("average_reward") or 0),
        "baseline_average_source_hit": float(baseline.get("average_source_hit") or 0),
        "baseline_average_topic_hit": float(baseline.get("average_topic_hit") or 0),
        "baseline_average_point_recall": float(baseline.get("average_point_recall") or 0),
    }
    metrics["reward_gain_vs_baseline"] = round(metrics["trained_average_reward"] - metrics["baseline_average_reward"], 4)
    metrics["source_hit_gain_vs_baseline"] = round(metrics["trained_average_source_hit"] - metrics["baseline_average_source_hit"], 4)
    metrics["topic_hit_gain_vs_baseline"] = round(metrics["trained_average_topic_hit"] - metrics["baseline_average_topic_hit"], 4)
    metrics["point_recall_gain_vs_baseline"] = round(metrics["trained_average_point_recall"] - metrics["baseline_average_point_recall"], 4)
    if extra:
        metrics.update(extra)
    return metrics


def _file_run_from_output(name: str, output_dir: Path, algorithm: str, metrics_hint: dict[str, Any] | None = None) -> dict[str, Any] | None:
    # 强化学习图表：从 outputs 目录读取某个算法的 evaluation/trace/checkpoint 信息。
    evaluation = _read_json(output_dir / "evaluation.json")
    if not evaluation:
        return None
    trace = _read_json(output_dir / "training_trace.json")
    if not isinstance(trace, list):
        trace = []
    metrics = _metrics_from_evaluation(evaluation, algorithm=algorithm, extra=metrics_hint)
    trained_episodes = ((evaluation.get("trained_policy") or {}).get("episodes") or []) if isinstance(evaluation.get("trained_policy"), dict) else []
    baseline_episodes = ((evaluation.get("baseline_policy") or {}).get("episodes") or []) if isinstance(evaluation.get("baseline_policy"), dict) else []
    return {
        "id": 0,
        "run_name": name,
        "data_path": "",
        "output_path": str(output_dir),
        "status": "completed",
        "metrics": metrics,
        "evaluation": evaluation,
        "episodes": {
            "training_trace": trace,
            "trained_episode": trained_episodes,
            "baseline_episode": baseline_episodes,
        },
        "created_at": "",
        "updated_at": "",
    }


def _best_file_retrieval_run() -> dict[str, Any] | None:
    # 强化学习图表：在文件输出中选择当前效果最好的检索策略。
    candidates = _file_retrieval_run_candidates()
    if not candidates:
        return None
    return _best_retrieval_run(candidates)


def _latest_file_retrieval_run() -> dict[str, Any] | None:
    # 强化学习图表：读取最近一次文件输出的检索策略结果。
    candidates = _file_retrieval_run_candidates()
    if not candidates:
        return None
    return max(candidates, key=lambda item: (Path(str(item.get("output_path") or "")) / "evaluation.json").stat().st_mtime)


def _file_retrieval_run_candidates() -> list[dict[str, Any]]:
    # 强化学习图表：收集 PPO、DPO、ORPO、LinUCB、Dueling DDQN 等候选结果。
    candidates: list[dict[str, Any]] = []
    linucb_run = _file_run_from_output("retrieval_policy_linucb", ROOT / "outputs" / "retrieval_policy_linucb", "LinUCB")
    if linucb_run:
        candidates.append(linucb_run)
    dueling_ddqn_run = _file_run_from_output("retrieval_policy_dueling_ddqn", ROOT / "outputs" / "retrieval_policy_dueling_ddqn", "Dueling DDQN")
    if dueling_ddqn_run:
        candidates.append(dueling_ddqn_run)
    orpo_torch_run = _file_run_from_output("retrieval_policy_orpo_torch", ROOT / "outputs" / "retrieval_policy_orpo_torch", "orpo")
    if orpo_torch_run:
        candidates.append(orpo_torch_run)
    dpo_torch_run = _file_run_from_output("retrieval_policy_dpo_torch", ROOT / "outputs" / "retrieval_policy_dpo_torch", "dpo")
    if dpo_torch_run:
        candidates.append(dpo_torch_run)
    ppo_torch_60_run = _file_run_from_output("retrieval_policy_ppo_torch_60", ROOT / "outputs" / "retrieval_policy_ppo_torch_60", "ppo")
    if ppo_torch_60_run:
        candidates.append(ppo_torch_60_run)
    ppo_torch_run = _file_run_from_output("retrieval_policy_ppo_torch", ROOT / "outputs" / "retrieval_policy_ppo_torch", "ppo")
    if ppo_torch_run:
        candidates.append(ppo_torch_run)
    ppo_run = _file_run_from_output("retrieval_policy_ppo", ROOT / "outputs" / "retrieval_policy_ppo", "ppo-linear")
    if ppo_run:
        candidates.append(ppo_run)

    sweep = _read_json(ROOT / "outputs" / "retrieval_reward_sweep" / "reward_sweep_summary.json")
    best = sweep.get("best", {}) if isinstance(sweep.get("best"), dict) else {}
    best_output = Path(str(best.get("output") or ""))
    if best_output.exists():
        metrics_hint = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
        metrics_hint = {"algorithm": "dqn", **metrics_hint}
        sweep_run = _file_run_from_output(str(best.get("name") or "retrieval_reward_sweep_best"), best_output, "dqn", metrics_hint=metrics_hint)
        if sweep_run:
            candidates.append(sweep_run)
    return candidates


def _comparison_file_retrieval_runs() -> list[dict[str, Any]]:
    # 强化学习图表：按固定顺序返回前端要对比展示的算法列表。
    specs = [
        ("retrieval_policy_ppo_torch_60", ROOT / "outputs" / "retrieval_policy_ppo_torch_60", "ppo"),
        ("retrieval_policy_dpo_torch", ROOT / "outputs" / "retrieval_policy_dpo_torch", "dpo"),
        ("retrieval_policy_linucb", ROOT / "outputs" / "retrieval_policy_linucb", "LinUCB"),
        ("retrieval_policy_dueling_ddqn", ROOT / "outputs" / "retrieval_policy_dueling_ddqn", "Dueling DDQN"),
        ("retrieval_policy_orpo_torch", ROOT / "outputs" / "retrieval_policy_orpo_torch", "orpo"),
    ]
    runs: list[dict[str, Any]] = []
    for name, path, algorithm in specs:
        run = _file_run_from_output(name, path, algorithm)
        if run:
            runs.append(run)
    return runs


def _best_retrieval_run(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    # 强化学习图表：按照 reward gain 选择最佳策略。
    if not candidates:
        return None
    return max(candidates, key=lambda item: float((item.get("metrics", {}) or {}).get("reward_gain_vs_baseline") or 0))


def _split_counts(total: int) -> dict[str, int]:
    # 数据集展示：按 3/1/1 粗略拆分训练、验证、测试数量。
    train = int(total * 3 / 5)
    validation = int(total * 1 / 5)
    test = max(total - train - validation, 0)
    return {"train": train, "validation": validation, "test": test}


# 核心5：强化学习效果图的数据接口。TrainingView/PolicyView 用这里的数据对比 Baseline、PPO、DPO、ORPO、LinUCB、DDQN。
def policy_evaluation_payload() -> dict[str, Any]:
    question_bank_path = ROOT / "training_data" / "agent_question_bank.jsonl"
    question_results_path = ROOT / "outputs" / "agent_question_bank_results.json"
    sweep_path = ROOT / "outputs" / "retrieval_reward_sweep" / "reward_sweep_summary.json"
    rag_eval_path = ROOT / "outputs" / "rag_eval_results.json"

    question_bank = _read_jsonl(question_bank_path)
    question_results = _read_json(question_results_path)
    sweep = _read_json(sweep_path)
    rag_eval = _read_json(rag_eval_path)

    completed_results = question_results.get("results", []) if isinstance(question_results.get("results"), list) else []
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    total_sources = 0
    total_tool_traces = 0
    for row in completed_results:
        category = str(row.get("category") or "unknown")
        difficulty = str(row.get("difficulty") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        total_sources += int(row.get("source_count") or 0)
        total_tool_traces += int(row.get("tool_trace_count") or 0)

    completed_count = len(completed_results)
    best = sweep.get("best", {}) if isinstance(sweep.get("best"), dict) else {}
    best_metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
    latest_rl = get_latest_retrieval_rl_run() or {}
    latest_rl_metrics = latest_rl.get("metrics", {}) if isinstance(latest_rl.get("metrics"), dict) else {}
    if latest_rl_metrics and not best_metrics:
        best_metrics = latest_rl_metrics
        best = {
            "name": latest_rl.get("run_name", ""),
            "metrics": latest_rl_metrics,
            "output": latest_rl.get("output_path", ""),
        }
    trials = sweep.get("trials", []) if isinstance(sweep.get("trials"), list) else []
    rag_metrics = ((rag_eval.get("summary") or {}).get("metrics") or {}) if isinstance(rag_eval.get("summary"), dict) else {}

    dataset_total = len(question_bank) or int(question_results.get("question_count") or 0)
    return {
        "ok": True,
        "available": bool(question_bank or completed_results or best_metrics or rag_metrics),
        "dataset": {
            "path": str(question_bank_path),
            "questionCount": dataset_total,
            "split": _split_counts(dataset_total),
        },
        "questionBank": {
            "path": str(question_results_path),
            "completed": completed_count,
            "total": int(question_results.get("question_count") or dataset_total or completed_count),
            "elapsedSeconds": float(question_results.get("elapsed_seconds") or 0),
            "averageSources": round(total_sources / completed_count, 4) if completed_count else 0,
            "averageToolTraces": round(total_tool_traces / completed_count, 4) if completed_count else 0,
            "categoryCounts": category_counts,
            "difficultyCounts": difficulty_counts,
        },
        "rag": {
            "path": str(rag_eval_path),
            "sourceHit": float(rag_metrics.get("source_hit") or 0),
            "topicHit": float(rag_metrics.get("topic_hit") or 0),
            "answerPointRecall": float(rag_metrics.get("answer_point_recall") or 0),
        },
        "optimization": {
            "path": str(sweep_path),
            "bestName": str(best.get("name") or ""),
            "latestRunName": str(latest_rl.get("run_name") or ""),
            "latestAlgorithm": str(latest_rl_metrics.get("algorithm") or ""),
            "trialCount": len(trials),
            "episodes": int(best_metrics.get("episodes") or 0),
            "baseline": {
                "reward": float(best_metrics.get("baseline_average_reward") or 0),
                "sourceHit": float(best_metrics.get("baseline_average_source_hit") or 0),
                "topicHit": float(best_metrics.get("baseline_average_topic_hit") or 0),
                "pointRecall": float(best_metrics.get("baseline_average_point_recall") or 0),
            },
            "trained": {
                "reward": float(best_metrics.get("trained_average_reward") or 0),
                "sourceHit": float(best_metrics.get("trained_average_source_hit") or 0),
                "topicHit": float(best_metrics.get("trained_average_topic_hit") or 0),
                "pointRecall": float(best_metrics.get("trained_average_point_recall") or 0),
            },
            "rewardGain": float(best_metrics.get("reward_gain_vs_baseline") or 0),
            "trials": [
                {
                    "name": str(trial.get("name") or f"trial_{index + 1}"),
                    "score": float(trial.get("score") or 0),
                    "rewardGain": float(((trial.get("metrics") or {}).get("reward_gain_vs_baseline")) or 0),
                }
                for index, trial in enumerate(trials)
                if isinstance(trial, dict)
            ],
        },
    }


def local_file_analysis_payload(path: str, prompt: str = "", max_chars: int = 12000) -> dict[str, Any]:
    # 核心4：本地文件分析入口，读取指定文件/目录并交给 Agent 总结。
    result = analyze_local_path(path, prompt=prompt, max_chars=max_chars)
    return {"ok": True, **result}


def local_file_read_payload(path: str, start_line: int = 1, end_line: int | None = None, max_bytes: int | None = None) -> dict[str, Any]:
    # 核心4：本地文件读取入口，返回带行号的文本内容。
    result = read_workspace_file(path=path, start_line=start_line, end_line=end_line, max_bytes=max_bytes)
    return {"ok": True, **result}


def propose_file_edit_payload(path: str, instruction: str, max_chars: int = 24000, model_id: str | None = None) -> dict[str, Any]:
    # 核心4：根据用户指令生成文件修改方案，但不直接写入文件。
    result = propose_file_edit(path, instruction=instruction, max_chars=max_chars, model_id=model_id)
    return {"ok": True, **result}


def apply_file_edit_payload(path: str, new_content: str, sha256_before: str = "", instruction: str = "") -> dict[str, Any]:
    # 核心4：真正应用文件修改；会结合 sha256_before 做安全校验，避免覆盖用户新改动。
    result = apply_file_edit(path, new_content=new_content, sha256_before=sha256_before, instruction=instruction)
    return {"ok": True, **result}


def repo_list_payload() -> dict[str, Any]:
    result = list_repo_directories()
    return {"ok": True, **result}


def clone_repo_payload(repo_url: str, branch: str = "", target_name: str = "") -> dict[str, Any]:
    result = clone_github_repo(repo_url, branch=branch, target_name=target_name)
    repos = list_repo_directories()
    return {"ok": True, **result, "repos": repos.get("items", [])}


# 核心6/7：地图实验保存入口。把 baseline 路线、DQN/PPO 方法、S3/RLTS/Mlsimp 压缩结果写入 MySQL。
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


def trajectory_runs_payload(limit: int = 20) -> dict[str, Any]:
    # 核心6/7：读取最近地图实验记录，供前端展示历史轨迹。
    return {"ok": True, "items": list_recent_trajectory_runs(limit=limit)}


def image_generation_payload(
    prompt: str,
    model: str = "",
    size: str = "1024x1024",
    quality: str = "",
    format: str = "",
    background: str = "",
    n: int = 1,
    style_notes: str = "",
    preset: str = "",
    quality_mode: str = "high",
    batch_size: int | None = None,
    allow_retry: bool = True,
    use_highres_fix: bool | None = True,
) -> dict[str, Any]:
    # 核心3：图片侧边栏的直接生成接口。聊天里明确要求画图时通常走 Agent -> generate_image_advanced。
    try:
        if get_image_provider() == "comfyui":
            return generate_image_advanced(
                prompt=prompt,
                style=style_notes,
                preset=preset,
                size=size,
                batch_size=batch_size or 1,
                quality_mode=quality_mode or "high",
                allow_retry=allow_retry,
                use_highres_fix=use_highres_fix,
            )
        return generate_image(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            format=format,
            background=background,
            n=n,
            style_notes=style_notes,
            user_visible_prompt=prompt,
        )
    except ImageGenerationError as error:
        return {"ok": False, "error": error.to_payload(), "detail": error.message}
    except Exception as error:
        try:
            return generate_image(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                format=format,
                background=background,
                n=n,
                style_notes=style_notes,
                user_visible_prompt=prompt,
            )
        except Exception as fallback_error:
            return {"ok": False, "detail": f"Image generation failed: {error}; fallback failed: {fallback_error}"}
