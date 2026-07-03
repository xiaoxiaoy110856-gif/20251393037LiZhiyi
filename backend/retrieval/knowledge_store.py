from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.bootstrap import ensure_python_paths
from backend.settings import (
    ensure_runtime_dirs,
    get_context_char_limit,
    get_embed_device,
    get_embed_model,
    get_index_dir,
    get_kb_json_path,
    get_raw_kb_dir,
    get_research_focus,
    get_top_k,
)

ensure_python_paths()


def _build_hf_embedding(HuggingFaceEmbedding: Any) -> Any:
    model_name = get_embed_model()
    device = get_embed_device()
    try:
        return HuggingFaceEmbedding(model_name=model_name, device=device)
    except TypeError:
        # Older llama-index wrappers may not accept device explicitly.
        return HuggingFaceEmbedding(model_name=model_name)


TOPIC_RULES: dict[str, list[str]] = {
    "trajectory_compression": [
        "trajectory compression",
        "trajectory simplification",
        "douglas-peucker",
        "compression ratio",
        "stream simplification",
        "trajectory stream",
        "轨迹压缩",
        "轨迹简化",
    ],
    "trajectory_prediction": [
        "trajectory prediction",
        "next location prediction",
        "mobility prediction",
        "trajectory forecasting",
        "轨迹预测",
        "下一位置预测",
    ],
    "path_planning": [
        "path planning",
        "trajectory planning",
        "motion planning",
        "trajectory optimization",
        "路径规划",
        "轨迹规划",
        "轨迹优化",
    ],
    "reinforcement_learning": [
        "reinforcement learning",
        "policy gradient",
        "actor-critic",
        "markov decision process",
        "强化学习",
        "策略梯度",
    ],
    "ppo": ["ppo", "proximal policy optimization"],
    "dqn": ["dqn", "deep q network", "deep q-learning"],
    "sac": ["sac", "soft actor critic"],
    "offline_rl": ["offline reinforcement learning", "batch reinforcement learning", "离线强化学习"],
    "reward_design": ["reward shaping", "reward model", "preference optimization", "inverse reinforcement learning", "奖励设计"],
    "trajectory_similarity": ["trajectory similarity", "subtrajectory search", "frechet", "hausdorff", "dtw", "轨迹相似性"],
    "experiment_design": ["ablation", "benchmark", "evaluation", "error metric", "实验设计", "评估指标"],
    "project_code": ["python", "script", "function", "class ", "def ", "代码", ".py"],
}

TOPIC_QUERY_HINTS: dict[str, list[str]] = {
    "trajectory_compression": ["trajectory compression", "trajectory simplification", "Douglas-Peucker", "DOTS", "轨迹压缩", "轨迹简化"],
    "trajectory_prediction": ["trajectory prediction", "next location prediction", "轨迹预测"],
    "path_planning": ["trajectory planning", "path planning", "motion planning", "轨迹规划", "路径规划"],
    "reinforcement_learning": ["reinforcement learning", "policy optimization", "actor-critic", "强化学习", "策略优化"],
    "ppo": ["PPO", "proximal policy optimization"],
    "dqn": ["DQN", "deep q network"],
    "sac": ["SAC", "soft actor critic"],
    "offline_rl": ["offline reinforcement learning", "离线强化学习"],
    "reward_design": ["reward design", "inverse reinforcement learning", "reward shaping", "奖励设计"],
    "trajectory_similarity": ["trajectory similarity", "subtrajectory search", "DTW", "轨迹相似性"],
}

SOURCE_ALIASES: dict[str, list[str]] = {
    "difftori": [
        "difftori",
        "differentiable trajectory optimization for deep reinforcement and imitation learning",
        "2024_neurips_differentiable trajectory optimization for deep reinforcement and imitation learning",
    ],
    "differentiable trajectory optimization": [
        "differentiable trajectory optimization as a policy class for reinforcement and imitation learning",
        "differentiable trajectory optimization",
    ],
    "dots": [
        "dots",
        "an online and near-optimal trajectory simplification algorithm",
    ],
    "trajectory simplification": [
        "trajectory simplification",
        "trajectory compression",
        "douglas-peucker",
    ],
    "fast trajectory simplification": [
        "fast trajectory simplification",
        "fast trajectory simplification algorithm",
    ],
    "similar subtrajectory search": [
        "similar subtrajectory search",
        "exact and efficient similar subtrajectory search",
        "subtrajectory search",
    ],
    "trajectory similarity": [
        "trajectory similarity",
        "subtrajectory search",
        "dtw",
    ],
    "goirl": [
        "goirl",
        "graph-oriented inverse reinforcement learning",
    ],
    "driveirl": [
        "driveirl",
        "driving in real life with inverse reinforcement learning",
    ],
}

RESEARCH_FOCUS_KEYWORDS = {
    "reinforcement_learning": [
        "reinforcement learning",
        "ppo",
        "dqn",
        "sac",
        "reward",
        "policy",
        "trajectory planning",
        "trajectory compression",
        "强化学习",
        "策略",
        "奖励",
        "轨迹",
    ],
}

TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".csv", ".log"}
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".json", ".jsonl", ".py", ".csv", ".log"}
STRUCTURED_SUFFIXES = {".md"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path, max_pages: int = 20) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return ""
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages[:max_pages]:
        parts.append(page.extract_text() or "")
    return "\n".join(part.strip() for part in parts if part.strip())


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in TEXT_SUFFIXES:
        return _read_text_file(path)
    return ""


def _safe_text(text: str) -> str:
    return (text or "").encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(text)).strip()


def _paragraphs(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append("\n".join(current).strip())
    return [item for item in paragraphs if item]


def _merge_with_overlap(parts: list[str], chunk_size: int = 900, overlap: int = 120) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = part if not buffer else f"{buffer}\n\n{part}"
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer.strip())
        if len(part) <= chunk_size:
            buffer = part
            continue
        start = 0
        while start < len(part):
            end = min(start + chunk_size, len(part))
            piece = part[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(part):
                break
            start = max(end - overlap, start + 1)
        buffer = ""
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def _split_markdown(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "root"
    current_lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip() or "section"
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(heading, body) for heading, body in sections if body.strip()]


def split_document(path: Path, text: str) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    chunks: list[dict[str, str]] = []
    if suffix in STRUCTURED_SUFFIXES:
        for heading, body in _split_markdown(text):
            for piece in _merge_with_overlap(_paragraphs(body)):
                chunks.append({"section": heading, "text": piece})
    else:
        for piece in _merge_with_overlap(_paragraphs(text)):
            chunks.append({"section": "content", "text": piece})
    if not chunks and text.strip():
        chunks.append({"section": "content", "text": text.strip()[:900]})
    return chunks


def infer_doc_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "paper_pdf"
    if suffix == ".md":
        return "markdown_note"
    if suffix == ".py":
        return "code"
    if suffix in {".json", ".jsonl", ".csv", ".log"}:
        return "data_or_log"
    return "text"


def infer_topics(text: str, title: str, doc_type: str | None = None) -> list[str]:
    haystack = f"{title}\n{text}".lower()
    topics: list[str] = []
    for topic, keywords in TOPIC_RULES.items():
        if topic == "project_code":
            continue
        if any(keyword in haystack for keyword in keywords):
            topics.append(topic)
    if doc_type in {"code", "markdown_note"} and any(token in title.lower() for token in [".py", "script", "code", "代码"]):
        topics.append("project_code")
    deduped: list[str] = []
    for topic in topics:
        if topic not in deduped:
            deduped.append(topic)
    return deduped or ["general"]


def _query_topics(query: str) -> list[str]:
    lowered = query.lower()
    topics = [topic for topic, keywords in TOPIC_RULES.items() if topic != "project_code" and any(keyword in lowered for keyword in keywords)]
    deduped: list[str] = []
    for topic in topics:
        if topic not in deduped:
            deduped.append(topic)
    return deduped


def expand_query(query: str) -> list[str]:
    rewrites = [query.strip()]
    lowered = query.lower()
    detected_topics = _query_topics(query)
    for topic in detected_topics:
        for hint in TOPIC_QUERY_HINTS.get(topic, [])[:4]:
            candidate = f"{query} {hint}".strip()
            if candidate not in rewrites:
                rewrites.append(candidate)
    if ("ppo" in lowered or "强化学习" in query or "reinforcement learning" in lowered) and (
        "trajectory" in lowered or "轨迹" in query or "planning" in lowered or "路径" in query
    ):
        for hint in [
            "PPO trajectory optimization",
            "trajectory planning reinforcement learning",
            "轨迹规划 强化学习",
            "策略优化 轨迹",
            "DiffTORI PPO trajectory optimization",
        ]:
            if hint not in rewrites:
                rewrites.append(hint)
    if "compression" in lowered or "simplification" in lowered or "压缩" in query or "简化" in query:
        for hint in [
            "Douglas-Peucker trajectory simplification",
            "DOTS trajectory simplification",
            "Fast Trajectory Simplification",
            "轨迹简化 基线 算法",
        ]:
            if hint not in rewrites:
                rewrites.append(hint)
    if "similarity" in lowered or "subtrajectory" in lowered or "相似" in query:
        for hint in [
            "trajectory similarity",
            "subtrajectory search",
            "轨迹相似性",
        ]:
            if hint not in rewrites:
                rewrites.append(hint)
    return rewrites[:6]


def build_kb(raw_dir: Path | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()
    source_dir = (raw_dir or get_raw_kb_dir()).resolve()
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _safe_text(_read_document(path)).strip()
        if not text:
            continue
        title = path.stem
        doc_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_") or "doc"
        doc_type = infer_doc_type(path)
        topics = infer_topics(text[:8000], title, doc_type)
        documents.append(
            {
                "id": doc_id,
                "title": title,
                "path": str(path),
                "topics": topics,
                "doc_type": doc_type,
                "preview": _clean_text(text[:1600]),
                "content": text[:30000],
            }
        )
        for index, item in enumerate(split_document(path, text), start=1):
            chunk_text = item["text"].strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "id": f"{doc_id}_chunk_{index:03d}",
                    "doc_id": doc_id,
                    "title": title,
                    "path": str(path),
                    "topics": topics,
                    "doc_type": doc_type,
                    "section": item["section"],
                    "text": chunk_text,
                }
            )

    payload = {
        "source": {
            "title": "Local trajectory and reinforcement learning knowledge base",
            "documentCount": len(documents),
            "chunkCount": len(chunks),
            "rawDirectory": str(source_dir),
        },
        "documents": documents,
        "chunks": chunks,
    }
    get_kb_json_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    get_cached_kb.cache_clear()
    get_cached_index.cache_clear()
    return payload


def get_llama_runtime() -> tuple[Any, Any, Any, Any, Any, Any, str | None]:
    try:
        from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage, HuggingFaceEmbedding, None
    except ImportError as error:
        detail = str(error)
        if "llama_index.embeddings.huggingface" in detail:
            detail += "\nInstall llama-index-embeddings-huggingface to enable HuggingFace embeddings."
        return None, None, None, None, None, None, detail


def build_index() -> dict[str, Any]:
    Document, Settings, _, VectorStoreIndex, _, HuggingFaceEmbedding, runtime_error = get_llama_runtime()
    if runtime_error:
        raise RuntimeError(runtime_error)

    payload = get_cached_kb()
    Settings.embed_model = _build_hf_embedding(HuggingFaceEmbedding)
    chunk_rows = payload.get("chunks", [])
    documents = [
        Document(
            text=chunk["text"],
            metadata={
                "chunk_id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "topics": ", ".join(chunk.get("topics", [])),
                "path": chunk["path"],
                "section": chunk.get("section", "content"),
                "doc_type": chunk.get("doc_type", "text"),
            },
        )
        for chunk in chunk_rows
    ]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(get_index_dir()))
    get_cached_index.cache_clear()
    return {"ok": True, "documentCount": len(payload.get("documents", [])), "chunkCount": len(documents)}


@lru_cache(maxsize=1)
def get_cached_kb() -> dict[str, Any]:
    path = get_kb_json_path()
    if not path.exists():
        return build_kb()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return build_kb()
    if "chunks" not in payload or not payload.get("chunks"):
        return build_kb()
    return payload


@lru_cache(maxsize=1)
def get_cached_index() -> Any:
    _, Settings, StorageContext, _, load_index_from_storage, HuggingFaceEmbedding, runtime_error = get_llama_runtime()
    if runtime_error:
        raise RuntimeError(runtime_error)
    if not (get_index_dir() / "docstore.json").exists():
        raise RuntimeError("Knowledge index is missing. Run scripts/build_rag_index.py or scripts/ingest_local_kb.py first.")
    Settings.embed_model = _build_hf_embedding(HuggingFaceEmbedding)
    storage_context = StorageContext.from_defaults(persist_dir=str(get_index_dir()))
    return load_index_from_storage(storage_context)


def _query_terms(query: str) -> list[str]:
    lowered = query.lower().strip()
    terms = [item for item in re.split(r"[\s,;:(){}\[\]\"'/\\|]+", lowered) if len(item) >= 2]
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", lowered))
    if cjk:
        for n in (2, 3):
            for index in range(max(len(cjk) - n + 1, 0)):
                terms.append(cjk[index : index + n])
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped or ([lowered] if lowered else [])


def _research_focus_boost(item: dict[str, Any], query: str) -> float:
    focus = get_research_focus()
    if focus not in RESEARCH_FOCUS_KEYWORDS:
        return 0.0
    keywords = RESEARCH_FOCUS_KEYWORDS[focus]
    text = f"{item.get('title', '')} {item.get('section', '')} {' '.join(item.get('topics', []))} {item.get('snippet', '')}".lower()
    query_lower = query.lower()
    query_is_focus = any(keyword in query_lower for keyword in keywords)
    hit_count = sum(1 for keyword in keywords if keyword in text)
    return min(hit_count * (0.7 if query_is_focus else 0.2), 4.0 if query_is_focus else 1.5)


def _doc_type_boost(item: dict[str, Any], query: str) -> float:
    lowered = query.lower()
    asks_for_code = any(token in lowered for token in ["code", "script", "file", "function", "class", "代码", "脚本", "文件", ".py"])
    doc_type = str(item.get("doc_type", "")).lower()
    if asks_for_code:
        return 1.5 if doc_type in {"code", "markdown_note"} else 0.0
    if doc_type == "paper_pdf":
        return 0.8
    if doc_type == "code":
        return -1.2
    return 0.0


def _alias_boost(item: dict[str, Any], query: str) -> float:
    text = f"{item.get('title', '')} {item.get('path', '')} {item.get('snippet', '')}".lower()
    query_lower = query.lower()
    boost = 0.0
    for aliases in SOURCE_ALIASES.values():
        if not any(alias.lower() in query_lower for alias in aliases):
            continue
        if any(alias.lower() in text for alias in aliases):
            boost += 2.0
    return boost


def _snippet_for(text: str, terms: list[str], limit: int = 700) -> str:
    compact = _clean_text(text)
    if not compact:
        return ""
    lowered = compact.lower()
    hit = min((lowered.find(term) for term in terms if term and lowered.find(term) >= 0), default=-1)
    if hit < 0:
        return compact[:limit]
    start = max(hit - limit // 3, 0)
    end = min(start + limit, len(compact))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


@lru_cache(maxsize=1)
def _fallback_search_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for chunk in get_cached_kb().get("chunks", []):
        rows.append(
            {
                "chunk": chunk,
                "haystack": f"{chunk['title']} {chunk['section']} {chunk['text']}".lower(),
                "title": chunk["title"].lower(),
            }
        )
    return tuple(rows)


def _fallback_search(query: str, top_k: int) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    hits: list[dict[str, Any]] = []
    for row in _fallback_search_rows():
        chunk = row["chunk"]
        haystack = row["haystack"]
        title = row["title"]
        score = sum(3 if term in title else 1 for term in terms if term in haystack)
        if score <= 0:
            continue
        result = {
            "title": chunk["title"],
            "topics": chunk.get("topics", []),
            "path": chunk["path"],
            "section": chunk.get("section", "content"),
            "doc_type": chunk.get("doc_type", "text"),
            "snippet": _snippet_for(chunk["text"], terms),
            "score": float(score) + _research_focus_boost(chunk, query),
        }
        hits.append(result)
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:top_k]


def _rerank_results(results: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    ranked = []
    for item in results:
        base_score = float(item.get("score", 0.0) or 0.0)
        title = str(item.get("title", "")).lower()
        section = str(item.get("section", "")).lower()
        text = f"{title} {section} {' '.join(item.get('topics', []))} {item.get('snippet', '')}".lower()
        lexical_boost = sum(0.6 for term in terms if term in text)
        focus_boost = _research_focus_boost(item, query)
        doc_type_boost = _doc_type_boost(item, query)
        alias_boost = _alias_boost(item, query)
        item["baseScore"] = round(base_score, 4)
        item["focusBoost"] = round(focus_boost, 4)
        item["lexicalBoost"] = round(lexical_boost, 4)
        item["docTypeBoost"] = round(doc_type_boost, 4)
        item["aliasBoost"] = round(alias_boost, 4)
        item["score"] = base_score + lexical_boost + focus_boost + doc_type_boost + alias_boost
        ranked.append(item)
    ranked.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return ranked[:limit]


def search_knowledge(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    limit = top_k or get_top_k()
    rewrites = expand_query(query)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    force_fallback = os.getenv("LOCAL_RAG_FORCE_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
    if not force_fallback:
        try:
            retriever = get_cached_index().as_retriever(similarity_top_k=max(limit * 2, limit))
            for rewritten in rewrites:
                nodes = retriever.retrieve(rewritten)
                for node in nodes:
                    result = {
                        "title": node.metadata.get("title", "Untitled"),
                        "topics": [item.strip() for item in node.metadata.get("topics", "").split(",") if item.strip()],
                        "path": node.metadata.get("path", ""),
                        "section": node.metadata.get("section", "content"),
                        "doc_type": node.metadata.get("doc_type", "text"),
                        "snippet": _snippet_for(node.text[:1200], _query_terms(query)),
                        "score": float(getattr(node, "score", 0.0) or 0.0),
                    }
                    key = (result["path"], result["section"])
                    existing = merged.get(key)
                    if existing is None or result["score"] > existing.get("score", 0.0):
                        merged[key] = result
            if merged:
                return _rerank_results(list(merged.values()), query, limit)
        except Exception:
            pass

    fallback_rewrites = rewrites
    if os.getenv("LOCAL_RAG_FAST_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}:
        fallback_rewrites = rewrites[:1]

    for rewritten in fallback_rewrites:
        for item in _fallback_search(rewritten, max(limit * 2, limit)):
            key = (item["path"], item.get("section", "content"))
            existing = merged.get(key)
            if existing is None or item["score"] > existing.get("score", 0.0):
                merged[key] = item
    return _rerank_results(list(merged.values()), query, limit)


def knowledge_overview() -> dict[str, Any]:
    payload = get_cached_kb()
    docs = payload.get("documents", [])
    topic_counts: dict[str, int] = {}
    for doc in docs:
        for topic in doc.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    return {
        "title": payload.get("source", {}).get("title", "Local trajectory and reinforcement learning knowledge base"),
        "documentCount": len(docs),
        "chunkCount": len(payload.get("chunks", [])),
        "rawDirectory": payload.get("source", {}).get("rawDirectory", ""),
        "topics": topic_counts,
        "researchFocus": get_research_focus(),
        "sampleTitles": [doc["title"] for doc in docs[:8]],
    }


def build_context_block(results: list[dict[str, Any]]) -> str:
    parts = []
    char_limit = get_context_char_limit()
    for index, item in enumerate(results, start=1):
        parts.append(
            "\n".join(
                [
                    f"[Evidence {index}]",
                    f"Title: {item['title']}",
                    f"Section: {item.get('section', 'content')}",
                    f"Topics: {', '.join(item.get('topics', []))}",
                    f"Path: {item.get('path', '')}",
                    f"Excerpt: {item.get('snippet', '')}",
                ]
            )
        )
    return "\n\n".join(parts)[:char_limit]
