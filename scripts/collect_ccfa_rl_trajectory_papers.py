from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kb" / "raw" / "trajectory_papers"
DEFAULT_MANIFEST = ROOT / "outputs" / "paper_crawl" / "ccfa_rl_trajectory_manifest.jsonl"


CCF_A_VENUES_2026: dict[str, list[str]] = {
    "AAAI": ["aaai", "aaai conference on artificial intelligence"],
    "NeurIPS": ["neurips", "nips", "neural information processing systems"],
    "ACL": ["acl", "annual meeting of the association for computational linguistics"],
    "CVPR": ["cvpr", "computer vision and pattern recognition"],
    "ICCV": ["iccv", "international conference on computer vision"],
    "ICML": ["icml", "international conference on machine learning"],
    "ICLR": ["iclr", "international conference on learning representations"],
    "SIGMOD": ["sigmod", "international conference on management of data"],
    "VLDB": ["vldb", "very large data bases", "proceedings of the vldb endowment", "pvldb"],
    "ICDE": ["icde", "international conference on data engineering"],
    "KDD": ["kdd", "knowledge discovery and data mining"],
    "SIGIR": ["sigir", "research and development in information retrieval"],
    "WWW": ["www", "the web conference", "world wide web conference"],
    "TKDE": ["ieee transactions on knowledge and data engineering", "tkde"],
    "TPAMI": ["ieee transactions on pattern analysis and machine intelligence", "tpami", "pami"],
    "IJCV": ["international journal of computer vision", "ijcv"],
    "JMLR": ["journal of machine learning research", "jmlr"],
    "TODS": ["acm transactions on database systems", "tods"],
    "TOIS": ["acm transactions on information systems", "tois"],
    "VLDBJ": ["vldb journal", "the vldb journal"],
}


QUERY_TEMPLATES = [
    "reinforcement learning trajectory",
    "deep reinforcement learning trajectory",
    "reinforcement learning trajectory planning",
    "reinforcement learning trajectory optimization",
    "reinforcement learning trajectory prediction",
    "reinforcement learning trajectory generation",
    "reinforcement learning trajectory compression",
    "reinforcement learning trajectory simplification",
    "reinforcement learning motion planning",
    "reinforcement learning path planning",
    "reinforcement learning autonomous driving trajectory",
    "offline reinforcement learning trajectory",
    "multi agent reinforcement learning trajectory",
    "PPO trajectory optimization",
    "SAC trajectory planning",
    "DQN trajectory",
]

for _venue in [
    "AAAI",
    "NeurIPS",
    "ACL",
    "CVPR",
    "ICCV",
    "ICML",
    "ICLR",
    "SIGMOD",
    "VLDB",
    "ICDE",
    "KDD",
    "SIGIR",
    "The Web Conference",
    "TKDE",
    "TPAMI",
    "JMLR",
]:
    QUERY_TEMPLATES.extend(
        [
            f"{_venue} reinforcement learning trajectory",
            f"{_venue} deep reinforcement learning trajectory planning",
            f"{_venue} reinforcement learning autonomous driving trajectory",
        ]
    )


RL_TERMS = [
    "reinforcement learning",
    "deep reinforcement learning",
    "offline reinforcement learning",
    "multi-agent reinforcement learning",
    "policy gradient",
    "policy optimization",
    "actor-critic",
    "ppo",
    "proximal policy optimization",
    "sac",
    "soft actor",
    "dqn",
    "q-learning",
]


TRAJECTORY_TERMS = [
    "trajectory",
    "trajectories",
    "path planning",
    "motion planning",
    "trajectory planning",
    "trajectory optimization",
    "trajectory prediction",
    "trajectory generation",
    "trajectory compression",
    "trajectory simplification",
    "autonomous driving",
    "vehicle routing",
    "mobility",
]


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def safe_filename(text: str, max_len: int = 130) -> str:
    cleaned = re.sub(r"[^\w\s.\-()\[\]]+", " ", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    if not cleaned:
        cleaned = "paper"
    return cleaned[:max_len].rstrip(" ._-")


def semantic_scholar_search(query: str, limit: int, offset: int = 0, retries: int = 2) -> list[dict[str, Any]]:
    fields = ",".join(
        [
            "paperId",
            "title",
            "abstract",
            "year",
            "venue",
            "publicationVenue",
            "url",
            "externalIds",
            "authors",
            "openAccessPdf",
            "isOpenAccess",
        ]
    )
    params = urllib.parse.urlencode({"query": query, "limit": min(limit, 100), "offset": offset, "fields": fields})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "trl-ccfa-open-paper-collector/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:200]
            if error.code == 429 and attempt < retries:
                time.sleep(10 + attempt * 10)
                continue
            raise RuntimeError(f"HTTP {error.code}: {body}") from error
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def venue_text(row: dict[str, Any]) -> str:
    venue = str(row.get("venue") or "")
    publication_venue = row.get("publicationVenue") or {}
    if isinstance(publication_venue, dict):
        venue = " ".join([venue, str(publication_venue.get("name") or ""), str(publication_venue.get("alternate_names") or "")])
    return normalize(venue)


def match_ccf_a_venue(row: dict[str, Any]) -> str | None:
    text = venue_text(row)
    if not text:
        return None
    for short_name, aliases in CCF_A_VENUES_2026.items():
        for alias in aliases:
            alias_norm = normalize(alias)
            if alias_norm and alias_norm in text:
                return short_name
    return None


def keyword_match(row: dict[str, Any]) -> bool:
    text = normalize(f"{row.get('title') or ''} {row.get('abstract') or ''}")
    has_rl = any(term in text for term in RL_TERMS)
    has_trajectory = any(term in text for term in TRAJECTORY_TERMS)
    return has_rl and has_trajectory


def paper_key(row: dict[str, Any]) -> str:
    external = row.get("externalIds") or {}
    if isinstance(external, dict):
        for key in ["DOI", "ArXiv", "MAG", "CorpusId"]:
            value = str(external.get(key) or "").strip().lower()
            if value:
                return f"{key.lower()}:{value}"
    paper_id = str(row.get("paperId") or "").strip().lower()
    if paper_id:
        return f"s2:{paper_id}"
    return "title:" + hashlib.sha1(normalize(str(row.get("title") or "")).encode("utf-8")).hexdigest()


def pdf_url(row: dict[str, Any]) -> str:
    pdf = row.get("openAccessPdf") or {}
    if isinstance(pdf, dict):
        url = str(pdf.get("url") or "").strip()
        if url:
            return url.replace("/abs/", "/pdf/") if "arxiv.org/abs/" in url else url
    external = row.get("externalIds") or {}
    if isinstance(external, dict) and external.get("ArXiv"):
        return f"https://arxiv.org/pdf/{external['ArXiv']}.pdf"
    return ""


def existing_pdf_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*.pdf") if item.is_file())


def existing_title_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for item in path.rglob("*.pdf"):
        keys.add(normalize(item.stem))
    return keys


def download_pdf(url: str, output: Path, sleep_seconds: float) -> tuple[bool, str, int]:
    if not url:
        return False, "missing_pdf_url", 0
    req = urllib.request.Request(url, headers={"User-Agent": "trl-ccfa-open-paper-collector/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            content_type = str(response.headers.get("content-type") or "").lower()
            data = response.read()
    except urllib.error.HTTPError as error:
        return False, f"http_{error.code}", 0
    except Exception as error:
        return False, f"download_error:{type(error).__name__}", 0

    if b"%PDF" not in data[:2048] and "pdf" not in content_type:
        return False, "not_pdf", len(data)
    if len(data) < 10_000:
        return False, "too_small", len(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return True, "downloaded", len(data)


def write_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    manifest_path: Path = args.manifest
    output_dir.mkdir(parents=True, exist_ok=True)

    current_count = existing_pdf_count(output_dir)
    remaining = max(args.target_total - current_count, 0)
    existing_keys = existing_title_keys(output_dir)
    seen_papers: set[str] = set()
    downloaded = 0
    accepted = 0
    rejected = 0
    errors: dict[str, int] = {}

    if remaining <= 0:
        return {
            "ok": True,
            "current_pdf_count": current_count,
            "downloaded": 0,
            "target_total": args.target_total,
            "detail": "target already satisfied",
        }

    for query in args.queries:
        if downloaded >= remaining:
            break
        for offset in range(0, args.max_results_per_query, 100):
            if downloaded >= remaining:
                break
            try:
                rows = semantic_scholar_search(query, limit=min(100, args.max_results_per_query - offset), offset=offset)
            except Exception as error:
                key = f"search_error:{str(error)[:120]}"
                errors[key] = errors.get(key, 0) + 1
                time.sleep(args.sleep)
                continue
            if not rows:
                break
            for row in rows:
                if downloaded >= remaining:
                    break
                key = paper_key(row)
                if key in seen_papers:
                    continue
                seen_papers.add(key)

                venue = match_ccf_a_venue(row)
                if not venue or not keyword_match(row):
                    rejected += 1
                    continue

                title = str(row.get("title") or "").strip()
                if not title or normalize(safe_filename(title)) in existing_keys:
                    rejected += 1
                    continue

                url = pdf_url(row)
                if not url:
                    rejected += 1
                    continue

                year = str(row.get("year") or "unknown")
                file_name = safe_filename(f"{venue}_{year}_{title}") + ".pdf"
                pdf_path = output_dir / "ccfa_rl_trajectory" / file_name
                if pdf_path.exists():
                    rejected += 1
                    continue

                ok, status, size = download_pdf(url, pdf_path, args.sleep)
                manifest_row = {
                    "status": status,
                    "downloaded": ok,
                    "title": title,
                    "year": row.get("year"),
                    "venue": venue,
                    "semantic_scholar_venue": row.get("venue"),
                    "paperId": row.get("paperId"),
                    "externalIds": row.get("externalIds"),
                    "url": row.get("url"),
                    "pdf_url": url,
                    "file": str(pdf_path) if ok else "",
                    "bytes": size,
                    "query": query,
                }
                write_manifest_row(manifest_path, manifest_row)
                accepted += 1
                if ok:
                    downloaded += 1
                    existing_keys.add(normalize(safe_filename(title)))
                else:
                    errors[status] = errors.get(status, 0) + 1
            time.sleep(args.sleep)

    return {
        "ok": True,
        "target_total": args.target_total,
        "initial_pdf_count": current_count,
        "final_pdf_count": existing_pdf_count(output_dir),
        "downloaded": downloaded,
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public CCF-A reinforcement-learning trajectory papers.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-total", type=int, default=200, help="Total PDF limit including existing PDFs under output-dir.")
    parser.add_argument("--max-results-per-query", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--query", action="append", dest="queries", default=[])
    args = parser.parse_args()
    args.queries = args.queries or QUERY_TEMPLATES
    return args


def main() -> None:
    args = parse_args()
    summary = collect(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
