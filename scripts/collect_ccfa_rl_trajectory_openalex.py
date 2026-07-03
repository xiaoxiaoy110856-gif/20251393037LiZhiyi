from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_ccfa_rl_trajectory_papers import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    QUERY_TEMPLATES,
    download_pdf,
    existing_pdf_count,
    existing_title_keys,
    keyword_match,
    match_ccf_a_venue,
    normalize,
    paper_key,
    pdf_url,
    safe_filename,
    write_manifest_row,
)


def inverted_abstract_to_text(index: dict[str, Any] | None) -> str:
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, offsets in index.items():
        if isinstance(offsets, list):
            for offset in offsets:
                try:
                    positions.append((int(offset), str(token)))
                except Exception:
                    continue
    return " ".join(token for _, token in sorted(positions))


def source_names(location: dict[str, Any] | None) -> list[str]:
    if not isinstance(location, dict):
        return []
    source = location.get("source") or {}
    if not isinstance(source, dict):
        return []
    names = [str(source.get("display_name") or "")]
    names.extend(str(item) for item in source.get("alternate_titles") or [])
    return [item for item in names if item.strip()]


def best_pdf_url(row: dict[str, Any]) -> str:
    for key in ["best_oa_location", "primary_location"]:
        location = row.get(key) or {}
        if isinstance(location, dict):
            url = str(location.get("pdf_url") or "").strip()
            if url:
                return url
    for location in row.get("locations") or []:
        if isinstance(location, dict):
            url = str(location.get("pdf_url") or "").strip()
            if url:
                return url
    return ""


def openalex_search(query: str, per_page: int, cursor: str) -> tuple[list[dict[str, Any]], str]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "open_access.is_oa:true",
            "per-page": min(per_page, 200),
            "cursor": cursor,
            "select": ",".join(
                [
                    "id",
                    "doi",
                    "display_name",
                    "publication_year",
                    "primary_location",
                    "best_oa_location",
                    "locations",
                    "open_access",
                    "abstract_inverted_index",
                ]
            ),
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "trl-ccfa-openalex-collector/1.0"})
    with urllib.request.urlopen(req, timeout=40) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    results = payload.get("results", [])
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    return (results if isinstance(results, list) else []), str(meta.get("next_cursor") or "")


def to_semantic_like(row: dict[str, Any]) -> dict[str, Any]:
    names: list[str] = []
    names.extend(source_names(row.get("primary_location")))
    names.extend(source_names(row.get("best_oa_location")))
    for location in row.get("locations") or []:
        names.extend(source_names(location))
    seen: set[str] = set()
    names = [name for name in names if not (normalize(name) in seen or seen.add(normalize(name)))]
    return {
        "paperId": str(row.get("id") or ""),
        "title": str(row.get("display_name") or ""),
        "abstract": inverted_abstract_to_text(row.get("abstract_inverted_index")),
        "year": row.get("publication_year"),
        "venue": " ".join(names),
        "publicationVenue": {"name": " ".join(names)},
        "url": row.get("id"),
        "externalIds": {"DOI": row.get("doi"), "OpenAlex": row.get("id")},
        "openAccessPdf": {"url": best_pdf_url(row)},
    }


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
        cursor = "*"
        for _page in range(args.max_pages_per_query):
            if downloaded >= remaining or not cursor:
                break
            try:
                rows, cursor = openalex_search(query, per_page=args.per_page, cursor=cursor)
            except Exception as error:
                key = f"openalex_error:{str(error)[:120]}"
                errors[key] = errors.get(key, 0) + 1
                time.sleep(max(args.sleep, 2.0))
                break
            if not rows:
                break
            for raw in rows:
                if downloaded >= remaining:
                    break
                row = to_semantic_like(raw)
                key = paper_key(row)
                if key in seen_papers:
                    continue
                seen_papers.add(key)

                venue = match_ccf_a_venue(row)
                if not venue or not keyword_match(row):
                    rejected += 1
                    continue

                title = str(row.get("title") or "").strip()
                title_key = normalize(safe_filename(title))
                if not title or title_key in existing_keys:
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
                    "source": "openalex",
                    "title": title,
                    "year": row.get("year"),
                    "venue": venue,
                    "openalex_venue_text": row.get("venue"),
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
                    existing_keys.add(title_key)
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
    parser = argparse.ArgumentParser(description="Collect public CCF-A RL trajectory papers from OpenAlex.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-total", type=int, default=200)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--max-pages-per-query", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--query", action="append", dest="queries", default=[])
    args = parser.parse_args()
    args.queries = args.queries or QUERY_TEMPLATES
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
