from __future__ import annotations

import json
import site
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from backend.settings import (
    get_db_backend,
    get_mysql_database,
    get_mysql_host,
    get_mysql_password,
    get_mysql_port,
    get_mysql_user,
    mysql_enabled,
)


def _ensure_db_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    for candidate in [root / "dbvendor_manual", root / "dbvendor", root / ".dbvendor"]:
        if candidate.exists():
            site.addsitedir(str(candidate))
    try:
        site.addsitedir(site.getusersitepackages())
    except Exception:
        pass


_ensure_db_paths()

try:
    import pymysql
    from pymysql.cursors import DictCursor

    PYMYSQL_AVAILABLE = True
except Exception:
    pymysql = None  # type: ignore[assignment]
    DictCursor = None  # type: ignore[assignment]
    PYMYSQL_AVAILABLE = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def database_ready() -> bool:
    return mysql_enabled() and PYMYSQL_AVAILABLE


def database_status() -> dict[str, Any]:
    if get_db_backend() != "mysql":
        return {"backend": "file", "enabled": False, "ready": False, "detail": "Using file storage"}
    if not PYMYSQL_AVAILABLE:
        return {"backend": "mysql", "enabled": True, "ready": False, "detail": "Missing pymysql"}
    try:
        init_db()
        return {
            "backend": "mysql",
            "enabled": True,
            "ready": True,
            "detail": f"{get_mysql_user()}@{get_mysql_host()}:{get_mysql_port()}/{get_mysql_database()}",
        }
    except Exception as error:
        return {"backend": "mysql", "enabled": True, "ready": False, "detail": str(error)}


def _connect(database: str | None = None):
    if not PYMYSQL_AVAILABLE:
        raise RuntimeError("pymysql is not installed.")
    return pymysql.connect(
        host=get_mysql_host(),
        port=get_mysql_port(),
        user=get_mysql_user(),
        password=get_mysql_password(),
        database=database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


def init_db() -> bool:
    if not database_ready():
        return False

    database_name = get_mysql_database().replace("`", "")
    admin_conn = _connect(None)
    try:
        with admin_conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        admin_conn.commit()
    finally:
        admin_conn.close()

    app_conn = _connect(database_name)
    try:
        with app_conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id VARCHAR(64) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    summary TEXT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    session_id VARCHAR(64) NOT NULL,
                    position INT NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content LONGTEXT NOT NULL,
                    token_count INT NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL,
                    INDEX idx_chat_messages_session (session_id, position),
                    CONSTRAINT fk_chat_messages_session
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try:
                cursor.execute("ALTER TABLE chat_messages ADD COLUMN token_count INT NOT NULL DEFAULT 0")
            except Exception:
                pass
            try:
                cursor.execute("CREATE FULLTEXT INDEX idx_chat_messages_content_ft ON chat_messages (content)")
            except Exception:
                pass
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    conversation_id VARCHAR(64) NOT NULL,
                    summary_type VARCHAR(32) NOT NULL,
                    start_message_id BIGINT NOT NULL,
                    end_message_id BIGINT NOT NULL,
                    content LONGTEXT NOT NULL,
                    structured_json JSON NULL,
                    token_count INT NOT NULL DEFAULT 0,
                    source_message_ids JSON NULL,
                    model VARCHAR(255) NULL,
                    version VARCHAR(64) NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    INDEX idx_conversation_summaries_conv (conversation_id),
                    INDEX idx_conversation_summaries_range (start_message_id, end_message_id),
                    INDEX idx_conversation_summaries_type (summary_type),
                    CONSTRAINT fk_conversation_summaries_session
                        FOREIGN KEY (conversation_id) REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            try:
                cursor.execute("CREATE FULLTEXT INDEX idx_conversation_summaries_content_ft ON conversation_summaries (content)")
            except Exception:
                pass
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_states (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    conversation_id VARCHAR(64) NOT NULL UNIQUE,
                    state_json JSON NOT NULL,
                    token_count INT NOT NULL DEFAULT 0,
                    last_compressed_message_id BIGINT NOT NULL DEFAULT 0,
                    version VARCHAR(64) NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    INDEX idx_conversation_states_conv (conversation_id),
                    INDEX idx_conversation_states_last (last_compressed_message_id),
                    CONSTRAINT fk_conversation_states_session
                        FOREIGN KEY (conversation_id) REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS context_build_logs (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    conversation_id VARCHAR(64) NOT NULL,
                    request_id VARCHAR(64) NOT NULL,
                    selected_recent_message_ids JSON NULL,
                    selected_summary_ids JSON NULL,
                    selected_relevant_message_ids JSON NULL,
                    total_input_tokens INT NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL,
                    INDEX idx_context_build_logs_conv (conversation_id),
                    INDEX idx_context_build_logs_request (request_id),
                    CONSTRAINT fk_context_build_logs_session
                        FOREIGN KEY (conversation_id) REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_artifacts (
                    id VARCHAR(64) PRIMARY KEY,
                    conversation_id VARCHAR(64) NULL,
                    prompt LONGTEXT NOT NULL,
                    positive_prompt LONGTEXT NULL,
                    negative_prompt LONGTEXT NULL,
                    preset VARCHAR(128) NULL,
                    workflow_name VARCHAR(255) NULL,
                    model VARCHAR(255) NULL,
                    seed VARCHAR(64) NULL,
                    width INT NOT NULL DEFAULT 0,
                    height INT NOT NULL DEFAULT 0,
                    file_path VARCHAR(1024) NOT NULL,
                    public_url VARCHAR(1024) NOT NULL,
                    score DOUBLE NULL,
                    quality_report_json JSON NULL,
                    generation_plan_json JSON NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_image_artifacts_conversation (conversation_id),
                    INDEX idx_image_artifacts_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    eval_type VARCHAR(64) NOT NULL,
                    dataset_path VARCHAR(1024) NOT NULL,
                    output_path VARCHAR(1024) NOT NULL,
                    metrics_json LONGTEXT NOT NULL,
                    details_json LONGTEXT NOT NULL,
                    created_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    run_type VARCHAR(64) NOT NULL,
                    model_path VARCHAR(1024) NOT NULL,
                    data_path VARCHAR(1024) NOT NULL,
                    output_path VARCHAR(1024) NOT NULL,
                    status VARCHAR(64) NOT NULL,
                    notes LONGTEXT NULL,
                    metrics_json LONGTEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_rl_runs (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    run_name VARCHAR(255) NOT NULL,
                    data_path VARCHAR(1024) NOT NULL,
                    output_path VARCHAR(1024) NOT NULL,
                    status VARCHAR(64) NOT NULL,
                    metrics_json LONGTEXT NOT NULL,
                    evaluation_json LONGTEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_rl_episodes (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    run_id BIGINT NOT NULL,
                    episode_kind VARCHAR(64) NOT NULL,
                    episode_index INT NOT NULL,
                    payload_json LONGTEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_retrieval_rl_episodes_run (run_id, episode_kind, episode_index),
                    CONSTRAINT fk_retrieval_rl_episodes_run
                        FOREIGN KEY (run_id) REFERENCES retrieval_rl_runs(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            # 核心6/7：地图演示结果表。rl_method 记录 DQN/PPO，compression_method 和 compression_json 记录 S3/RLTS/Mlsimp。
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectory_runs (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    trajectory_type VARCHAR(64) NOT NULL,
                    scenario_id VARCHAR(128) NULL,
                    scenario_label VARCHAR(255) NULL,
                    rl_method VARCHAR(64) NULL,
                    compression_method VARCHAR(64) NULL,
                    map_provider VARCHAR(128) NOT NULL,
                    route_provider VARCHAR(128) NOT NULL,
                    start_lat DOUBLE NOT NULL,
                    start_lng DOUBLE NOT NULL,
                    end_lat DOUBLE NOT NULL,
                    end_lng DOUBLE NOT NULL,
                    distance_km DOUBLE NOT NULL,
                    duration_min DOUBLE NOT NULL,
                    route_geometry_json LONGTEXT NOT NULL,
                    compression_json LONGTEXT NOT NULL,
                    metadata_json LONGTEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_trajectory_runs_created (created_at),
                    INDEX idx_trajectory_runs_type (trajectory_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        app_conn.commit()
    finally:
        app_conn.close()
    return True


@contextmanager
def connection_scope() -> Iterator[Any]:
    if not database_ready():
        raise RuntimeError("MySQL backend is not enabled or pymysql is missing.")
    init_db()
    conn = _connect(get_mysql_database())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_eval_run(eval_type: str, dataset_path: str, output_path: str, metrics: dict[str, Any], details: list[dict[str, Any]]) -> None:
    if not database_ready():
        return
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO eval_runs (eval_type, dataset_path, output_path, metrics_json, details_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    eval_type,
                    dataset_path,
                    output_path,
                    json.dumps(metrics, ensure_ascii=False),
                    json.dumps(details, ensure_ascii=False),
                    datetime.now(),
                ),
            )


def save_training_run(
    run_type: str,
    model_path: str,
    data_path: str,
    output_path: str,
    status: str,
    notes: str = "",
    metrics: dict[str, Any] | None = None,
) -> None:
    if not database_ready():
        return
    now = datetime.now()
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO training_runs (run_type, model_path, data_path, output_path, status, notes, metrics_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_type,
                    model_path,
                    data_path,
                    output_path,
                    status,
                    notes,
                    json.dumps(metrics or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )


def create_retrieval_rl_run(run_name: str, data_path: str, output_path: str, status: str = "started") -> int | None:
    if not database_ready():
        return None
    now = datetime.now()
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO retrieval_rl_runs (run_name, data_path, output_path, status, metrics_json, evaluation_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_name, data_path, output_path, status, "{}", "{}", now, now),
            )
            return int(cursor.lastrowid)


def finalize_retrieval_rl_run(
    run_id: int,
    status: str,
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    trace: list[dict[str, Any]],
) -> None:
    if not database_ready():
        return
    now = datetime.now()
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE retrieval_rl_runs
                SET status=%s, metrics_json=%s, evaluation_json=%s, updated_at=%s
                WHERE id=%s
                """,
                (status, json.dumps(metrics, ensure_ascii=False), json.dumps(evaluation, ensure_ascii=False), now, run_id),
            )
            cursor.execute("DELETE FROM retrieval_rl_episodes WHERE run_id=%s", (run_id,))
            for index, row in enumerate(trace, start=1):
                cursor.execute(
                    """
                    INSERT INTO retrieval_rl_episodes (run_id, episode_kind, episode_index, payload_json, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (run_id, "training_trace", index, json.dumps(row, ensure_ascii=False), now),
                )
            trained_episodes = evaluation.get("trained_policy", {}).get("episodes", [])
            for index, row in enumerate(trained_episodes, start=1):
                cursor.execute(
                    """
                    INSERT INTO retrieval_rl_episodes (run_id, episode_kind, episode_index, payload_json, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (run_id, "trained_episode", index, json.dumps(row, ensure_ascii=False), now),
                )
            baseline_episodes = evaluation.get("baseline_policy", {}).get("episodes", [])
            for index, row in enumerate(baseline_episodes, start=1):
                cursor.execute(
                    """
                    INSERT INTO retrieval_rl_episodes (run_id, episode_kind, episode_index, payload_json, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (run_id, "baseline_episode", index, json.dumps(row, ensure_ascii=False), now),
                )


def fail_retrieval_rl_run(run_id: int, notes: str) -> None:
    if not database_ready():
        return
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE retrieval_rl_runs
                SET status=%s, metrics_json=%s, evaluation_json=%s, updated_at=%s
                WHERE id=%s
                """,
                ("failed", json.dumps({}, ensure_ascii=False), json.dumps({"error": notes}, ensure_ascii=False), datetime.now(), run_id),
            )


def get_latest_retrieval_rl_run() -> dict[str, Any] | None:
    if not database_ready():
        return None
    init_db()
    conn = _connect(get_mysql_database())
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, run_name, data_path, output_path, status, metrics_json, evaluation_json, created_at, updated_at
                FROM retrieval_rl_runs
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                return None
            run_id = int(row["id"])
            cursor.execute(
                """
                SELECT episode_kind, episode_index, payload_json
                FROM retrieval_rl_episodes
                WHERE run_id=%s
                ORDER BY episode_kind ASC, episode_index ASC
                """,
                (run_id,),
            )
            grouped: dict[str, list[dict[str, Any]]] = {}
            for episode in cursor.fetchall():
                grouped.setdefault(str(episode["episode_kind"]), []).append(json.loads(episode["payload_json"]))
            return {
                "id": run_id,
                "run_name": row["run_name"],
                "data_path": row["data_path"],
                "output_path": row["output_path"],
                "status": row["status"],
                "metrics": json.loads(row["metrics_json"] or "{}"),
                "evaluation": json.loads(row["evaluation_json"] or "{}"),
                "episodes": grouped,
                "created_at": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else "",
                "updated_at": row["updated_at"].isoformat(timespec="seconds") if row["updated_at"] else "",
            }
    finally:
        conn.close()


# 核心6/7：地图轨迹和算法叠加结果的数据库写入函数。
def save_trajectory_run(
    trajectory_type: str,
    scenario_id: str,
    scenario_label: str,
    rl_method: str,
    compression_method: str,
    map_provider: str,
    route_provider: str,
    start_coords: list[float],
    end_coords: list[float],
    distance_km: float,
    duration_min: float,
    route_geometry: list[list[float]],
    compression: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    if not database_ready():
        return None
    with connection_scope() as conn:
        with conn.cursor() as cursor:
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
                    scenario_id or None,
                    scenario_label or None,
                    rl_method or None,
                    compression_method or None,
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
            return int(cursor.lastrowid)


# 核心6/7：读取最近地图实验记录，供前端“最近轨迹”面板展示。
def list_recent_trajectory_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not database_ready():
        return []
    init_db()
    conn = _connect(get_mysql_database())
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, trajectory_type, scenario_id, scenario_label, rl_method, compression_method,
                       map_provider, route_provider, start_lat, start_lng, end_lat, end_lng,
                       distance_km, duration_min, route_geometry_json, compression_json, metadata_json, created_at
                FROM trajectory_runs
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": int(row["id"]),
                    "trajectoryType": row["trajectory_type"],
                    "scenarioId": row["scenario_id"] or "",
                    "scenarioLabel": row["scenario_label"] or "",
                    "rlMethod": row["rl_method"] or "",
                    "compressionMethod": row["compression_method"] or "",
                    "mapProvider": row["map_provider"],
                    "routeProvider": row["route_provider"],
                    "start": [float(row["start_lat"]), float(row["start_lng"])],
                    "end": [float(row["end_lat"]), float(row["end_lng"])],
                    "distanceKm": float(row["distance_km"]),
                    "durationMin": float(row["duration_min"]),
                    "routeGeometry": json.loads(row["route_geometry_json"] or "[]"),
                    "compression": json.loads(row["compression_json"] or "{}"),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "createdAt": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else "",
                }
                for row in rows
            ]
    finally:
        conn.close()
