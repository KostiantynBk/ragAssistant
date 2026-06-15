import os
import sqlite3
from contextlib import contextmanager

from src.api.schemas import EvalResult

_DB_PATH = os.getenv("SQLITE_DB_PATH", "db/eval.db")

_DDL = """
CREATE TABLE IF NOT EXISTS queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    question     TEXT,
    chunking     TEXT,
    use_reranker INTEGER,
    answer       TEXT,
    grounded     INTEGER,
    latency_ms   REAL,
    timestamp    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    config         TEXT,
    hit_at_1       INTEGER,
    hit_at_5       INTEGER,
    hit_at_10      INTEGER,
    mrr            REAL,
    faithfulness   REAL,
    avg_latency_ms REAL,
    timestamp      TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def _conn(db_path: str = _DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_DDL)
        yield con
        con.commit()
    finally:
        con.close()


def log_query(
    question: str,
    chunking: str,
    use_reranker: bool,
    answer: str,
    grounded: bool,
    latency_ms: float,
    db_path: str = _DB_PATH,
) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO queries (question, chunking, use_reranker, answer, grounded, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, chunking, int(use_reranker), answer, int(grounded), latency_ms),
        )
        return cur.lastrowid


def log_eval(
    config: str,
    hit_at_1: bool,
    hit_at_5: bool,
    hit_at_10: bool,
    mrr: float,
    faithfulness: float,
    avg_latency_ms: float,
    db_path: str = _DB_PATH,
) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO eval_results "
            "(config, hit_at_1, hit_at_5, hit_at_10, mrr, faithfulness, avg_latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (config, int(hit_at_1), int(hit_at_5), int(hit_at_10), mrr, faithfulness, avg_latency_ms),
        )
        return cur.lastrowid


def get_latest_evals(n: int = 10, db_path: str = _DB_PATH) -> list[EvalResult]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM eval_results ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [EvalResult(**dict(r)) for r in rows]
