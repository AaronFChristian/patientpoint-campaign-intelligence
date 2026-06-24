"""
data/load_sqlite.py

Creates the SQLite database schema and bulk-loads all clean CSV data.
Also creates an empty anomaly_log table for Agent 2 to populate.

Run from project root:
    python data/load_sqlite.py

Output:
    data/campaign_data.db   (the single database file used by all agents)
"""

import os
import sys
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.getenv("DB_PATH", os.path.join(DATA_DIR, "campaign_data.db"))


# ── Schema DDL ────────────────────────────────────────────────────────────────
# All date fields stored as TEXT (ISO 8601: YYYY-MM-DD) — SQLite has no DATE type.
# Booleans stored as INTEGER (0/1) — SQLite has no BOOLEAN type.

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    health_condition   TEXT NOT NULL,
    specialty_target   TEXT NOT NULL,
    start_date         TEXT NOT NULL,
    end_date           TEXT NOT NULL,
    budget             REAL NOT NULL,
    target_completion  REAL NOT NULL,
    is_ab_demo         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS physician_offices (
    office_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    specialty  TEXT NOT NULL,
    state      TEXT NOT NULL,
    region     TEXT NOT NULL,
    tier       TEXT NOT NULL CHECK(tier IN ('A','B','C')),
    city       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign_placements (
    placement_id  TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(campaign_id),
    office_id     TEXT NOT NULL REFERENCES physician_offices(office_id),
    screen_type   TEXT NOT NULL CHECK(screen_type IN ('waiting_room','exam_room')),
    tier          TEXT NOT NULL CHECK(tier IN ('A','B','C'))
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    metric_id       TEXT PRIMARY KEY,
    placement_id    TEXT NOT NULL REFERENCES campaign_placements(placement_id),
    date            TEXT NOT NULL,
    impressions     INTEGER NOT NULL DEFAULT 0,
    completions     INTEGER NOT NULL DEFAULT 0,
    skips           INTEGER NOT NULL DEFAULT 0,
    ctr             REAL    NOT NULL DEFAULT 0.0,
    completion_rate REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS weekly_benchmarks (
    benchmark_id           TEXT PRIMARY KEY,
    campaign_id            TEXT NOT NULL REFERENCES campaigns(campaign_id),
    week_start             TEXT NOT NULL,
    week_end               TEXT NOT NULL,
    week_number            INTEGER NOT NULL,
    target_completion_rate REAL NOT NULL,
    national_avg           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS anomaly_log (
    anomaly_id     TEXT PRIMARY KEY,
    campaign_id    TEXT NOT NULL,
    office_id      TEXT NOT NULL,
    placement_id   TEXT NOT NULL,
    detected_at    TEXT NOT NULL,
    week_start     TEXT NOT NULL,
    severity       TEXT NOT NULL CHECK(severity IN ('critical','high','medium')),
    metric_value   REAL NOT NULL,
    baseline_value REAL NOT NULL,
    z_score        REAL NOT NULL,
    explanation    TEXT
);
"""

# ── Indexes — critical for Agent 2's rolling baseline queries ─────────────────
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_metrics_placement
    ON daily_metrics(placement_id);

CREATE INDEX IF NOT EXISTS idx_metrics_date
    ON daily_metrics(date);

CREATE INDEX IF NOT EXISTS idx_metrics_placement_date
    ON daily_metrics(placement_id, date);

CREATE INDEX IF NOT EXISTS idx_placements_campaign
    ON campaign_placements(campaign_id);

CREATE INDEX IF NOT EXISTS idx_placements_office
    ON campaign_placements(office_id);

CREATE INDEX IF NOT EXISTS idx_placements_tier
    ON campaign_placements(tier);

CREATE INDEX IF NOT EXISTS idx_benchmarks_campaign
    ON weekly_benchmarks(campaign_id);

CREATE INDEX IF NOT EXISTS idx_anomaly_campaign
    ON anomaly_log(campaign_id);

CREATE INDEX IF NOT EXISTS idx_anomaly_severity
    ON anomaly_log(severity);
"""


# ── Load helper ───────────────────────────────────────────────────────────────
def load_clean(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  ERROR: {filename} not found. Run clean_pipeline.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def bulk_insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame, chunksize: int = 50_000):
    """
    Uses pandas to_sql for fast bulk insert.
    'replace' on first chunk so re-runs are idempotent (it truncates then reloads).
    """
    n = len(df)
    loaded = 0
    for i, chunk_start in enumerate(range(0, n, chunksize)):
        chunk = df.iloc[chunk_start : chunk_start + chunksize]
        if_exists = "replace" if i == 0 else "append"
        chunk.to_sql(table, conn, if_exists=if_exists, index=False)
        loaded += len(chunk)
        print(f"    {table}: {loaded:,}/{n:,} rows", end="\r")
    print(f"    {table}: {loaded:,} rows loaded        ")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  PatientPoint — SQLite Loader")
    print("=" * 50)
    print(f"\n  Database path: {DB_PATH}")

    # Remove stale DB so re-runs start clean
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  Removed existing database (fresh load)")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")   # faster concurrent writes
    conn.execute("PRAGMA synchronous=NORMAL;") # safe but not slow
    conn.execute("PRAGMA foreign_keys=ON;")

    print("\n  Creating schema...")
    conn.executescript(SCHEMA)
    print("  ✓ 6 tables created (anomaly_log starts empty)")

    # ── Load order matters: parents before children ───────────────────────────
    print("\n  Bulk loading data...")
    tables = [
        ("campaigns",           "clean_campaigns.csv"),
        ("physician_offices",   "clean_physician_offices.csv"),
        ("campaign_placements", "clean_campaign_placements.csv"),
        ("daily_metrics",       "clean_daily_metrics.csv"),
        ("weekly_benchmarks",   "clean_weekly_benchmarks.csv"),
    ]
    for table_name, filename in tables:
        df = load_clean(filename)
        bulk_insert(conn, table_name, df)

    print("\n  Creating indexes...")
    conn.executescript(INDEXES)
    print("  ✓ 9 indexes created")

    conn.commit()

    # ── Verification queries ──────────────────────────────────────────────────
    print("\n  Verifying row counts:")
    verify_tables = [
        "campaigns", "physician_offices", "campaign_placements",
        "daily_metrics", "weekly_benchmarks", "anomaly_log",
    ]
    total_rows = 0
    for t in verify_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total_rows += count
        print(f"    {t:<30} {count:>10,}")

    # Quick sanity query — same one you'd run in an interview
    print("\n  Sample query — top 5 campaigns by total impressions:")
    result = conn.execute("""
        SELECT c.campaign_id,
               c.health_condition,
               SUM(dm.impressions)  AS total_impressions,
               ROUND(AVG(dm.completion_rate), 4) AS avg_completion
        FROM daily_metrics dm
        JOIN campaign_placements cp ON dm.placement_id = cp.placement_id
        JOIN campaigns c            ON cp.campaign_id  = c.campaign_id
        WHERE dm.impressions > 0
        GROUP BY c.campaign_id
        ORDER BY total_impressions DESC
        LIMIT 5
    """).fetchall()
    print(f"    {'campaign_id':<16} {'condition':<16} {'impressions':>12} {'avg_completion':>14}")
    print(f"    {'─'*16} {'─'*16} {'─'*12} {'─'*14}")
    for row in result:
        print(f"    {row[0]:<16} {row[1]:<16} {row[2]:>12,} {row[3]:>14.4f}")

    # A/B demo check
    print("\n  A/B demo validation:")
    ab = conn.execute("""
        SELECT c.campaign_id,
               ROUND(AVG(dm.completion_rate), 4) AS avg_cr,
               COUNT(DISTINCT dm.placement_id)   AS placements
        FROM daily_metrics dm
        JOIN campaign_placements cp ON dm.placement_id = cp.placement_id
        JOIN campaigns c            ON cp.campaign_id  = c.campaign_id
        WHERE c.is_ab_demo = 1 AND dm.impressions > 0
        GROUP BY c.campaign_id
    """).fetchall()
    for row in ab:
        print(f"    {row[0]:<16}  avg_cr={row[1]}  placements={row[2]}")

    conn.close()

    db_size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"\n  Database size: {db_size_mb:.1f} MB")
    print(f"\n  ✓ Done. campaign_data.db is ready for the agent pipeline.\n")
    print("=" * 50)


if __name__ == "__main__":
    main()
