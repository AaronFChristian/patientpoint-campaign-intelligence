"""
agents/sql_analyst.py

Agent 1 — SQL Analyst
Translates a natural language question into SQL, executes it against
campaign_data.db, and returns a structured result ready for Plotly rendering.

Flow:
    user_prompt → Haiku (generate SQL) → validate SELECT-only →
    execute SQLite → [retry once on error] → detect chart type → return state

The result dict written to state["sql_result"]:
    {
        "query":      str,        # the final SQL that ran
        "columns":    list[str],  # column names
        "rows":       list[list], # result rows (max 500)
        "row_count":  int,        # actual rows returned
        "chart_type": str,        # "line" | "bar" | "table" | "kpi_card"
        "x_col":      str | None, # suggested x-axis column
        "y_col":      str | None, # suggested y-axis column
    }
"""

import os
import re
import sqlite3
from typing import Optional
from dotenv import load_dotenv
import anthropic

load_dotenv()

client     = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
HAIKU      = os.getenv("HAIKU_MODEL", "claude-haiku-4-5")
DB_PATH    = os.getenv("DB_PATH", "data/campaign_data.db")
MAX_ROWS   = 500

# ── Compact schema injected into every SQL generation prompt ──────────────────
SCHEMA = """
SQLite database: campaign_data.db

Tables and columns:
  campaigns(campaign_id TEXT PK, name TEXT, health_condition TEXT,
            specialty_target TEXT, start_date TEXT, end_date TEXT,
            budget REAL, target_completion REAL, is_ab_demo INTEGER)

  physician_offices(office_id TEXT PK, name TEXT, specialty TEXT,
                    state TEXT, region TEXT, tier TEXT, city TEXT)
            -- tier values: 'A' (high volume), 'B' (mid), 'C' (low)
            -- region values: 'Northeast','Midwest','South','West'

  campaign_placements(placement_id TEXT PK, campaign_id TEXT FK,
                      office_id TEXT FK, screen_type TEXT, tier TEXT)
            -- screen_type: 'waiting_room' | 'exam_room'

  daily_metrics(metric_id TEXT PK, placement_id TEXT FK, date TEXT,
                impressions INTEGER, completions INTEGER, skips INTEGER,
                ctr REAL, completion_rate REAL)
            -- date format: 'YYYY-MM-DD'
            -- completion_rate = completions / impressions (already computed)
            -- ctr = click-through rate

  weekly_benchmarks(benchmark_id TEXT PK, campaign_id TEXT FK,
                    week_start TEXT, week_end TEXT, week_number INTEGER,
                    target_completion_rate REAL, national_avg REAL)

  anomaly_log(anomaly_id TEXT PK, campaign_id TEXT, office_id TEXT,
              placement_id TEXT, detected_at TEXT, week_start TEXT,
              severity TEXT, metric_value REAL, baseline_value REAL,
              z_score REAL, explanation TEXT)
            -- severity: 'critical' | 'high' | 'medium'

Key joins (always use these):
  daily_metrics.placement_id → campaign_placements.placement_id
  campaign_placements.campaign_id → campaigns.campaign_id
  campaign_placements.office_id  → physician_offices.office_id
  weekly_benchmarks.campaign_id  → campaigns.campaign_id
"""

SQL_SYSTEM = f"""You are an expert SQL analyst for a healthcare media analytics platform.
Generate SQLite-compatible SQL queries to answer business questions about campaign performance.

{SCHEMA}

Rules:
1. Return ONLY the SQL query — no explanation, no markdown, no backticks.
2. Always use SELECT statements only. Never use INSERT, UPDATE, DELETE, DROP, or CREATE.
3. Always filter out rows where impressions = 0 when computing rates (avoids division errors).
4. Limit results to {MAX_ROWS} rows using LIMIT.
5. Use ROUND(value, 4) for all float calculations.
6. IMPORTANT: All data is from 2025 (2025-01-01 to 2025-12-25). Never use DATE('now') — it returns 2026 dates with no data. Use explicit 2025 date literals. For "last month" use date >= '2025-11-25'. For "last week" or "recent" use date >= '2025-12-11'. For "this quarter" use date >= '2025-10-01'.
7. Always add meaningful column aliases (AS) so output is self-explanatory.
8. Use CTEs (WITH ...) for complex queries to aid readability."""


# ── SQL generation ─────────────────────────────────────────────────────────────
def _generate_sql(prompt: str, prior_error: Optional[str] = None) -> str:
    """Call Haiku to generate SQL. On retry, appends the prior error for self-correction."""
    user_content = f"Question: {prompt}"
    if prior_error:
        user_content += f"\n\nThe previous SQL attempt failed with this error:\n{prior_error}\nPlease fix the SQL."

    response = client.messages.create(
        model=HAIKU,
        max_tokens=1200,
        system=SQL_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"```sql|```", "", raw).strip()
    return raw


# ── Safety validation ──────────────────────────────────────────────────────────
def _is_safe_select(sql: str) -> bool:
    """Block any non-SELECT statements."""
    clean = sql.strip().upper()
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                 "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA")
    if not clean.startswith("SELECT") and not clean.startswith("WITH"):
        return False
    for word in forbidden:
        if re.search(rf"\b{word}\b", clean):
            return False
    return True


# ── Chart type detection ───────────────────────────────────────────────────────
def _detect_chart(columns: list, rows: list) -> tuple[str, Optional[str], Optional[str]]:
    """
    Infer the best Plotly chart type from the result shape.
    Returns: (chart_type, x_col, y_col)
    """
    if not rows:
        return "table", None, None

    n_cols = len(columns)
    n_rows = len(rows)
    cols_lower = [c.lower() for c in columns]

    # Single number → KPI card
    if n_rows == 1 and n_cols == 1:
        return "kpi_card", None, columns[0]

    # Date/week column present → line chart
    date_cols = [c for c in cols_lower if any(
        kw in c for kw in ("date", "week", "month", "iso_week", "period")
    )]
    numeric_cols = [columns[i] for i, _ in enumerate(columns)
                    if rows and isinstance(rows[0][i], (int, float))]

    if date_cols and numeric_cols:
        x = columns[cols_lower.index(date_cols[0])]
        y = numeric_cols[0]
        return "line", x, y

    # Two columns — string label + number → bar chart
    if n_cols == 2:
        if isinstance(rows[0][1], (int, float)):
            return "bar", columns[0], columns[1]
        if isinstance(rows[0][0], (int, float)):
            return "bar", columns[1], columns[0]

    # 3-column with numeric last → horizontal bar
    if n_cols == 3 and isinstance(rows[0][-1], (int, float)):
        return "bar", columns[0], columns[-1]

    # Default → table
    return "table", None, None


# ── SQLite execution ───────────────────────────────────────────────────────────
def _execute_sql(sql: str) -> tuple[list, list]:
    """Execute SQL and return (columns, rows). Raises on error."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS)
        return columns, [list(r) for r in rows]
    finally:
        conn.close()


# ── Main agent function ────────────────────────────────────────────────────────
def sql_analyst_node(state: dict) -> dict:
    """
    LangGraph node — replaces the stub in orchestrator.py.
    Generates SQL, validates, executes, retries once on error.
    """
    prompt = state["user_prompt"]
    print(f"\n[SQL Analyst] Generating SQL for: \"{prompt[:70]}\"")

    sql = None
    columns, rows = [], []
    last_error = None

    for attempt in range(1, 3):  # max 2 attempts
        try:
            sql = _generate_sql(prompt, prior_error=last_error if attempt > 1 else None)
            print(f"[SQL Analyst] Attempt {attempt} — SQL generated ({len(sql)} chars)")

            if not _is_safe_select(sql):
                raise ValueError(f"Unsafe SQL blocked: starts with {sql[:40]}")

            columns, rows = _execute_sql(sql)
            print(f"[SQL Analyst] Executed OK — {len(rows)} rows, {len(columns)} columns")
            last_error = None
            break  # success

        except Exception as e:
            last_error = str(e)
            print(f"[SQL Analyst] Attempt {attempt} failed: {last_error[:120]}")
            if attempt == 2:
                return {
                    **state,
                    "error": f"SQL Analyst failed after 2 attempts: {last_error}",
                    "sql_result": None,
                }

    chart_type, x_col, y_col = _detect_chart(columns, rows)
    print(f"[SQL Analyst] Chart type → {chart_type}  (x={x_col}, y={y_col})")

    return {
        **state,
        "sql_result": {
            "query":      sql,
            "columns":    columns,
            "rows":       rows,
            "row_count":  len(rows),
            "chart_type": chart_type,
            "x_col":      x_col,
            "y_col":      y_col,
        },
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_prompts = [
        "Which region has the highest average campaign completion rate?",
        "Show me the top 5 campaigns by total impressions",
        "What is the week-over-week completion rate trend for CAMP_035 over the last 6 weeks?",
    ]

    print("=" * 65)
    print("Agent 1 — SQL Analyst standalone test")
    print("=" * 65)

    for prompt in test_prompts:
        print(f"\nQ: {prompt}")
        result = sql_analyst_node({
            "user_prompt":     prompt,
            "intent":          "ask_question",
            "sql_result":      None,
            "anomalies":       None,
            "report_md":       None,
            "error":           None,
            "iteration_count": 0,
        })
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            continue
        sr = result["sql_result"]
        print(f"  SQL      : {sr['query'][:120]}...")
        print(f"  Rows     : {sr['row_count']}  Columns: {sr['columns']}")
        print(f"  Chart    : {sr['chart_type']}  (x={sr['x_col']}, y={sr['y_col']})")
        if sr["rows"]:
            print(f"  First row: {sr['rows'][0]}")
