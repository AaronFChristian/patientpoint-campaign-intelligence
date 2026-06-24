"""
agents/report_writer.py

Agent 3 — Report Writer
Takes sql_result (campaign KPI data) and anomalies (from Agent 2) from state,
calls Claude Sonnet to produce a structured, client-ready markdown report.

Uses Sonnet (not Haiku) because this output goes to pharma brand partners —
narrative quality and clinical tone matter here. This is the one node where
the premium model earns its cost.

Run standalone from project root:
    python agents/report_writer.py
"""

import os
import json
import sqlite3
from datetime import date, datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
SONNET  = os.getenv("SONNET_MODEL", "claude-sonnet-4-6")
DB_PATH = os.getenv("DB_PATH", "data/campaign_data.db")

MAX_ANOMALIES_IN_REPORT = 8    # top N anomalies to include in the report
MAX_SQL_ROWS_IN_REPORT  = 10   # max rows from sql_result to include


# ── Pull campaign summary from DB (used when sql_result is sparse) ─────────────
def _pull_campaign_summary(campaign_id: str = None) -> dict:
    """
    Fetches a high-level campaign summary from SQLite to ground the report.
    If campaign_id is provided, pulls that campaign specifically.
    Otherwise pulls the top 5 campaigns by impressions for a portfolio summary.
    """
    conn = sqlite3.connect(DB_PATH)

    if campaign_id:
        rows = conn.execute("""
            SELECT c.campaign_id, c.name, c.health_condition, c.specialty_target,
                   c.start_date, c.end_date, c.target_completion,
                   COUNT(DISTINCT cp.office_id)         AS offices,
                   SUM(dm.impressions)                  AS total_impressions,
                   ROUND(AVG(dm.completion_rate), 4)    AS avg_completion,
                   ROUND(c.target_completion, 4)        AS kpi_target,
                   ROUND(AVG(dm.completion_rate) - c.target_completion, 4) AS delta_vs_kpi
            FROM campaigns c
            JOIN campaign_placements cp ON c.campaign_id  = cp.campaign_id
            JOIN daily_metrics dm       ON cp.placement_id = dm.placement_id
            WHERE c.campaign_id = ? AND dm.impressions > 0
            GROUP BY c.campaign_id
        """, (campaign_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT c.campaign_id, c.name, c.health_condition, c.specialty_target,
                   c.start_date, c.end_date, c.target_completion,
                   COUNT(DISTINCT cp.office_id)         AS offices,
                   SUM(dm.impressions)                  AS total_impressions,
                   ROUND(AVG(dm.completion_rate), 4)    AS avg_completion,
                   ROUND(c.target_completion, 4)        AS kpi_target,
                   ROUND(AVG(dm.completion_rate) - c.target_completion, 4) AS delta_vs_kpi
            FROM campaigns c
            JOIN campaign_placements cp ON c.campaign_id  = cp.campaign_id
            JOIN daily_metrics dm       ON cp.placement_id = dm.placement_id
            WHERE dm.impressions > 0
            GROUP BY c.campaign_id
            ORDER BY total_impressions DESC
            LIMIT 5
        """).fetchall()

    cols = ["campaign_id", "name", "health_condition", "specialty_target",
            "start_date", "end_date", "target_completion", "offices",
            "total_impressions", "avg_completion", "kpi_target", "delta_vs_kpi"]

    # Benchmark data for national avg
    benchmarks = conn.execute("""
        SELECT ROUND(AVG(national_avg), 4) AS national_avg,
               ROUND(AVG(target_completion_rate), 4) AS avg_target
        FROM weekly_benchmarks
        WHERE week_start >= '2025-12-01'
    """).fetchone()

    conn.close()

    return {
        "campaigns":   [dict(zip(cols, r)) for r in rows],
        "national_avg": benchmarks[0] if benchmarks else 0.622,
        "avg_target":   benchmarks[1] if benchmarks else 0.660,
        "report_date":  str(date.today()),
        "data_period":  "2025-01-04 to 2025-12-25",
    }


# ── Build report prompt ────────────────────────────────────────────────────────
def _build_prompt(
    campaign_summary: dict,
    sql_result: dict | None,
    anomalies: list | None,
    user_prompt: str,
) -> str:
    """Constructs the Sonnet prompt with all data context."""

    # Campaign data section
    campaigns_json = json.dumps(campaign_summary["campaigns"][:5], indent=2)

    # SQL result section (if available from Agent 1)
    sql_section = ""
    if sql_result and sql_result.get("rows"):
        cols = sql_result.get("columns", [])
        rows = sql_result.get("rows", [])[:MAX_SQL_ROWS_IN_REPORT]
        sql_section = f"""
## Additional Query Results (from analyst query)
Query: {sql_result.get('query', '')[:300]}
Columns: {cols}
Top rows: {json.dumps(rows[:5])}
"""

    # Anomaly section
    anomaly_section = "No anomalies detected this period."
    if anomalies:
        top_anomalies = sorted(
            anomalies, key=lambda x: abs(x.get("z_score", 0)), reverse=True
        )[:MAX_ANOMALIES_IN_REPORT]
        anomaly_section = json.dumps(top_anomalies, indent=2)

    return f"""You are a senior data analyst at PatientPoint, a point-of-care healthcare media company.
Write a professional weekly campaign performance report for pharma brand partners.

Analyst request: "{user_prompt}"
Report date: {campaign_summary['report_date']}
Data period: {campaign_summary['data_period']}
National average completion rate: {campaign_summary['national_avg']}
Portfolio KPI target: {campaign_summary['avg_target']}

## Campaign Performance Data
{campaigns_json}
{sql_section}
## Anomalies Detected (Z-score > 1.8 vs 90-day baseline)
{anomaly_section}

Write the report using EXACTLY this markdown structure:

# PatientPoint Weekly Campaign Report
**Report Date:** [date]
**Data Period:** [period]

## Executive Summary
[2-3 sentences: overall portfolio health, highlight 1 win and 1 concern]

## Campaign Performance
[For each campaign in the data: campaign name, health condition, total impressions,
completion rate vs KPI target, delta, and one-line status. Use a markdown table.]

| Campaign | Condition | Impressions | Completion Rate | KPI Target | Delta | Status |
|---|---|---|---|---|---|---|
[rows]

## Anomalies Flagged
[List top anomalies with campaign, office region, severity, and the explanation.
If no anomalies, say "No anomalies detected this reporting period."]

## Regional Highlights
[2-3 bullets on regional performance patterns from the data]

## Recommendations for Next Week
[3 specific, actionable recommendations based on the data — one per bullet]

---
*Report generated by PatientPoint Campaign Intelligence Agent*
*Data source: campaign_data.db | Model: Claude Sonnet*

Write in a professional but accessible tone. Use precise numbers from the data.
Do not invent statistics not present in the data provided."""


# ── Main agent function ────────────────────────────────────────────────────────
def report_writer_node(state: dict) -> dict:
    print("\n[Report Writer] Generating markdown report with Sonnet...")

    try:
        state = {**state, "error": None}   # ← ADD THIS LINE HERE
        
        # Extract campaign_id from user prompt if mentioned (e.g. "CAMP_035")
        import re
        prompt = state.get("user_prompt", "")
        camp_match = re.search(r"CAMP_[A-Z0-9_]+", prompt.upper())
        campaign_id = camp_match.group(0) if camp_match else None

        # Pull fresh campaign summary from DB
        campaign_summary = _pull_campaign_summary(campaign_id)
        print(f"  Campaign summary: {len(campaign_summary['campaigns'])} campaigns loaded")

        # Build the prompt
        full_prompt = _build_prompt(
            campaign_summary  = campaign_summary,
            sql_result        = state.get("sql_result"),
            anomalies         = state.get("anomalies"),
            user_prompt       = prompt,
        )

        # Sonnet call — quality matters here, this goes to clients
        response = client.messages.create(
            model=SONNET,
            max_tokens=2000,
            messages=[{"role": "user", "content": full_prompt}],
        )

        report_md = response.content[0].text.strip()
        print(f"  Report generated: {len(report_md)} chars, "
              f"~{len(report_md.splitlines())} lines")

        # Save to exports/ for download
        os.makedirs("exports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"exports/weekly_report_{timestamp}.md"
        with open(export_path, "w") as f:
            f.write(report_md)
        print(f"  Saved to {export_path}")

        return {**state, "report_md": report_md}

    except Exception as e:
        print(f"[Report Writer] ERROR: {e}")
        return {
            **state,
            "error":     f"Report Writer failed: {str(e)}",
            "report_md": None,
        }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Agent 3 — Report Writer standalone test")
    print("=" * 65)

    # Test with a full run_report state (simulates what orchestrator passes after Agents 1+2)
    mock_anomalies = [
        {
            "campaign_id": "CAMP_041", "office_id": "OFF_04174", "region": "South",
            "specialty": "cardiology", "tier": "B", "severity": "critical",
            "date": "2025-12-20", "completion_rate": 0.393, "baseline": 0.714,
            "z_score": -3.68,
            "explanation": "Completion rate dropped 32% below the 90-day baseline at this cardiology office.",
        },
        {
            "campaign_id": "CAMP_010", "office_id": "OFF_00347", "region": "Midwest",
            "specialty": "primary_care", "tier": "A", "severity": "critical",
            "date": "2025-12-19", "completion_rate": 0.209, "baseline": 0.652,
            "z_score": -3.66,
            "explanation": "Completion rate dropped 44% below baseline at a high-volume Midwest primary care office.",
        },
    ]

    result = report_writer_node({
        "user_prompt":     "Generate the weekly performance report for CAMP_035",
        "intent":          "run_report",
        "sql_result":      None,
        "anomalies":       mock_anomalies,
        "report_md":       None,
        "error":           None,
        "iteration_count": 0,
    })

    if result.get("report_md"):
        print("\n" + "─" * 65)
        print("REPORT PREVIEW (first 50 lines):")
        print("─" * 65)
        lines = result["report_md"].splitlines()
        for line in lines[:50]:
            print(line)
        if len(lines) > 50:
            print(f"\n... ({len(lines) - 50} more lines in saved file)")
    else:
        print(f"Error: {result.get('error')}")

    print("\n" + "=" * 65)
