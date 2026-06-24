"""
agents/anomaly_detector.py

Agent 2 — Anomaly Detector
Loads daily_metrics from SQLite, computes rolling 90-day Z-scores per placement,
flags outliers from the MOST RECENT 14 days only (production-realistic: you don't
alert on old anomalies), then calls Claude Haiku in batches of 50 for explanations.

Severity thresholds (|Z-score|):
    >= 2.5 → critical
    >= 2.0 → high
    >= 1.8 → medium

Run standalone from project root:
    python agents/anomaly_detector.py
"""

import os
import json
import re
import sqlite3
from datetime import datetime, date
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()

client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
HAIKU   = os.getenv("HAIKU_MODEL", "claude-haiku-4-5")
DB_PATH = os.getenv("DB_PATH", "data/campaign_data.db")

Z_MEDIUM        = 1.8
Z_HIGH          = 2.0
Z_CRITICAL      = 2.5
MIN_WINDOW      = 14   # days of baseline needed before flagging
ROLLING_DAYS    = 90   # rolling baseline window
RECENT_WINDOW   = 14   # only alert on anomalies in the last N days of data
EXPLAIN_BATCH   = 50   # max anomalies per Haiku API call
MAX_LLM_EXPLAIN = 150  # LLM explains top 150; rest get template fallback


# ── Load data ──────────────────────────────────────────────────────────────────
def _load_metrics() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT dm.placement_id, dm.date, dm.completion_rate,
               dm.impressions, dm.completions,
               cp.campaign_id, cp.office_id, cp.screen_type,
               po.region, po.specialty, po.tier, po.state
        FROM daily_metrics dm
        JOIN campaign_placements cp ON dm.placement_id = cp.placement_id
        JOIN physician_offices   po ON cp.office_id    = po.office_id
        WHERE dm.impressions > 0
        ORDER BY dm.placement_id, dm.date
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Z-score on full history ────────────────────────────────────────────────────
def _compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 90-day Z-score per placement. Uses full history as baseline."""
    print(f"  Computing Z-scores across {len(df):,} rows...")
    results = []
    for pid, group in df.groupby("placement_id", sort=False):
        g = group.sort_values("date").copy()
        rolling = g["completion_rate"].rolling(window=ROLLING_DAYS, min_periods=MIN_WINDOW)
        g["rolling_mean"] = rolling.mean().shift(1)
        g["rolling_std"]  = rolling.std().shift(1)
        results.append(g)

    out = pd.concat(results, ignore_index=True)
    mask = out["rolling_std"].notna() & (out["rolling_std"] > 0.001)
    out.loc[mask, "z_score"] = (
        (out.loc[mask, "completion_rate"] - out.loc[mask, "rolling_mean"])
        / out.loc[mask, "rolling_std"]
    )
    return out


# ── Flag anomalies — RECENT WINDOW ONLY ───────────────────────────────────────
def _flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to anomalies in the most recent RECENT_WINDOW days of data.
    This is what a production system does: compute baselines from full history
    but only alert on what's happening RIGHT NOW.
    """
    latest_date = df["date"].max()
    cutoff      = latest_date - pd.Timedelta(days=RECENT_WINDOW)

    flagged = df[
        df["z_score"].notna() &
        (df["z_score"].abs() >= Z_MEDIUM) &
        (df["date"] >= cutoff)                 # ← recency filter
    ].copy()

    if flagged.empty:
        return flagged

    flagged["severity"] = "medium"
    flagged.loc[flagged["z_score"].abs() >= Z_HIGH,     "severity"] = "high"
    flagged.loc[flagged["z_score"].abs() >= Z_CRITICAL, "severity"] = "critical"
    flagged["abs_z"] = flagged["z_score"].abs()

    # One worst anomaly per campaign-office pair
    worst = (
        flagged
        .sort_values("abs_z", ascending=False)
        .drop_duplicates(subset=["campaign_id", "office_id"])
        .reset_index(drop=True)
    )
    print(f"  Recent window ({cutoff.date()} → {latest_date.date()}): {len(worst)} anomalous campaign-office pairs")
    return worst


# ── Template fallback explanation ─────────────────────────────────────────────
def _template_explanation(row: dict) -> str:
    direction = "dropped" if row["z_score"] < 0 else "spiked"
    delta_pct  = abs(row["completion_rate"] - row["baseline"]) * 100
    return (
        f"Completion rate {direction} {delta_pct:.0f}% below the 90-day "
        f"baseline at this {row['specialty']} office ({row['region']})."
    )


# ── LLM explanations in batches of 50 ────────────────────────────────────────
def _explain_batch(batch: list[dict]) -> list[dict]:
    """Single Haiku call for one batch of up to EXPLAIN_BATCH anomalies."""
    prompt = f"""You are analyzing campaign performance anomalies for PatientPoint, a healthcare media platform.

For each anomaly below, write exactly ONE sentence (max 20 words) that:
- States what happened in plain English (no z-scores or statistics)
- Mentions the direction (drop or spike) and rough magnitude
- Is suitable for a non-technical client success manager

Return ONLY a valid JSON array with all original fields plus one new field: "explanation".
No markdown, no code blocks, no preamble.

{json.dumps(batch, indent=2)}"""

    response = client.messages.create(
        model=HAIKU,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Add template fallback to each item
            for item in batch:
                item["explanation"] = _template_explanation(item)
            return batch


def _explain_anomalies(flagged: pd.DataFrame) -> list[dict]:
    """
    Top MAX_LLM_EXPLAIN anomalies get LLM explanations (in batches of EXPLAIN_BATCH).
    Remaining get fast template explanations — no extra API cost.
    """
    if flagged.empty:
        return []

    # Build payload rows
    all_items = []
    for _, row in flagged.iterrows():
        all_items.append({
            "campaign_id":     row["campaign_id"],
            "office_id":       row["office_id"],
            "region":          row["region"],
            "specialty":       row["specialty"],
            "tier":            row["tier"],
            "severity":        row["severity"],
            "date":            str(row["date"].date()),
            "completion_rate": round(float(row["completion_rate"]), 4),
            "baseline":        round(float(row["rolling_mean"]), 4),
            "z_score":         round(float(row["z_score"]), 2),
        })

    # Split: top N get LLM, rest get templates
    llm_batch   = all_items[:MAX_LLM_EXPLAIN]
    tmpl_batch  = all_items[MAX_LLM_EXPLAIN:]

    # Template explanations first (free)
    for item in tmpl_batch:
        item["explanation"] = _template_explanation(item)

    # LLM explanations in chunks
    explained_llm = []
    n_batches = (len(llm_batch) + EXPLAIN_BATCH - 1) // EXPLAIN_BATCH
    print(f"  Calling Haiku: {len(llm_batch)} LLM + {len(tmpl_batch)} template explanations "
          f"({n_batches} batch{'es' if n_batches > 1 else ''})")

    for i in range(0, len(llm_batch), EXPLAIN_BATCH):
        chunk = llm_batch[i : i + EXPLAIN_BATCH]
        print(f"    Batch {i//EXPLAIN_BATCH + 1}/{n_batches} ({len(chunk)} anomalies)...", end=" ")
        result = _explain_batch(chunk)
        explained_llm.extend(result)
        print("done")

    return explained_llm + tmpl_batch


# ── Write to anomaly_log ───────────────────────────────────────────────────────
def _write_to_db(anomalies: list[dict], flagged_df: pd.DataFrame):
    if not anomalies:
        return
    placement_lookup = (
        flagged_df.set_index(["campaign_id", "office_id"])["placement_id"].to_dict()
    )
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM anomaly_log")
    records = [(
        f"ANOM_{a.get('campaign_id','')}_{a.get('office_id','')}",
        a.get("campaign_id", ""),
        a.get("office_id", ""),
        placement_lookup.get((a.get("campaign_id",""), a.get("office_id","")), ""),
        datetime.now().isoformat(),
        a.get("date", str(date.today())),
        a.get("severity", "medium"),
        a.get("completion_rate", 0.0),
        a.get("baseline", 0.0),
        a.get("z_score", 0.0),
        a.get("explanation", ""),
    ) for a in anomalies]

    conn.executemany("""
        INSERT OR REPLACE INTO anomaly_log
        (anomaly_id, campaign_id, office_id, placement_id, detected_at,
         week_start, severity, metric_value, baseline_value, z_score, explanation)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, records)
    conn.commit()
    conn.close()


# ── LangGraph node ─────────────────────────────────────────────────────────────
def anomaly_detector_node(state: dict) -> dict:
    print("\n[Anomaly Detector] Starting Z-score scan...")
    try:
        df      = _load_metrics()
        print(f"  Loaded {len(df):,} metric rows")
        df      = _compute_zscores(df)
        flagged = _flag_anomalies(df)

        if flagged.empty:
            return {**state, "anomalies": []}

        for sev in ["critical", "high", "medium"]:
            c = (flagged["severity"] == sev).sum()
            if c: print(f"    {sev.capitalize():<10} {c}")

        anomalies = _explain_anomalies(flagged)
        _write_to_db(anomalies, flagged)
        print(f"  Wrote {len(anomalies)} records to anomaly_log")
        return {**state, "anomalies": anomalies}

    except Exception as e:
        print(f"[Anomaly Detector] ERROR: {e}")
        return {**state, "error": f"Anomaly detector failed: {str(e)}", "anomalies": []}


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter
    print("=" * 65)
    print("Agent 2 — Anomaly Detector standalone test")
    print("=" * 65)

    result = anomaly_detector_node({
        "user_prompt": "Show me anomalies", "intent": "show_anomalies",
        "sql_result": None, "anomalies": None, "report_md": None,
        "error": None, "iteration_count": 0,
    })

    anomalies = result.get("anomalies", [])
    print(f"\nTotal anomalies: {len(anomalies)}")

    if anomalies:
        print(f"Severity breakdown: {dict(Counter(a.get('severity') for a in anomalies))}")
        sorted_a = sorted(anomalies, key=lambda x: abs(x.get("z_score", 0)), reverse=True)
        print(f"\nTop 5 most severe:")
        print(f"  {'Campaign':<14} {'Office':<12} {'Sev':<10} {'Z':>6}  Explanation")
        print(f"  {'─'*14} {'─'*12} {'─'*10} {'─'*6}  {'─'*40}")
        for a in sorted_a[:5]:
            print(f"  {a.get('campaign_id',''):<14} {a.get('office_id',''):<12} "
                  f"{a.get('severity',''):<10} {a.get('z_score',0):>6.2f}  "
                  f"{a.get('explanation','')[:50]}")

        conn = sqlite3.connect(DB_PATH)
        db_count = conn.execute("SELECT COUNT(*) FROM anomaly_log").fetchone()[0]
        conn.close()
        print(f"\nRows in anomaly_log: {db_count}")

    if result.get("error"):
        print(f"\nError: {result['error']}")
    print("\n" + "=" * 65)
