"""
app/streamlit_app.py

PatientPoint Campaign Intelligence — Streamlit UI
4-tab analyst interface powered by the LangGraph agent pipeline.

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import sqlite3
import json
import re
from datetime import datetime, date

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── Path setup (app/ → project root) ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import run_agent
from agents.sql_analyst import sql_analyst_node
from agents.anomaly_detector import anomaly_detector_node
from agents.report_writer import report_writer_node

DB_PATH = os.getenv("DB_PATH", "data/campaign_data.db")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PatientPoint Campaign Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0; }
    .sub-header  { font-size: 0.9rem; color: #64748b; margin-top: 0; margin-bottom: 1.5rem; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                   padding: 1rem; text-align: center; }
    .severity-critical { background-color: #fee2e2; color: #991b1b;
                         padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .severity-high     { background-color: #ffedd5; color: #9a3412;
                         padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .severity-medium   { background-color: #fef9c3; color: #854d0e;
                         padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .sql-block { background: #1e293b; color: #94a3b8; padding: 1rem;
                 border-radius: 6px; font-family: monospace; font-size: 0.82rem;
                 white-space: pre-wrap; overflow-x: auto; }
    .stTabs [data-baseweb="tab"] { font-size: 0.92rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏥 PatientPoint Campaign Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered analytics for point-of-care campaign performance</p>', unsafe_allow_html=True)

# ── DB helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_campaign_list() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT campaign_id FROM campaigns ORDER BY campaign_id").fetchall()
    conn.close()
    return [r[0] for r in rows]

@st.cache_data(ttl=300)
def get_campaign_metrics(campaign_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT dm.date, dm.completion_rate, dm.impressions, dm.completions, dm.ctr
        FROM daily_metrics dm
        JOIN campaign_placements cp ON dm.placement_id = cp.placement_id
        WHERE cp.campaign_id = ? AND dm.impressions > 0
        ORDER BY dm.date
    """, conn, params=(campaign_id,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_anomaly_log() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("""
            SELECT al.*, c.name AS campaign_name, c.health_condition
            FROM anomaly_log al
            LEFT JOIN campaigns c ON al.campaign_id = c.campaign_id
            ORDER BY ABS(al.z_score) DESC
        """, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

# ── Plotly chart renderer ──────────────────────────────────────────────────────
def render_chart(sql_result: dict):
    cols  = sql_result.get("columns", [])
    rows  = sql_result.get("rows", [])
    ctype = sql_result.get("chart_type", "table")
    x_col = sql_result.get("x_col")
    y_col = sql_result.get("y_col")

    if not rows:
        st.info("Query returned no rows.")
        return

    df = pd.DataFrame(rows, columns=cols)

    if ctype == "kpi_card" and y_col:
        st.metric(label=y_col.replace("_", " ").title(), value=rows[0][0])

    elif ctype == "bar" and x_col and y_col:
        fig = px.bar(
            df, x=x_col, y=y_col,
            color_discrete_sequence=["#2563EB"],
            title=f"{y_col.replace('_',' ').title()} by {x_col.replace('_',' ').title()}",
        )
        fig.update_layout(plot_bgcolor="white", showlegend=False,
                          xaxis_title=x_col.replace("_", " ").title(),
                          yaxis_title=y_col.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    elif ctype == "line" and x_col and y_col:
        fig = px.line(
            df, x=x_col, y=y_col, markers=True,
            color_discrete_sequence=["#2563EB"],
            title=f"{y_col.replace('_',' ').title()} over Time",
        )
        fig.update_layout(plot_bgcolor="white",
                          xaxis_title=x_col.replace("_", " ").title(),
                          yaxis_title=y_col.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.dataframe(df, use_container_width=True, height=min(400, 50 + len(df) * 35))


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask a Question",
    "📋 Weekly Report",
    "⚠️ Anomaly Feed",
    "🧪 A/B Tester",
])


# ── TAB 1: Natural Language Q&A ───────────────────────────────────────────────
with tab1:
    st.subheader("Ask a Data Question")
    st.caption("Type any business question about campaign performance. The AI generates and runs SQL automatically.")

    st.markdown("**Try these:**")
    examples = [
        "Which region has the highest average completion rate?",
        "Show me the top 5 campaigns by total impressions",
        "Which cardiology offices in the Midwest had the lowest completion rate?",
        "How many offices are in each tier?",
    ]
    col_ex = st.columns(2)
    for i, ex in enumerate(examples):
        if col_ex[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["tab1_input"] = ex   # ← writes directly to widget key

    st.divider()

    prompt = st.text_input(
        "Your question",
        placeholder="e.g. Which specialty has the highest completion rate for cardiovascular campaigns?",
        key="tab1_input",                          # ← no value= param; session_state drives it
    )

    if st.button("Run Query", type="primary", key="tab1_run") and prompt:
        with st.spinner("Generating SQL and querying database..."):
            state = {
                "user_prompt":     prompt,
                "intent":          "ask_question",
                "sql_result":      None,
                "anomalies":       None,
                "report_md":       None,
                "error":           None,
                "iteration_count": 0,
            }
            result = sql_analyst_node(state)

        if result.get("error"):
            st.error(f"Query failed: {result['error']}")
        elif result.get("sql_result"):
            sr = result["sql_result"]
            st.success(f"Returned {sr['row_count']} rows")
            render_chart(sr)
            with st.expander("View generated SQL", expanded=False):
                st.markdown(
                    f'<div class="sql-block">{sr["query"]}</div>',
                    unsafe_allow_html=True,
                )
            if sr["rows"] and sr["chart_type"] != "table":
                with st.expander("View raw data", expanded=False):
                    df_raw = pd.DataFrame(sr["rows"], columns=sr["columns"])
                    st.dataframe(df_raw, use_container_width=True)
        else:
            st.warning("No results returned. Try rephrasing your question.")


# ── TAB 2: Weekly Report Generator ───────────────────────────────────────────
with tab2:
    st.subheader("Weekly Campaign Report")
    st.caption("Select a campaign and generate a client-ready markdown report using the full 3-agent pipeline.")

    campaigns = get_campaign_list()

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        selected_campaign = st.selectbox(
            "Campaign", campaigns,
            index=campaigns.index("CAMP_035") if "CAMP_035" in campaigns else 0,
        )
    with col_r2:
        report_prompt = st.text_input(
            "Report focus (optional)",
            value=f"Generate the full weekly performance report for {selected_campaign}",
            key="tab2_prompt",
        )

    run_report = st.button("Generate Report", type="primary", key="tab2_run")

    if run_report:
        report_state = {
            "user_prompt":     report_prompt,
            "intent":          "run_report",
            "sql_result":      None,
            "anomalies":       None,
            "report_md":       None,
            "error":           None,
            "iteration_count": 0,
        }

        with st.status("Running 3-agent pipeline...", expanded=True) as status:
            st.write("**Agent 1:** Querying campaign performance data...")
            report_state = sql_analyst_node(report_state)
            if report_state.get("sql_result"):
                sr = report_state["sql_result"]
                st.write(f"  Retrieved {sr['row_count']} rows across {len(sr['columns'])} metrics")

            st.write("**Agent 2:** Scanning for anomalies (Z-score across 528K rows)...")
            report_state["error"] = None
            report_state = anomaly_detector_node(report_state)
            n_anom = len(report_state.get("anomalies") or [])
            st.write(f"  Flagged {n_anom} anomalies in the recent window")

            st.write("**Agent 3:** Generating client report with Claude Sonnet...")
            report_state["error"] = None
            report_state = report_writer_node(report_state)

            if report_state.get("report_md"):
                status.update(label="Report ready!", state="complete", expanded=False)
            else:
                status.update(label="Report failed", state="error", expanded=True)

        if report_state.get("report_md"):
            st.divider()
            st.download_button(
                label="Download Report (.md)",
                data=report_state["report_md"],
                file_name=f"patientpoint_report_{selected_campaign}_{date.today()}.md",
                mime="text/markdown",
            )
            st.markdown(report_state["report_md"])
        elif report_state.get("error"):
            st.error(f"Report generation failed: {report_state['error']}")


# ── TAB 3: Anomaly Feed ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Anomaly Feed")
    st.caption("Statistical anomalies detected via rolling 90-day Z-score baseline. Updated each time the pipeline runs.")

    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        sev_filter = st.multiselect(
            "Severity",
            ["critical", "high", "medium"],
            default=["critical", "high"],
        )
    with col_f2:
        region_filter = st.multiselect("Region", ["Northeast", "Midwest", "South", "West"])
    with col_f3:
        rescan = st.button("Re-run anomaly scan", key="tab3_rescan")

    if rescan:
        with st.spinner("Scanning 528K metric rows..."):
            scan_state = {
                "user_prompt": "show anomalies", "intent": "show_anomalies",
                "sql_result": None, "anomalies": None, "report_md": None,
                "error": None, "iteration_count": 0,
            }
            anomaly_detector_node(scan_state)
        st.success("Anomaly scan complete — table refreshed.")
        st.cache_data.clear()

    anomaly_df = get_anomaly_log()

    if anomaly_df.empty:
        st.info("No anomalies in the log yet. Run the pipeline or click 'Re-run anomaly scan'.")
    else:
        if sev_filter:
            anomaly_df = anomaly_df[anomaly_df["severity"].isin(sev_filter)]

        if "region" not in anomaly_df.columns:
            conn = sqlite3.connect(DB_PATH)
            offices = pd.read_sql_query("SELECT office_id, region, specialty FROM physician_offices", conn)
            conn.close()
            anomaly_df = anomaly_df.merge(offices, on="office_id", how="left")

        if region_filter:
            anomaly_df = anomaly_df[anomaly_df["region"].isin(region_filter)]

        st.caption(f"Showing {len(anomaly_df)} anomalies")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Flagged", len(anomaly_df))
        k2.metric("Critical", int((anomaly_df["severity"] == "critical").sum()))
        k3.metric("High", int((anomaly_df["severity"] == "high").sum()))
        k4.metric("Avg |Z-score|", f"{anomaly_df['z_score'].abs().mean():.2f}")

        st.divider()

        def color_severity(val):
            colors = {"critical": "background-color: #fee2e2",
                      "high":     "background-color: #ffedd5",
                      "medium":   "background-color: #fef9c3"}
            return colors.get(val, "")

        display_cols = ["campaign_id", "office_id", "severity", "metric_value",
                        "baseline_value", "z_score", "week_start"]
        if "region" in anomaly_df.columns:
            display_cols.insert(2, "region")

        display_df = anomaly_df[
            [c for c in display_cols if c in anomaly_df.columns]
        ].copy()
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

        styled = display_df.style.map(
            color_severity,
            subset=["Severity"] if "Severity" in display_df.columns else [],
        ).format(
            {"Metric Value": "{:.4f}", "Baseline Value": "{:.4f}", "Z Score": "{:.2f}"}
            if "Metric Value" in display_df.columns else {}
        )
        st.dataframe(styled, use_container_width=True, height=400)

        st.subheader("Anomaly Details")
        for _, row in anomaly_df.head(10).iterrows():
            sev = row.get("severity", "medium")
            with st.expander(
                f"{row['campaign_id']} / {row['office_id']} — {sev.upper()} (Z={row['z_score']:.2f})"
            ):
                dcol1, dcol2 = st.columns(2)
                dcol1.metric("Completion Rate", f"{row['metric_value']:.1%}")
                dcol2.metric("90-Day Baseline", f"{row['baseline_value']:.1%}")
                dcol1.metric("Z-Score", f"{row['z_score']:.2f}")
                dcol2.metric("Drop vs Baseline",
                             f"{(row['metric_value'] - row['baseline_value']):.1%}")
                if row.get("explanation"):
                    st.info(f"**AI Explanation:** {row['explanation']}")

        csv_data = anomaly_df.to_csv(index=False)
        st.download_button(
            "Export anomalies to CSV",
            data=csv_data,
            file_name=f"anomalies_{date.today()}.csv",
            mime="text/csv",
        )


# ── TAB 4: A/B Significance Tester ───────────────────────────────────────────
with tab4:
    st.subheader("A/B Campaign Significance Tester")
    st.caption("Welch's t-test on daily completion rates. Use CAMP_AB_HIGH vs CAMP_AB_LOW for a pre-seeded significant result.")

    campaigns = get_campaign_list()

    ab_col1, ab_col2 = st.columns(2)
    with ab_col1:
        campaign_a = st.selectbox(
            "Campaign A", campaigns,
            index=campaigns.index("CAMP_AB_HIGH") if "CAMP_AB_HIGH" in campaigns else 0,
            key="ab_camp_a",
        )
    with ab_col2:
        campaign_b = st.selectbox(
            "Campaign B", campaigns,
            index=campaigns.index("CAMP_AB_LOW") if "CAMP_AB_LOW" in campaigns else 1,
            key="ab_camp_b",
        )

    ab_col3, ab_col4 = st.columns(2)
    with ab_col3:
        date_from = st.date_input("From", value=date(2025, 2, 1), key="ab_from")
    with ab_col4:
        date_to   = st.date_input("To",   value=date(2025, 5, 1), key="ab_to")

    run_ab = st.button("Run A/B Test", type="primary", key="ab_run")

    if run_ab:
        if campaign_a == campaign_b:
            st.warning("Select two different campaigns to compare.")
        else:
            df_a = get_campaign_metrics(campaign_a)
            df_b = get_campaign_metrics(campaign_b)

            df_a = df_a[(df_a["date"] >= pd.Timestamp(date_from)) &
                        (df_a["date"] <= pd.Timestamp(date_to))]
            df_b = df_b[(df_b["date"] >= pd.Timestamp(date_from)) &
                        (df_b["date"] <= pd.Timestamp(date_to))]

            if len(df_a) < 7 or len(df_b) < 7:
                st.warning("Not enough data in this date range. Try widening the window.")
            else:
                cr_a = df_a["completion_rate"].dropna()
                cr_b = df_b["completion_rate"].dropna()

                t_stat, p_value = stats.ttest_ind(cr_a, cr_b, equal_var=False)
                mean_a  = cr_a.mean()
                mean_b  = cr_b.mean()
                diff    = mean_a - mean_b
                se      = np.sqrt(cr_a.var()/len(cr_a) + cr_b.var()/len(cr_b))
                ci_low  = diff - 1.96 * se
                ci_high = diff + 1.96 * se

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{campaign_a} mean", f"{mean_a:.4f}")
                m2.metric(f"{campaign_b} mean", f"{mean_b:.4f}")
                m3.metric("Mean difference", f"{diff:+.4f}")
                m4.metric("p-value", f"{p_value:.4f}")

                alpha = 0.05
                if p_value < alpha:
                    st.success(
                        f"**Statistically significant** (p={p_value:.4f} < {alpha}). "
                        f"{campaign_a} outperforms {campaign_b} by {abs(diff):.1%} "
                        f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}]). "
                        f"This difference is unlikely due to chance."
                    )
                else:
                    st.warning(
                        f"**Not statistically significant** (p={p_value:.4f} ≥ {alpha}). "
                        f"Cannot conclude a real performance difference with 95% confidence."
                    )

                st.caption(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}] | "
                           f"t-statistic: {t_stat:.3f} | n_A={len(cr_a)}, n_B={len(cr_b)}")

                st.divider()

                df_combined = pd.DataFrame({
                    "Completion Rate": pd.concat([cr_a, cr_b], ignore_index=True),
                    "Campaign": [campaign_a] * len(cr_a) + [campaign_b] * len(cr_b),
                })
                fig = px.box(
                    df_combined, x="Campaign", y="Completion Rate",
                    color="Campaign",
                    color_discrete_map={campaign_a: "#2563EB", campaign_b: "#DC2626"},
                    points="outliers",
                    title=f"Completion Rate Distribution: {campaign_a} vs {campaign_b}",
                )
                fig.update_layout(plot_bgcolor="white", showlegend=False,
                                  yaxis_title="Daily Completion Rate")
                st.plotly_chart(fig, use_container_width=True)

                df_a_plot = df_a[["date", "completion_rate"]].copy()
                df_a_plot["Campaign"] = campaign_a
                df_b_plot = df_b[["date", "completion_rate"]].copy()
                df_b_plot["Campaign"] = campaign_b
                df_ts = pd.concat([df_a_plot, df_b_plot])

                fig2 = px.line(
                    df_ts, x="date", y="completion_rate", color="Campaign",
                    color_discrete_map={campaign_a: "#2563EB", campaign_b: "#DC2626"},
                    title="Daily Completion Rate Over Time",
                )
                fig2.update_layout(plot_bgcolor="white",
                                   xaxis_title="Date", yaxis_title="Completion Rate")
                st.plotly_chart(fig2, use_container_width=True)
