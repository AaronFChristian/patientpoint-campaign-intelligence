"""
agents/orchestrator.py

LangGraph StateGraph orchestrator for the PatientPoint Campaign Intelligence Agent.
Classifies user intent and routes to the appropriate agent node(s).

Intent routing:
    ask_question   → sql_analyst → END
    show_anomalies → anomaly_detector → END
    run_report     → sql_analyst → anomaly_detector → report_writer → END

Run test harness from project root:
    python agents/orchestrator.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from typing import TypedDict, Optional, Literal
from dotenv import load_dotenv
import anthropic
from langgraph.graph import StateGraph, END

load_dotenv()

from agents.sql_analyst import sql_analyst_node
from agents.anomaly_detector import anomaly_detector_node
from agents.report_writer import report_writer_node

# ── Anthropic client ───────────────────────────────────────────────────────────
_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not set. Check your .env file.")

client = anthropic.Anthropic(api_key=_api_key)

HAIKU_MODEL  = os.getenv("HAIKU_MODEL",  "claude-haiku-4-5")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-6")
MAX_ITERATIONS = 3


# ── State schema ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    Shared state object passed between every node in the graph.
    Each agent reads what it needs and writes its output back here.
    """
    user_prompt:     str
    intent:          Optional[str]   # ask_question | show_anomalies | run_report
    sql_result:      Optional[dict]  # {query, columns, rows, row_count, chart_type}
    anomalies:       Optional[list]  # list of anomaly dicts from Agent 2
    report_md:       Optional[str]   # full markdown report from Agent 3
    error:           Optional[str]   # set by any node on failure
    iteration_count: int             # safety guard — prevents infinite loops


# ── Node: Classifier ──────────────────────────────────────────────────────────
def classifier_node(state: AgentState) -> AgentState:
    """
    Calls Claude Haiku to classify the user's prompt into one of three intents.
    This is the only node that fires on every request — it gates the whole graph.
    Uses Haiku (not Sonnet) because intent classification is a simple, structured
    output task that doesn't need narrative quality.
    """
    print(f"\n[Classifier] Classifying: \"{state['user_prompt'][:80]}...\"")

    if state["iteration_count"] >= MAX_ITERATIONS:
        print(f"[Classifier] Max iterations ({MAX_ITERATIONS}) reached — stopping.")
        return {**state, "error": "max_iterations_exceeded"}

    prompt = f"""Classify the following user request into exactly one of these three intents:

- ask_question   : The user wants to query data, get a specific metric, see a chart,
                   or ask a factual question about campaign or office performance.
- show_anomalies : The user wants to see anomalies, outliers, performance problems,
                   campaigns that are flagged, or anything that "looks off".
- run_report     : The user wants a full report, weekly summary, comprehensive analysis,
                   executive summary, or a complete overview of a campaign.

User request: "{state['user_prompt']}"

Return ONLY one of these three strings, nothing else:
ask_question
show_anomalies
run_report"""

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip().lower()

        # Validate — default to ask_question if unexpected response
        valid_intents = {"ask_question", "show_anomalies", "run_report"}
        intent = raw if raw in valid_intents else "ask_question"

        print(f"[Classifier] Intent → {intent}")
        return {
            **state,
            "intent": intent,
            "iteration_count": state["iteration_count"] + 1,
        }

    except Exception as e:
        print(f"[Classifier] ERROR: {e}")
        return {**state, "error": str(e), "intent": "ask_question"}

# ── Routing functions ──────────────────────────────────────────────────────────
def route_after_classifier(
    state: AgentState,
) -> Literal["sql_analyst", "anomaly_detector", "__end__"]:
    """After classification: errors go to END, others route to first agent."""
    if state.get("error"):
        return END
    intent = state.get("intent", "ask_question")
    if intent in ("ask_question", "run_report"):
        return "sql_analyst"
    return "anomaly_detector"  # show_anomalies skips sql_analyst


def route_after_sql(
    state: AgentState,
) -> Literal["anomaly_detector", "__end__"]:
    """After SQL Analyst: run_report always continues even if SQL failed
    (report_writer pulls from DB directly). ask_question stops here."""
    if state.get("intent") == "run_report":
        return "anomaly_detector"
    return END  # ask_question is done after SQL


def route_after_anomaly(
    state: AgentState,
) -> Literal["report_writer", "__end__"]:
    """After Anomaly Detector: run_report continues to report_writer, others stop."""
    if state.get("error"):
        return END
    if state.get("intent") == "run_report":
        return "report_writer"
    return END  # show_anomalies is done after anomaly scan


# ── Build the graph ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classifier",       classifier_node)
    builder.add_node("sql_analyst",      sql_analyst_node)
    builder.add_node("anomaly_detector", anomaly_detector_node)
    builder.add_node("report_writer",    report_writer_node)

    # Entry point
    builder.set_entry_point("classifier")

    # Edges
    builder.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "sql_analyst":      "sql_analyst",
            "anomaly_detector": "anomaly_detector",
            END:                END,
        },
    )
    builder.add_conditional_edges(
        "sql_analyst",
        route_after_sql,
        {
            "anomaly_detector": "anomaly_detector",
            END:                END,
        },
    )
    builder.add_conditional_edges(
        "anomaly_detector",
        route_after_anomaly,
        {
            "report_writer": "report_writer",
            END:             END,
        },
    )
    builder.add_edge("report_writer", END)

    return builder.compile()


# Compiled graph — imported by Streamlit app and other modules
graph = build_graph()


# ── Public helper: invoke the graph ───────────────────────────────────────────
def run_agent(user_prompt: str) -> AgentState:
    """
    Single entry point for all callers (Streamlit tabs, tests, CLI).
    Initialises state with safe defaults and invokes the graph.
    """
    initial_state: AgentState = {
        "user_prompt":     user_prompt,
        "intent":          None,
        "sql_result":      None,
        "anomalies":       None,
        "report_md":       None,
        "error":           None,
        "iteration_count": 0,
    }
    return graph.invoke(initial_state)


# ── Test harness ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("PatientPoint Agent — Orchestrator Test (stubs active)")
    print("=" * 65)

    test_cases = [
        (
            "ask_question",
            "Which cardiology offices in the Midwest had the lowest completion rate last month?",
        ),
        (
            "show_anomalies",
            "Show me any campaigns with performance anomalies or unusual drops this week",
        ),
        (
            "run_report",
            "Generate the full weekly performance report for Campaign CAMP_035",
        ),
    ]

    passed = 0
    for expected_intent, prompt in test_cases:
        print(f"\n{'─' * 65}")
        print(f"Prompt    : {prompt[:80]}")
        print(f"Expected  : {expected_intent}")

        result = run_agent(prompt)

        actual = result.get("intent", "unknown")
        ok = actual == expected_intent
        passed += 1 if ok else 0

        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"Classified: {actual}  →  {status}")

        if result.get("error"):
            print(f"Error     : {result['error']}")

        # Show which nodes were reached based on what got populated
        nodes_reached = ["classifier"]
        if result.get("sql_result"):
            nodes_reached.append("sql_analyst")
        if result.get("anomalies") is not None:
            nodes_reached.append("anomaly_detector")
        if result.get("report_md"):
            nodes_reached.append("report_writer")
        print(f"Path      : {' → '.join(nodes_reached)} → END")

    print(f"\n{'=' * 65}")
    print(f"Results: {passed}/{len(test_cases)} intent classifications correct")
    print(f"{'=' * 65}\n")
