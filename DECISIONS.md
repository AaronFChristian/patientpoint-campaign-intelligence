# Technical Decisions

Architecture and tooling choices for the PatientPoint Campaign Intelligence Agent, with rationale.

---

## 1. Haiku for Agents 1 & 2, Sonnet for Agent 3

**Agent 1 (SQL Analyst)** and **Agent 2 (Anomaly Detector)** call Claude Haiku. Both tasks are structured output jobs — generate a SQL query, or return a JSON array of anomalies with explanations. Haiku handles structured extraction reliably at a fraction of Sonnet's cost. In a production system running this pipeline daily across 100+ clients, using Haiku here keeps per-run API cost under $0.05.

**Agent 3 (Report Writer)** calls Claude Sonnet. This is the only node where output quality is what the client actually reads. A weekly report that goes to a pharma brand partner needs coherent narrative, proper section structure, and appropriate clinical tone. That's where Sonnet earns its cost.

---

## 2. LangGraph over raw function calls

A plain Python chain of `sql_analyst() → anomaly_detector() → report_writer()` would work for the happy path. LangGraph adds three things that matter for a production-realistic demo:

- **Conditional routing** — the orchestrator classifies intent and can skip agents. An ad-hoc question doesn't need the report writer. This mirrors how a real automation system would work.
- **Typed state** — the StateGraph enforces a shared state schema (TypedDict). Every agent reads from and writes to the same state object. This makes the data flow explicit and debuggable.
- **Retry and error isolation** — each node can fail and recover independently without crashing the whole pipeline. The `max_iterations` guard prevents infinite loops.

---

## 3. SQLite over PostgreSQL

This project runs locally on a MacBook for interview demo purposes. SQLite gives a single-file database with zero setup, zero running process, and identical SQL semantics to PostgreSQL for everything this project does (JOINs, window functions, GROUP BY, subqueries). The patterns in `ad_hoc_library.sql` are directly portable to Snowflake or BigQuery — the SQL dialect gap is minimal.

A production PatientPoint analytics system would run on Snowflake or Redshift. The interview answer is: "I used SQLite to keep setup friction near zero for the demo, but every query in the library runs unchanged against PostgreSQL — I tested the window functions specifically."

---

## 4. Streamlit over React/FastAPI

A data analyst building an internal analytics tool would reach for Streamlit. It lets the data layer and the UI live in the same Python process, which means the dataframes from Agent 1 flow directly into Plotly charts without a REST serialization layer. React + FastAPI would add 200 lines of API plumbing for no user-facing benefit in this context, and it would signal "backend engineer" rather than "analyst who automates their own workflow."

---

## 5. AWS S3 as an additive layer

S3 is not a dependency — the pipeline runs fully offline. S3 is added for two reasons: first, it demonstrates production thinking (raw CSVs versioned in object storage is standard practice for healthcare analytics audit trails). Second, it's a talking point: "In a real PatientPoint deployment, every weekly export would be archived to S3 with a timestamped key so you can replay any week's report and audit the underlying data."
