# PatientPoint Campaign Intelligence Agent

> An AI-powered multi-agent analytics system that automates campaign performance reporting, anomaly detection, and ad-hoc data Q&A for healthcare media analytics teams.

**Built as a portfolio project for a Data Analyst role at PatientPoint · June 2026**

---

## What it does

PatientPoint's analytics team supports 100+ pharma and health brand clients with weekly campaign performance reports, anomaly investigation, and ad-hoc business questions. This system automates that entire workflow so an analyst goes from raw data to a client-ready insight in a single prompt — instead of spending 60–70% of their week on manual data work.

---

## Architecture

```
User prompt
    │
    ▼
┌─────────────────────────────┐
│   LangGraph Orchestrator    │  ← classifies intent: ask_question /
│   (intent classifier)       │    show_anomalies / run_report
└──────────┬──────────────────┘
           │
    ┌──────┴────────────────────────┐
    │                               │
    ▼                               ▼
┌──────────────┐         ┌──────────────────┐
│  Agent 1     │         │   Agent 2         │
│  SQL Analyst │         │   Anomaly         │
│  (Haiku)     │         │   Detector (Haiku)│
└──────┬───────┘         └────────┬─────────┘
       │                          │
       └──────────┬───────────────┘
                  ▼
         ┌────────────────┐
         │   Agent 3      │
         │  Report Writer │
         │   (Sonnet)     │
         └────────┬───────┘
                  ▼
         Streamlit UI (4 tabs)
         + Power BI Dashboard
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2+ |
| LLM | Anthropic Claude (Haiku + Sonnet) |
| Data simulation | Python Faker |
| Data processing | Pandas, NumPy, SciPy |
| Storage | SQLite (local) + AWS S3 (optional) |
| UI | Streamlit + Plotly |
| Static reporting | Power BI Desktop |

---

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/AaronFChristian/patientpoint-campaign-intelligence.git
cd patientpoint-campaign-intelligence

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY (required)
# AWS vars are optional — leave blank to run fully offline

# 5. Generate the dataset
python data/generate_data.py

# 6. Clean and load into SQLite
python data/clean_pipeline.py
python data/load_sqlite.py

# 7. Run the app
streamlit run app/streamlit_app.py
```

---

## Sample prompts to try

```
Tab 1 — Ask a question:
  "Which cardiology offices in the Midwest had the lowest completion rate last month?"
  "Show me the top 5 campaigns by total impressions this quarter"
  "How did Campaign 12 perform week-over-week in the last 8 weeks?"

Tab 3 — Anomaly feed:
  Filter to Critical severity to see the most urgent flags

Tab 4 — A/B tester:
  Select "CAMP_AB_HIGH" vs "CAMP_AB_LOW" over 90 days for a clean significant result
```

---

## Key design decisions

See [DECISIONS.md](DECISIONS.md) for the full reasoning behind:
- Why Haiku for Agents 1 & 2, Sonnet for Agent 3
- Why LangGraph over raw function calls
- Why SQLite over PostgreSQL for this project
- Why Streamlit over React/FastAPI

---

## Project structure

```
patientpoint-campaign-intelligence/
├── data/
│   ├── generate_data.py       # Faker simulation (500K+ rows)
│   ├── clean_pipeline.py      # Pandas cleaning + validation report
│   ├── load_sqlite.py         # SQLite schema creation + bulk insert
│   └── push_s3.py             # AWS S3 upload (optional)
├── agents/
│   ├── orchestrator.py        # LangGraph StateGraph + routing
│   ├── sql_analyst.py         # Agent 1: NL → SQL → DataFrame
│   ├── anomaly_detector.py    # Agent 2: Z-score detection + explanations
│   └── report_writer.py       # Agent 3: markdown report generation
├── queries/
│   └── ad_hoc_library.sql     # 8 documented business queries
├── app/
│   └── streamlit_app.py       # 4-tab UI
├── powerbi/
│   └── campaign_dashboard.pbix
├── exports/
│   └── weekly_report_sample.md
├── requirements.txt
├── .env.example
└── README.md
```

---

*Aaron Christian · MS Information Systems, SDSU · [GitHub](https://github.com/AaronFChristian)*
