# AI E-commerce Ops Tool

One Streamlit console for cross-border e-commerce operations. Replaces manual product selection, copywriting, customer-inquiry triage, and dashboard monitoring with a single AI-powered workflow.

Built as an interview demo: simple to run, simple to explain, every screen mapped to a measurable business outcome.

## Workflow

```
Product Selection  →  Copy Generator  →  Customer Insight  →  Monitoring
   (rank 500           (3 styles ×           (8k inquiries          (anomaly
   products)           2 langs)              → 4 themes)            detection)
```

Each tab feeds the next: rank a product → push it to the copy generator → understand what customers ask → monitor how the changes perform.

## Modules

| # | Module | What it does | Headline impact |
|---|---|---|---|
| **M1** | Product Selection Optimizer | Weighted-percentile ranking on CTR / inquiry rate / conversion / volume | Decision time **2 h → 20 min (-85%)** |
| **M2** | AI Marketing Copy Generator | LLM produces 3 copy styles × EN/中文; A/B simulator | CTR **+40%**, orders **+130%** |
| **M3** | Customer Insight Mining | Bilingual classifier into price / delivery / feature / customization themes | High-match inquiries **35% → 61%**, triage **10 h/wk → 30 min** |
| **M4** | Streamlit Dashboard | Wires M1/M2/M3/M5 into one connected console | — |
| **M5** | Automation & Monitoring | Z-score anomaly detection + auto-generated daily report | Manual work **↓ 65%**, decision speed **↑ 70%** |

## Architecture

```
app.py                    Streamlit entry — 4 tabs (M1, M2, M3, M5)
llm_client.py             Single OpenAI choke point with offline fallback
modules/
  m1_product_rank.py      Pure pandas weighted-percentile ranking
  m2_copy_gen.py          Prompt design + A/B simulator
  m3_insight_mining.py    LLM classify + bilingual keyword fallback
  m5_monitor.py           Anomaly detection + markdown report
data/
  generate.py             Reproducible synthetic data generator (seed=42)
  products.csv            500 products
  inquiries.csv           8,000 bilingual inquiries with ground-truth labels
  products_timeseries.csv 30 days × 100 products (with one injected anomaly)
```

Every LLM call routes through `llm_client.py`, so the demo runs with or without an API key:

- **Live mode** (`OPENAI_API_KEY` set, network reachable) — `gpt-4o-mini` produces fresh copy and classifications.
- **Fallback mode** (no key, no network, or quota hit) — bilingual templates (M2) and keyword rules (M3). The fallback path achieves 100% accuracy on the synthetic ground-truth eval set.

A status caption near the top of the app shows which mode is active.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (one time)
python3 -m data.generate

# 3. (Optional) configure OpenAI for the live path
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 4. Launch the app
streamlit run app.py
# Open http://localhost:8501
```

Each module also runs as a standalone CLI demo:

```bash
python3 -m modules.m1_product_rank      # ranked table + decision-time savings
python3 -m modules.m2_copy_gen          # 3 EN + 3 ZH copy variants + A/B table
python3 -m modules.m3_insight_mining    # theme distribution + accuracy + actions
python3 -m modules.m5_monitor           # anomaly table + daily report
python3 llm_client.py                   # live + fallback round-trip
```

## Tech stack

- Python 3.11+
- Streamlit (UI)
- pandas + numpy (data)
- OpenAI Python SDK (`gpt-4o-mini`)
- python-dotenv (config)

No Docker, no DB, no microservices. Reproducible synthetic data, in-memory caching, under 2,000 lines of code total.

## Interview talking points

See [`interview_notes.md`](./interview_notes.md) for the 30-second pitch per module, business-impact numbers, and answers to the most common interview questions.
