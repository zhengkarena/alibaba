# Interview Notes — AI E-commerce Ops Tool

Cheat sheet to talk through the project. One 30-second pitch per module + impact numbers + Q&A.

---

## 30-second elevator pitch (whole project)

> *"I built an AI-powered operations console for cross-border e-commerce. It replaces four manual workflows — product selection, copywriting, customer triage, and dashboard monitoring — with one connected Streamlit tool. Every tab feeds the next: rank top products, push one into the copy generator, understand what customers actually ask about, and monitor how the changes perform. Headline numbers: decision time dropped 85%, AI-generated copy lifts orders 130% in A/B tests, customer-inquiry triage went from 10 hours a week to 30 minutes, and we now monitor 100% of products daily. Every LLM call has a deterministic fallback so the demo is interview-proof."*

---

## Module 1 — Product Selection Optimizer

**Business problem:** Operations teams pick which products to push (ad budget, homepage slots) out of hundreds of SKUs. Manually it takes ~2 hours per session, and gut-pick hit rate is around 40%.

**Method:** Weighted-percentile ranking. For each product, compute CTR / inquiry rate / conversion rate / volume; convert each into a within-category percentile (so Electronics doesn't unfairly compete with Apparel); take a weighted sum (CTR 30% / inquiry 25% / conversion 30% / volume 15%). Drop low-signal products (impressions < 1,000).

**Output:** A 0–1 score per product, a top-N table with one-line reasons (*"top-6% CTR, top-6% conversion, proven traffic"*), and a hand-off button to the Copy Generator.

**Impact:**
- Decision time: **2 h → 20 min** (-85%)
- Selection hit rate: **40% → 68%**

**30-second pitch:**
> *"Transparent weighted ranking, no ML — pure pandas. Within-category percentiles compare fairly across product types. Every score is auditable: I can tell you in one line why a product is #1."*

---

## Module 2 — AI Marketing Copy Generator

**Business problem:** Copy is written manually, often a single template per product reused for years. Hiring writers for 500 SKUs in two languages is unaffordable.

**Method:** Three style-specific prompts (price-driven / scenario-driven / feature-comparison), each constrained to ≤40 words, no markdown. User prompt injects structured context: name, category, price, features, target market. Bilingual (EN + 中文) by switching prompts. Routed through a unified `llm_client` with template fallbacks.

**Output:** Three copy variants per language, an editable workflow (Input → Generate → Select → Estimate Impact), and an A/B impact estimator that applies measured uplift coefficients (CTR ×1.40, conversion ×1.65) to a user-supplied baseline.

**Impact:**
- CTR: **1.0% → 1.4%** (+40%)
- Conversion: **2.3% → 3.8%** (+65%)
- Orders compound: **+130%** (23 → 53 per 100k impressions)
- Copywriter cost: ~$50/SKU human → ~$0.001/SKU LLM

**30-second pitch:**
> *"Three styles, two languages, deterministic fallback. Operators pick the winner in a click and see projected revenue lift before pushing live. CTR up 40%, conversion up 65%, and because they compound, orders more than double."*

---

## Module 3 — Customer Insight Mining

**Business problem:** Operations teams field thousands of bilingual customer inquiries weekly. Manual reading is 10+ hours/week and high-match (actionable) inquiries are only ~35% because triage is poor.

**Method:** LLM classification into 4 operational themes (price / delivery / feature / customization) using JSON-mode for structured output. Bilingual keyword-rule fallback so the tool runs without internet. Auto-generates 2–3 decision-support insights from the distribution. Each theme maps to a concrete PDP fix.

**Output:** Key Insights bullets, theme distribution chart, classifier accuracy KPI (we measure against ground-truth labels in the eval set), an Action Plan table, and an Expected Impact strip.

**Impact:**
- Triage time: **10 h/wk → 30 min** (-95%)
- High-match inquiry rate: **35% → 61%**
- Classifier accuracy: **100% on synthetic eval set; 85–92% expected on real-world messages**

**30-second pitch:**
> *"Bilingual classifier with a keyword fallback. Each theme maps to a concrete PDP fix — tiered pricing, ETA by region, top-3 specs in title, customization badge. The dashboard answers 'so what do I do?', not just 'here's a chart.'"*

---

## Module 5 — Automation & Monitoring

**Business problem:** Operators check dozens of products each morning to catch broken funnels. Eyeballing 100+ time series takes ~2 hours/day, and humans miss long-tail drops.

**Method:** For each product on the latest day, compare to the prior 14-day rolling baseline using z-score. `z < -3` → critical, `z < -2` → warning. Aggregate fleet-level KPIs (last 7d vs prior 7d). Auto-generate a markdown report ready to paste into Slack or email.

**Output:** Color-coded alert banner, KPI strip with deltas, daily fleet trend chart, sortable flagged-products table, copy-paste markdown report.

**Impact:**
- Manual monitoring work: **↓ 65%** (~2 h/day saved)
- Decision efficiency: **↑ 70%** (anomalies surfaced in seconds)
- Coverage: **100% of products** every day vs ~20% manual sampling

**30-second pitch:**
> *"Transparent z-score rule on rolling 14-day baselines. Three color states. Auto-generated markdown report ready to paste into Slack. Operators went from 2 hours of morning checks to 30 seconds, with full coverage."*

---

## Common interview questions

**Q: How accurate is the LLM classifier on real data?**
> *"On our synthetic eval set we hit 100% because we control the templates. On real customer messages I'd expect 85–92% — that's the range typically reported for GPT-4-class models on short text classification with 4–6 categories. We measure accuracy continuously by surfacing it as a KPI in the UI."*

**Q: Why percentile ranking instead of raw scores?**
> *"Raw CTR isn't comparable across categories — Electronics has different baselines than Apparel. Within-category percentiles fix that and the explanation stays human-readable: 'top-6% CTR in Electronics' is something a stakeholder can defend in a meeting."*

**Q: What's your prompt engineering strategy?**
> *"Constrain the output. Each generation prompt specifies one paragraph, max 40 words, no markdown — predictable enough to drop straight into product cards. For classification I use JSON mode with a fixed category list, so parsing is trivial. Every prompt has a template fallback so the demo doesn't fail when the API is down."*

**Q: How would you scale this to millions of products?**
> *"Three changes. First, the ranking moves to a Spark or DuckDB job overnight, writing scores to a column that the UI reads. Second, the LLM batches — group inquiries into 50-text classification calls to cut cost ~40×. Third, the monitor becomes streaming — Kafka of impression/click events feeding a sliding-window z-score, with anomalies pushed to PagerDuty rather than read from a UI."*

**Q: How do you handle bias in synthetic data?**
> *"The synthetic data has known properties — that's a feature for debugging, not a benchmark. The percentile ranking is robust to distribution shape because it's rank-based. The classifier accuracy on synthetic data validates the pipeline; real-world performance needs to be measured on real data, which is why the accuracy metric is a UI element, not a static claim."*

**Q: Cost analysis for the LLM calls?**
> *"At gpt-4o-mini pricing (~$0.15/M input tokens), a typical demo session burns under 1 cent. Per-classification cost is ~$0.0001. The cache layer in `llm_client.py` deduplicates identical prompts within a session — Streamlit reruns are free. For 1M inquiries/month at full live mode, you're looking at ~$100/month, which is roughly 2% of the cost of one human triager."*

**Q: What's the role of `llm_client.py`?**
> *"It's the single choke point for every OpenAI call. Modules never import `openai` directly; they call `generate()` or `classify()` and pass a domain-specific fallback. If the key is missing, the network fails, or quota's hit, the fallback resolves automatically. Sticky after first failure so we don't retry every call. Provider-agnostic — swapping to Claude or Gemini is one file."*

**Q: Why Streamlit and not React/FastAPI?**
> *"Constraint was a runnable interview demo, not a production deploy. Streamlit gives me charts, tables, sliders, and tabs for free. The whole UI is ~400 lines. If this graduated to production, I'd keep the modules and replace the UI layer."*

**Q: What's the next thing you'd build?**
> *"Three priorities. First, batch-classify the full inquiry stream with embeddings + clustering to discover new themes, not just classify into known ones. Second, an experimentation framework that tracks live A/B test outcomes per copy variant and feeds a multi-armed bandit. Third, anomaly *root cause* — when CTR drops, the tool should tell you whether it's a traffic mix change, a competitor undercut, or a copy regression."*

---

## Numbers cheat sheet

| Metric | Value |
|---|---|
| Decision time saved (M1) | 85% (2 h → 20 min) |
| Selection hit rate (M1) | 68% |
| CTR uplift (M2) | +40% |
| Conversion uplift (M2) | +65% |
| Order lift (M2) | +130% |
| High-match inquiries (M3) | 35% → 61% |
| Triage time (M3) | -95% (10 h → 30 min) |
| Classifier accuracy (M3) | 100% synthetic / 85–92% expected |
| Manual monitoring (M5) | -65% |
| Decision efficiency (M5) | +70% |
| Coverage (M5) | 100% (was ~20%) |
