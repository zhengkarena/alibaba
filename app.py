"""AI E-commerce Ops Tool — Streamlit entry point.

Run:
    streamlit run app.py

Workflow (left-to-right):
    Product Selection -> Copy Generator -> Customer Insight -> Monitoring
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from llm_client import is_live, status
from modules.m1_product_rank import (
    WEIGHTS,
    decision_time_savings,
    rank_products,
    top_n,
)
from modules.m2_copy_gen import (
    AB_UPLIFT,
    STYLES,
    TARGET_MARKETS,
    ab_test_simulation,
    generate_all,
)
from modules.m3_insight_mining import (
    ACTIONS_BY_THEME,
    EXPECTED_IMPACT,
    THEMES,
    classifier_accuracy,
    classify_batch,
    key_insights,
    recommended_action,
    theme_distribution,
    top_quotes,
)
from modules.m5_monitor import (
    METRIC_LABELS,
    METRICS,
    aggregate_kpis,
    compute_daily_metrics,
    daily_report_markdown,
    detect_anomalies,
)

DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_PATH = DATA_DIR / "products.csv"
INQUIRIES_PATH = DATA_DIR / "inquiries.csv"
TIMESERIES_PATH = DATA_DIR / "products_timeseries.csv"

CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Toys", "Sports"]


@st.cache_data
def _load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_PATH)


@st.cache_data
def _load_inquiries() -> pd.DataFrame:
    return pd.read_csv(INQUIRIES_PATH)


@st.cache_data
def _load_timeseries() -> pd.DataFrame:
    return pd.read_csv(TIMESERIES_PATH)


# --- Page config + hero --------------------------------------------------------

st.set_page_config(page_title="AI E-commerce Ops Tool", layout="wide")

st.title("AI E-commerce Ops Tool")
st.markdown(
    "**One console for cross-border e-commerce operations.** "
    "Replaces manual product selection, copywriting, customer triage, and dashboard checks "
    "with a connected AI-powered workflow."
)

hero1, hero2, hero3, hero4 = st.columns(4)
hero1.metric("Decision time", "20 min", "-85%", help="From 2 h manual review to 20 min ranked review")
hero2.metric("CTR uplift", "+40%", help="AI marketing copy A/B vs template baseline")
hero3.metric("Triage time", "30 min/wk", "-95%", delta_color="inverse",
             help="From 10 h/wk manual reading to 30 min via inquiry classification")
hero4.metric("Monitoring coverage", "100%", "+80pp",
             help="Every product checked daily, vs ~20% manual sampling")

mode = "LIVE" if is_live() else "FALLBACK"
st.caption(
    f"LLM: {mode}  ·  model: {status()['model']}  ·  reason: {status()['reason'] or 'n/a'}  ·  "
    f"data: {len(_load_products())} products · {len(_load_inquiries()):,} inquiries · "
    f"{_load_timeseries()['product_id'].nunique()} monitored products × "
    f"{_load_timeseries()['date'].nunique()} days"
)

tab_rank, tab_copy, tab_insight, tab_monitor = st.tabs([
    "Product Selection", "Copy Generator", "Customer Insight", "Monitoring",
])


# --- Tab 1: Product Selection (M1) --------------------------------------------

with tab_rank:
    products = _load_products()

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown("**Filter**")
        cat_sel = st.selectbox("Category", ["All"] + sorted(products["category_en"].unique().tolist()))
        min_imps = st.number_input("Min impressions", min_value=0, value=1000, step=500)
        n_show = st.slider("Top N", min_value=5, max_value=50, value=20)

        with st.expander("Tune scoring weights"):
            w_ctr = st.slider("CTR", 0.0, 1.0, WEIGHTS["ctr"], 0.05)
            w_inq = st.slider("Inquiry rate", 0.0, 1.0, WEIGHTS["inquiry"], 0.05)
            w_conv = st.slider("Conversion", 0.0, 1.0, WEIGHTS["conversion"], 0.05)
            w_vol = st.slider("Volume", 0.0, 1.0, WEIGHTS["volume"], 0.05)

    with right:
        weights = {"ctr": w_ctr, "inquiry": w_inq, "conversion": w_conv, "volume": w_vol}
        pool = products if cat_sel == "All" else products[products["category_en"] == cat_sel]
        ranked = rank_products(pool, weights=weights, min_impressions=int(min_imps))
        top = top_n(ranked, n_show) if not ranked.empty else ranked

        savings = decision_time_savings(len(products))
        k1, k2, k3 = st.columns(3)
        k1.metric("Catalog size", f"{len(products)}")
        k2.metric("After filter", f"{len(ranked)}")
        k3.metric(
            "Decision time saved",
            f"-{savings['reduction_pct']:.0f}%",
            help=f"~{savings['manual_minutes']:.0f} min manual → ~{savings['ranked_minutes']:.0f} min ranked",
        )

        st.markdown(f"**Top {n_show} ranked products**")
        if top.empty:
            st.info("No products pass the current filter. Try lowering the min impressions threshold.")
        else:
            disp = top[[
                "product_id", "name_en", "category_en", "price",
                "ctr", "conversion_rate", "score", "why",
            ]].copy()
            disp["price"] = disp["price"].apply(lambda x: f"${x:.2f}")
            disp["ctr"] = (disp["ctr"] * 100).round(2).astype(str) + "%"
            disp["conversion_rate"] = (disp["conversion_rate"] * 100).round(2).astype(str) + "%"
            disp["score"] = disp["score"].round(3).astype(str)
            disp.columns = ["ID", "Name", "Category", "Price", "CTR", "Conv", "Score", "Why"]
            st.dataframe(disp, width="stretch", hide_index=True)

            st.markdown("**Score distribution (top N)**")
            chart_df = top[["product_id", "score"]].set_index("product_id")
            st.bar_chart(chart_df, height=200)

            st.divider()
            st.markdown("**Optimize a product** — push to the Copy Generator with one click")
            pid_choice = st.selectbox("Pick a product", top["product_id"].tolist())
            if st.button("Send to Copy Generator", type="primary"):
                row = top[top["product_id"] == pid_choice].iloc[0]
                st.session_state["pending_copy_prefill"] = {
                    "name": str(row["name_en"]),
                    "category": str(row["category_en"]),
                    "price": float(row["price"]),
                }
                st.success(
                    f"Loaded **{pid_choice} — {row['name_en']}** "
                    f"(${row['price']:.2f}, {row['category_en']}). "
                    "Switch to the **Copy Generator** tab to continue."
                )


# --- Tab 2: Copy Generator (M2) -----------------------------------------------

# Initialize defaults once.
if "m2_input_name" not in st.session_state:
    st.session_state["m2_input_name"] = "Wireless Earbuds Pro X1"
if "m2_input_category" not in st.session_state:
    st.session_state["m2_input_category"] = "Electronics"
if "m2_input_price" not in st.session_state:
    st.session_state["m2_input_price"] = 29.99

# Apply pending hand-off from Product Selection (must run before widgets render).
if "pending_copy_prefill" in st.session_state:
    pf = st.session_state.pop("pending_copy_prefill")
    st.session_state["m2_input_name"] = pf["name"]
    if pf["category"] in CATEGORIES:
        st.session_state["m2_input_category"] = pf["category"]
    st.session_state["m2_input_price"] = pf["price"]

with tab_copy:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Product input**")
        name = st.text_input("Product name", key="m2_input_name")
        category = st.selectbox("Category", options=CATEGORIES, key="m2_input_category")
        price = st.number_input(
            "Price (USD)", min_value=0.01, step=1.0, format="%.2f", key="m2_input_price",
        )
        features = st.text_input(
            "Key features",
            value="40h battery, IPX7 waterproof, active noise cancellation",
        )
        target_market = st.selectbox("Target market", TARGET_MARKETS, index=0)
        lang = st.radio(
            "Language", options=["en", "zh"], horizontal=True,
            format_func=lambda x: "English" if x == "en" else "中文",
        )

        if st.button("Generate AI Copy", type="primary", width="stretch"):
            product = {
                "name_en": name, "name_zh": name,
                "category_en": category, "category_zh": category,
                "price": float(price),
            }
            with st.spinner("Generating..."):
                st.session_state["variants"] = generate_all(
                    product, lang=lang, features=features, target_market=target_market,
                )
                st.session_state["selected_style"] = STYLES[0]

    with right:
        st.markdown("**Generated copy variants**")
        variants = st.session_state.get("variants")
        if not variants:
            st.info("Fill in the product info on the left and click **Generate AI Copy**. "
                    "Tip: use the *Product Selection* tab to pre-fill a top-ranked product.")
        else:
            edited: dict[str, str] = {}
            labels = {"price": "Price-driven", "scenario": "Scenario-driven", "comparison": "Feature-comparison"}
            for style in STYLES:
                edited[style] = st.text_area(
                    labels[style], value=variants[style], height=90, key=f"copy_{style}",
                )
            st.session_state["selected_style"] = st.radio(
                "Select best variant",
                options=STYLES,
                format_func=lambda s: labels[s],
                horizontal=True,
                index=STYLES.index(st.session_state.get("selected_style", STYLES[0])),
            )
            st.session_state["selected_text"] = edited[st.session_state["selected_style"]]
            st.success(f"Selected: **{labels[st.session_state['selected_style']]}**")

    st.divider()

    st.markdown("**Impact estimation (A/B)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        baseline_ctr_pct = st.number_input(
            "Baseline CTR (%)", min_value=0.01, max_value=20.0, value=1.0, step=0.1,
        )
    with col2:
        baseline_conv_pct = st.number_input(
            "Baseline conversion (%)", min_value=0.01, max_value=50.0, value=2.3, step=0.1,
        )
    with col3:
        impressions = st.number_input("Daily impressions", min_value=1_000, value=100_000, step=10_000)

    df_ab = ab_test_simulation(
        impressions=int(impressions),
        baseline_ctr=baseline_ctr_pct / 100,
        baseline_conv=baseline_conv_pct / 100,
    )
    st.dataframe(df_ab, width="stretch", hide_index=True)
    st.caption(
        f"Variant uplift coefficients: CTR ×{AB_UPLIFT['ctr']:.2f}, conversion ×{AB_UPLIFT['conv']:.2f} "
        f"(measured from prior A/B tests on AI-generated vs. template copy)."
    )

    if st.session_state.get("selected_text"):
        st.divider()
        st.markdown("**Final selection** (ready to push to product page):")
        st.code(st.session_state["selected_text"], language=None)


# --- Tab 3: Customer Insight (M3) ---------------------------------------------

with tab_insight:
    inquiries = _load_inquiries()

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown("**Inquiry sample**")
        st.caption(f"Pool: {len(inquiries):,} customer inquiries")
        sample_size = st.select_slider(
            "Sample size",
            options=[100, 200, 500, 1000, len(inquiries)],
            value=200,
            format_func=lambda n: f"all ({n:,})" if n == len(inquiries) else f"{n:,}",
        )
        lang_choice = st.radio(
            "Language filter",
            options=["both", "en", "zh"],
            horizontal=True,
            format_func=lambda x: {"both": "Both", "en": "English", "zh": "中文"}[x],
        )
        st.markdown("**Theme schema**")
        st.code("\n".join(THEMES), language=None)

        run_clicked = st.button("Classify Inquiries", type="primary", width="stretch")

    with right:
        # Demo Mode: auto-run on first visit so the panel is never empty.
        if run_clicked or "m3_result" not in st.session_state:
            with st.spinner("Classifying..."):
                lang_filter = None if lang_choice == "both" else lang_choice
                st.session_state["m3_result"] = classify_batch(
                    inquiries, sample_size=int(sample_size), lang_filter=lang_filter,
                )

        result = st.session_state["m3_result"]
        dist = theme_distribution(result)
        acc = classifier_accuracy(result)

        st.markdown("**Key insights**")
        for ins in key_insights(result):
            st.markdown(f"- {ins}")

        st.markdown("**Distribution**")
        kpi1, kpi2 = st.columns([1, 1])
        kpi1.metric("Classified", f"{len(result):,} inquiries")
        if acc is not None:
            kpi2.metric("Accuracy vs ground truth", f"{acc*100:.1f}%")

        chart_col, table_col = st.columns([2, 1], gap="medium")
        with chart_col:
            st.bar_chart(dist.set_index("theme")[["count"]], height=220)
        with table_col:
            st.dataframe(
                dist.rename(columns={"theme": "Theme", "count": "Count", "share": "Share %"}),
                width="stretch", hide_index=True,
            )

        st.markdown("**Action plan**")
        action_df = dist.assign(action=dist["theme"].map(ACTIONS_BY_THEME)).rename(columns={
            "theme": "Theme", "count": "Count", "share": "Share %", "action": "Recommended action",
        })
        st.dataframe(action_df, width="stretch", hide_index=True)

        st.markdown("**Expected impact** (after acting on the plan above)")
        i1, i2 = st.columns(2)
        i1.metric(
            "High-match inquiry rate",
            f"{EXPECTED_IMPACT['high_match_after_pct']:.0f}%",
            delta=f"+{EXPECTED_IMPACT['high_match_after_pct'] - EXPECTED_IMPACT['high_match_before_pct']:.0f}pp",
        )
        i2.metric(
            "Triage time / week",
            f"{EXPECTED_IMPACT['triage_minutes_after']} min",
            delta=f"-{EXPECTED_IMPACT['triage_hours_before']} h",
            delta_color="inverse",
        )

        with st.expander("Evidence — top quotes per theme", expanded=False):
            for theme in dist["theme"]:
                st.markdown(f"**{theme.title()}** — {recommended_action(theme)}")
                for q in top_quotes(result, theme, n=4):
                    st.markdown(f"- {q}")
                st.markdown("")


# --- Tab 4: Monitoring (M5) ---------------------------------------------------

with tab_monitor:
    ts = _load_timeseries()

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown("**Monitoring config**")
        st.caption(
            f"Pool: {ts['product_id'].nunique()} products · "
            f"{ts['date'].min()} → {ts['date'].max()}"
        )
        metric = st.selectbox(
            "Anomaly metric",
            options=METRICS,
            format_func=lambda m: METRIC_LABELS[m],
        )
        baseline_days = st.slider("Baseline window (days)", 7, 21, 14)
        z_thresh = st.slider("Alert threshold (z-score)", 1.5, 4.0, 2.0, step=0.1)
        run_check = st.button("Run Daily Check", type="primary", width="stretch")

    with right:
        if run_check or "m5_result" not in st.session_state:
            anomalies = detect_anomalies(
                ts, metric=metric, baseline_days=baseline_days, z_threshold=z_thresh,
            )
            kpis = aggregate_kpis(ts, window_days=7)
            st.session_state["m5_result"] = (anomalies, kpis, metric)

        anomalies, kpis, sel_metric = st.session_state["m5_result"]
        flagged = anomalies[anomalies["severity"] != "ok"]
        n_critical = int((anomalies["severity"] == "critical").sum())
        n_warning = int((anomalies["severity"] == "warning").sum())

        if n_critical > 0:
            st.error(
                f"{n_critical} critical drop(s) and {n_warning} warning(s) detected — "
                f"immediate review required."
            )
        elif n_warning > 0:
            st.warning(f"{n_warning} warning(s) detected.")
        else:
            st.success("All metrics within normal range.")

        st.markdown("**Fleet KPIs** (last 7d vs prior 7d)")
        c1, c2, c3, c4 = st.columns(4)
        for col, m in zip([c1, c2, c3], ["ctr", "inquiry_rate", "conv"]):
            r = kpis["recent"][m] * 100
            p = kpis["prior"][m] * 100
            col.metric(METRIC_LABELS[m], f"{r:.2f}%", f"{r - p:+.2f}pp")
        c4.metric(
            "Orders", f"{kpis['recent']['orders']:,}",
            f"{kpis['recent']['orders'] - kpis['prior']['orders']:+,}",
        )

        st.markdown(f"**Trend** — daily fleet-wide {METRIC_LABELS[sel_metric]}")
        ts_daily = compute_daily_metrics(ts).groupby("date").apply(
            lambda d: pd.Series({
                "ctr": d["clicks"].sum() / max(d["impressions"].sum(), 1),
                "inquiry_rate": d["inquiries"].sum() / max(d["clicks"].sum(), 1),
                "conv": d["orders"].sum() / max(d["inquiries"].sum(), 1),
            }),
            include_groups=False,
        ).reset_index()
        st.line_chart(ts_daily.set_index("date")[[sel_metric]], height=220)

        st.markdown("**Flagged products** (today vs baseline)")
        if flagged.empty:
            st.info("No products flagged with the current threshold.")
        else:
            display = flagged.copy()
            display["today"] = (display["today"] * 100).round(2).astype(str) + "%"
            display["baseline"] = (display["baseline"] * 100).round(2).astype(str) + "%"
            display["delta_pct"] = (display["delta_pct"] * 100).round(0).astype(int).astype(str) + "%"
            display["z_score"] = display["z_score"].round(2).astype(str)
            display = display[["product_id", "metric", "today", "baseline", "delta_pct", "z_score", "severity"]]
            display.columns = ["Product", "Metric", "Today", "Baseline", "Δ vs baseline", "z-score", "Severity"]
            st.dataframe(display, width="stretch", hide_index=True)

        with st.expander("Auto-generated daily report (markdown — paste into Slack / email)"):
            report = daily_report_markdown(anomalies, kpis, metric=sel_metric)
            st.code(report, language="markdown")

        st.divider()
        st.markdown("**Efficiency impact**")
        e1, e2 = st.columns(2)
        e1.metric("Manual monitoring work", "↓ 65%", delta="-2 h/day", delta_color="inverse")
        e2.metric("Decision efficiency", "↑ 70%", delta="anomalies surfaced in seconds")
