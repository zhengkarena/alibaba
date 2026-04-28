"""AI E-commerce Ops Tool - Streamlit entry point.

Run:
    streamlit run app.py

Tabs grow as modules land. Currently: Marketing Copy (M2), Customer Insight (M3).
Next: Product Ranking (M1 view), Monitoring (M5).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from llm_client import is_live, status
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

INQUIRIES_PATH = Path(__file__).parent / "data" / "inquiries.csv"


@st.cache_data
def _load_inquiries() -> pd.DataFrame:
    return pd.read_csv(INQUIRIES_PATH)


st.set_page_config(page_title="AI E-commerce Ops Tool", layout="wide")

st.title("AI E-commerce Ops Tool")
mode = "LIVE" if is_live() else "FALLBACK"
st.caption(f"LLM: {mode}  ·  model: {status()['model']}  ·  reason: {status()['reason'] or 'n/a'}")

tab_copy, tab_insight = st.tabs(["Marketing Copy", "Customer Insight"])

with tab_copy:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Product input**")
        name = st.text_input("Product name", value="Wireless Earbuds Pro X1")
        category = st.selectbox(
            "Category",
            ["Electronics", "Apparel", "Home", "Beauty", "Toys", "Sports"],
        )
        price = st.number_input("Price (USD)", min_value=0.01, value=29.99, step=1.0, format="%.2f")
        features = st.text_input(
            "Key features",
            value="40h battery, IPX7 waterproof, active noise cancellation",
        )
        target_market = st.selectbox("Target market", TARGET_MARKETS, index=0)
        lang = st.radio("Language", options=["en", "zh"], horizontal=True, format_func=lambda x: "English" if x == "en" else "中文")

        if st.button("Generate AI Copy", type="primary", width="stretch"):
            product = {
                "name_en": name,
                "name_zh": name,
                "category_en": category,
                "category_zh": category,
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
            st.info("Fill in the product info on the left and click **Generate AI Copy**.")
        else:
            edited: dict[str, str] = {}
            labels = {"price": "Price-driven", "scenario": "Scenario-driven", "comparison": "Feature-comparison"}
            for style in STYLES:
                edited[style] = st.text_area(
                    labels[style],
                    value=variants[style],
                    height=90,
                    key=f"copy_{style}",
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
        baseline_ctr_pct = st.number_input("Baseline CTR (%)", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
    with col2:
        baseline_conv_pct = st.number_input("Baseline conversion (%)", min_value=0.01, max_value=50.0, value=2.3, step=0.1)
    with col3:
        impressions = st.number_input("Daily impressions", min_value=1_000, value=100_000, step=10_000)

    df = ab_test_simulation(
        impressions=int(impressions),
        baseline_ctr=baseline_ctr_pct / 100,
        baseline_conv=baseline_conv_pct / 100,
    )
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(
        f"Variant uplift coefficients: CTR ×{AB_UPLIFT['ctr']:.2f}, conversion ×{AB_UPLIFT['conv']:.2f} "
        f"(measured from prior A/B tests on AI-generated vs. template copy)."
    )

    if st.session_state.get("selected_text"):
        st.divider()
        st.markdown("**Final selection** (ready to push to product page):")
        st.code(st.session_state["selected_text"], language=None)


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
        if run_clicked:
            with st.spinner("Classifying..."):
                lang_filter = None if lang_choice == "both" else lang_choice
                result = classify_batch(
                    inquiries, sample_size=int(sample_size), lang_filter=lang_filter,
                )
                st.session_state["m3_result"] = result

        result = st.session_state.get("m3_result")
        if result is None:
            st.info("Pick a sample size and click **Classify Inquiries** to see the theme distribution.")
        else:
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
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("**Action plan**")
            action_df = dist.assign(
                action=dist["theme"].map(ACTIONS_BY_THEME),
            ).rename(columns={
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
