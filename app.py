"""AI E-commerce Ops Tool - Streamlit entry point.

Run:
    streamlit run app.py

Tabs grow as modules land. Currently: Marketing Copy (M2).
Next: Customer Insight (M3), Product Ranking (M1 view), Monitoring (M5).
"""
from __future__ import annotations

import streamlit as st

from llm_client import is_live, status
from modules.m2_copy_gen import (
    AB_UPLIFT,
    STYLES,
    TARGET_MARKETS,
    ab_test_simulation,
    generate_all,
)

st.set_page_config(page_title="AI E-commerce Ops Tool", layout="wide")

st.title("AI E-commerce Ops Tool")
mode = "LIVE" if is_live() else "FALLBACK"
st.caption(f"LLM: {mode}  ·  model: {status()['model']}  ·  reason: {status()['reason'] or 'n/a'}")

tab_copy, = st.tabs(["Marketing Copy"])

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
