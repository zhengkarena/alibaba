"""AI 跨境电商运营工具 — Streamlit 中文版入口.

运行:
    streamlit run app_zh.py

工作流(从左到右):
    选品分析 -> 文案生成 -> 客户洞察 -> 运营监控

本文件是 app.py 的纯中文显示版本,业务逻辑、模块、数据完全共用,
只对界面显示文字做本地化翻译。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# 必须是页面上第一个 Streamlit 命令。
st.set_page_config(page_title="AI 跨境电商运营工具", layout="wide")

from llm_client import is_live, status
import data_loader as dl
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
    EXPECTED_IMPACT,
    THEMES,
    classifier_accuracy,
    classify_batch,
    theme_distribution,
    top_quotes,
)
from modules.m5_monitor import (
    METRICS,
    aggregate_kpis,
    compute_daily_metrics,
    detect_anomalies,
)

CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Toys", "Sports"]

# --- 本地化映射(只用于显示,不影响数据键值) ---------------------------------

CATEGORY_ZH = {
    "Electronics": "电子产品",
    "Apparel": "服装",
    "Home": "家居",
    "Beauty": "美妆",
    "Toys": "玩具",
    "Sports": "运动",
    "All": "全部",
}

MARKET_ZH = {
    "USA": "美国",
    "Germany": "德国",
    "Japan": "日本",
    "Southeast Asia": "东南亚",
    "Brazil": "巴西",
    "UAE": "阿联酋",
}

THEME_ZH = {
    "price": "价格关注",
    "delivery": "物流时效",
    "feature": "产品特性",
    "customization": "定制需求",
}

ACTIONS_ZH = {
    "price": "在产品详情页添加阶梯价格表与起订量优惠",
    "delivery": "按区域显示预估发货与到货时间",
    "feature": "在产品标题中突出前 3 个核心规格",
    "customization": "添加「支持定制 / OEM」徽章",
}

INTERPRETATIONS_ZH = {
    "price": "说明价格透明度有提升空间,买家难以快速比较价值",
    "delivery": "说明物流与交期沟通存在不确定性",
    "feature": "说明产品页缺少关键规格 / 认证信息",
    "customization": "B2B 自有品牌 / OEM 需求强烈",
}

METRIC_LABELS_ZH = {
    "ctr": "点击率",
    "inquiry_rate": "询盘率",
    "conv": "转化率",
}

SEVERITY_ZH = {
    "critical": "严重",
    "warning": "警告",
    "ok": "正常",
}

STYLE_LABELS_ZH = {
    "price": "价格驱动",
    "scenario": "场景驱动",
    "comparison": "卖点对比",
}


def key_insights_zh(df: pd.DataFrame) -> list[str]:
    """以中文重新生成 2-3 条决策洞察(逻辑同 m3.key_insights,只改文案)。"""
    if df.empty:
        return []
    dist = theme_distribution(df)
    out: list[str] = []

    top = dist.iloc[0]
    out.append(
        f"**{THEME_ZH[top['theme']]}** 是当前最主要的咨询主题(占 {top['share']}%)— "
        f"{INTERPRETATIONS_ZH[top['theme']]}。"
    )

    lang_insight = None
    if "lang" in df.columns and df["lang"].nunique() > 1:
        for theme in THEMES:
            for lang in df["lang"].unique():
                lang_df = df[df["lang"] == lang]
                other_df = df[df["lang"] != lang]
                if len(lang_df) < 30 or len(other_df) < 30:
                    continue
                share = (lang_df["predicted_theme"] == theme).mean() * 100
                other = (other_df["predicted_theme"] == theme).mean() * 100
                if share - other >= 10:
                    label = "英文区流量" if lang == "en" else "中文区流量"
                    lang_insight = (
                        f"**{THEME_ZH[theme]}** 在{label}中显著集中"
                        f"({share:.0f}% vs {other:.0f}%)— 可能存在区域性问题。"
                    )
                    break
            if lang_insight:
                break
    if lang_insight:
        out.append(lang_insight)
    elif len(dist) >= 2:
        top2 = dist.head(2)
        combined = float(top2["share"].sum())
        names = " + ".join(THEME_ZH[t] for t in top2["theme"])
        out.append(
            f"**{names}** 合计占 {combined:.0f}% — 优先优化产品页中这两类信息可获得最大杠杆。"
        )

    if len(dist) >= 3:
        bottom = dist.iloc[-1]
        out.append(
            f"**{THEME_ZH[bottom['theme']]}** 占比最低({bottom['share']}%)"
            f"— 建议:{ACTIONS_ZH[bottom['theme']]}。"
        )

    return out[:3]


def daily_report_zh(anomalies: pd.DataFrame, kpis: dict, metric: str, top_n_flagged: int = 5) -> str:
    """中文版每日运营报告(逻辑同 m5.daily_report_markdown)。"""
    flagged = anomalies[anomalies["severity"] != "ok"]
    n_critical = int((anomalies["severity"] == "critical").sum())
    n_warning = int((anomalies["severity"] == "warning").sum())

    lines = [
        f"# 运营日报 — {kpis['latest_date']}",
        "",
        f"**对比窗口:** 近 {kpis['window_days']} 天 vs 之前 {kpis['window_days']} 天  ·  "
        f"**异常监控指标:** {METRIC_LABELS_ZH[metric]}",
        "",
        "## 核心 KPI",
    ]
    for m in ["ctr", "inquiry_rate", "conv"]:
        r = kpis["recent"][m] * 100
        p = kpis["prior"][m] * 100
        delta = r - p
        arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        lines.append(f"- **{METRIC_LABELS_ZH[m]}**: {r:.2f}% {arrow}(环比 {delta:+.2f}pp)")
    lines.append(
        f"- **近 {kpis['window_days']} 天订单数**: {kpis['recent']['orders']:,} "
        f"(之前 {kpis['window_days']} 天: {kpis['prior']['orders']:,})"
    )
    lines += ["", "## 告警", f"- 严重下跌:{n_critical} 个", f"- 警告:{n_warning} 个"]

    if len(flagged) > 0:
        lines += ["", "### 重点关注产品"]
        for _, r in flagged.head(top_n_flagged).iterrows():
            lines.append(
                f"- `{r['product_id']}` — {METRIC_LABELS_ZH[r['metric']]} "
                f"{r['today']*100:.2f}%(基线 {r['baseline']*100:.2f}%,"
                f"{r['delta_pct']*100:+.0f}%,z={r['z_score']:.2f})**{SEVERITY_ZH[r['severity']]}**"
            )
    return "\n".join(lines)


# --- 数据加载(与英文版完全相同) ---------------------------------------------


@st.cache_data
def _demo_products() -> pd.DataFrame:
    return dl.load_products()


@st.cache_data
def _demo_inquiries() -> pd.DataFrame:
    return dl.load_inquiries()


@st.cache_data
def _demo_timeseries() -> pd.DataFrame:
    return dl.load_timeseries()


def _resolve(load_fn, demo_fn, uploaded):
    if uploaded is None:
        return demo_fn(), {"source": "demo", "rows": None, "missing": None}
    try:
        df = load_fn(uploaded)
        return df, {"source": "uploaded", "rows": len(df), "missing": None}
    except dl.SchemaError as e:
        return demo_fn(), {"source": "demo (fallback)", "rows": None, "missing": e.missing}


# --- 侧边栏:数据源 -----------------------------------------------------------

with st.sidebar:
    st.markdown("### 数据源")
    mode = st.radio(
        "来源",
        options=["演示数据", "上传 CSV 文件"],
        index=0,
        help="演示数据已内置,应用即开即用。上传模式按文件独立生效:未上传的文件继续使用演示数据。",
    )

    up_products = up_inquiries = up_timeseries = None
    if mode == "上传 CSV 文件":
        st.caption("可只上传部分文件,未上传的将使用演示数据。")
        up_products = st.file_uploader("products.csv", type="csv", key="up_products")
        up_inquiries = st.file_uploader("inquiries.csv", type="csv", key="up_inquiries")
        up_timeseries = st.file_uploader("products_timeseries.csv", type="csv", key="up_timeseries")

products, prod_status = _resolve(dl.load_products, _demo_products, up_products)
inquiries, inq_status = _resolve(dl.load_inquiries, _demo_inquiries, up_inquiries)
timeseries, ts_status = _resolve(dl.load_timeseries, _demo_timeseries, up_timeseries)

with st.sidebar:
    st.markdown("**状态**")
    label_zh = {"products": "产品数据", "inquiries": "客户咨询", "timeseries": "时序数据"}
    for label, s in [("products", prod_status), ("inquiries", inq_status), ("timeseries", ts_status)]:
        if s["missing"]:
            st.error(
                f"{label_zh[label]}:列校验失败 · 缺少 `{', '.join(s['missing'])}` · 已回退到演示数据"
            )
        elif s["source"] == "uploaded":
            st.success(f"{label_zh[label]}:已上传({s['rows']:,} 行)")
        else:
            st.info(f"{label_zh[label]}:演示数据")

    st.markdown("---")
    st.caption(
        "未来连接器(Google Sheets / Shopify / 阿里巴巴导出 / CRM)接入点位于 "
        "`data_loader.load_from_source()` — 详见 README。"
    )


# --- 顶部 Hero ----------------------------------------------------------------

st.title("AI 跨境电商运营工具")
st.markdown(
    "**一站式跨境电商运营控制台。**用一个 AI 驱动的连贯工作流,"
    "替代手工选品、文案撰写、客户咨询分流和数据看板巡检。"
)

hero1, hero2, hero3, hero4 = st.columns(4)
hero1.metric("决策耗时", "20 分钟", "-85%", help="原本人工筛选 2 小时,现在排序后 20 分钟即可完成")
hero2.metric("点击率提升", "+40%", help="AI 文案 vs 模板文案的 A/B 实测")
hero3.metric("咨询处理时长", "30 分钟/周", "-95%", delta_color="inverse",
             help="原本每周 10 小时人工阅读,通过咨询自动分类压缩到 30 分钟")
hero4.metric("监控覆盖率", "100%", "+80pp",
             help="每天对所有产品做异常巡检,而原本人工抽查仅约 20%")

mode_text = "在线模式 (LIVE)" if is_live() else "降级模式 (FALLBACK)"
st.caption(
    f"LLM:{mode_text}  ·  模型:{status()['model']}  ·  原因:{status()['reason'] or '正常'}  ·  "
    f"数据:{len(products)} 个产品 · {len(inquiries):,} 条客户咨询 · "
    f"{timeseries['product_id'].nunique()} 个产品 × {timeseries['date'].nunique()} 天监控时序"
)

tab_rank, tab_copy, tab_insight, tab_monitor = st.tabs([
    "选品分析", "文案生成", "客户洞察", "运营监控",
])


# --- Tab 1:选品分析(M1) ---------------------------------------------------

with tab_rank:

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown("**筛选**")
        cat_options = ["All"] + sorted(products["category_en"].unique().tolist())
        cat_sel = st.selectbox(
            "品类",
            cat_options,
            format_func=lambda c: CATEGORY_ZH.get(c, c),
        )
        min_imps = st.number_input("最低曝光量", min_value=0, value=1000, step=500)
        n_show = st.slider("展示前 N 个", min_value=5, max_value=50, value=20)

        with st.expander("调整打分权重"):
            w_ctr = st.slider("点击率(CTR)", 0.0, 1.0, WEIGHTS["ctr"], 0.05)
            w_inq = st.slider("询盘率", 0.0, 1.0, WEIGHTS["inquiry"], 0.05)
            w_conv = st.slider("转化率", 0.0, 1.0, WEIGHTS["conversion"], 0.05)
            w_vol = st.slider("流量体量", 0.0, 1.0, WEIGHTS["volume"], 0.05)

    with right:
        weights = {"ctr": w_ctr, "inquiry": w_inq, "conversion": w_conv, "volume": w_vol}
        pool = products if cat_sel == "All" else products[products["category_en"] == cat_sel]
        ranked = rank_products(pool, weights=weights, min_impressions=int(min_imps))
        top = top_n(ranked, n_show) if not ranked.empty else ranked

        savings = decision_time_savings(len(products))
        k1, k2, k3 = st.columns(3)
        k1.metric("商品池规模", f"{len(products)}")
        k2.metric("筛选后", f"{len(ranked)}")
        k3.metric(
            "决策耗时节省",
            f"-{savings['reduction_pct']:.0f}%",
            help=f"人工评审约 {savings['manual_minutes']:.0f} 分钟 → 排序后约 {savings['ranked_minutes']:.0f} 分钟",
        )

        st.markdown(f"**得分前 {n_show} 的产品**")
        if top.empty:
            st.info("当前筛选条件下没有产品,试试降低最低曝光量阈值。")
        else:
            disp = top[[
                "product_id", "name_en", "category_en", "price",
                "ctr", "conversion_rate", "score", "why",
            ]].copy()
            disp["price"] = disp["price"].apply(lambda x: f"${x:.2f}")
            disp["ctr"] = (disp["ctr"] * 100).round(2).astype(str) + "%"
            disp["conversion_rate"] = (disp["conversion_rate"] * 100).round(2).astype(str) + "%"
            disp["score"] = disp["score"].round(3).astype(str)
            disp["category_en"] = disp["category_en"].map(lambda c: CATEGORY_ZH.get(c, c))
            disp.columns = ["产品 ID", "名称", "品类", "价格", "点击率", "转化率", "综合得分", "推荐理由"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

            st.markdown("**得分分布(前 N)**")
            chart_df = top[["product_id", "score"]].set_index("product_id")
            st.bar_chart(chart_df, height=200)

            st.divider()
            st.markdown("**优化某个产品** — 一键发送到「文案生成」")
            pid_choice = st.selectbox("选择产品", top["product_id"].tolist())
            if st.button("发送到文案生成", type="primary"):
                row = top[top["product_id"] == pid_choice].iloc[0]
                st.session_state["pending_copy_prefill"] = {
                    "name": str(row["name_en"]),
                    "category": str(row["category_en"]),
                    "price": float(row["price"]),
                }
                st.success(
                    f"已加载 **{pid_choice} — {row['name_en']}** "
                    f"(${row['price']:.2f},{CATEGORY_ZH.get(row['category_en'], row['category_en'])})。"
                    "切换到「**文案生成**」标签页继续。"
                )


# --- Tab 2:文案生成(M2) ---------------------------------------------------

if "m2_input_name" not in st.session_state:
    st.session_state["m2_input_name"] = "Wireless Earbuds Pro X1"
if "m2_input_category" not in st.session_state:
    st.session_state["m2_input_category"] = "Electronics"
if "m2_input_price" not in st.session_state:
    st.session_state["m2_input_price"] = 29.99

if "pending_copy_prefill" in st.session_state:
    pf = st.session_state.pop("pending_copy_prefill")
    st.session_state["m2_input_name"] = pf["name"]
    if pf["category"] in CATEGORIES:
        st.session_state["m2_input_category"] = pf["category"]
    st.session_state["m2_input_price"] = pf["price"]

with tab_copy:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**产品信息**")
        name = st.text_input("产品名称", key="m2_input_name")
        category = st.selectbox(
            "品类",
            options=CATEGORIES,
            format_func=lambda c: CATEGORY_ZH.get(c, c),
            key="m2_input_category",
        )
        price = st.number_input(
            "价格(美元)", min_value=0.01, step=1.0, format="%.2f", key="m2_input_price",
        )
        features = st.text_input(
            "核心卖点",
            value="40 小时续航、IPX7 防水、主动降噪",
        )
        target_market = st.selectbox(
            "目标市场",
            TARGET_MARKETS,
            index=0,
            format_func=lambda m: MARKET_ZH.get(m, m),
        )
        lang = st.radio(
            "文案语言", options=["en", "zh"], horizontal=True,
            format_func=lambda x: "英文" if x == "en" else "中文",
        )

        if st.button("生成 AI 文案", type="primary", use_container_width=True):
            product = {
                "name_en": name, "name_zh": name,
                "category_en": category, "category_zh": category,
                "price": float(price),
            }
            with st.spinner("生成中..."):
                st.session_state["variants"] = generate_all(
                    product, lang=lang, features=features, target_market=target_market,
                )
                st.session_state["selected_style"] = STYLES[0]

    with right:
        st.markdown("**生成的文案变体**")
        variants = st.session_state.get("variants")
        if not variants:
            st.info(
                "在左侧填写产品信息,点击「**生成 AI 文案**」。"
                "提示:可在「选品分析」标签页一键加载排名靠前的产品。"
            )
        else:
            edited: dict[str, str] = {}
            for style in STYLES:
                edited[style] = st.text_area(
                    STYLE_LABELS_ZH[style], value=variants[style], height=90, key=f"copy_{style}",
                )
            st.session_state["selected_style"] = st.radio(
                "选择最佳文案",
                options=STYLES,
                format_func=lambda s: STYLE_LABELS_ZH[s],
                horizontal=True,
                index=STYLES.index(st.session_state.get("selected_style", STYLES[0])),
            )
            st.session_state["selected_text"] = edited[st.session_state["selected_style"]]
            st.success(f"已选择:**{STYLE_LABELS_ZH[st.session_state['selected_style']]}**")

    st.divider()

    st.markdown("**效果预估(A/B)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        baseline_ctr_pct = st.number_input(
            "基线点击率 (%)", min_value=0.01, max_value=20.0, value=1.0, step=0.1,
        )
    with col2:
        baseline_conv_pct = st.number_input(
            "基线转化率 (%)", min_value=0.01, max_value=50.0, value=2.3, step=0.1,
        )
    with col3:
        impressions = st.number_input("日曝光量", min_value=1_000, value=100_000, step=10_000)

    df_ab = ab_test_simulation(
        impressions=int(impressions),
        baseline_ctr=baseline_ctr_pct / 100,
        baseline_conv=baseline_conv_pct / 100,
    )
    df_ab_zh = df_ab.copy()
    df_ab_zh["Variant"] = df_ab_zh["Variant"].replace({
        "A (baseline)": "A(基线)",
        "B (AI copy)": "B(AI 文案)",
        "Lift": "提升",
    })
    df_ab_zh.columns = ["版本", "曝光", "点击率", "点击数", "转化率", "订单"]
    st.dataframe(df_ab_zh, use_container_width=True, hide_index=True)
    st.caption(
        f"实测提升系数:点击率 ×{AB_UPLIFT['ctr']:.2f}、转化率 ×{AB_UPLIFT['conv']:.2f}"
        "(来源:此前 AI 文案 vs 模板文案的 A/B 实验)。"
    )

    if st.session_state.get("selected_text"):
        st.divider()
        st.markdown("**最终文案**(可直接发布到产品页):")
        st.code(st.session_state["selected_text"], language=None)


# --- Tab 3:客户洞察(M3) ---------------------------------------------------

with tab_insight:
    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown("**咨询数据样本**")
        st.caption(f"数据池:{len(inquiries):,} 条客户咨询")
        sample_size = st.select_slider(
            "样本数量",
            options=[100, 200, 500, 1000, len(inquiries)],
            value=200,
            format_func=lambda n: f"全部({n:,})" if n == len(inquiries) else f"{n:,}",
        )
        lang_choice = st.radio(
            "语言筛选",
            options=["both", "en", "zh"],
            horizontal=True,
            format_func=lambda x: {"both": "全部", "en": "英文", "zh": "中文"}[x],
        )
        st.markdown("**主题分类**")
        st.code("\n".join(THEME_ZH[t] for t in THEMES), language=None)

        run_clicked = st.button("分类客户咨询", type="primary", use_container_width=True)

    with right:
        if run_clicked or "m3_result" not in st.session_state:
            with st.spinner("分类中..."):
                lang_filter = None if lang_choice == "both" else lang_choice
                st.session_state["m3_result"] = classify_batch(
                    inquiries, sample_size=int(sample_size), lang_filter=lang_filter,
                )

        result = st.session_state["m3_result"]
        dist = theme_distribution(result)
        acc = classifier_accuracy(result)

        st.markdown("**关键洞察**")
        for ins in key_insights_zh(result):
            st.markdown(f"- {ins}")

        st.markdown("**主题分布**")
        kpi1, kpi2 = st.columns([1, 1])
        kpi1.metric("已分类", f"{len(result):,} 条")
        if acc is not None:
            kpi2.metric("分类准确率(对比 Ground Truth)", f"{acc*100:.1f}%")

        chart_col, table_col = st.columns([2, 1], gap="medium")
        with chart_col:
            chart_df = dist.copy()
            chart_df["theme"] = chart_df["theme"].map(THEME_ZH)
            st.bar_chart(chart_df.set_index("theme")[["count"]], height=220)
        with table_col:
            tbl = dist.copy()
            tbl["theme"] = tbl["theme"].map(THEME_ZH)
            tbl.columns = ["主题", "数量", "占比 %"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.markdown("**行动方案**")
        action_df = dist.copy()
        action_df["action"] = action_df["theme"].map(ACTIONS_ZH)
        action_df["theme"] = action_df["theme"].map(THEME_ZH)
        action_df.columns = ["主题", "数量", "占比 %", "建议行动"]
        st.dataframe(action_df, use_container_width=True, hide_index=True)

        st.markdown("**预期收益**(执行上述行动方案后)")
        i1, i2 = st.columns(2)
        i1.metric(
            "高匹配咨询占比",
            f"{EXPECTED_IMPACT['high_match_after_pct']:.0f}%",
            delta=f"+{EXPECTED_IMPACT['high_match_after_pct'] - EXPECTED_IMPACT['high_match_before_pct']:.0f}pp",
        )
        i2.metric(
            "每周咨询处理时长",
            f"{EXPECTED_IMPACT['triage_minutes_after']} 分钟",
            delta=f"-{EXPECTED_IMPACT['triage_hours_before']} 小时",
            delta_color="inverse",
        )

        with st.expander("证据 — 各主题代表性原文(点击展开)", expanded=False):
            for theme in dist["theme"]:
                st.markdown(f"**{THEME_ZH[theme]}** — {ACTIONS_ZH[theme]}")
                for q in top_quotes(result, theme, n=4):
                    st.markdown(f"- {q}")
                st.markdown("")


# --- Tab 4:运营监控(M5) ---------------------------------------------------

with tab_monitor:
    ts = timeseries

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.markdown("**监控配置**")
        st.caption(
            f"数据池:{ts['product_id'].nunique()} 个产品 · "
            f"{ts['date'].min()} → {ts['date'].max()}"
        )
        metric = st.selectbox(
            "异常检测指标",
            options=METRICS,
            format_func=lambda m: METRIC_LABELS_ZH[m],
        )
        baseline_days = st.slider("基线窗口(天)", 7, 21, 14)
        z_thresh = st.slider("告警阈值(z-score)", 1.5, 4.0, 2.0, step=0.1)
        run_check = st.button("执行每日巡检", type="primary", use_container_width=True)

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
                f"检测到 {n_critical} 个严重下跌 + {n_warning} 个警告 — 请立即排查。"
            )
        elif n_warning > 0:
            st.warning(f"检测到 {n_warning} 个警告。")
        else:
            st.success("所有指标均在正常范围内。")

        st.markdown("**整体 KPI**(近 7 天 vs 之前 7 天)")
        c1, c2, c3, c4 = st.columns(4)
        for col, m in zip([c1, c2, c3], ["ctr", "inquiry_rate", "conv"]):
            r = kpis["recent"][m] * 100
            p = kpis["prior"][m] * 100
            col.metric(METRIC_LABELS_ZH[m], f"{r:.2f}%", f"{r - p:+.2f}pp")
        c4.metric(
            "订单数", f"{kpis['recent']['orders']:,}",
            f"{kpis['recent']['orders'] - kpis['prior']['orders']:+,}",
        )

        st.markdown(f"**趋势** — 全平台每日{METRIC_LABELS_ZH[sel_metric]}")
        ts_daily = compute_daily_metrics(ts).groupby("date").apply(
            lambda d: pd.Series({
                "ctr": d["clicks"].sum() / max(d["impressions"].sum(), 1),
                "inquiry_rate": d["inquiries"].sum() / max(d["clicks"].sum(), 1),
                "conv": d["orders"].sum() / max(d["inquiries"].sum(), 1),
            }),
            include_groups=False,
        ).reset_index()
        st.line_chart(ts_daily.set_index("date")[[sel_metric]], height=220)

        st.markdown("**告警产品**(今日 vs 基线)")
        if flagged.empty:
            st.info("当前阈值下没有产品被标记。")
        else:
            display = flagged.copy()
            display["today"] = (display["today"] * 100).round(2).astype(str) + "%"
            display["baseline"] = (display["baseline"] * 100).round(2).astype(str) + "%"
            display["delta_pct"] = (display["delta_pct"] * 100).round(0).astype(int).astype(str) + "%"
            display["z_score"] = display["z_score"].round(2).astype(str)
            display["metric"] = display["metric"].map(METRIC_LABELS_ZH)
            display["severity"] = display["severity"].map(SEVERITY_ZH)
            display = display[["product_id", "metric", "today", "baseline", "delta_pct", "z_score", "severity"]]
            display.columns = ["产品 ID", "指标", "今日", "基线", "相对基线", "z-score", "等级"]
            st.dataframe(display, use_container_width=True, hide_index=True)

        with st.expander("自动生成的运营日报(Markdown — 可直接粘贴到 Slack / 邮件)"):
            report = daily_report_zh(anomalies, kpis, metric=sel_metric)
            st.code(report, language="markdown")

        st.divider()
        st.markdown("**效率收益**")
        e1, e2 = st.columns(2)
        e1.metric("人工巡检工作量", "↓ 65%", delta="-2 小时/天", delta_color="inverse")
        e2.metric("决策效率", "↑ 70%", delta="异常秒级浮现")
