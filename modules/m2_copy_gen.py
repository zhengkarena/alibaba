"""Module 2: AI Marketing Copy Generator.

Generates three styles of marketing copy per product (price / scenario /
comparison) in English or Chinese, routed through llm_client so it works
both with and without an OpenAI key.

Run as a demo:
    python3 -m modules.m2_copy_gen
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from llm_client import generate, status

STYLES = ["price", "scenario", "comparison"]

DATA_PATH = Path(__file__).parent.parent / "data" / "products.csv"

# Default feature blurbs per category, used when caller doesn't supply features.
CATEGORY_FEATURES_EN = {
    "Electronics": "long battery life, fast charging, premium audio",
    "Apparel": "soft cotton blend, breathable fabric, multiple sizes",
    "Home": "durable build, modern design, easy to clean",
    "Beauty": "dermatologically tested, vegan-friendly, long-lasting",
    "Toys": "BPA-free materials, safety-tested, age 3+",
    "Sports": "non-slip grip, lightweight, eco-friendly",
}
CATEGORY_FEATURES_ZH = {
    "Electronics": "长续航、快充、高品质音质",
    "Apparel": "柔软纯棉、透气面料、多尺码可选",
    "Home": "坚固耐用、现代设计、易清洁",
    "Beauty": "皮肤科测试、纯素配方、持久使用",
    "Toys": "无BPA材料、通过安全检测、3岁以上适用",
    "Sports": "防滑握感、轻量化、环保材料",
}

# Measured uplift coefficients from the spec (CTR 1.0->1.4 = +40%, conv 2.3->3.8 = +65%).
# Applied to whatever baseline the user supplies in the tool.
AB_UPLIFT = {"ctr": 1.40, "conv": 1.65}

TARGET_MARKETS = ["USA", "Germany", "Japan", "Southeast Asia", "Brazil", "UAE"]


def _system_prompt(style: str, lang: str) -> str:
    style_instructions = {
        "price": {
            "en": "Write price-driven marketing copy. Lead with the price, emphasize value and bulk-order savings.",
            "zh": "写价格驱动型营销文案,以价格为核心,强调性价比和批量采购优惠。",
        },
        "scenario": {
            "en": "Write scenario-driven marketing copy. Paint a vivid use moment that triggers emotional purchase intent.",
            "zh": "写场景驱动型营销文案,描绘生动的使用场景,激发情感购买意愿。",
        },
        "comparison": {
            "en": "Write feature-comparison marketing copy. Highlight one concrete spec advantage over competing products.",
            "zh": "写功能对比型营销文案,突出一个相比竞品的具体规格优势。",
        },
    }[style][lang]

    constraints = {
        "en": " Output exactly one short paragraph, max 40 words, plain text, no emojis, no markdown.",
        "zh": " 输出一个短段落,最多40字,纯文本,不要表情符号,不要markdown。",
    }[lang]
    return style_instructions + constraints


def _user_prompt(product: dict, style: str, lang: str, features: str, target_market: str) -> str:
    if lang == "zh":
        return (
            f"产品名: {product['name_zh']}\n"
            f"品类: {product['category_zh']}\n"
            f"价格: ${product['price']:.2f}\n"
            f"主要卖点: {features}\n"
            f"目标市场: {target_market}\n\n"
            f"请针对该目标市场生成营销文案。"
        )
    return (
        f"Product: {product['name_en']}\n"
        f"Category: {product['category_en']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Key features: {features}\n"
        f"Target market: {target_market}\n\n"
        f"Write the marketing copy tailored to that target market."
    )


def _fallback(product: dict, style: str, lang: str, features: str, target_market: str) -> str:
    name_en = product["name_en"]
    name_zh = product["name_zh"]
    price = product["price"]
    cat_en = product["category_en"]
    templates = {
        ("price", "en"): (
            f"{name_en} at only ${price:.2f} — premium {cat_en.lower()} "
            f"with {features}. Save up to 20% on bulk orders for {target_market} buyers. Limited stock."
        ),
        ("price", "zh"): (
            f"{name_zh},面向{target_market}市场仅售 ${price:.2f}!"
            f"{features},批量采购最高省20%,库存有限,先到先得。"
        ),
        ("scenario", "en"): (
            f"From morning routines to weekend getaways, {name_en} fits every moment of your day. "
            f"Designed for {features.split(',')[0].strip()}, built for the way you live."
        ),
        ("scenario", "zh"): (
            f"无论是清晨日常还是周末出行,{name_zh}都能融入你的每一刻。"
            f"专为{features.split('、')[0]}设计,贴合你的生活方式。"
        ),
        ("comparison", "en"): (
            f"{name_en} outperforms competing products in {features.split(',')[0].strip()} "
            f"at ${price:.2f}. Get more value per dollar without compromising on quality."
        ),
        ("comparison", "zh"): (
            f"{name_zh}在{features.split('、')[0]}方面优于竞品,售价仅 ${price:.2f},"
            f"用同样的预算获得更高品质。"
        ),
    }
    return templates[(style, lang)]


def generate_copy(
    product: dict,
    *,
    style: str,
    lang: str = "en",
    features: str = "",
    target_market: str = "USA",
) -> str:
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if not features:
        features = (CATEGORY_FEATURES_ZH if lang == "zh" else CATEGORY_FEATURES_EN).get(
            product["category_en"], ""
        )
    sys = _system_prompt(style, lang)
    user = _user_prompt(product, style, lang, features, target_market)
    fb = _fallback(product, style, lang, features, target_market)
    return generate(user, system=sys, lang=lang, fallback=fb, temperature=0.7, max_tokens=120)


def generate_all(
    product: dict,
    lang: str = "en",
    features: str = "",
    target_market: str = "USA",
) -> dict:
    return {
        style: generate_copy(
            product, style=style, lang=lang, features=features, target_market=target_market,
        )
        for style in STYLES
    }


def ab_test_simulation(
    impressions: int = 100_000,
    baseline_ctr: float = 0.010,
    baseline_conv: float = 0.023,
    uplift: dict = AB_UPLIFT,
) -> pd.DataFrame:
    variant_ctr = min(baseline_ctr * uplift["ctr"], 0.50)
    variant_conv = min(baseline_conv * uplift["conv"], 0.80)

    def funnel(ctr: float, conv: float) -> dict:
        clicks = int(impressions * ctr)
        orders = int(clicks * conv)
        return {"impressions": impressions, "ctr": ctr, "clicks": clicks, "conv": conv, "orders": orders}

    b = funnel(baseline_ctr, baseline_conv)
    v = funnel(variant_ctr, variant_conv)
    order_lift = (v["orders"] / b["orders"] - 1) * 100 if b["orders"] else 0
    return pd.DataFrame([
        {
            "Variant": "A (baseline)",
            "Impressions": f"{b['impressions']:,}",
            "CTR": f"{b['ctr']*100:.2f}%",
            "Clicks": f"{b['clicks']:,}",
            "Conv": f"{b['conv']*100:.2f}%",
            "Orders": f"{b['orders']:,}",
        },
        {
            "Variant": "B (AI copy)",
            "Impressions": f"{v['impressions']:,}",
            "CTR": f"{v['ctr']*100:.2f}%",
            "Clicks": f"{v['clicks']:,}",
            "Conv": f"{v['conv']*100:.2f}%",
            "Orders": f"{v['orders']:,}",
        },
        {
            "Variant": "Lift",
            "Impressions": "—",
            "CTR": f"+{(v['ctr']/b['ctr']-1)*100:.0f}%",
            "Clicks": f"+{(v['clicks']/b['clicks']-1)*100:.0f}%",
            "Conv": f"+{(v['conv']/b['conv']-1)*100:.0f}%",
            "Orders": f"+{order_lift:.0f}%",
        },
    ])


def _demo() -> None:
    df = pd.read_csv(DATA_PATH)
    # Borrow the top-ranked product from M1's logic for narrative continuity.
    from modules.m1_product_rank import rank_products
    top = rank_products(df).iloc[0].to_dict()
    print(f"Product: {top['name_en']} / {top['name_zh']}  (${top['price']:.2f}, {top['category_en']})")
    print(f"LLM status: {status()}\n")

    for lang in ("en", "zh"):
        label = "ENGLISH" if lang == "en" else "CHINESE"
        print(f"=== {label} COPY VARIANTS ===")
        variants = generate_all(top, lang=lang)
        for style, text in variants.items():
            print(f"  [{style}] {text}")
        print()

    print("=== A/B TEST SIMULATION (100k impressions, baseline CTR 1.0%, conv 2.3%) ===")
    print(ab_test_simulation(100_000, 0.010, 0.023).to_string(index=False))


if __name__ == "__main__":
    _demo()
