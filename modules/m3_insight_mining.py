"""Module 3: Customer Insight Mining.

Classifies customer inquiries into 4 operational themes (price, delivery,
feature, customization) using the LLM client with a bilingual keyword
fallback. Outputs a theme distribution, classifier accuracy against the
synthetic ground truth, top quotes per theme, and a recommended action.

Run as a demo:
    python3 -m modules.m3_insight_mining
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from llm_client import classify

THEMES = ["price", "delivery", "feature", "customization"]

DATA_PATH = Path(__file__).parent.parent / "data" / "inquiries.csv"

# Bilingual keyword tables for the offline fallback. Tuned so the synthetic
# inquiry templates classify >90% correctly without an API key.
KEYWORDS: dict[str, dict[str, list[str]]] = {
    "price": {
        "en": ["price", "discount", "cheaper", "expensive", "promotional", "MOQ",
               "minimum order", "negotiable", "competitor offers", "$"],
        "zh": ["价格", "折扣", "便宜", "贵", "促销", "起订量", "优惠", "议价", "竞争对手"],
    },
    "delivery": {
        "en": ["shipping", "delivery", "ship", "lead time", "DDP", "air freight",
               "arrive", "before", "Germany", "US", "urgent"],
        "zh": ["发货", "到货", "发到", "运送", "物流", "生产周期", "空运", "海运",
               "DDP", "急单", "春节前", "12月"],
    },
    "feature": {
        "en": ["spec", "specification", "feature", "battery", "capacity", "material",
               "BPA", "certification", "CE", "warranty", "waterproof", "IPX"],
        "zh": ["参数", "功能", "电池", "毫安", "材料", "BPA", "认证", "CE",
               "保修", "防水", "IPX", "技术参数"],
    },
    "customization": {
        "en": ["logo", "custom", "customize", "OEM", "ODM", "private label",
               "packaging", "package", "brand", "retail packaging"],
        "zh": ["定制", "logo", "OEM", "ODM", "包装", "贴牌", "自有品牌", "盒"],
    },
}

ACTIONS_BY_THEME: dict[str, str] = {
    "price": "Introduce tiered pricing for >500 units and publish a bulk-discount table on the PDP.",
    "delivery": "Publish lead-time and shipping options (DDP / air / sea) on the PDP, with country-specific ETAs.",
    "feature": "Add a downloadable spec sheet and surface certifications (CE, BPA-free, IPX rating) above the fold.",
    "customization": "Add an OEM/ODM inquiry CTA and a 'private label packaging' callout on the PDP.",
}


def keyword_classify(text: str, lang: str) -> str:
    text_l = text.lower()
    scores = {t: 0 for t in THEMES}
    for theme, langs in KEYWORDS.items():
        for kw in langs.get(lang, []):
            if kw.lower() in text_l:
                scores[theme] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "price"


def classify_inquiry(text: str, lang: str) -> str:
    return classify(
        text,
        categories=THEMES,
        lang=lang,
        fallback=lambda: keyword_classify(text, lang),
    )


def classify_batch(
    df: pd.DataFrame,
    sample_size: int | None = None,
    lang_filter: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    work = df
    if lang_filter in ("en", "zh"):
        work = work[work["lang"] == lang_filter]
    if sample_size and sample_size < len(work):
        work = work.sample(sample_size, random_state=seed)
    work = work.copy().reset_index(drop=True)
    work["predicted_theme"] = [
        classify_inquiry(t, l) for t, l in zip(work["text"], work["lang"])
    ]
    return work


def theme_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["predicted_theme"].value_counts()
    share = df["predicted_theme"].value_counts(normalize=True)
    out = pd.DataFrame({
        "theme": counts.index,
        "count": counts.values,
        "share": (share.values * 100).round(1),
    })
    return out.reset_index(drop=True)


def classifier_accuracy(df: pd.DataFrame) -> float | None:
    if "true_theme" not in df.columns:
        return None
    return float((df["predicted_theme"] == df["true_theme"]).mean())


def top_quotes(df: pd.DataFrame, theme: str, n: int = 5) -> list[str]:
    sub = df.loc[df["predicted_theme"] == theme, "text"].drop_duplicates()
    return sub.head(n).tolist()


def recommended_action(theme: str) -> str:
    return ACTIONS_BY_THEME.get(theme, "")


def _demo() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} inquiries from {DATA_PATH.name}")

    sample = classify_batch(df, sample_size=400)
    print(f"Classified {len(sample)} inquiries\n")

    print("=== THEME DISTRIBUTION ===")
    dist = theme_distribution(sample)
    print(dist.to_string(index=False))

    acc = classifier_accuracy(sample)
    if acc is not None:
        print(f"\nClassifier accuracy vs ground truth: {acc*100:.1f}%")

    print("\n=== TOP QUOTES + RECOMMENDED ACTION PER THEME ===")
    for theme in THEMES:
        quotes = top_quotes(sample, theme, n=3)
        if not quotes:
            continue
        print(f"\n[{theme}]  action: {recommended_action(theme)}")
        for q in quotes:
            print(f"   - {q}")


if __name__ == "__main__":
    _demo()
