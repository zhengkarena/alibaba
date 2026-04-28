"""Module 1: Product Selection Optimizer.

Transparent weighted-percentile ranking. No ML, no LLM -- pure pandas so
the math is auditable in an interview.

Score formula (per product, range 0-1):
    score = 0.30 * CTR_percentile_in_category
          + 0.25 * inquiry_rate_percentile_in_category
          + 0.30 * conversion_rate_percentile_in_category
          + 0.15 * impressions_percentile_global

Run as a demo:
    python3 -m modules.m1_product_rank
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

WEIGHTS = {"ctr": 0.30, "inquiry": 0.25, "conversion": 0.30, "volume": 0.15}
MIN_IMPRESSIONS = 1000

DATA_PATH = Path(__file__).parent.parent / "data" / "products.csv"


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ctr"] = df["clicks"] / df["impressions"].clip(lower=1)
    df["inquiry_rate"] = df["inquiries"] / df["clicks"].clip(lower=1)
    df["conversion_rate"] = df["orders"] / df["inquiries"].clip(lower=1)
    return df


def rank_products(
    df: pd.DataFrame,
    weights: dict = WEIGHTS,
    min_impressions: int = MIN_IMPRESSIONS,
) -> pd.DataFrame:
    df = compute_metrics(df)
    df = df[df["impressions"] >= min_impressions].copy()

    # Within-category percentiles (fair across Electronics vs Apparel etc.)
    for col, key in [("ctr", "ctr"), ("inquiry_rate", "inquiry"), ("conversion_rate", "conversion")]:
        df[f"{key}_pct"] = df.groupby("category_en")[col].rank(pct=True)
    # Global volume percentile -- rewards products with proven traffic
    df["volume_pct"] = df["impressions"].rank(pct=True)

    df["score"] = (
        weights["ctr"] * df["ctr_pct"]
        + weights["inquiry"] * df["inquiry_pct"]
        + weights["conversion"] * df["conversion_pct"]
        + weights["volume"] * df["volume_pct"]
    )
    df["score"] = df["score"].round(3)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def explain_row(row: pd.Series) -> str:
    parts = []
    if row["ctr_pct"] >= 0.8:
        parts.append(f"top-{int((1 - row['ctr_pct']) * 100) + 1}% CTR")
    if row["conversion_pct"] >= 0.8:
        parts.append(f"top-{int((1 - row['conversion_pct']) * 100) + 1}% conversion")
    if row["inquiry_pct"] >= 0.8:
        parts.append(f"strong B2B intent (top-{int((1 - row['inquiry_pct']) * 100) + 1}% inquiry rate)")
    if row["volume_pct"] >= 0.7:
        parts.append("proven traffic")
    if not parts:
        parts.append("balanced metrics across the board")
    return f"{row['category_en']}: " + ", ".join(parts) + "."


def top_n(ranked: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    out = ranked.head(n).copy()
    out["why"] = out.apply(explain_row, axis=1)
    return out


def decision_time_savings(n_products: int, manual_sec: int = 14, ranked_sec: int = 30, top_pct: float = 0.05) -> dict:
    manual_min = (n_products * manual_sec) / 60
    reviewed = max(int(n_products * top_pct), 5)
    ranked_min = (reviewed * ranked_sec) / 60
    return {
        "manual_minutes": round(manual_min, 1),
        "ranked_minutes": round(ranked_min, 1),
        "reviewed_count": reviewed,
        "reduction_pct": round((1 - ranked_min / manual_min) * 100, 1),
    }


def _demo() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} products from {DATA_PATH.name}\n")

    ranked = rank_products(df)
    print(f"Ranked {len(ranked)} products (after filter impressions >= {MIN_IMPRESSIONS})\n")

    cols = ["product_id", "category_en", "name_en", "price", "ctr", "conversion_rate", "score"]
    fmt = ranked[cols].head(20).copy()
    fmt["ctr"] = (fmt["ctr"] * 100).round(2).astype(str) + "%"
    fmt["conversion_rate"] = (fmt["conversion_rate"] * 100).round(2).astype(str) + "%"
    fmt["price"] = fmt["price"].apply(lambda x: f"${x:.2f}")
    print("=== TOP 20 RANKED ===")
    print(fmt.to_string(index=False))

    print("\n=== TOP 5 RECOMMENDED (with reasons) ===")
    top5 = top_n(ranked, 5)
    for _, r in top5.iterrows():
        print(f"  [{r['score']:.3f}] {r['product_id']} {r['name_en']:<35} -> {r['why']}")

    print("\n=== DECISION TIME SAVINGS ===")
    s = decision_time_savings(len(df))
    print(f"  Manual review of all {len(df)} products: ~{s['manual_minutes']} min")
    print(f"  Ranked top-5% review ({s['reviewed_count']} products): ~{s['ranked_minutes']} min")
    print(f"  Reduction: {s['reduction_pct']}%   (matches the '2h -> 20min' interview talking point)")


if __name__ == "__main__":
    _demo()
