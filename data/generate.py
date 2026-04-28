"""Generate synthetic e-commerce datasets for the demo.

Run once:  python -m data.generate
Outputs:   data/products.csv, data/inquiries.csv, data/products_timeseries.csv
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PRODUCTS = 500
N_INQUIRIES = 8000
TIMESERIES_DAYS = 30
TIMESERIES_PRODUCTS = 100

OUT_DIR = Path(__file__).parent

CATEGORIES = [
    ("Electronics", "电子产品"),
    ("Apparel", "服装"),
    ("Home", "家居"),
    ("Beauty", "美妆"),
    ("Toys", "玩具"),
    ("Sports", "运动"),
]

PRODUCT_WORDS_EN = {
    "Electronics": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Power Bank", "USB-C Hub"],
    "Apparel": ["Cotton T-Shirt", "Denim Jacket", "Running Shorts", "Wool Sweater", "Yoga Pants"],
    "Home": ["LED Desk Lamp", "Memory Foam Pillow", "Ceramic Mug Set", "Storage Basket", "Bath Towel"],
    "Beauty": ["Vitamin C Serum", "Matte Lipstick", "Hair Dryer", "Face Mask Pack", "Nail Polish Kit"],
    "Toys": ["Building Blocks", "Plush Bear", "RC Car", "Puzzle 1000pc", "Educational Robot"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Resistance Bands", "Cycling Helmet", "Football"],
}
PRODUCT_WORDS_ZH = {
    "Electronics": ["无线耳机", "智能手表", "蓝牙音箱", "充电宝", "USB-C 扩展坞"],
    "Apparel": ["纯棉T恤", "牛仔外套", "跑步短裤", "羊毛毛衣", "瑜伽裤"],
    "Home": ["LED 台灯", "记忆棉枕头", "陶瓷杯套装", "收纳篮", "浴巾"],
    "Beauty": ["维C精华液", "哑光口红", "电吹风", "面膜组合", "指甲油套装"],
    "Toys": ["积木", "毛绒熊", "遥控车", "1000片拼图", "益智机器人"],
    "Sports": ["瑜伽垫", "哑铃套装", "弹力带", "骑行头盔", "足球"],
}


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def generate_products() -> pd.DataFrame:
    rng = _rng()
    rows = []
    for i in range(N_PRODUCTS):
        cat_en, cat_zh = CATEGORIES[i % len(CATEGORIES)]
        base_en = rng.choice(PRODUCT_WORDS_EN[cat_en])
        idx_zh = PRODUCT_WORDS_EN[cat_en].index(base_en)
        base_zh = PRODUCT_WORDS_ZH[cat_en][idx_zh]
        price = round(float(np.exp(rng.normal(3.2, 0.8))), 2)  # ~$10-$200, log-normal
        impressions = int(np.exp(rng.normal(9.5, 1.0)))         # ~hundreds to tens of thousands
        ctr = float(np.clip(rng.lognormal(mean=np.log(0.018), sigma=0.5), 0.001, 0.10))
        clicks = int(impressions * ctr)
        inquiry_rate = float(np.clip(rng.beta(2, 18), 0.01, 0.30))  # ~5-15%
        inquiries = int(clicks * inquiry_rate)
        conv = float(np.clip(rng.beta(2, 12), 0.02, 0.40))           # ~10-25% of inquiries
        orders = int(inquiries * conv)
        rows.append({
            "product_id": f"P{i:04d}",
            "category_en": cat_en,
            "category_zh": cat_zh,
            "name_en": f"{base_en} Model {i:03d}",
            "name_zh": f"{base_zh} 型号{i:03d}",
            "price": price,
            "impressions": impressions,
            "clicks": clicks,
            "inquiries": inquiries,
            "orders": orders,
        })
    return pd.DataFrame(rows)


INQUIRY_TEMPLATES = {
    "price": {
        "en": [
            "Can you offer a better price for bulk orders?",
            "What is the minimum order quantity to get a discount?",
            "Is the price negotiable for 500 units?",
            "Your competitor offers $X less, can you match it?",
            "Do you have any promotional pricing this month?",
            "The price seems high compared to similar products.",
            "Any chance of a 10% discount for first-time buyers?",
        ],
        "zh": [
            "大批量采购可以给个更好的价格吗?",
            "最低起订量多少才能拿到折扣?",
            "500件的话价格能再优惠吗?",
            "你们竞争对手报价更低,能否匹配?",
            "这个月有促销价吗?",
            "和同类产品相比价格偏高。",
            "首次合作能给9折吗?",
        ],
    },
    "delivery": {
        "en": [
            "How long is the shipping time to Germany?",
            "Can you guarantee delivery before December 15?",
            "Do you support DDP shipping to the US?",
            "What's the lead time for 1000 units?",
            "Is air freight available for urgent orders?",
            "Will the order arrive before Chinese New Year?",
        ],
        "zh": [
            "发到德国需要多久?",
            "能保证12月15号前到货吗?",
            "支持DDP发到美国吗?",
            "1000件的生产周期多久?",
            "急单能走空运吗?",
            "春节前能到货吗?",
        ],
    },
    "feature": {
        "en": [
            "Does this product support waterproof IPX7?",
            "What is the battery capacity?",
            "Is the material BPA-free?",
            "Can you share the technical spec sheet?",
            "Does it come with a CE certification?",
            "What's the warranty period?",
        ],
        "zh": [
            "这个产品支持IPX7防水吗?",
            "电池容量多少毫安?",
            "材料是否不含BPA?",
            "能发一下技术参数表吗?",
            "有CE认证吗?",
            "保修期多久?",
        ],
    },
    "customization": {
        "en": [
            "Can we add our logo on the packaging?",
            "Do you support OEM/ODM service?",
            "Can the color be customized?",
            "Is private label packaging available?",
            "Can you change the box design to our brand?",
            "We need custom retail packaging, is that possible?",
        ],
        "zh": [
            "包装上能加我们的logo吗?",
            "支持OEM/ODM吗?",
            "颜色可以定制吗?",
            "可以做自有品牌包装吗?",
            "包装盒能换成我们的品牌设计吗?",
            "需要定制零售包装,可以做吗?",
        ],
    },
}


def generate_inquiries(products: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    themes = list(INQUIRY_TEMPLATES.keys())
    theme_weights = [0.35, 0.20, 0.30, 0.15]  # price-heavy, like real life
    rows = []
    for i in range(N_INQUIRIES):
        theme = rng.choice(themes, p=theme_weights)
        lang = "en" if rng.random() < 0.5 else "zh"
        text = rng.choice(INQUIRY_TEMPLATES[theme][lang])
        product_id = rng.choice(products["product_id"].values)
        rows.append({
            "inquiry_id": f"Q{i:05d}",
            "product_id": product_id,
            "lang": lang,
            "text": text,
            "true_theme": theme,
        })
    return pd.DataFrame(rows)


def generate_timeseries(products: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    sample = products.sample(TIMESERIES_PRODUCTS, random_state=SEED).reset_index(drop=True)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=TIMESERIES_DAYS)
    rows = []
    for _, p in sample.iterrows():
        base_imps = max(p["impressions"] // TIMESERIES_DAYS, 50)
        base_ctr = p["clicks"] / max(p["impressions"], 1)
        base_inq_rate = p["inquiries"] / max(p["clicks"], 1)
        base_conv = p["orders"] / max(p["inquiries"], 1)
        for d_idx, date in enumerate(dates):
            imps = int(base_imps * rng.normal(1.0, 0.15))
            imps = max(imps, 10)
            ctr = float(np.clip(base_ctr * rng.normal(1.0, 0.20), 0.001, 0.15))
            clicks = int(imps * ctr)
            inquiries = int(clicks * float(np.clip(base_inq_rate * rng.normal(1.0, 0.25), 0.01, 0.5)))
            orders = int(inquiries * float(np.clip(base_conv * rng.normal(1.0, 0.25), 0.01, 0.6)))
            rows.append({
                "date": date.date().isoformat(),
                "product_id": p["product_id"],
                "impressions": imps,
                "clicks": clicks,
                "inquiries": inquiries,
                "orders": orders,
            })
    df = pd.DataFrame(rows)

    # Inject one obvious anomaly: pick a product, drop CTR to ~10% of normal on the last day.
    anomaly_pid = sample.iloc[0]["product_id"]
    last_date = dates[-1].date().isoformat()
    mask = (df["product_id"] == anomaly_pid) & (df["date"] == last_date)
    df.loc[mask, "clicks"] = (df.loc[mask, "clicks"] * 0.1).astype(int)
    df.loc[mask, "inquiries"] = (df.loc[mask, "inquiries"] * 0.1).astype(int)
    df.loc[mask, "orders"] = 0
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    products = generate_products()
    inquiries = generate_inquiries(products)
    timeseries = generate_timeseries(products)

    products.to_csv(OUT_DIR / "products.csv", index=False)
    inquiries.to_csv(OUT_DIR / "inquiries.csv", index=False)
    timeseries.to_csv(OUT_DIR / "products_timeseries.csv", index=False)

    print(f"products.csv           rows={len(products):>6}")
    print(f"inquiries.csv          rows={len(inquiries):>6}")
    print(f"products_timeseries.csv rows={len(timeseries):>6}")


if __name__ == "__main__":
    main()
