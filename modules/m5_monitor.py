"""Module 5: Automation & Monitoring.

Detects per-product metric anomalies (CTR / inquiry rate / conversion)
on the latest day vs a rolling baseline, aggregates fleet-level KPIs
for last-7d vs prior-7d, and produces a markdown daily/weekly report
that ops can paste straight into Slack or email.

Run as a demo:
    python3 -m modules.m5_monitor
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "products_timeseries.csv"

METRICS = ["ctr", "inquiry_rate", "conv"]
METRIC_LABELS = {"ctr": "CTR", "inquiry_rate": "Inquiry rate", "conv": "Conversion"}


def compute_daily_metrics(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()
    df["ctr"] = df["clicks"] / df["impressions"].clip(lower=1)
    df["inquiry_rate"] = df["inquiries"] / df["clicks"].clip(lower=1)
    df["conv"] = df["orders"] / df["inquiries"].clip(lower=1)
    return df


def detect_anomalies(
    ts: pd.DataFrame,
    *,
    metric: str = "ctr",
    baseline_days: int = 14,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Flag products whose latest day is z-threshold below baseline."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    df = compute_daily_metrics(ts).sort_values(["product_id", "date"])

    rows = []
    for pid, g in df.groupby("product_id"):
        if len(g) < baseline_days + 1:
            continue
        latest = g.iloc[-1]
        history = g.iloc[-(baseline_days + 1):-1]
        baseline = float(history[metric].mean())
        std = float(history[metric].std()) or 1e-9
        z = (float(latest[metric]) - baseline) / std
        delta_pct = ((latest[metric] - baseline) / baseline) if baseline > 0 else 0.0
        if z < -3:
            severity = "critical"
        elif z < -z_threshold:
            severity = "warning"
        else:
            severity = "ok"
        rows.append({
            "product_id": pid,
            "date": latest["date"],
            "metric": metric,
            "today": float(latest[metric]),
            "baseline": baseline,
            "delta_pct": float(delta_pct),
            "z_score": float(z),
            "severity": severity,
        })
    return pd.DataFrame(rows).sort_values("z_score").reset_index(drop=True)


def aggregate_kpis(ts: pd.DataFrame, window_days: int = 7) -> dict:
    df = compute_daily_metrics(ts)
    df["date_dt"] = pd.to_datetime(df["date"])
    latest = df["date_dt"].max()
    cutoff = latest - pd.Timedelta(days=window_days - 1)
    prev_start = cutoff - pd.Timedelta(days=window_days)

    recent = df[df["date_dt"] >= cutoff]
    prior = df[(df["date_dt"] >= prev_start) & (df["date_dt"] < cutoff)]

    def _agg(d: pd.DataFrame) -> dict:
        if d.empty:
            return {"ctr": 0.0, "inquiry_rate": 0.0, "conv": 0.0, "orders": 0}
        imps = max(int(d["impressions"].sum()), 1)
        clk = max(int(d["clicks"].sum()), 1)
        inq = max(int(d["inquiries"].sum()), 1)
        return {
            "ctr": d["clicks"].sum() / imps,
            "inquiry_rate": d["inquiries"].sum() / clk,
            "conv": d["orders"].sum() / inq,
            "orders": int(d["orders"].sum()),
        }

    return {
        "recent": _agg(recent),
        "prior": _agg(prior),
        "window_days": window_days,
        "latest_date": latest.strftime("%Y-%m-%d"),
    }


def daily_report_markdown(
    anomalies: pd.DataFrame,
    kpis: dict,
    metric: str,
    top_n_flagged: int = 5,
) -> str:
    flagged = anomalies[anomalies["severity"] != "ok"]
    n_critical = int((anomalies["severity"] == "critical").sum())
    n_warning = int((anomalies["severity"] == "warning").sum())

    lines = [
        f"# Operations Report — {kpis['latest_date']}",
        "",
        f"**Window:** last {kpis['window_days']} days vs prior {kpis['window_days']} days  ·  "
        f"**Anomaly metric:** {METRIC_LABELS[metric]}",
        "",
        "## KPIs",
    ]
    for m in ["ctr", "inquiry_rate", "conv"]:
        r = kpis["recent"][m] * 100
        p = kpis["prior"][m] * 100
        delta = r - p
        arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        lines.append(f"- **{METRIC_LABELS[m]}**: {r:.2f}% {arrow} ({delta:+.2f}pp vs prior)")
    lines.append(
        f"- **Orders (last {kpis['window_days']}d)**: {kpis['recent']['orders']:,} "
        f"vs prior {kpis['prior']['orders']:,}"
    )

    lines += ["", "## Alerts", f"- {n_critical} critical drops", f"- {n_warning} warnings"]

    if len(flagged) > 0:
        lines += ["", "### Top flagged products"]
        for _, r in flagged.head(top_n_flagged).iterrows():
            lines.append(
                f"- `{r['product_id']}` — {METRIC_LABELS[r['metric']]} "
                f"{r['today']*100:.2f}% (baseline {r['baseline']*100:.2f}%, "
                f"{r['delta_pct']*100:+.0f}%, z={r['z_score']:.2f}) **{r['severity']}**"
            )

    return "\n".join(lines)


def _demo() -> None:
    ts = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(ts)} rows from {DATA_PATH.name}, "
          f"{ts['product_id'].nunique()} products, "
          f"dates {ts['date'].min()} to {ts['date'].max()}\n")

    kpis = aggregate_kpis(ts, window_days=7)
    print("=== FLEET KPIs (last 7d vs prior 7d) ===")
    for m in METRICS:
        r = kpis["recent"][m] * 100
        p = kpis["prior"][m] * 100
        print(f"  {METRIC_LABELS[m]:<14} recent {r:5.2f}%  prior {p:5.2f}%  delta {r-p:+.2f}pp")
    print(f"  Orders         recent {kpis['recent']['orders']:,}  prior {kpis['prior']['orders']:,}")

    print("\n=== ANOMALY DETECTION (metric=ctr, baseline=14d, z<-2) ===")
    anomalies = detect_anomalies(ts, metric="ctr", baseline_days=14, z_threshold=2.0)
    flagged = anomalies[anomalies["severity"] != "ok"]
    if flagged.empty:
        print("  (none)")
    else:
        for _, r in flagged.iterrows():
            print(f"  {r['severity']:<8} {r['product_id']}  today={r['today']*100:.2f}%  "
                  f"baseline={r['baseline']*100:.2f}%  delta={r['delta_pct']*100:+.0f}%  z={r['z_score']:.2f}")

    print("\n=== AUTO-GENERATED REPORT ===")
    print(daily_report_markdown(anomalies, kpis, metric="ctr"))


if __name__ == "__main__":
    _demo()
