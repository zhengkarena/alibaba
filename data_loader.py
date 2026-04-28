"""Data source layer.

Lets the app run on either bundled demo CSVs (default) or uploaded user
CSVs (per file, independently). Modules never call this directly — the
Streamlit app holds the single read site. Loaders return DataFrames in
the *internal* column schema modules already expect, normalizing from
the user-facing canonical schema documented in README.md.

Future connectors (Google Sheets, Shopify, Alibaba export, CRM) plug
into load_from_source() — currently a stub that raises NotImplementedError.
"""
from __future__ import annotations

from pathlib import Path
from typing import IO, Optional

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Canonical (user-facing) schemas — what an uploaded CSV must contain.
REQUIRED_PRODUCTS = [
    "product_id", "product_name", "category", "price",
    "impressions", "clicks", "inquiries", "orders",
]
REQUIRED_INQUIRIES = ["inquiry_id", "text", "language"]
REQUIRED_TIMESERIES = [
    "date", "product_id", "impressions", "clicks", "inquiries", "orders",
]

OPTIONAL_INQUIRIES = ["product_id", "true_theme"]


class SchemaError(Exception):
    """Raised when an uploaded CSV is missing required columns."""
    def __init__(self, file_kind: str, missing: list[str]):
        self.file_kind = file_kind
        self.missing = missing
        super().__init__(
            f"{file_kind}: missing required columns: {', '.join(missing)}"
        )


def validate_schema(df: pd.DataFrame, required: list[str]) -> tuple[bool, list[str]]:
    """Return (ok, missing_columns)."""
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


def _read_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or a file-like (Streamlit UploadedFile)."""
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source)


def _normalize_products_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Map canonical upload schema to internal schema used by M1/M2."""
    df = df.copy()
    df["name_en"] = df["product_name"].astype(str)
    df["name_zh"] = df["product_name"].astype(str)
    df["category_en"] = df["category"].astype(str)
    df["category_zh"] = df["category"].astype(str)
    return df


def _normalize_inquiries_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Map canonical upload schema to internal schema used by M3."""
    df = df.copy()
    df = df.rename(columns={"language": "lang"})
    return df


def load_products(uploaded: Optional[IO] = None) -> pd.DataFrame:
    if uploaded is None:
        return _read_csv(DATA_DIR / "products.csv")
    df = _read_csv(uploaded)
    ok, missing = validate_schema(df, REQUIRED_PRODUCTS)
    if not ok:
        raise SchemaError("products.csv", missing)
    return _normalize_products_upload(df)


def load_inquiries(uploaded: Optional[IO] = None) -> pd.DataFrame:
    if uploaded is None:
        return _read_csv(DATA_DIR / "inquiries.csv")
    df = _read_csv(uploaded)
    ok, missing = validate_schema(df, REQUIRED_INQUIRIES)
    if not ok:
        raise SchemaError("inquiries.csv", missing)
    return _normalize_inquiries_upload(df)


def load_timeseries(uploaded: Optional[IO] = None) -> pd.DataFrame:
    if uploaded is None:
        return _read_csv(DATA_DIR / "products_timeseries.csv")
    df = _read_csv(uploaded)
    ok, missing = validate_schema(df, REQUIRED_TIMESERIES)
    if not ok:
        raise SchemaError("products_timeseries.csv", missing)
    return df


# --- Future connectors --------------------------------------------------------


def load_from_source(kind: str, source: str) -> pd.DataFrame:
    """Future external connectors.

    kind:    one of {'products', 'inquiries', 'timeseries'}
    source:  scheme-prefixed string, e.g. 'gsheets:<url>',
             'shopify:<token>', 'alibaba:<export_id>', 'crm:<endpoint>'.

    Currently only CSV upload via the Streamlit UI is wired up. New
    connectors slot into this function so call sites never change.
    """
    raise NotImplementedError(
        f"Connector for {kind!r} via {source!r} not implemented yet. "
        "Roadmap: gsheets, shopify, alibaba, crm. Use CSV upload for now."
    )
