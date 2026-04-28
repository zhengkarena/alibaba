"""Unified LLM wrapper with offline fallback.

Public API:
    is_live() -> bool                     -- whether OpenAI is reachable+configured
    generate(prompt, *, system=None, lang="en", fallback=None, temperature=0.7) -> str
    classify(text, categories, *, lang="en", fallback=None) -> str
    status() -> dict                      -- {"mode": "live"|"fallback", "model": ..., "reason": ...}

Design:
- Downstream modules never import openai directly. They call generate/classify
  and pass a domain-specific fallback (template string or keyword rule).
- If OPENAI_API_KEY is missing, calls fail, or quota is hit, the fallback is
  used. Status is sticky once a live call fails — we don't retry every call.
- Responses are cached in-memory by request hash so Streamlit reruns are free.

Demo:
    python3 llm_client.py
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


FallbackArg = Union[str, Callable[[], str], None]


@dataclass
class _State:
    mode: str = "unknown"   # "live" | "fallback" | "unknown"
    reason: str = ""
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


_state = _State()
_cache: dict[str, str] = {}


def _client():
    try:
        from openai import OpenAI
    except ImportError:
        _state.mode = "fallback"
        _state.reason = "openai package not installed"
        return None
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-your-key"):
        _state.mode = "fallback"
        _state.reason = "OPENAI_API_KEY not set"
        return None
    return OpenAI(api_key=key)


def _resolve_fallback(fallback: FallbackArg, default: str) -> str:
    if fallback is None:
        return default
    if callable(fallback):
        return fallback()
    return str(fallback)


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def status() -> dict:
    return {"mode": _state.mode, "model": _state.model, "reason": _state.reason}


def is_live() -> bool:
    if _state.mode == "unknown":
        # Probe lazily on first call; until then assume live if key looks set.
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        return bool(key) and not key.startswith("sk-your-key")
    return _state.mode == "live"


def generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    lang: str = "en",
    fallback: FallbackArg = None,
    temperature: float = 0.7,
    max_tokens: int = 400,
) -> str:
    cache_key = _hash("gen", _state.model, system or "", prompt, lang, str(temperature))
    if cache_key in _cache:
        return _cache[cache_key]

    client = _client()
    default_fb = f"[fallback:{lang}] {prompt[:80]}..."
    if client is None:
        out = _resolve_fallback(fallback, default_fb)
        _cache[cache_key] = out
        return out

    sys_msg = system or (
        "You are a concise e-commerce marketing assistant. "
        "Reply in Chinese." if lang == "zh" else
        "You are a concise e-commerce marketing assistant. Reply in English."
    )
    try:
        resp = client.chat.completions.create(
            model=_state.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        _state.mode = "live"
        _state.reason = "ok"
        _cache[cache_key] = text
        return text
    except Exception as e:
        _state.mode = "fallback"
        _state.reason = f"api error: {type(e).__name__}"
        out = _resolve_fallback(fallback, default_fb)
        _cache[cache_key] = out
        return out


def classify(
    text: str,
    categories: list[str],
    *,
    lang: str = "en",
    fallback: FallbackArg = None,
) -> str:
    cache_key = _hash("cls", _state.model, text, ",".join(categories), lang)
    if cache_key in _cache:
        return _cache[cache_key]

    client = _client()
    default_fb = categories[0]
    if client is None:
        out = _resolve_fallback(fallback, default_fb)
        if out not in categories:
            out = default_fb
        _cache[cache_key] = out
        return out

    cats = ", ".join(categories)
    sys_msg = (
        f"Classify the user message into exactly one of these categories: {cats}. "
        f"Respond with strict JSON: {{\"category\": \"<one of: {cats}>\"}}."
    )
    try:
        resp = client.chat.completions.create(
            model=_state.model,
            temperature=0.0,
            max_tokens=40,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        cat = json.loads(raw).get("category", "").strip()
        if cat not in categories:
            cat = default_fb
        _state.mode = "live"
        _state.reason = "ok"
        _cache[cache_key] = cat
        return cat
    except Exception as e:
        _state.mode = "fallback"
        _state.reason = f"api error: {type(e).__name__}"
        out = _resolve_fallback(fallback, default_fb)
        if out not in categories:
            out = default_fb
        _cache[cache_key] = out
        return out


def _demo():
    print("=" * 60)
    print("LLM CLIENT DEMO")
    print("=" * 60)

    # --- Mode 1: live (uses key from .env) ---
    print("\n[1] LIVE MODE (using .env key if present)")
    out_en = generate("Write a 1-line product tagline for wireless earbuds.", lang="en")
    print(f"  EN tagline: {out_en}")
    print(f"  status: {status()}")

    out_zh = generate("用一句话写无线耳机的卖点。", lang="zh")
    print(f"  ZH tagline: {out_zh}")

    cat = classify(
        "Can you offer a discount for 1000 units?",
        ["price", "delivery", "feature", "customization"],
    )
    print(f"  classify EN: {cat}")
    cat_zh = classify(
        "包装上能加我们的logo吗?",
        ["price", "delivery", "feature", "customization"],
        lang="zh",
    )
    print(f"  classify ZH: {cat_zh}")
    print(f"  status: {status()}")

    # --- Mode 2: forced fallback (simulate no key) ---
    print("\n[2] FALLBACK MODE (simulated by clearing key)")
    saved = os.environ.pop("OPENAI_API_KEY", None)
    _cache.clear()
    _state.mode = "unknown"
    out_fb = generate(
        "Write a 1-line product tagline for wireless earbuds.",
        fallback="Crystal-clear sound, all-day battery — your perfect commute companion.",
    )
    print(f"  EN fallback: {out_fb}")
    cat_fb = classify(
        "Can you ship to Germany before Dec 15?",
        ["price", "delivery", "feature", "customization"],
        fallback=lambda: "delivery",
    )
    print(f"  classify fallback: {cat_fb}")
    print(f"  status: {status()}")

    if saved:
        os.environ["OPENAI_API_KEY"] = saved


if __name__ == "__main__":
    _demo()
