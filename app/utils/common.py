from __future__ import annotations
from typing import Any
from app.config import POINTS_REASON_FILTER, POINTS_REASON_REFUND

def normalize_style_strength(value: Any) -> float:
    if value is None or value == "":
        return 0.7
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return 0.7
    if strength > 1.0:
        if strength <= 100.0:
            strength = strength / 100.0
        else:
            strength = 1.0
    return max(0.0, min(1.0, strength))

def describe_style_strength(strength: float) -> str:
    if strength <= 0.35:
        return "subtle"
    if strength <= 0.7:
        return "balanced"
    return "pronounced"

def require_dependencies(require_db: bool = True) -> None:
    missing = []
    try:
        import requests  # noqa: F401
    except Exception:
        missing.append("requests")
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        missing.append("Pillow")
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        import yaml  # noqa: F401
    except Exception:
        missing.append("PyYAML")
    try:
        import flask  # noqa: F401
    except Exception:
        missing.append("Flask")
    if require_db:
        try:
            import aiomysql  # noqa: F401
        except Exception:
            missing.append("aiomysql")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"缺少依赖: {joined}。请先安装后再运行。")

def is_safe_run_id(value: str) -> bool:
    if not value:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return all(char in allowed for char in value)

def format_points_note(note: str | None, reason: str) -> str:
    if not note:
        return "-"
    if reason not in {POINTS_REASON_FILTER, POINTS_REASON_REFUND}:
        return note
    styles = None
    run_id = None
    for token in note.split():
        if token.startswith("styles="):
            styles = token.split("=", 1)[1]
        elif token.startswith("run_id="):
            run_id = token.split("=", 1)[1]
    if not styles and not run_id:
        return note
    if styles and run_id:
        return f"风格={styles} 任务={run_id}"
    if styles:
        return f"风格={styles}"
    return f"任务={run_id}"
