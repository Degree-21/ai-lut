from __future__ import annotations
from functools import wraps
from flask import session, request, jsonify, redirect, url_for

def clean_next_url(value: str | None) -> str | None:
    if not value:
        return None
    if not value.startswith("/") or value.startswith("//"):
        return None
    return value

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("user_id"):
            return view(*args, **kwargs)
        if request.path.startswith("/api/"):
            return jsonify({"error": "未登录，请先登录。"}), 401
        next_url = request.full_path
        if next_url.endswith("?"):
            next_url = request.path
        return redirect(url_for("auth.login", next=next_url))

    return wrapped

def get_session_user_id() -> int | None:
    value = session.get("user_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
