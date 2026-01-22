from __future__ import annotations
from werkzeug.security import generate_password_hash
from app.config import MIN_USERNAME_LENGTH, MAX_USERNAME_LENGTH, MIN_PASSWORD_LENGTH
from app.models.db import count_users, fetch_user_by_username, create_user

def registration_open(database_url: str, allow_register: bool) -> bool:
    if allow_register:
        return True
    try:
        return count_users(database_url) == 0
    except Exception:
        return False

def validate_registration(username: str, password: str, confirm: str) -> str | None:
    if not username:
        return "请输入用户名。"
    if len(username) < MIN_USERNAME_LENGTH or len(username) > MAX_USERNAME_LENGTH:
        return f"用户名长度需在 {MIN_USERNAME_LENGTH}-{MAX_USERNAME_LENGTH} 个字符。"
    if not password:
        return "请输入密码。"
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"密码至少 {MIN_PASSWORD_LENGTH} 位。"
    if password != confirm:
        return "两次输入的密码不一致。"
    return None

def ensure_admin_user(database_url: str, username: str, password: str) -> None:
    if not username or not password:
        return
    if len(password) < MIN_PASSWORD_LENGTH:
        raise RuntimeError(f"管理员密码至少 {MIN_PASSWORD_LENGTH} 位。")
    existing = fetch_user_by_username(database_url, username)
    if existing:
        return
    create_user(database_url, username, generate_password_hash(password))

def is_admin_user(settings: dict, username: str | None) -> bool:
    admin_username = str(settings.get("admin_username", "")).strip()
    if not admin_username or not username:
        return False
    return username == admin_username
