from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
from werkzeug.security import check_password_hash, generate_password_hash
from app.models.db import (
    fetch_user_by_username,
    mark_last_login,
    create_user,
    UserExistsError,
    apply_points_change,
)
from app.services.settings import load_effective_settings
from app.services.user import registration_open, validate_registration
from app.utils.auth import clean_next_url
from app.config import POINTS_REASON_REGISTER, POINTS_SOURCE_REGISTER

bp = Blueprint("auth", __name__)

@bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = clean_next_url(request.args.get("next"))
    database_url = current_app.config["DATABASE_URL"]
    register_allowed = registration_open(database_url, current_app.config["ALLOW_REGISTER"])
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = clean_next_url(request.form.get("next")) or next_url
        if not username or not password:
            error = "请输入用户名和密码。"
        else:
            try:
                user = fetch_user_by_username(database_url, username)
            except Exception:
                error = "数据库连接失败，请稍后重试。"
            else:
                if not user or not check_password_hash(user["password_hash"], password):
                    error = "用户名或密码错误。"
                else:
                    session.clear()
                    session["user_id"] = user["id"]
                    session["username"] = user["username"]
                    session.permanent = True
                    mark_last_login(database_url, user["id"])
                    return redirect(next_url or url_for("main.index"))
    return render_template(
        "login.html",
        error=error,
        allow_register=register_allowed,
        next_url=next_url,
    )

@bp.route("/register", methods=["GET", "POST"])
def register():
    error = None
    database_url = current_app.config["DATABASE_URL"]
    register_allowed = registration_open(database_url, current_app.config["ALLOW_REGISTER"])
    if not register_allowed:
        return redirect(url_for("auth.login"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        error = validate_registration(username, password, confirm)
        if not error:
            try:
                user_id = create_user(
                    database_url, username, generate_password_hash(password)
                )
                bonus_points = int(
                    load_effective_settings(database_url).get(
                        "register_bonus_points", 0
                    )
                )
                if bonus_points > 0:
                    try:
                        apply_points_change(
                            database_url,
                            user_id,
                            bonus_points,
                            reason=POINTS_REASON_REGISTER,
                            source=POINTS_SOURCE_REGISTER,
                            note="注册赠送积分",
                        )
                    except Exception:
                        pass
                session.clear()
                session["user_id"] = user_id
                session["username"] = username
                session.permanent = True
                mark_last_login(database_url, user_id)
                return redirect(url_for("main.index"))
            except UserExistsError as exc:
                error = str(exc)
            except Exception:
                error = "注册失败，请稍后重试。"
    return render_template("register.html", error=error)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
