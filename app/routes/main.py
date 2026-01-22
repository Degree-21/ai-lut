from __future__ import annotations
from flask import Blueprint, render_template, request, session, current_app, send_from_directory, Response, jsonify, redirect, url_for
from pathlib import Path

from app.models.db import (
    get_user_points,
    list_points_transactions,
    list_analysis_records,
    fetch_analysis_record,
    list_image_records,
    list_lut_records,
    fetch_lut_content,
    upsert_settings
)
from app.services.settings import load_effective_settings
from app.services.user import is_admin_user
from app.services.styles import normalize_scene_type, normalize_lut_space, STYLE_PRESETS
from app.utils.auth import login_required, get_session_user_id
from app.utils.common import normalize_style_strength, format_points_note
from app.config import (
    DEFAULT_SCENE_TYPE, DEFAULT_LUT_SPACE, DEFAULT_STYLE_STRENGTH,
    POINTS_REASON_LABELS, POINTS_SOURCE_LABELS, load_settings
)

bp = Blueprint("main", __name__)

@bp.route("/")
def home():
    if get_session_user_id():
        return redirect(url_for("main.index"))
    return render_template("home.html")

@bp.route("/app")
@login_required
def index():
    database_url = current_app.config["DATABASE_URL"]
    settings = load_effective_settings(database_url)
    is_admin = is_admin_user(settings, session.get("username"))
    user_id = get_session_user_id()
    points_balance = get_user_points(database_url, user_id) if user_id else 0
    scene_type = normalize_scene_type(settings.get("scene_type", DEFAULT_SCENE_TYPE))
    style_strength = normalize_style_strength(
        settings.get("style_strength", DEFAULT_STYLE_STRENGTH)
    )
    return render_template(
        "index.html",
        api_key=str(settings.get("api_key", "")) if is_admin else "",
        doubao_api_key=str(settings.get("doubao_api_key", "")),
        analysis_model=str(settings.get("analysis_model", "gemini-1.5-flash")),
        image_model=str(settings.get("image_model", "gemini-1.5-flash")),
        lut_space=normalize_lut_space(settings.get("lut_space", DEFAULT_LUT_SPACE)),
        scene_type=scene_type,
        style_strength_percent=int(round(style_strength * 100)),
        current_user=session.get("username"),
        is_admin=is_admin,
        points_balance=points_balance,
    )

@bp.route("/image-gen")
@login_required
def image_gen():
    database_url = current_app.config["DATABASE_URL"]
    settings = load_effective_settings(database_url)
    user_id = get_session_user_id()
    points_balance = get_user_points(database_url, user_id) if user_id else 0
    is_admin = is_admin_user(settings, session.get("username"))
    
    return render_template(
        "image_gen.html",
        current_user=session.get("username"),
        points_balance=points_balance,
        is_admin=is_admin,
    )

@bp.route("/points")
@login_required
def points():
    database_url = current_app.config["DATABASE_URL"]
    user_id = get_session_user_id()
    balance = get_user_points(database_url, user_id) if user_id else 0
    run_id = (request.args.get("run_id") or "").strip()
    transactions = (
        list_points_transactions(
            database_url, user_id, limit=100, run_id=run_id or None
        )
        if user_id
        else []
    )
    for item in transactions:
        item["reason_label"] = POINTS_REASON_LABELS.get(
            str(item.get("reason", "")), str(item.get("reason", ""))
        )
        item["source_label"] = POINTS_SOURCE_LABELS.get(
            str(item.get("source", "")), str(item.get("source", ""))
        )
        item["note_label"] = format_points_note(
            item.get("note"), str(item.get("reason", ""))
        )
        # Extract run_id from note for linking
        note = str(item.get("note") or "")
        if "run_id=" in note:
            try:
                # Simple extraction assuming run_id is the last part or space-separated
                parts = note.split("run_id=")
                if len(parts) > 1:
                    # Take the part after run_id= and stop at next space or end
                    candidate = parts[1].split()[0].strip()
                    if candidate:
                        item["run_id"] = candidate
            except Exception:
                pass
    return render_template(
        "points.html",
        current_user=session.get("username"),
        balance=balance,
        transactions=transactions,
        run_id=run_id,
    )

@bp.route("/history")
@login_required
def history():
    database_url = current_app.config["DATABASE_URL"]
    user_id = get_session_user_id()
    records = list_analysis_records(database_url, user_id, limit=50) if user_id else []
    return render_template(
        "history.html",
        current_user=session.get("username"),
        records=records,
    )

@bp.route("/api/history/<run_id>")
@login_required
def api_history_detail(run_id: str):
    database_url = current_app.config["DATABASE_URL"]
    user_id = get_session_user_id()
    if not user_id:
        return jsonify({"error": "未登录，请先登录。"}), 401
    record = fetch_analysis_record(database_url, user_id, run_id)
    if not record:
        return jsonify({"error": "记录不存在。"}), 404
    style_ids_raw = str(record.get("style_ids", ""))
    style_ids = [item for item in style_ids_raw.split(",") if item]
    style_map = {style.id: style for style in STYLE_PRESETS}
    image_records = list_image_records(database_url, user_id, run_id)
    lut_records = list_lut_records(database_url, user_id, run_id)
    image_map = {row.get("style_id"): row for row in image_records}
    lut_map = {row.get("style_id"): row for row in lut_records}
    results = []
    for style_id in style_ids:
        style = style_map.get(style_id)
        image_row = image_map.get(style_id) or {}
        image_url = image_row.get("image_url")
        image_filename = image_row.get("image_filename")
        if not image_url and image_filename:
            image_url = f"/api/download/{run_id}/{image_filename}"
        lut_row = lut_map.get(style_id) or {}
        lut_url = lut_row.get("lut_url")
        lut_filename = lut_row.get("lut_filename")
        if not lut_url and lut_filename:
            lut_url = f"/api/download/{run_id}/{lut_filename}"
        results.append(
            {
                "id": style_id,
                "name": style.name if style else style_id,
                "description": style.description if style else "",
                "image": image_url or "",
                "image_url": image_url,
                "lut_url": lut_url,
            }
        )
    return jsonify(
        {
            "analysis": record.get("analysis_text") or "",
            "analysis_url": record.get("analysis_url") or "",
            "source_url": record.get("source_url") or "",
            "results": results,
            "run_id": record.get("run_id") or run_id,
        }
    )

@bp.route("/admin/config", methods=["GET", "POST"])
@login_required
def admin_config():
    database_url = current_app.config["DATABASE_URL"]
    settings = load_effective_settings(database_url)
    if not is_admin_user(settings, session.get("username")):
        return "无权限访问。", 403

    error = None
    message = None
    if request.method == "POST":
        analysis_model = request.form.get("analysis_model", "").strip()
        image_model = request.form.get("image_model", "").strip()
        api_key = request.form.get("api_key", "").strip()
        doubao_api_key = request.form.get("doubao_api_key", "").strip()
        register_bonus_raw = request.form.get("register_bonus_points", "").strip()
        if not analysis_model:
            error = "解析模型不能为空。"
        elif not image_model:
            error = "输出模型不能为空。"
        else:
            try:
                register_bonus_points = int(register_bonus_raw or 0)
                if register_bonus_points < 0:
                    raise ValueError
            except ValueError:
                error = "注册赠送积分需为非负整数。"
                register_bonus_points = 0
            if error:
                return render_template(
                    "admin_config.html",
                    analysis_model=analysis_model,
                    image_model=image_model,
                    api_key=api_key,
                    doubao_api_key=doubao_api_key,
                    register_bonus_points=register_bonus_raw,
                    current_user=session.get("username"),
                    error=error,
                    message=message,
                )
            try:
                upsert_settings(
                    database_url,
                    {
                        "analysis_model": analysis_model,
                        "image_model": image_model,
                        "api_key": api_key,
                        "doubao_api_key": doubao_api_key,
                        "register_bonus_points": str(register_bonus_points),
                    },
                )
            except Exception:
                error = "保存失败，请检查数据库连接。"
            else:
                settings = load_effective_settings(database_url)
                message = "系统配置已保存。"

    return render_template(
        "admin_config.html",
        analysis_model=str(settings.get("analysis_model", "")),
        image_model=str(settings.get("image_model", "")),
        api_key=str(settings.get("api_key", "")),
        doubao_api_key=str(settings.get("doubao_api_key", "")),
        register_bonus_points=str(settings.get("register_bonus_points", 0)),
        current_user=session.get("username"),
        error=error,
        message=message,
    )

@bp.route("/api/download/<run_id>/<path:filename>")
@login_required
def api_download(run_id: str, filename: str):
    base_out_dir = Path(load_settings().get("out_dir", "outputs"))
    safe_dir = (base_out_dir / run_id).resolve()
    file_path = (safe_dir / filename).resolve()
    if safe_dir not in file_path.parents:
        return jsonify({"error": "非法路径。"}), 400
    if not file_path.exists():
        if filename.lower().endswith(".cube"):
            database_url = current_app.config["DATABASE_URL"]
            user_id = get_session_user_id()
            if user_id:
                lut_content = fetch_lut_content(
                    database_url,
                    user_id=user_id,
                    run_id=run_id,
                    lut_filename=filename,
                )
                if lut_content:
                    response = Response(lut_content, mimetype="text/plain; charset=utf-8")
                    response.headers["Content-Disposition"] = (
                        f'attachment; filename="{filename}"'
                    )
                    return response
        return jsonify({"error": "文件不存在。"}), 404
    
    mimetype = None
    if filename.lower().endswith(".txt"):
        mimetype = "text/plain; charset=utf-8"
        
    return send_from_directory(safe_dir, filename, as_attachment=True, mimetype=mimetype)
