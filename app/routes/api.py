from __future__ import annotations
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
import time
import uuid
import base64
import tempfile
from pathlib import Path
from typing import Dict

from app.models.db import (
    apply_points_change,
    fetch_analysis_record,
    create_analysis_record,
    update_analysis_record,
    create_lut_record,
    create_image_record,
)
from app.services.settings import load_effective_settings
from app.services.storage import load_qiniu_config, upload_to_qiniu, upload_run_outputs, append_qiniu_image_suffix
from app.services.pipeline import run_pipeline
from app.services.styles import STYLE_PRESETS, normalize_lut_space, normalize_scene_type
from app.services.ai import stream_analyze_scene, resolve_doubao_image_input
from app.utils.auth import login_required, get_session_user_id
from app.utils.common import is_safe_run_id, normalize_style_strength, require_dependencies
from app.utils.image import normalize_image_input, resolve_image_suffix, is_heif_image, HEIF_EXTENSIONS, guess_mime_type
from app.config import (
    AppConfig, DEFAULT_LUT_SPACE, DEFAULT_SCENE_TYPE, DEFAULT_STYLE_STRENGTH,
    POINTS_REASON_FILTER, POINTS_SOURCE_FILTER, POINTS_REASON_REFUND
)

bp = Blueprint("api", __name__)

@bp.route("/api/generate", methods=["POST"])
@login_required
def api_generate():
    require_dependencies()
    database_url = current_app.config["DATABASE_URL"]
    settings = load_effective_settings(database_url)
    api_key = str(settings.get("api_key", ""))
    doubao_api_key = str(settings.get("doubao_api_key", ""))
    if api_key and not doubao_api_key:
        doubao_api_key = api_key
    if doubao_api_key and not api_key:
        api_key = doubao_api_key
    analysis_override = request.form.get("analysis", "").strip()
    analysis_model = str(settings.get("analysis_model", "gemini-1.5-flash"))
    image_model = str(settings.get("image_model", "gemini-1.5-flash"))
    if (
            not api_key
            and not analysis_model.startswith("doubao-")
            and not image_model.startswith("doubao-")
    ):
        return jsonify({"error": "未提供 API Key，请先填写。"}), 400
    if (
            not doubao_api_key
            and (analysis_model.startswith("doubao-") or image_model.startswith("doubao-"))
    ):
        return jsonify({"error": "未提供豆包 API Key，请先填写。"}), 400

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "请先上传静帧。"}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "上传的图片为空。"}), 400

    generate_lut_param = request.form.get("generate_lut", "1").strip().lower()
    generate_lut_flag = generate_lut_param not in {"0", "false", "no", "off"}
    lut_space = normalize_lut_space(
        request.form.get("lut_space", settings.get("lut_space", DEFAULT_LUT_SPACE))
    )
    debug_requests = request.form.get("debug_requests", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    style_ids = request.form.getlist("styles")
    if not style_ids:
        style_ids = [style.id for style in STYLE_PRESETS]

    needs_url_input = analysis_model.startswith("doubao-") or image_model.startswith("doubao-")
    qiniu_config = load_qiniu_config(settings)
    if needs_url_input and not qiniu_config:
        return jsonify({"error": "豆包模型需要配置七牛云以使用图片 URL 输入。"}), 400

    requested_run_id = request.form.get("run_id", "").strip()
    if requested_run_id and not is_safe_run_id(requested_run_id):
        return jsonify({"error": "无效的任务编号。"}), 400
    user_id = get_session_user_id()
    if not user_id:
        return jsonify({"error": "未登录，请先登录。"}), 401
    existing_record = None
    if requested_run_id:
        existing_record = fetch_analysis_record(database_url, user_id, requested_run_id)
        if not existing_record:
            return jsonify({"error": "记录不存在或无权限访问。"}), 404
        run_id = requested_run_id
    else:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    cost = max(1, len(style_ids))
    try:
        apply_points_change(
            database_url,
            user_id,
            -cost,
            reason=POINTS_REASON_FILTER,
            source=POINTS_SOURCE_FILTER,
            note=f"styles={','.join(style_ids)} run_id={run_id}",
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "积分扣减失败，请稍后重试。"}), 500
    base_out_dir = Path(settings.get("out_dir", "outputs"))
    out_dir = base_out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        image_path=Path(settings.get("image_path", "input.png")),
        api_key=api_key,
        doubao_api_key=doubao_api_key,
        analysis_model=analysis_model,
        image_model=image_model,
        out_dir=out_dir,
        styles=",".join(style_ids),
        no_lut=not generate_lut_flag,
        sample_size=int(settings.get("sample_size", 128)),
        lut_size=int(settings.get("lut_size", 65)),
        lut_space=lut_space,
        scene_type=normalize_scene_type(
            request.form.get("scene_type", settings.get("scene_type", DEFAULT_SCENE_TYPE))
        ),
        style_strength=normalize_style_strength(
            request.form.get("style_strength", settings.get("style_strength", DEFAULT_STYLE_STRENGTH))
        ),
        retries=int(settings.get("retries", 5)),
        debug_requests=bool(settings.get("debug_requests", False)),
    )

    try:
        image_bytes, source_image, mime_type, original_suffix = normalize_image_input(
            image_bytes,
            filename=image_file.filename or "",
            mime_type=image_file.mimetype or "",
        )
    except Exception as exc:
        return jsonify({"error": f"图片解析失败: {exc}"}), 400
    source_filename = f"source{original_suffix}"
    source_path = out_dir / source_filename
    source_path.write_bytes(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    uploaded_files: Dict[str, str] = {}
    image_url = None
    if needs_url_input:
        try:
            source_upload_url = upload_to_qiniu(
                qiniu_config, source_path, f"{run_id}/{source_filename}"
            )
        except Exception as exc:
            try:
                apply_points_change(
                    database_url,
                    user_id,
                    cost,
                    reason=POINTS_REASON_REFUND,
                    source=POINTS_SOURCE_FILTER,
                    note=f"upload_failed run_id={run_id}",
                )
            except Exception:
                pass
            return jsonify({"error": str(exc)}), 500
        uploaded_files[source_filename] = source_upload_url
        image_url = append_qiniu_image_suffix(source_upload_url)

    try:
        scene_description, results = run_pipeline(
            config,
            image_b64,
            source_image,
            mime_type,
            style_ids,
            image_url=image_url,
            analysis_override=analysis_override or None,
            debug_requests=debug_requests,
        )
    except Exception as exc:
        try:
            apply_points_change(
                database_url,
                user_id,
                cost,
                reason=POINTS_REASON_REFUND,
                source=POINTS_SOURCE_FILTER,
                note=f"styles={','.join(style_ids)} run_id={run_id}",
            )
        except Exception:
            pass
        return jsonify({"error": str(exc)}), 400

    if qiniu_config:
        try:
            uploaded_files = upload_run_outputs(
                out_dir, run_id, qiniu_config, existing=uploaded_files
            )
        except Exception as exc:
            try:
                apply_points_change(
                    database_url,
                    user_id,
                    cost,
                    reason=POINTS_REASON_REFUND,
                    source=POINTS_SOURCE_FILTER,
                    note=f"upload_failed run_id={run_id}",
                )
            except Exception:
                pass
            return jsonify({"error": str(exc)}), 500

    analysis_url = uploaded_files.get("analysis.txt")
    if not analysis_url:
        analysis_url = f"/api/download/{out_dir.name}/analysis.txt"
    source_url = uploaded_files.get(source_filename)
    if not source_url:
        source_url = f"/api/download/{out_dir.name}/{source_filename}"
    try:
        if existing_record:
            existing_style_ids = [
                item
                for item in str(existing_record.get("style_ids", "")).split(",")
                if item
            ]
            merged_style_ids = existing_style_ids + [
                item for item in style_ids if item not in existing_style_ids
            ]
            updated_cost = int(existing_record.get("cost") or 0) + cost
            update_analysis_record(
                database_url,
                user_id=user_id,
                run_id=run_id,
                style_ids=merged_style_ids,
                cost=updated_cost,
                source_filename=source_filename,
                source_url=source_url,
                analysis_text=scene_description,
                analysis_url=analysis_url,
            )
        else:
            create_analysis_record(
                database_url,
                user_id=user_id,
                run_id=run_id,
                style_ids=style_ids,
                cost=cost,
                source_filename=source_filename,
                source_url=source_url,
                analysis_text=scene_description,
                analysis_url=analysis_url,
            )
    except Exception:
        pass

    payload_results = []
    for item in results:
        image_bytes = item["image_bytes"]
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        lut_filename = item.get("lut_filename")
        lut_content = item.get("lut_content")
        lut_url = None
        if lut_filename:
            lut_url = uploaded_files.get(lut_filename)
            if not lut_url:
                lut_url = f"/api/download/{out_dir.name}/{lut_filename}"
            if lut_content:
                try:
                    create_lut_record(
                        database_url,
                        user_id=user_id,
                        run_id=out_dir.name,
                        style_id=str(item.get("id", "")),
                        lut_space=config.lut_space,
                        lut_size=config.lut_size,
                        lut_filename=lut_filename,
                        lut_content=lut_content,
                        lut_url=lut_url,
                    )
                except Exception:
                    pass
        image_filename = f"ref_{item['id']}.png"
        image_url = uploaded_files.get(image_filename)
        if not image_url:
            image_url = f"/api/download/{out_dir.name}/{image_filename}"
        try:
            create_image_record(
                database_url,
                user_id=user_id,
                run_id=out_dir.name,
                style_id=str(item.get("id", "")),
                image_filename=image_filename,
                image_url=image_url,
            )
        except Exception:
            pass
        payload_results.append(
            {
                "id": item["id"],
                "name": item["name"],
                "description": item["description"],
                "image": f"data:image/png;base64,{image_base64}",
                "image_url": image_url,
                "lut_url": lut_url,
            }
        )

    return jsonify(
        {
            "analysis": scene_description,
            "analysis_url": analysis_url,
            "source_url": source_url,
            "results": payload_results,
            "run_id": out_dir.name,
        }
    )

@bp.route("/api/analyze_stream", methods=["POST"])
@login_required
def api_analyze_stream():
    require_dependencies()
    database_url = current_app.config["DATABASE_URL"]
    settings = load_effective_settings(database_url)
    api_key = str(settings.get("api_key", ""))
    doubao_api_key = str(settings.get("doubao_api_key", ""))
    if api_key and not doubao_api_key:
        doubao_api_key = api_key
    if doubao_api_key and not api_key:
        api_key = doubao_api_key
    analysis_model = str(settings.get("analysis_model", "gemini-1.5-flash"))
    if not api_key and not analysis_model.startswith("doubao-"):
        return jsonify({"error": "未提供 API Key，请先填写。"}), 400
    if not doubao_api_key and analysis_model.startswith("doubao-"):
        return jsonify({"error": "未提供豆包 API Key，请先填写。"}), 400

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "请先上传静帧。"}), 400
    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "上传的图片为空。"}), 400

    debug_requests = request.form.get("debug_requests", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    config = AppConfig(
        image_path=Path(settings.get("image_path", "input.png")),
        api_key=api_key,
        doubao_api_key=doubao_api_key,
        analysis_model=analysis_model,
        image_model=str(settings.get("image_model", "gemini-1.5-flash")),
        out_dir=Path(settings.get("out_dir", "outputs")),
        styles=str(settings.get("styles", "all")),
        no_lut=bool(settings.get("no_lut", False)),
        sample_size=int(settings.get("sample_size", 128)),
        lut_size=int(settings.get("lut_size", 65)),
        lut_space=normalize_lut_space(settings.get("lut_space", DEFAULT_LUT_SPACE)),
        scene_type=normalize_scene_type(settings.get("scene_type", DEFAULT_SCENE_TYPE)),
        style_strength=normalize_style_strength(
            settings.get("style_strength", DEFAULT_STYLE_STRENGTH)
        ),
        retries=int(settings.get("retries", 5)),
        debug_requests=bool(settings.get("debug_requests", False)),
    )

    mime_type = (image_file.mimetype or "").strip().lower()
    original_suffix = resolve_image_suffix(image_file.filename or "", mime_type)
    if is_heif_image(image_file.filename or "", mime_type) or original_suffix in HEIF_EXTENSIONS:
        try:
            image_bytes, _, mime_type, original_suffix = normalize_image_input(
                image_bytes,
                filename=image_file.filename or "",
                mime_type=mime_type,
            )
        except Exception as exc:
            return jsonify({"error": f"图片解析失败: {exc}"}), 400
    elif not mime_type:
        mime_type = guess_mime_type(Path(f"file{original_suffix}"))
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_url = None
    if analysis_model.startswith("doubao-"):
        qiniu_config = load_qiniu_config(settings)
        if not qiniu_config:
            return jsonify({"error": "豆包模型需要配置七牛云以使用图片 URL 输入。"}), 400
        upload_id = f"stream-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        key = f"{upload_id}/source{original_suffix}"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="ai_lut_", suffix=original_suffix, delete=False
            ) as temp_file:
                temp_file.write(image_bytes)
                temp_path = Path(temp_file.name)
            source_upload_url = upload_to_qiniu(qiniu_config, temp_path, key)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        finally:
            if temp_path:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
        image_url = append_qiniu_image_suffix(source_upload_url)

    def generate():
        # Global debug state is problematic here, passing config is better
        # But stream_analyze_scene might depend on config.debug_requests
        try:
            for chunk in stream_analyze_scene(
                image_b64,
                mime_type,
                config,
                config.retries,
                image_url=image_url,
            ):
                yield chunk
        finally:
            pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )
