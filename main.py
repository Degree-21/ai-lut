from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    stream_with_context,
    url_for,
)
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash

from config import load_settings
from user_store import (
    UserExistsError,
    apply_points_change,
    count_users,
    create_user,
    create_analysis_record,
    create_image_record,
    create_lut_record,
    fetch_analysis_record,
    fetch_user_by_username,
    fetch_lut_content,
    fetch_settings,
    get_user_points,
    init_db,
    list_analysis_records,
    list_image_records,
    list_lut_records,
    list_points_transactions,
    mark_last_login,
    update_analysis_record,
    upsert_settings,
)

DEBUG_REQUESTS_STATE = False
MIN_PASSWORD_LENGTH = 8
MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 32
DB_SETTING_KEYS = (
    "analysis_model",
    "image_model",
    "api_key",
    "doubao_api_key",
    "register_bonus_points",
)
POINTS_REASON_REGISTER = "register_bonus"
POINTS_SOURCE_REGISTER = "register"
POINTS_REASON_FILTER = "filter_use"
POINTS_SOURCE_FILTER = "generate"
POINTS_REASON_REFUND = "filter_refund"
ICP_RECORD = "闽ICP备2025083568号-1"
DEFAULT_LUT_SPACE = "rec709_sdr"
DEFAULT_SCENE_TYPE = "auto"
DEFAULT_STYLE_STRENGTH = 0.7
QINIU_IMAGE_SUFFIX = "?imageMogr2/thumbnail/200x/strip/format/jpg"
POINTS_REASON_LABELS = {
    POINTS_REASON_REGISTER: "注册赠送",
    POINTS_REASON_FILTER: "调色生成",
    POINTS_REASON_REFUND: "生成失败返还",
}
POINTS_SOURCE_LABELS = {
    POINTS_SOURCE_REGISTER: "注册",
    POINTS_SOURCE_FILTER: "生成",
}


@dataclass(frozen=True)
class StylePreset:
    id: str
    name: str
    description: str


STYLE_PRESETS = [
    StylePreset("cinematic", "好莱坞电影", "青橙色调 (Teal & Orange)，高对比度，深邃阴影，极具戏剧感。"),
    StylePreset("vintage", "经典胶片", "Kodak 暖黄色调，柔和的高光溢出，低饱和度，怀旧质感。"),
    StylePreset("minimal", "清新日系", "高调照明 (High-key)，低对比度，淡蓝色或偏白影调，干净明亮。"),
    StylePreset("noir", "暗黑悬疑", "低色温，强调阴影细节，冷峻的青蓝色系，压抑且迷人。"),
    StylePreset("commercial", "时尚商业", "高饱和，色彩还原准确且明亮，光影分布均匀，质感通透。"),
    StylePreset("cyber", "赛博都市", "霓虹冷暖色差，强烈的紫色与青色碰撞，极具现代冲击力。"),
]


ANALYSIS_PROMPT = (
    "请作为电影调色师和摄影指导，详细描述这张静帧。包含："
    "1.画面构图和所有主体元素。2.现有的光源位置和性质。3.画面的物理结构。"
    "请输出一段极其详尽的描述，用于指导另一个AI生成相同结构但不同影调的图片。"
)

LUT_SPACE_PRESETS = {
    "rec709_sdr": {
        "label": "Rec.709 SDR (BT.1886)",
        "gamma": 2.4,
        "file_tag": "709SDR",
    },
    "rec709a": {
        "label": "Rec.709-A (Apple Display Gamma)",
        "gamma": 1.96,
        "file_tag": "709A",
    },
}

SCENE_TYPE_PRESETS = {
    "auto": {
        "label": "自动",
        "prompt": "",
        "chroma_scale": 1.0,
    },
    "portrait": {
        "label": "人像",
        "prompt": (
            "Portrait scene. Preserve natural skin tones, avoid hue shifts, "
            "keep skin texture clean and realistic."
        ),
        "chroma_scale": 0.55,
    },
    "landscape": {
        "label": "风光",
        "prompt": (
            "Landscape scene. Keep skies and foliage natural, avoid oversaturation."
        ),
        "chroma_scale": 1.0,
    },
    "city": {
        "label": "城市/建筑",
        "prompt": (
            "Urban/architecture scene. Keep neutral grays stable, avoid neon clipping."
        ),
        "chroma_scale": 0.9,
    },
    "night": {
        "label": "夜景",
        "prompt": (
            "Night scene. Preserve highlight detail, avoid crushed blacks and color clipping."
        ),
        "chroma_scale": 0.8,
    },
    "indoor": {
        "label": "室内",
        "prompt": (
            "Indoor scene. Maintain realistic white balance, avoid strong color casts."
        ),
        "chroma_scale": 0.8,
    },
}


def normalize_lut_space(value: str | None) -> str:
    if not value:
        return DEFAULT_LUT_SPACE
    key = str(value).strip().lower().replace(" ", "")
    if key in {"rec709a", "rec.709a", "rec709-a", "709a"}:
        return "rec709a"
    if key in {
        "rec709_sdr",
        "rec709sdr",
        "rec.709sdr",
        "rec709_24",
        "rec709",
        "rec.709",
        "709",
        "g24",
        "gamma24",
        "2.4",
        "rec709-g24",
        "bt1886",
        "bt.1886",
    }:
        return "rec709_sdr"
    return DEFAULT_LUT_SPACE


def get_lut_space_info(lut_space: str | None) -> Tuple[str, float, str, str]:
    normalized = normalize_lut_space(lut_space)
    info = LUT_SPACE_PRESETS[normalized]
    return normalized, float(info["gamma"]), str(info["file_tag"]), str(info["label"])


def normalize_scene_type(value: str | None) -> str:
    if not value:
        return DEFAULT_SCENE_TYPE
    key = str(value).strip().lower().replace(" ", "")
    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "portrait": "portrait",
        "people": "portrait",
        "person": "portrait",
        "human": "portrait",
        "face": "portrait",
        "人像": "portrait",
        "landscape": "landscape",
        "scenery": "landscape",
        "风光": "landscape",
        "city": "city",
        "urban": "city",
        "architecture": "city",
        "城市": "city",
        "night": "night",
        "夜景": "night",
        "indoor": "indoor",
        "interior": "indoor",
        "室内": "indoor",
    }
    return aliases.get(key, DEFAULT_SCENE_TYPE)


def get_scene_profile(scene_type: str | None) -> Dict[str, object]:
    normalized = normalize_scene_type(scene_type)
    return SCENE_TYPE_PRESETS[normalized]


def normalize_style_strength(value: object) -> float:
    if value is None or value == "":
        return DEFAULT_STYLE_STRENGTH
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return DEFAULT_STYLE_STRENGTH
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


def build_style_prompt(scene_description: str, style: StylePreset, config: AppConfig) -> str:
    scene_profile = get_scene_profile(config.scene_type)
    scene_hint = str(scene_profile.get("prompt", "")).strip()
    strength = normalize_style_strength(config.style_strength)
    strength_label = describe_style_strength(strength)
    parts = [
        f"A professional movie frame with {style.name} color grade. {style.description}.",
        f"Base scene description: {scene_description}.",
    ]
    if scene_hint:
        parts.append(scene_hint)
    parts.append(
        f"Intensity: {strength_label} color grading, refined and controlled."
    )
    parts.append(
        "CRITICAL: Keep the original composition, subject identity, and physical structure 100% identical."
    )
    parts.append("Only change lighting and color grading. High detail, cinematic lighting, 4k quality.")
    return " ".join(parts)


class ColorTools:
    @staticmethod
    def gamma_to_linear(v: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.maximum(0.0, v), gamma)

    @staticmethod
    def linear_to_gamma(v: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.maximum(0.0, v), 1 / gamma)

    @staticmethod
    def rgb_to_oklab(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = np.cbrt(l)
        m_ = np.cbrt(m)
        s_ = np.cbrt(s)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720403 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return L, a, b

    @staticmethod
    def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        return r, g, b


@dataclass(frozen=True)
class AppConfig:
    image_path: Path
    api_key: str
    doubao_api_key: str
    analysis_model: str
    image_model: str
    out_dir: Path
    styles: str
    no_lut: bool
    sample_size: int
    lut_size: int
    lut_space: str
    scene_type: str
    style_strength: float
    retries: int
    debug_requests: bool


@dataclass(frozen=True)
class QiniuConfig:
    access_key: str
    secret_key: str
    bucket: str
    domain: str


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def registration_open(database_url: str, allow_register: bool) -> bool:
    if allow_register:
        return True
    try:
        return count_users(database_url) == 0
    except Exception:
        return False


def clean_next_url(value: str | None) -> str | None:
    if not value:
        return None
    if not value.startswith("/") or value.startswith("//"):
        return None
    return value


def is_safe_run_id(value: str) -> bool:
    if not value:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return all(char in allowed for char in value)


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
        return redirect(url_for("login", next=next_url))

    return wrapped


def get_session_user_id() -> int | None:
    value = session.get("user_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def is_admin_user(settings: Dict[str, object], username: str | None) -> bool:
    admin_username = str(settings.get("admin_username", "")).strip()
    if not admin_username or not username:
        return False
    return username == admin_username


def load_effective_settings(database_url: str) -> Dict[str, object]:
    settings = load_settings()
    try:
        db_settings = fetch_settings(database_url, DB_SETTING_KEYS)
    except Exception:
        return settings
    for key, value in db_settings.items():
        settings[key] = value
    if "register_bonus_points" in settings:
        try:
            settings["register_bonus_points"] = int(settings["register_bonus_points"])
        except (TypeError, ValueError):
            settings["register_bonus_points"] = 0
    return settings


def extension_from_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type in {"image/heic", "image/heif"}:
        return ".heic"
    if mime_type == "image/png":
        return ".png"
    return ""


def load_qiniu_config(settings: Dict[str, object]) -> QiniuConfig | None:
    access_key = str(settings.get("qiniu_access_key", "")).strip()
    secret_key = str(settings.get("qiniu_secret_key", "")).strip()
    bucket = str(settings.get("qiniu_bucket", "")).strip()
    domain = str(settings.get("qiniu_domain", "")).strip()
    if not access_key or not secret_key or not bucket:
        return None
    if domain and not domain.startswith("http"):
        domain = f"https://{domain}"
    domain = domain.rstrip("/")
    if not domain:
        return None
    return QiniuConfig(
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        domain=domain,
    )


def upload_to_qiniu(config: QiniuConfig, file_path: Path, key: str) -> str:
    try:
        from qiniu import Auth, put_file
    except Exception as exc:
        raise RuntimeError("缺少七牛 SDK，请先安装 requirements.txt。") from exc
    auth = Auth(config.access_key, config.secret_key)
    token = auth.upload_token(config.bucket, key)
    ret, info = put_file(token, key, str(file_path))
    if info.status_code != 200:
        raise RuntimeError(f"七牛上传失败: {info.status_code}")
    return f"{config.domain}/{key}"


def upload_run_outputs(
    out_dir: Path,
    run_id: str,
    config: QiniuConfig,
    existing: Dict[str, str] | None = None,
) -> Dict[str, str]:
    uploads: Dict[str, str] = dict(existing or {})
    for path in sorted(out_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in uploads:
            continue
        key = f"{run_id}/{path.name}"
        uploads[path.name] = upload_to_qiniu(config, path, key)
    return uploads


def append_qiniu_image_suffix(url: str, suffix: str = QINIU_IMAGE_SUFFIX) -> str:
    if not suffix:
        return url
    if "?" in url:
        if suffix.startswith("?"):
            return f"{url}&{suffix[1:]}"
        return f"{url}&{suffix}"
    if suffix.startswith("?"):
        return f"{url}{suffix}"
    return f"{url}?{suffix}"


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


def load_config() -> AppConfig:
    settings = load_settings()
    image_path = Path(settings.get("image_path", "input.png"))
    out_dir = Path(settings.get("out_dir", "outputs"))
    styles = str(settings.get("styles", "all"))
    no_lut = bool(settings.get("no_lut", False))
    sample_size = int(settings.get("sample_size", 128))
    lut_size = int(settings.get("lut_size", 65))
    lut_space = normalize_lut_space(settings.get("lut_space", DEFAULT_LUT_SPACE))
    scene_type = normalize_scene_type(settings.get("scene_type", DEFAULT_SCENE_TYPE))
    style_strength = normalize_style_strength(
        settings.get("style_strength", DEFAULT_STYLE_STRENGTH)
    )
    retries = int(settings.get("retries", 5))
    return AppConfig(
        image_path=image_path,
        api_key=str(settings.get("api_key", "")),
        doubao_api_key=str(settings.get("doubao_api_key", "")),
        analysis_model=str(settings.get("analysis_model", "gemini-1.5-flash")),
        image_model=str(settings.get("image_model", "gemini-1.5-flash")),
        out_dir=out_dir,
        styles=styles,
        no_lut=no_lut,
        sample_size=sample_size,
        lut_size=lut_size,
        lut_space=lut_space,
        scene_type=scene_type,
        style_strength=style_strength,
        retries=retries,
        debug_requests=bool(settings.get("debug_requests", False)),
    )


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


def guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext in {".heic", ".heif"}:
        return "image/heic"
    return "image/png"


HEIF_MIME_TYPES = {"image/heic", "image/heif"}
HEIF_EXTENSIONS = {".heic", ".heif"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"} | HEIF_EXTENSIONS


def is_heif_image(filename: str | None, mime_type: str | None) -> bool:
    if mime_type and mime_type.lower() in HEIF_MIME_TYPES:
        return True
    suffix = Path(filename or "").suffix.lower()
    return suffix in HEIF_EXTENSIONS


def ensure_heif_support() -> None:
    if try_register_heif_opener():
        return
    raise RuntimeError(
        "检测到 HEIC/HEIF 图片，请先安装 pillow-heif 以解码。"
    )


def try_register_heif_opener() -> bool:
    try:
        import pillow_heif
    except Exception:
        return False
    try:
        pillow_heif.register_heif_opener()
    except Exception:
        return False
    return True


def resolve_image_suffix(filename: str | None, mime_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return suffix
    if mime_type:
        ext = extension_from_mime(mime_type.lower())
        if ext:
            return ext
    return ".png"


def normalize_image_input(
    image_bytes: bytes,
    *,
    filename: str | None,
    mime_type: str | None,
) -> Tuple[bytes, Image.Image, str, str]:
    try_register_heif_opener()
    resolved_mime = (mime_type or "").strip().lower()
    resolved_suffix = resolve_image_suffix(filename, resolved_mime)
    if is_heif_image(filename, resolved_mime) or resolved_suffix in HEIF_EXTENSIONS:
        ensure_heif_support()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), image, "image/png", ".png"
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not resolved_mime:
        resolved_mime = guess_mime_type(Path(f"file{resolved_suffix}"))
    return image_bytes, image, resolved_mime, resolved_suffix


def load_image_base64(path: Path) -> Tuple[str, Image.Image, str]:
    image_bytes = path.read_bytes()
    mime_type = guess_mime_type(path)
    image_bytes, image, mime_type, _ = normalize_image_input(
        image_bytes, filename=path.name, mime_type=mime_type
    )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return encoded, image, mime_type


def fetch_with_retry(
        url: str,
        payload: Dict,
        headers: Dict | None = None,
        retries: int = 5,
        backoff: float = 1.0,
) -> Dict:
    if headers is None:
        headers = {"Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            if DEBUG_REQUESTS_STATE or env_flag("DEBUG_REQUESTS", "0"):
                log_request(url, payload, attempt)
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if response.status_code == 401:
                raise RuntimeError("API授权失败 (401)，请检查 Key 或模型访问权限。")
            if not response.ok:
                detail = ""
                try:
                    detail = json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    detail = response.text
                raise RuntimeError(f"HTTP {response.status_code} 请求失败: {detail}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt >= retries or "401" in str(exc):
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("请求失败")


def log_request(url: str, payload: Dict, attempt: int) -> None:
    redacted_url = redact_key(url)
    print(f"[debug] 请求 {attempt + 1}: POST {redacted_url}", flush=True)
    try:
        scrubbed = scrub_payload(payload)
        print(json.dumps(scrubbed, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print(f"[debug] 请求体序列化失败: {exc}", flush=True)


def redact_key(url: str) -> str:
    if "key=" not in url:
        return url
    prefix, rest = url.split("key=", 1)
    if "&" in rest:
        _, suffix = rest.split("&", 1)
        return f"{prefix}key=***&{suffix}"
    return f"{prefix}key=***"


def scrub_payload(value):
    if isinstance(value, dict):
        scrubbed = {}
        for key, item in value.items():
            if key == "data" and isinstance(item, str):
                scrubbed[key] = f"<{len(item)} chars base64>"
            else:
                scrubbed[key] = scrub_payload(item)
        return scrubbed
    if isinstance(value, list):
        return [scrub_payload(item) for item in value]
    return value


def resolve_style_ids(styles_value: str) -> List[str]:
    if not styles_value or styles_value.strip().lower() == "all":
        return [style.id for style in STYLE_PRESETS]
    wanted = {item.strip() for item in styles_value.split(",") if item.strip()}
    return [style.id for style in STYLE_PRESETS if style.id in wanted]


def analyze_scene(
    image_b64: str,
    mime_type: str,
    config: AppConfig,
    retries: int,
    image_url: str | None = None,
) -> str:
    if config.analysis_model.startswith("doubao-"):
        if not config.doubao_api_key:
            raise RuntimeError("未提供豆包 API Key，请通过环境变量 ARK_API_KEY 设置。")

        image_input = resolve_doubao_image_input(image_b64, mime_type, image_url)
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=config.doubao_api_key,
        )

        response = client.chat.completions.create(
            model=config.analysis_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_input
                            },
                        },
                        {
                            "type": "text",
                            "text": ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content or "检测到复杂场景"

    if not config.api_key:
        raise RuntimeError(
            "未提供 API Key，请通过环境变量 GOOGLE_API_KEY/GEMINI_API_KEY/API_KEY 设置。"
        )

    ANALYSIS_MODEL = config.analysis_model
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            ANALYSIS_PROMPT
                        )
                    },
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                ]
            }
        ]
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{ANALYSIS_MODEL}:generateContent?key={config.api_key}"
    )
    result = fetch_with_retry(url, payload, retries=retries)
    return (
        result.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "检测到复杂场景")
    )


def extract_image_bytes(result: Dict) -> bytes:
    candidates = result.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("未获取到生成图像数据")


def resolve_doubao_image_input(
    image_b64: str, mime_type: str, image_url: str | None = None
) -> str:
    if image_url:
        return image_url
    if image_b64.startswith(("http://", "https://")):
        return image_b64
    return f"data:{mime_type};base64,{image_b64}"


def extract_doubao_image_bytes(result: Dict) -> bytes:
    data_item = result.get("data", [{}])[0]
    b64_data = data_item.get("b64_json")
    if b64_data:
        return base64.b64decode(b64_data)
    image_url = data_item.get("url")
    if image_url:
        image_response = requests.get(image_url, timeout=120)
        image_response.raise_for_status()
        return image_response.content
    raise RuntimeError("未获取到生成图像数据")


def stream_analyze_scene(
    image_b64: str,
    mime_type: str,
    config: AppConfig,
    retries: int,
    image_url: str | None = None,
):
    if not config.analysis_model.startswith("doubao-"):
        yield analyze_scene(image_b64, mime_type, config, retries, image_url=image_url)
        return
    if not config.doubao_api_key:
        raise RuntimeError("未提供豆包 API Key，请通过环境变量 ARK_API_KEY 设置。")

    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    image_input = resolve_doubao_image_input(image_b64, mime_type, image_url)
    payload = {
        "model": config.analysis_model,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_input}},
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.doubao_api_key}",
    }

    response = fetch_stream_with_retry(url, payload, headers=headers, retries=retries)
    try:
        for chunk in iter_doubao_stream(response):
            yield chunk
    finally:
        response.close()


def fetch_stream_with_retry(
    url: str,
    payload: Dict,
    headers: Dict | None = None,
    retries: int = 5,
    backoff: float = 1.0,
) -> requests.Response:
    if headers is None:
        headers = {"Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            if DEBUG_REQUESTS_STATE or env_flag("DEBUG_REQUESTS", "0"):
                log_request(url, payload, attempt)
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=120, stream=True
            )
            if response.status_code == 401:
                raise RuntimeError("API授权失败 (401)，请检查 Key 或模型访问权限。")
            if not response.ok:
                detail = response.text
                raise RuntimeError(f"HTTP {response.status_code} 请求失败: {detail}")
            response.raise_for_status()
            return response
        except Exception as exc:
            if attempt >= retries or "401" in str(exc):
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("请求失败")


def iter_doubao_stream(response: requests.Response):
    for line in response.iter_lines(decode_unicode=False):
        if not line:
            continue
        try:
            chunk = line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if chunk.startswith("data:"):
            chunk = chunk[len("data:") :].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content


def generate_style_image(
        scene_description: str,
        style: StylePreset,
        image_b64: str,
        mime_type: str,
        config: AppConfig,
        retries: int,
        image_url: str | None = None,
) -> bytes:
    if config.image_model.startswith("doubao-"):
        if not config.doubao_api_key:
            raise RuntimeError("未提供豆包 API Key，请通过环境变量 ARK_API_KEY 设置。")
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        image_input = resolve_doubao_image_input(image_b64, mime_type, image_url)
        prompt = build_style_prompt(scene_description, style, config)
        payload = {
            "model": config.image_model,
            "prompt": prompt,
            "image": [image_input],
            "size": "2k",
            "watermark": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.doubao_api_key}",
        }
        result = fetch_with_retry(url, payload, headers=headers, retries=retries)
        return extract_doubao_image_bytes(result)
    else:
        if not config.api_key:
            raise RuntimeError("未提供 API Key，请通过环境变量 GOOGLE_API_KEY/GEMINI_API_KEY/API_KEY 设置。")
        prompt = build_style_prompt(scene_description, style, config)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.image_model}:generateContent?key={config.api_key}"
        )
        result = fetch_with_retry(url, payload, retries=retries)
        try:
            return extract_image_bytes(result)
        except RuntimeError as exc:
            raise RuntimeError(f"未获取到 {style.id} 的生成结果") from exc


def extract_histogram(
    image: Image.Image,
    sample_size: int,
    gamma: float,
) -> Dict[str, np.ndarray]:
    resized = image.resize((sample_size, sample_size), Image.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    linear = ColorTools.gamma_to_linear(pixels, gamma)
    r = linear[:, :, 0]
    g = linear[:, :, 1]
    b = linear[:, :, 2]
    L, a, b = ColorTools.rgb_to_oklab(r, g, b)
    l_idx = np.clip((L * 255).astype(np.int32), 0, 255)
    a_idx = np.clip(((a + 0.4) * (255 / 0.8)).astype(np.int32), 0, 255)
    b_idx = np.clip(((b + 0.4) * (255 / 0.8)).astype(np.int32), 0, 255)

    hist = {
        "L": np.bincount(l_idx.ravel(), minlength=256).astype(np.float64),
        "a": np.bincount(a_idx.ravel(), minlength=256).astype(np.float64),
        "b": np.bincount(b_idx.ravel(), minlength=256).astype(np.float64),
    }
    return hist


def get_cdf(hist: np.ndarray) -> np.ndarray:
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return np.cumsum(hist) / total


def build_lookup(src_cdf: np.ndarray, tgt_cdf: np.ndarray) -> np.ndarray:
    return np.searchsorted(tgt_cdf, src_cdf, side="left").clip(0, 255)


def generate_lut(
        source: Image.Image,
        target: Image.Image,
        style_id: str,
        output_path: Path,
        sample_size: int = 128,
        lut_size: int = 65,
        lut_space: str = DEFAULT_LUT_SPACE,
        scene_type: str = DEFAULT_SCENE_TYPE,
        style_strength: float = DEFAULT_STYLE_STRENGTH,
) -> str:
    _, gamma, file_tag, space_label = get_lut_space_info(lut_space)
    scene_profile = get_scene_profile(scene_type)
    chroma_scale = float(scene_profile.get("chroma_scale", 1.0))
    strength = normalize_style_strength(style_strength)
    chroma_strength = strength * chroma_scale
    src_hist = extract_histogram(source, sample_size, gamma)
    tgt_hist = extract_histogram(target, sample_size, gamma)

    lookups = {
        "L": build_lookup(get_cdf(src_hist["L"]), get_cdf(tgt_hist["L"])),
        "a": build_lookup(get_cdf(src_hist["a"]), get_cdf(tgt_hist["a"])),
        "b": build_lookup(get_cdf(src_hist["b"]), get_cdf(tgt_hist["b"])),
    }

    levels = np.linspace(0.0, 1.0, lut_size)
    linear_levels = ColorTools.gamma_to_linear(levels, gamma)

    lines: List[str] = [
        f'TITLE "AI_{style_id.upper()}_{file_tag}"',
        f"LUT_3D_SIZE {lut_size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        f"# LUT_SPACE {space_label}",
        f"# GAMMA {gamma:.3f}",
        "",
    ]

    for b in range(lut_size):
        lb = linear_levels[b]
        for g in range(lut_size):
            lg = linear_levels[g]
            for r in range(lut_size):
                lr = linear_levels[r]
                L, a, b_val = ColorTools.rgb_to_oklab(np.array(lr), np.array(lg), np.array(lb))
                L = float(L)
                a = float(a)
                b_val = float(b_val)
                l_idx = int(np.clip(np.floor(L * 255), 0, 255))
                a_idx = int(np.clip(np.floor((a + 0.4) * (255 / 0.8)), 0, 255))
                b_idx = int(np.clip(np.floor((b_val + 0.4) * (255 / 0.8)), 0, 255))
                nL = float(lookups["L"][l_idx] / 255.0)
                na = float((lookups["a"][a_idx] / (255 / 0.8)) - 0.4)
                nb = float((lookups["b"][b_idx] / (255 / 0.8)) - 0.4)
                out_L = L + strength * (nL - L)
                out_a = a + chroma_strength * (na - a)
                out_b = b_val + chroma_strength * (nb - b_val)
                fr, fg, fb = ColorTools.oklab_to_rgb(out_L, out_a, out_b)
                fr = ColorTools.linear_to_gamma(np.array(fr), gamma)
                fg = ColorTools.linear_to_gamma(np.array(fg), gamma)
                fb = ColorTools.linear_to_gamma(np.array(fb), gamma)
                fr = float(np.clip(fr, 0.0, 1.0))
                fg = float(np.clip(fg, 0.0, 1.0))
                fb = float(np.clip(fb, 0.0, 1.0))
                lines.append(f"{fr:.6f} {fg:.6f} {fb:.6f}")

    lut_text = "\n".join(lines)
    output_path.write_text(lut_text, encoding="utf-8")
    return lut_text


def run_pipeline(
    config: AppConfig,
    image_b64: str,
    source_image: Image.Image,
    mime_type: str,
    style_ids: List[str],
    image_url: str | None = None,
    analysis_override: str | None = None,
    debug_requests: bool = False,
) -> Tuple[str, List[Dict[str, object]]]:
    global DEBUG_REQUESTS_STATE
    previous_debug = DEBUG_REQUESTS_STATE
    DEBUG_REQUESTS_STATE = debug_requests or config.debug_requests
    try:
        out_dir = config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_styles = [style for style in STYLE_PRESETS if style.id in style_ids]
        if not selected_styles:
            raise RuntimeError("未选择任何风格。")

        if analysis_override:
            scene_description = analysis_override
        else:
            scene_description = analyze_scene(
                image_b64, mime_type, config, config.retries, image_url=image_url
            )
        analysis_path = out_dir / "analysis.txt"
        analysis_path.write_text(scene_description, encoding="utf-8")

        results: List[Dict[str, object]] = []
        _, _, lut_file_tag, _ = get_lut_space_info(config.lut_space)
        for style in selected_styles:
            image_bytes = generate_style_image(
                scene_description,
                style,
                image_b64,
                mime_type,
                config,
                config.retries,
                image_url=image_url,
            )
            image_out = out_dir / f"ref_{style.id}.png"
            image_out.write_bytes(image_bytes)

            lut_filename = None
            lut_content = None
            if not config.no_lut:
                target_image = Image.open(image_out).convert("RGB")
                lut_out = out_dir / f"AI_Grade_{style.id}_{config.lut_size}_{lut_file_tag}.cube"
                lut_content = generate_lut(
                    source_image,
                    target_image,
                    style.id,
                    lut_out,
                    sample_size=config.sample_size,
                    lut_size=config.lut_size,
                    lut_space=config.lut_space,
                    scene_type=config.scene_type,
                    style_strength=config.style_strength,
                )
                lut_filename = lut_out.name

            results.append(
                {
                    "id": style.id,
                    "name": style.name,
                    "description": style.description,
                    "image_bytes": image_bytes,
                    "lut_filename": lut_filename,
                    "lut_content": lut_content,
                }
            )

        return scene_description, results
    finally:
        DEBUG_REQUESTS_STATE = previous_debug


def run_cli() -> None:
    require_dependencies(require_db=False)
    config = load_config()
    if not config.api_key:
        print(
            "未提供 API Key，已跳过分析流程。设置环境变量后重新运行即可开始生成。",
            file=sys.stderr,
        )
        return
    if not config.image_path.exists():
        raise FileNotFoundError(f"找不到输入图片: {config.image_path}，请设置 IMAGE_PATH。")

    image_b64, source_image, mime_type = load_image_base64(config.image_path)
    style_ids = resolve_style_ids(config.styles)
    scene_description, _ = run_pipeline(
        config,
        image_b64,
        source_image,
        mime_type,
        style_ids,
        debug_requests=config.debug_requests,
    )
    print(f"场景分析已保存: {config.out_dir / 'analysis.txt'}")
    print(scene_description)


def create_app() -> Flask:
    settings = load_settings()
    app = Flask(__name__, static_folder="static", template_folder="templates")
    require_dependencies()
    database_url = str(settings.get("database_url", "")).strip()
    if not database_url:
        raise RuntimeError("未配置 database_url，请在 config.yaml 或环境变量 DATABASE_URL 中设置。")
    app.config["DATABASE_URL"] = database_url
    app.config["ALLOW_REGISTER"] = bool(settings.get("allow_register", True))
    app.secret_key = str(settings.get("secret_key") or os.getenv("SECRET_KEY") or uuid.uuid4().hex)
    session_hours = int(settings.get("session_expire_hours", 12))
    app.permanent_session_lifetime = timedelta(hours=session_hours)
    init_db(database_url)
    ensure_admin_user(
        database_url,
        str(settings.get("admin_username", "")).strip(),
        str(settings.get("admin_password", "")).strip(),
    )

    @app.context_processor
    def inject_icp():
        return {"icp_record": ICP_RECORD}

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        next_url = clean_next_url(request.args.get("next"))
        register_allowed = registration_open(database_url, app.config["ALLOW_REGISTER"])
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
                        return redirect(next_url or url_for("index"))
        return render_template(
            "login.html",
            error=error,
            allow_register=register_allowed,
            next_url=next_url,
        )

    @app.route("/register", methods=["GET", "POST"])
    def register():
        error = None
        register_allowed = registration_open(database_url, app.config["ALLOW_REGISTER"])
        if not register_allowed:
            return redirect(url_for("login"))
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
                    return redirect(url_for("index"))
                except UserExistsError as exc:
                    error = str(exc)
                except Exception:
                    error = "注册失败，请稍后重试。"
        return render_template("register.html", error=error)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    def index():
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

    @app.route("/admin/config", methods=["GET", "POST"])
    @login_required
    def admin_config():
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

    @app.route("/points")
    @login_required
    def points():
        user_id = get_session_user_id()
        balance = get_user_points(database_url, user_id) if user_id else 0
        transactions = (
            list_points_transactions(database_url, user_id, limit=100) if user_id else []
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
        return render_template(
            "points.html",
            current_user=session.get("username"),
            balance=balance,
            transactions=transactions,
        )

    @app.route("/history")
    @login_required
    def history():
        user_id = get_session_user_id()
        records = list_analysis_records(database_url, user_id, limit=50) if user_id else []
        return render_template(
            "history.html",
            current_user=session.get("username"),
            records=records,
        )

    @app.route("/api/history/<run_id>")
    @login_required
    def api_history_detail(run_id: str):
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

    @app.route("/api/generate", methods=["POST"])
    @login_required
    def api_generate():
        require_dependencies()
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

        generate_lut_flag = request.form.get("generate_lut", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
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

    @app.route("/api/analyze_stream", methods=["POST"])
    @login_required
    def api_analyze_stream():
        require_dependencies()
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
            global DEBUG_REQUESTS_STATE
            previous_debug = DEBUG_REQUESTS_STATE
            DEBUG_REQUESTS_STATE = debug_requests or config.debug_requests
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
                DEBUG_REQUESTS_STATE = previous_debug

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.route("/api/download/<run_id>/<path:filename>")
    @login_required
    def api_download(run_id: str, filename: str):
        base_out_dir = Path(load_settings().get("out_dir", "outputs"))
        safe_dir = (base_out_dir / run_id).resolve()
        file_path = (safe_dir / filename).resolve()
        if safe_dir not in file_path.parents:
            return jsonify({"error": "非法路径。"}), 400
        if not file_path.exists():
            if filename.lower().endswith(".cube"):
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
        return send_from_directory(safe_dir, filename, as_attachment=True)

    return app


def main() -> None:
    if env_flag("CLI_MODE", "0"):
        run_cli()
    else:
        app = create_app()
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "7860"))

        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
