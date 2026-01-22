from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "doubao_api_key": "",
    "analysis_model": "doubao-1.5-vision-pro-250328",
    "image_model": "doubao-seedream-4-0-250828",
    "image_path": "input.png",
    "out_dir": "outputs",
    "styles": "all",
    "no_lut": False,
    "sample_size": 128,
    "lut_size": 65,
    "lut_space": "rec709_sdr",
    "scene_type": "auto",
    "style_strength": 0.7,
    "retries": 5,
    "debug_requests": False,
    "database_url": "mysql://ai_lut:ai_lut_password@localhost:3306/ai_lut?charset=utf8mb4",
    "secret_key": "",
    "allow_register": True,
    "session_expire_hours": 12,
    "admin_username": "",
    "admin_password": "",
    "register_bonus_points": 0,
    "qiniu_access_key": "",
    "qiniu_secret_key": "",
    "qiniu_bucket": "",
    "qiniu_domain": "https://qn.3xx3x.cn",
    "grsai_api_key": "",
    "grsai_api_url": "https://grsai.dakka.com.cn/v1/draw/completions",
    "grsai_model": "gpt-image-1.5",
}

DEFAULT_LUT_SPACE = "rec709_sdr"
DEFAULT_SCENE_TYPE = "auto"
DEFAULT_STYLE_STRENGTH = 0.7

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
POINTS_REASON_LABELS = {
    POINTS_REASON_REGISTER: "注册赠送",
    POINTS_REASON_FILTER: "调色生成",
    POINTS_REASON_REFUND: "生成失败返还",
}
POINTS_SOURCE_LABELS = {
    POINTS_SOURCE_REGISTER: "注册",
    POINTS_SOURCE_FILTER: "生成",
}
MIN_PASSWORD_LENGTH = 8
MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 32


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
    grsai_api_key: str
    grsai_api_url: str
    grsai_model: str


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    settings = dict(DEFAULT_CONFIG)
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            settings.update(data)

    api_key_env = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("API_KEY")
    )
    if api_key_env:
        settings["api_key"] = api_key_env

    doubao_api_key_env = os.getenv("ARK_API_KEY")
    if doubao_api_key_env:
        settings["doubao_api_key"] = doubao_api_key_env

    database_url_env = os.getenv("DATABASE_URL")
    if database_url_env:
        settings["database_url"] = database_url_env

    secret_key_env = os.getenv("SECRET_KEY")
    if secret_key_env:
        settings["secret_key"] = secret_key_env

    allow_register_env = os.getenv("ALLOW_REGISTER")
    if allow_register_env is not None and allow_register_env != "":
        settings["allow_register"] = env_flag("ALLOW_REGISTER", "0")

    session_expire_env = os.getenv("SESSION_EXPIRE_HOURS")
    if session_expire_env:
        try:
            settings["session_expire_hours"] = int(session_expire_env)
        except ValueError:
            pass

    admin_username_env = os.getenv("ADMIN_USERNAME")
    if admin_username_env:
        settings["admin_username"] = admin_username_env

    admin_password_env = os.getenv("ADMIN_PASSWORD")
    if admin_password_env:
        settings["admin_password"] = admin_password_env

    register_bonus_env = os.getenv("REGISTER_BONUS_POINTS")
    if register_bonus_env:
        try:
            settings["register_bonus_points"] = int(register_bonus_env)
        except ValueError:
            pass

    qiniu_access_env = os.getenv("QINIU_ACCESS_KEY")
    if qiniu_access_env:
        settings["qiniu_access_key"] = qiniu_access_env
    qiniu_secret_env = os.getenv("QINIU_SECRET_KEY")
    if qiniu_secret_env:
        settings["qiniu_secret_key"] = qiniu_secret_env
    qiniu_bucket_env = os.getenv("QINIU_BUCKET")
    if qiniu_bucket_env:
        settings["qiniu_bucket"] = qiniu_bucket_env
    qiniu_domain_env = os.getenv("QINIU_DOMAIN")
    if qiniu_domain_env:
        settings["qiniu_domain"] = qiniu_domain_env

    grsai_api_key_env = os.getenv("GRSAI_API_KEY")
    if grsai_api_key_env:
        settings["grsai_api_key"] = grsai_api_key_env
    
    grsai_api_url_env = os.getenv("GRSAI_API_URL")
    if grsai_api_url_env:
        settings["grsai_api_url"] = grsai_api_url_env

    if not settings.get("api_key") and settings.get("doubao_api_key"):
        settings["api_key"] = settings["doubao_api_key"]
    if not settings.get("doubao_api_key") and settings.get("api_key"):
        settings["doubao_api_key"] = settings["api_key"]
    return settings


def save_settings(updates: Dict[str, Any], config_path: str | Path = "config.yaml") -> None:
    path = Path(config_path)
    settings: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            settings.update(data)
    settings.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, allow_unicode=True, sort_keys=False)
