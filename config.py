from __future__ import annotations

import os
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
    "retries": 5,
    "debug_requests": False,
}


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

    if not settings.get("api_key") and settings.get("doubao_api_key"):
        settings["api_key"] = settings["doubao_api_key"]
    if not settings.get("doubao_api_key") and settings.get("api_key"):
        settings["doubao_api_key"] = settings["api_key"]
    return settings
