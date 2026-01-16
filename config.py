from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "image_path": "input.png",
    "out_dir": "outputs",
    "styles": "all",
    "no_lut": False,
    "sample_size": 128,
    "lut_size": 65,
    "retries": 5,
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
    return settings
