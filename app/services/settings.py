from __future__ import annotations
from typing import Dict
from app.config import load_settings, DB_SETTING_KEYS
from app.models.db import fetch_settings

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
