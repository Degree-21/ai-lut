from __future__ import annotations

import sys

from config import load_settings
from user_store import init_db, upsert_settings


DB_SETTING_KEYS = ("analysis_model", "image_model", "api_key", "doubao_api_key")


def _mask_value(key: str, value: str) -> str:
    if "key" not in key or not value:
        return value
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def main() -> None:
    settings = load_settings()
    database_url = str(settings.get("database_url", "")).strip()
    if not database_url:
        print("Missing database_url in config.yaml or env.", file=sys.stderr)
        sys.exit(1)

    init_db(database_url)
    updates = {key: str(settings.get(key, "")) for key in DB_SETTING_KEYS}
    upsert_settings(database_url, updates)

    print("Migrated settings to database:")
    for key in DB_SETTING_KEYS:
        print(f"- {key}: {_mask_value(key, updates.get(key, ''))}")


if __name__ == "__main__":
    main()
