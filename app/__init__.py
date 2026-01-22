from __future__ import annotations
from flask import Flask
from datetime import timedelta
import os
import uuid

from app.config import load_settings, ICP_RECORD
from app.models.db import init_db
from app.services.user import ensure_admin_user
from app.utils.common import require_dependencies
from app.routes import auth, main, api

def create_app() -> Flask:
    settings = load_settings()
    # Note: templates and static are in root, so we need to point to them
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
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

    app.register_blueprint(auth.bp)
    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp)

    return app
