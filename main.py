from __future__ import annotations
import sys
import os
from app import create_app
from app.cli import run_cli
from app.config import env_flag

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
