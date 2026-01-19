from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import aiomysql
import pymysql.err


class UserExistsError(Exception):
    pass


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("mysql+"):
        scheme, rest = database_url.split("://", 1)
        return f"mysql://{rest}"
    return database_url


def _parse_database_url(database_url: str) -> Dict[str, object]:
    normalized = _normalize_database_url(database_url)
    parsed = urlparse(normalized)
    if parsed.scheme != "mysql":
        raise ValueError("database_url 必须使用 mysql://")
    db_name = parsed.path.lstrip("/")
    if not db_name:
        raise ValueError("database_url 缺少数据库名")
    query = parse_qs(parsed.query)
    charset = (query.get("charset") or ["utf8mb4"])[0]
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "db": db_name,
        "charset": charset,
        "autocommit": True,
    }


def _run(coro):
    return asyncio.run(coro)


async def _open_conn(database_url: str) -> aiomysql.Connection:
    config = _parse_database_url(database_url)
    return await aiomysql.connect(**config)


async def _execute(
    database_url: str,
    query: str,
    args: tuple | None = None,
    *,
    fetchone: bool = False,
) -> Optional[Dict[str, object]]:
    conn = await _open_conn(database_url)
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, args or ())
            if fetchone:
                return await cur.fetchone()
    finally:
        conn.close()
    return None


async def _fetch_all(
    database_url: str, query: str, args: tuple | None = None
) -> List[Dict[str, object]]:
    conn = await _open_conn(database_url)
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, args or ())
            rows = await cur.fetchall()
            return list(rows or [])
    finally:
        conn.close()


async def _execute_scalar(database_url: str, query: str) -> int:
    conn = await _open_conn(database_url)
    try:
        async with conn.cursor() as cur:
            await cur.execute(query)
            row = await cur.fetchone()
            if not row:
                return 0
            return int(row[0])
    finally:
        conn.close()


def init_db(database_url: str) -> None:
    async def _init() -> None:
        conn = await _open_conn(database_url)
        try:
            async with conn.cursor() as cur:
                async def _column_exists(table: str, column: str) -> bool:
                    await cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = %s
                          AND COLUMN_NAME = %s
                        """,
                        (table, column),
                    )
                    row = await cur.fetchone()
                    return bool(row and int(row[0]) > 0)

                async def _index_exists(table: str, index: str) -> bool:
                    await cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM INFORMATION_SCHEMA.STATISTICS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = %s
                          AND INDEX_NAME = %s
                        """,
                        (table, index),
                    )
                    row = await cur.fetchone()
                    return bool(row and int(row[0]) > 0)

                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id CHAR(32) PRIMARY KEY,
                        username VARCHAR(32) NOT NULL UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        last_login_at DATETIME NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_settings (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(64) NOT NULL UNIQUE,
                        value TEXT NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_points (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_id CHAR(32) NOT NULL UNIQUE,
                        balance INT NOT NULL DEFAULT 0,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        CONSTRAINT fk_user_points_user
                            FOREIGN KEY (user_id) REFERENCES users(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS points_transactions (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_id CHAR(32) NOT NULL,
                        change_amount INT NOT NULL,
                        balance_after INT NOT NULL,
                        reason VARCHAR(64) NOT NULL,
                        source VARCHAR(64) NOT NULL,
                        note TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_points_transactions_user (user_id),
                        INDEX idx_points_transactions_created (created_at),
                        CONSTRAINT fk_points_transactions_user
                            FOREIGN KEY (user_id) REFERENCES users(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'app_settings'
                      AND COLUMN_NAME = 'id'
                    """
                )
                row = await cur.fetchone()
                needs_migration = bool(row and int(row[0]) == 0)
                if needs_migration:
                    await cur.execute("ALTER TABLE app_settings DROP PRIMARY KEY")
                    await cur.execute(
                        "ALTER TABLE app_settings "
                        "ADD COLUMN id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST"
                    )
                    await cur.execute(
                        "ALTER TABLE app_settings "
                        "ADD UNIQUE KEY uniq_app_settings_name (name)"
                    )
                if not await _column_exists("app_settings", "created_at"):
                    await cur.execute(
                        "ALTER TABLE app_settings "
                        "ADD COLUMN created_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP"
                    )
                if not await _column_exists("app_settings", "updated_at"):
                    await cur.execute(
                        "ALTER TABLE app_settings "
                        "ADD COLUMN updated_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
                    )

                if not await _column_exists("users", "updated_at"):
                    await cur.execute(
                        "ALTER TABLE users "
                        "ADD COLUMN updated_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP "
                        "AFTER created_at"
                    )

                if not await _column_exists("user_points", "id"):
                    if not await _index_exists("user_points", "uniq_user_points_user"):
                        await cur.execute(
                            "ALTER TABLE user_points "
                            "ADD UNIQUE KEY uniq_user_points_user (user_id)"
                        )
                    await cur.execute("ALTER TABLE user_points DROP PRIMARY KEY")
                    await cur.execute(
                        "ALTER TABLE user_points "
                        "ADD COLUMN id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST"
                    )
                if not await _column_exists("user_points", "created_at"):
                    await cur.execute(
                        "ALTER TABLE user_points "
                        "ADD COLUMN created_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP"
                    )
                if not await _column_exists("user_points", "updated_at"):
                    await cur.execute(
                        "ALTER TABLE user_points "
                        "ADD COLUMN updated_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
                    )

                if not await _column_exists("points_transactions", "created_at"):
                    await cur.execute(
                        "ALTER TABLE points_transactions "
                        "ADD COLUMN created_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP"
                    )
                if not await _column_exists("points_transactions", "updated_at"):
                    await cur.execute(
                        "ALTER TABLE points_transactions "
                        "ADD COLUMN updated_at DATETIME NOT NULL "
                        "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
                    )
        finally:
            conn.close()

    _run(_init())


def fetch_user_by_username(database_url: str, username: str) -> Optional[Dict[str, str]]:
    return _run(
        _execute(
            database_url,
            "SELECT id, username, password_hash FROM users WHERE username = %s",
            (username,),
            fetchone=True,
        )
    )


def create_user(database_url: str, username: str, password_hash: str) -> str:
    user_id = uuid.uuid4().hex

    async def _create() -> None:
        conn = await _open_conn(database_url)
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (id, username, password_hash) VALUES (%s, %s, %s)",
                    (user_id, username, password_hash),
                )
                await cur.execute(
                    "INSERT INTO user_points (user_id, balance) VALUES (%s, %s)",
                    (user_id, 0),
                )
        finally:
            conn.close()

    try:
        _run(_create())
    except pymysql.err.IntegrityError as exc:
        if exc.args and exc.args[0] == 1062:
            raise UserExistsError("用户名已存在") from exc
        raise
    return user_id


def count_users(database_url: str) -> int:
    return _run(_execute_scalar(database_url, "SELECT COUNT(*) FROM users"))


def mark_last_login(database_url: str, user_id: str) -> None:
    _run(
        _execute(
            database_url,
            "UPDATE users SET last_login_at = NOW() WHERE id = %s",
            (user_id,),
        )
    )


def fetch_settings(database_url: str, names: Iterable[str]) -> Dict[str, str]:
    names_list = [name for name in names if name]
    if not names_list:
        return {}
    placeholders = ", ".join(["%s"] * len(names_list))
    query = f"SELECT name, value FROM app_settings WHERE name IN ({placeholders})"
    rows = _run(_fetch_all(database_url, query, tuple(names_list)))
    return {str(row["name"]): str(row["value"]) for row in rows if row}


def upsert_settings(database_url: str, settings: Dict[str, str]) -> None:
    items = [(name, value) for name, value in settings.items()]
    if not items:
        return

    async def _upsert() -> None:
        conn = await _open_conn(database_url)
        try:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO app_settings (name, value)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE value = VALUES(value)
                    """,
                    items,
                )
        finally:
            conn.close()

    _run(_upsert())


def get_user_points(database_url: str, user_id: str) -> int:
    row = _run(
        _execute(
            database_url,
            "SELECT balance FROM user_points WHERE user_id = %s",
            (user_id,),
            fetchone=True,
        )
    )
    if not row:
        return 0
    return int(row.get("balance", 0))


def list_points_transactions(
    database_url: str, user_id: str, limit: int = 50
) -> List[Dict[str, object]]:
    safe_limit = max(1, min(int(limit), 200))
    query = (
        "SELECT id, change_amount, balance_after, reason, source, note, created_at "
        "FROM points_transactions WHERE user_id = %s "
        "ORDER BY id DESC LIMIT %s"
    )
    return _run(_fetch_all(database_url, query, (user_id, safe_limit)))


def apply_points_change(
    database_url: str,
    user_id: str,
    change_amount: int,
    *,
    reason: str,
    source: str,
    note: str | None = None,
) -> int:
    async def _apply() -> int:
        conn = await _open_conn(database_url)
        try:
            await conn.begin()
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "INSERT INTO user_points (user_id, balance) "
                    "VALUES (%s, %s) ON DUPLICATE KEY UPDATE balance = balance",
                    (user_id, 0),
                )
                await cur.execute(
                    "SELECT balance FROM user_points WHERE user_id = %s FOR UPDATE",
                    (user_id,),
                )
                row = await cur.fetchone()
                current = int(row["balance"]) if row else 0
                new_balance = current + int(change_amount)
                if new_balance < 0:
                    raise ValueError("积分不足，无法扣减。")
                await cur.execute(
                    "UPDATE user_points SET balance = %s WHERE user_id = %s",
                    (new_balance, user_id),
                )
                await cur.execute(
                    """
                    INSERT INTO points_transactions
                        (user_id, change_amount, balance_after, reason, source, note)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, change_amount, new_balance, reason, source, note),
                )
            await conn.commit()
            return new_balance
        except Exception:
            await conn.rollback()
            raise
        finally:
            conn.close()

    return _run(_apply())
