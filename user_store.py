from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Optional
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
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id CHAR(32) PRIMARY KEY,
                        username VARCHAR(32) NOT NULL UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_login_at DATETIME NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
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
