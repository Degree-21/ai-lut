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
    conn = await aiomysql.connect(**config)
    try:
        await conn.set_charset("utf8mb4")
    except AttributeError:
        async with conn.cursor() as cur:
            await cur.execute("SET NAMES utf8mb4")
    return conn


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
                async def _table_exists(table: str) -> bool:
                    await cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = %s
                        """,
                        (table,),
                    )
                    row = await cur.fetchone()
                    return bool(row and int(row[0]) > 0)

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

                async def _column_type(table: str, column: str) -> str | None:
                    await cur.execute(
                        """
                        SELECT DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = %s
                          AND COLUMN_NAME = %s
                        """,
                        (table, column),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return str(row[0]).lower()

                def _is_int_type(value: str | None) -> bool:
                    return value in {"int", "bigint", "mediumint", "smallint", "tinyint"}

                async def _guard_schema() -> None:
                    if await _table_exists("users"):
                        id_type = await _column_type("users", "id")
                        has_uuid = await _column_exists("users", "user_uuid")
                        if not _is_int_type(id_type) or not has_uuid:
                            raise RuntimeError(
                                "检测到旧用户表结构。由于无需迁移，请删除旧表后重启。"
                            )

                    tables = [
                        ("app_settings", None),
                        ("user_points", "user_id"),
                        ("points_transactions", "user_id"),
                        ("analysis_records", "user_id"),
                        ("analysis_images", "user_id"),
                        ("analysis_luts", "user_id"),
                    ]
                    for table, user_col in tables:
                        if not await _table_exists(table):
                            continue
                        id_type = await _column_type(table, "id")
                        if not _is_int_type(id_type):
                            raise RuntimeError(
                                f"检测到旧表 {table} 结构。由于无需迁移，请删除旧表后重启。"
                            )
                        if user_col:
                            user_type = await _column_type(table, user_col)
                            if not _is_int_type(user_type):
                                raise RuntimeError(
                                    f"检测到旧表 {table} 结构。由于无需迁移，请删除旧表后重启。"
                                )

                await _guard_schema()

                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_uuid CHAR(32) NOT NULL UNIQUE,
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
                        user_id BIGINT NOT NULL UNIQUE,
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
                        user_id BIGINT NOT NULL,
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
                    CREATE TABLE IF NOT EXISTS analysis_records (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        run_id VARCHAR(64) NOT NULL UNIQUE,
                        style_ids TEXT NOT NULL,
                        cost INT NOT NULL,
                        source_filename VARCHAR(255) NOT NULL,
                        source_url TEXT NULL,
                        analysis_text MEDIUMTEXT NOT NULL,
                        analysis_url TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_analysis_records_user (user_id),
                        INDEX idx_analysis_records_created (created_at),
                        CONSTRAINT fk_analysis_records_user
                            FOREIGN KEY (user_id) REFERENCES users(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_images (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        run_id VARCHAR(64) NOT NULL,
                        style_id VARCHAR(64) NOT NULL,
                        image_filename VARCHAR(255) NOT NULL,
                        image_url TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_analysis_images_user (user_id),
                        INDEX idx_analysis_images_run (run_id),
                        INDEX idx_analysis_images_style (style_id),
                        UNIQUE KEY uniq_analysis_images_filename (run_id, image_filename),
                        CONSTRAINT fk_analysis_images_user
                            FOREIGN KEY (user_id) REFERENCES users(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_luts (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        run_id VARCHAR(64) NOT NULL,
                        style_id VARCHAR(64) NOT NULL,
                        lut_space VARCHAR(32) NOT NULL,
                        lut_size INT NOT NULL,
                        lut_filename VARCHAR(255) NOT NULL,
                        lut_content MEDIUMTEXT NOT NULL,
                        lut_url TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_analysis_luts_user (user_id),
                        INDEX idx_analysis_luts_run (run_id),
                        INDEX idx_analysis_luts_style (style_id),
                        UNIQUE KEY uniq_analysis_luts_filename (run_id, lut_filename),
                        CONSTRAINT fk_analysis_luts_user
                            FOREIGN KEY (user_id) REFERENCES users(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                )
                await cur.execute(
                    """
                    INSERT INTO user_points (user_id, balance)
                    SELECT u.id, 0
                    FROM users u
                    LEFT JOIN user_points up ON up.user_id = u.id
                    WHERE up.user_id IS NULL
                    """
                )
        finally:
            conn.close()

    _run(_init())


def fetch_user_by_username(database_url: str, username: str) -> Optional[Dict[str, object]]:
    return _run(
        _execute(
            database_url,
            "SELECT id, username, password_hash FROM users WHERE username = %s",
            (username,),
            fetchone=True,
        )
    )


def create_user(database_url: str, username: str, password_hash: str) -> int:
    user_uuid = uuid.uuid4().hex
    user_id = 0

    async def _create() -> None:
        nonlocal user_id
        conn = await _open_conn(database_url)
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (user_uuid, username, password_hash) VALUES (%s, %s, %s)",
                    (user_uuid, username, password_hash),
                )
                user_id = int(cur.lastrowid)
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


def mark_last_login(database_url: str, user_id: int) -> None:
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


def get_user_points(database_url: str, user_id: int) -> int:
    _run(
        _execute(
            database_url,
            "INSERT INTO user_points (user_id, balance) "
            "VALUES (%s, %s) ON DUPLICATE KEY UPDATE balance = balance",
            (user_id, 0),
        )
    )
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
    database_url: str,
    user_id: int,
    limit: int = 50,
    run_id: str | None = None,
) -> List[Dict[str, object]]:
    safe_limit = max(1, min(int(limit), 200))
    params: List[object] = [user_id]
    query = (
        "SELECT id, change_amount, balance_after, reason, source, note, created_at "
        "FROM points_transactions WHERE user_id = %s"
    )
    if run_id:
        query += " AND note LIKE %s"
        params.append(f"%run_id={run_id}%")
    query += " ORDER BY id DESC LIMIT %s"
    params.append(safe_limit)
    return _run(_fetch_all(database_url, query, tuple(params)))


def apply_points_change(
    database_url: str,
    user_id: int,
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


def create_analysis_record(
    database_url: str,
    *,
    user_id: int,
    run_id: str,
    style_ids: List[str],
    cost: int,
    source_filename: str,
    source_url: str,
    analysis_text: str,
    analysis_url: str,
) -> None:
    _run(
        _execute(
            database_url,
            """
            INSERT INTO analysis_records
                (user_id, run_id, style_ids, cost, source_filename, source_url, analysis_text, analysis_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                run_id,
                ",".join(style_ids),
                cost,
                source_filename,
                source_url,
                analysis_text,
                analysis_url,
            ),
        )
    )


def update_analysis_record(
    database_url: str,
    *,
    user_id: int,
    run_id: str,
    style_ids: List[str],
    cost: int,
    source_filename: str,
    source_url: str,
    analysis_text: str,
    analysis_url: str,
) -> None:
    _run(
        _execute(
            database_url,
            """
            UPDATE analysis_records
            SET style_ids = %s,
                cost = %s,
                source_filename = %s,
                source_url = %s,
                analysis_text = %s,
                analysis_url = %s
            WHERE user_id = %s AND run_id = %s
            """,
            (
                ",".join(style_ids),
                cost,
                source_filename,
                source_url,
                analysis_text,
                analysis_url,
                user_id,
                run_id,
            ),
        )
    )


def fetch_analysis_record(
    database_url: str, user_id: int, run_id: str
) -> Optional[Dict[str, object]]:
    return _run(
        _execute(
            database_url,
            """
            SELECT id, run_id, style_ids, cost, source_filename, source_url,
                   analysis_text, analysis_url, created_at
            FROM analysis_records
            WHERE user_id = %s AND run_id = %s
            """,
            (user_id, run_id),
            fetchone=True,
        )
    )


def create_image_record(
    database_url: str,
    *,
    user_id: int,
    run_id: str,
    style_id: str,
    image_filename: str,
    image_url: str | None,
) -> None:
    _run(
        _execute(
            database_url,
            """
            INSERT INTO analysis_images
                (user_id, run_id, style_id, image_filename, image_url)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                image_url = VALUES(image_url)
            """,
            (user_id, run_id, style_id, image_filename, image_url),
        )
    )


def create_lut_record(
    database_url: str,
    *,
    user_id: int,
    run_id: str,
    style_id: str,
    lut_space: str,
    lut_size: int,
    lut_filename: str,
    lut_content: str,
    lut_url: str | None,
) -> None:
    _run(
        _execute(
            database_url,
            """
            INSERT INTO analysis_luts
                (user_id, run_id, style_id, lut_space, lut_size, lut_filename, lut_content, lut_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) AS new
            ON DUPLICATE KEY UPDATE
                lut_content = new.lut_content,
                lut_url = new.lut_url
            """,
            (
                user_id,
                run_id,
                style_id,
                lut_space,
                lut_size,
                lut_filename,
                lut_content,
                lut_url,
            ),
        )
    )


def list_image_records(
    database_url: str, user_id: int, run_id: str
) -> List[Dict[str, object]]:
    query = (
        "SELECT style_id, image_filename, image_url "
        "FROM analysis_images WHERE user_id = %s AND run_id = %s "
        "ORDER BY id ASC"
    )
    return _run(_fetch_all(database_url, query, (user_id, run_id)))


def list_lut_records(
    database_url: str, user_id: int, run_id: str
) -> List[Dict[str, object]]:
    query = (
        "SELECT style_id, lut_filename, lut_url "
        "FROM analysis_luts WHERE user_id = %s AND run_id = %s "
        "ORDER BY id ASC"
    )
    return _run(_fetch_all(database_url, query, (user_id, run_id)))


def fetch_lut_content(
    database_url: str,
    *,
    user_id: int,
    run_id: str,
    lut_filename: str,
) -> Optional[str]:
    row = _run(
        _execute(
            database_url,
            """
            SELECT lut_content
            FROM analysis_luts
            WHERE user_id = %s AND run_id = %s AND lut_filename = %s
            """,
            (user_id, run_id, lut_filename),
            fetchone=True,
        )
    )
    if not row:
        return None
    return str(row.get("lut_content") or "")


def list_analysis_records(
    database_url: str, user_id: int, limit: int = 50
) -> List[Dict[str, object]]:
    safe_limit = max(1, min(int(limit), 200))
    query = (
        "SELECT id, run_id, style_ids, cost, source_filename, source_url, "
        "analysis_text, analysis_url, created_at "
        "FROM analysis_records WHERE user_id = %s "
        "ORDER BY id DESC LIMIT %s"
    )
    return _run(_fetch_all(database_url, query, (user_id, safe_limit)))
