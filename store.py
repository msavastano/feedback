"""Postgres-backed persistence for the skill-memory agent.

Replaces the per-user filesystem tree under `data/users/<user_id>/` with four
tables (users, skills, sessions, turns). See `schema.sql` for the DDL.

All public callables are sync because the FastAPI handlers in `server.py` are
sync. A module-level `psycopg_pool.ConnectionPool` is created lazily on the
first call so that import-time side effects don't run without a `DATABASE_URL`.

The shape of each function mirrors the file-based version it replaces so the
swap in `agent.py` and `server.py` is a near 1:1 substitution.
"""

from __future__ import annotations

import json
import os
import re

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# ---------- pool ----------

_POOL: ConnectionPool | None = None


def _dsn() -> str:
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it in your environment "
            "(or in Vercel project settings) to point at Postgres."
        )
    return dsn


def get_pool() -> ConnectionPool:
    """Lazily build a small pool sized for serverless cold starts."""
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(
            conninfo=_dsn(),
            min_size=0,
            max_size=int(os.environ.get("DB_POOL_MAX", "4")),
            max_idle=30,
            kwargs={"autocommit": True},
        )
    return _POOL


def close_pool() -> None:
    """Tear down the pool. Useful in tests."""
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


# ---------- user store ----------

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _check_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
        raise ValueError(f"invalid user_id: {user_id!r}")
    return user_id


def _check_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not SESSION_ID_RE.match(session_id):
        raise ValueError(f"invalid session_id: {session_id!r}")
    return session_id


class UserStore:
    """Profile cache populated on Google login."""

    @staticmethod
    def upsert(profile: dict) -> dict:
        user_id = _check_user_id(profile["user_id"])
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO users (user_id, email, name, sub, picture)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        email      = EXCLUDED.email,
                        name       = EXCLUDED.name,
                        sub        = EXCLUDED.sub,
                        picture    = EXCLUDED.picture,
                        updated_at = now()
                    RETURNING user_id, email, name, sub, picture
                    """,
                    (
                        user_id,
                        profile.get("email", ""),
                        profile.get("name", ""),
                        profile.get("sub", ""),
                        profile.get("picture", ""),
                    ),
                )
                row = cur.fetchone()
                return dict(row) if row else {}

    @staticmethod
    def get(user_id: str) -> dict:
        _check_user_id(user_id)
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT user_id, email, name, sub, picture "
                    "FROM users WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else {}

    @staticmethod
    def list_user_ids() -> list[str]:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users ORDER BY user_id")
                return [r[0] for r in cur.fetchall()]


# ---------- skill store ----------

# Row shape returned by load_all: dict with tier/name/description/body keys.

class SkillStore:
    """All skills for a user. Mirrors the old `data/users/<uid>/skills/` tree."""

    @staticmethod
    def load_all(user_id: str) -> list[dict]:
        """Return every skill for a user as dicts ordered (system, active, archive)
        then by name. Matches the order produced by the old `load_skills`."""
        _check_user_id(user_id)
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT tier, name, description, body
                    FROM skills
                    WHERE user_id = %s
                    ORDER BY
                      CASE tier
                        WHEN 'system'  THEN 0
                        WHEN 'active'  THEN 1
                        WHEN 'archive' THEN 2
                        ELSE 3
                      END,
                      name
                    """,
                    (user_id,),
                )
                return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def upsert(
        user_id: str, name: str, description: str, body: str, tier: str = "active"
    ) -> str:
        """Insert-or-update a skill. Returns the canonical name."""
        _check_user_id(user_id)
        if tier not in ("system", "active", "archive"):
            tier = "active"
        with get_pool().connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Snapshot the pre-image so a lossy LLM merge is recoverable.
                    # Skipped when the body is unchanged (no-op rewrites don't
                    # spam versions).
                    cur.execute(
                        """
                        INSERT INTO skill_versions
                            (user_id, name, tier, description, body, op)
                        SELECT user_id, name, tier, description, body, 'update'
                        FROM skills
                        WHERE user_id = %s AND name = %s
                          AND body IS DISTINCT FROM %s
                        """,
                        (user_id, name, body),
                    )
                    cur.execute(
                        """
                        INSERT INTO skills (user_id, tier, name, description, body)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, name) DO UPDATE SET
                            tier        = EXCLUDED.tier,
                            description = EXCLUDED.description,
                            body        = EXCLUDED.body,
                            updated_at  = now()
                        """,
                        (user_id, tier, name, description, body),
                    )
        return name

    @staticmethod
    def delete(user_id: str, name: str) -> bool:
        _check_user_id(user_id)
        with get_pool().connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO skill_versions
                            (user_id, name, tier, description, body, op)
                        SELECT user_id, name, tier, description, body, 'delete'
                        FROM skills
                        WHERE user_id = %s AND name = %s
                        """,
                        (user_id, name),
                    )
                    cur.execute(
                        "DELETE FROM skills WHERE user_id = %s AND name = %s",
                        (user_id, name),
                    )
                    return cur.rowcount > 0


# ---------- session store ----------

class SessionStore:
    """Sessions + turns. Replaces `sessions/<sid>.jsonl`."""

    @staticmethod
    def create(user_id: str, session_id: str) -> None:
        _check_user_id(user_id)
        _check_session_id(session_id)
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (user_id, session_id)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id, session_id) DO NOTHING
                    """,
                    (user_id, session_id),
                )

    @staticmethod
    def append_turn(
        user_id: str,
        session_id: str,
        role: str,
        text: str,
        tokens: dict | None = None,
    ) -> int:
        """Append a turn. Returns the assigned `idx`."""
        _check_user_id(user_id)
        _check_session_id(session_id)
        if role not in ("user", "model"):
            raise ValueError(f"invalid role: {role!r}")
        with get_pool().connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Ensure session row exists. A new turn invalidates any
                    # prior rollup: the session's captured-in-memory snapshot is
                    # now stale, so clear rolled_up. This lets a returned-to chat
                    # be re-summarized on the next end. (No-op on a fresh session
                    # where rolled_up is already FALSE.)
                    cur.execute(
                        """
                        INSERT INTO sessions (user_id, session_id)
                        VALUES (%s, %s)
                        ON CONFLICT (user_id, session_id)
                        DO UPDATE SET rolled_up = FALSE
                        WHERE sessions.rolled_up
                        """,
                        (user_id, session_id),
                    )
                    cur.execute(
                        """
                        INSERT INTO turns (user_id, session_id, idx, role, text, tokens)
                        VALUES (
                            %s, %s,
                            COALESCE(
                                (SELECT MAX(idx)+1 FROM turns
                                 WHERE user_id=%s AND session_id=%s),
                                0
                            ),
                            %s, %s, %s
                        )
                        RETURNING idx
                        """,
                        (
                            user_id,
                            session_id,
                            user_id,
                            session_id,
                            role,
                            text,
                            json.dumps(tokens) if tokens else None,
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0

    @staticmethod
    def load_turns(user_id: str, session_id: str) -> list[dict]:
        """Return ordered turns as [{role, text, ts, tokens}]."""
        _check_user_id(user_id)
        _check_session_id(session_id)
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT role, text, ts, tokens
                    FROM turns
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY idx ASC
                    """,
                    (user_id, session_id),
                )
                return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def list_sessions(user_id: str) -> list[dict]:
        """Return summary rows matching the old `list_sessions` shape."""
        _check_user_id(user_id)
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        s.session_id,
                        s.rolled_up,
                        COUNT(t.idx) FILTER (WHERE t.role IN ('user','model')) AS turns,
                        MAX(t.ts)    FILTER (WHERE t.role IN ('user','model')) AS last_ts,
                        (
                            SELECT t2.text
                            FROM turns t2
                            WHERE t2.user_id = s.user_id
                              AND t2.session_id = s.session_id
                              AND t2.role = 'user'
                            ORDER BY t2.idx ASC
                            LIMIT 1
                        ) AS first_message
                    FROM sessions s
                    LEFT JOIN turns t
                      ON t.user_id = s.user_id AND t.session_id = s.session_id
                    WHERE s.user_id = %s
                    GROUP BY s.user_id, s.session_id, s.created_at, s.rolled_up
                    ORDER BY s.created_at DESC
                    """,
                    (user_id,),
                )
                out: list[dict] = []
                for r in cur.fetchall():
                    first_msg = (r.get("first_message") or "")[:80]
                    last_ts = r.get("last_ts")
                    out.append(
                        {
                            "session_id": r["session_id"],
                            "first_message": first_msg,
                            "last_ts": last_ts.isoformat() if last_ts else "",
                            "turns": int(r.get("turns") or 0),
                            "rolled_up": bool(r.get("rolled_up")),
                        }
                    )
                return out

    @staticmethod
    def is_rolled_up(user_id: str, session_id: str) -> bool:
        _check_user_id(user_id)
        _check_session_id(session_id)
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT rolled_up FROM sessions "
                    "WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                row = cur.fetchone()
                return bool(row and row[0])

    @staticmethod
    def rollup_skill(user_id: str, session_id: str) -> str | None:
        """Name of the skill this session was last rolled up into, or None.
        Lets a re-summarize overwrite the same skill instead of minting a dupe."""
        _check_user_id(user_id)
        _check_session_id(session_id)
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT rollup_skill FROM sessions "
                    "WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
                row = cur.fetchone()
                return row[0] if row and row[0] else None

    @staticmethod
    def mark_rolled_up(user_id: str, session_id: str, skill_name: str) -> None:
        _check_user_id(user_id)
        _check_session_id(session_id)
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions
                    SET rolled_up = TRUE, ended_at = now(), rollup_skill = %s
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (skill_name, user_id, session_id),
                )
