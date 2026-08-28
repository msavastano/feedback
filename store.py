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
import math
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

def normalize_skill_name(name: str) -> str:
    """Normalize a skill name into a clean, lowercase slug identifier."""
    if not name:
        return name
    s = re.sub(r'[^a-zA-Z0-9_-]+', '-', name).strip('-').lower()
    return s or name

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
    def search_bodies(
        user_id: str,
        terms: list[str],
        limit: int = 3,
        min_rank: float = 0.0,
    ) -> list[tuple[str, float]]:
        """(name, ts_rank) for skills matching ANY of `terms`, best match first.

        Fallback for the picker, which only ever sees names and descriptions: a
        fact filed inside a skill about something else is invisible to it, and
        the description becomes a single point of failure for that fact. This
        reads the bodies. System tier is excluded — it loads unconditionally.

        `min_rank` drops weak hits. The query ORs every term, so a match on one
        incidental word is enough to surface a skill about a completely
        different subject; without a floor those ride into the prompt on every
        turn. Sampled on a real store, genuine topic matches score ~0.03-0.09
        and pure noise ~0.009-0.02, so the caller's default sits in the gap.
        Note ts_rank here is unnormalised (no length divisor), so scores drift
        up as bodies grow — re-sample if the floor starts letting junk through.

        # ponytail: seq scan with the tsvector computed per row. A user's
        # catalog is a few dozen small rows, so this is cheaper than a stored
        # tsv column + GIN index + migration. Add those if a catalog grows
        # enough to feel it.
        """
        _check_user_id(user_id)
        if not terms:
            return []
        query = " | ".join(terms)
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, ts_rank(tsv, q) AS rank
                    FROM skills,
                         to_tsquery('english', %s) AS q,
                         to_tsvector('english',
                             name || ' ' || description || ' ' || body) AS tsv
                    WHERE user_id = %s AND tier <> 'system' AND tsv @@ q
                      AND ts_rank(tsv, q) >= %s
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, user_id, min_rank, limit),
                )
                return [(r[0], float(r[1])) for r in cur.fetchall()]

    @staticmethod
    def upsert(
        user_id: str, name: str, description: str, body: str, tier: str = "active"
    ) -> str:
        """Insert-or-update a skill. Returns the canonical name."""
        _check_user_id(user_id)
        name = normalize_skill_name(name)
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
        name = normalize_skill_name(name)
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
        picked: dict | None = None,
    ) -> int:
        """Append a turn. Returns the assigned `idx`.

        `picked` is the skill picker's scores for this turn (model rows only);
        see the `turns.picked` comment in schema.sql for the shape.
        """
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
                        INSERT INTO turns
                            (user_id, session_id, idx, role, text, tokens, picked)
                        VALUES (
                            %s, %s,
                            COALESCE(
                                (SELECT MAX(idx)+1 FROM turns
                                 WHERE user_id=%s AND session_id=%s),
                                0
                            ),
                            %s, %s, %s, %s
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
                            json.dumps(picked) if picked else None,
                        ),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0

    @staticmethod
    def load_turns(user_id: str, session_id: str) -> list[dict]:
        """Return ordered turns as [{role, text, ts, tokens, picked}]."""
        _check_user_id(user_id)
        _check_session_id(session_id)
        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT role, text, ts, tokens, picked
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
    def stale_unrolled(
        user_id: str,
        idle_minutes: int,
        exclude_session_id: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Sessions with real turns that were never rolled up and have gone quiet.

        A session only becomes memory when something calls
        `summarize_session_to_skill`, and on the web that depends on the browser
        firing its pagehide beacon. A closed laptop, a crashed tab, or a lost
        network drops the session on the floor: its turns stay in `turns`, which
        no retrieval path reads, so the conversation is unrecallable forever.
        This finds those so a sweep can roll them up late.

        `idle_minutes` keeps a session the user is still using out of the set;
        `exclude_session_id` protects the caller's own live session. `limit`
        bounds the work, since each hit costs one LLM summarization — the
        remainder is picked up by the next sweep.
        """
        _check_user_id(user_id)
        if exclude_session_id is not None:
            _check_session_id(exclude_session_id)
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.session_id
                    FROM sessions s
                    JOIN turns t
                      ON t.user_id = s.user_id AND t.session_id = s.session_id
                    WHERE s.user_id = %s
                      AND s.rolled_up = FALSE
                      AND (%s::text IS NULL OR s.session_id <> %s)
                    GROUP BY s.session_id
                    HAVING count(t.idx) >= 2
                       AND max(t.ts) < now() - make_interval(mins => %s)
                    ORDER BY max(t.ts)
                    LIMIT %s
                    """,
                    (
                        user_id,
                        exclude_session_id,
                        exclude_session_id,
                        idle_minutes,
                        limit,
                    ),
                )
                return [r[0] for r in cur.fetchall()]
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


# ---------- skill graph ----------

# `session-*` / `sessions-archive-*` are *containers*, not subjects: a rollup
# holds one conversation and an archive holds dozens of unrelated ones. They
# therefore relate to everything, on either signal — co-activation (the picker
# names them alongside whatever topic the session covered) and content overlap
# (their bodies literally contain every other skill's vocabulary). Left in, a
# single archive node becomes a hub wiring unrelated subjects together, which
# is most of what made the map's connections look arbitrary. They stay as
# nodes; they are just never edge endpoints.
_CONTAINER_RE = re.compile(r"^(session-|sessions-archive-)")


def _is_container(name: str) -> bool:
    return bool(_CONTAINER_RE.match(name))


COACT_MIN_WEIGHT = 2    # one shared turn is a coincidence, not a relation

# Content-similarity tuning. Similarity is idf-weighted cosine over the token
# set of `name + description + body`, then thinned to a mutual k-nearest-
# neighbour graph so one big skill can't cable itself to half the catalog.
# Sampled against a real 54-skill store, the same way FALLBACK_MIN_RANK was:
# genuine subject pairs ran 0.084-0.370 down to `interest-books` /
# `interest-books-children-of-time`, and clear junk (`topic-ai-ipo-landscape` /
# `topic-fifa-world-cup-2026-odds`, `image-generations` /
# `interest-books-children-of-time`) sat at 0.074 and below, so the floor goes
# in the gap. Unlike ts_rank, cosine is length-normalised, so this does not
# drift upward as bodies grow — but re-sample if junk starts appearing.
SIM_MIN = 0.08
SIM_K = 3               # max content edges per skill
SIM_MIN_TOKEN = 3       # shorter tokens are almost all stopwords
# Deliberately small: idf already suppresses anything common across the store,
# so this only has to catch words too frequent to be distinctive but too rare
# for idf to flatten in a ~100-document corpus.
_SIM_STOP = frozenset("""
the and for that with this from was were are you your they them their there here
what when where which who whom how why not but all any can could will would should
may might must have has had does did done its it's about into over under more most
some such than then also very much many other same each both few own too own via
user asked said says note see using used use like just get got new one two three
""".split())

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


def _sim_tokens(text: str) -> set[str]:
    """Token set for similarity: lowercase words, hyphens split, stopwords out."""
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        for part in tok.split("-"):
            if len(part) >= SIM_MIN_TOKEN and part not in _SIM_STOP:
                out.add(part)
    return out


def _content_edges(docs: dict[str, set[str]]) -> list[dict]:
    """Mutual-kNN content-similarity edges over skill token sets.

    Why this exists at all: co-activation only ever describes skills the picker
    has actually named together, which on a young store is a handful of them —
    24 of 101 on the store this was built against, leaving three quarters of
    the map as unconnected dots whose position means nothing. Body overlap
    needs no pick history, so the taxonomy the flat namespace can't express
    (`interest-sports-rules-fifa` / `-fifa-subs` / `-fifa-tie-breakers`) draws
    itself. No LLM call and no embedding service: idf-weighted cosine over
    token sets, which is cheap enough to run per request.
    """
    names = sorted(docs)
    n_docs = len(names)
    if n_docs < 2:
        return []
    df: dict[str, int] = {}
    for toks in docs.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1
    # Smoothed idf. Plain log(N/(1+df)) hits exactly zero when a token is in
    # half the corpus, which on a small store silently zeroes every shared
    # token and returns no edges at all; the +1 keeps the weight positive
    # while preserving the ranking on a full-sized catalog.
    idf = {t: math.log(1 + n_docs / (1 + c)) for t, c in df.items()}
    # Precompute norms once; the pair loop is O(n^2) and does enough work already.
    norm = {
        name: math.sqrt(sum(idf[t] * idf[t] for t in toks)) or 1.0
        for name, toks in docs.items()
    }
    # Only pairs sharing a rare-ish token can clear SIM_MIN, so invert the token
    # index and score those instead of all n^2 pairs.
    postings: dict[str, list[str]] = {}
    for name, toks in docs.items():
        for t in toks:
            postings.setdefault(t, []).append(name)
    scored: dict[tuple[str, str], float] = {}
    for t, holders in postings.items():
        # A token in most of the catalog carries no signal and would make this
        # loop quadratic for nothing. The floor of 4 keeps a small store from
        # having every shared token cut as "too common".
        if len(holders) < 2 or len(holders) > max(4, n_docs * 0.5):
            continue
        w = idf[t] * idf[t]
        for i, a in enumerate(holders):
            for b in holders[i + 1:]:
                key = (a, b) if a < b else (b, a)
                scored[key] = scored.get(key, 0.0) + w
    pairs = []
    for (a, b), acc in scored.items():
        s = acc / (norm[a] * norm[b])
        if s >= SIM_MIN:
            pairs.append((s, a, b))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    kept: dict[str, int] = {}
    out = []
    for s, a, b in pairs:
        if kept.get(a, 0) >= SIM_K or kept.get(b, 0) >= SIM_K:
            continue
        kept[a] = kept.get(a, 0) + 1
        kept[b] = kept.get(b, 0) + 1
        out.append(
            {"source": a, "target": b, "kind": "content", "strength": round(s, 3)}
        )
    return out


def _build_graph(
    catalog: list[dict],
    picks: list[dict],
    edges: list[dict],
    bodies: dict[str, str] | None = None,
    turns: int = 0,
) -> dict:
    """Join the skill catalog with retrieval stats into a node/edge graph.

    Pure so it can be tested without a database. Anything naming a skill that is
    no longer in the catalog is dropped: `consolidate()` deletes the `session-*`
    skills it folds into archive, so old `turns.picked` rows routinely reference
    skills that no longer exist, and the graph is a picture of current memory.

    Two kinds of edge, and they answer different questions. `content` is what
    the memory is *about* — idf-weighted body overlap, available for every
    skill from the moment it is written. `coactivation` is what retrieval
    actually *did* — skills the picker named in the same turn.

    Co-activation is gated at `COACT_MIN_WEIGHT`. Raw co-occurrence was the
    only relation this graph had, and on real stores two thirds of those edges
    came from a single shared turn: a coincidence drawn as a permanent line
    (`interest-diesel-crack-spread — interest-nest-wifi-pro-vs-tplink-m4` is a
    real example). Each surviving edge also carries a Jaccard `strength`, so a
    pair that is always retrieved together is distinguishable from one that
    overlapped twice out of thirty — raw counts alone just favour whichever
    skill gets picked most.
    """
    stats = {p["name"]: p for p in picks}
    nodes = []
    for row in catalog:
        p = stats.get(row["name"], {})
        avg = p.get("avg_score")
        last = p.get("last_picked")
        updated = row.get("updated_at")
        nodes.append(
            {
                "name": row["name"],
                "tier": row["tier"],
                "description": row.get("description", ""),
                "chars": row.get("chars", 0),
                # System skills bypass the picker entirely (auto-included, never
                # scored), so a pick count on them would read as "never used".
                "always": row["tier"] == "system",
                "picks": int(p.get("picks", 0)),
                "avg_score": None if avg is None else round(float(avg), 2),
                "last_picked": last.isoformat() if hasattr(last, "isoformat") else last,
                "updated_at": updated.isoformat() if hasattr(updated, "isoformat") else updated,
            }
        )
    # System skills bypass the picker, so they can never be a co-activation
    # endpoint; containers are excluded for the reason at _CONTAINER_RE.
    linkable = {
        n["name"] for n in nodes if not n["always"] and not _is_container(n["name"])
    }
    freq = {p["name"]: int(p["picks"]) for p in picks}

    out_edges = []
    for e in edges:
        a, b = e["src"], e["dst"]
        if a not in linkable or b not in linkable:
            continue
        w = int(e["weight"])
        if w < COACT_MIN_WEIGHT:
            continue
        union = freq.get(a, w) + freq.get(b, w) - w
        out_edges.append(
            {
                "source": a,
                "target": b,
                "kind": "coactivation",
                "weight": w,
                # Jaccard: of the turns that used either skill, the share that
                # used both. Unlike the raw count this does not simply rank the
                # busiest skills first.
                "strength": round(w / union, 3) if union > 0 else 1.0,
            }
        )
    out_edges.sort(key=lambda e: (-e["strength"], e["source"], e["target"]))

    if bodies:
        docs = {}
        for n in nodes:
            name = n["name"]
            if name not in linkable:
                continue
            toks = _sim_tokens(
                f"{name} {n.get('description') or ''} {bodies.get(name) or ''}"
            )
            if toks:
                docs[name] = toks
        seen = {(e["source"], e["target"]) for e in out_edges}
        out_edges += [
            e
            for e in _content_edges(docs)
            if (e["source"], e["target"]) not in seen
        ]

    return {"nodes": nodes, "edges": out_edges, "turns": turns}


def skill_graph(user_id: str) -> dict:
    """Skill catalog + retrieval stats + content/co-activation edges for the map.

    Co-activation reads `turns.picked` (written by the pick stage, never read
    back by the agent itself); turns without a `picked` map -- user rows and
    the image-generation fast path -- contribute nothing. Content edges read
    the bodies, so they exist for skills the picker has never touched.
    """
    _check_user_id(user_id)
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # Bodies come back whole for the similarity pass. That is ~130 KB
            # on a mature store, and only on a Map open, so it is not worth a
            # second round trip or a server-side tsvector to avoid.
            cur.execute(
                """
                SELECT name, tier, description, body, length(body) AS chars,
                       updated_at
                FROM skills WHERE user_id = %s
                ORDER BY name
                """,
                (user_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            bodies = {r["name"]: r.pop("body") or "" for r in rows}
            catalog = rows

            cur.execute(
                """
                SELECT k.key AS name, count(*) AS picks,
                       avg((k.value)::numeric) AS avg_score, max(t.ts) AS last_picked
                FROM turns t, jsonb_each_text(t.picked->'scores') k
                WHERE t.user_id = %s AND t.picked IS NOT NULL
                GROUP BY 1
                """,
                (user_id,),
            )
            picks = [dict(r) for r in cur.fetchall()]

            # a.key < b.key dedupes each pair and drops self-edges.
            cur.execute(
                """
                SELECT a.key AS src, b.key AS dst, count(*) AS weight
                FROM turns t,
                     jsonb_object_keys(t.picked->'scores') AS a(key),
                     jsonb_object_keys(t.picked->'scores') AS b(key)
                WHERE t.user_id = %s AND t.picked IS NOT NULL AND a.key < b.key
                GROUP BY 1, 2
                """,
                (user_id,),
            )
            edges = [dict(r) for r in cur.fetchall()]

            # `picked` was added to turns after this app had real history, so a
            # zero pick count means "not picked since scores started", not
            # "never retrieved". Ship the window so the UI can say which.
            cur.execute(
                """
                SELECT count(*) FILTER (WHERE picked IS NOT NULL) AS scored,
                       count(*) FILTER (WHERE role = 'model')     AS model_turns,
                       min(ts)  FILTER (WHERE picked IS NOT NULL) AS since
                FROM turns WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()

    graph = _build_graph(catalog, picks, edges, bodies=bodies, turns=int(row["scored"]))
    graph["model_turns"] = int(row["model_turns"])
    graph["since"] = row["since"].isoformat() if row["since"] else None
    return graph
