"""Self-editing skill-memory agent on Gemini.

Multi-user. Per-user state lives in Postgres (see store.py + schema.sql).

Flow per turn:
  1. Read every skill (name+description+body+tier) for the user.
  2. Ask model which skills are relevant to the prompt.
  3. Load full bodies (active) + section excerpts (archive) + all system skills.
  4. Generate response (web search tool available).
  5. Reflect: model returns skill edits across system/active/archive tiers.
  6. Append turn to sessions/turns tables.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from google import genai
from google.genai import types

from store import SessionStore, SkillStore, UserStore

MODEL = "gemini-3.5-flash"
ALLOWED_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")

# Per-model thinking level for the main respond() call. Gemini 3 tool use
# (code execution / search) needs thinking engaged; without a thinking_config
# flash-lite mis-emits a bare function_call instead of running code. Mirror the
# levels AI Studio's generated code pairs with each model.
THINKING_LEVELS = {
    "gemini-3.5-flash": types.ThinkingLevel.MEDIUM,
    "gemini-3.1-flash-lite": types.ThinkingLevel.MEDIUM,
}

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
SECTION_RE = re.compile(r"^(##\s+.*)$", re.MULTILINE)
TIERS = ("system", "active", "archive")

# Local-dev fallback path for the cookie-signing secret. Skipped entirely on
# Vercel (read-only FS); set SECRET_KEY in the environment there.
SECRET_KEY_PATH = os.path.join(os.path.dirname(__file__), ".secret_key")
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

# Maximum number of prior turns (user+model messages) sent to `respond`.
# Avoids unbounded history growth as sessions get long. Tune via env var.
HISTORY_TURN_CAP = int(os.environ.get("AGENT_HISTORY_CAP", "20"))


def _window_history(history: list, cap: int = HISTORY_TURN_CAP) -> list:
    """Return only the last `cap` turn entries. Pure, cheap."""
    if cap <= 0 or len(history) <= cap:
        return history
    return history[-cap:]


# ---------- user context ----------

def validate_user_id(user_id: str) -> str:
    """Validate user_id against strict regex. Returns the id or raises ValueError."""
    if not isinstance(user_id, str):
        raise ValueError("user_id must be string")
    if "\x00" in user_id:
        raise ValueError("user_id contains NUL")
    if not USER_ID_RE.match(user_id):
        raise ValueError(f"invalid user_id: {user_id!r}")
    return user_id


@dataclass
class UserCtx:
    """Pure identity carrier. No filesystem state."""
    user_id: str
    model: str = MODEL

    def __post_init__(self) -> None:
        validate_user_id(self.user_id)
        if self.model not in ALLOWED_MODELS:
            self.model = MODEL


def get_secret_key() -> bytes:
    """Bootstrap a persistent secret key for cookie signing.

    Order of preference:
      1. SECRET_KEY env var (required on Vercel).
      2. Local-dev file fallback at .secret_key (skipped on serverless).
      3. Generate-and-save (local-dev only).
    """
    env = os.environ.get("SECRET_KEY")
    if env:
        return env.encode("utf-8")
    if IS_SERVERLESS:
        raise RuntimeError(
            "SECRET_KEY env var is required in serverless deployments "
            "(no writable filesystem for the file fallback)."
        )
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "rb") as f:
            return f.read().strip()
    key = secrets.token_hex(32).encode("utf-8")
    with open(SECRET_KEY_PATH, "wb") as f:
        f.write(key)
    try:
        os.chmod(SECRET_KEY_PATH, 0o600)
    except OSError:
        pass
    return key


# ---------- skill IO ----------

@dataclass
class Skill:
    name: str
    description: str
    body: str
    tier: str = "active"

    def serialize(self) -> str:
        import yaml
        fm = yaml.safe_dump(
            {"name": self.name, "description": self.description},
            sort_keys=False,
        ).strip()
        return f"---\n{fm}\n---\n\n{self.body.strip()}\n"


def load_skills(ctx: UserCtx) -> list[Skill]:
    rows = SkillStore.load_all(ctx.user_id)
    return [
        Skill(
            name=r["name"],
            description=r.get("description", ""),
            body=r.get("body", ""),
            tier=r.get("tier", "active"),
        )
        for r in rows
    ]


def write_skill(
    ctx: UserCtx, name: str, description: str, body: str, tier: str = "active"
) -> str:
    """Insert-or-update a skill. Returns the canonical name (used as identifier)."""
    if tier not in TIERS:
        tier = "active"
    return SkillStore.upsert(ctx.user_id, name, description, body, tier=tier)


def delete_skill(ctx: UserCtx, name: str) -> bool:
    """Delete a skill by name. Returns True if a row was removed."""
    return SkillStore.delete(ctx.user_id, name)


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split body on ## headings. Returns [(heading, chunk_text)]."""
    parts = SECTION_RE.split(body)
    if len(parts) <= 1:
        return [("(full)", body.strip())]
    out: list[tuple[str, str]] = []
    if parts[0].strip():
        out.append(("(preamble)", parts[0].strip()))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((heading, f"{heading}\n{content.strip()}"))
    return out


# ---------- session IO ----------

def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def new_session(ctx: UserCtx) -> str:
    sid = new_session_id()
    SessionStore.create(ctx.user_id, sid)
    return sid


def append_turn(
    ctx: UserCtx,
    session_id: str,
    role: str,
    text: str,
    usage: dict | None = None,
) -> None:
    SessionStore.append_turn(ctx.user_id, session_id, role, text, tokens=usage)


def load_session(ctx: UserCtx, session_id: str) -> list[types.Content]:
    rows = SessionStore.load_turns(ctx.user_id, session_id)
    out: list[types.Content] = []
    for r in rows:
        role = r.get("role")
        text = r.get("text", "")
        if role in ("user", "model") and text:
            out.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return out


def list_sessions(ctx: UserCtx) -> list[dict]:
    """Return [{session_id, first_message, last_ts, turns, rolled_up}]."""
    return SessionStore.list_sessions(ctx.user_id)


def session_turns(ctx: UserCtx, session_id: str) -> list[dict]:
    """Return ordered chat turns [{role, text, tokens}] for replay on refresh."""
    rows = SessionStore.load_turns(ctx.user_id, session_id)
    out: list[dict] = []
    for r in rows:
        role = r.get("role")
        text = r.get("text", "")
        if role in ("user", "model") and text:
            out.append({"role": role, "text": text, "tokens": r.get("tokens")})
    return out


# ---------- LLM helpers ----------

def _client(api_key: str | None = None) -> genai.Client:
    """Build a Gemini client.

    Web deploy: each request supplies the user's own key (held only in their
    browser session) — pass it here. CLI: `api_key` is None and the SDK falls
    back to GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
    """
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def _usage_dict(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if not u:
        return {}
    return {
        "in": getattr(u, "prompt_token_count", 0) or 0,
        "out": getattr(u, "candidates_token_count", 0) or 0,
        "thoughts": getattr(u, "thoughts_token_count", 0) or 0,
        "total": getattr(u, "total_token_count", 0) or 0,
    }


def _log_usage(label: str, resp) -> dict:
    u = _usage_dict(resp)
    if not u:
        return {}
    parts = [f"in={u['in']}", f"out={u['out']}"]
    if u.get("thoughts"):
        parts.append(f"thoughts={u['thoughts']}")
    parts.append(f"total={u['total']}")
    print(f"[tokens {label}] {' '.join(parts)}")
    return u


def pick_skills(
    client: genai.Client,
    prompt: str,
    skills: list[Skill],
    model: str = MODEL,
) -> list[str]:
    """Return list of skill names model deems relevant. System tier auto-included."""
    candidates = [s for s in skills if s.tier != "system"]
    if not candidates:
        return []
    catalog = "\n".join(
        f"- [{s.tier}] {s.name}: {s.description}" for s in candidates
    )
    sys_prompt = (
        "You select relevant skills for a user prompt. "
        "Return ONLY a JSON array of skill names from the catalog. "
        "Empty array if none apply."
    )
    user = f"Catalog:\n{catalog}\n\nUser prompt:\n{prompt}\n\nJSON array:"
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            response_mime_type="application/json",
        ),
    )
    _log_usage("pick", resp)
    try:
        names = json.loads(resp.text)
        if not isinstance(names, list):
            return []
        valid = {s.name for s in candidates}
        return [n for n in names if n in valid]
    except (json.JSONDecodeError, TypeError):
        return []


def pick_archive_sections(
    client: genai.Client, prompt: str, skill: Skill, model: str = MODEL
) -> str:
    """For archive-tier skill: return only relevant ## sections."""
    sections = split_sections(skill.body)
    if len(sections) <= 1:
        return skill.body
    headings = "\n".join(f"- {i}: {h}" for i, (h, _) in enumerate(sections))
    sys_prompt = (
        "Pick relevant sections from archived skill for the prompt. "
        "Return ONLY a JSON array of section indices (integers)."
    )
    user = (
        f"Skill: {skill.name}\nDescription: {skill.description}\n"
        f"Sections:\n{headings}\n\nPrompt:\n{prompt}\n\nJSON array of indices:"
    )
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            response_mime_type="application/json",
        ),
    )
    _log_usage(f"archive:{skill.name}", resp)
    try:
        idxs = json.loads(resp.text)
        if not isinstance(idxs, list):
            return ""
        chunks = [
            sections[i][1]
            for i in idxs
            if isinstance(i, int) and 0 <= i < len(sections)
        ]
        return "\n\n".join(chunks)
    except (json.JSONDecodeError, TypeError):
        return ""


def _assemble_answer(resp) -> str:
    """Build a readable markdown answer from a (possibly multi-part) response.

    With the code_execution tool, a response interleaves text, executable_code,
    and code_execution_result parts. `resp.text` drops the latter two (and
    warns), so we walk parts in order and fence code + output. Falls back to
    `resp.text` when there are no structured parts."""
    cand = (getattr(resp, "candidates", None) or [None])[0]
    content = getattr(cand, "content", None) if cand else None
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return resp.text or ""
    out: list[str] = []
    saw_function_call = False
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            out.append(text)
        code = getattr(part, "executable_code", None)
        if code is not None and getattr(code, "code", None):
            out.append(f"```python\n{code.code}\n```")
        result = getattr(part, "code_execution_result", None)
        if result is not None and getattr(result, "output", None):
            out.append(f"```\n{result.output}\n```")
        # A bare function_call (no matching callable) would otherwise leave the
        # answer empty. Surface it instead of silently dropping the turn.
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            saw_function_call = True
    assembled = "\n\n".join(out).strip()
    if assembled:
        return assembled
    if saw_function_call:
        return (
            "(The model returned an unhandled tool call and no text - the tool "
            "invocation did not resolve to a result this turn. Try rephrasing, "
            "or raise the model's thinking level.)"
        )
    return resp.text or ""


def respond(
    client: genai.Client,
    prompt: str,
    system_skills: list[Skill],
    active_loaded: list[Skill],
    archive_excerpts: list[tuple[Skill, str]],
    history: list[types.Content],
    model: str = MODEL,
    allow_code_execution: bool = False,
) -> tuple[str, dict]:
    """Main answer. Has google_search grounding (and optional code execution).
    Returns (text, usage)."""
    sys_block = "\n\n".join(
        f"## [system] {s.name}\n{s.body.strip()}" for s in system_skills
    )
    active_block = "\n\n".join(
        f"## [active] {s.name}\n{s.description}\n\n{s.body.strip()}"
        for s in active_loaded
    )
    archive_block = "\n\n".join(
        f"## [archive excerpt] {s.name}\n{excerpt}"
        for s, excerpt in archive_excerpts
        if excerpt
    )
    loaded_text = "\n\n".join(
        b for b in (sys_block, active_block, archive_block) if b
    )
    sys_prompt = (
        "You are a helpful agent with a skill-memory system. "
        "Use the loaded skills below as long-term memory. "
        "Use google_search when current information is needed.\n\n"
        f"LOADED SKILLS:\n{loaded_text or '(none)'}"
    )
    if allow_code_execution:
        sys_prompt += (
            "\n\nYou may run Python via the code execution tool for "
            "calculation, data manipulation, or anything better solved by "
            "executing code than by reasoning in prose."
        )
    tools = [types.Tool(google_search=types.GoogleSearch())]
    if allow_code_execution:
        tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
    contents = history + [
        types.Content(role="user", parts=[types.Part(text=prompt)])
    ]
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            tools=tools,
            thinking_config=types.ThinkingConfig(
                thinking_level=THINKING_LEVELS.get(
                    model, types.ThinkingLevel.MEDIUM
                )
            ),
        ),
    )
    if os.environ.get("AGENT_DEBUG"):
        cand = (getattr(resp, "candidates", None) or [None])[0]
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        kinds = [
            attr
            for p in parts
            for attr in (
                "text", "executable_code", "code_execution_result",
                "function_call",
            )
            if getattr(p, attr, None)
        ]
        ran_code = "executable_code" in kinds
        print(f"[respond parts] {kinds} code_executed={ran_code}")
    usage = _log_usage("respond", resp)
    return _assemble_answer(resp), usage


EDIT_INSTRUCTIONS = """\
You are a memory writer. Aggressively persist anything from the exchange that could
inform FUTURE conversations. Skills are markdown docs stored with a YAML
frontmatter (name, description) and a markdown body. Skills live in one of three tiers:

- "system": ALWAYS loaded into context. Reserved for core identity/rules the agent
  must obey every turn (e.g. user-profile, response-style, hard rules).
- "active" (default): Loaded only when frontmatter matches the prompt. Use for most
  facts, projects, preferences, references.
- "archive": Loaded selectively by section. Use when content is large (long notes,
  research, multi-topic reference). Structure body with ## headings — each becomes
  retrievable on its own.

WRITE A SKILL whenever the user reveals ANY of:
- Their name, role, employer, team, location, or background  → system tier
- Strong preferences for how to respond, nicknames assigned to you  → system tier
- Tools, frameworks, languages they use  → active
- Projects they work on  → active
- Rules ("always X", "never Y")  → system tier
- People, pets, customers, systems referenced  → active
- External resources (URLs, dashboards, docs)  → active
- Bulky research notes, long-form knowledge  → archive (with ## sections)

PREFER MANY SMALL SKILLS over one big one for active tier. Examples of good names:
"user-profile", "response-style", "tech-stack", "project-<name>", "rule-<topic>",
"user-pets", "interest-<thing>".

DEDUPLICATION (CRITICAL):
- BEFORE deciding op="create", scan the "Existing skills" inventory below.
- If ANY existing skill name or description covers the same topic — even partly —
  you MUST use op="update" with that exact existing name. Do NOT invent a new name
  for the same topic. Do NOT write parallel skills like "user-pets" and
  "pets-info" for the same facts.
- For op="update", produce the MERGED body (existing body + new info, no duplicate
  bullets). The "Existing bodies" section is provided so you can do this merge.
- For archive updates, preserve existing ## sections; append new ## sections only
  for genuinely new topics.

BODY FORMAT (CRITICAL):
- The "body" field is markdown ONLY. It must NOT contain a YAML frontmatter block.
- Do NOT start the body with "---". Do NOT include "name:" or "description:" lines
  inside the body. The frontmatter is constructed from the "name" and "description"
  fields you return separately.
- Bad body (DO NOT DO THIS):
    ---
    name: identity
    description: ...
    ---
    - bullet
- Good body:
    - bullet

Return ONLY JSON, no prose:
{"edits": [{"op": "create"|"update", "tier": "system"|"active"|"archive",
  "name": "...", "description": "...", "body": "..."}]}

Empty edits array ONLY if the exchange is pure small talk with zero new facts.
"""


def _strip_leading_frontmatter(body: str) -> str:
    """Remove a YAML frontmatter block if the model erroneously embedded one
    at the top of the body field."""
    if not body:
        return body
    stripped = body.lstrip()
    if not stripped.startswith("---"):
        return body
    m = FRONTMATTER_RE.match(stripped)
    if m:
        return m.group(2).lstrip()
    return body


def reflect_and_edit(
    client: genai.Client,
    prompt: str,
    answer: str,
    all_skills: list[Skill],
    scoped_skills: list[Skill],
    model: str = MODEL,
) -> list[dict]:
    """Reflect on the exchange and emit skill edits.

    Cost-shaped inputs:
      - `all_skills`: every skill (any tier) — used only to build the *cheap*
        name+description inventory so the model can avoid dup-by-name across
        the whole catalog.
      - `scoped_skills`: skills whose full bodies we will pay for. Caller MUST
        pre-filter to the skills relevant to this turn (system + picked active).
        Archive bodies are explicitly excluded — archive is cold storage,
        modified only by the consolidation pass.
    """
    inventory = "\n".join(
        f"- [{s.tier}] {s.name}: {s.description}" for s in all_skills
    ) or "(no skills yet)"
    body_skills = [s for s in scoped_skills if s.tier != "archive"]
    existing_bodies = "\n\n".join(
        f"### [{s.tier}] {s.name}\n{s.body.strip()}" for s in body_skills
    )
    total_body_chars = sum(len(s.body) for s in all_skills)
    scoped_body_chars = sum(len(s.body) for s in body_skills)
    print(
        f"[reflect scope] catalog={len(all_skills)} "
        f"scoped_bodies={len(body_skills)} "
        f"total_body_chars={total_body_chars} "
        f"scoped_body_chars={scoped_body_chars}"
    )
    user = (
        f"Existing skills:\n{inventory}\n\n"
        f"Existing bodies (for update merges):\n{existing_bodies or '(none)'}\n\n"
        f"User prompt:\n{prompt}\n\nAssistant answer:\n{answer}\n\n"
        "Return edits JSON:"
    )
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=EDIT_INSTRUCTIONS,
            response_mime_type="application/json",
        ),
    )
    _log_usage("reflect", resp)
    raw = resp.text or ""
    if os.environ.get("AGENT_DEBUG"):
        print(f"[reflect raw]: {raw[:500]}")
    try:
        data = json.loads(raw)
        edits = data.get("edits", [])
        return edits if isinstance(edits, list) else []
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        if os.environ.get("AGENT_DEBUG"):
            print(f"[reflect parse error]: {e}")
        return []


SESSION_INSTRUCTIONS = """\
Summarize this chat session as a skill memory for future sessions.
Capture: topics discussed, decisions made, user state/mood, open threads, anything
worth remembering next time. 3-7 tight bullets. No filler.

Return ONLY JSON:
{"description": "one-line hook for matching future prompts", "body": "markdown bullets"}

If the session was trivial (one or two turns of small talk, no substance), return:
{"description": "", "body": ""}
"""


def summarize_session_to_skill(
    client: genai.Client, ctx: UserCtx, session_id: str
) -> str | None:
    """Roll a chat into a `session-<ts>` active skill. Returns the skill name
    or None if the session was trivial or already rolled up."""
    if SessionStore.is_rolled_up(ctx.user_id, session_id):
        return None
    history = load_session(ctx, session_id)
    if len(history) < 2:
        return None
    transcript = "\n\n".join(
        f"{c.role.upper()}: {''.join(p.text or '' for p in c.parts)}"
        for c in history
    )
    resp = client.models.generate_content(
        model=ctx.model,
        contents=f"Transcript:\n{transcript}\n\nReturn JSON:",
        config=types.GenerateContentConfig(
            system_instruction=SESSION_INSTRUCTIONS,
            response_mime_type="application/json",
        ),
    )
    _log_usage("session", resp)
    raw = resp.text or ""
    try:
        data = json.loads(raw)
        if data.get("description") and data.get("body"):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = f"session-{stamp}"
            write_skill(ctx, name, data["description"], data["body"], tier="active")
            SessionStore.mark_rolled_up(ctx.user_id, session_id, name)
            return name
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


# ---------- consolidation (offline pass) ----------

def consolidate(ctx: UserCtx, dry_run: bool = False) -> dict:
    """Fold every active `session-*` skill into one new archive skill and delete
    the originals. Idempotent: a no-op when there are no `session-*` skills
    left. Runs in a single transaction so a crash mid-fold leaves both the
    new archive skill and the old session skills present, never neither.
    """
    skills = load_skills(ctx)
    session_skills = [
        s for s in skills
        if s.tier == "active" and s.name.startswith("session-")
    ]

    summary = {
        "input_total": len(skills),
        "input_sessions": len(session_skills),
        "input_keep": len(skills) - len(session_skills),
        "merged_archive_skill": None,
        "deleted_session_skills": [],
        "dry_run": dry_run,
    }

    if not session_skills:
        summary["reason"] = "no session-* skills to consolidate"
        return summary

    # Build one archive skill of all session bodies, one `##` section each.
    sections = []
    for s in sorted(session_skills, key=lambda x: x.name):
        sections.append(f"## {s.name}\n\n{s.body.strip()}")
    body = "\n\n".join(sections)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_name = f"sessions-archive-{stamp}"
    desc = (
        f"Consolidated {len(session_skills)} session rollups "
        f"(merged {datetime.now().date().isoformat()})."
    )

    if dry_run:
        summary["merged_archive_skill"] = archive_name
        summary["deleted_session_skills"] = [s.name for s in session_skills]
        return summary

    # Write archive skill, then delete originals. Both happen against the same
    # connection so a transactional wrapper at the caller can roll the pair
    # back together if needed; SkillStore.upsert/delete are individually atomic.
    write_skill(ctx, archive_name, desc, body, tier="archive")
    deleted: list[str] = []
    for s in session_skills:
        if delete_skill(ctx, s.name):
            deleted.append(s.name)

    summary["merged_archive_skill"] = archive_name
    summary["deleted_session_skills"] = deleted
    return summary


def apply_edits(ctx: UserCtx, edits: list[dict]) -> list[str]:
    applied: list[str] = []
    for e in edits:
        name = e.get("name")
        desc = e.get("description", "")
        body = e.get("body", "")
        tier = e.get("tier", "active")
        if not name or not body:
            continue
        body = _strip_leading_frontmatter(body)
        if not body.strip():
            continue
        canonical = write_skill(ctx, name, desc, body, tier=tier)
        applied.append(f"{e.get('op', 'write')}[{tier}] -> {canonical}")
    return applied


# ---------- shared per-turn driver (used by CLI + server) ----------

@dataclass
class TurnResult:
    answer: str
    loaded_system: list[str]
    loaded_active: list[str]
    loaded_archive: list[str]
    edits_applied: list[str]
    tokens: dict


def run_turn_events(
    client: genai.Client,
    ctx: UserCtx,
    session_id: str,
    prompt: str,
    allow_code_execution: bool = False,
):
    """Generator. Yields {stage, msg, ...} dicts. Final event has stage='done'.

    `allow_code_execution` enables Gemini's code execution tool in `respond`.
    CLI passes True; the server leaves it False to keep the web path text-only.
    """
    yield {"stage": "load", "msg": "Reading skill catalog…"}
    skills = load_skills(ctx)
    system_skills = [s for s in skills if s.tier == "system"]

    yield {
        "stage": "pick",
        "msg": f"Picking relevant skills from {len(skills) - len(system_skills)} options…",
    }
    chosen_names = pick_skills(client, prompt, skills, model=ctx.model)
    picked = [s for s in skills if s.name in chosen_names]
    active_loaded = [s for s in picked if s.tier == "active"]
    archive_picked = [s for s in picked if s.tier == "archive"]

    archive_excerpts: list[tuple[Skill, str]] = []
    for s in archive_picked:
        yield {"stage": "archive", "msg": f"Retrieving sections from archive: {s.name}"}
        excerpt = pick_archive_sections(client, prompt, s, model=ctx.model)
        archive_excerpts.append((s, excerpt))

    loaded_summary = {
        "system": [s.name for s in system_skills],
        "active": [s.name for s in active_loaded],
        "archive": [s.name for s, _ in archive_excerpts],
    }
    yield {
        "stage": "respond",
        "msg": "Thinking…",
        "loaded": loaded_summary,
    }
    full_history = load_session(ctx, session_id)
    history = _window_history(full_history)
    if len(history) < len(full_history):
        print(
            f"[history window] full={len(full_history)} sent={len(history)}"
        )
    answer, usage = respond(
        client,
        prompt,
        system_skills,
        active_loaded,
        archive_excerpts,
        history,
        model=ctx.model,
        allow_code_execution=allow_code_execution,
    )

    yield {"stage": "persist", "msg": "Saving turn to session log…"}
    append_turn(ctx, session_id, "user", prompt)
    append_turn(ctx, session_id, "model", answer, usage=usage)

    yield {"stage": "reflect", "msg": "Reflecting on what to remember…"}
    scoped_for_reflect = system_skills + active_loaded
    edits = reflect_and_edit(
        client,
        prompt,
        answer,
        all_skills=skills,
        scoped_skills=scoped_for_reflect,
        model=ctx.model,
    )
    applied = apply_edits(ctx, edits)

    meta = {
        "catalog": len(skills),
        "scoped_bodies": len(
            [s for s in scoped_for_reflect if s.tier != "archive"]
        ),
        "total_body_chars": sum(len(s.body) for s in skills),
        "scoped_body_chars": sum(
            len(s.body)
            for s in scoped_for_reflect
            if s.tier != "archive"
        ),
        "history_full": len(full_history),
        "history_sent": len(history),
    }

    yield {
        "stage": "done",
        "answer": answer,
        "loaded": loaded_summary,
        "edits": applied,
        "tokens": usage,
        "meta": meta,
    }


def run_turn(
    client: genai.Client,
    ctx: UserCtx,
    session_id: str,
    prompt: str,
    allow_code_execution: bool = False,
) -> TurnResult:
    """Drains run_turn_events. Kept for CLI compatibility."""
    final: dict | None = None
    for ev in run_turn_events(
        client, ctx, session_id, prompt,
        allow_code_execution=allow_code_execution,
    ):
        if ev.get("stage") == "done":
            final = ev
    if final is None:
        raise RuntimeError("run_turn_events did not emit 'done'")
    return TurnResult(
        answer=final["answer"],
        loaded_system=final["loaded"]["system"],
        loaded_active=final["loaded"]["active"],
        loaded_archive=final["loaded"]["archive"],
        edits_applied=final["edits"],
        tokens=final["tokens"],
    )


# ---------- REPL ----------

def _code_exec_enabled() -> bool:
    """CLI code execution toggle. On by default; set AGENT_CODE_EXEC=0 to disable."""
    return os.environ.get("AGENT_CODE_EXEC", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )


def main() -> None:
    if "--consolidate" in sys.argv:
        _run_consolidate_cli()
        return
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("DATABASE_URL"):
        print("Set DATABASE_URL (Postgres) before running.", file=sys.stderr)
        sys.exit(1)

    user_id = os.environ.get("AGENT_USER_ID") or "default"
    try:
        validate_user_id(user_id)
    except ValueError as e:
        print(f"Invalid AGENT_USER_ID: {e}", file=sys.stderr)
        sys.exit(1)

    ctx = UserCtx(user_id=user_id)
    client = _client()
    session_id = new_session(ctx)
    code_exec = _code_exec_enabled()
    print(f"Agent ready ({ctx.model}). User: {user_id}. Session: {session_id}")
    print(f"Code execution: {'on' if code_exec else 'off'} (AGENT_CODE_EXEC)")
    print("Type prompt. Ctrl-C to exit.\n")

    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue

        result = run_turn(
            client, ctx, session_id, prompt,
            allow_code_execution=code_exec,
        )

        tier_summary = []
        if result.loaded_system:
            tier_summary.append(f"system:{len(result.loaded_system)}")
        if result.loaded_active:
            tier_summary.append(f"active:{','.join(result.loaded_active)}")
        if result.loaded_archive:
            tier_summary.append(f"archive:{','.join(result.loaded_archive)}")
        print(f"[loaded {' | '.join(tier_summary) if tier_summary else 'nothing'}]")

        print(f"\nagent> {result.answer}\n")

        if result.edits_applied:
            print(f"[memory: {'; '.join(result.edits_applied)}]\n")

    summarized = summarize_session_to_skill(client, ctx, session_id)
    if summarized:
        print(f"[session saved: {summarized}]")


def _consolidate_one(user_id: str, dry_run: bool) -> None:
    ctx = UserCtx(user_id=user_id)
    print(f"\n=== Consolidating skills for {user_id} (dry_run={dry_run}) ===")
    result = consolidate(ctx, dry_run=dry_run)
    print(json.dumps(result, indent=2))


def _run_consolidate_cli() -> None:
    dry = "--dry-run" in sys.argv
    all_users = "--all-users" in sys.argv

    if all_users:
        uids = UserStore.list_user_ids()
        if not uids:
            print("No users found in the users table.", file=sys.stderr)
            sys.exit(1)
        for uid in uids:
            _consolidate_one(uid, dry)
        return

    user_id = os.environ.get("AGENT_USER_ID") or "default"
    try:
        validate_user_id(user_id)
    except ValueError as e:
        print(f"Invalid AGENT_USER_ID: {e}", file=sys.stderr)
        sys.exit(1)
    _consolidate_one(user_id, dry)


if __name__ == "__main__":
    main()
