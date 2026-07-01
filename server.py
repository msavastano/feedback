"""FastAPI server for the skill-memory agent. Google OAuth (ID-token JS flow).

Single auth seam: `/api/login/google` verifies a Google ID token and issues a
signed cookie. `current_user` dependency verifies the cookie on every request.
Downstream code is identity-source-agnostic — cookie payload is
`{"user_id": "..."}` regardless of how the user was authenticated.

`user_id` is derived from the email local-part (with `+tag` stripped and `.`
replaced by `-`) so it stays a stable, URL-safe identifier. Collisions across
email domains are intentional for personal/loopback use.

All persistent state lives in Postgres (see store.py + schema.sql). Designed
for Vercel: no filesystem writes outside `/tmp`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, Field

from agent import (
    ALLOWED_MODELS,
    MODEL,
    THINKING_CAPABLE_MODELS,
    THINKING_LEVEL_NAMES,
    Clients,
    UserCtx,
    _anthropic_client,
    _client,
    apply_edits,
    consolidate,
    delete_skill,
    get_secret_key,
    list_sessions,
    load_skills,
    new_session,
    prepare_attachments,
    provider_of,
    run_turn,
    run_turn_events,
    session_turns,
    summarize_session_to_skill,
    validate_user_id,
)
from store import UserStore, get_pool

STATIC_DIR = Path(__file__).parent / "static"
COOKIE_NAME = "agent_user"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
CONSOLIDATE_TOKEN = os.environ.get("CONSOLIDATE_TOKEN", "").strip()
CRON_SECRET = os.environ.get("CRON_SECRET", "").strip()  # Vercel Cron auto-injects

# Comma-separated user_ids allowed to use Gemini code execution on the web path.
# Code runs in Google's sandbox (no host access), but it is billable and abusable,
# so it is an opt-in entitlement gated on the authenticated identity. Off for
# everyone when unset.
CODE_EXEC_USERS = {
    u for u in (
        s.strip() for s in os.environ.get("CODE_EXEC_USERS", "").split(",")
    ) if u
}

app = FastAPI(title="Skill-memory agent")

_serializer = URLSafeSerializer(get_secret_key(), salt="agent-user-cookie")


def _user_id_from_email(email: str) -> str:
    """email local-part → user_id. Strip +tag, replace . with -, lowercase."""
    local = email.split("@", 1)[0].lower()
    local = local.split("+", 1)[0]
    local = local.replace(".", "-")
    return local


# ---------- auth ----------

def current_user(agent_user: str | None = Cookie(default=None)) -> str:
    if not agent_user:
        raise HTTPException(status_code=401, detail="not logged in")
    try:
        payload = _serializer.loads(agent_user)
    except BadSignature:
        raise HTTPException(status_code=401, detail="bad cookie")
    user_id = payload.get("user_id") if isinstance(payload, dict) else None
    if not user_id:
        raise HTTPException(status_code=401, detail="malformed cookie")
    try:
        validate_user_id(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid user_id")
    return user_id


def _ctx(user_id: str = Depends(current_user)) -> UserCtx:
    return UserCtx(user_id=user_id)


def gemini_key(x_gemini_key: str | None = Header(default=None)) -> str | None:
    """The caller's Gemini API key, supplied per-request via the `X-Gemini-Key`
    header. The key lives only in the user's browser session — the server never
    stores it. Optional: returns None when absent; `api_chat` enforces the key
    for whichever provider the chosen model needs."""
    return (x_gemini_key or "").strip() or None


def anthropic_key(x_anthropic_key: str | None = Header(default=None)) -> str | None:
    """The caller's Anthropic API key, supplied per-request via the
    `X-Anthropic-Key` header. Same BYOK contract as `gemini_key`; optional."""
    return (x_anthropic_key or "").strip() or None


class GoogleLoginBody(BaseModel):
    credential: str = Field(min_length=1)


@app.get("/api/config")
def api_config() -> dict:
    """Public — boot data the login UI needs (Google client ID)."""
    return {"google_client_id": GOOGLE_CLIENT_ID}


@app.get("/api/health")
def api_health() -> dict:
    """Readiness check — verifies the DB pool can hand out a connection."""
    try:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"db unavailable: {e}")


@app.post("/api/login/google")
def api_login_google(body: GoogleLoginBody, response: Response) -> dict:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")
    try:
        info = google_id_token.verify_oauth2_token(
            body.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"invalid id_token: {e}")

    email = info.get("email")
    if not email or not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="email not verified")

    user_id = _user_id_from_email(email)
    try:
        validate_user_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"cannot derive user_id: {e}")

    profile = {
        "user_id": user_id,
        "email": email,
        "name": info.get("name", ""),
        "sub": info.get("sub", ""),
        "picture": info.get("picture", ""),
    }
    UserStore.upsert(profile)

    token = _serializer.dumps({"user_id": user_id})
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return {"user_id": user_id, "email": email, "name": profile["name"]}


@app.post("/api/logout")
def api_logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me")
def api_me(user_id: str = Depends(current_user)) -> dict:
    profile = UserStore.get(user_id)
    display = profile.get("name") or profile.get("email") or user_id
    return {
        "user_id": user_id,
        "display_name": display,
        "email": profile.get("email", ""),
        "picture": profile.get("picture", ""),
        "model": MODEL,
        "allowed_models": list(ALLOWED_MODELS),
        "allowed_thinking_levels": list(THINKING_LEVEL_NAMES.keys()),
        "thinking_capable_models": list(THINKING_CAPABLE_MODELS),
    }


# ---------- chat ----------

class ChatAttachment(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    mime: str = ""
    data: str = Field(min_length=1)  # base64 file bytes


class ChatBody(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    model: str | None = None
    thinking_level: str | None = None
    # Session-only uploads: bytes go to Gemini for this turn, never to the DB.
    attachments: list[ChatAttachment] | None = None


@app.post("/api/chat")
def api_chat(
    body: ChatBody,
    user_id: str = Depends(current_user),
    gkey: str | None = Depends(gemini_key),
    akey: str | None = Depends(anthropic_key),
) -> StreamingResponse:
    """NDJSON stream of stage events. First line: {session_id}. Last: {stage:'done', ...}."""
    chosen = body.model if body.model in ALLOWED_MODELS else MODEL
    ctx = UserCtx(
        user_id=user_id, model=chosen, thinking_level=body.thinking_level
    )
    # Require the key for the chosen model's provider. 400 (the UI treats it as
    # "supply your key"); attachment errors below use 422 to stay distinct.
    if provider_of(chosen) == "anthropic" and not akey:
        raise HTTPException(
            status_code=400,
            detail="missing Anthropic API key (X-Anthropic-Key header)",
        )
    if provider_of(chosen) == "gemini" and not gkey:
        raise HTTPException(
            status_code=400,
            detail="missing Gemini API key (X-Gemini-Key header)",
        )
    attachments = None
    if body.attachments:
        try:
            attachments = prepare_attachments(
                [a.model_dump() for a in body.attachments]
            )
        except ValueError as e:
            # 422 (not 400): the UI treats 400 as "missing API key".
            raise HTTPException(status_code=422, detail=str(e))
    clients = Clients(
        gemini=_client(gkey) if gkey else None,
        anthropic=_anthropic_client(akey) if akey else None,
    )
    session_id = body.session_id or new_session(ctx)

    allow_code_execution = user_id in CODE_EXEC_USERS

    def gen():
        yield json.dumps({"stage": "init", "session_id": session_id}) + "\n"
        try:
            for ev in run_turn_events(
                clients, ctx, session_id, body.message,
                allow_code_execution=allow_code_execution,
                attachments=attachments,
            ):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"stage": "error", "msg": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class SessionEndBody(BaseModel):
    session_id: str
    # The session has no stored model, so the browser passes the model it was
    # chatting with so the summary routes to the right provider.
    model: str | None = None
    # sendBeacon (fired on pagehide) cannot set headers, so the browser passes
    # the key(s) in the body on that path. The header forms are also accepted.
    gemini_key: str | None = None
    anthropic_key: str | None = None


@app.post("/api/session/end")
def api_session_end(
    body: SessionEndBody,
    user_id: str = Depends(current_user),
    x_gemini_key: str | None = Header(default=None),
    x_anthropic_key: str | None = Header(default=None),
) -> dict:
    chosen = body.model if body.model in ALLOWED_MODELS else MODEL
    ctx = UserCtx(user_id=user_id, model=chosen)
    gkey = (body.gemini_key or x_gemini_key or "").strip()
    akey = (body.anthropic_key or x_anthropic_key or "").strip()
    provider = provider_of(chosen)
    needed = akey if provider == "anthropic" else gkey
    if not needed:
        # No key for the session's provider (e.g. session already gone). Skip
        # summarization rather than failing — chat history is already persisted.
        return {"saved": None, "reason": f"no {provider} API key supplied"}
    clients = Clients(
        gemini=_client(gkey) if gkey else None,
        anthropic=_anthropic_client(akey) if akey else None,
    )
    name = summarize_session_to_skill(clients, ctx, body.session_id)
    return {"saved": name}


@app.get("/api/sessions")
def api_sessions(ctx: UserCtx = Depends(_ctx)) -> list[dict]:
    return list_sessions(ctx)


@app.get("/api/session/{session_id}")
def api_session_get(session_id: str, ctx: UserCtx = Depends(_ctx)) -> dict:
    return {"session_id": session_id, "turns": session_turns(ctx, session_id)}


# ---------- skills ----------

@app.get("/api/skills")
def api_skills(ctx: UserCtx = Depends(_ctx)) -> list[dict]:
    return [
        {"name": s.name, "tier": s.tier, "description": s.description}
        for s in load_skills(ctx)
    ]


@app.get("/api/skill/{name}")
def api_skill_get(name: str, ctx: UserCtx = Depends(_ctx)) -> dict:
    for s in load_skills(ctx):
        if s.name == name:
            return {
                "name": s.name,
                "tier": s.tier,
                "description": s.description,
                "body": s.body,
            }
    raise HTTPException(status_code=404, detail="skill not found")


@app.delete("/api/skill/{name}")
def api_skill_delete(name: str, ctx: UserCtx = Depends(_ctx)) -> dict:
    ok = delete_skill(ctx, name)
    if not ok:
        raise HTTPException(status_code=404, detail="skill not found")
    return {"deleted": name}


class ConsolidateBody(BaseModel):
    dry_run: bool = False


@app.post("/api/consolidate")
def api_consolidate(
    body: ConsolidateBody = ConsolidateBody(),
    ctx: UserCtx = Depends(_ctx),
) -> dict:
    """Per-user consolidation. Folds active session-* skills into one archive
    skill and deletes the originals. Idempotent."""
    return consolidate(ctx, dry_run=body.dry_run)


# ---------- admin / cron ----------

class AdminConsolidateBody(BaseModel):
    user_id: str | None = None     # if None, iterate every user
    dry_run: bool = False


def _require_consolidate_token(x_consolidate_token: str | None = Header(default=None)) -> None:
    if not CONSOLIDATE_TOKEN:
        raise HTTPException(status_code=500, detail="CONSOLIDATE_TOKEN not configured")
    if x_consolidate_token != CONSOLIDATE_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


@app.post("/api/admin/consolidate")
def api_admin_consolidate(
    body: AdminConsolidateBody = AdminConsolidateBody(),
    _: None = Depends(_require_consolidate_token),
) -> dict:
    """Manual admin endpoint. Guarded by `X-Consolidate-Token` header
    matching the `CONSOLIDATE_TOKEN` env var. Targets one user (`user_id`) or
    iterates every row in the `users` table when omitted."""
    return _do_consolidate(body.user_id, body.dry_run)


@app.get("/api/cron/consolidate")
def api_cron_consolidate(
    authorization: str | None = Header(default=None),
) -> dict:
    """Vercel Cron entrypoint. Vercel auto-injects `Authorization: Bearer
    $CRON_SECRET` on scheduled invocations — we verify against the env var.
    Iterates every user and runs consolidation."""
    if not CRON_SECRET:
        raise HTTPException(status_code=500, detail="CRON_SECRET not configured")
    if authorization != f"Bearer {CRON_SECRET}":
        raise HTTPException(status_code=401, detail="bad cron auth")
    return _do_consolidate(user_id=None, dry_run=False)


def _do_consolidate(user_id: str | None, dry_run: bool) -> dict:
    if user_id:
        validate_user_id(user_id)
        ctx = UserCtx(user_id=user_id)
        return {"results": [{"user_id": user_id, **consolidate(ctx, dry_run=dry_run)}]}

    out = []
    for uid in UserStore.list_user_ids():
        try:
            validate_user_id(uid)
        except ValueError:
            continue
        ctx = UserCtx(user_id=uid)
        out.append({"user_id": uid, **consolidate(ctx, dry_run=dry_run)})
    return {"results": out}


# ---------- static ----------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "static/index.html missing"}, status_code=500)
    return FileResponse(idx)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    """Browsers probe /favicon.ico at the root; serve the SVG icon."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")
