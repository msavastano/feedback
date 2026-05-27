"""FastAPI server for the skill-memory agent. Google OAuth (ID-token JS flow).

Single auth seam: `/api/login/google` verifies a Google ID token and issues a
signed cookie. `current_user` dependency verifies the cookie on every request.
Downstream code is identity-source-agnostic — cookie payload is
`{"user_id": "..."}` regardless of how the user was authenticated.

`user_id` is derived from the email local-part (with `+tag` stripped and `.`
replaced by `-`) so filesystem paths under `data/users/<user_id>/` stay readable.
Collisions across email domains are intentional for personal/loopback use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, Field

from agent import (
    UserCtx,
    _client,
    apply_edits,
    consolidate,
    delete_skill,
    get_secret_key,
    list_sessions,
    load_skills,
    new_session,
    run_turn,
    run_turn_events,
    summarize_session_to_skill,
    validate_user_id,
)

STATIC_DIR = Path(__file__).parent / "static"
COOKIE_NAME = "agent_user"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()

app = FastAPI(title="Skill-memory agent")

_serializer = URLSafeSerializer(get_secret_key(), salt="agent-user-cookie")


def _user_id_from_email(email: str) -> str:
    """email local-part → user_id. Strip +tag, replace . with -, lowercase."""
    local = email.split("@", 1)[0].lower()
    local = local.split("+", 1)[0]
    local = local.replace(".", "-")
    return local


def _profile_path(user_id: str) -> Path:
    return UserCtx(user_id=user_id).root / "profile.json"


def _load_profile(user_id: str) -> dict:
    p = _profile_path(user_id)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_profile(user_id: str, profile: dict) -> None:
    p = _profile_path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2), encoding="utf-8")


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


class GoogleLoginBody(BaseModel):
    credential: str = Field(min_length=1)


@app.get("/api/config")
def api_config() -> dict:
    """Public — boot data the login UI needs (Google client ID)."""
    return {"google_client_id": GOOGLE_CLIENT_ID}


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
    _save_profile(user_id, profile)

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
    profile = _load_profile(user_id)
    display = profile.get("name") or profile.get("email") or user_id
    ctx = UserCtx(user_id=user_id)
    return {
        "user_id": user_id,
        "display_name": display,
        "email": profile.get("email", ""),
        "picture": profile.get("picture", ""),
        "model": ctx.model,
    }


# ---------- chat ----------

class ChatBody(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


@app.post("/api/chat")
def api_chat(body: ChatBody, ctx: UserCtx = Depends(_ctx)) -> StreamingResponse:
    """NDJSON stream of stage events. First line: {session_id}. Last: {stage:'done', ...}."""
    client = _client()
    session_id = body.session_id or new_session(ctx)

    def gen():
        yield json.dumps({"stage": "init", "session_id": session_id}) + "\n"
        try:
            for ev in run_turn_events(client, ctx, session_id, body.message):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"stage": "error", "msg": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class SessionEndBody(BaseModel):
    session_id: str


@app.post("/api/session/end")
def api_session_end(
    body: SessionEndBody, ctx: UserCtx = Depends(_ctx)
) -> dict:
    client = _client()
    path = summarize_session_to_skill(client, ctx, body.session_id)
    return {"saved": path.name if path else None}


@app.get("/api/sessions")
def api_sessions(ctx: UserCtx = Depends(_ctx)) -> list[dict]:
    return list_sessions(ctx)


# ---------- skills ----------

@app.get("/api/skills")
def api_skills(ctx: UserCtx = Depends(_ctx)) -> list[dict]:
    return [
        {"name": s.name, "tier": s.tier, "description": s.description}
        for s in load_skills(ctx)
    ]


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
    """Offline consolidation pass. Writes to skills.consolidated/ side path
    — original skills tree untouched. Idempotent."""
    return consolidate(ctx, dry_run=body.dry_run)


# ---------- static ----------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "static/index.html missing"}, status_code=500)
    return FileResponse(idx)
