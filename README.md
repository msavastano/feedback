# Skill-memory agent

Self-editing agent on Gemini 3.5 Flash. Three-tier markdown memory (`system` / `active` / `archive`), per-user isolation, web UI + CLI.

## Setup

```
pip install -r requirements.txt
$env:GEMINI_API_KEY = "..."
```

## Run

**Web server** (multi-user, mock login):
```
uvicorn server:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000 → pick a mock user → chat.

**CLI** (single user per process):
```
$env:AGENT_USER_ID = "alice"
python agent.py
```

## Per-turn flow

1. Read frontmatter of every skill under `data/users/<user_id>/skills/`.
2. Gemini picks relevant skills from non-system tiers.
3. System tier always loads. Active picks load whole-body. Archive picks load chosen `##` sections.
4. Gemini answers (google_search grounding available).
5. Reflect call returns skill edits — applied to disk.
6. Turn appended to `sessions/<session_id>.jsonl`.
7. On `/api/session/end` (or Ctrl-C in CLI) → session summarized into a new active skill.

## Data layout

```
data/users/<user_id>/
  skills/
    system/*.md         # always loaded (identity, response style, hard rules)
    *.md                # active tier, frontmatter-matched
    archive/*.md        # archive tier, section-retrieved by ## heading
  sessions/
    <session_id>.jsonl  # one turn per line
```

## Skill file format

```markdown
---
name: skill-name
description: One-line description used for matching against prompts
---

Body. For archive-tier, structure with ## sections for granular retrieval.
```

## Mock-account auth (interim)

`mock_users.json` lists allowed user ids. Login POSTs to `/api/login` which sets a signed cookie. `current_user` dependency in [server.py](server.py) re-validates the cookie on every request.

**Single seam for Google OAuth:** replace `/api/login` with the OAuth callback. Cookie payload shape (`{"user_id": "..."}`) and `current_user` dependency stay identical. Nothing else in the codebase touches identity.

Mock login has no password — bind to `127.0.0.1` only.

## Env vars

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` — required.
- `AGENT_USER_ID` — CLI only. Must exist in `mock_users.json`.
- `SECRET_KEY` — optional. Auto-generated at `.secret_key` if absent.
- `AGENT_DEBUG=1` — prints raw LLM JSON from reflect/session calls.

## API

| Method | Path                     | Body / Notes                      |
| ------ | ------------------------ | --------------------------------- |
| GET    | `/api/users`             | Public. Mock account list.        |
| POST   | `/api/login`             | `{user_id}` → sets cookie.        |
| POST   | `/api/logout`            | Clears cookie.                    |
| GET    | `/api/me`                | Current user.                     |
| POST   | `/api/chat`              | `{session_id?, message}`          |
| POST   | `/api/session/end`       | `{session_id}` → writes skill.    |
| GET    | `/api/sessions`          | List user's sessions.             |
| GET    | `/api/skills`            | List user's skills.               |
| DELETE | `/api/skill/{name}`      | Remove a skill.                   |
