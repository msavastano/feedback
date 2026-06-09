# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# install
pip install -r requirements.txt
$env:GEMINI_API_KEY = "..."   # or GOOGLE_API_KEY
$env:GOOGLE_CLIENT_ID = "..." # OAuth 2.0 Web client ID from Google Cloud Console
                              # authorized JS origin: http://127.0.0.1:8000

# web server (Google OAuth login, bind to loopback only)
uvicorn server:app --host 127.0.0.1 --port 8000
$env:CODE_EXEC_USERS = "alice,bob"  # opt-in: user_ids allowed code execution on
                                    # the web path (off for everyone when unset)

# CLI (single user per process, no Google flow — AGENT_USER_ID is the identity)
$env:AGENT_USER_ID = "alice"
python agent.py

# offline consolidation pass (folds session-* rollups into archive)
python agent.py --consolidate              # one user (AGENT_USER_ID or "default")
python agent.py --consolidate --all-users  # iterate every data/users/*/ dir
python agent.py --consolidate --dry-run

# debug: dump raw LLM JSON from reflect/session calls
$env:AGENT_DEBUG = "1"

# history window cap (default 20 turns sent to respond)
$env:AGENT_HISTORY_CAP = "40"

# skip per-turn reflect call to save one Gemini request/turn (on by default).
# session-end summarization still persists memory. Helps free-tier rate limits.
$env:AGENT_REFLECT = "0"   # disable

# CLI code execution (Gemini built-in sandbox; on by default, CLI only — never web)
# respond() sets a per-model thinking_level (see THINKING_LEVELS in agent.py);
# tool use needs thinking engaged or the model mis-emits a bare function_call.
$env:AGENT_CODE_EXEC = "0"   # disable

# Override the per-model thinking level for respond() (minimal|low|medium|high,
# case-insensitive). Invalid/unset => per-model THINKING_LEVELS default.
$env:AGENT_THINKING_LEVEL = "high"

# Image generation (native-image model IMAGE_MODEL = gemini-3.1-flash-image).
# On by default; a cheap per-turn classifier (detect_image_intent) routes
# "draw/generate an image" prompts to the image model. Disable to skip the
# classifier call and save one Gemini request/turn.
$env:AGENT_IMAGE_GEN = "0"   # disable

# Cheap-storage opt-in for generated images. Unset => images are shown/downloaded
# live but NOT persisted (only the prompt is remembered in the image-generations
# skill). Set to a Vercel Blob RW token => images are uploaded and the URL is
# saved, so they reappear on session reload.
$env:BLOB_READ_WRITE_TOKEN = "vercel_blob_rw_..."
```

No test suite, no linter config. Manual testing via CLI or `/api/chat`.

## Architecture

Self-editing agent on Gemini. Per-user filesystem state under `data/users/<user_id>/`. All real logic lives in [agent.py](agent.py); [server.py](server.py) is a thin FastAPI wrapper.

### Three-tier skill memory

Skills are markdown files with YAML frontmatter (`name`, `description`). Tier == directory:

- `skills/system/*.md` — **always** loaded into every turn. Identity, hard rules, response style.
- `skills/*.md` (top-level) — **active** tier. Loaded when frontmatter matches the prompt.
- `skills/archive/*.md` — **archive** tier. Loaded section-by-section; body split on `##` headings, model picks which sections.

`load_skills()` enumerates all three; top-level glob excludes subdirs so active vs archive don't collide.

### Per-turn pipeline ([agent.py](agent.py) `run_turn_events`)

1. **load** — read frontmatter of every skill (cheap; bodies not loaded yet).
2. **pick** — `pick_skills()` gives the model the catalog (name + description + tier) and asks for a JSON array of relevant names. System tier is auto-included.
3. **archive** — for each picked archive skill, `pick_archive_sections()` asks the model which `##` sections to load (second cheap call per archive hit).
4. **respond** — `respond()` builds the prompt with system bodies + active bodies + archive excerpts, plus a windowed history (`HISTORY_TURN_CAP`), and calls Gemini with `google_search` grounding enabled.
5. **persist** — append user + model turns to `sessions/<sid>.jsonl`.
6. **reflect** — `reflect_and_edit()` emits skill edits (`create` / `update`, tier, name, description, body). **Cost-shaped inputs**: the full catalog is sent as name+description inventory only; full bodies are sent only for `scoped_skills` (system + picked active). Archive bodies are explicitly excluded — archive is cold storage, only the consolidation pass mutates it.
7. **apply** — `apply_edits()` writes files. `_strip_leading_frontmatter()` defends against the model embedding a YAML block inside `body`.

`run_turn_events` is a generator yielding `{stage, msg, ...}` dicts; the server streams these as NDJSON. `run_turn` drains it for CLI.

### Session-only attachments

Uploads (PDF, images, .xlsx/.csv spreadsheets, text docs) are **never stored**.
The browser keeps files in JS memory (`sessionAttachments` in
[static/index.html](static/index.html)) and re-sends them with every `/api/chat`
call so multi-turn Q&A over a document works; they vanish on reload/new
session. Server side, `prepare_attachments()` ([agent.py](agent.py)) decodes +
validates (5 files, 8MB each, 16MB total), `attachment_parts()` turns them into
Gemini parts — PDF/image as native bytes, text inline, .xlsx converted to CSV
text via openpyxl (capped at `ATTACH_SHEET_ROW_CAP` rows/sheet). Only a
`[attached: name (mime, size)]` marker line is appended to the persisted user
turn, so reflect/session-summary still record what was discussed. Attachment
turns skip the image-generation fast path. CLI: `/attach <path>` adds a file
(re-sent each turn), `/detach` clears.

### Image generation fast-path

Before step 2, `run_turn_events` runs `detect_image_intent()` (cheap classifier on the user's chat model). On a hit it routes to `generate_image()` (model `IMAGE_MODEL` = `gemini-3.1-flash-image`, `response_modalities=[TEXT, IMAGE]`) and **returns early** — no pick/archive/respond/reflect. The image bytes ride in the final `done` event (base64) for live display + download; they are **never written to the turns table**. Only a prompt note is persisted via `_remember_image()` into an `image-generations` active skill. `_store_image_blob()` is the opt-in "cheap storage" path (Vercel Blob, gated on `BLOB_READ_WRITE_TOKEN`): when present the image URL is uploaded, embedded in the saved turn as markdown, and logged in the skill so it survives reload. Toggle the whole feature with `AGENT_IMAGE_GEN`.

### Session lifecycle

Each chat creates `sessions/<16-hex>.jsonl` (one JSON record per line). On `/api/session/end` (or Ctrl-C in CLI), `summarize_session_to_skill()` produces a `session-<timestamp>` active skill and appends a `{"_rollup": true}` marker line for idempotency.

### Consolidation (offline)

`consolidate()` folds all `session-*` active skills into one `sessions-archive-<timestamp>` archive skill (one `##` section per session). **Non-destructive**: writes to `skills.consolidated/` side path, leaves the live tree untouched. Idempotent via `.consolidated` marker comparing input file mtimes. Manual swap required.

### Auth (single seam)

`/api/login/google` accepts a Google ID-token (JS GSI flow), verifies it with `google.oauth2.id_token.verify_oauth2_token` against `GOOGLE_CLIENT_ID`, derives `user_id` from the email local-part (strip `+tag`, replace `.` with `-`, lowercase), and issues a signed cookie via `itsdangerous.URLSafeSerializer` keyed on `get_secret_key()` (env `SECRET_KEY` → `.secret_key` file → freshly generated). `current_user` dependency validates the cookie signature + `user_id` regex on every request — no per-request token re-verification. Profile metadata (email/name/picture) is cached at `data/users/<user_id>/profile.json`.

Cookie payload shape is `{"user_id": "..."}`. Downstream code is identity-source-agnostic. Bind to `127.0.0.1` only.

### Model

Default `MODEL = "gemini-3.5-flash"` in [agent.py](agent.py). All users run the same model — per-user override removed with mock_users.json.

### Token logging

Every Gemini call passes through `_log_usage()` which prints `[tokens <label>] in= out= [thoughts=] total=`. Useful for debugging reflect-cost regressions.

## Conventions

- `user_id` and `session_id` validated against `^[A-Za-z0-9_-]{1,64}$` before any path construction. Never bypass `validate_user_id()` / `_session_path()`.
- Skill `body` is markdown only — no embedded frontmatter. `_strip_leading_frontmatter()` is a safety net, not an excuse.
- Active tier holds many small skills (`user-profile`, `tech-stack`, `project-<name>`); see `EDIT_INSTRUCTIONS` in [agent.py](agent.py) for the prompt that governs this.
- `skills.old/` is legacy and not loaded.
