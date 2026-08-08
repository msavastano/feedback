# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# install
pip install -r requirements.txt
$env:GEMINI_API_KEY = "..."   # or GOOGLE_API_KEY (Gemini models + image gen)
$env:ANTHROPIC_API_KEY = "..." # Claude Haiku 4.5 chat model (optional)
$env:GOOGLE_CLIENT_ID = "..." # OAuth 2.0 Web client ID from Google Cloud Console
                              # authorized JS origin: http://127.0.0.1:8000

# CLI model selection (which provider the single-process CLI talks to). Defaults
# to MODEL (gemini-3.6-flash); set to a Claude id to exercise the Anthropic path.
$env:AGENT_MODEL = "claude-haiku-4-5"  # or "claude-sonnet-5"; needs ANTHROPIC_API_KEY, no Gemini key required

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
2. **pick** — `pick_skills()` gives the model the catalog (name + description + tier) plus the last 4 turns of the session (so pronoun-only follow-ups still match) and asks for a JSON array of `{name, score}` objects. System tier is auto-included. `_parse_picks()` normalises the reply (tolerates bare strings, drops+logs unknown names, dedupes, clamps 0–1) and sorts best-first; that ranking is preserved downstream. The score is the model's own confidence in the same call — a relative ranking for inspection/plotting, **not** a calibrated probability, and nothing thresholds on it. Scores ride in the `loaded.scores` map on the stream's `respond`/`done` events, show in the UI badges and the CLI `[loaded …]` line, and persist to `turns.picked` (JSONB, model rows only) as `{"scores": {...}, "catalog": <candidates offered>}` — never read back by the agent itself, but `store.skill_graph()` reads it for `GET /api/graph`, which backs the UI's **Map** modal — a 3D skill co-activation graph rendered with three.js (import map + module script at the end of [static/index.html](static/index.html)), laid out by `d3-force-3d`. A synthetic `memory` hub is pinned at the origin; system skills form an inner core, picked skills a middle band, unpicked ones the outer shell. Materials, radius ratios (3 : 1.82 : 1) and the hub ring come verbatim from `design-files/mind-map.{mtl,glb}` — glTF base colours are LINEAR, so they are set via `setRGB(..., LinearSRGBColorSpace)`, not as hex. Link colour encodes co-activation weight (violet = hub spine, saffron ≥ 2, teal = 1). The modal is always dark; the design's materials are lit for a dark stage and do not follow the app theme. Note `picked` was added after real history existed, so pick counts cover only turns recorded since then — `/api/graph` returns `turns` / `model_turns` / `since` so the UI can scope its claims instead of implying a lifetime count. Query it directly for retrieval drift:

```sql
-- how often each skill gets picked, and how confidently
SELECT k.key AS skill, count(*) AS picks, round(avg((k.value)::numeric), 2) AS avg_score
FROM turns t, jsonb_each_text(t.picked->'scores') k
WHERE t.user_id = 'alice'
GROUP BY 1 ORDER BY picks DESC;
```
2b. **fallback** — the picker only ever sees names and descriptions, so a fact filed inside a skill about another subject is invisible to it and the description becomes that fact's single point of failure. When `pick_skills()` returns **nothing**, `fallback_body_picks()` runs one Postgres full-text OR-query over `name || description || body` (`SkillStore.search_bodies`, ranked by `ts_rank`, top 3, system tier excluded) and uses the hits instead. No LLM call. `_search_terms()` reduces the prompt to ≤8 distinctive alphanumeric tokens — alphanumeric because the terms are interpolated into a `to_tsquery`, where a stray `&`/`|`/`:`/`!` would be a syntax error, not a miss (see [test_search_terms.py](test_search_terms.py)). Fallback hits carry `FALLBACK_SCORE = 0.1` so they're distinguishable from real picker confidences everywhere scores surface. Any exception is caught and logged — a search miss must never cost the user their turn.

3. **archive** — for each picked archive skill, `pick_archive_sections()` asks the model which `##` sections to load (second cheap call per archive hit).
4. **respond** — `respond()` builds the prompt with system bodies + active bodies + archive excerpts, plus a windowed history (`HISTORY_TURN_CAP`), and calls Gemini with `google_search` grounding enabled.
5. **persist** — append user + model turns to `sessions/<sid>.jsonl`.
6. **reflect** — `reflect_and_edit()` emits skill edits (`create` / `update`, tier, name, description, body). **Cost-shaped inputs**: the full catalog is sent as name+description inventory only; full bodies are sent only for `scoped_skills` (system + picked active). Archive bodies are explicitly excluded — archive is cold storage, only the consolidation pass mutates it.
7. **apply** — `apply_edits()` writes files. `_strip_leading_frontmatter()` defends against the model embedding a YAML block inside `body`. Every skill update/delete snapshots the pre-image row into the `skill_versions` table (write-only audit trail; recover via SQL).

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
turn, so reflect/session-summary still record what was discussed. Turns with
any non-image attachment (pdf, text, spreadsheet) skip the image-generation
fast path; image-only attachments stay eligible and ride into `generate_image()`
as reference inputs (image-to-image). CLI: `/attach <path>` adds a file
(re-sent each turn), `/detach` clears.

### Image generation fast-path

Before step 2, `run_turn_events` runs `detect_image_intent()` (cheap classifier on the user's chat model; a reference-image variant of the instructions is used when image attachments are present, so "redraw me as X" over an upload counts as generation). On a hit it routes to `generate_image()` (model `IMAGE_MODEL` = `gemini-3.1-flash-image`, `response_modalities=[TEXT, IMAGE]`; attached images are passed as input parts for image-to-image) and **returns early** — no pick/archive/respond/reflect. The image bytes ride in the final `done` event (base64) for live display + download; they are **never written to the turns table**. Only a prompt note is persisted via `_remember_image()` into an `image-generations` active skill. `_store_image_blob()` is the opt-in "cheap storage" path (Vercel Blob, gated on `BLOB_READ_WRITE_TOKEN`): when present the image URL is uploaded, embedded in the saved turn as markdown, and logged in the skill so it survives reload. Toggle the whole feature with `AGENT_IMAGE_GEN`.

### Session lifecycle

Each chat creates `sessions/<16-hex>.jsonl` (one JSON record per line). On `/api/session/end` (or Ctrl-C in CLI), `summarize_session_to_skill()` produces a `session-<timestamp>` active skill and appends a `{"_rollup": true}` marker line for idempotency.

### Consolidation (offline)

`consolidate()` folds all `session-*` active skills into one `sessions-archive-<timestamp>` archive skill (one `##` section per session, headed `<name> — <description>` so `pick_archive_sections` can match on topic, not timestamp), then deletes the originals from the live store.

**Lossless despite the delete**: the fold is verbatim string concatenation of `s.body` — no LLM call, so nothing is summarized away — and `SkillStore.delete` snapshots each pre-image into `skill_versions` first. Idempotent because retired sessions no longer match the `active` + `session-*` filter. Each run mints a *new* timestamped archive skill rather than extending the previous one, so the archive tier grows by one skill per run (49 of them on the main user as of Aug 2026).

(Pre-Postgres this wrote to a `skills.consolidated/` side path and required a manual swap. That path is gone; `--dry-run` is the preview now.)

### Auth (single seam)

`/api/login/google` accepts a Google ID-token (JS GSI flow), verifies it with `google.oauth2.id_token.verify_oauth2_token` against `GOOGLE_CLIENT_ID`, derives `user_id` from the email local-part (strip `+tag`, replace `.` with `-`, lowercase), and issues a signed cookie via `itsdangerous.URLSafeSerializer` keyed on `get_secret_key()` (env `SECRET_KEY` → `.secret_key` file → freshly generated). `current_user` dependency validates the cookie signature + `user_id` regex on every request — no per-request token re-verification. Profile metadata (email/name/picture) is cached at `data/users/<user_id>/profile.json`.

Cookie payload shape is `{"user_id": "..."}`. Downstream code is identity-source-agnostic. Bind to `127.0.0.1` only.

### Model providers

Default `MODEL = "gemini-3.6-flash"` in [agent.py](agent.py). `ALLOWED_MODELS` also
includes two Anthropic models, `CLAUDE_MODEL = "claude-haiku-4-5"` and
`CLAUDE_SONNET_MODEL = "claude-sonnet-5"`. `provider_of(model)` routes a chat
model to its SDK by id prefix (`claude*` → Anthropic, else Gemini), so both
Claude ids share the same v1 path (web search, no image gen / code exec).

A `Clients` dataclass (`{gemini, anthropic}`, either may be `None`) is threaded
where the single `client` used to be. Two adapters absorb all provider branching
so the seven helpers stay thin: `_chat_json` (pick / archive / image-intent /
reflect / session-summary) and `_chat_respond` (the `respond` tail). The Gemini
branch is unchanged (`_generate` + `google_search` + `ThinkingConfig`, plus
optional code execution). The Claude branch calls `messages.create` with the
server-side `web_search_20250305` tool, loops on `stop_reason == "pause_turn"`,
and parses JSON via `_extract_json` (fence/balance-tolerant — Haiku structured
output isn't assumed). Anthropic usage feeds the same `[tokens <label>]` print
via `_log_anthropic_usage`.

**Claude path is v1-scoped:** no image generation (the fast-path is gated to
Gemini — a Claude user needs no Gemini key), no code execution. `thinking_level`
maps to manual extended thinking (`thinking.budget_tokens`) only for
`CLAUDE_MODEL` (Haiku 4.5) via `CLAUDE_THINKING_BUDGETS`; Sonnet 5 replaced
manual budget_tokens with always-on adaptive thinking + an `effort` param (not
wired here) and 400s if sent, so it's excluded from `CLAUDE_THINKING_MODELS`
and the field is a no-op there. `THINKING_CAPABLE_MODELS` (Gemini models +
Haiku) is exposed via `/api/me` as `thinking_capable_models` so the UI can
disable the "think" dropdown (shows "NA") for models where it wouldn't do
anything, e.g. Sonnet 5. BYOK: the web path takes the Anthropic key via the `X-Anthropic-Key`
header (mirrors `X-Gemini-Key`); `api_chat` requires the key for the chosen
model's provider. `api_session_end` takes the session's `model` so the rollup
summary routes to the right provider.

### Token logging

Every Gemini call passes through `_log_usage()` which prints `[tokens <label>] in= out= [thoughts=] total=`. Useful for debugging reflect-cost regressions.

## Conventions

- `user_id` and `session_id` validated against `^[A-Za-z0-9_-]{1,64}$` before any path construction. Never bypass `validate_user_id()` / `_session_path()`.
- Skill `body` is markdown only — no embedded frontmatter. `_strip_leading_frontmatter()` is a safety net, not an excuse.
- Active tier holds many small skills (`user-profile`, `tech-stack`, `project-<name>`); see `EDIT_INSTRUCTIONS` in [agent.py](agent.py) for the prompt that governs this.
- `skills.old/` is legacy and not loaded.

## Memory hygiene rules (why they're in the prompts)

Three rules in the writer prompts exist to counter documented failure modes of self-editing filesystem memory ([arXiv 2607.26637](https://arxiv.org/abs/2607.26637), which studies exactly this store class). Don't relax them without re-reading that:

- **Supersession** (`EDIT_INSTRUCTIONS`) — when a new fact contradicts a stored one, retire the old line with a date and add the new one; never silently overwrite. Counters "stale facts standing as live traits", worst in the `system` tier, which is injected every turn with no relevance check. `reflect_and_edit` passes `TODAY` into the prompt so relative phrases ("yesterday") resolve before storage. Baseline when this landed: 4 of 107 bodies carried any ISO date.
- **Keep every fact** (`SESSION_INSTRUCTIONS`) — the session rollup *replaces* the transcript as memory, so the bullet count is a target, not a budget. Counters silent condensation, the paper's one degenerate-by-default behavior (measured at ~2x correctness loss on one benchmark). Tightening wording is fine; dropping a name/number/date/decision is not.
- **Delete is for containers, never facts** (`EDIT_INSTRUCTIONS`) — `apply_edits` has always handled `op="delete"`, but the prompt didn't document it, so it was dead code and the store could only grow. It's now documented, scoped to redundant skills whose content was merged elsewhere.

Store shape as of Aug 2026 (main user): 107 skills / 129 KB — 2 system, 52 active, 53 archive; 58 subject skills, 49 `sessions-archive-*`. Note the subject names encode a taxonomy the flat namespace can't hold (`interest-sports-rules-fifa`, `-fifa-subs`, `-fifa-tie-breakers`). Byte size is comparable to the paper's curated stores; file count is ~40x theirs.
