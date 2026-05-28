-- Skill-memory agent: Postgres schema for Vercel deployment.
-- Run once against your DATABASE_URL (Vercel Postgres / Neon).
--   psql "$DATABASE_URL" -f schema.sql

CREATE TABLE IF NOT EXISTS users (
  user_id    TEXT PRIMARY KEY,
  email      TEXT NOT NULL,
  name       TEXT NOT NULL DEFAULT '',
  sub        TEXT NOT NULL DEFAULT '',          -- Google subject id
  picture    TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS skills (
  user_id     TEXT NOT NULL,
  tier        TEXT NOT NULL CHECK (tier IN ('system','active','archive')),
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  body        TEXT NOT NULL DEFAULT '',
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, name)
);
CREATE INDEX IF NOT EXISTS skills_catalog ON skills (user_id, tier);

CREATE TABLE IF NOT EXISTS sessions (
  user_id     TEXT NOT NULL,
  session_id  TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at    TIMESTAMPTZ,
  rolled_up   BOOLEAN NOT NULL DEFAULT FALSE,
  rollup_skill TEXT,
  PRIMARY KEY (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS sessions_recent ON sessions (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS turns (
  user_id    TEXT NOT NULL,
  session_id TEXT NOT NULL,
  idx        INTEGER NOT NULL,                  -- monotonic per session
  ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
  role       TEXT NOT NULL CHECK (role IN ('user','model')),
  text       TEXT NOT NULL,
  tokens     JSONB,
  PRIMARY KEY (user_id, session_id, idx),
  FOREIGN KEY (user_id, session_id)
    REFERENCES sessions(user_id, session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS turns_session ON turns (user_id, session_id, idx);
