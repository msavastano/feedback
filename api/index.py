"""Vercel entrypoint. Re-exports the FastAPI app from server.py at the repo root."""
import sys
from pathlib import Path

# Make repo-root modules (server, agent, store) importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app  # noqa: E402,F401
