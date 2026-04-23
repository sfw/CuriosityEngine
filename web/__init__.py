"""Curiosity Engine web UI.

FastAPI + Jinja2 + htmx + sigma.js (via CDN). Single-user, local-only,
listens on 127.0.0.1:8000 by default. Shares state with the CLI engine via
the same ~/.CuriosityEngine/engine.toml config and ./data/ journal files.
"""
