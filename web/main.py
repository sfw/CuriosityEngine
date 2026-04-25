"""Curiosity Engine — FastAPI web UI.

Mostly thin glue over the existing engine package. Reads journals via the `Journal`
class, writes via the same methods `--set-focus`/`--add-question`/`--review-register`
would use in CLI mode. Long-running cycles are spawned as subprocesses and their
stdout streamed to the browser via Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from journal import Journal
from models import CrossReference, Insight, Prediction, RegisterEntry  # noqa: F401
from config import CuriosityEngineConfig

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CURIOSITY_DATA_DIR", "/workspace"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Curiosity Engine")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# In-memory tracking of active cycle runs keyed by opaque run_id.
_active_runs: dict[str, dict] = {}


def _reconcile_orphaned_runs():
    """Scan all run-meta files at startup and mark any `status=running`
    entries as interrupted. The in-memory _active_runs map is empty on a
    fresh container start, so by definition any meta still claiming to be
    "running" is orphaned by a container restart. Without this fixup,
    stale meta files render as active runs forever in the UI but lack
    a Stop button (because they're not in _active_runs).
    """
    runs_root = DATA_DIR / "_runs"
    if not runs_root.exists():
        return
    fixed = 0
    for meta_path in runs_root.glob("*/*.meta.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if (meta.get("status") or "") != "running":
            continue
        # Found a stale "running" meta. Container restart killed its
        # subprocess; mark it interrupted with a clear note.
        meta["status"] = "failed"
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        meta["returncode"] = None  # we don't know what it would have been
        meta["interrupted_by_restart"] = True
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
            fixed += 1
        except OSError:
            pass
    if fixed:
        print(f"[startup] reconciled {fixed} orphaned run meta file(s) "
              f"(status=running with no live subprocess → marked failed/interrupted)")


# Reconcile at module load — runs once when uvicorn imports web.main.
_reconcile_orphaned_runs()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _journal_path(name: str) -> Path:
    """Resolve a journal filename to an absolute path inside DATA_DIR.

    Filters path-traversal; only allows simple names and relative paths inside DATA_DIR.
    """
    safe = Path(name).name  # strip any directory parts
    if not safe or safe.startswith("."):
        raise HTTPException(400, f"invalid journal name: {name!r}")
    if not safe.endswith(".json"):
        safe = safe + ".json"
    return DATA_DIR / safe


def _load_journal(name: str) -> Journal:
    path = _journal_path(name)
    if not path.exists():
        raise HTTPException(404, f"journal not found: {path.name}")
    return Journal(str(path))


def _list_journals() -> list[dict]:
    """Discover journal files in DATA_DIR and return a summary of each."""
    out: list[dict] = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text() or "{}")
        except (json.JSONDecodeError, OSError):
            continue
        if "entries" not in data:
            continue
        out.append({
            "name": p.stem,
            "filename": p.name,
            "entries": len(data.get("entries") or []),
            "cross_references": len(data.get("cross_references") or []),
            "insights": len(data.get("insights") or []),
            "register": len(data.get("register") or []),
            "predictions": len(data.get("predictions") or []),
            "focus": str(data.get("focus", "")),
            "last_updated": data.get("metadata", {}).get("last_updated", ""),
            "domains": sorted({t for e in data.get("entries") or [] for t in (e.get("domain_tags") or [])})[:10],
        })
    out.sort(key=lambda j: j["last_updated"], reverse=True)
    return out


def _connection() -> CuriosityEngineConfig:
    return CuriosityEngineConfig.load()


# ─────────────────────────────────────────────
# Run persistence
# ─────────────────────────────────────────────

def _runs_root() -> Path:
    d = DATA_DIR / "_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _runs_dir(journal_stem: str) -> Path:
    safe = Path(journal_stem).name
    d = _runs_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_log_path(journal_stem: str, run_id: str) -> Path:
    return _runs_dir(journal_stem) / f"{run_id}.log"


def _run_meta_path(journal_stem: str, run_id: str) -> Path:
    return _runs_dir(journal_stem) / f"{run_id}.meta.json"


def _list_runs_for_journal(journal_stem: str) -> list[dict]:
    d = _runs_dir(journal_stem)
    out = []
    for meta_path in sorted(d.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _update_run_meta(journal_stem: str, run_id: str, **updates):
    p = _run_meta_path(journal_stem, run_id)
    if not p.exists():
        return
    try:
        meta = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return
    meta.update(updates)
    p.write_text(json.dumps(meta, indent=2))


# ─────────────────────────────────────────────
# Routes: listing + overview
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "journals.html", {
        "journals": _list_journals(),
    })


@app.get("/journals", response_class=HTMLResponse)
def journals_list(request: Request):
    return templates.TemplateResponse(request, "journals.html", {
        "journals": _list_journals(),
    })


@app.get("/journals/{name}", response_class=HTMLResponse)
def journal_view(request: Request, name: str, tab: str = "overview"):
    journal = _load_journal(name)
    connection = _connection()
    return templates.TemplateResponse(request, "journal.html", {
        "name": name,
        "journal": journal,
        "tab": tab,
        "connection": connection,
        "domains": sorted({t for e in journal.entries for t in (e.get("domain_tags") or [])}),
    })


# ─────────────────────────────────────────────
# Tab partials (htmx swaps)
# ─────────────────────────────────────────────

@app.get("/journals/{name}/overview", response_class=HTMLResponse)
def journal_overview(request: Request, name: str):
    journal = _load_journal(name)
    high_surprise = sorted(
        [e for e in journal.entries if float(e.get("surprise_delta") or 0.0) >= 0.6],
        key=lambda e: float(e.get("surprise_delta") or 0.0),
        reverse=True,
    )
    recent = journal.entries[-5:][::-1]
    return templates.TemplateResponse(request, "partials/overview.html", {
        "name": name,
        "journal": journal,
        "recent": recent,
        "high_surprise": high_surprise[:5],
    })


@app.get("/journals/{name}/entries", response_class=HTMLResponse)
def journal_entries(request: Request, name: str, tag: Optional[str] = None):
    journal = _load_journal(name)
    entries = list(reversed(journal.entries))
    if tag:
        entries = [e for e in entries if tag in (e.get("domain_tags") or [])]
    domains = sorted({t for e in journal.entries for t in (e.get("domain_tags") or [])})
    return templates.TemplateResponse(request, "partials/entries.html", {
        "name": name,
        "entries": entries,
        "domains": domains,
        "filter_tag": tag or "",
    })


@app.get("/journals/{name}/entries/{entry_id}", response_class=HTMLResponse)
def journal_entry(request: Request, name: str, entry_id: str):
    journal = _load_journal(name)
    entry = next((e for e in journal.entries if e.get("id") == entry_id), None)
    if entry is None:
        raise HTTPException(404, f"entry not found: {entry_id}")
    return templates.TemplateResponse(request, "partials/entry_detail.html", {
        "name": name,
        "entry": entry,
    })


@app.get("/journals/{name}/insights", response_class=HTMLResponse)
def journal_insights(request: Request, name: str):
    journal = _load_journal(name)
    insights = list(reversed(journal.insights))
    return templates.TemplateResponse(request, "partials/insights.html", {
        "name": name,
        "insights": insights,
    })


def _effective_novelty_for_entry(entry: dict) -> str:
    """Latest reverification's new_novelty_type if the entry's been audited,
    otherwise the originally-stored novelty_type. See the audit-aware rule
    the user called out when we shipped client-side filtering — this is the
    authoritative server-side implementation."""
    rv_log = entry.get("reverification_log") or []
    if rv_log:
        last = rv_log[-1]
        nov = (last.get("new_novelty_type") or "").strip()
        if nov:
            return nov
    return (entry.get("novelty_type") or "").strip() or "unknown"


@app.get("/journals/{name}/register", response_class=HTMLResponse)
def journal_register(
    request: Request,
    name: str,
    filter_status: Optional[str] = None,
    lifecycle: Optional[str] = None,
    novelty_type: Optional[str] = None,
):
    journal = _load_journal(name)
    register = list(reversed(journal.register))
    # Compute per-entry effective novelty (audit-aware). Stored on the dict
    # so the template can filter + render without redoing the computation.
    for r in register:
        r["_effective_novelty"] = _effective_novelty_for_entry(r)
    if filter_status:
        register = [r for r in register if r.get("human_review_status", "unreviewed") == filter_status]
    if lifecycle and lifecycle in ("active", "held"):
        if lifecycle == "active":
            register = [r for r in register if r.get("status") != "held"]
        else:  # "held"
            register = [r for r in register if r.get("status") == "held"]
    # Compute novelty counts over the lifecycle/review-filtered set BEFORE
    # applying the novelty filter, so the filter-bar counts reflect the full
    # pool the user could switch to (not just the current slice).
    novelty_counts: dict[str, int] = {}
    for r in register:
        eff = r.get("_effective_novelty") or "unknown"
        novelty_counts[eff] = novelty_counts.get(eff, 0) + 1
    total_for_all_button = len(register)
    # Novelty filter — uses effective_novelty (post-audit source of truth).
    if novelty_type:
        nt = novelty_type.strip()
        if nt:
            register = [r for r in register if r.get("_effective_novelty") == nt]
    held_count = sum(1 for r in journal.register if r.get("status") == "held")
    active_count = len(journal.register) - held_count
    # Map register-id → directive sidecar verdict for entries with a generated
    # directive on disk. Lets the template render a "View directive" link
    # alongside the export button + a verdict chip (clean / needs_fixes / fatal).
    directive_status = _directive_status_map(name)
    return templates.TemplateResponse(request, "partials/register.html", {
        "name": name,
        "register": register,
        "predictions": journal.predictions,
        "filter_status": filter_status or "",
        "lifecycle": lifecycle or "",
        "novelty_type": novelty_type or "",
        "novelty_counts": novelty_counts,
        "novelty_total": total_for_all_button,
        "held_count": held_count,
        "active_count": active_count,
        "directive_status": directive_status,
    })


def _directive_status_map(name: str) -> dict:
    """Walk the journal's directives directory and build a map of
    register_id → {filename, verdict, flagged_issues_count, generated_at}.
    The verdict comes from the .verification.json sidecar when present,
    so the register UI can render a status chip without re-parsing the markdown.
    """
    journal_path = _journal_path(name)
    d = journal_path.parent / f"{journal_path.stem}_directives"
    if not d.exists():
        return {}
    out: dict = {}
    for md_file in d.glob("r-*.md"):
        rid = md_file.stem  # "r-<id>"
        sidecar = md_file.with_suffix(".verification.json")
        info: dict = {"filename": md_file.name, "verdict": "", "flagged_issues_count": 0}
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    s = json.load(f)
                info["verdict"] = s.get("verdict", "")
                info["flagged_issues_count"] = len(s.get("flagged_issues") or [])
                info["generated_at"] = s.get("generated_at", "")
            except (OSError, json.JSONDecodeError):
                pass
        out[rid] = info
    return out


@app.post("/journals/{name}/register/{entry_id}/promote")
def promote_register_entry(
    name: str,
    entry_id: str,
    reviewer: str = Form(""),
    convert_triggers: str = Form(""),
):
    """Promote a held register entry to active. Optionally convert
    settlement_triggers into Predictions so --check-predictions can track them."""
    journal = _load_journal(name)
    entry = next((r for r in journal.register if r.get("id") == entry_id), None)
    if entry is None:
        raise HTTPException(404, f"register entry not found: {entry_id}")
    if entry.get("status") != "held":
        raise HTTPException(400, f"entry {entry_id} is not in held state")
    promoted = journal.promote_register_entry(
        entry_id, promoted_by=f"human:{(reviewer or 'unknown').strip()}",
    )
    if not promoted:
        raise HTTPException(500, "promotion failed")

    # Optionally persist settlement_triggers as predictions.
    if convert_triggers.strip():
        from uuid import uuid4 as _uuid4
        horizon = (entry.get("settlement_horizon") or "").strip()
        for t in (entry.get("settlement_triggers") or []):
            trigger = (t or "").strip()
            if not trigger:
                continue
            pred = Prediction(
                id=f"p-{_uuid4().hex[:8]}",
                register_entry_id=entry_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                target_date=horizon,
                claim=f"Settlement trigger for promoted entry: {trigger}",
                falsifiable_condition=trigger,
                check_method="Reuse the settlement_method recorded on the originally-held register entry.",
            )
            journal.add_prediction(pred)

    # Also mark human_review_status so the entry doesn't clutter the unreviewed list.
    journal.update_register_entry_review(
        entry_id, status="approved", notes="promoted from held", reviewer=reviewer,
    )
    return RedirectResponse(f"/journals/{name}?tab=register", status_code=303)


async def _spawn_maintenance_subprocess(
    journal_path: Path,
    extra_args: list[str],
    *,
    kind: str,
) -> dict:
    """Spawn a curiosity_engine.py subprocess for a maintenance operation and
    register it with the standard run-tracking infrastructure (shared meta/log,
    top-bar indicator, Runs tab stream).

    `extra_args` is appended after `--journal <path>`. `kind` is stored on the
    meta file so the UI can distinguish reverify/synth-orphan/cross-ref-only/
    check-predictions from actual cycle runs.
    """
    if not journal_path.exists():
        raise HTTPException(404, f"journal not found: {journal_path.name}")

    cmd = ["python", "/app/curiosity_engine.py", "--journal", str(journal_path)]
    cmd.extend(extra_args)

    run_id = uuid.uuid4().hex
    journal_stem = journal_path.stem
    meta = {
        "run_id": run_id,
        "journal": journal_stem,
        "cmd": " ".join(cmd),
        "domain": "",
        "cycles": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "returncode": None,
        "status": "running",
        "kind": kind,
    }
    _run_meta_path(journal_stem, run_id).write_text(json.dumps(meta, indent=2))
    log_path = _run_log_path(journal_stem, run_id)
    log_path.write_text(f"$ {' '.join(cmd)}\n")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd="/workspace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _active_runs[run_id] = {
        "proc": proc,
        "lines": [],
        "done": False,
        "cmd": " ".join(cmd),
        "journal": journal_stem,
        "log_path": log_path,
    }
    asyncio.create_task(_collect_run_output(run_id, proc))
    return {"run_id": run_id, "cmd": " ".join(cmd), "journal": journal_stem}


@app.post("/journals/{name}/runs/{run_id}/kill")
async def kill_run(name: str, run_id: str):
    """Stop a streaming run by sending SIGTERM to its subprocess. If the
    process doesn't exit within 3s, escalates to SIGKILL. Updates run
    meta so the Runs tab shows the run as failed/killed once refreshed."""
    state = _active_runs.get(run_id)
    if state is None or state.get("done"):
        return JSONResponse(
            {"ok": False, "reason": "run not found in active set (already completed?)"},
            status_code=404,
        )
    proc = state.get("proc")
    if proc is None:
        return JSONResponse({"ok": False, "reason": "no process handle"}, status_code=500)

    journal_stem = state.get("journal", "")

    # Try graceful shutdown first.
    try:
        proc.terminate()
    except ProcessLookupError:
        pass  # already dead

    # Wait up to 3s for graceful exit, then SIGKILL.
    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
        kind = "terminated"
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        kind = "killed"

    # Mark the run as failed in meta — so the Runs tab badges it correctly
    # on next render. The collector task in _collect_run_output will also
    # set returncode + completed_at when it observes the proc finish.
    if journal_stem:
        _update_run_meta(
            journal_stem, run_id,
            status="failed",
            killed_by_user=True,
        )
    return JSONResponse({"ok": True, "outcome": kind, "run_id": run_id})


@app.post("/journals/{name}/insights/reverify")
async def insights_reverify(name: str, insight_id: str = Form("")):
    """Reverify unregistered insights (or a specific one) under current rules."""
    extra = ["--reverify-insight", insight_id.strip()] if insight_id.strip() else ["--reverify-insights"]
    result = await _spawn_maintenance_subprocess(_journal_path(name), extra, kind="reverify")
    return JSONResponse(result)


@app.post("/journals/{name}/maintenance/cross-ref")
async def maintenance_cross_ref(
    name: str,
    cross_ref_window: str = Form(""),
    cross_ref_role: str = Form(""),
    primary_role: str = Form(""),
    verifier_role: str = Form(""),
):
    """Re-run the cross-reference phase: generates NEW xrefs from the current
    journal state, synthesizes high-novelty ones into insights, and verifies
    them. Dedup is handled by the cross-ref phase's existing participant-set
    check + anti-attractor gate, so it's safe to re-run repeatedly.

    Optional overrides (only propagated when they differ from defaults):
    cross_ref_window slims the prompt; primary_role/verifier_role copy a
    configured profile (by role — e.g. 'verifier') into the slot, swapping
    the whole provider/base_url/api_key/name, not just the model name.
    """
    extra = ["--cross-ref-only"]
    conn = _connection()

    raw = (cross_ref_window or "").strip()
    if raw:
        try:
            v = int(raw)
            if v != int(conn.engine.cross_ref_window):
                extra += ["--cross-ref-window", str(v)]
        except ValueError:
            pass

    # Dirty-detection against the *currently effective* cross_ref role (engine.toml
    # default). If user didn't change the dropdown, no flag is sent.
    effective_cr_role = "primary"
    if conn.cross_ref is not None:
        if conn.cross_ref is conn.verifier or conn.cross_ref.name == conn.verifier.name:
            effective_cr_role = "verifier"
        else:
            effective_cr_role = "cross_ref"  # configured via [models.cross_ref]
    cr = cross_ref_role.strip().lower()
    if cr and cr != effective_cr_role:
        extra += ["--cross-ref-role", cr]

    pr = primary_role.strip().lower()
    if pr and pr != "primary":
        extra += ["--primary-role", pr]
    vr = verifier_role.strip().lower()
    if vr and vr != "verifier":
        extra += ["--verifier-role", vr]

    result = await _spawn_maintenance_subprocess(
        _journal_path(name), extra, kind="cross-ref-only",
    )
    return JSONResponse(result)


@app.post("/journals/{name}/maintenance/synth-orphaned")
async def maintenance_synth_orphaned(name: str):
    """Synthesize + verify every cross-reference that doesn't yet have a
    matching insight (e.g. after a mid-run failure between cross-ref and
    synthesis). Inherently dedup'd: skips any xref already in an insight's
    supporting_evidence."""
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), ["--synth-orphaned-xrefs"], kind="synth-orphaned",
    )
    return JSONResponse(result)


@app.post("/journals/{name}/maintenance/reverify")
async def maintenance_reverify(name: str):
    """Batch re-verify every unregistered insight. Equivalent to the
    `Re-verify unregistered` button on the Insights tab."""
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), ["--reverify-insights"], kind="reverify",
    )
    return JSONResponse(result)


@app.post("/journals/{name}/known-prior-art/add")
def add_known_prior_art(
    name: str,
    domain: str = Form(...),
    system_name: str = Form(...),
    url: str = Form(""),
    notes: str = Form(""),
):
    """Append a known-prior-art anchor. The verifier will evaluate it on
    every future register-candidate whose domain matches. See
    journal.add_known_prior_art for the matching rule."""
    journal = _load_journal(name)
    journal.add_known_prior_art(
        domain=domain, system_name=system_name, url=url, notes=notes,
    )
    return RedirectResponse(f"/journals/{name}?tab=admin", status_code=303)


@app.post("/journals/{name}/known-prior-art/{entry_id}/delete")
def delete_known_prior_art(name: str, entry_id: str):
    """Remove a known-prior-art anchor by id."""
    journal = _load_journal(name)
    journal.remove_known_prior_art(entry_id)
    return RedirectResponse(f"/journals/{name}?tab=admin", status_code=303)


@app.post("/journals/{name}/maintenance/reverify-register")
async def maintenance_reverify_register(
    name: str,
    max_confidence: str = Form(""),
    novelty_types: str = Form(""),
):
    """Batch re-verify existing register entries (audit under updated rules).
    Appends reverification_log to each entry; never overwrites the original
    verdict. Optional filters: max_confidence (float 0-1), novelty_types
    (comma-sep list, e.g. 'new_synthesis,correction')."""
    extra: list[str] = ["--reverify-register"]
    mc = (max_confidence or "").strip()
    if mc:
        try:
            float(mc)
            extra.extend(["--reverify-register-max-confidence", mc])
        except ValueError:
            pass
    nt = (novelty_types or "").strip()
    if nt:
        extra.extend(["--reverify-register-novelty-types", nt])
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), extra, kind="reverify-register",
    )
    return JSONResponse(result)


@app.post("/journals/{name}/register/{entry_id}/directive")
async def export_register_directive(name: str, entry_id: str):
    """Generate a research directive (markdown) for one register entry.
    Spawns `curiosity_engine.py --export-directive <id>`; streams log to
    the live-log panel. Output written to data/{journal}_directives/."""
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), ["--export-directive", entry_id], kind="export-directive",
    )
    return JSONResponse(result)


@app.post("/journals/{name}/maintenance/export-directives-bundle")
async def maintenance_export_directives_bundle(name: str):
    """Generate a research-directives bundle covering all qualifying entries.
    Slow — runs the primary+verifier pipeline once per qualifying entry.
    Output: data/{journal}_directives/bundle-<timestamp>.md + sidecar."""
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), ["--export-directives-bundle"], kind="export-directives-bundle",
    )
    return JSONResponse(result)


@app.get("/journals/{name}/directives/{filename}")
def serve_directive_file(name: str, filename: str):
    """Serve a generated directive markdown (or sidecar JSON) for download/view.
    Path sanitised against traversal. Only files in the journal's directives/
    directory are reachable."""
    journal_path = _journal_path(name)
    directives_dir = journal_path.parent / f"{journal_path.stem}_directives"
    # Sanitise: reject any path traversal attempts. Only a plain filename
    # (no slashes, no parent refs) is allowed.
    if "/" in filename or "\\" in filename or ".." in filename:
        return PlainTextResponse("invalid filename", status_code=400)
    target = directives_dir / filename
    if not target.exists() or not target.is_file():
        return PlainTextResponse("not found", status_code=404)
    content = target.read_text()
    # Use text/markdown for .md so browsers don't download-prompt uselessly;
    # JSON for sidecars.
    if filename.endswith(".json"):
        return PlainTextResponse(content, media_type="application/json")
    return PlainTextResponse(content, media_type="text/markdown")


@app.post("/journals/{name}/maintenance/scan-gaps")
async def maintenance_scan_gaps(name: str):
    """Run a negative-space scan on this journal. Spawns `curiosity_engine.py
    --scan-gaps`. Gated server-side by [engine].negative_space_min_entries —
    the engine aborts with a message if the journal is too young."""
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), ["--scan-gaps"], kind="scan-gaps",
    )
    return JSONResponse(result)


@app.get("/journals/{name}/coverage", response_class=HTMLResponse)
def journal_coverage(request: Request, name: str):
    """Coverage tab — shows the latest negative-space scan as a method × problem
    matrix with empty cells color-coded by classification. When ≥2 scans exist,
    also surfaces a diff against the prior scan: gaps filled, still open, newly
    emerged."""
    journal = _load_journal(name)
    scans = journal.coverage_scans
    latest = scans[-1] if scans else None
    diff = None
    if len(scans) >= 2:
        from engine.negative_space import diff_coverage_scans
        diff = diff_coverage_scans(scans[-2], scans[-1])
    return templates.TemplateResponse(request, "partials/coverage.html", {
        "name": name,
        "scans": list(reversed(scans)),  # newest first in list view
        "latest": latest,
        "diff": diff,
        "min_entries": int(_connection().engine.negative_space_min_entries),
        "current_entries": len(journal.entries),
    })


@app.post("/journals/{name}/maintenance/check-predictions")
async def maintenance_check_predictions(name: str, all_pending: str = Form("")):
    """Check due predictions against current reality (verifier + tools).
    When `all_pending` is truthy, checks every pending prediction regardless
    of target_date."""
    flag = "--check-predictions-all" if all_pending.strip() else "--check-predictions"
    result = await _spawn_maintenance_subprocess(
        _journal_path(name), [flag], kind="check-predictions",
    )
    return JSONResponse(result)


@app.get("/journals/{name}/admin", response_class=HTMLResponse)
def journal_admin(request: Request, name: str):
    """Admin tab — consolidated maintenance operations with counters showing
    how much work each one has to do (so no-ops are visible up front)."""
    journal = _load_journal(name)
    conn = _connection()

    # Orphaned xrefs: xref.id not in any insight.supporting_evidence.
    insight_supports: set[str] = set()
    for i in journal.insights:
        for sid in (i.get("supporting_evidence") or []):
            insight_supports.add(sid)
    threshold = float(conn.engine.novelty_threshold)
    orphan_xrefs = [
        x for x in journal.cross_references
        if x.get("id") and x.get("id") not in insight_supports
        and float(x.get("novelty_score", 0.0) or 0.0) >= threshold
    ]

    # Unregistered insights: insight.id not in any register.insight_id
    registered_iids = {r.get("insight_id") for r in journal.register}
    unregistered_insights = [
        i for i in journal.insights if i.get("id") not in registered_iids
    ]

    # Predictions due
    due_predictions = journal.due_predictions(include_overdue=True)
    pending_predictions = [p for p in journal.predictions if p.get("status") == "pending"]

    # Held entries awaiting settlement
    held_entries = [r for r in journal.register if r.get("status") == "held"]

    # Model profiles for the cross-ref override dropdowns.
    model_profiles: list[dict] = [
        {"role": "primary",  "name": conn.primary.name},
        {"role": "verifier", "name": conn.verifier.name},
    ]
    for role, profile in (getattr(conn, "extras", {}) or {}).items():
        model_profiles.append({"role": role, "name": profile.name})

    # Effective cross_ref role: whatever engine.toml currently resolves to.
    # This is what the "Cross-ref model" dropdown defaults to; dirty-detection
    # only fires when the user picks something different.
    effective_cr_role = "primary"
    if conn.cross_ref is not None:
        if conn.cross_ref.name == conn.verifier.name and conn.cross_ref.base_url == conn.verifier.base_url:
            effective_cr_role = "verifier"
        else:
            for role, profile in (getattr(conn, "extras", {}) or {}).items():
                if profile.name == conn.cross_ref.name and profile.base_url == conn.cross_ref.base_url:
                    effective_cr_role = role
                    break

    return templates.TemplateResponse(request, "partials/admin.html", {
        "name": name,
        "orphan_xref_count": len(orphan_xrefs),
        "unregistered_insight_count": len(unregistered_insights),
        "due_prediction_count": len(due_predictions),
        "pending_prediction_count": len(pending_predictions),
        "held_count": len(held_entries),
        "novelty_threshold": threshold,
        "cross_ref_window": int(conn.engine.cross_ref_window),
        "model_profiles": model_profiles,
        "effective_cross_ref_role": effective_cr_role,
        "entries_count": len(journal.entries),
        "gap_min_entries": int(conn.engine.negative_space_min_entries),
        "coverage_scans_count": len(journal.coverage_scans),
        "register_count": len(journal.register),
        "known_prior_art": list(journal.known_prior_art),
        "engine_domain": journal.last_domain or "",
        "directive_bundle_count": _count_directive_bundles(name),
        "latest_directives": _list_directive_files(name),
    })


def _count_directive_bundles(name: str) -> int:
    d = _journal_path(name).parent / f"{_journal_path(name).stem}_directives"
    if not d.exists():
        return 0
    return sum(1 for f in d.glob("bundle-*.md"))


def _list_directive_files(name: str) -> list[dict]:
    """List directive .md files (both per-record and bundle), newest first."""
    d = _journal_path(name).parent / f"{_journal_path(name).stem}_directives"
    if not d.exists():
        return []
    files = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        st = f.stat()
        out.append({
            "filename": f.name,
            "size_kb": f"{st.st_size / 1024:.1f}",
            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        })
    return out


@app.get("/journals/{name}/predictions", response_class=HTMLResponse)
def journal_predictions(request: Request, name: str):
    journal = _load_journal(name)
    return templates.TemplateResponse(request, "partials/predictions.html", {
        "name": name,
        "predictions": list(reversed(journal.predictions)),
        "register_by_id": {r["id"]: r for r in journal.register},
    })


@app.get("/journals/{name}/queue", response_class=HTMLResponse)
def journal_queue(request: Request, name: str):
    journal = _load_journal(name)
    return templates.TemplateResponse(request, "partials/queue.html", {
        "name": name,
        "queue": journal.question_queue,
        "focus": journal.focus,
    })


@app.get("/journals/{name}/graph-view", response_class=HTMLResponse)
def journal_graph_view(request: Request, name: str):
    return templates.TemplateResponse(request, "partials/graph.html", {"name": name})


@app.get("/journals/{name}/runs", response_class=HTMLResponse)
def journal_runs(request: Request, name: str):
    stem = _journal_path(name).stem
    runs = _list_runs_for_journal(stem)
    active_ids = {
        rid for rid, r in _active_runs.items()
        if r.get("journal") == stem and not r.get("done")
    }
    return templates.TemplateResponse(request, "partials/runs.html", {
        "name": name,
        "runs": runs,
        "active_ids": active_ids,
    })


@app.get("/journals/{name}/runs/{run_id}", response_class=HTMLResponse)
def journal_run_detail(request: Request, name: str, run_id: str):
    stem = _journal_path(name).stem
    meta_path = _run_meta_path(stem, run_id)
    log_path = _run_log_path(stem, run_id)
    if not meta_path.exists():
        raise HTTPException(404, f"run not found: {run_id}")
    meta = json.loads(meta_path.read_text())
    log_content = log_path.read_text() if log_path.exists() else ""
    is_active = run_id in _active_runs and not _active_runs[run_id].get("done")
    return templates.TemplateResponse(request, "partials/run_detail.html", {
        "name": name,
        "run_id": run_id,
        "meta": meta,
        "log_content": log_content,
        "is_active": is_active,
    })


# ─────────────────────────────────────────────
# Graph data endpoint (JSON for sigma.js)
# ─────────────────────────────────────────────

@app.get("/active-runs.json")
def active_runs_json():
    """Snapshot of runs currently streaming. Polled by the top-bar activity dot.

    Source of truth is the in-memory `_active_runs` dict — it reflects what the
    web process can actually stream. Meta files on disk are used only to recover
    the journal name and started_at for display.
    """
    out = []
    for run_id, run in _active_runs.items():
        if run.get("done"):
            continue
        out.append({
            "run_id": run_id,
            "journal": run.get("journal", ""),
            "cmd": run.get("cmd", ""),
        })
    return JSONResponse({"active": out, "count": len(out)})


@app.get("/journals/{name}/graph.json")
def journal_graph_json(name: str):
    journal = _load_journal(name)
    from engine.graph import build_graph
    g = build_graph(journal)
    # Sigma.js consumes a simple node/edge array format.
    nodes = []
    for node, data in g.nodes(data=True):
        kind = data.get("kind", "entry")
        raw_label = (
            data.get("label")
            or data.get("question")
            or data.get("title")
            or data.get("claim")
            or data.get("description")
            or node
        )
        label = str(raw_label)[:120]
        # Size for source/tag scales with reference count so bridges stand out.
        base_size = {
            "entry": 8,
            "xref": 10,
            "insight": 14,
            "register": 16,
            "prediction": 6,
            "source": 4,
            "tag": 5,
        }.get(kind, 4)
        if kind == "source":
            base_size = min(16, base_size + int(data.get("citation_count", 1)))
        elif kind == "tag":
            base_size = min(18, base_size + int(data.get("usage", 1)))
        nodes.append({
            "key": node,
            "attributes": {
                "label": label,
                "kind": kind,
                "size": base_size,
                "citation_count": int(data.get("citation_count", 0) or 0),
                "usage": int(data.get("usage", 0) or 0),
                "raw_id": node.split(":", 1)[1] if ":" in node else node,
                # status lets the client colour held register nodes distinctly.
                "status": str(data.get("status", "") or ""),
            },
        })

    edges = []
    for i, (u, v, data) in enumerate(g.edges(data=True)):
        kind = data.get("kind", "related")
        edges.append({
            "key": f"e-{i}",
            "source": u,
            "target": v,
            "attributes": {
                "kind": kind,
                "size": 1,
                "weight": float(data.get("weight", 1)),
            },
        })

    return JSONResponse({"nodes": nodes, "edges": edges})


@app.get("/journals/{name}/graph-node", response_class=HTMLResponse)
def graph_node_detail(request: Request, name: str, key: str):
    """Return a rich HTML fragment describing a graph node.

    `key` is the full graph node key (e.g. "entry:j-a1b2c3d4", "source:<hash>",
    "tag:<tag_name>"). We look the node up in the current journal state and
    render the kind-specific view with full (uncropped) text.
    """
    journal = _load_journal(name)
    if ":" not in key:
        raise HTTPException(400, f"invalid node key: {key!r}")
    kind, raw_id = key.split(":", 1)

    ctx: dict = {"name": name, "kind": kind, "node_key": key, "raw_id": raw_id}

    if kind == "entry":
        entry = next((e for e in journal.entries if e.get("id") == raw_id), None)
        if entry is None:
            raise HTTPException(404, f"entry not found: {raw_id}")
        ctx["entry"] = entry
    elif kind == "xref":
        xref = next((x for x in journal.cross_references if x.get("id") == raw_id), None)
        if xref is None:
            raise HTTPException(404, f"cross-reference not found: {raw_id}")
        ctx["xref"] = xref
        ctx["source_entries"] = [
            e for e in journal.entries if e.get("id") in (xref.get("source_entries") or [])
        ]
    elif kind == "insight":
        insight = next((i for i in journal.insights if i.get("id") == raw_id), None)
        if insight is None:
            raise HTTPException(404, f"insight not found: {raw_id}")
        ctx["insight"] = insight
    elif kind == "register":
        entry = next((r for r in journal.register if r.get("id") == raw_id), None)
        if entry is None:
            raise HTTPException(404, f"register entry not found: {raw_id}")
        ctx["register"] = entry
        ctx["register_predictions"] = journal.predictions_for_entry(raw_id)
    elif kind == "prediction":
        pred = next((p for p in journal.predictions if p.get("id") == raw_id), None)
        if pred is None:
            raise HTTPException(404, f"prediction not found: {raw_id}")
        ctx["prediction"] = pred
        parent_id = pred.get("register_entry_id", "")
        ctx["parent_register"] = next(
            (r for r in journal.register if r.get("id") == parent_id), None,
        )
    elif kind == "source":
        # raw_id is a SHA1 hash of the NORMALIZED source string (arxiv IDs
        # lowercased, DOIs de-prefixed, titles whitespace-collapsed — see
        # engine.graph._normalize_source). Match that exact hashing so the
        # lookup agrees with the graph node's key. Earlier we hashed the raw
        # string here, which worked by accident for already-canonical sources
        # but 404'd any time the original string needed normalization
        # (e.g. "arXiv:2302.03668 (Geiping et al., 2023) — ..." vs the
        # canonical "arxiv:2302.03668").
        import hashlib
        from engine.graph import _normalize_source
        citing: list[dict] = []
        url = ""
        for e in journal.entries:
            for src in e.get("sources", []) or []:
                canonical = _normalize_source(str(src))
                h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
                if h == raw_id:
                    url = str(src)
                    citing.append(e)
                    break
        if not citing:
            raise HTTPException(404, f"source not found: {raw_id}")
        ctx["source_url"] = url
        ctx["source_citing_entries"] = citing
    elif kind == "tag":
        citing = [e for e in journal.entries if raw_id in (e.get("domain_tags") or [])]
        if not citing:
            raise HTTPException(404, f"tag not in use: {raw_id}")
        ctx["tag_name"] = raw_id
        ctx["tag_entries"] = citing
    else:
        raise HTTPException(400, f"unknown node kind: {kind}")

    return templates.TemplateResponse(request, "partials/graph_node.html", ctx)


# ─────────────────────────────────────────────
# Mutations — focus / questions / review
# ─────────────────────────────────────────────

@app.post("/journals/{name}/focus")
def set_focus(name: str, focus: str = Form(...)):
    journal = _load_journal(name)
    journal.set_focus(focus)
    return RedirectResponse(f"/journals/{name}?tab=queue", status_code=303)


@app.post("/journals/{name}/focus/clear")
def clear_focus(name: str):
    journal = _load_journal(name)
    journal.clear_focus()
    return RedirectResponse(f"/journals/{name}?tab=queue", status_code=303)


@app.post("/journals/{name}/questions/add")
def add_question(name: str, question: str = Form(...)):
    journal = _load_journal(name)
    question = (question or "").strip()
    if question:
        journal.enqueue_questions([question], source="human", priority=1.0)
    return RedirectResponse(f"/journals/{name}?tab=queue", status_code=303)


@app.post("/journals/{name}/questions/clear")
def clear_questions(name: str):
    journal = _load_journal(name)
    journal.clear_question_queue()
    return RedirectResponse(f"/journals/{name}?tab=queue", status_code=303)


@app.post("/journals/{name}/questions/delete")
def delete_question(name: str, question_text: str = Form(...)):
    """Remove a single queued question matching text verbatim."""
    journal = _load_journal(name)
    before = len(journal.question_queue)
    journal.question_queue = [q for q in journal.question_queue if q.get("question") != question_text]
    if len(journal.question_queue) != before:
        journal.save()
    return RedirectResponse(f"/journals/{name}?tab=queue", status_code=303)


@app.post("/journals/{name}/queue/reorder")
def reorder_queued_question(
    name: str,
    question_text: str = Form(...),
    new_priority: float = Form(...),
):
    """Update a queued question's priority. Used by the drag-and-drop UI in
    the queue panel — SortableJS sends (question_text, new_priority) after
    a drop. The frontend computes new_priority = target_neighbor.priority
    + 0.001 so the dropped item sorts above its new neighbor within the
    same source bucket. Responds with a tiny JSON blob for the JS caller."""
    journal = _load_journal(name)
    ok = journal.update_queued_question_priority(question_text, new_priority)
    return JSONResponse({"ok": ok, "priority": max(0.0, min(1.0, new_priority))})


@app.post("/journals/rename")
def rename_journal(old_name: str = Form(...), new_name: str = Form(...)):
    old_path = _journal_path(old_name)
    new_path = _journal_path(new_name)
    if not old_path.exists():
        raise HTTPException(404, f"source journal not found: {old_path.name}")
    if new_path.exists():
        raise HTTPException(409, f"target already exists: {new_path.name}")
    old_path.rename(new_path)
    return RedirectResponse(f"/journals/{new_path.stem}", status_code=303)


@app.post("/journals/delete")
def delete_journal(name: str = Form(...), confirm: str = Form("")):
    if confirm != name:
        raise HTTPException(400, "confirm must equal the journal name")
    path = _journal_path(name)
    if not path.exists():
        raise HTTPException(404, f"journal not found: {path.name}")
    path.unlink()
    return RedirectResponse("/journals", status_code=303)


@app.post("/journals/{name}/register/{entry_id}/review")
def review_entry(
    name: str,
    entry_id: str,
    action: str = Form(...),
    rejection_reason: str = Form(""),
    notes: str = Form(""),
    reviewer: str = Form(""),
):
    journal = _load_journal(name)
    if action not in ("approved", "rejected", "deferred"):
        raise HTTPException(400, f"invalid action: {action}")
    if action == "rejected" and not rejection_reason.strip():
        raise HTTPException(400, "rejection requires a rejection_reason")
    journal.update_register_entry_review(
        entry_id,
        status=action,
        notes=notes,
        rejection_reason=rejection_reason,
        reviewer=reviewer,
    )
    return RedirectResponse(f"/journals/{name}/register", status_code=303)


# ─────────────────────────────────────────────
# Semantic search
# ─────────────────────────────────────────────

@app.post("/journals/{name}/find-similar", response_class=HTMLResponse)
def find_similar(request: Request, name: str, query: str = Form(...), top_k: int = Form(10)):
    journal = _load_journal(name)
    connection = _connection()
    from providers import build_embedding_client
    try:
        client = build_embedding_client(connection.verifier)
    except Exception:
        try:
            client = build_embedding_client(connection.primary)
        except Exception:
            return templates.TemplateResponse(request, "partials/similar_results.html", {
                "error": "No embedding-capable provider configured.",
                "hits": [],
            })

    from engine.embeddings import find_similar as do_find
    hits = do_find(query, journal, client, top_k=top_k)
    entries_by_id = {e.get("id"): e for e in journal.entries}
    return templates.TemplateResponse(request, "partials/similar_results.html", {
        "name": name,
        "query": query,
        "hits": hits,
        "entries_by_id": entries_by_id,
        "error": "",
    })


# ─────────────────────────────────────────────
# Run pages + SSE
# ─────────────────────────────────────────────

@app.get("/run", response_class=HTMLResponse)
def run_form(request: Request, journal: Optional[str] = None):
    existing = None
    if journal:
        path = DATA_DIR / (journal if journal.endswith(".json") else f"{journal}.json")
        if path.exists():
            try:
                data = json.loads(path.read_text() or "{}")
                existing = {
                    "name": path.stem,
                    "entries": len(data.get("entries") or []),
                    "insights": len(data.get("insights") or []),
                    "register": len(data.get("register") or []),
                    "predictions": len(data.get("predictions") or []),
                    "focus": str(data.get("focus", "")),
                    "last_domain": str(data.get("last_domain", "")),
                    "queue_count": len(data.get("question_queue") or []),
                    "human_queue_count": sum(
                        1 for q in (data.get("question_queue") or [])
                        if str(q.get("source", "")).startswith("human")
                    ),
                }
            except (json.JSONDecodeError, OSError):
                existing = None

    conn = _connection()
    # Every configured model profile becomes a selectable option in the
    # primary / verifier dropdowns. Primary + verifier are always present;
    # additional profiles added under [models.*] in engine.toml extend this.
    model_profiles: list[dict] = [
        {"role": "primary",  "name": conn.primary.name,  "provider": conn.primary.provider},
        {"role": "verifier", "name": conn.verifier.name, "provider": conn.verifier.provider},
    ]
    for role, profile in (getattr(conn, "extras", {}) or {}).items():
        model_profiles.append({"role": role, "name": profile.name, "provider": profile.provider})

    return templates.TemplateResponse(request, "run.html", {
        "journals": _list_journals(),
        "connection": conn,
        "model_profiles": model_profiles,
        "existing": existing,
    })


@app.post("/run/start")
async def run_start(
    journal: str = Form(...),
    domain: str = Form(""),
    cycles: int = Form(1),
    primary_model: str = Form(""),
    verifier_model: str = Form(""),
    primary_role: str = Form(""),
    verifier_role: str = Form(""),
    focus: str = Form(""),
    questions: str = Form(""),
    # Optional per-run engine-knob overrides. Empty string / default means "inherit from engine.toml".
    cross_ref_freq: str = Form(""),
    investigations_per_cycle: str = Form(""),
    novelty_threshold: str = Form(""),
    register_confidence_floor: str = Form(""),
    verify_insights: str = Form(""),                 # "on" if checkbox ticked, "" otherwise
    analog_probe_enabled: str = Form(""),            # same
    analog_probe_threshold: str = Form(""),
):
    journal_path = _journal_path(journal)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply direction BEFORE cycles spawn so the first phase prompts already see
    # the focus / human-queued questions.
    focus = (focus or "").strip()
    domain_stripped = (domain or "").strip()
    raw_questions = [line.strip() for line in (questions or "").splitlines() if line.strip()]
    if focus or raw_questions or domain_stripped:
        j = Journal(str(journal_path))
        if focus:
            j.set_focus(focus)
        if raw_questions:
            j.enqueue_questions(raw_questions, source="human", priority=1.0)
        if domain_stripped:
            j.set_last_domain(domain_stripped)

    cmd = [
        "python", "/app/curiosity_engine.py",
        "--cycles", str(max(0, min(200, int(cycles)))),
        "--journal", str(journal_path),
    ]
    if domain.strip():
        cmd += ["--domain", domain.strip()]
    # Model dropdowns post the role name ("primary" / "verifier") so only
    # propagate when the selection differs from the default role for that slot.
    # Role-based swap copies the whole profile; name-only override (if present)
    # is still accepted for backward compat with CLI callers.
    pr = (primary_role or "").strip().lower()
    if pr and pr != "primary":
        cmd += ["--primary-role", pr]
    vr = (verifier_role or "").strip().lower()
    if vr and vr != "verifier":
        cmd += ["--verifier-role", vr]

    conn_for_cmd = _connection()
    if primary_model.strip() and primary_model.strip() != conn_for_cmd.primary.name:
        cmd += ["--primary-model", primary_model.strip()]
    if verifier_model.strip() and verifier_model.strip() != conn_for_cmd.verifier.name:
        cmd += ["--verifier-model", verifier_model.strip()]

    # Per-run engine-knob overrides. Form is pre-filled with engine.toml defaults,
    # so only propagate a flag when the submitted value differs from the current
    # default — that way the cmd line reflects what the user actually changed.
    eng_defaults = _connection().engine

    def _int_override(raw: str, default: int, flag: str):
        raw = (raw or "").strip()
        if not raw:
            return
        try:
            v = int(raw)
        except ValueError:
            return
        if v != default:
            cmd.extend([flag, str(v)])

    def _float_override(raw: str, default: float, flag: str):
        raw = (raw or "").strip()
        if not raw:
            return
        try:
            v = float(raw)
        except ValueError:
            return
        # Float tolerance matches the form's step=0.05; anything inside this
        # window is a visual no-op.
        if abs(v - default) > 1e-9:
            cmd.extend([flag, f"{v:g}"])

    def _bool_override(raw: str, default: bool, flag_on: str, flag_off: str):
        submitted = bool(raw.strip())
        if submitted != default:
            cmd.append(flag_on if submitted else flag_off)

    _int_override(cross_ref_freq, eng_defaults.cross_ref_frequency, "--cross-ref-freq")
    _int_override(investigations_per_cycle, eng_defaults.investigations_per_cycle, "--investigations-per-cycle")
    _float_override(novelty_threshold, eng_defaults.novelty_threshold, "--novelty-threshold")
    _float_override(register_confidence_floor, eng_defaults.register_confidence_floor, "--register-confidence-floor")
    _float_override(analog_probe_threshold, eng_defaults.analog_probe_surprise_threshold, "--analog-probe-threshold")
    _bool_override(verify_insights, eng_defaults.verify_insights, "--verify-insights", "--no-verify-insights")
    _bool_override(analog_probe_enabled, eng_defaults.analog_probe_enabled, "--analog-probe-enabled", "--no-analog-probe-enabled")

    run_id = uuid.uuid4().hex
    journal_stem = journal_path.stem

    # Write initial meta + empty log so the run shows up in the Runs tab immediately.
    meta = {
        "run_id": run_id,
        "journal": journal_stem,
        "cmd": " ".join(cmd),
        "domain": domain.strip(),
        "cycles": int(cycles),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "returncode": None,
        "status": "running",
    }
    _run_meta_path(journal_stem, run_id).write_text(json.dumps(meta, indent=2))
    log_path = _run_log_path(journal_stem, run_id)
    log_path.write_text(f"$ {' '.join(cmd)}\n")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd="/workspace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _active_runs[run_id] = {
        "proc": proc,
        "lines": [],
        "done": False,
        "cmd": " ".join(cmd),
        "journal": journal_stem,
        "log_path": log_path,
    }

    asyncio.create_task(_collect_run_output(run_id, proc))
    return JSONResponse({"run_id": run_id, "cmd": " ".join(cmd), "journal": journal_stem})


async def _collect_run_output(run_id: str, proc: asyncio.subprocess.Process):
    if proc.stdout is None:
        return
    run = _active_runs[run_id]
    log_path: Path = run.get("log_path")
    # Append stdout lines to disk as we receive them so the log persists even
    # if the web server restarts mid-run.
    log_fh = open(log_path, "a") if log_path else None
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            run["lines"].append(text.rstrip())
            if log_fh is not None:
                log_fh.write(text)
                log_fh.flush()
    finally:
        if log_fh is not None:
            log_fh.close()
    await proc.wait()
    run["done"] = True
    run["returncode"] = proc.returncode

    journal_stem = run.get("journal") or ""
    if journal_stem:
        _update_run_meta(
            journal_stem,
            run_id,
            completed_at=datetime.now(timezone.utc).isoformat(),
            returncode=proc.returncode,
            status=("complete" if proc.returncode == 0 else "failed"),
        )


@app.get("/run/stream/{run_id}")
async def run_stream(run_id: str):
    if run_id not in _active_runs:
        raise HTTPException(404, f"unknown run: {run_id}")

    async def gen() -> AsyncGenerator[bytes, None]:
        sent = 0
        while True:
            run = _active_runs.get(run_id)
            if run is None:
                break
            lines = run["lines"]
            while sent < len(lines):
                line = lines[sent]
                sent += 1
                payload = json.dumps({"line": line}).encode("utf-8")
                yield b"data: " + payload + b"\n\n"
            if run.get("done"):
                payload = json.dumps({"done": True, "returncode": run.get("returncode")}).encode("utf-8")
                yield b"data: " + payload + b"\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# Settings (read-only)
# ─────────────────────────────────────────────

@app.get("/settings", response_class=HTMLResponse)
def settings_view(request: Request, saved: int = 0):
    connection = _connection()
    return templates.TemplateResponse(request, "settings.html", {
        "connection": connection,
        "env_keys": {
            "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "E2B_API_KEY": bool(os.environ.get("E2B_API_KEY")),
            "SEMANTIC_SCHOLAR_API_KEY": bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY")),
        },
        "saved": bool(saved),
    })


@app.post("/settings/save")
def settings_save(
    primary_provider: str = Form(...),
    primary_name: str = Form(...),
    primary_base_url: str = Form(""),
    primary_max_tokens: int = Form(4096),
    primary_investigation_max_tokens: int = Form(8192),
    primary_api_key: str = Form(""),
    primary_temperature: float = Form(1.0),
    primary_timeout_seconds: float = Form(300.0),
    verifier_provider: str = Form(...),
    verifier_name: str = Form(...),
    verifier_base_url: str = Form(""),
    verifier_max_tokens: int = Form(4096),
    verifier_investigation_max_tokens: int = Form(8192),
    verifier_api_key: str = Form(""),
    verifier_temperature: float = Form(1.0),
    verifier_timeout_seconds: float = Form(300.0),
    retry_max_attempts: int = Form(10),
    retry_base_delay_seconds: float = Form(0.5),
    retry_max_delay_seconds: float = Form(90.0),
    retry_jitter_seconds: float = Form(0.25),
    engine_cross_ref_window: int = Form(20),
    engine_questions_per_cycle: int = Form(3),
    engine_investigations_per_cycle: int = Form(1),
    engine_cross_ref_frequency: int = Form(3),
    engine_novelty_threshold: float = Form(0.7),
    engine_register_confidence_floor: float = Form(0.6),
    engine_verify_insights: str = Form(""),
    engine_analog_probe_enabled: str = Form(""),
    engine_analog_probe_surprise_threshold: float = Form(0.5),
    engine_assumption_probe_enabled: str = Form(""),
    engine_assumption_probe_surprise_threshold: float = Form(0.3),
    engine_held_entries_enabled: str = Form(""),
    engine_held_confidence_floor: float = Form(0.7),
    engine_cross_ref_role: str = Form(""),
    engine_directive_primary_role: str = Form(""),
    engine_directive_verifier_role: str = Form(""),
    engine_directive_max_verification_passes: int = Form(3),
    engine_gap_scan_extract_role: str = Form(""),
    engine_gap_scan_classify_role: str = Form(""),
    engine_parallel_investigations: int = Form(1),
    engine_parallel_xref_pipeline: int = Form(1),
    engine_negative_space_min_entries: int = Form(15),
    engine_gap_verification_hit_threshold: int = Form(5),
    engine_confidence_drop_on_downgrade: float = Form(0.10),
    engine_question_priority_floor: float = Form(0.70),
    engine_analog_probe_max_analogs: int = Form(3),
    engine_assumption_probe_max_assumptions: int = Form(3),
):
    """Write engine.toml. Empty api_key fields keep the existing value (don't clobber).
    Every knob the engine recognises is written here so Saving doesn't silently
    revert previously-configured values to their code defaults."""
    connection = _connection()

    def _profile_toml(role: str, provider: str, name: str, base_url: str,
                      max_tokens: int, inv_max_tokens: int, api_key: str,
                      existing_api_key: str, temperature: float,
                      timeout_seconds: float) -> str:
        key = api_key if api_key.strip() else existing_api_key
        key_line = f'api_key = "{key}"' if key else '# api_key = "..."'
        base_line = f'base_url = "{base_url.strip()}"' if base_url.strip() else '# base_url = "..."'
        return (
            f"[models.{role}]\n"
            f'provider = "{provider}"\n'
            f'name = "{name}"\n'
            f"{key_line}\n"
            f"{base_line}\n"
            f"max_tokens = {max_tokens}\n"
            f"investigation_max_tokens = {inv_max_tokens}\n"
            f"temperature = {temperature}\n"
            f"timeout_seconds = {timeout_seconds}\n"
        )

    # Checkbox semantics: browsers omit the field entirely when unchecked.
    def _checkbox(v: str) -> bool:
        return v.strip().lower() in ("on", "true", "1", "yes")

    verify_insights_on     = _checkbox(engine_verify_insights)
    analog_probe_on        = _checkbox(engine_analog_probe_enabled)
    assumption_probe_on    = _checkbox(engine_assumption_probe_enabled)
    held_entries_on        = _checkbox(engine_held_entries_enabled)

    # Validate cross_ref_role against the configured roles we know about,
    # so a typo doesn't silently save a broken config.
    known_roles = {"", "primary", "verifier"}
    known_roles.update(getattr(connection, "extras", {}) or {})
    cross_ref_role = engine_cross_ref_role.strip().lower()
    if cross_ref_role and cross_ref_role not in known_roles:
        cross_ref_role = ""  # treat unknown role as "default"
    directive_primary_role = engine_directive_primary_role.strip().lower()
    if directive_primary_role and directive_primary_role not in known_roles:
        directive_primary_role = ""
    directive_verifier_role = engine_directive_verifier_role.strip().lower()
    if directive_verifier_role and directive_verifier_role not in known_roles:
        directive_verifier_role = ""
    gap_scan_extract_role = engine_gap_scan_extract_role.strip().lower()
    if gap_scan_extract_role and gap_scan_extract_role not in known_roles:
        gap_scan_extract_role = ""
    gap_scan_classify_role = engine_gap_scan_classify_role.strip().lower()
    if gap_scan_classify_role and gap_scan_classify_role not in known_roles:
        gap_scan_classify_role = ""

    # Render any extra [models.<name>] profiles back out verbatim — we don't
    # expose them in the web form yet, but we mustn't erase them on save.
    extras_toml = ""
    for role_name, profile in (getattr(connection, "extras", {}) or {}).items():
        extras_toml += _profile_toml(
            role_name,
            profile.provider, profile.name, profile.base_url,
            profile.max_tokens, profile.investigation_max_tokens,
            "", profile.api_key, profile.temperature,
            profile.timeout_seconds,
        ) + "\n"

    toml_text = (
        "# Curiosity Engine — model connection + engine settings.\n"
        "# Edited via web UI.\n\n"
        + _profile_toml("primary", primary_provider, primary_name, primary_base_url,
                        primary_max_tokens, primary_investigation_max_tokens,
                        primary_api_key, connection.primary.api_key, primary_temperature,
                        primary_timeout_seconds)
        + "\n"
        + _profile_toml("verifier", verifier_provider, verifier_name, verifier_base_url,
                        verifier_max_tokens, verifier_investigation_max_tokens,
                        verifier_api_key, connection.verifier.api_key, verifier_temperature,
                        verifier_timeout_seconds)
        + "\n"
        + extras_toml
        + (
            "[retry]\n"
            f"max_attempts = {retry_max_attempts}\n"
            f"base_delay_seconds = {retry_base_delay_seconds}\n"
            f"max_delay_seconds = {retry_max_delay_seconds}\n"
            f"jitter_seconds = {retry_jitter_seconds}\n"
        )
        + "\n"
        + (
            "[engine]\n"
            f"cross_ref_window = {max(2, min(500, engine_cross_ref_window))}\n"
            f"questions_per_cycle = {max(1, min(10, engine_questions_per_cycle))}\n"
            f"investigations_per_cycle = {max(1, min(5, engine_investigations_per_cycle))}\n"
            f"cross_ref_frequency = {max(1, min(20, engine_cross_ref_frequency))}\n"
            f"novelty_threshold = {max(0.0, min(1.0, engine_novelty_threshold))}\n"
            f"register_confidence_floor = {max(0.0, min(1.0, engine_register_confidence_floor))}\n"
            f"verify_insights = {str(verify_insights_on).lower()}\n"
            f"analog_probe_enabled = {str(analog_probe_on).lower()}\n"
            f"analog_probe_surprise_threshold = {max(0.0, min(1.0, engine_analog_probe_surprise_threshold))}\n"
            f"assumption_probe_enabled = {str(assumption_probe_on).lower()}\n"
            f"assumption_probe_surprise_threshold = {max(0.0, min(1.0, engine_assumption_probe_surprise_threshold))}\n"
            f"held_entries_enabled = {str(held_entries_on).lower()}\n"
            f"held_confidence_floor = {max(0.0, min(1.0, engine_held_confidence_floor))}\n"
            f'cross_ref_role = "{cross_ref_role}"\n'
            f'directive_primary_role = "{directive_primary_role}"\n'
            f'directive_verifier_role = "{directive_verifier_role}"\n'
            f"directive_max_verification_passes = {max(1, min(10, engine_directive_max_verification_passes))}\n"
            f'gap_scan_extract_role = "{gap_scan_extract_role}"\n'
            f'gap_scan_classify_role = "{gap_scan_classify_role}"\n'
            f"parallel_investigations = {max(1, min(5, engine_parallel_investigations))}\n"
            f"parallel_xref_pipeline = {max(1, min(5, engine_parallel_xref_pipeline))}\n"
            f"negative_space_min_entries = {max(1, min(500, engine_negative_space_min_entries))}\n"
            f"gap_verification_hit_threshold = {max(1, min(100, engine_gap_verification_hit_threshold))}\n"
            f"confidence_drop_on_downgrade = {max(0.0, min(0.5, engine_confidence_drop_on_downgrade))}\n"
            f"question_priority_floor = {max(0.0, min(1.0, engine_question_priority_floor))}\n"
            f"analog_probe_max_analogs = {max(1, min(10, engine_analog_probe_max_analogs))}\n"
            f"assumption_probe_max_assumptions = {max(1, min(10, engine_assumption_probe_max_assumptions))}\n"
        )
    )

    from config import CONFIG_PATH
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(toml_text)
    return RedirectResponse("/settings?saved=1", status_code=303)
