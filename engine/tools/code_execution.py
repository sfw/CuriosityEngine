"""code_execution — run a Python snippet with timeout + output caps.

Two backends:

- **local subprocess** (default, no setup): spawn `python -c <code>` in a fresh
  workdir under $TMPDIR, with CPU-time/timeout enforcement, output truncated
  to a configured cap, and each call hermetic (no persistence between calls).
  NOT a security sandbox. Use at your own risk — the model can read / write
  files under your home dir if it wants to. Appropriate for a research engine
  you're driving on your own machine.

- **E2B hosted sandbox** (enabled when the `E2B_API_KEY` env var is set):
  each call runs in an isolated container with the scientific Python stack
  (numpy, scipy, pandas, sklearn, matplotlib, etc.) pre-installed. Safer and
  more reproducible; costs E2B API usage.

Both return the same shape: a structured result string with stdout / stderr /
exit code / runtime / truncation flag.

Note: when the primary is an Anthropic model, Anthropic's native server-side
`code_execution_20250825` tool takes over this name. Our client tool is the
fallback for non-Anthropic primaries (Kimi, GPT, Gemini, Ollama, etc.) and
for the verifier when it's on a non-Anthropic provider.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from engine.tools.base import Tool, ToolError

_DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_TIMEOUT_SECONDS = 300.0
_MAX_OUTPUT_BYTES = 200_000          # 200 KB stdout/stderr each
_MAX_CODE_LEN = 50_000


def _apply_child_limits(max_cpu_seconds: int):
    """preexec_fn to cap CPU time and restrict core dumps (Unix only)."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def _truncate(text: str, limit: int = _MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: limit - 200]
    return head + f"\n\n[truncated; original length {len(text)} bytes]", True


def _format_result(
    *,
    stdout: str,
    stderr: str,
    exit_code: int,
    runtime_seconds: float,
    backend: str,
    interrupted: bool = False,
    workdir: Optional[str] = None,
) -> str:
    stdout_trunc, stdout_was_truncated = _truncate(stdout)
    stderr_trunc, stderr_was_truncated = _truncate(stderr)

    lines: list[str] = []
    lines.append(f"# code_execution ({backend})")
    lines.append(f"# exit_code={exit_code}  runtime={runtime_seconds:.2f}s")
    if interrupted:
        lines.append("# TIMEOUT/INTERRUPTED — process was killed")
    if workdir:
        lines.append(f"# workdir={workdir}")
    if stdout_was_truncated:
        lines.append(f"# stdout truncated to {_MAX_OUTPUT_BYTES} bytes")
    if stderr_was_truncated:
        lines.append(f"# stderr truncated to {_MAX_OUTPUT_BYTES} bytes")
    lines.append("")
    lines.append("## stdout")
    lines.append(stdout_trunc or "[empty]")
    lines.append("")
    lines.append("## stderr")
    lines.append(stderr_trunc or "[empty]")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Local subprocess backend
# ─────────────────────────────────────────────

def _run_local(code: str, timeout: float, python_bin: str) -> str:
    """Run `code` with `python_bin` in a fresh temp dir under $TMPDIR."""
    workdir = Path(tempfile.mkdtemp(prefix="curiosity_code_"))
    start = time.perf_counter()
    interrupted = False
    stdout, stderr = "", ""
    exit_code = -1
    try:
        try:
            proc = subprocess.run(
                [python_bin, "-I", "-u", "-c", code],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                # Pass a minimal env; don't inherit anything the model could exploit.
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": str(workdir),             # redirect $HOME into workdir
                    "TMPDIR": str(workdir),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "MPLBACKEND": "Agg",              # headless matplotlib
                },
                preexec_fn=lambda: _apply_child_limits(int(timeout) + 2),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as e:
            interrupted = True
            stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = ((e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\n[killed: timeout after {timeout:.1f}s]"
            exit_code = -9
    finally:
        # Hermetic: nuke the workdir regardless of outcome.
        shutil.rmtree(workdir, ignore_errors=True)

    runtime = time.perf_counter() - start
    return _format_result(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        runtime_seconds=runtime,
        backend="local_subprocess",
        interrupted=interrupted,
    )


# ─────────────────────────────────────────────
# E2B hosted sandbox backend (optional)
# ─────────────────────────────────────────────

def _run_e2b(code: str, timeout: float) -> str:
    """Run `code` in an E2B sandbox. Requires `e2b_code_interpreter` installed and
    E2B_API_KEY env var set. Each call is a fresh sandbox."""
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError as e:
        raise ToolError(
            "E2B_API_KEY is set but the e2b_code_interpreter package is not installed. "
            "`pip install e2b-code-interpreter` or unset E2B_API_KEY to fall back to local subprocess."
        ) from e

    start = time.perf_counter()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    exit_code = 0
    interrupted = False

    try:
        with Sandbox(timeout=int(timeout) + 10) as sb:
            exec_result = sb.run_code(code, timeout=int(timeout))
            for item in exec_result.logs.stdout or []:
                stdout_chunks.append(item)
            for item in exec_result.logs.stderr or []:
                stderr_chunks.append(item)
            if exec_result.error:
                exit_code = 1
                stderr_chunks.append(
                    f"\n[e2b error: {exec_result.error.name}: {exec_result.error.value}]"
                )
            for res in exec_result.results or []:
                # Rendered results (text / markdown) — appended as structured output.
                text = getattr(res, "text", None) or getattr(res, "markdown", None)
                if text:
                    stdout_chunks.append(str(text))
    except TimeoutError:
        interrupted = True
        exit_code = -9
        stderr_chunks.append(f"[killed: timeout after {timeout:.1f}s]")
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"e2b backend failed: {type(e).__name__}: {e}") from e

    runtime = time.perf_counter() - start
    return _format_result(
        stdout="\n".join(stdout_chunks),
        stderr="\n".join(stderr_chunks),
        exit_code=exit_code,
        runtime_seconds=runtime,
        backend="e2b",
        interrupted=interrupted,
    )


# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────


class CodeExecutionTool(Tool):
    name = "code_execution"
    description = (
        "Execute a Python snippet and return its stdout/stderr/exit_code. Use for "
        "testing claims computationally, recomputing numbers cited in papers, "
        "running small simulations, or sanity-checking math. Each call is hermetic "
        "(fresh working directory, no state carried between calls). Import anything "
        "available in the host Python environment (numpy/scipy/pandas/sklearn/"
        "matplotlib etc. if installed). Timeout default 60s, max 300s. Output "
        "truncated at 200 KB per stream. Set E2B_API_KEY to run in a hosted "
        "sandbox with scientific Python pre-installed instead of the local "
        "subprocess."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must be self-contained — no state persists between calls.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": f"Max execution time in seconds (default {_DEFAULT_TIMEOUT_SECONDS}, cap {_MAX_TIMEOUT_SECONDS}).",
                "default": _DEFAULT_TIMEOUT_SECONDS,
                "minimum": 1,
                "maximum": _MAX_TIMEOUT_SECONDS,
            },
        },
        "required": ["code"],
    }
    timeout_seconds = _MAX_TIMEOUT_SECONDS + 30.0

    def execute(self, args: dict) -> str:
        code = args.get("code") or ""
        if not code.strip():
            raise ToolError("no code provided")
        if len(code) > _MAX_CODE_LEN:
            raise ToolError(f"code too long (max {_MAX_CODE_LEN} chars)")

        timeout = float(args.get("timeout_seconds") or _DEFAULT_TIMEOUT_SECONDS)
        timeout = max(1.0, min(_MAX_TIMEOUT_SECONDS, timeout))

        if os.environ.get("E2B_API_KEY"):
            return _run_e2b(code, timeout)

        python_bin = sys.executable or "python3"
        return _run_local(code, timeout, python_bin)
