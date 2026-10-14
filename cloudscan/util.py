"""Small shared helpers: terminal logging, subprocess execution, misc parsing."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
_VERBOSE = False
_QUIET = False


class _C:
    RESET = "\033[0m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    RED = "\033[31m" if _USE_COLOR else ""
    GREEN = "\033[32m" if _USE_COLOR else ""
    YELLOW = "\033[33m" if _USE_COLOR else ""
    BLUE = "\033[34m" if _USE_COLOR else ""
    CYAN = "\033[36m" if _USE_COLOR else ""


def configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    global _VERBOSE, _QUIET
    _VERBOSE = verbose
    _QUIET = quiet


def _emit(prefix: str, color: str, msg: str) -> None:
    print(f"{color}{prefix}{_C.RESET} {msg}", file=sys.stderr, flush=True)


def info(msg: str) -> None:
    if not _QUIET:
        _emit("[*]", _C.CYAN, msg)


def step(msg: str) -> None:
    if not _QUIET:
        _emit("[>]", _C.BOLD, msg)


def ok(msg: str) -> None:
    if not _QUIET:
        _emit("[+]", _C.GREEN, msg)


def warn(msg: str) -> None:
    _emit("[!]", _C.YELLOW, msg)


def error(msg: str) -> None:
    _emit("[-]", _C.RED, msg)


def debug(msg: str) -> None:
    if _VERBOSE:
        _emit("[d]", _C.DIM, msg)


# ---------------------------------------------------------------------------
# Reasoning display (stdout)
# ---------------------------------------------------------------------------
#
# The model's thinking goes to stdout, while every log line above goes to
# stderr. That separation lets ``awscloudscan --show-thinking > think.txt``
# capture the reasoning on its own and still watch the run on the terminal.
# The output is deliberately plain text (no colour, no prefixes) so a saved
# copy reads cleanly.


def thinking_header(label: Optional[str] = None, *, model: Optional[str] = None) -> None:
    """Announce a block of model reasoning on stdout."""
    if _QUIET:
        return
    title = "DeepSeek reasoning"
    if label:
        title = f"{title} — {label}"
    if model:
        title = f"{title} ({model})"
    print(f"\n=== {title} ===", file=sys.stdout, flush=True)


def thinking_chunk(text: str) -> None:
    """Write one streamed piece of model reasoning to stdout."""
    if _QUIET or not text:
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def thinking_end() -> None:
    """Close a reasoning block on stdout with a trailing newline."""
    if _QUIET:
        return
    sys.stdout.write("\n")
    sys.stdout.flush()


def die(msg: str, code: int = 1) -> "None":
    error(msg)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


@dataclass
class Completed:
    """Result of a subprocess run."""

    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def cmd(self) -> str:
        return " ".join(str(a) for a in self.args)


def which(cmd: str) -> Optional[str]:
    """Return the resolved path of *cmd*, or None."""
    return shutil.which(cmd)


def run(
    args: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stream: bool = False,
    stream_prefix: str = "  | ",
    timeout: Optional[int] = None,
    check: bool = False,
    log_command: bool = True,
) -> Completed:
    """Run a command, capturing stdout/stderr (and optionally echoing them live).

    ``stream=True`` prints output as it arrives (useful for long CloudFox runs)
    while still returning the captured text.
    """
    args = [str(a) for a in args]
    if log_command:
        debug(f"exec: {' '.join(args)}")

    if not stream:
        try:
            proc = subprocess.run(
                args,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            result = Completed(args, 124, exc.stdout or "", (exc.stderr or "") + "\n[timeout]")
            if check:
                raise RuntimeError(f"command timed out: {result.cmd}")
            return result
        except OSError as exc:
            # Covers FileNotFoundError, PermissionError, Exec format error, etc.
            result = Completed(args, 127, "", f"{type(exc).__name__}: {exc}")
            if check:
                raise RuntimeError(f"command could not be executed: {result.cmd}: {exc}") from exc
            debug(f"exec failed: {result.cmd}: {exc}")
            return result
        result = Completed(args, proc.returncode, proc.stdout or "", proc.stderr or "")
        if check and not result.ok:
            raise RuntimeError(f"command failed ({result.returncode}): {result.cmd}\n{result.stderr}")
        return result

    # Streaming mode.
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if _QUIET else subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        result = Completed(args, 127, "", f"{type(exc).__name__}: {exc}")
        if check:
            raise RuntimeError(f"command could not be executed: {result.cmd}: {exc}") from exc
        debug(f"exec failed: {result.cmd}: {exc}")
        return result
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_chunks.append(line)
        print(f"{_C.DIM}{stream_prefix}{line.rstrip()}{_C.RESET}", file=sys.stderr, flush=True)
    if proc.stderr is not None:
        for line in proc.stderr:
            stderr_chunks.append(line)
            print(f"{_C.RED}{stream_prefix}{line.rstrip()}{_C.RESET}", file=sys.stderr, flush=True)
    proc.wait()
    return Completed(args, proc.returncode, "".join(stdout_chunks), "".join(stderr_chunks))


# ---------------------------------------------------------------------------
# Misc parsing
# ---------------------------------------------------------------------------


def parse_version(text: str) -> Optional[tuple]:
    """Extract the first ``x.y.z`` version tuple from *text*."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    if not match:
        return None
    return tuple(int(g) for g in match.groups())


def arn_account(arn: str) -> Optional[str]:
    """Return the 12-digit account id embedded in an ARN, if present.

    Handles both ``arn:aws:iam::123456789012:role/x`` and bare 12-digit ids.
    """
    if not arn:
        return None
    match = re.search(r"arn:[^:]*:[^:]*:[^:]*:(\d{12}):", arn)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{12}", arn.strip()):
        return arn.strip()
    return None


_ACCOUNT_RE = re.compile(r"\b(\d{12})\b")


def find_account_ids(text: str) -> List[str]:
    """Return every 12-digit account id appearing in *text*, de-duplicated."""
    if not text:
        return []
    seen: List[str] = []
    for match in _ACCOUNT_RE.finditer(text):
        acct = match.group(1)
        if acct not in seen and not acct.startswith("0000"):
            seen.append(acct)
    return seen


def truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters with an explicit marker."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def slugify(text: str) -> str:
    """Make *text* safe for use in a filename."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text or "").strip("-")
    return text or "unknown"


def human_bytes(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"
