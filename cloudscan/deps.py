"""Dependency detection and self-provisioning.

awscloudscan needs CloudFox, PMapper and the AWS CLI. If any of them are missing it
will install them instead of failing:

* CloudFox  -> prebuilt release binary (falls back to ``go install``)
* PMapper   -> ``pip install principalmapper``
* AWS CLI   -> ``pip install awscli`` (only if no existing aws CLI is found)
* requests  -> ``pip install requests`` (the only third-party runtime dep)
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import config
from .util import Completed, debug, error, info, ok, parse_version, run, warn, which

LOCAL_BIN = os.path.join(os.path.expanduser("~"), ".local", "bin")


@dataclass
class DepStatus:
    """The state of a single external dependency."""

    name: str
    required: bool
    found: bool = False
    path: Optional[str] = None
    version: Optional[str] = None
    installed_now: bool = False
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        return self.found

    def line(self) -> str:
        if not self.found:
            return f"{self.name:<10} MISSING"
        marker = " (installed)" if self.installed_now else ""
        ver = f" {self.version}" if self.version else ""
        loc = f" {self.path}" if self.path else ""
        return f"{self.name:<10} OK{loc}{ver}{marker}"


@dataclass
class Dependencies:
    """Aggregate dependency state, resolved once per run."""

    python_ok: bool = True
    cloudfox: Optional[str] = None
    pmapper: Optional[str] = None
    aws: Optional[str] = None
    statuses: List[DepStatus] = field(default_factory=list)

    @property
    def missing_required(self) -> List[DepStatus]:
        return [s for s in self.statuses if s.required and not s.found]


# ---------------------------------------------------------------------------
# pip helpers
# ---------------------------------------------------------------------------


def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def pip_install(package: str) -> bool:
    """Install *package* with pip, handling PEP 668 and --user gracefully."""
    base = [sys.executable, "-m", "pip", "install", "--upgrade"]
    # Only add --user when not inside a virtualenv (harmless but noisy otherwise).
    attempts = [base + ([package])] if _in_venv() else [base + ["--user", package]]
    # Retry with --break-system-packages for externally-managed environments.
    attempts.append(base + ["--user", "--break-system-packages", package])

    for cmd in attempts:
        info(f"installing {package}: {' '.join(cmd)}")
        result: Completed = run(cmd, timeout=900)
        if result.ok:
            ok(f"installed {package}")
            return True
        if "externally-managed-environment" in (result.stderr or ""):
            debug("PEP 668 environment detected, retrying with --break-system-packages")
            continue
        error(f"pip install failed for {package}: {(result.stderr or '').strip()[:400]}")
        return False

    error(f"could not install {package} (all pip strategies failed)")
    return False


def _pip_show_version(dist_name: str) -> Optional[str]:
    result = run([sys.executable, "-m", "pip", "show", dist_name], log_command=False)
    if not result.ok:
        return None
    for line in result.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def check_python() -> DepStatus:
    major, minor = sys.version_info[:2]
    vers = f"{major}.{minor}.{sys.version_info[2]}"
    status = DepStatus("python3", required=True, found=(major, minor) >= (3, 9), version=vers, path=sys.executable)
    if not status.found:
        status.detail = "Python 3.9+ is required"
    return status


def check_cloudfox(install: Callable[[], bool]) -> DepStatus:
    path = which("cloudfox")
    status = DepStatus("cloudfox", required=True, path=path)
    if path:
        result = run([path, "--version"], log_command=False)
        version = parse_version(result.stdout + result.stderr)
        status.found = True
        status.version = ".".join(map(str, version)) if version else "unknown"
        if version and version < config.CLOUDFOX_MIN_VERSION:
            warn(f"cloudfox {status.version} is older than the supported minimum "
                 f"{'.'.join(map(str, config.CLOUDFOX_MIN_VERSION))}; attempting upgrade")
            if install():
                status.path = which("cloudfox") or path
                status.installed_now = True
        return status

    warn("cloudfox not found on PATH; installing it now")
    if install():
        status.path = which("cloudfox") or os.path.join(LOCAL_BIN, "cloudfox")
        status.installed_now = True
        status.found = os.path.exists(status.path)
        probe = run([status.path, "--version"], log_command=False)
        if probe.ok:
            version = parse_version(probe.stdout + probe.stderr)
            status.version = ".".join(map(str, version)) if version else None
        else:
            warn(f"cloudfox was installed but does not run: {(probe.stderr or '').strip()[:200]}")
            status.found = False
            status.detail = "installed binary could not be executed"
    return status


def check_pmapper(install: Callable[[], bool]) -> DepStatus:
    path = which("pmapper")
    status = DepStatus("pmapper", required=True, path=path)
    if path:
        status.found = True
        status.version = _pip_show_version("principalmapper")
        return status

    warn("pmapper not found on PATH; installing it now")
    if install():
        status.path = which("pmapper") or os.path.join(LOCAL_BIN, "pmapper")
        status.found = os.path.exists(status.path)
        status.installed_now = True
        status.version = _pip_show_version("principalmapper")
    return status


def check_awscli(install: Callable[[], bool]) -> DepStatus:
    path = which("aws")
    status = DepStatus("aws", required=False, path=path)
    if path:
        status.found = True
        result = run([path, "--version"], log_command=False)
        status.version = (result.stdout or result.stderr).strip().split(" ")[0]
        return status

    warn("aws CLI not found; installing awscli via pip (only needed as a fallback)")
    if install():
        status.path = which("aws") or os.path.join(LOCAL_BIN, "aws")
        status.found = os.path.exists(status.path)
        status.installed_now = True
    return status


def check_requests(install: Callable[[], bool]) -> DepStatus:
    try:
        import requests  # noqa: F401

        import requests as _r

        return DepStatus("requests", required=True, found=True, version=_r.__version__)
    except ImportError:
        warn("python 'requests' missing; installing it now")
        if install():
            return DepStatus("requests", required=True, found=True, installed_now=True)
        return DepStatus("requests", required=True, found=False)


# ---------------------------------------------------------------------------
# CloudFox installers
# ---------------------------------------------------------------------------


def _arch_slug() -> Optional[str]:
    machine = platform.machine().lower()
    return {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "i386": "386",
        "i686": "386",
    }.get(machine)


def _download(url: str, dest: str) -> bool:
    info(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as handle:
            shutil.copyfileobj(response, handle)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        error(f"download failed: {exc}")
        return False


def _install_cloudfox_binary() -> bool:
    """Download the pinned CloudFox release and drop the binary in ~/.local/bin."""
    arch = _arch_slug()
    if not arch:
        warn(f"unsupported architecture for prebuilt CloudFox: {platform.machine()}")
        return False

    asset = f"cloudfox-linux-{arch}.zip"
    url = (f"https://github.com/{config.CLOUDFOX_REPO}/releases/download/"
           f"v{config.CLOUDFOX_PINNED_VERSION}/{asset}")

    tmpdir = tempfile.mkdtemp(prefix="awscloudscan-cf-")
    zip_path = os.path.join(tmpdir, asset)
    try:
        if not _download(url, zip_path):
            return False
        with zipfile.ZipFile(zip_path) as archive:
            member = next((n for n in archive.namelist() if n.endswith("cloudfox")), None)
            if member is None:
                error("cloudfox binary not found inside the release archive")
                return False
            archive.extract(member, tmpdir)
            extracted = os.path.join(tmpdir, member)
        os.makedirs(LOCAL_BIN, exist_ok=True)
        target = os.path.join(LOCAL_BIN, "cloudfox")
        shutil.copy2(extracted, target)
        os.chmod(target, 0o755)
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _install_cloudfox_go() -> bool:
    if not which("go"):
        return False
    gopath = run(["go", "env", "GOPATH"], log_command=False).stdout.strip() or os.path.expanduser("~/go")
    info("installing cloudfox via 'go install' (this builds from source, may take a minute)")
    result = run(["go", "install", f"github.com/{config.CLOUDFOX_REPO}@latest"], timeout=1800)
    if not result.ok:
        error(f"go install failed: {(result.stderr or '').strip()[:400]}")
        return False
    built = os.path.join(gopath, "bin", "cloudfox")
    if os.path.exists(built):
        os.makedirs(LOCAL_BIN, exist_ok=True)
        target = os.path.join(LOCAL_BIN, "cloudfox")
        try:
            shutil.copy2(built, target)
            os.chmod(target, 0o755)
        except OSError:
            pass
    return which("cloudfox") is not None


def install_cloudfox() -> bool:
    warn("cloudfox is required but was not found on PATH")
    if _install_cloudfox_binary():
        ok("cloudfox installed from the official GitHub release")
        return True
    warn("prebuilt download failed; falling back to 'go install'")
    if _install_cloudfox_go():
        ok("cloudfox installed via go")
        return True
    error("automatic CloudFox installation failed. Install manually: "
          "https://github.com/BishopFox/cloudfox#installation")
    return False


def install_pmapper() -> bool:
    return pip_install(config.PMAPPER_PIP_PACKAGE)


def install_awscli() -> bool:
    return pip_install("awscli")


def install_requests() -> bool:
    return pip_install("requests")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def ensure_dependencies(*, auto_install: bool = True, install_python_deps: bool = True) -> Dependencies:
    """Resolve every dependency, installing missing ones when allowed."""
    deps = Dependencies()

    def maybe(installer: Callable[[], bool]) -> Callable[[], bool]:
        return installer if auto_install else (lambda: False)

    deps.statuses.append(check_python())
    deps.statuses.append(check_requests(maybe(install_requests) if install_python_deps else (lambda: False)))
    deps.statuses.append(check_cloudfox(maybe(install_cloudfox)))
    deps.statuses.append(check_pmapper(maybe(install_pmapper)))
    deps.statuses.append(check_awscli(maybe(install_awscli)))

    for status in deps.statuses:
        if status.name == "cloudfox" and status.found:
            deps.cloudfox = status.path
        elif status.name == "pmapper" and status.found:
            deps.pmapper = status.path
        elif status.name == "aws" and status.found:
            deps.aws = status.path

    deps.python_ok = deps.statuses[0].found
    return deps


def print_dependency_report(deps: Dependencies) -> None:
    info("dependency status")
    for status in deps.statuses:
        line = status.line()
        if status.found:
            ok(line)
        elif status.required:
            error(line)
        else:
            warn(line)
        if status.detail:
            warn(f"           {status.detail}")

    # ~/.local/bin may have just received new tools that are not on PATH yet.
    if LOCAL_BIN not in os.environ.get("PATH", "") and any(s.installed_now for s in deps.statuses):
        warn(f"{LOCAL_BIN} is not on your PATH — awscloudscan will still call the tools by absolute path, "
             f"but you may want to add it: export PATH=\"$PATH:{LOCAL_BIN}\"")
