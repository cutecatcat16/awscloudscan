"""Execution layer: identity resolution plus CloudFox and PMapper invocations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import config
from .util import Completed, debug, error, info, ok, run, step, warn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Identity:
    account: str
    arn: str
    user_id: str
    profile: str
    region: str

    @property
    def is_assumed_role(self) -> bool:
        return ":assumed-role/" in (self.arn or "")

    @property
    def principal_name(self) -> str:
        return (self.arn or "").split("/")[-1]


@dataclass
class RunLogEntry:
    tool: str
    command: str
    returncode: int
    duration_s: float
    note: str = ""


@dataclass
class RunResult:
    identity: Optional[Identity] = None
    outdir: str = ""
    cloudfox_output_dir: Optional[str] = None
    pmapper_storage: Optional[str] = None
    pmapper_analysis: Optional[dict] = None
    pmapper_analysis_text: str = ""
    cloudfox_pmapper_dir: Optional[str] = None
    log: List[RunLogEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_env(region: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["AWS_DEFAULT_REGION"] = region
    env["AWS_REGION"] = region
    # CloudFox/PMapper benefit from unbuffered, non-paged output.
    env.setdefault("AWS_PAGER", "")
    env.setdefault("PAGER", "")
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def pmapper_storage_root() -> str:
    root = os.getenv("PMAPPER_STORAGE")
    if root:
        return root
    if os.getenv("XDG_DATA_HOME"):
        return os.path.join(os.getenv("XDG_DATA_HOME"), "principalmapper")
    return os.path.join(os.path.expanduser("~"), ".local", "share", "principalmapper")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def resolve_identity(profile: str, region: str, aws_path: Optional[str] = None) -> Identity:
    """Resolve the caller identity for *profile* via boto3, falling back to the aws CLI."""
    env = build_env(region)
    try:
        import boto3  # type: ignore
        from botocore.exceptions import (ClientError, NoCredentialsError,
                                         ProfileNotFound)

        try:
            session = boto3.Session(profile_name=profile, region_name=region)
            identity = session.client("sts").get_caller_identity()
            return Identity(
                account=identity["Account"],
                arn=identity["Arn"],
                user_id=identity["UserId"],
                profile=profile,
                region=region,
            )
        except ProfileNotFound:
            raise SystemExit(
                f"AWS profile '{profile}' was not found. Check ~/.aws/credentials and ~/.aws/config."
            )
        except (NoCredentialsError, ClientError) as exc:
            warn(f"boto3 could not resolve the identity ({exc}); trying the aws CLI")
    except ImportError:
        debug("boto3 not available; using the aws CLI for identity resolution")

    aws_cmd = aws_path or "aws"
    result = run([aws_cmd, "sts", "get-caller-identity", "--profile", profile, "--output", "json"], env=env)
    if not result.ok:
        error(result.stderr.strip() or "aws sts get-caller-identity failed")
        raise SystemExit(
            f"Unable to resolve identity for profile '{profile}'. "
            "Verify the profile has valid credentials and (if required) an MFA/session token."
        )
    payload = json.loads(result.stdout)
    return Identity(
        account=payload["Account"],
        arn=payload["Arn"],
        user_id=payload["UserId"],
        profile=profile,
        region=region,
    )


# ---------------------------------------------------------------------------
# CloudFox
# ---------------------------------------------------------------------------


def _cloudfox_argv(cloudfox: str, profile: str, region: str, outdir: str, modules: Sequence[str],
                   mfa_token: Optional[str] = None) -> List[str]:
    # CloudFox v2: `--outdir` sets the output directory. The short flag `-o` is
    # NOT an alias for it — `-o/--output` selects the table format
    # ("brief"/"wide"). Passing a path to `-o` silently leaves the output in
    # CloudFox's default location (~/.cloudfox) and awscloudscan then finds nothing.
    argv = [cloudfox, "aws", "-p", profile, "--outdir", outdir, "-v", "2", "-y"]
    if mfa_token:
        argv += ["--mfa-token", mfa_token]
    argv += list(modules)
    return argv


def cloudfox_aws_bases(outdir: str) -> List[str]:
    """Candidate ``.../aws`` output bases under a CloudFox output root.

    CloudFox v2 nests module output under ``<outdir>/cloudfox-output/aws/``;
    older layouts wrote directly to ``<outdir>/aws/``. Return whichever exist.
    """
    candidates = [
        os.path.join(outdir, "cloudfox-output", "aws"),
        os.path.join(outdir, "aws"),
    ]
    return [c for c in candidates if os.path.isdir(c)]


def _find_cloudfox_output_dir(outdir: str, profile: str, account: str) -> Optional[str]:
    for base in cloudfox_aws_bases(outdir):
        for candidate in (f"{profile}-{account}", profile, account):
            path = os.path.join(base, candidate)
            if os.path.isdir(path):
                return path
        dirs = [os.path.join(base, e) for e in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, e))]
        if len(dirs) == 1:
            return dirs[0]
        if dirs:
            # Several module/profile directories: prefer the most recently written.
            return max(dirs, key=os.path.getmtime)
    return None


def run_cloudfox(
    cloudfox: str,
    profile: str,
    region: str,
    outdir: str,
    modules: Sequence[str],
    result: RunResult,
    *,
    mfa_token: Optional[str] = None,
    stream: bool = True,
) -> None:
    """Run CloudFox modules, falling back to one-module-per-call if batching fails."""
    import time

    os.makedirs(outdir, exist_ok=True)
    env = build_env(region)

    argv = _cloudfox_argv(cloudfox, profile, region, outdir, modules, mfa_token)
    step(f"running CloudFox ({len(modules)} modules): {' '.join(modules)}")
    started = time.time()
    completed = run(argv, env=env, stream=stream, timeout=60 * 60 * 4)
    elapsed = time.time() - started
    result.log.append(RunLogEntry("cloudfox", completed.cmd, completed.returncode, elapsed))

    if completed.ok:
        ok(f"CloudFox finished in {elapsed:.0f}s")
        return

    warn(f"batched CloudFox run failed (rc={completed.returncode}); retrying module-by-module")
    for module in modules:
        argv = _cloudfox_argv(cloudfox, profile, region, outdir, [module], mfa_token)
        started = time.time()
        completed = run(argv, env=env, stream=stream, timeout=60 * 30)
        elapsed = time.time() - started
        note = "" if completed.ok else (completed.stderr or "").strip()[:300]
        result.log.append(RunLogEntry("cloudfox", completed.cmd, completed.returncode, elapsed, note))
        if completed.ok:
            ok(f"cloudfox {module} finished in {elapsed:.0f}s")
        else:
            warn(f"cloudfox {module} failed (rc={completed.returncode})")
            result.errors.append(f"cloudfox module '{module}' failed: {note}")


def find_cloudfox_output(result: RunResult, outdir: str, profile: str) -> None:
    if result.identity:
        result.cloudfox_output_dir = _find_cloudfox_output_dir(outdir, profile, result.identity.account)


# ---------------------------------------------------------------------------
# PMapper
# ---------------------------------------------------------------------------


# Matches a regional AWS endpoint host such as ``autoscaling.me-south-1.amazonaws.com``
# and captures the region label (``me-south-1``).
_REGIONAL_ENDPOINT_RE = re.compile(r"[a-z0-9-]+\.([a-z]{2}-[a-z]+-\d+)\.amazonaws\.com")


def _region_from_endpoint_error(text: str) -> Optional[str]:
    """Extract the AWS region from an ``EndpointConnectionError`` message."""
    match = _REGIONAL_ENDPOINT_RE.search(text or "")
    return match.group(1) if match else None


def run_pmapper(
    pmapper: str,
    profile: str,
    region: str,
    result: RunResult,
    *,
    workdir: str,
    ignore_orgs: bool = True,
    skip_create: bool = False,
    include_regions: Optional[Sequence[str]] = None,
    exclude_regions: Optional[Sequence[str]] = None,
    include_services: Optional[Sequence[str]] = None,
    exclude_services: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> None:
    """Create the PMapper graph (unless it exists) and run the analysis."""
    import time

    env = build_env(region)
    storage = pmapper_storage_root()
    result.pmapper_storage = storage

    account = result.identity.account if result.identity else None
    graph_exists = bool(account and os.path.isdir(os.path.join(storage, account, "graph")))

    if skip_create:
        info("skipping PMapper graph creation (--skip-pmapper-create)")
    elif graph_exists:
        info(f"reusing existing PMapper graph at {os.path.join(storage, account)}")
    else:
        # PMapper's edge checkers sweep every AWS region by default. That both
        # wastes time and aborts graph creation outright when any regional
        # endpoint is unreachable (blocked network / disabled region) — botocore
        # raises EndpointConnectionError, which PMapper does not catch.
        #
        # Default: constrain PMapper to an explicit --include-regions allow-list
        # (the engagement region), so it never touches unreachable endpoints.
        # `global` is always included by PMapper, so the IAM edges are unaffected.
        #
        # All-regions mode (empty include list): fall back to the full sweep but
        # self-heal — parse the offending region out of the error and retry with
        # it added to PMapper's --exclude-regions deny-list.
        included = [r.strip() for r in (include_regions or []) if r and r.strip()]
        excluded: List[str] = []
        for r in (exclude_regions or []):
            r = (r or "").strip()
            if r and r not in excluded:
                excluded.append(r)

        completed: Optional[Completed] = None
        elapsed = 0.0
        note = ""
        attempts = 0
        while attempts < 6:
            attempts += 1
            argv = [pmapper, "--profile", profile]
            if debug:
                # --debug is a *global* PMapper flag: it must precede the
                # subcommand, otherwise argparse rejects it.
                argv.append("--debug")
            argv += ["graph", "create"]
            if ignore_orgs:
                argv.append("--ignore-orgs")
            if included:
                argv += ["--include-regions", *included]
                scope = f" [regions: {', '.join(included)}]"
            else:
                if excluded:
                    argv += ["--exclude-regions", *excluded]
                scope = f" [excluding: {', '.join(excluded)}]" if excluded else " [all regions]"
            # --include-services and --exclude-services are mutually exclusive
            # in PMapper; callers must not supply both.
            if include_services:
                argv += ["--include-services", *include_services]
                scope += f" [services: {', '.join(include_services)}]"
            elif exclude_services:
                argv += ["--exclude-services", *exclude_services]
                scope += f" [skipping services: {', '.join(exclude_services)}]"
            step("building PMapper IAM graph (this can take a few minutes)" + scope)
            started = time.time()
            completed = run(argv, env=env, stream=True, timeout=60 * 60)
            elapsed = time.time() - started
            note = "" if completed.ok else (completed.stderr or "").strip()[:300]
            result.log.append(RunLogEntry("pmapper", completed.cmd, completed.returncode, elapsed, note))
            if completed.ok:
                break
            if included:
                # Regions were set explicitly; there is nothing to auto-exclude.
                break
            bad_region = _region_from_endpoint_error((completed.stderr or "") + (completed.stdout or ""))
            if bad_region and bad_region not in excluded:
                excluded.append(bad_region)
                warn(f"PMapper could not reach AWS region '{bad_region}' "
                     f"(endpoint unreachable); retrying without it")
                continue
            break

        if completed is not None and completed.ok:
            ok(f"PMapper graph created in {elapsed:.0f}s")
            if excluded:
                warn("PMapper graph built without unreachable regions: " + ", ".join(excluded))
        else:
            rc = completed.returncode if completed else "?"
            warn(f"PMapper graph creation failed (rc={rc}): {note}")
            result.errors.append(f"pmapper graph create failed: {note}")
            return

    # Analysis (JSON for the LLM, text for humans/report appendix).
    argv = [pmapper, "--profile", profile, "analysis", "--output-type", "json"]
    started = time.time()
    completed = run(argv, env=env, timeout=60 * 30)
    elapsed = time.time() - started
    result.log.append(RunLogEntry("pmapper", completed.cmd, completed.returncode, elapsed))

    if completed.ok and completed.stdout.strip():
        try:
            result.pmapper_analysis = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            warn(f"could not parse PMapper JSON analysis: {exc}")
            result.pmapper_analysis_text = completed.stdout
    else:
        warn("PMapper analysis produced no JSON output")
        if completed.stderr:
            result.errors.append(f"pmapper analysis: {completed.stderr.strip()[:300]}")

    # Persist a text analysis alongside, for the report appendix.
    text_run = run([pmapper, "--profile", profile, "analysis", "--output-type", "text"], env=env)
    if text_run.ok:
        result.pmapper_analysis_text = text_run.stdout
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "pmapper-analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(result.pmapper_analysis or {}, handle, indent=2)
    with open(os.path.join(workdir, "pmapper-analysis.txt"), "w", encoding="utf-8") as handle:
        handle.write(result.pmapper_analysis_text)


def run_cloudfox_pmapper(cloudfox: str, profile: str, region: str, outdir: str, result: RunResult) -> None:
    """Ask CloudFox to turn the PMapper graph into a privesc graph."""
    import time

    if not result.pmapper_storage or not result.identity:
        return
    env = build_env(region)
    argv = [
        cloudfox, "aws", "pmapper",
        "-p", profile, "--outdir", outdir, "-v", "2", "-y",
        "--pmapper-data-basepath", result.pmapper_storage,
    ]
    started = time.time()
    completed = run(argv, env=env, stream=True, timeout=60 * 30)
    elapsed = time.time() - started
    note = "" if completed.ok else (completed.stderr or "").strip()[:300]
    result.log.append(RunLogEntry("cloudfox-pmapper", completed.cmd, completed.returncode, elapsed, note))
    if completed.ok:
        ok(f"CloudFox PMapper graph built in {elapsed:.0f}s")
        # The pmapper module writes its artifacts (pmapper-privesc-paths-*) into
        # the same <outdir>/cloudfox-output/aws/<profile>-<account> directory as
        # every other module, so the main evidence walk already picks them up.
        result.cloudfox_pmapper_dir = result.cloudfox_output_dir
    else:
        warn(f"CloudFox PMapper module failed (rc={completed.returncode}): {note}")
