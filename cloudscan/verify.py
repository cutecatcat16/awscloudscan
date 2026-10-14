"""Verify CloudFox privilege-escalation candidates against the AWS API.

CloudFox enumerates what a principal *may* be able to do, based only on the
policies it can read at scan time. That is a candidate index, not a verdict: it
ignores resource scoping, condition keys, SCPs and permission boundaries. It is
the reason a role holding ``lambda:UpdateFunction*`` on one specific function
looks dangerous when evaluated against ``*`` and harmless against the function
it actually names.

This module closes that gap. It takes the interesting ``(principal, action,
resource)`` tuples CloudFox harvested, asks IAM's own evaluator
(``iam:SimulatePrincipalPolicy``) what each one really resolves to, and
classifies the answer:

===========================  ==================================================
``confirmed``                The API allows it. The finding is real.
``conditional``              Allowed only while a condition (often MFA) holds.
``not_allowed``              Implicit deny at this resource: the candidate was noise.
``denied``                   An explicit deny exists; itself a control worth noting.
``unverified``               The API refused or failed; fall back to local review.
===========================  ==================================================

``SimulatePrincipalPolicy`` is a read-only calculator. It mutates nothing, does
not act as the target principal, and needs only the caller's own permission
(``iam:Simulate*``, granted by ``IAMReadOnlyAccess``, ``SecurityAudit`` and the
full ``ReadOnlyAccess`` policy), so it is available on a read-only engagement.

Design rules enforced here:

1. **CloudFox tells you what to ask; AWS tells you the answer.** CloudFox is
   never re-run and its verdicts are never trusted.
2. **Simulate the resource the row names, never ``*``.** Simulating ``*``
   produces ``implicitDeny`` for anything resource-scoped, which files a false
   negative. ``*`` is only used when the policy statement itself says ``*``.
3. **Cache every answer to disk, keyed by a hash of the question.** A re-run
   costs nothing, and the cache doubles as the audit trail for the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config
# Grading and the probe set come from the shared taxonomy, so the verifier and
# the prompt cannot disagree about what a grant is or what it grades to.
from .issues import (CRITICAL_ACTIONS as _CRITICAL_ACTIONS,
                     DATA_ACCESS_ACTIONS as _DATA_ACCESS_ACTIONS,
                     HIGH_ACTIONS as _HIGH_ACTIONS,
                     HIGH_IMPACT_DATA as _HIGH_IMPACT_DATA,
                     PRIVESC_ACTIONS,
                     SEVERITY_RANK as _SEVERITY_RANK,
                     category_for as _category_for,
                     severity_for as _severity_for)
from .runner import build_env
from .util import debug, info, read_json, run, warn

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# ``SimulatePrincipalPolicy`` accepts up to 100 action names per call. Batching
# by (principal, resource) collapses ~1,000 candidates into a few hundred calls;
# 50 keeps each call comfortably inside the documented limit.
DEFAULT_BATCH_SIZE = 50
MAX_ACTIONS_PER_CALL = 100
# Retries for a throttled or transiently failing call.
MAX_API_RETRIES = 3
CACHE_FILENAME = "verify-cache.json"
# Flush the on-disk cache this often (in API calls), so an interrupted run keeps
# whatever it has already paid for.
CACHE_FLUSH_EVERY = 25

# Verdicts.
CONFIRMED = "confirmed"
CONDITIONAL = "conditional"
NOT_ALLOWED = "not_allowed"
DENIED = "denied"
UNVERIFIED = "unverified"

VERDICT_LABELS = {
    CONFIRMED: "Confirmed",
    CONDITIONAL: "Conditional",
    NOT_ALLOWED: "Not allowed",
    DENIED: "Explicitly denied",
    UNVERIFIED: "Unverified",
}

# ``_CRITICAL_ACTIONS``, ``_HIGH_ACTIONS``, ``_DATA_ACCESS_ACTIONS``,
# ``_HIGH_IMPACT_DATA`` and ``_SEVERITY_RANK`` are imported from
# :mod:`cloudscan.issues` above — see the module docstring there for why the
# data-access set is graded apart from escalation.

# Throttling / transient error codes worth retrying.
_RETRYABLE_CODES = {
    "Throttling", "ThrottlingException", "RequestLimitExceeded",
    "ServiceFailure", "InternalFailure", "InternalError", "RequestTimeout",
}
# Error codes that mean "we are not allowed to ask" rather than "the answer is no".
_PERMISSION_CODES = {
    "AccessDenied", "AccessDeniedException", "UnauthorizedOperation",
    "NotAuthorized", "AuthorizationError", "NoPermission",
}
# Errors that will fail identically for every remaining candidate, so the run
# should stop instead of warning once per batch.
_FATAL_CODES = {
    "ProfileNotFound", "InvalidClientTokenId", "InvalidClientTokenIdException",
    "SignatureDoesNotMatch", "ExpiredToken", "ExpiredTokenException",
    "UnrecognizedClientException", "InvalidSignatureException",
}
# The AWS CLI reports these in prose rather than as a code token.
_FATAL_TEXT = (
    "could not be found",
    "Unable to locate credentials",
    "The security token included in the request is invalid",
    "The request signature we calculated does not match",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One ``(principal, action, resource)`` tuple worth asking the API about."""

    principal_arn: str
    action: str
    resource: str
    principal_name: str = ""
    principal_type: str = ""
    policy: str = ""
    policy_name: str = ""
    condition: bool = False
    account: str = ""
    # Set when *action* came from expanding a wildcard statement (e.g. ``s3:*``).
    source_action: str = ""

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.principal_arn, self.action, self.resource)

    @property
    def action_lower(self) -> str:
        return self.action.strip().lower()


@dataclass
class Decision:
    """The API's answer for a single candidate."""

    candidate: Candidate
    verdict: str
    eval_decision: str = ""            # allowed | implicitDeny | explicitDeny
    missing_context: List[str] = field(default_factory=list)
    context_values: Dict[str, str] = field(default_factory=dict)
    matched_statements: List[dict] = field(default_factory=list)
    org_allowed: Optional[bool] = None
    mfa_required: Optional[bool] = None
    reason: str = ""
    from_cache: bool = False
    simulated_resource: str = ""

    @property
    def confirmed(self) -> bool:
        return self.verdict in (CONFIRMED, CONDITIONAL)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "principal_arn": self.candidate.principal_arn,
            "principal_name": self.candidate.principal_name,
            "action": self.candidate.action,
            "resource": self.candidate.resource,
            "verdict": self.verdict,
            "eval_decision": self.eval_decision,
            "missing_context": list(self.missing_context),
            "context_values": dict(self.context_values),
            "matched_statements": len(self.matched_statements),
            "organizations_allowed": self.org_allowed,
            "mfa_required": self.mfa_required,
            "policy": self.candidate.policy,
            "policy_name": self.candidate.policy_name,
            "source_action": self.candidate.source_action,
            "reason": self.reason,
        }


@dataclass
class VerifyResult:
    """Aggregate outcome of a verification pass."""

    decisions: List[Decision] = field(default_factory=list)
    api_calls: int = 0
    cache_hits: int = 0
    errors: List[str] = field(default_factory=list)
    degraded: bool = False            # set when the API became unavailable
    skipped: int = 0                  # candidates not attempted
    # ARN -> reason, for principals whose grants are expected rather than risky
    # (administrators, AWS service-linked roles). They reach no finding.
    suppressed: Dict[str, str] = field(default_factory=dict)

    def by_verdict(self) -> Dict[str, List[Decision]]:
        buckets: Dict[str, List[Decision]] = {}
        for decision in self.decisions:
            buckets.setdefault(decision.verdict, []).append(decision)
        return buckets

    def confirmed(self) -> List[Decision]:
        return [d for d in self.decisions if d.verdict == CONFIRMED]

    def conditional(self) -> List[Decision]:
        return [d for d in self.decisions if d.verdict == CONDITIONAL]

    def actionable(self) -> List[Decision]:
        """Decisions worth writing up: confirmed and conditional grants."""
        return [d for d in self.decisions if d.confirmed]

    def to_findings(self) -> List[Dict[str, Any]]:
        """Shape the verified grants into the report's finding schema."""
        return decisions_to_findings(self.actionable(), suppressed=self.suppressed)

    def suppressed_grants(self) -> List[Decision]:
        """Actionable grants withheld because the principal's privilege is expected."""
        return [d for d in self.actionable() if d.candidate.principal_arn in self.suppressed]

    def summary(self) -> Dict[str, Any]:
        counts = {v: len(vs) for v, vs in self.by_verdict().items()}
        withheld = self.suppressed_grants()
        return {
            "candidates": len(self.decisions) + self.skipped,
            "checked": len(self.decisions),
            "skipped": self.skipped,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "degraded": self.degraded,
            "counts": counts,
            # Kept so a reviewer can see what was set aside, and why, without
            # having to re-derive it from the principals inventory.
            "suppressed_principals": len(self.suppressed),
            "suppressed_grants": len(withheld),
            "suppressed": {arn: reason for arn, reason in sorted(self.suppressed.items())},
        }


# ---------------------------------------------------------------------------
# Candidate selection (local, no API calls)
# ---------------------------------------------------------------------------

# Wildcard statements cannot be simulated: ``SimulatePrincipalPolicy`` rejects
# action names shorter than three characters and knows nothing about globs. So a
# statement is expanded into the concrete escalation primitives it covers, and
# each primitive is verified in its own right. This is also the correct reading
# of the policy: ``iam:*`` genuinely grants every one of these actions.
_WILDCARD_CONCRETE_HINTS = {
    "organizations:*": ["organizations:leaveorganization"],
}


def _concrete_privesc_actions() -> List[str]:
    actions = {a for a in PRIVESC_ACTIONS if "*" not in a}
    for hint in _WILDCARD_CONCRETE_HINTS.values():
        actions.update(hint)
    return sorted(actions)


_CONCRETE_PRIVESC_ACTIONS = _concrete_privesc_actions()


def _wildcard_covers(pattern: str, action: str) -> bool:
    """Case-insensitive glob match: does *pattern* grant *action*?"""
    return fnmatchcase(action.strip().lower(), pattern.strip().lower())


def expand_action(action: str) -> List[str]:
    """Return the concrete actions a statement grants, expanding wildcards.

    A concrete privileged action maps to itself; ``*`` maps to every escalation
    primitive; ``svc:*`` and ``svc:Prefix*`` map to the primitives they cover.
    Anything covering no privilege-escalation primitive maps to nothing.
    """
    a = (action or "").strip()
    if not a:
        return []
    if "*" not in a:
        return [a] if a.lower() in PRIVESC_ACTIONS else []
    if a == "*":
        return list(_CONCRETE_PRIVESC_ACTIONS)
    covered = [p for p in _CONCRETE_PRIVESC_ACTIONS if _wildcard_covers(a, p)]
    if covered:
        return covered
    return [h for pattern, hints in _WILDCARD_CONCRETE_HINTS.items()
            if _wildcard_covers(a, pattern) for h in hints]


def is_candidate_action(action: str) -> bool:
    """True when a statement could grant any escalation primitive."""
    return bool(expand_action(action))


def _norm_resource(value: str) -> str:
    resource = (value or "").strip()
    return resource or "*"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in ("true", "yes", "1", "y")


def build_candidates(
    rows: Iterable[Dict[str, str]],
    *,
    account: str = "",
    expand: Callable[[str], List[str]] = expand_action,
) -> List[Candidate]:
    """Turn ``permissions.csv`` rows into the distinct tuples worth verifying.

    Keeps ``Allow`` statements that could grant a privilege-escalation primitive,
    expanding wildcards into concrete actions, and deduplicates on
    ``(arn, action, resource)``. No API calls, no I/O.
    """
    seen: Dict[Tuple[str, str, str], Candidate] = {}
    for row in rows:
        if str(row.get("Effect", "")).strip().lower() != "allow":
            continue
        arn = (row.get("Arn") or "").strip()
        statement_action = (row.get("Action") or "").strip()
        if not arn or not statement_action:
            continue
        resource = _norm_resource(row.get("Resource", ""))
        for action in expand(statement_action):
            key = (arn, action, resource)
            if key in seen:
                continue
            seen[key] = Candidate(
                principal_arn=arn,
                action=action,
                resource=resource,
                source_action=statement_action if statement_action != action else "",
                principal_name=(row.get("Name") or "").strip(),
                principal_type=(row.get("Type") or "").strip(),
                policy=(row.get("Policy") or "").strip(),
                policy_name=(row.get("Policy Name") or "").strip(),
                condition=_truthy(row.get("Condition")),
                account=(row.get("Account") or account or "").strip(),
            )
    return sorted(seen.values(), key=lambda c: (c.principal_arn, c.resource, c.action.lower()))


def group_candidates(
    candidates: Sequence[Candidate], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> List[Tuple[str, str, List[Candidate]]]:
    """Group candidates by ``(principal, resource)`` then chunk the actions.

    One API call answers a whole batch of actions against a single resource, so
    this is what collapses ~1,000 candidates into a few hundred calls.
    """
    batch_size = max(1, min(int(batch_size or DEFAULT_BATCH_SIZE), MAX_ACTIONS_PER_CALL))
    grouped: Dict[Tuple[str, str], List[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.principal_arn, candidate.resource), []).append(candidate)

    batches: List[Tuple[str, str, List[Candidate]]] = []
    for (arn, resource), items in sorted(grouped.items()):
        items = sorted(items, key=lambda c: c.action.lower())
        for start in range(0, len(items), batch_size):
            batches.append((arn, resource, items[start:start + batch_size]))
    return batches


def _dedupe_actions(actions: Iterable[str]) -> List[str]:
    """Collapse action names that differ only in case, preserving first spelling.

    CloudFox records the same action with whatever case each policy used, and
    ``SimulatePrincipalPolicy`` rejects a request holding two spellings of one
    action ("require different authorization information"). Responses are looked
    up case-insensitively, so every candidate still receives its verdict.
    """
    unique: Dict[str, str] = {}
    for action in actions:
        name = (action or "").strip()
        if name:
            unique.setdefault(name.lower(), name)
    return list(unique.values())


# ---------------------------------------------------------------------------
# Simulator backends
# ---------------------------------------------------------------------------


class SimulateError(RuntimeError):
    """Raised when a simulate call fails for a reason worth surfacing."""

    def __init__(self, message: str, *, code: str = "", permission: bool = False,
                 retryable: bool = False, fatal: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.permission = permission
        self.retryable = retryable
        self.fatal = fatal


def _classify_error_code(code: str) -> Tuple[bool, bool]:
    """Return ``(is_permission_error, is_retryable)`` for an AWS error code."""
    code = (code or "").strip()
    if code in _PERMISSION_CODES:
        return True, False
    if code in _RETRYABLE_CODES:
        return False, True
    if code.endswith("AccessDenied") or code.startswith("AccessDenied"):
        return True, False
    return False, False


def _is_fatal(code: str, text: str = "") -> bool:
    """True when the failure would repeat for every remaining candidate."""
    if (code or "").strip() in _FATAL_CODES:
        return True
    return any(marker in (text or "") for marker in _FATAL_TEXT)


class Simulator:
    """Base class: ask AWS what a tuple resolves to."""

    def simulate(
        self,
        principal_arn: str,
        actions: Sequence[str],
        resource: str,
        context: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


class BotoSimulator(Simulator):
    """Simulate via boto3 (preferred: structured errors, no process spawn)."""

    def __init__(self, profile: str, region: str) -> None:
        import boto3  # type: ignore

        self.profile = profile
        self.region = region
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("iam")

    def describe(self) -> str:
        return f"boto3 iam:SimulatePrincipalPolicy (profile={self.profile}, region={self.region})"

    def simulate(self, principal_arn, actions, resource, context=None):
        kwargs: Dict[str, Any] = {
            "PolicySourceArn": principal_arn,
            "ActionNames": list(actions),
            "ResourceArns": [resource],
        }
        if context:
            kwargs["ContextEntries"] = list(context)
        from botocore.exceptions import ClientError, ParamValidationError  # type: ignore

        try:
            response = self.client.simulate_principal_policy(**kwargs)
        except ParamValidationError as exc:
            # A malformed action name is our bug, not an AWS refusal: fail this
            # batch rather than the whole run.
            raise SimulateError(f"invalid simulate request: {exc}") from exc
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", ""))
            message = str(error.get("Message", str(exc)))
            permission, retryable = _classify_error_code(code)
            raise SimulateError(
                f"{code}: {message}",
                code=code, permission=permission, retryable=retryable,
                fatal=_is_fatal(code, message),
            ) from exc
        return _index_results(response.get("EvaluationResults", []), resource)


class CliSimulator(Simulator):
    """Simulate by shelling out to the AWS CLI (fallback when boto3 is absent)."""

    def __init__(self, profile: str, region: str, aws_path: Optional[str] = None) -> None:
        self.profile = profile
        self.region = region
        self.aws = aws_path or "aws"
        self.env = build_env(region)

    def describe(self) -> str:
        return f"aws iam simulate-principal-policy (profile={self.profile}, region={self.region})"

    def simulate(self, principal_arn, actions, resource, context=None):
        argv = [
            self.aws, "iam", "simulate-principal-policy",
            "--policy-source-arn", principal_arn,
            "--action-names", *[str(a) for a in actions],
            "--resource-arns", resource,
            "--profile", self.profile,
            "--region", self.region,
            "--output", "json",
        ]
        for entry in context or []:
            values = entry.get("ContextKeyValues") or []
            if not values:
                continue
            argv += [
                "--context-entries",
                f"ContextKeyName={entry.get('ContextKeyName', '')},"
                f"ContextKeyType={entry.get('ContextKeyType', 'string')},"
                f"ContextKeyValues={','.join(str(v) for v in values)}",
            ]
        completed = run(argv, env=self.env, timeout=120, log_command=False)
        if not completed.ok:
            stderr = (completed.stderr or "").strip()
            code = _error_code_from_cli(stderr)
            permission, retryable = _classify_error_code(code)
            if not code and permission is False and "AccessDenied" in stderr:
                permission = True
            raise SimulateError(stderr or f"aws cli exited {completed.returncode}",
                                code=code, permission=permission, retryable=retryable,
                                fatal=_is_fatal(code, stderr))
        try:
            payload = json.loads(completed.stdout or "{}")
        except ValueError as exc:
            raise SimulateError(f"could not parse AWS CLI output: {exc}") from exc
        return _index_results(payload.get("EvaluationResults", []), resource)


def _error_code_from_cli(stderr: str) -> str:
    """Pull the AWS error code out of a CLI error message."""
    for token in (stderr or "").replace("\n", " ").split():
        cleaned = token.strip("\"'(),.:")
        if cleaned[:1].isupper() and ("Denied" in cleaned or cleaned in _RETRYABLE_CODES):
            return cleaned
    return ""


def make_simulator(
    profile: str,
    region: str,
    *,
    aws_path: Optional[str] = None,
    prefer_cli: bool = False,
) -> Simulator:
    """Return a working simulator, preferring boto3 and falling back to the CLI."""
    if not prefer_cli:
        try:
            return BotoSimulator(profile, region)
        except ImportError:
            debug("boto3 not available; using the AWS CLI for policy simulation")
        except Exception as exc:  # noqa: BLE001 - fall back rather than abort
            debug(f"boto3 simulator unavailable ({exc}); using the AWS CLI")
    return CliSimulator(profile, region, aws_path=aws_path)


def _index_results(results: Sequence[dict], resource: str) -> Dict[str, Dict[str, Any]]:
    """Collapse ``EvaluationResults`` into ``{action: summary}``.

    A statement that matches every resource is reported against the resource we
    asked about, so an action is only recorded when its result actually covers
    the requested resource (or a wildcard).
    """
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in results or []:
        action = str(item.get("EvalActionName", "")).strip()
        if not action:
            continue
        eval_resource = str(item.get("EvalResourceName", "")).strip() or resource
        if eval_resource not in (resource, "*") and resource != "*":
            # The API evaluated a different resource; for a single-resource call
            # this should not happen, so prefer the entry that names our resource.
            if action in indexed:
                continue
        org_detail = item.get("OrganizationsDecisionDetail") or {}
        entry = indexed.setdefault(action, {})
        decision = str(item.get("EvalDecision", "")).strip()
        # An explicit deny against the same action always wins.
        previous = entry.get("decision", "")
        if previous == "explicitDeny":
            continue
        entry.update({
            "decision": decision,
            "missing_context": list(item.get("MissingContextValues") or []),
            "matched_statements": list(item.get("MatchedStatements") or []),
            "org_allowed": org_detail.get("AllowedByOrganizations"),
            "resource": eval_resource,
            "decision_details": dict(item.get("EvalDecisionDetails") or {}),
        })
    return indexed


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def _cache_key(
    principal_arn: str,
    actions: Sequence[str],
    resource: str,
    context: Optional[Sequence[Dict[str, Any]]],
) -> str:
    canonical = json.dumps(
        {
            "arn": principal_arn,
            "actions": sorted(str(a) for a in actions),
            "resource": resource,
            "context": _canonical_context(context),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_context(context: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in context or []:
        out.append({
            "key": entry.get("ContextKeyName", ""),
            "type": entry.get("ContextKeyType", "string"),
            "values": sorted(str(v) for v in (entry.get("ContextKeyValues") or [])),
        })
    return sorted(out, key=lambda e: (e["key"], e["type"]))


class SimulateCache:
    """Disk-backed memo of API answers, keyed by a hash of the question.

    Deterministic: the same question never costs a second API call, which makes
    re-runs free and gives the report a verifiable audit trail.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0
        self.dirty = False
        if path and os.path.isfile(path):
            payload = read_json(path)
            if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
                self.entries = payload["entries"]

    def get(self, key: str) -> Optional[Dict[str, Dict[str, Any]]]:
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: Dict[str, Dict[str, Any]]) -> None:
        self.entries[key] = value
        self.dirty = True

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        payload = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": self.entries,
        }
        tmp = f"{self.path}.tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True, default=str)
            os.replace(tmp, self.path)
            self.dirty = False
        except OSError as exc:
            warn(f"could not write the verification cache: {exc}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _mfa_context(present: bool) -> List[Dict[str, Any]]:
    return [{
        "ContextKeyName": "aws:MultiFactorAuthPresent",
        "ContextKeyType": "boolean",
        "ContextKeyValues": ["true" if present else "false"],
    }]


def _verdict_from_eval(eval_decision: str, missing_context: Sequence[str]) -> str:
    if eval_decision == "allowed":
        return CONDITIONAL if missing_context else CONFIRMED
    if eval_decision == "explicitDeny":
        return DENIED
    # implicitDeny with missing context means the grant may exist behind a condition.
    if missing_context:
        return CONDITIONAL
    return NOT_ALLOWED


def _call_with_retries(
    simulator: Simulator,
    arn: str,
    actions: Sequence[str],
    resource: str,
    context: Optional[Sequence[Dict[str, Any]]],
    *,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = MAX_API_RETRIES,
) -> Dict[str, Dict[str, Any]]:
    last: Optional[SimulateError] = None
    for attempt in range(max(1, retries + 1)):
        try:
            return simulator.simulate(arn, actions, resource, context)
        except SimulateError as exc:
            last = exc
            if exc.permission or not exc.retryable:
                raise
            if attempt < retries:
                sleep(min(2 ** attempt, 8))
    raise last or SimulateError("simulate failed")


def verify(
    candidates: Sequence[Candidate],
    simulator: Simulator,
    *,
    cache: Optional[SimulateCache] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mfa_probe: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    progress: Optional[Callable[[int, int], None]] = None,
) -> VerifyResult:
    """Ask AWS about every candidate and classify the answers.

    Degrades gracefully: the first ``AccessDenied`` stops further calls and
    marks the remainder ``unverified`` rather than failing the run.
    """
    result = VerifyResult()
    batches = group_candidates(candidates, batch_size=batch_size)
    total = len(candidates)
    done = 0

    def _simulate(arn: str, actions: Sequence[str], resource: str) -> Dict[str, Dict[str, Any]]:
        """Call the simulator, splitting the batch if AWS rejects the request outright.

        The API refuses a request that carries two spellings of the same action
        ("require different authorization information") and names every action in
        the batch, so a single bad entry would otherwise cost the whole batch its
        verdicts. Halving down to one isolates the offender.
        """
        result.api_calls += 1
        try:
            return _call_with_retries(simulator, arn, actions, resource, None, sleep=sleep)
        except SimulateError as exc:
            if len(actions) > 1 and not (exc.permission or exc.fatal):
                warn(f"simulator rejected a {len(actions)}-action request for {arn} ({exc}); "
                     f"retrying it in halves")
                middle = len(actions) // 2
                merged = _simulate(arn, actions[:middle], resource)
                merged.update(_simulate(arn, actions[middle:], resource))
                return merged
            raise

    try:
        for arn, resource, items in batches:
            actions = _dedupe_actions([c.action for c in items])
            key = _cache_key(arn, actions, resource, None)
            indexed = cache.get(key) if cache else None
            served_from_cache = indexed is not None
            if indexed is None:
                try:
                    indexed = _simulate(arn, actions, resource)
                    if cache:
                        cache.put(key, indexed)
                except SimulateError as exc:
                    message = f"{arn} @ {resource}: {exc}"
                    result.errors.append(message)
                    if exc.permission or exc.fatal:
                        if exc.permission:
                            warn(f"policy simulation is not permitted ({exc.code or 'AccessDenied'})")
                        else:
                            warn(f"policy simulation cannot run: {exc}")
                        warn("marking the remaining candidates unverified — the report "
                             "will flag them as static-only rather than claiming a verdict")
                        result.degraded = True
                        result.skipped += total - done
                        break
                    warn(f"simulation failed for {arn}: {exc}")
                    for candidate in items:
                        result.decisions.append(Decision(
                            candidate=candidate, verdict=UNVERIFIED,
                            simulated_resource=resource, reason=str(exc),
                        ))
                    done += len(items)
                    continue
            else:
                result.cache_hits += 1

            for candidate in items:
                entry = indexed.get(candidate.action) or _case_insensitive_lookup(indexed, candidate.action)
                result.decisions.append(
                    _decision_for(candidate, entry, resource, simulator, cache=cache,
                                  mfa_probe=mfa_probe, sleep=sleep, from_cache=served_from_cache)
                )
            done += len(items)
            if progress:
                progress(done, total)
            # Flush periodically: a crash or Ctrl-C halfway through a 500-call
            # run must not throw away the answers already paid for.
            if cache and result.api_calls and result.api_calls % CACHE_FLUSH_EVERY == 0:
                cache.save()
    finally:
        if cache:
            cache.save()

    if result.cache_hits:
        info(f"verification: {result.api_calls} API calls, "
             f"{result.cache_hits} cache hits ({len(result.decisions)} candidates checked)")
    return result


def _case_insensitive_lookup(indexed: Dict[str, Dict[str, Any]], action: str) -> Optional[Dict[str, Any]]:
    wanted = action.strip().lower()
    for name, entry in indexed.items():
        if name.strip().lower() == wanted:
            return entry
    return None


def _decision_for(
    candidate: Candidate,
    entry: Optional[Dict[str, Any]],
    resource: str,
    simulator: Simulator,
    *,
    cache: Optional[SimulateCache],
    mfa_probe: bool,
    sleep: Callable[[float], None],
    from_cache: bool = False,
) -> Decision:
    if entry is None:
        return Decision(
            candidate=candidate, verdict=UNVERIFIED, simulated_resource=resource,
            reason="the API returned no evaluation for this action",
            from_cache=from_cache,
        )

    eval_decision = str(entry.get("decision", ""))
    missing = list(entry.get("missing_context") or [])
    verdict = _verdict_from_eval(eval_decision, missing)
    decision = Decision(
        candidate=candidate,
        verdict=verdict,
        eval_decision=eval_decision,
        missing_context=missing,
        matched_statements=list(entry.get("matched_statements") or []),
        org_allowed=entry.get("org_allowed"),
        simulated_resource=entry.get("resource", resource),
        from_cache=from_cache,
    )

    # An implicit deny that hinges on MFA is the single most common shape in a
    # real account, so probe it explicitly rather than reporting "unknown".
    if mfa_probe and "aws:MultiFactorAuthPresent" in missing:
        probed = _probe_mfa(candidate, resource, simulator, cache=cache, sleep=sleep)
        if probed is not None:
            decision.mfa_required = probed.mfa_required
            decision.verdict = probed.verdict
            decision.reason = probed.reason
            if probed.verdict == CONFIRMED:
                decision.context_values = {"aws:MultiFactorAuthPresent": "true and false"}
            else:
                decision.context_values = {"aws:MultiFactorAuthPresent": "true"}
    return decision


def _probe_mfa(
    candidate: Candidate,
    resource: str,
    simulator: Simulator,
    *,
    cache: Optional[SimulateCache],
    sleep: Callable[[float], None],
) -> Optional[Decision]:
    """Ask what happens with and without MFA.

    ``present=true`` allowed and ``present=false`` denied means the grant is real
    but gated by MFA: that is a *conditional* finding, not a false positive.
    Allowed either way means MFA is irrelevant and the grant is confirmed.
    Denied both ways means the MFA condition was incidental and the deny stands.
    """
    outcomes: Dict[bool, Optional[bool]] = {}
    for present in (True, False):
        context = _mfa_context(present)
        key = _cache_key(candidate.principal_arn, [candidate.action], resource, context)
        indexed = cache.get(key) if cache else None
        if indexed is None:
            try:
                indexed = _call_with_retries(
                    simulator, candidate.principal_arn, [candidate.action], resource, context, sleep=sleep
                )
                if cache:
                    cache.put(key, indexed)
            except SimulateError as exc:
                debug(f"MFA probe failed for {candidate.action}: {exc}")
                outcomes[present] = None
                continue
        entry = indexed.get(candidate.action) or _case_insensitive_lookup(indexed, candidate.action)
        if entry is None:
            outcomes[present] = None
            continue
        outcomes[present] = str(entry.get("decision", "")) == "allowed"

    with_mfa = outcomes.get(True)
    without_mfa = outcomes.get(False)
    if with_mfa and without_mfa is True:
        return Decision(candidate=candidate, verdict=CONFIRMED, mfa_required=False,
                        reason="allowed regardless of MFA")
    if with_mfa is True:
        return Decision(candidate=candidate, verdict=CONDITIONAL, mfa_required=True,
                        reason="allowed only when MFA is present")
    return None


# ---------------------------------------------------------------------------
# Reporting glue
# ---------------------------------------------------------------------------


def admin_principals(evidence: Any) -> set:
    """Principals CloudFox already rates as administrative.

    PMapper refuses to originate an edge from a principal it knows to be admin
    (``if node_source.is_admin: continue`` in its Lambda checker): an admin can
    already do everything, so listing its "escalation" paths describes the
    baseline rather than a finding. On this account that distinction is the
    whole difference between a usable report and a wall of noise — four admin
    roles account for 456 of 528 raw edges. The join follows PMapper's rule so
    that dropping its checkers does not change what gets reported.
    """
    admins: set = set()
    for row in evidence.csv("principals"):
        arn = (row.get("Arn") or "").strip()
        if arn and _truthy(row.get("IsAdminRole?")):
            admins.add(arn)
    return admins


def is_service_linked(arn: str, name: str = "") -> bool:
    """True for AWS service-linked roles, which no user can assume."""
    lowered = (arn or "").lower()
    return "/aws-service-role/" in lowered or (name or "").startswith("AWSServiceRoleFor")


def suppressed_principals(evidence: Any) -> Dict[str, str]:
    """Principals that must not originate findings, mapped to the reason.

    A grant held by one of these describes the account's intended design, not a
    weakness, and reporting it buries the findings that matter:

    * **administrative principals** — CloudFox's ``IsAdminRole?``. "An
      administrator can create roles" is a tautology. On this account the four
      such roles held 259 of 764 findings.
    * **AWS service-linked roles** — created and assumed by an AWS service, not
      by a person. Nothing an attacker holds can act as them, so their grants
      are unreachable.

    Suppressed principals are recorded, not silently dropped: their grant count
    reaches ``analysis.json`` so a reviewer can see what was set aside and why.
    """
    reasons: Dict[str, str] = {}
    for arn in admin_principals(evidence):
        reasons[arn] = "administrative in its own right (CloudFox IsAdminRole? = YES)"
    for row in evidence.csv("principals"):
        arn = (row.get("Arn") or "").strip()
        name = (row.get("Name") or "").strip()
        if arn and arn not in reasons and is_service_linked(arn, name):
            reasons[arn] = "AWS service-linked role, not assumable by a user"
    return reasons


def _collapse_actions(items: Sequence[Decision]) -> List[Dict[str, Any]]:
    """Collapse one principal's decisions into one entry per action.

    Actions are grouped case-insensitively because CloudFox records
    ``ssm:StartAutomationExecution`` from one statement and
    ``ssm:startautomationexecution`` from another, and the two are one grant.
    """
    grouped: Dict[str, List[Decision]] = {}
    for decision in items:
        grouped.setdefault(decision.candidate.action.strip().lower(), []).append(decision)

    grants: List[Dict[str, Any]] = []
    for _, decisions in grouped.items():
        # Lead with the statement that names the action outright, so the report
        # shows the operator's spelling rather than a wildcard expansion of it.
        decisions = sorted(decisions, key=lambda d: (bool(d.candidate.source_action), d.candidate.action))
        first = decisions[0]
        resources = sorted({d.candidate.resource for d in decisions})
        grant = {
            "action": first.candidate.action,
            "resources": resources,
            "severity": _severity_for(first.candidate.action, unscoped="*" in resources),
            "category": _category_for(first.candidate.action),
            "conditional": any(d.verdict == CONDITIONAL for d in decisions),
            "mfa_required": any(bool(d.mfa_required) for d in decisions),
            "source_actions": sorted({d.candidate.source_action for d in decisions if d.candidate.source_action}),
            "policy": first.candidate.policy,
            "policy_name": first.candidate.policy_name,
            "eval_decision": first.eval_decision,
            "org_allowed": first.org_allowed,
            "matched_statements": len(first.matched_statements),
            "verdicts": sorted({d.verdict for d in decisions}),
        }
        grants.append(grant)
    grants.sort(key=lambda g: (-_SEVERITY_RANK.get(g["severity"], 0), g["action"]))
    return grants


def decisions_to_findings(
    decisions: Sequence[Decision], *, suppressed: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """Convert verified grants into the report's finding schema.

    One finding per **principal**, listing every verified grant it holds,
    worst first. The unit an operator fixes is the principal's policy, so
    emitting one finding per (principal, action) turned a single over-privileged
    role into 66 separate "issues" — 102 of the 141 principals here did that,
    and the report became unreadable. Per-grant detail is preserved in
    ``analysis.json`` under ``verification.actions``.
    """
    suppressed = suppressed or {}
    grouped: Dict[str, List[Decision]] = {}
    for decision in decisions:
        arn = decision.candidate.principal_arn
        if arn in suppressed:
            continue
        grouped.setdefault(arn, []).append(decision)

    findings: List[Dict[str, Any]] = []
    for index, (arn, items) in enumerate(sorted(grouped.items()), start=1):
        grants = _collapse_actions(items)
        worst = grants[0]
        count = len(grants)
        principal_name = items[0].candidate.principal_name or arn
        gate = "only when MFA is present" if worst["mfa_required"] else "only under a condition"
        qualifier = f" {gate}" if (worst["conditional"] or worst["mfa_required"]) else ""

        if count == 1:
            title = f"{principal_name} can perform {worst['action']}{qualifier}"
        else:
            title = (f"{principal_name} can perform {worst['action']}{qualifier} "
                     f"and {count - 1} other verified action(s)")

        resources = sorted({r for grant in grants for r in grant["resources"]})
        evidence_lines = [
            f"Verified with iam:SimulatePrincipalPolicy: {count} grant(s) confirmed for {arn}.",
            "Actions, most severe first:",
        ]
        for grant in grants:
            flags = []
            if grant["conditional"]:
                flags.append("conditional")
            if grant["mfa_required"]:
                flags.append("MFA-gated")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            evidence_lines.append(
                f"- {grant['action']} ({grant['severity']}, {grant['category']}) on "
                f"{', '.join(grant['resources'])}{suffix}"
            )
        wildcards = sorted({w for grant in grants for w in grant["source_actions"]})
        for wildcard in wildcards:
            evidence_lines.append(f"Granted by a wildcard statement: {wildcard}")
        if worst["policy"]:
            evidence_lines.append(f"Source statement for {worst['action']}: "
                                  f"{worst['policy']} {worst['policy_name']}".strip())
        if worst["mfa_required"]:
            evidence_lines.append("Re-simulating with aws:MultiFactorAuthPresent=false returns a deny, "
                                  "so the grant is gated by MFA.")

        impact = _impact_for(worst["action"])
        if count > 1:
            impact = (f"{count} verified grant(s); the most severe is {worst['action']}. ") + impact

        findings.append({
            "id": f"VER-{index:03d}",
            "category": worst["category"],
            # A grant that only works with MFA is real, but it is not freely
            # usable, so it does not carry the same confidence as a plain allow.
            "confidence": "medium" if (worst["conditional"] or worst["mfa_required"]) else "high",
            "title": title,
            "description": (
                f"AWS confirms that {arn} is allowed to call {count} action(s) of "
                f"interest, led by {worst['action']} on {', '.join(worst['resources'])}. "
                "This was verified against the live IAM evaluation engine, not "
                "inferred from policy text."
            ),
            "evidence": "\n".join(evidence_lines),
            "impact": impact,
            "severity": worst["severity"],
            "recommendation": _recommendation_for(worst["action"]),
            "affected_principals": [arn],
            "affected_resources": resources,
            "from_principal": arn,
            "to_principal": "",
            "primitives": [grant["action"] for grant in grants],
            "reason": items[0].reason or "verified by AWS policy simulation",
            "verified": True,
            "verification": {
                "verdicts": sorted({d.verdict for d in items}),
                "mfa_required": any(grant["mfa_required"] for grant in grants),
                "eval_decision": worst["eval_decision"],
                "org_allowed": worst["org_allowed"],
                "source_actions": wildcards,
                "grant_count": count,
                "actions": grants,
            },
        })
    return findings


def _impact_for(action: str) -> str:
    lowered = (action or "").strip().lower()
    if lowered in _DATA_ACCESS_ACTIONS:
        return ("Sensitive-data exposure: this action reads or writes data the principal may "
                "have no business touching, including secrets, keys or customer objects.")
    if lowered in _CRITICAL_ACTIONS or lowered == "*":
        return ("Direct privilege escalation: this action can grant the principal "
                "additional permissions or credentials, up to full administrative control.")
    if lowered in _HIGH_ACTIONS:
        return ("Privilege escalation or lateral movement: this action can execute "
                "attacker-influenced code, assume another identity, or expose data.")
    return "Expands the principal's effective access beyond its intended role."


def _recommendation_for(action: str) -> str:
    lowered = (action or "").strip().lower()
    if lowered in _DATA_ACCESS_ACTIONS:
        return ("Scope the action to the resources the principal actually needs and confirm "
                "the data it reaches is meant to be readable by it.")
    if lowered in _CRITICAL_ACTIONS or lowered == "*":
        return ("Remove this action, scope it to specific resources, and require MFA "
                "for any remaining administrative calls.")
    if lowered in _HIGH_ACTIONS:
        return ("Scope the action to the specific resources the principal needs and "
                "add a condition (for example aws:SourceArn or MFA) to bound its use.")
    return "Confirm the principal needs this action; if not, remove it from the policy."


def attach_to_analysis(analysis: Any, verify_result: VerifyResult) -> int:
    """Merge verified findings into an :class:`Analysis`, returning how many were added.

    Verified grants are placed first so the report leads with what AWS confirmed.
    """
    findings = verify_result.to_findings()
    if not findings:
        return 0
    existing = list(getattr(analysis, "findings", []) or [])
    existing_titles = {str(f.get("title", "")).strip() for f in existing if isinstance(f, dict)}
    fresh = [f for f in findings if str(f.get("title", "")).strip() not in existing_titles]
    analysis.findings = fresh + existing
    summary = getattr(analysis, "summary", "") or ""
    counts = verify_result.summary().get("counts", {})
    line = (f"Verified against the AWS IAM evaluation engine: "
            f"{counts.get(CONFIRMED, 0)} confirmed, {counts.get(CONDITIONAL, 0)} conditional "
            f"grant(s), written up as {len(fresh)} principal-level finding(s).")
    withheld = verify_result.suppressed_grants()
    if withheld:
        line += (f" A further {len(withheld)} confirmed grant(s) held by {len(verify_result.suppressed)} "
                 f"administrative or service-linked principal(s) are not findings — they describe the "
                 f"account's intended design and are recorded in analysis.json instead.")
    analysis.summary = (summary + "\n\n" + line).strip() if summary else line
    return len(fresh)


def verify_evidence(
    evidence: Any,
    *,
    profile: str = "",
    region: str = "",
    cache_path: Optional[str] = None,
    simulator: Optional[Simulator] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mfa_probe: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> VerifyResult:
    """Convenience entry point: verify the ``permissions`` artifact of an Evidence.

    Reads only local CloudFox output until the API calls begin, so it can run
    while other collection is still in flight.
    """
    rows = evidence.csv("permissions") if hasattr(evidence, "csv") else []
    if not rows:
        warn("no permissions.csv in the evidence — nothing to verify")
        return VerifyResult()
    candidates = build_candidates(rows, account=getattr(evidence, "account", ""))
    if not candidates:
        info("no privilege-escalation candidates found in the permissions artifact")
        return VerifyResult()
    info(f"verification: {len(candidates)} candidate tuple(s) over "
         f"{len({c.principal_arn for c in candidates})} principal(s)")

    if simulator is None:
        if not profile:
            warn("no AWS profile supplied — cannot reach the simulate API")
            return VerifyResult(degraded=True, skipped=len(candidates))
        simulator = make_simulator(profile, region)
    info(f"verifier: {simulator.describe()}")

    cache = SimulateCache(cache_path) if cache_path else None
    if cache and cache.entries:
        info(f"verification cache: {len(cache.entries)} previous answer(s) loaded")

    result = verify(candidates, simulator, cache=cache, batch_size=batch_size,
                    mfa_probe=mfa_probe, sleep=sleep)
    result.suppressed = suppressed_principals(evidence)
    if result.suppressed:
        info(f"suppression: {len(result.suppressed)} principal(s) whose privilege is expected "
             f"(administrators and AWS service-linked roles) will not originate findings; "
             f"{len(result.suppressed_grants())} grant(s) set aside")
    return result
