"""Account scope logic.

The report must cover issues *inside* the target account and deliberately exclude
anything that belongs to the parent AWS Organization (the management account, org
members other than ours, org-level constructs). This module classifies findings
deterministically using the account ids that appear in them, with the LLM's own
``scope`` hint used only as a tie-breaker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .util import find_account_ids

# Wider than util's account regex on purpose: a model that hallucinates an
# account id usually loses digits, so the truncated form never matches \d{12}.
_ID_RE = re.compile(r"\b(\d{9,12})\b")

ACCOUNT = "account"
MANAGEMENT_ACCOUNT = "management-account"
ORGANIZATION = "organization"
EXTERNAL_ACCOUNT = "external-account"
UNKNOWN = "unknown"

CATEGORY_LABELS = {
    ACCOUNT: "In-account",
    MANAGEMENT_ACCOUNT: "Parent organisation (management account)",
    ORGANIZATION: "Parent organisation (org-level)",
    EXTERNAL_ACCOUNT: "External / other account",
    UNKNOWN: "Undetermined scope",
}

# Phrases that indicate a finding is really about the parent organization.
_ORG_KEYWORDS = (
    "service control policy",
    "scp ",
    "organizations:",
    "organization trail",
    "org trail",
    "management account",
    "master account",
    "organizationaccountaccessrole",
    "organizational unit",
    "identity center",
    "aws organizations",
)

_TARGET_HINTS = (
    "this account",
    "in-account",
    "target account",
)


@dataclass
class ScopeDecision:
    in_scope: bool
    category: str
    reason: str
    accounts_seen: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)


@dataclass
class Scope:
    """The account boundary for a run."""

    target_account: str
    management_account: Optional[str] = None
    org_members: List[str] = field(default_factory=list)
    extra_excluded: List[str] = field(default_factory=list)

    @classmethod
    def from_evidence(cls, evidence, extra_excluded: Optional[Sequence[str]] = None) -> "Scope":
        return cls(
            target_account=evidence.account or "",
            management_account=evidence.management_account(),
            org_members=evidence.member_accounts(),
            extra_excluded=list(extra_excluded or []),
        )

    # -- account-level helpers ------------------------------------------------

    def is_target(self, account: str) -> bool:
        return bool(account) and account == self.target_account

    def is_management(self, account: str) -> bool:
        return bool(account) and account == self.management_account

    def is_excluded(self, account: str) -> bool:
        return bool(account) and account in self.extra_excluded

    def is_org_member(self, account: str) -> bool:
        return bool(account) and account in self.org_members

    def category_for_account(self, account: str) -> str:
        if self.is_target(account):
            return ACCOUNT
        if self.is_management(account):
            return MANAGEMENT_ACCOUNT
        if self.is_excluded(account):
            return EXTERNAL_ACCOUNT
        if self.is_org_member(account):
            return ORGANIZATION
        return EXTERNAL_ACCOUNT

    # -- text-level classification -------------------------------------------

    def classify_text(self, text: str, llm_hint: Optional[str] = None) -> ScopeDecision:
        """Classify free text (a finding body) into a scope category."""
        accounts = find_account_ids(text or "")
        target_present = self.is_target(self.target_account) and self.target_account in accounts

        if target_present:
            return ScopeDecision(
                True, ACCOUNT,
                f"references the target account {self.target_account}",
                accounts,
            )

        if accounts:
            categories = {self.category_for_account(a) for a in accounts}
            if categories == {MANAGEMENT_ACCOUNT}:
                return ScopeDecision(False, MANAGEMENT_ACCOUNT,
                                     f"only references the management account {self.management_account}", accounts)
            if categories <= {ORGANIZATION, MANAGEMENT_ACCOUNT}:
                return ScopeDecision(False, ORGANIZATION,
                                     "references parent-organisation accounts only", accounts)
            if categories == {EXTERNAL_ACCOUNT}:
                return ScopeDecision(False, EXTERNAL_ACCOUNT,
                                     "references external account(s) only", accounts)

        lowered = (text or "").lower()
        mentions_org = any(k in lowered for k in _ORG_KEYWORDS)
        if mentions_org and not target_present:
            return ScopeDecision(False, ORGANIZATION,
                                 "describes an organisation-level control, not an in-account resource", accounts)

        # No account ids were found: fall back to the LLM hint, then default in-scope.
        if llm_hint in (MANAGEMENT_ACCOUNT, ORGANIZATION, EXTERNAL_ACCOUNT):
            return ScopeDecision(False, llm_hint, "classified by model as out-of-scope", accounts)
        return ScopeDecision(True, ACCOUNT,
                             "no foreign account references; treated as in-account", accounts)


def _finding_text(finding: Dict[str, Any]) -> str:
    """Flatten the textual fields of a finding dict for classification."""
    parts: List[str] = []
    for key in ("title", "description", "evidence", "impact", "affected_principals",
                "affected_resources", "from_principal", "to_principal", "primitives",
                "recommendation", "reason"):
        value = finding.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            parts.append(" ".join(str(v) for v in value))
        else:
            parts.append(str(value))
    return "\n".join(parts)


def split_findings(
    scope: Scope, findings: Iterable[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split findings into (in_scope, out_of_scope), annotating each with its decision."""
    in_scope: List[Dict[str, Any]] = []
    out_of_scope: List[Dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        hint = str(finding.get("scope") or finding.get("in_scope_hint") or "").strip().lower()
        decision = scope.classify_text(_finding_text(finding), hint or None)
        annotated = dict(finding)
        annotated["_scope"] = {
            "in_scope": decision.in_scope,
            "category": decision.category,
            "label": decision.label,
            "reason": decision.reason,
        }
        (in_scope if decision.in_scope else out_of_scope).append(annotated)
    return in_scope, out_of_scope


def summarize_scope(scope: Scope, findings_in: int, findings_out: int) -> str:
    mgmt = scope.management_account or "n/a"
    return (
        f"Target account **{scope.target_account}**; "
        f"parent management account **{mgmt}**; "
        f"{findings_in} in-scope finding(s), {findings_out} excluded as parent-organisation/other-account."
    )


# ---------------------------------------------------------------------------
# Identifier hygiene
# ---------------------------------------------------------------------------

@dataclass
class IdRepairs:
    """Outcome of auditing the account ids a model wrote into its prose."""

    text: str
    fixed: List[Tuple[str, str]] = field(default_factory=list)      # (wrong, corrected)
    suspicious: List[str] = field(default_factory=list)             # unknown, near-miss ids


def _levenshtein(a: str, b: str, limit: int) -> Optional[int]:
    """Edit distance between two short strings, abandoned once it exceeds *limit*."""
    if abs(len(a) - len(b)) > limit:
        return None
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        best = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > limit:
            return None
        previous = current
    return previous[-1] if previous[-1] <= limit else None


def repair_account_ids(text: str, known: Sequence[str], *, max_distance: int = 3) -> IdRepairs:
    """Correct account ids that were mistyped or truncated, and flag near misses.

    Models write 12-digit account ids from memory, and nothing downstream checked
    them. On this engagement the executive summary asserted the parent management
    account was ``676830375`` — ``676830062375`` with two digits lost — which
    reads as authoritative because it is stated as fact.

    A known id is left untouched. An unknown one within *max_distance* edits of a
    known id is snapped to it and recorded as a correction. An id that is merely
    *close* (up to twice that distance) is reported without being changed, which
    catches a plausible fabrication without false-alarming on the genuinely
    foreign accounts that CloudFox's trust data legitimately names.
    """
    known = [k for k in dict.fromkeys(known or []) if k]
    if not text or not known:
        return IdRepairs(text)
    result = IdRepairs(text)

    def replace(match: "re.Match[str]") -> str:
        found = match.group(1)
        if found.startswith("0000"):
            return found
        if len(found) == 12 and found in known:
            return found
        # A truncated id is not 12 digits, so this scan is deliberately wider
        # than util.find_account_ids: 676830375 never reaches that regex.
        best: Optional[str] = None
        best_distance: Optional[int] = None
        for candidate in known:
            distance = _levenshtein(found, candidate, max_distance)
            if distance is not None and (best_distance is None or distance < best_distance):
                best, best_distance = candidate, distance
        if best is not None:
            result.fixed.append((found, best))
            return best
        if len(found) == 12 and any(_levenshtein(found, candidate, max_distance * 2) is not None
                                    for candidate in known):
            if found not in result.suspicious:
                result.suspicious.append(found)
        return found

    result.text = _ID_RE.sub(replace, text)
    return result


def _walk_strings(node: Any, fn) -> Any:
    """Apply *fn* to every string inside a JSON-ish structure."""
    if isinstance(node, str):
        return fn(node)
    if isinstance(node, list):
        return [_walk_strings(item, fn) for item in node]
    if isinstance(node, dict):
        return {key: _walk_strings(value, fn) for key, value in node.items()}
    return node


def repair_analysis(analysis: Any, scope: Scope) -> IdRepairs:
    """Audit the model's prose for account ids that do not exist, in place.

    Only what the model authored is rewritten: ``summary``, ``escalation_paths``
    and ``findings``. ``escalations_raw`` and ``findings_raw`` keep its verbatim
    reply as the audit record of what was actually said.
    """
    known = [scope.target_account, scope.management_account,
             *scope.org_members, *scope.extra_excluded]
    combined = IdRepairs("")

    def fix(text: str) -> str:
        outcome = repair_account_ids(text, known)
        combined.fixed.extend(outcome.fixed)
        for account in outcome.suspicious:
            if account not in combined.suspicious:
                combined.suspicious.append(account)
        return outcome.text

    if analysis is None:
        return combined
    analysis.summary = fix(analysis.summary or "")
    analysis.escalation_paths = _walk_strings(getattr(analysis, "escalation_paths", []) or [], fix)
    analysis.findings = _walk_strings(getattr(analysis, "findings", []) or [], fix)
    return combined


# ---------------------------------------------------------------------------
# Escalation-path provenance
# ---------------------------------------------------------------------------

def label_escalation_sources(scope: Scope, paths: Optional[Sequence[Dict[str, Any]]]) -> int:
    """Annotate escalation paths whose starting principal lies outside the account.

    A role in this account whose trust policy names the management account root
    *is* an in-account finding — the risky object lives here — but the path
    starts in an account we did not assess. Unlabelled, "the management account
    escalates into this account" contradicts the report's own scope rule; the
    label is what makes the two statements compatible.

    Returns how many paths were found to originate externally.
    """
    external = 0
    for path in paths or []:
        if not isinstance(path, dict):
            continue
        accounts = find_account_ids(str(path.get("from_principal") or ""))
        foreign = [a for a in accounts if not scope.is_target(a)]
        if foreign:
            account = foreign[0]
            path["_source_scope"] = {
                "external": True,
                "account": account,
                "label": CATEGORY_LABELS.get(scope.category_for_account(account),
                                             CATEGORY_LABELS[EXTERNAL_ACCOUNT]),
            }
            external += 1
        else:
            path["_source_scope"] = {
                "external": False,
                "account": scope.target_account,
                "label": CATEGORY_LABELS[ACCOUNT],
            }
    return external
