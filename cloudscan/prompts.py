"""Evidence digesting plus the DeepSeek prompt/analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config, priority
from .deepseek import DeepSeekClient, DeepSeekError, DeepSeekTruncation
# Everything the prompt *defines* lives in the taxonomy module, so the prompt and
# the verifier cannot disagree. ``PRIVESC_ACTIONS`` stays re-exported here for
# callers that still reach for it.
from .issues import (CATEGORY_PRIVESC, LATERAL_MOVEMENT, PRIVESC_ACTIONS,
                     PRIVILEGE_ESCALATION, SEVERITY_SCALE,
                     is_escalation as _escalation_action,
                     report_categories, taxonomy_text)
from .scope import Scope
from .util import arn_account, debug, info, truncate, warn

# The schema enums are the taxonomy's own vocabulary, rendered — never a
# hand-written list. A severity the taxonomy does not grade to, or a category it
# does not define, cannot appear in the JSON the model is asked for.
_SEVERITY_CHOICES = "|".join(SEVERITY_SCALE)
_CATEGORY_CHOICES = "|".join(report_categories())


# Substrings that mark a row as privileged regardless of its module.
_ADMIN_MARKERS = ("administratoraccess", "administrator", "full access", "fullaccess",
                  "poweruser", "power user", "iam:*", "sts:*")
# Number of characters held back from the evidence budget so the coverage footer
# (dropped / truncated markers) is never itself the thing that gets dropped.
_FOOTER_RESERVE = 800

# Upper bounds on what one LLM stage may return. The tool's failure mode is
# volume, not omission: an uncapped "list every path" is the wall of noise this
# tool exists to replace. Both are stated in the instruction *and* enforced on the
# reply (:func:`_worst_first`), so a model that ignores the instruction cannot
# inflate the report. The verified grants merged in afterwards are deliberately
# not subject to the cap — those are backed by the IAM evaluation engine, and
# dropping a confirmed grant to hit a number would hide the one thing that is real.
_MAX_ESCALATION_PATHS = 20
_MAX_FINDINGS = 40

# Names each reasoning block in the live stdout display and the saved copy.
_STAGE_ESCALATION = "stage 1 — escalation paths"
_STAGE_FINDINGS = "stage 2 — findings review"

# Applies to both stages, and mirrors :func:`cloudscan.verify.suppressed_principals`
# — the verifier already refuses to originate a finding from an administrative or
# service-linked principal, so telling the model the same thing stops it generating
# candidates that are discarded downstream.
_BASELINE_NOTE = (
    "Administrative principals — anything CloudFox rates IsAdminRole? = YES, or a PMapper node marked "
    "admin — and AWS service-linked roles are the account's baseline: they can already do everything, "
    "or are not assumable by a person, so a grant they hold is not a finding and is not the start of an "
    "escalation path. The digest lists them for context only.\n\n"
)


def _signal_score(module: str, row: Dict[str, str], account: str,
                  management_account: Optional[str]) -> int:
    """Heuristic 0..N "how much does an analyst care about this row" score.

    Higher is more interesting. Used to *rank* rows before a per-table cap is
    applied, so truncation keeps signal instead of file order. The scoring is
    deliberately generous (lots of small contributions) — ties fall back to the
    original file order because ``sorted`` is stable.
    """
    score = 0
    values = [str(v).strip() for v in row.values() if v is not None]
    if not values:
        return 0
    lowered = " ".join(v.lower() for v in values)

    # Wildcards and admin-looking grants are the strongest single signals.
    if any(v == "*" for v in values):
        score += 60
    if any(marker in lowered for marker in _ADMIN_MARKERS):
        score += 40
    if any(_escalation_action(v) for v in values):
        score += 30

    # Trust-relationship rows: rank by who is trusted.
    if module.startswith("role-trusts") or module == "resource-trusts":
        principal = (row.get("Principal") or row.get("principal")
                     or row.get("Principal ARN") or row.get("Trusted Principal") or "").strip()
        plower = principal.lower()
        if principal == "*" or plower.endswith("::*") or "::*:" in plower or plower == "arn:aws:iam::*:root":
            score += 120  # trust of everyone
        trusted_account = arn_account(principal) or (principal if principal.isdigit() else None)
        if trusted_account and trusted_account != account:
            # An external account is more interesting than the org's own mgmt account.
            score += 80 if trusted_account == management_account else 110
        if "federated" in module:
            score += 25  # federated trusts often accept a wider identity space
        if "without-external-id" in module:
            score += 40

    # Long-lived active keys are a credential-exposure signal.
    if module == "access-keys":
        status = " ".join(str(row.get(k, "")) for k in ("Status", "Key Status", "Active")).lower()
        if "active" in status:
            score += 20

    return score


def _ranked_rows(rows: Sequence[Dict[str, str]], module: str, account: str,
                 management_account: Optional[str]) -> List[Dict[str, str]]:
    """Return *rows* ordered by descending signal score (stable on ties)."""
    return sorted(rows, key=lambda r: _signal_score(module, r, account, management_account),
                  reverse=True)


# ---------------------------------------------------------------------------
# Digest building
# ---------------------------------------------------------------------------


def _block(title: str, body: str, budget: int) -> Optional[str]:
    """Render one ``###`` section, or ``None`` when it cannot fit at all.

    Returning ``None`` still drops the body, but the caller records the drop and
    surfaces it in the coverage footer — a section is never lost *silently*.
    """
    if not body or budget <= 0:
        return None
    rendered = f"\n### {title}\n\n{body.strip()}\n"
    if len(rendered) > budget:
        rendered = truncate(rendered, budget)
    return rendered


def _csv_block(rows: Sequence[Dict[str, str]], *, limit: int,
               columns: Optional[Sequence[str]] = None, ranked: bool = False) -> str:
    """Render rows as a markdown table, capped at *limit* with an explicit note."""
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    for row in rows[:limit]:
        lines.append(" | ".join(str(row.get(c, "")) for c in columns))
    omitted = len(rows) - limit
    if omitted > 0:
        order = " (highest-signal rows first)" if ranked else ""
        lines.append(f"... [truncated: {limit} of {len(rows)} rows shown; {omitted} omitted{order}]")
    elif ranked:
        lines.append(f"... [all {len(rows)} rows shown, highest-signal first]")
    return "\n".join(lines)


def _collected_index(evidence) -> str:
    """A one-line-per-artifact inventory of everything that was collected.

    This is emitted unconditionally (it never competes for the evidence budget),
    so even if every detailed section is dropped the model can still see *what
    was collected* — the detail is what it loses, not the knowledge that the
    artifact exists. Sections whose detail is dropped are named in the coverage
    footer, so a dropped section is an unknown rather than an absence. The index
    itself is not a finding source: collection coverage is not a reported class.
    """
    lines: List[str] = []
    for module in sorted(evidence.csv_data):
        lines.append(f"- {module}.csv — {len(evidence.csv_data[module])} rows")
    for module in sorted(evidence.tables):
        lines.append(f"- {module} (table) — {len(evidence.tables[module])} chars")
    for module in sorted(evidence.json_data):
        lines.append(f"- {module}.json — present")
    for module in sorted(evidence.loot):
        lines.append(f"- {module} (loot) — {len(evidence.loot[module])} chars")
    graph = evidence.pmapper_graph or {}
    if graph.get("nodes"):
        lines.append(f"- pmapper graph — {len(graph.get('nodes') or [])} nodes, "
                     f"{len(graph.get('edges') or [])} edges")
    if evidence.pmapper_analysis:
        lines.append(f"- pmapper analysis — {len(evidence.pmapper_analysis.get('findings') or [])} native findings")
    if not lines:
        return ""
    return "\n".join(lines)


def _pmapper_edge_block(graph: Dict[str, Any], limit: int = 120) -> str:
    edges = graph.get("edges") or []
    if not edges:
        return ""
    lines = []
    for edge in edges[:limit]:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source") or edge.get("Source") or "?"
        dst = edge.get("destination") or edge.get("Destination") or "?"
        reason = edge.get("short_reason") or edge.get("shortReason") or edge.get("reason") or ""
        lines.append(f"- {src} -> {dst} :: {truncate(str(reason), 200)}")
    if len(edges) > limit:
        lines.append(f"... [truncated: {limit} of {len(edges)} edges shown; {len(edges) - limit} omitted]")
    return "\n".join(lines)


def _node_summary(graph: Dict[str, Any], limit: int = 60) -> str:
    nodes = graph.get("nodes") or []
    if not nodes:
        return ""
    admins = [n for n in nodes if isinstance(n, dict) and (n.get("is_admin") or n.get("isAdmin"))]
    lines = [f"- total nodes: {len(nodes)}",
             f"- administrative principals (baseline — context, never a finding): {len(admins)}"]
    for node in admins[:limit]:
        arn = node.get("arn") or node.get("Arn") or "?"
        lines.append(f"  - ADMIN {arn}")
    return "\n".join(lines)


def _permissions_summary(rows: Sequence[Dict[str, str]], limit: int = 200,
                         account: str = "", management_account: Optional[str] = None) -> str:
    interesting = [
        r for r in rows
        if _escalation_action(r.get("Action", "")) and (r.get("Effect", "Allow") == "Allow")
    ]
    header = (f"permissions.csv: {len(rows)} total statements; {len(interesting)} grant an "
              "escalation primitive. Statements that only read data — S3 objects, secrets, KMS — are "
              "omitted here: they grade as Data Access, and the simulator confirms them.")
    if not interesting:
        return header
    columns = ["Type", "Name", "Policy Name", "Effect", "Action", "Resource", "Condition"]
    ranked = _ranked_rows(interesting, "permissions", account, management_account)
    return header + "\n\n" + _csv_block(ranked, limit=limit, columns=columns, ranked=True)


def build_evidence_digest(evidence, scope: Scope, *, budget: Optional[int] = None) -> str:
    """Render a budgeted textual digest of all collected evidence.

    Two guarantees hold regardless of budget:

    * a **collected-artifacts index** is always emitted, so the model knows what
      exists even when a detailed section could not fit; and
    * every section that is dropped or truncated is named in an ``### Analysis
      coverage`` footer — nothing disappears silently.

    Rows are signal-ranked (see :func:`_signal_score`) before the per-table cap
    is applied, so what survives truncation is the interesting part.
    """
    budget = budget or config.MAX_EVIDENCE_CHARS
    limit = config.MAX_ROWS_PER_TABLE
    account = evidence.account
    mgmt = scope.management_account

    header = (
        f"# CloudFox + PMapper evidence digest\n\n"
        f"- Target account: {evidence.account}\n"
        f"- Collection profile: {evidence.profile}\n"
        f"- Collection region: {evidence.region}\n"
        f"- Caller identity: {evidence.identity_arn}\n"
        f"- Parent management account: {scope.management_account or 'unknown'}\n"
        f"- Organisation members seen by CloudFox: {', '.join(scope.org_members) or 'none'}\n"
    )
    sections: List[str] = [header]
    used = len(header)

    # The index is unconditional: it is small, and it is what lets the model
    # reason about evidence it cannot see rendered in full below.
    index = _collected_index(evidence)
    if index:
        index_block = ("\n### Collected artifacts (index)\n\n"
                       "Everything the tools returned; detail follows below where it fits.\n\n"
                       f"{index}\n")
        sections.append(index_block)
        used += len(index_block)

    dropped: List[str] = []
    truncated: List[str] = []

    def add(title: str, body: str, *, rows: Optional[int] = None) -> None:
        nonlocal used
        text = (body or "").strip()
        if not text:
            if rows:
                dropped.append(f"[dropped: {title}, {rows} rows]")
            return
        remaining = budget - used - _FOOTER_RESERVE
        block = _block(title, body, remaining)
        if block is None:
            count = rows if rows is not None else "?"
            dropped.append(f"[dropped: {title}, {count} rows]")
            return
        sections.append(block)
        used += len(block)
        if len(block) < len(f"\n### {title}\n\n{text}\n"):
            truncated.append(f"[truncated: {title}]")

    def add_csv(title: str, module: str, *, columns=None, rows=None, limit_override=None) -> None:
        data = rows if rows is not None else evidence.csv(module)
        if not data:
            return
        ranked = _ranked_rows(data, module, account, mgmt)
        body = _csv_block(ranked, limit=limit_override or limit, columns=columns, ranked=True)
        add(title, body, rows=len(data))

    # -- PMapper --------------------------------------------------------------
    if evidence.pmapper_analysis:
        native = evidence.pmapper_analysis.get("findings")
        add("PMapper analysis (native findings)",
            "```json\n" + truncate(_json(native), config.MAX_FIELD_CHARS) + "\n```",
            rows=len(native or []))
    if evidence.pmapper_graph:
        graph = evidence.pmapper_graph
        add("PMapper graph — administrative principals (baseline; context, not findings)",
            _node_summary(graph), rows=len(graph.get("nodes") or []))
        add("PMapper graph — privesc edges", _pmapper_edge_block(graph),
            rows=len(graph.get("edges") or []))

    # -- CloudFox high-signal ------------------------------------------------
    if evidence.has_csv("workloads-admin"):
        add_csv("CloudFox workloads-admin (workloads with admin or path to admin)", "workloads-admin")
    if evidence.has_csv("workloads"):
        add_csv("CloudFox workloads", "workloads")
    if evidence.has_csv("iam-simulator"):
        rows = evidence.csv("iam-simulator")
        admins = [r for r in rows if "administrator" in (r.get("Query", "") or "").lower()]
        if admins:
            add_csv("CloudFox iam-simulator — principals that appear to be administrators "
                    "(baseline; context, not findings)", "iam-simulator", rows=admins)
        else:
            add("CloudFox iam-simulator",
                f"{len(rows)} rows; the simulator flagged none as administrator.", rows=len(rows))
    for module in ("role-trusts-principals", "role-trusts-federated", "role-trusts-services",
                   "role-trusts-principals-root-trusts-without-external-id", "resource-trusts"):
        if evidence.has_csv(module):
            add_csv(f"CloudFox {module}", module)
    if evidence.has_csv("permissions"):
        add("CloudFox permissions (statements granting an escalation primitive)",
            _permissions_summary(evidence.csv("permissions"), account=account, management_account=mgmt),
            rows=len(evidence.csv("permissions")))
    if evidence.has_csv("access-keys"):
        add_csv("CloudFox access-keys", "access-keys")
    if evidence.csv("org"):
        add_csv("CloudFox org membership", "org", limit_override=20)

    # -- CloudFox general inventory & resources ------------------------------
    # Deliberately absent: ``tags`` (thousands of generic tag rows, no exploitation
    # step consumes them) and ``route53`` (DNS records plus a loot file of A records
    # to *scan*; CloudFox detects no takeover, and no class rests on it).
    for module in ("principals", "secrets", "buckets", "databases", "endpoints",
                   "env-vars-interesting", "ecs-tasks", "lambda", "instances",
                   "network-ports", "ram", "cloudformation", "sns", "sqs", "api-gw",
                   "eks", "ecr", "filesystems", "codebuild"):
        if evidence.has_csv(module):
            add_csv(f"CloudFox {module}", module)
    if evidence.tables.get("inventory"):
        inventory = evidence.tables["inventory"]
        add("CloudFox inventory summary", inventory,
            rows=len([line for line in inventory.splitlines() if line.strip()]))

    # -- Coverage footer (always emitted) ------------------------------------
    coverage: List[str] = []
    if truncated:
        coverage.append("Sections truncated mid-render (highest-signal rows kept first):")
        coverage.extend(f"  {item}" for item in truncated)
    if dropped:
        coverage.append("Sections dropped for budget — this data WAS collected but is NOT shown to you; "
                        "treat it as an unknown, NOT as absent:")
        coverage.extend(f"  {item}" for item in dropped)
    if coverage:
        sections.append("\n### Analysis coverage\n\n" + "\n".join(coverage) + "\n")

    digest = "\n".join(sections)
    debug(f"evidence digest: {len(digest)} chars (budget {budget}, "
          f"{len(truncated)} truncated, {len(dropped)} dropped)")
    return digest


def _json(value: Any) -> str:
    import json

    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The invariants that hold for both stages: what this is, what it may not do, and
# the shape of the reply. There is deliberately no persona. Each call is a
# single-shot extraction, not a turn in an agent loop — the task lives in the
# stage's taxonomy, and "be a thorough expert" only primes the over-reporting
# this tool exists to avoid. Uncertainty is carried by the schema's `confidence`
# field, so it is not something the model is asked to announce.
SYSTEM_PROMPT = (
    "This is an authorised, read-only assessment of an AWS account: propose nothing that "
    "changes it. Answer only from the evidence provided — never invent a resource, action "
    "or path that is not in it. Reply with a single valid JSON object and nothing else."
)

_ESCALATION_SCHEMA = f"""{{
  "summary": "<3-6 sentence executive summary of the account's escalation posture>",
  "escalation_paths": [
    {{
      "id": "ESC-01",
      "title": "<short title>",
      "severity": "{_SEVERITY_CHOICES}",
      "from_principal": "<starting principal ARN or 'external attacker'>",
      "to_principal": "<the principal or resource the path ends at, or the impact it reaches>",
      "hops": ["<step 1>", "<step 2>"],
      "primitives": ["iam:PassRole", "lambda:CreateFunction"],
      "evidence": "<concrete evidence from the digest: ARNs, policy actions, trust statements>",
      "preconditions": "<what an attacker needs first>",
      "confidence": "high|medium|low",
      "recommendation": "<how to break the path>"
    }}
  ]
}}"""

_FINDING_SCHEMA = f"""{{
  "findings": [
    {{
      "id": "AWSSCLOUDSCAN-01",
      "title": "<short title>",
      "severity": "{_SEVERITY_CHOICES}",
      "category": "{_CATEGORY_CHOICES}",
      "scope": "account|management-account|organization|external-account|unknown",
      "affected_principals": ["<ARN>"],
      "affected_resources": ["<ARN or name>"],
      "evidence": "<exact evidence from the digest>",
      "impact": "<what an attacker achieves>",
      "recommendation": "<remediation>",
      "confidence": "high|medium|low"
    }}
  ]
}}"""


def build_escalation_messages(digest: str, evidence, scope: Scope) -> List[Dict[str, str]]:
    instruction = (
        f"Analyse the following CloudFox and PMapper evidence for AWS account {evidence.account}.\n\n"
        "Your task is to identify the privilege-escalation and lateral-movement paths available "
        "inside this account, by the definitions below.\n\n"
        f"{taxonomy_text([PRIVILEGE_ESCALATION, LATERAL_MOVEMENT])}\n"
        f"{_BASELINE_NOTE}"
        "Ground every claim in the evidence. If PMapper already reports a privesc path, include it and "
        "cite the edge.\n\n"
        f"Report at most {_MAX_ESCALATION_PATHS} paths, worst first — order them by how few "
        "preconditions they need, and stop there rather than padding the list.\n\n"
        f"Respond with a single JSON object exactly matching this schema:\n{_ESCALATION_SCHEMA}\n\n"
        f"--- EVIDENCE START ---\n{digest}\n--- EVIDENCE END ---"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def build_findings_messages(digest: str, escalations: Dict[str, Any], evidence, scope: Scope) -> List[Dict[str, str]]:
    escalation_digest = _json(escalations.get("escalation_paths", []))
    instruction = (
        f"Review the CloudFox and PMapper evidence for AWS account {evidence.account} and produce the "
        "security findings, by the definitions below.\n\n"
        "IMPORTANT SCOPE RULE: report only issues that concern resources or principals owned by "
        f"account {evidence.account} itself. Issues that belong to the parent AWS Organisation "
        f"(management account {scope.management_account or 'unknown'}, organisation-level SCPs/trails, or "
        "other member accounts) must be tagged with the matching scope value and will be excluded from the "
        "main report. A role in this account that trusts an external principal IS an in-account finding "
        "(scope \"account\"); a misconfiguration in another account is not.\n\n"
        "For each finding give concrete evidence (ARNs, policy actions, trust statements) drawn from the "
        "digest. Do not invent resources.\n\n"
        f"{_BASELINE_NOTE}"
        f"{taxonomy_text()}\n"
        f"Escalation and lateral-movement paths already reported and final — do not restate them as "
        f"findings, and do not report a finding whose only content is one of them:\n"
        f"{truncate(escalation_digest, 8000)}\n\n"
        f"Report at most {_MAX_FINDINGS} findings, worst first. Precision beats recall: if the same "
        "weakness appears on several resources, report it once and list the resources; omit anything "
        "you cannot evidence from the digest.\n\n"
        f"Respond with a single JSON object exactly matching this schema:\n{_FINDING_SCHEMA}\n\n"
        f"--- EVIDENCE START ---\n{digest}\n--- EVIDENCE END ---"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


# ---------------------------------------------------------------------------
# Analysis orchestration
# ---------------------------------------------------------------------------


def _worst_first(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Cap a stage's reply at *limit*, keeping the items that matter most.

    Ranking is the deterministic priority table (:mod:`cloudscan.priority`), not
    a bare severity sort: two High findings are not equally interesting, and the
    table says which is which. The cap is enforced here as well as stated in the
    instruction, so a reply that ignores the instruction cannot inflate the
    report. ``sorted`` is stable, so items a rank and severity tie keep the order
    the model gave them — its judgement inside a band survives the cut.
    """
    if len(items) <= limit:
        return items
    ranked = priority.sort_findings(items)
    info(f"trimmed {len(items)} items to the {limit} highest priority")
    return ranked[:limit]


@dataclass
class Analysis:
    summary: str = ""
    escalation_paths: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    escalations_raw: Dict[str, Any] = field(default_factory=dict)
    findings_raw: Dict[str, Any] = field(default_factory=dict)
    #: The model's chain of thought for each stage, when thinking is enabled.
    #: Kept so the run can show it live and save it beside the report.
    escalation_reasoning: str = ""
    findings_reasoning: str = ""
    warnings: List[str] = field(default_factory=list)
    #: Names of the stages whose reply was cut off at its token budget. Kept
    #: apart from ``warnings`` so the caller can tell a stage that needed more
    #: room from one that could not reach the API at all — the two need opposite
    #: fixes, and the run's closing advice must not blame the network for a
    #: budget problem.
    truncated_stages: List[str] = field(default_factory=list)
    used_ai: bool = False


def pmapper_fallback_findings(evidence) -> List[Dict[str, Any]]:
    """Deterministic findings straight from PMapper, used when the LLM is unavailable."""
    results: List[Dict[str, Any]] = []
    analysis = evidence.pmapper_analysis or {}
    for index, finding in enumerate(analysis.get("findings", []) or [], start=1):
        if not isinstance(finding, dict):
            continue
        results.append({
            "id": f"PMAPPER-{index:02d}",
            "title": finding.get("title", "PMapper finding"),
            "severity": finding.get("severity", "Unknown"),
            "category": CATEGORY_PRIVESC,
            "scope": "account",
            "affected_principals": [],
            "affected_resources": [],
            "evidence": truncate(finding.get("description", ""), config.MAX_FIELD_CHARS),
            "impact": finding.get("impact", ""),
            "recommendation": finding.get("recommendation", ""),
            "confidence": "high",
        })
    return results


def _last_reasoning(client: DeepSeekClient) -> str:
    """The chain of thought behind the client's most recent completion.

    :meth:`DeepSeekClient.chat_json` returns only the parsed JSON, so the
    thinking is read back off the client. A double that does not track it
    yields an empty string.
    """
    return getattr(getattr(client, "last_result", None), "reasoning", "") or ""


def _forget_last_result(client: DeepSeekClient) -> None:
    """Clear the cached result so a stage never inherits the previous one's thinking."""
    if hasattr(client, "last_result"):
        client.last_result = None


def _fallback_summary(analysis: Analysis) -> str:
    """An honest summary when the model produced none.

    It must not claim more is missing than really is. The findings themselves are
    deterministic (the simulator, the resource-to-role join, PMapper), they are
    ordered by the priority table, and they are all present — what is absent is
    the model's *narrative*: the prose, and the escalation walk-through.
    """
    if analysis.truncated_stages:
        where = ("both stages" if len(analysis.truncated_stages) > 1
                 else analysis.truncated_stages[0])
        reason = f"the DeepSeek reply for {where} was cut off at its token budget"
    else:
        reason = "no DeepSeek analysis could be produced"
    return ("DeepSeek review was unavailable, so this report carries no AI-written narrative and "
            f"no AI escalation walk-through ({reason}). The findings themselves are deterministic "
            "and complete: verified grants come from iam:SimulatePrincipalPolicy, escalation "
            "edges from the resource-to-role join, and PMapper findings straight from PMapper. "
            "They are ordered worst first by the built-in priority table, not by the model.")


def analyze(client: DeepSeekClient, evidence, scope: Scope, *, digest_budget: Optional[int] = None,
            stream_reasoning: bool = False, max_tokens: Optional[int] = None) -> Analysis:
    """Run the two-stage DeepSeek analysis, degrading gracefully on failure.

    With ``stream_reasoning=True`` both stages request SSE and echo their
    thinking to stdout as it is generated; the text is kept on the returned
    :class:`Analysis` as well, so the caller can save it beside the report.

    ``max_tokens`` overrides the per-stage completion budget for both stages;
    without it each stage uses its own default from :mod:`cloudscan.config`. A
    stage cut off at its budget is recorded in ``truncated_stages`` rather than
    lumped in with a connectivity failure — the fixes are different.
    """
    analysis = Analysis()
    digest = build_evidence_digest(evidence, scope, budget=digest_budget)
    stage1_tokens, stage2_tokens = config.deepseek_stage_budgets(max_tokens)

    # Stage 1: escalation paths.
    try:
        _forget_last_result(client)
        raw = client.chat_json(build_escalation_messages(digest, evidence, scope),
                               max_tokens=stage1_tokens,
                               stream=stream_reasoning, label=_STAGE_ESCALATION)
        analysis.escalations_raw = raw
        analysis.summary = str(raw.get("summary", "")).strip()
        paths = raw.get("escalation_paths")
        if isinstance(paths, list):
            analysis.escalation_paths = _worst_first(
                [p for p in paths if isinstance(p, dict)], _MAX_ESCALATION_PATHS)
        analysis.used_ai = True
    except DeepSeekTruncation as exc:
        analysis.truncated_stages.append(_STAGE_ESCALATION)
        analysis.warnings.append(f"escalation analysis was truncated: {exc}")
        warn(str(exc))
    except DeepSeekError as exc:
        analysis.warnings.append(f"escalation analysis failed: {exc}")
        warn(str(exc))
    # Whatever thinking arrived — even from a stage that then failed — is kept.
    analysis.escalation_reasoning = _last_reasoning(client)

    # Stage 2: findings review.
    try:
        _forget_last_result(client)
        raw = client.chat_json(build_findings_messages(digest, analysis.escalations_raw, evidence, scope),
                               max_tokens=stage2_tokens, stream=stream_reasoning,
                               label=_STAGE_FINDINGS)
        analysis.findings_raw = raw
        findings = raw.get("findings")
        if isinstance(findings, list):
            analysis.findings = _worst_first([f for f in findings if isinstance(f, dict)],
                                             _MAX_FINDINGS)
        analysis.used_ai = True
    except DeepSeekTruncation as exc:
        analysis.truncated_stages.append(_STAGE_FINDINGS)
        analysis.warnings.append(f"findings review was truncated: {exc}")
        warn(str(exc))
    except DeepSeekError as exc:
        analysis.warnings.append(f"findings review failed: {exc}")
        warn(str(exc))
    analysis.findings_reasoning = _last_reasoning(client)

    if not analysis.findings:
        fallback = pmapper_fallback_findings(evidence)
        if fallback:
            info(f"using {len(fallback)} deterministic PMapper finding(s) as fallback")
            analysis.findings = fallback
    # The summary is filled whenever it is missing, not only when there are no
    # findings. A run whose stage 1 failed but whose verified findings merged in
    # downstream still needs the report to say what is and is not AI-authored —
    # and must not claim to hold nothing but raw evidence when it holds 115
    # simulator-confirmed findings.
    if not analysis.summary:
        analysis.summary = _fallback_summary(analysis)
    return analysis
