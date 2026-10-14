"""Render the final Markdown report."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, priority
from .scope import Scope
from .util import truncate

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _sev_rank(value: str) -> int:
    return _SEVERITY_ORDER.get((value or "").strip().lower(), 5)


def _sort_findings(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order items worst first, by the deterministic priority table.

    See :mod:`cloudscan.priority`: severity alone cannot rank a ``*`` trust above
    a stale public load balancer, so the ordering lives in a rule table rather
    than in an ad-hoc sort here. Severity is the tiebreaker; ties keep their
    incoming order.
    """
    return priority.sort_findings(items)


def _cell(value: Any) -> str:
    """Escape a value for a Markdown table cell."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _bullets(values: Any) -> str:
    if not values:
        return "_none recorded_"
    if isinstance(values, str):
        values = [values]
    return "\n".join(f"- {v}" for v in values)


def _origin_label(path: Dict[str, Any]) -> str:
    """Short origin tag for the escalation-path table."""
    source = path.get("_source_scope") or {}
    if source.get("external"):
        account = source.get("account")
        return f"External ({account})" if account else "External"
    return "In-account"


def _origin_indicator(path: Dict[str, Any]) -> str:
    """Longer origin indicator for a path's detail block."""
    source = path.get("_source_scope") or {}
    if source.get("external"):
        account = source.get("account")
        label = source.get("label") or "starts outside the target account"
        return f"{label}" + (f" (`{account}`)" if account and account not in label else "")
    return source.get("label") or "In-account (starts and stays inside the target account)"


def _section_severity_counts(findings: Sequence[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for finding in findings:
        sev = str(finding.get("severity", "Unknown")).capitalize()
        counts[sev] = counts.get(sev, 0) + 1
    if not counts:
        return "_none_"
    ordered = sorted(counts.items(), key=lambda kv: _sev_rank(kv[0]))
    return ", ".join(f"**{k}**: {v}" for k, v in ordered)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    *,
    identity,
    scope: Scope,
    analysis,
    in_scope_findings: List[Dict[str, Any]],
    out_of_scope_findings: List[Dict[str, Any]],
    evidence,
    run_result,
    generated_at: Optional[dt.datetime] = None,
    include_out_of_scope: bool = False,
    tool_versions: Optional[Dict[str, str]] = None,
    ai_model: Optional[str] = None,
    ai_effort: Optional[str] = None,
    ai_used: bool = False,
    id_repairs: Any = None,
    external_paths: int = 0,
    thinking_files: Optional[List[str]] = None,
) -> str:
    generated_at = generated_at or dt.datetime.now(dt.timezone.utc)
    lines: List[str] = []

    target = scope.target_account or (identity.account if identity else "unknown")
    lines.append(f"# AWSCloudScan Report — AWS account {target}")
    lines.append("")
    lines.append(f"_Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} by awscloudscan v{__version__}._")
    lines.append("")

    # -- metadata -------------------------------------------------------------
    lines.append("## Engagement metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Target account | `{target}` |")
    lines.append(f"| AWS profile | `{getattr(identity, 'profile', evidence.profile)}` |")
    lines.append(f"| Region | `{getattr(identity, 'region', evidence.region)}` |")
    lines.append(f"| Caller identity | `{getattr(identity, 'arn', evidence.identity_arn)}` |")
    lines.append(f"| Parent management account | `{scope.management_account or 'unknown'}` |")
    lines.append(f"| Organisation members seen | {', '.join(f'`{a}`' for a in scope.org_members) or '_none_'} |")
    if tool_versions:
        lines.append(f"| CloudFox | `{tool_versions.get('cloudfox', 'unknown')}` |")
        lines.append(f"| PMapper | `{tool_versions.get('pmapper', 'unknown')}` |")
    lines.append(f"| DeepSeek review | {'completed' if ai_used else 'not performed'} |")
    if ai_model:
        lines.append(f"| DeepSeek model | `{ai_model}` |")
    if ai_used:
        lines.append(f"| DeepSeek thinking effort | `{ai_effort or 'default'}` |")
    if thinking_files:
        saved = ", ".join(f"`{name}`" for name in thinking_files)
        lines.append(f"| DeepSeek reasoning | saved beside this report: {saved} |")
    lines.append("")
    lines.append("**Scope rule.** Findings cover issues *inside the target account only* — "
                 "resources, identities and policies that live here. Organisation-level controls "
                 "(SCPs, organisation trails) and resources owned by other accounts are excluded and "
                 "listed in Appendix A. Where an in-account role's trust policy names a principal "
                 "outside this account (for example the management account's root), the role **is** "
                 "reported, because the risky object lives here, but the path is labelled as starting "
                 "elsewhere and the foreign account is not itself assessed.")
    lines.append("")

    # -- executive summary ----------------------------------------------------
    lines.append("## Executive summary")
    lines.append("")
    summary = (analysis.summary or "").strip() if analysis else ""
    lines.append(summary or "_No AI summary available._")
    lines.append("")
    lines.append(f"- In-scope findings: {len(in_scope_findings)} ({_section_severity_counts(in_scope_findings)})")
    lines.append(f"- Escalation paths identified: {len(analysis.escalation_paths) if analysis else 0}")
    if external_paths:
        lines.append(f"- Escalation paths that start outside this account: {external_paths} "
                     "(the in-account role each one reaches is still reported; the foreign account is not assessed)")
    lines.append(f"- Items excluded as parent-organisation/other-account: {len(out_of_scope_findings)}")
    if id_repairs is not None and getattr(id_repairs, "fixed", None):
        note = ", ".join(f"`{wrong}` → `{right}`" for wrong, right in id_repairs.fixed)
        lines.append(f"- Account identifiers corrected in AI-authored text: {len(id_repairs.fixed)} ({note})")
    if id_repairs is not None and getattr(id_repairs, "suspicious", None):
        note = ", ".join(f"`{a}`" for a in id_repairs.suspicious)
        lines.append(f"- Account identifiers cited but not seen in evidence (left as written): {note}")
    if analysis and analysis.warnings:
        lines.append("")
        lines.append("> Analysis warnings: " + "; ".join(analysis.warnings))
    lines.append("")

    # -- escalation paths -----------------------------------------------------
    lines.append("## Possible privilege-escalation paths")
    lines.append("")
    escalation_paths = _sort_findings(analysis.escalation_paths) if analysis else []
    if not escalation_paths:
        lines.append("_No escalation paths were identified from the available evidence._")
        lines.append("")
    else:
        lines.append("| ID | Severity | Origin | Path | Primitives | Confidence |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for path in escalation_paths:
            frm = path.get("from_principal", "?")
            to = path.get("to_principal", "?")
            lines.append(
                f"| {_cell(path.get('id'))} | {_cell(path.get('severity'))} | "
                f"{_cell(_origin_label(path))} | "
                f"{_cell(frm)} → {_cell(to)} | {_cell(path.get('primitives'))} | {_cell(path.get('confidence'))} |"
            )
        lines.append("")
        for path in escalation_paths:
            lines.append(f"### {path.get('id', 'ESC')} — {path.get('title', 'Escalation path')}")
            lines.append("")
            lines.append(f"**Severity:** {path.get('severity', 'Unknown')}  ")
            lines.append(f"**Confidence:** {path.get('confidence', 'unknown')}  ")
            lines.append(f"**Origin:** {_origin_indicator(path)}  ")
            lines.append(f"**From:** `{path.get('from_principal', '?')}`  ")
            lines.append(f"**To:** `{path.get('to_principal', '?')}`")
            lines.append("")
            _source = path.get("_source_scope") or {}
            if _source.get("external"):
                lines.append("> This path begins at a principal outside the target account. The in-account "
                             "role it reaches is the thing to fix — its trust policy admits the external "
                             "principal. The external principal's own account is not assessed in this report.")
                lines.append("")
            hops = path.get("hops") or []
            if hops:
                lines.append("**Steps:**")
                lines.append("")
                for index, hop in enumerate(hops, start=1):
                    lines.append(f"{index}. {hop}")
                lines.append("")
            if path.get("primitives"):
                lines.append("**Primitives:**")
                lines.append("")
                lines.append(_bullets(path.get("primitives")))
                lines.append("")
            if path.get("preconditions"):
                lines.append(f"**Preconditions:** {path.get('preconditions')}")
                lines.append("")
            if path.get("evidence"):
                lines.append(f"**Evidence:** {path.get('evidence')}")
                lines.append("")
            if path.get("recommendation"):
                lines.append(f"**Recommendation:** {path.get('recommendation')}")
                lines.append("")

    # -- findings -------------------------------------------------------------
    lines.append("## Findings (in-account)")
    lines.append("")
    if not in_scope_findings:
        lines.append("_No in-account findings were identified._")
        lines.append("")
    else:
        lines.append("| ID | Severity | Category | Finding | Confidence |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in _sort_findings(in_scope_findings):
            lines.append(
                f"| {_cell(finding.get('id'))} | {_cell(finding.get('severity'))} | "
                f"{_cell(finding.get('category'))} | {_cell(finding.get('title'))} | "
                f"{_cell(finding.get('confidence'))} |"
            )
        lines.append("")
        for finding in _sort_findings(in_scope_findings):
            lines.append(f"### {finding.get('id', 'FINDING')} — {finding.get('title', 'Finding')}")
            lines.append("")
            lines.append(f"**Severity:** {finding.get('severity', 'Unknown')}  ")
            lines.append(f"**Category:** {finding.get('category', 'Other')}  ")
            lines.append(f"**Confidence:** {finding.get('confidence', 'unknown')}  ")
            lines.append(f"**Scope:** {finding.get('_scope', {}).get('label', 'In-account')}")
            lines.append("")
            if finding.get("affected_principals"):
                lines.append("**Affected principals:**")
                lines.append("")
                lines.append(_bullets(finding.get("affected_principals")))
                lines.append("")
            if finding.get("affected_resources"):
                lines.append("**Affected resources:**")
                lines.append("")
                lines.append(_bullets(finding.get("affected_resources")))
                lines.append("")
            if finding.get("evidence"):
                lines.append(f"**Evidence:** {finding.get('evidence')}")
                lines.append("")
            if finding.get("impact"):
                lines.append(f"**Impact:** {finding.get('impact')}")
                lines.append("")
            if finding.get("recommendation"):
                lines.append(f"**Recommendation:** {finding.get('recommendation')}")
                lines.append("")

    # -- merged out-of-scope (optional) ---------------------------------------
    if include_out_of_scope and out_of_scope_findings:
        lines.append("## Out-of-scope items (included on request)")
        lines.append("")
        for finding in _sort_findings(out_of_scope_findings):
            lines.append(f"### {finding.get('id', 'FINDING')} — {finding.get('title', 'Finding')}")
            lines.append("")
            lines.append(f"**Severity:** {finding.get('severity', 'Unknown')}  ")
            lines.append(f"**Scope:** {finding.get('_scope', {}).get('label', 'Out of scope')} "
                         f"({finding.get('_scope', {}).get('reason', '')})")
            lines.append("")
            if finding.get("evidence"):
                lines.append(f"**Evidence:** {finding.get('evidence')}")
                lines.append("")
            if finding.get("recommendation"):
                lines.append(f"**Recommendation:** {finding.get('recommendation')}")
                lines.append("")

    # -- appendices -----------------------------------------------------------
    lines.append("## Appendix A — Excluded / out-of-scope items")
    lines.append("")
    if not out_of_scope_findings:
        lines.append("_None._")
        lines.append("")
    else:
        lines.append("These items were identified but excluded because they fall outside the target account "
                     "(parent organisation, management account, or other accounts):")
        lines.append("")
        lines.append("| ID | Severity | Scope | Finding | Reason |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in _sort_findings(out_of_scope_findings):
            meta = finding.get("_scope", {})
            lines.append(
                f"| {_cell(finding.get('id'))} | {_cell(finding.get('severity'))} | "
                f"{_cell(meta.get('label'))} | {_cell(finding.get('title'))} | {_cell(meta.get('reason'))} |"
            )
        lines.append("")

    lines.append("## Appendix B — PMapper native findings")
    lines.append("")
    pmapper_text = (evidence.pmapper_text or "").strip()
    lines.append("```text")
    lines.append(truncate(pmapper_text, 20000) if pmapper_text else "No PMapper text analysis available.")
    lines.append("```")
    lines.append("")

    lines.append("## Appendix C — Collection log")
    lines.append("")
    lines.append("| Tool | Command | Exit | Duration | Note |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in run_result.log:
        lines.append(
            f"| {_cell(entry.tool)} | `{_cell(entry.command)}` | {entry.returncode} | "
            f"{entry.duration_s:.0f}s | {_cell(entry.note)} |"
        )
    lines.append("")
    if run_result.errors:
        lines.append("**Errors encountered:**")
        lines.append("")
        for err in run_result.errors:
            lines.append(f"- {err}")
        lines.append("")

    lines.append("## Appendix D — CloudFox artifacts")
    lines.append("")
    if evidence.file_inventory:
        lines.append(f"Collected {len(evidence.file_inventory)} artifacts from `{evidence.cloudfox_dir}`:")
        lines.append("")
        lines.append("```text")
        for rel, size in sorted(evidence.file_inventory):
            lines.append(f"{size:>10,}  {rel}")
        lines.append("```")
    else:
        lines.append("_No CloudFox artifacts were collected._")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_This report was produced from read-only CloudFox and PMapper enumeration, with DeepSeek used to "
                 "assist the analysis. AI-assisted findings are a starting point for human review and should be "
                 "validated before remediation or disclosure._")
    lines.append("")
    return "\n".join(lines)
