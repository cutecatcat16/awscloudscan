"""Deterministic worst-first ordering for findings and escalation paths.

Severity alone cannot say *which* of two High findings matters more — a role that
trusts ``*`` and a stale public load balancer are both High, and alphabetical
order is not an answer. This module is the operator's answer, written down as
data: an ordered rule table, worst first, matched against each finding's
category, verified primitives, escalation-edge shape and text.

Ordering is a pure function of the finding, so it is deterministic — the same
findings always come out in the same order, whatever the model did or did not
return, and a run with the AI unavailable is ordered exactly like one with it.

The head of the table is the operator's own ranking of the classes that matter in
this engagement (see :data:`RULES`). Anything matching no rule falls to the tail,
ordered by severity, so a finding the table has never heard of still appears — it
just does not outrank a known one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .issues import SEVERITY_RANK

__all__ = ["Rule", "RULES", "rule_for", "rank_of", "order_key", "sort_findings", "PRIORITY_TABLE"]


@dataclass(frozen=True)
class Rule:
    """One line of the ranking table.

    Every attribute is a condition; a rule matches only when *all* the conditions
    it states hold. Within an attribute the values are alternatives — any one is
    enough. A rule that states nothing at all would match everything, so
    :data:`RULES` never contains one.

    ``category`` is the report category (``Trust``, ``Network``, ``Compute``,
    ``Data Access``, ``Privilege Escalation``). ``actions`` are lowercased
    primitives. ``edge_shapes`` are escalation-edge shapes (``ec2``, ``lambda``,
    ...). ``patterns`` are regular expressions over the finding's lowercased text
    (title, evidence, resources, edge fields), used for the inventory classes,
    which carry no structured primitive of their own.
    """

    key: str
    label: str
    category: Optional[str] = None
    actions: Tuple[str, ...] = ()
    edge_shapes: Tuple[str, ...] = ()
    patterns: Tuple[str, ...] = ()
    _compiled: Tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", tuple(re.compile(p) for p in self.patterns))

    def matches(self, category: str, actions: set, shapes: set, text: str) -> bool:
        if self.category is not None and category != self.category.lower():
            return False
        if self.actions and not (actions & set(self.actions)):
            return False
        if self.edge_shapes and not (shapes & set(self.edge_shapes)):
            return False
        if self._compiled and not any(pattern.search(text) for pattern in self._compiled):
            return False
        return True


# Worst first. The first fifteen are the operator's ordering of the classes that
# matter in this environment; the tail is severity-ordered, so a class the table
# does not name still lands somewhere sane.
RULES: Tuple[Rule, ...] = (
    Rule("wildcard-trust", "Trust of every principal (Principal \"*\")",
         category="Trust",
         patterns=(r'principal["\s:=]{0,4}\*', r"\ball principals\b", r"\bany principal\b")),
    Rule("passrole-to-ec2", "iam:PassRole into EC2",
         actions=("iam:passrole",), edge_shapes=("ec2",)),
    Rule("public-api-gateway", "Public API Gateway fronting an internal service",
         category="Network", patterns=(r"\bapi[ -]?gateway\b", r"\bapigw\b", r"\bapi-gw\b")),
    Rule("ssm-parameter-secrets", "SSM Parameter Store secrets readable",
         actions=("ssm:getparameter", "ssm:getparameters")),
    Rule("root-trust-without-external-id", "Root trust without sts:ExternalId",
         category="Trust",
         patterns=(r"without (an )?external[ -]?id", r"\bno external[ -]?id\b",
                   r"missing external[ -]?id")),
    Rule("management-root-trust", "Trust of the management-account root",
         category="Trust",
         patterns=(r"management[ -]account", r"\bmanagement root\b", r"\borgani[sz]ation root\b")),
    Rule("external-account-trust", "Trust of an external account",
         category="Trust",
         patterns=(r"external", r"cross[ -]account", r"third[ -]party", r"another account")),
    Rule("federated-trust", "Federated trust without conditions",
         category="Trust", patterns=(r"federat",)),
    Rule("public-load-balancer", "Public load balancer",
         category="Network",
         patterns=(r"load balancer", r"\b(elb|alb|nlb)\b", r"listener")),
    Rule("internet-open-ports", "Internet-reaching security-group ports on EC2",
         category="Network",
         patterns=(r"0\.0\.0\.0/0", r"\bwide[ -]open\b", r"internet", r"\bpublic\b")),
    Rule("ecs-task-role", "ECS task role reachable by the task's workload",
         category="Compute",
         patterns=(r"\becs\b", r"task role", r"task definition")),
    Rule("update-assume-role-policy", "iam:UpdateAssumeRolePolicy",
         actions=("iam:updateassumerolepolicy",)),
    Rule("create-access-key", "iam:CreateAccessKey",
         actions=("iam:createaccesskey",)),
    Rule("public-bucket", "Public S3 bucket",
         category="Data Access",
         patterns=(r"public.{0,40}bucket", r"bucket.{0,40}public", r"public access block",
                   r"public read")),
    Rule("cross-account-bucket", "S3 bucket reachable from another account",
         category="Data Access",
         patterns=(r"cross[ -]account", r"external account", r"another account")),
)

# A human-readable rendering of the ranking, for docs and tests.
PRIORITY_TABLE: str = "\n".join(
    f"{index:>2}. {rule.label}" for index, rule in enumerate(RULES, start=1))

_UNRANKED = len(RULES)


def _category(finding: Dict[str, Any]) -> str:
    return str(finding.get("category", "") or "").strip().lower()


def _actions(finding: Dict[str, Any]) -> set:
    """The finding's verified primitives, from ``primitives`` and ``verification``."""
    actions = set()
    primitives = finding.get("primitives")
    if isinstance(primitives, (list, tuple)):
        actions.update(str(v).strip().lower() for v in primitives)
    elif isinstance(primitives, str):
        actions.add(primitives.strip().lower())
    verification = finding.get("verification")
    if isinstance(verification, dict):
        actions.update(str(k).strip().lower() for k in verification)
    return actions


def _edge_shapes(finding: Dict[str, Any]) -> set:
    edge = finding.get("edge")
    shapes = set()
    if isinstance(edge, dict):
        for key in ("shape", "service"):
            value = edge.get(key)
            if value:
                shapes.add(str(value).strip().lower())
    return shapes


def _text(finding: Dict[str, Any]) -> str:
    """The finding's searchable text, lowercased.

    Structured fields first, then everything a finding might carry as prose. The
    inventory classes have no structured primitive — until their own detectors
    exist, the model's title and evidence are the only place their nature is
    written down, so the rules read them.
    """
    parts: List[str] = []
    for key in ("title", "description", "evidence", "impact", "category", "reason"):
        value = finding.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("affected_principals", "affected_resources"):
        value = finding.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value)
        elif isinstance(value, str):
            parts.append(value)
    edge = finding.get("edge")
    if isinstance(edge, dict):
        for key in ("shape", "service", "resource", "resources", "target_role"):
            value = edge.get(key)
            if isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value)
            elif value:
                parts.append(str(value))
    return " ".join(parts).lower()


def rule_for(finding: Dict[str, Any]) -> Optional[Rule]:
    """The first rule that pins this finding, or ``None`` for the severity tail."""
    if not isinstance(finding, dict):
        return None
    category = _category(finding)
    actions = _actions(finding)
    shapes = _edge_shapes(finding)
    text = _text(finding)
    for rule in RULES:
        if rule.matches(category, actions, shapes, text):
            return rule
    return None


def rank_of(finding: Dict[str, Any]) -> int:
    """The finding's position in the table; :data:`_UNRANKED` for the tail."""
    rule = rule_for(finding)
    if rule is None:
        return _UNRANKED
    return RULES.index(rule)


def order_key(finding: Dict[str, Any]) -> Tuple[int, int]:
    """Sort key: table rank first, then severity.

    Rank dominates severity — that is the point of the table: a known class beats
    an unknown one whatever the grader said. Severity breaks ties inside a rank,
    and because the sort is stable, findings a rank and severity tie keep the
    order they arrived in (for escalation paths, the model's own worst-first
    ordering; for findings, the deterministic merge order).
    """
    severity = str(finding.get("severity", "") if isinstance(finding, dict) else "").strip()
    severity = severity[:1].upper() + severity[1:].lower() if severity else ""
    return (rank_of(finding), -SEVERITY_RANK.get(severity, 0))


def sort_findings(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """*items* ordered worst first: the priority table, then severity."""
    return sorted(items, key=order_key)
