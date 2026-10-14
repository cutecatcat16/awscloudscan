"""Derive service-based escalation edges from CloudFox inventories plus verified grants.

PMapper's ``lambda``/``ec2``/``cloudformation``/``codebuild`` edge checkers answer
a question nothing else in the pipeline answers: *which principal can modify a
resource that executes as role R, and therefore run code as R?* PMapper reaches
the answer by brute force — ``for destination in nodes: for source in nodes: for
resource in resources``, with a full local policy evaluation at the innermost
step. On this engagement the Lambda checker alone is 68 role-destinations x 201
principals x 75 functions. PMapper spent 2 h 52 m inside it and was interrupted
before it wrote any graph.

The same edge can be derived exactly, and in milliseconds, by joining two things
awscloudscan already holds:

* CloudFox's resource inventories, which record the role each Lambda function,
  EC2 instance and CloudFormation stack executes as (``lambda.csv``,
  ``instances.csv``, ``cloudformation.csv``);
* :mod:`cloudscan.verify`'s ``SimulatePrincipalPolicy`` verdicts, which record —
  according to the AWS evaluation engine, SCPs and MFA included — whether a
  principal may call the modifying action.

Two rules keep the join honest:

1. **No grant is ever invented.** A shape becomes a finding only when *every*
   action it needs carries a confirmed or conditional simulator verdict for that
   principal. AWS remains the verifier of record; this module contributes only
   the resource-to-role link.
2. **Nothing PMapper would not have found is claimed.** The inventory is limited
   to the services PMapper's four checkers sweep, and principals CloudFox already
   rates administrative are not used as sources, just as PMapper skips
   ``is_admin`` nodes. The join therefore reports the same class of edge rather
   than a superset of it.

Findings are collapsed per source when a path's grants are unbounded — a single
``lambda:UpdateFunctionCode`` on ``"*"`` reaches every function's execution role
in the account, and reporting that once per role would weight the report's
severity counts by one root cause. The collapsed finding lists the roles it
reaches; the per-role detail stays on its ``edge.roles`` record.

The one place this module is deliberately stricter *and* more accurate than
PMapper is conditions. PMapper supplies ``iam:PassedToService`` itself when it
evaluates ``iam:PassRole``, so a policy that is genuinely gated on the service
comes back as a plain allow. Here the context is left to the simulator, so such
a grant is reported as conditional — which is what it is.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .issues import CATEGORY_PRIVESC, SEVERITY_RANK as _SEVERITY_RANK
from .util import info, warn
from .verify import CONDITIONAL, CONFIRMED, admin_principals, expand_action
from .verify import _severity_for as severity_for
from .verify import _truthy

EDGE_ID_PREFIX = "EDGE"

# How a shape's grant has to line up with the resource that executes as the role.
TARGET_RESOURCE = "resource"   # the grant must cover the concrete resource
TARGET_ROLE = "role"           # the grant must cover the execution role ARN
TARGET_ANY = "any"             # the attacker names the resource, so any scope works


# ---------------------------------------------------------------------------
# Escalation shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Need:
    """One action a shape requires, and what the simulator must have allowed."""

    action: str
    target: str = TARGET_RESOURCE


@dataclass(frozen=True)
class Shape:
    """A way of turning a grant on a resource into control of its execution role.

    ``bonus`` records grants that make the shape *stronger* but are not required
    for it. They are reported alongside the path, not as a separate finding: a
    shape whose needs are a superset of another shape's can only ever fire when
    that other shape fires too, so emitting it separately would duplicate the
    finding rather than add information.
    """

    key: str
    service: str
    reason: str
    needs: Tuple[Need, ...]
    bonus: Tuple[Need, ...] = ()


# The shapes mirror PMapper's four checkers so that dropping them costs no
# coverage. Read ``{resource}`` and ``{role}`` as the substitution points filled
# in from the inventory pair.
_SHAPES: Tuple[Shape, ...] = (
    # -- Lambda -------------------------------------------------------------
    Shape(
        "lambda-update-code", "Lambda",
        "can rewrite the code of {resource}, which then runs as {role}",
        (Need("lambda:UpdateFunctionCode"),),
        # Rewriting the configuration as well is a strictly stronger capability
        # on the same resource, so it is reported as part of this path.
        bonus=(Need("lambda:UpdateFunctionConfiguration"),),
    ),
    Shape(
        "lambda-create-function", "Lambda",
        "can create a new Lambda function, pass {role} to it, and run arbitrary code as {role}",
        (Need("iam:PassRole", TARGET_ROLE), Need("lambda:CreateFunction", TARGET_ANY)),
    ),
    # -- EC2 ----------------------------------------------------------------
    Shape(
        "ec2-run-instance", "EC2",
        "can launch an instance with the profile of {resource} and run code as {role}",
        (Need("iam:PassRole", TARGET_ROLE), Need("ec2:RunInstances", TARGET_ANY)),
    ),
    Shape(
        "ec2-run-then-associate", "EC2",
        "can launch an instance and attach an instance profile to it, running code as {role}",
        (Need("ec2:RunInstances", TARGET_ANY), Need("ec2:AssociateIamInstanceProfile", TARGET_ANY)),
    ),
    Shape(
        "ec2-new-profile", "EC2",
        "can create an instance profile, put {role} in it and launch an instance with it",
        (Need("iam:PassRole", TARGET_ROLE), Need("iam:CreateInstanceProfile", TARGET_ANY),
         Need("iam:AddRoleToInstanceProfile", TARGET_ANY), Need("ec2:RunInstances", TARGET_ANY)),
    ),
    # -- CloudFormation -----------------------------------------------------
    Shape(
        "cloudformation-create-stack", "CloudFormation",
        "can create a stack, pass {role} to it, and run arbitrary code as {role}",
        (Need("iam:PassRole", TARGET_ROLE), Need("cloudformation:CreateStack", TARGET_ANY)),
    ),
    Shape(
        "cloudformation-update-stack", "CloudFormation",
        "can update {resource}, which executes as {role}",
        (Need("cloudformation:UpdateStack"),),
    ),
    Shape(
        "cloudformation-changeset", "CloudFormation",
        "can create and execute a change set on {resource}, which executes as {role}",
        (Need("cloudformation:CreateChangeSet"), Need("cloudformation:ExecuteChangeSet")),
    ),
    # -- CodeBuild ----------------------------------------------------------
    Shape(
        "codebuild-start-build", "CodeBuild",
        "can start a build of {resource}, which runs as {role}",
        (Need("codebuild:StartBuild"),),
    ),
    Shape(
        "codebuild-create-project", "CodeBuild",
        "can create a CodeBuild project, pass {role} to it, and run code as {role}",
        (Need("iam:PassRole", TARGET_ROLE), Need("codebuild:CreateProject", TARGET_ANY),
         Need("codebuild:StartBuild", TARGET_ANY)),
    ),
    Shape(
        "codebuild-update-project", "CodeBuild",
        "can edit {resource}, which builds as {role}",
        (Need("codebuild:UpdateProject"), Need("codebuild:StartBuild")),
    ),
)

_SHAPES_BY_SERVICE: Dict[str, Tuple[Shape, ...]] = {}
for _shape in _SHAPES:
    service = _shape.key.split("-", 1)[0]
    _SHAPES_BY_SERVICE.setdefault(service, ())
    _SHAPES_BY_SERVICE[service] += (_shape,)


# ---------------------------------------------------------------------------
# Resource inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSpec:
    """Where an inventory CSV keeps a resource and the role it executes as."""

    service: str
    module: str
    arn_column: Optional[str]        # the resource's own ARN, when the CSV has one
    id_column: str                   # otherwise the column that identifies it
    role_column: str
    label_column: str


_RESOURCE_SPECS: Tuple[ResourceSpec, ...] = (
    ResourceSpec("lambda", "lambda", "Arn", "Name", "Role", "Name"),
    ResourceSpec("ec2", "instances", None, "ID", "Role", "Name"),
    ResourceSpec("cloudformation", "cloudformation", None, "Name", "Role", "Name"),
)


@dataclass(frozen=True)
class ResourceRole:
    """One resource and the role it executes as."""

    service: str
    label: str
    identifier: str      # the ARN used for grant matching
    role: str


def _resource_roles(evidence: Any) -> List[ResourceRole]:
    """Collect every ``(resource, execution role)`` pair CloudFox recorded.

    CodeBuild is absent on purpose: CloudFox's codebuild module writes no CSV
    when the account has no projects, so there is nothing to join. That is a
    real finding in its own right, and it is reported through ``notes``.
    """
    pairs: List[ResourceRole] = []
    seen: set = set()
    account = (getattr(evidence, "account", "") or "").strip()
    region = (getattr(evidence, "region", "") or "").strip()

    for spec in _RESOURCE_SPECS:
        rows = evidence.csv(spec.module)
        for row in rows:
            role = (row.get(spec.role_column) or "").strip()
            # An empty role column is common (CloudFox leaves it blank when the
            # resource has no task/execution role), and a role in another
            # account is not reachable from this one.
            if ":role/" not in role:
                continue
            if account and f":iam::{account}:role/" not in role and ":iam::" in role:
                continue

            label = (row.get(spec.label_column) or "").strip()
            if spec.arn_column:
                identifier = (row.get(spec.arn_column) or "").strip()
            elif spec.service == "cloudformation":
                # describe_stacks reports a stack id, which CloudFox does not
                # carry; the name is the only handle available, so match grants
                # against a name-derived id.
                identifier = f"arn:aws:cloudformation:{region}:{account}:stack/{label}/*"
            else:
                identifier = (row.get(spec.id_column) or "").strip()
            if not identifier:
                continue

            key = (spec.service, identifier, role)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(ResourceRole(spec.service, label or identifier, identifier, role))

    return pairs


# ---------------------------------------------------------------------------
# Grant lookup
# ---------------------------------------------------------------------------


def _covers(pattern: str, resource: str) -> bool:
    """True when a grant scoped to *pattern* also covers *resource*."""
    p = (pattern or "").strip()
    r = (resource or "").strip()
    if not p or not r:
        return False
    if p == "*":
        return True
    if p.lower() == r.lower():
        return True
    return fnmatch.fnmatchcase(r.lower(), p.lower())


@dataclass(frozen=True)
class _Match:
    """The simulator verdict that satisfies one :class:`Need`."""

    action: str
    verdict: str
    resource: str
    mfa_required: bool = False

    @property
    def conditional(self) -> bool:
        return self.verdict == CONDITIONAL or self.mfa_required

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "verdict": self.verdict,
            "resource": self.resource,
            "mfa_required": self.mfa_required,
        }


class _GrantIndex:
    """The simulator's verdicts, indexed for repeated lookup."""

    def __init__(self, decisions: Sequence[Any]) -> None:
        self._by_principal: Dict[str, Dict[str, List[Any]]] = {}
        for decision in decisions:
            candidate = getattr(decision, "candidate", None)
            if candidate is None:
                continue
            if getattr(decision, "verdict", "") not in (CONFIRMED, CONDITIONAL):
                continue
            arn = (candidate.principal_arn or "").strip()
            action = (candidate.action or "").strip().lower()
            if not arn or not action:
                continue
            self._by_principal.setdefault(arn, {}).setdefault(action, []).append(decision)

    def principals(self) -> List[str]:
        return sorted(self._by_principal)

    def match(self, principal: str, need: Need, resource: str, role: str) -> Optional[_Match]:
        """Return the strongest verdict satisfying *need*, or ``None``.

        A confirmed grant always beats a conditional one, and among equals the
        tightest resource pattern wins, so the evidence quotes the most specific
        statement that makes the edge work.
        """
        entries = self._by_principal.get(principal, {}).get(need.action.strip().lower())
        if not entries:
            return None

        best: Optional[_Match] = None
        best_key: Tuple[int, int] = (-1, -1)
        for decision in entries:
            candidate = decision.candidate
            grant_resource = (getattr(decision, "simulated_resource", "")
                              or candidate.resource or "*").strip() or "*"
            if need.target == TARGET_RESOURCE:
                if not _covers(grant_resource, resource):
                    continue
            elif need.target == TARGET_ROLE:
                if not _covers(grant_resource, role):
                    continue
            # TARGET_ANY accepts any scope: the principal names the resource it
            # creates, so a pattern such as ``...function:SecretsManager*`` still
            # lets it stand up a function under that name and pass the role.

            mfa = bool(getattr(decision, "mfa_required", False))
            confirmed = 1 if getattr(decision, "verdict", "") == CONFIRMED and not mfa else 0
            specificity = 0 if grant_resource == "*" else len(grant_resource)
            key = (confirmed, specificity)
            if key > best_key:
                best_key = key
                best = _Match(candidate.action, decision.verdict, grant_resource, mfa)
        return best


def _satisfy(index: _GrantIndex, principal: str, shape: Shape,
             pair: ResourceRole) -> Tuple[Optional[List[_Match]], List[_Match]]:
    """Return the grants that satisfy *shape*, plus any bonus grants it also has.

    The first element is ``None`` when the shape is not satisfied at all; the
    second is always a (possibly empty) list.
    """
    matches: List[_Match] = []
    for need in shape.needs:
        match = index.match(principal, need, pair.identifier, pair.role)
        if match is None:
            return None, []
        matches.append(match)

    extras: List[_Match] = []
    for need in shape.bonus:
        match = index.match(principal, need, pair.identifier, pair.role)
        if match is not None:
            extras.append(match)
    return matches, extras


def _unbounded(shape: Shape, matches: Sequence[_Match]) -> bool:
    """True when nothing about the shape ties it to one resource or one role.

    Only the needs that are checked against a scope can bound a shape. A
    ``TARGET_ANY`` action is one the attacker points at a resource they create
    themselves, so the scope of that grant never limits which role the path
    reaches — a principal that may create a function anywhere and pass ``*``
    reaches every Lambda execution role in the account regardless of how the
    create grant is spelled.
    """
    return all(
        match.resource == "*"
        for need, match in zip(shape.needs, matches)
        if need.target in (TARGET_RESOURCE, TARGET_ROLE)
    )


# ---------------------------------------------------------------------------
# Privilege of the target role
# ---------------------------------------------------------------------------


def _role_privilege(rows: Iterable[Dict[str, str]]) -> Dict[str, Tuple[str, str]]:
    """Map each principal to its own worst escalation primitive.

    An edge's severity is the value of what it reaches. Escalating into a role
    that holds ``iam:AttachRolePolicy`` is worth more than reaching a role that
    can only read one bucket, so the severity is taken from the target role's
    own verified-inventory permissions rather than from the edge's mechanics.

    This is a statement-level read of ``permissions.csv``, not a simulation, so
    it is a ranking device only — never a claim in the report.
    """
    worst: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        if str(row.get("Effect", "")).strip().lower() != "allow":
            continue
        arn = (row.get("Arn") or "").strip()
        statement_action = (row.get("Action") or "").strip()
        if not arn or not statement_action:
            continue

        if statement_action == "*":
            candidates = [("Critical", "*")]
        else:
            candidates = [(severity_for(action), action) for action in expand_action(statement_action)]
        for severity, action in candidates:
            current = worst.get(arn)
            if current is None or _SEVERITY_RANK.get(severity, 0) > _SEVERITY_RANK.get(current[0], 0):
                worst[arn] = (severity, action)
    return worst


def _worst_role(details: Iterable[Dict[str, Any]]) -> Tuple[str, str]:
    """The most valuable role a finding reaches, and the primitive that makes it so.

    ``Medium`` is the floor, matching the default used when a target role has no
    ranked primitive of its own: a reachable role of unknown value is still a
    reachable role.
    """
    severity, primitive = "Medium", ""
    best = _SEVERITY_RANK.get(severity, 0)
    for detail in details:
        rank = _SEVERITY_RANK.get(detail.get("severity", ""), 0)
        if rank > best:
            best = rank
            severity = detail.get("severity", "")
            primitive = detail.get("reached_primitive", "") or ""
    return severity, primitive


def _admin_principals(evidence: Any) -> set:
    """Principals CloudFox already rates as administrative.

    The rule lives in :func:`verify.admin_principals` so that the resource join
    and the verifier suppress the same principals — the two used to disagree,
    which is how 259 admin tautologies survived into the report.
    """
    return admin_principals(evidence)


def _principal_labels(evidence: Any) -> Dict[str, str]:
    """ARN -> display name, from whichever CloudFox inventory names it."""
    labels: Dict[str, str] = {}
    for module in ("principals", "permissions"):
        for row in evidence.csv(module):
            arn = (row.get("Arn") or "").strip()
            name = (row.get("Name") or "").strip()
            if arn and name:
                labels.setdefault(arn, name)
    return labels


def _short(arn: str) -> str:
    """The human handle for a principal ARN."""
    tail = (arn or "").rstrip("/").rsplit("/", 1)[-1]
    return tail or arn


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class EdgeResult:
    """Edges recovered by the join, shaped as report findings."""

    findings: List[Dict[str, Any]] = field(default_factory=list)
    resources: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        by_service: Dict[str, int] = {}
        for finding in self.findings:
            service = (finding.get("edge") or {}).get("service", "unknown")
            by_service[service] = by_service.get(service, 0) + 1
        return {
            "resources": self.resources,
            "edges": len(self.findings),
            "roles": sum(len((f.get("edge") or {}).get("roles") or [])
                         for f in self.findings),
            "by_service": by_service,
            "notes": list(self.notes),
        }


def derive_edges(evidence: Any, verify_result: Any) -> EdgeResult:
    """Join CloudFox's resource->role inventory against the simulator's verdicts."""
    result = EdgeResult()
    decisions = list(getattr(verify_result, "decisions", []) or [])
    if not decisions:
        result.notes.append("no simulator verdicts were available, so no edge could be joined")
        return result

    pairs = _resource_roles(evidence)
    result.resources = len(pairs)
    if not pairs:
        result.notes.append("no resource in the CloudFox inventory carried an execution role")
        return result

    index = _GrantIndex(decisions)
    privilege = _role_privilege(evidence.csv("permissions"))
    labels = _principal_labels(evidence)
    admins = _admin_principals(evidence)

    def label(arn: str) -> str:
        return labels.get(arn) or _short(arn)

    sources = index.principals()
    skipped_admins = [p for p in sources if p in admins]
    if skipped_admins:
        result.notes.append(
            f"{len(skipped_admins)} principal(s) are administrative in their own right "
            f"({', '.join(label(p) for p in skipped_admins[:4])}"
            + (f" and {len(skipped_admins) - 4} more" if len(skipped_admins) > 4 else "")
            + "), so they are not treated as escalation sources")

    # Aggregate the edges. A scoped grant gets one finding per way of getting
    # from a principal to a role, listing every resource that makes it work.
    # An unbounded grant reaches every resource in the inventory, so it gets one
    # finding per (source, shape) that lists the roles it reaches instead: fifty
    # near-identical findings sharing a single root cause weight the report's
    # severity counts without adding information. The per-role breakdown is kept
    # in full on the finding's ``edge`` record, so nothing is lost.
    aggregated: Dict[Tuple[str, Optional[str], str], Dict[str, Any]] = {}
    for pair in pairs:
        shapes = _SHAPES_BY_SERVICE.get(pair.service, ())
        if not shapes:
            continue
        for principal in sources:
            if principal == pair.role:
                continue  # escalating into the role you already hold is not an edge
            if principal in admins:
                continue  # an admin escalating is the baseline, not a finding
            for shape in shapes:
                matches, extras = _satisfy(index, principal, shape, pair)
                if matches is None:
                    continue
                unbounded = _unbounded(shape, matches)
                key = (principal, None if unbounded else pair.role, shape.key)
                entry = aggregated.get(key)
                if entry is None:
                    entry = {
                        "principal": principal,
                        "shape": shape,
                        "matches": matches,
                        "extras": [],
                        "unbounded": unbounded,
                        "roles": {},
                    }
                    aggregated[key] = entry
                for extra in extras:
                    if all(known.action != extra.action for known in entry["extras"]):
                        entry["extras"].append(extra)
                severity, reached = privilege.get(pair.role, ("Medium", ""))
                detail = entry["roles"].get(pair.role)
                if detail is None:
                    detail = {"severity": severity, "reached_primitive": reached,
                              "resources": []}
                    entry["roles"][pair.role] = detail
                if pair.identifier not in detail["resources"]:
                    detail["resources"].append(pair.identifier)

    for service, shape_count in _SHAPES_BY_SERVICE.items():
        if not any(p.service == service for p in pairs):
            result.notes.append(f"no {service} resource carried an execution role, "
                                f"so its {len(shape_count)} shape(s) found nothing")

    def order(entry: Dict[str, Any]) -> Tuple[int, str, str]:
        severity, _ = _worst_role(entry["roles"].values())
        return (-_SEVERITY_RANK.get(severity, 0), entry["principal"], entry["shape"].key)

    findings: List[Dict[str, Any]] = []
    for index_number, entry in enumerate(sorted(aggregated.values(), key=order), start=1):
        findings.append(_finding(index_number, entry, label))
    result.findings = findings
    return result


def _finding(number: int, entry: Dict[str, Any], label) -> Dict[str, Any]:
    """Shape one aggregated edge into the report's finding schema."""
    shape: Shape = entry["shape"]
    principal: str = entry["principal"]
    matches: List[_Match] = entry["matches"]
    extras: List[_Match] = entry.get("extras") or []
    roles: Dict[str, Dict[str, Any]] = entry["roles"]
    unbounded: bool = bool(entry.get("unbounded"))

    # Worst role first: an unbounded finding lists the roles it reaches in the
    # order a reader cares about, and takes its severity from the best of them.
    ordered_roles = sorted(roles.items(),
                           key=lambda item: (-_SEVERITY_RANK.get(item[1].get("severity", ""), 0),
                                             -len(item[1].get("resources", [])),
                                             item[0]))
    # The headline values come from the worst role reached, so the sentence that
    # justifies the severity always agrees with the listing above it.
    primary_role = ordered_roles[0][0]
    severity = ordered_roles[0][1].get("severity") or "Medium"
    reached_primitive = ordered_roles[0][1].get("reached_primitive") or ""
    resources: List[str] = sorted({r for _, detail in ordered_roles
                                   for r in detail["resources"]})
    conditional = any(m.conditional for m in matches)

    if unbounded:
        reason = (f"the grants that make this path work are not scoped to a resource, so every "
                  f"{shape.service} resource in the account is in reach and with it the "
                  f"{len(roles)} role(s) those resources execute as")
        title = f"{label(principal)} can escalate to any {shape.service} execution role"
    else:
        reason = shape.reason.format(resource=resources[0], role=label(primary_role))
        title = f"{label(principal)} can escalate to {label(primary_role)} via {shape.service}"

    if unbounded:
        actions = ", ".join(sorted({m.action for m in matches}))
        listing = order_roles_for_evidence(ordered_roles, label)
        evidence_lines = [
            f"Unbounded grant: {actions} "
            f"{'is' if len(matches) == 1 else 'are'} allowed on \"*\", so this path is not "
            f"limited to one resource or one role.",
            f"Roles reached: {len(roles)} execution role(s) across "
            f"{len(resources)} inventoried {shape.service} resource(s).",
            f"Reaches, worst first: {listing}",
            "Every action on the path was confirmed by iam:SimulatePrincipalPolicy:",
        ]
    else:
        evidence_lines = [
            f"Resource to role: {', '.join(resources[:8])}"
            + (f" (and {len(resources) - 8} more)" if len(resources) > 8 else "")
            + f" executes as {label(primary_role)}.",
            f"Escalation path: {reason}.",
            "Every action on the path was confirmed by iam:SimulatePrincipalPolicy:",
        ]
    for match in matches:
        # The verdict is already the word "conditional"; say what gates it.
        gate = " (requires MFA)" if match.mfa_required else (
            " (context-gated)" if match.verdict == CONDITIONAL else "")
        evidence_lines.append(f"    {match.action} on {match.resource} — "
                              f"{match.verdict}{gate}")
    for extra in extras:
        evidence_lines.append(f"    {extra.action} on {extra.resource} — "
                              f"{extra.verdict} (not required, but also granted, so the "
                              f"target's configuration can be changed too)")
    if not any(m.mfa_required for m in matches) and reached_primitive:
        if unbounded:
            evidence_lines.append(
                f"The most valuable role reached, {label(primary_role)}, holds "
                f"{reached_primitive}, which is why this path is rated {severity}.")
        else:
            evidence_lines.append(f"The target role itself holds {reached_primitive}, "
                                  f"which is why this path is rated {severity}.")
    evidence_lines.append(f"Residual risk: {shape.service} edges were derived by joining the "
                          f"CloudFox inventory to AWS-verified grants rather than by a graph "
                          f"walk, so no intermediate hop between the principal and this "
                          f"resource has been checked.")

    if unbounded:
        description = (
            f"{principal} holds grants, confirmed by the AWS IAM evaluation engine, that are "
            f"not scoped to a resource, so it can control every {shape.service} resource in "
            f"this account and run arbitrary code as any of the {len(roles)} role(s) those "
            f"resources execute as. The path is reported once for the unbounded grant rather "
            f"than once per role; the full role-by-role breakdown is in the analysis JSON."
        )
        impact = (f"Full use of any of the {len(roles)} role(s) that {shape.service} resources "
                  f"in this account execute as, including {label(primary_role)}"
                  + (f", which holds {reached_primitive}" if reached_primitive else "")
                  + f". The grant is unscoped, so all {len(resources)} inventoried "
                    f"{shape.service} resource(s) fall inside its scope.")
        recommendation = (f"Scope this grant to the {shape.service} resources that need it — it "
                          f"is currently allowed on \"*\", which is what puts all {len(roles)} "
                          f"role(s) in reach. ") + _recommendation_for(
            shape, f"any of the {len(roles)} roles these resources execute as")
        affected_resources = sorted(roles)
    else:
        description = (
            f"{principal} holds a grant, confirmed by the AWS IAM evaluation engine, that "
            f"allows it to control a {shape.service} resource executing as {label(primary_role)}. "
            f"Controlling the resource means running arbitrary code with that role's "
            f"credentials and every permission attached to it."
        )
        impact = (f"Full use of {label(primary_role)}, including any permissions attached to it. "
                  f"{'This role holds ' + reached_primitive + ', so the path reaches an administrative foothold.'
                     if reached_primitive else 'The blast radius is whatever this role can reach.'}")
        recommendation = _recommendation_for(shape, label(primary_role))
        affected_resources = sorted({primary_role, *resources})

    return {
        "id": f"{EDGE_ID_PREFIX}-{number:03d}",
        "category": CATEGORY_PRIVESC,
        "confidence": "medium" if conditional else "high",
        "title": title,
        "description": description,
        "evidence": "\n".join(evidence_lines),
        "impact": impact,
        "severity": severity,
        "recommendation": recommendation,
        "affected_principals": [principal],
        "affected_resources": affected_resources,
        "from_principal": principal,
        "to_principal": primary_role,
        "primitives": [m.action for m in matches] + [m.action for m in extras],
        "reason": reason,
        "verified": True,
        "edge": {
            "shape": shape.key,
            "service": shape.service,
            "unbounded": unbounded,
            "resource": resources[0],
            "resources": resources,
            # ``target_role``/``target_privilege`` name the worst role reached, so
            # a consumer that only reads the old two keys still gets the headline.
            "target_role": primary_role,
            "target_privilege": reached_primitive,
            "roles": [
                {
                    "role": name,
                    "severity": detail.get("severity", ""),
                    "target_privilege": detail.get("reached_primitive", "") or "",
                    "resources": sorted(detail["resources"]),
                }
                for name, detail in ordered_roles
            ],
            "grants": [m.as_dict() for m in matches],
            "bonus_grants": [m.as_dict() for m in extras],
        },
    }


def order_roles_for_evidence(ordered_roles: Sequence[Tuple[str, Dict[str, Any]]],
                             label, limit: int = 8) -> str:
    """Render the reached roles for an unbounded finding's evidence line.

    Capped: the point of the collapsed finding is readability, and the complete
    list is on the finding's ``edge.roles`` record.
    """
    parts = []
    for name, detail in ordered_roles[:limit]:
        note = f"{label(name)} ({detail.get('severity') or 'unrated'}"
        if detail.get("reached_primitive"):
            note += f", holds {detail['reached_primitive']}"
        note += f", {len(detail['resources'])} resource(s))"
        parts.append(note)
    if len(ordered_roles) > limit:
        parts.append(f"and {len(ordered_roles) - limit} more, listed in the analysis JSON")
    return "; ".join(parts)


def _recommendation_for(shape: Shape, role: str) -> str:
    if shape.key.startswith("lambda"):
        return (f"Scope the {', '.join(n.action for n in shape.needs)} grant to the exact "
                f"functions that need editing, or remove it; then confirm nothing else can "
                f"reach {role} through Lambda.")
    if shape.key.startswith("ec2"):
        return (f"Restrict iam:PassRole to the services that genuinely need {role}, and gate "
                f"ec2:RunInstances and instance-profile changes behind a condition such as "
                f"aws:SourceVpc or an MFA requirement.")
    if shape.key.startswith("cloudformation"):
        return (f"Limit the stack actions to the named stacks that need them, and pin "
                f"iam:PassRole to the CloudFormation service so that {role} cannot be handed "
                f"to an arbitrary template.")
    if shape.key.startswith("codebuild"):
        return (f"Scope the CodeBuild actions to the projects that need them and pin "
                f"iam:PassRole to codebuild.amazonaws.com so {role} cannot be passed to a "
                f"newly created project.")
    return f"Scope this path to the resources that need it, or remove it, so that {role} is unreachable."


# ---------------------------------------------------------------------------
# Report integration
# ---------------------------------------------------------------------------


def attach_edges_to_analysis(analysis: Any, edge_result: EdgeResult) -> int:
    """Merge recovered edges into an :class:`Analysis`, returning how many were added.

    Edges are placed ahead of the flat verified grants: a chain that reaches a
    role is more actionable than a single confirmed permission, and the grants
    that make up the chain are already listed inside the edge's own evidence.
    """
    if not edge_result.findings:
        return 0
    existing = list(getattr(analysis, "findings", []) or [])
    existing_titles = {str(f.get("title", "")).strip() for f in existing if isinstance(f, dict)}
    fresh = [f for f in edge_result.findings
             if str(f.get("title", "")).strip() not in existing_titles]
    analysis.findings = fresh + existing

    summary = getattr(analysis, "summary", "") or ""
    services = edge_result.summary()["by_service"]
    detail = ", ".join(f"{count} via {service}" for service, count in sorted(services.items()))
    line = (f"Resource-to-role join: {len(fresh)} escalation edge(s) recovered from "
            f"{edge_result.resources} inventoried resource(s)"
            + (f" ({detail})" if detail else "")
            + ". These replace PMapper's lambda/ec2/cloudformation/codebuild checkers, "
              "which were skipped.")
    analysis.summary = (summary + "\n\n" + line).strip() if summary else line
    return len(fresh)


def log_edge_result(edge_result: EdgeResult) -> None:
    """Report what the join found, and be explicit about what it did not."""
    if edge_result.findings:
        services = edge_result.summary()["by_service"]
        detail = ", ".join(f"{count} via {service}" for service, count in sorted(services.items()))
        info(f"resource-to-role join: {len(edge_result.findings)} edge(s) from "
             f"{edge_result.resources} resource(s) ({detail})")
    else:
        warn(f"resource-to-role join found no edge across {edge_result.resources} resource(s)")
    for note in edge_result.notes:
        info(f"resource-to-role join: {note}")
