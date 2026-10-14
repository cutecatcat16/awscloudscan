"""The issue taxonomy for an AWS account scan.

One source of truth for *what this tool reports, what each class means, and what
each confirmed grant grades to*. Both the prompt builder
(:mod:`cloudscan.prompts`) and the verifier (:mod:`cloudscan.verify`) import from
here, so the two cannot drift apart.

They used to. The prompt carried its own ``PRIVESC_ACTIONS`` and told the model to
chase admin-role workloads and to treat a bucket read as privilege escalation; the
verifier carried separate ``_CRITICAL_ACTIONS`` / ``_HIGH_ACTIONS`` /
``_DATA_ACCESS_ACTIONS`` sets, re-graded data reads as *Data Access*, and refused
to originate findings from administrative or service-linked principals at all. The
model therefore generated precisely the noise the verifier discarded.

Seven classes are reported. Two are deliberately *not* — detection/logging
coverage and account hygiene — because neither is exploitable on its own; they
padded the report with compliance boilerplate. See :data:`EXCLUDED_CLASSES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

__all__ = [
    # action sets
    "PRIVESC_ACTIONS", "CRITICAL_ACTIONS", "HIGH_ACTIONS",
    "SECRET_ACTIONS", "OBJECT_ACTIONS", "DATA_ACCESS_ACTIONS", "HIGH_IMPACT_DATA",
    "SEVERITY_RANK",
    # report categories
    "CATEGORY_PRIVESC", "CATEGORY_DATA_ACCESS",
    # class identifiers
    "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT", "TRUST_ABUSE", "CREDENTIAL_EXPOSURE",
    "DATA_EXPOSURE", "NETWORK_EXPOSURE", "COMPUTE_MISCONFIGURATION",
    # registry
    "IssueClass", "CLASSES", "CLASSES_BY_KEY", "EXCLUDED_CLASSES",
    # grading and lookup
    "severity_for", "category_for", "class_of",
    "is_data_access", "is_secret_access", "is_probe_action", "is_escalation",
    # vocabulary the prompt renders from
    "SEVERITY_SCALE", "report_categories", "taxonomy_text",
]


# ---------------------------------------------------------------------------
# Class identifiers
# ---------------------------------------------------------------------------

PRIVILEGE_ESCALATION = "privilege-escalation"
LATERAL_MOVEMENT = "lateral-movement"
TRUST_ABUSE = "cross-account-trust"
CREDENTIAL_EXPOSURE = "credential-exposure"
DATA_EXPOSURE = "data-exposure"
NETWORK_EXPOSURE = "network-exposure"
COMPUTE_MISCONFIGURATION = "compute-misconfiguration"

# The two category strings a *verified* grant is rendered under. The finer class
# labels above are what the prompt and the prose use.
CATEGORY_PRIVESC = "Privilege Escalation"
CATEGORY_DATA_ACCESS = "Data Access"


# ---------------------------------------------------------------------------
# Action sets
# ---------------------------------------------------------------------------

# The probe set: every action worth putting to ``iam:SimulatePrincipalPolicy``.
# Deliberately a superset — it includes the data-access primitives below, which
# are probed for confirmation but graded in their own class, not as escalation.
PRIVESC_ACTIONS = frozenset({
    "iam:attachrolepolicy", "iam:attachuserpolicy", "iam:attachgrouppolicy",
    "iam:createrole", "iam:createuser", "iam:createloginprofile", "iam:createaccesskey",
    "iam:createservicespecificcredential", "iam:createpolicyversion", "iam:setdefaultpolicyversion",
    "iam:putrolepolicy", "iam:putuserpolicy", "iam:putgrouppolicy",
    "iam:addusertogroup", "iam:updateassumerolepolicy", "iam:passrole",
    "iam:createinstanceprofile", "iam:addroletoinstanceprofile",
    "iam:updateaccesskey", "iam:deactivateaccesskey", "iam:resyncmfadevice",
    "sts:assumerole", "sts:assumerolewithsaml", "sts:assumerolewithwebidentity",
    "sts:getfederationtoken", "sts:getservicebearertoken",
    "lambda:createfunction", "lambda:invokefunction", "lambda:updatefunctioncode",
    "lambda:addpermission", "lambda:createeventsourcemapping",
    "lambda:updatefunctionconfiguration",
    "ssm:sendcommand", "ssm:startautomationexecution", "ssm:createassociation",
    "ec2:runinstances", "ec2:associateiaminstanceprofile",
    "cloudformation:createstack", "cloudformation:updatestack",
    "cloudformation:createchangeset", "cloudformation:executechangeset",
    "datapipeline:createpipeline", "datapipeline:putpipelinedefinition",
    "glue:createdevendpoint", "glue:createjob", "glue:updatedevendpoint",
    "codebuild:createproject", "codebuild:startbuild", "codebuild:updateproject",
    "sagemaker:createpresignednotebookinstanceurl", "sagemaker:createnotebookinstance",
    "events:putrule", "events:puttargets", "ecs:registertaskdefinition", "ecs:runtask",
    "ecs:createservice", "ecs:updateservice", "batch:submitjob",
    "organizations:*", "secretsmanager:getsecretvalue", "ssm:getparameter",
    "ssm:getparameters", "kms:decrypt", "s3:getobject", "s3:putobject",
    "iam:updateassumerolepolicy", "iam:createsamlprovider", "iam:updateuser",
})

# Escalation primitives whose confirmation is, on its own, an administrative foothold.
CRITICAL_ACTIONS = frozenset({
    "iam:attachrolepolicy", "iam:attachuserpolicy", "iam:attachgrouppolicy",
    "iam:createrole", "iam:createpolicyversion", "iam:setdefaultpolicyversion",
    "iam:putrolepolicy", "iam:putuserpolicy", "iam:updateassumerolepolicy",
    "iam:createaccesskey", "iam:createloginprofile", "iam:addusertogroup",
    "iam:passrole", "organizations:*",
})
HIGH_ACTIONS = frozenset({
    "lambda:createfunction", "lambda:updatefunctioncode", "lambda:invokefunction",
    "lambda:addpermission", "lambda:createeventsourcemapping",
    "lambda:updatefunctionconfiguration",
    "ssm:sendcommand", "ssm:startautomationexecution",
    "ec2:runinstances", "ec2:associateiaminstanceprofile",
    "iam:createinstanceprofile", "iam:addroletoinstanceprofile",
    "cloudformation:createstack", "cloudformation:updatestack",
    "cloudformation:createchangeset", "cloudformation:executechangeset",
    "datapipeline:createpipeline", "glue:createdevendpoint", "glue:createjob",
    "codebuild:createproject", "codebuild:updateproject",
    "sagemaker:createnotebookinstance",
    "ecs:registertaskdefinition", "ecs:runtask", "ecs:createservice",
    "events:putrule", "events:puttargets", "sts:assumerole",
})

# Data-access primitives. Worth reporting, but they are not privilege escalation
# and must not be filed as such: a principal that can read an object has not
# gained a new identity, new code execution, or new permissions. Reading a secret
# is a confidentiality problem; reading a bucket is usually the job.
SECRET_ACTIONS = frozenset({
    "secretsmanager:getsecretvalue", "ssm:getparameter", "ssm:getparameters",
    "kms:decrypt",
})
OBJECT_ACTIONS = frozenset({"s3:getobject", "s3:putobject"})
DATA_ACCESS_ACTIONS = SECRET_ACTIONS | OBJECT_ACTIONS
# Data-access primitives whose *unscoped* form is a serious exposure: ``*`` means
# every secret in the account, not one object. Scoped, they stay Medium.
HIGH_IMPACT_DATA = frozenset({
    "kms:decrypt", "secretsmanager:getsecretvalue",
    "ssm:getparameter", "ssm:getparameters",
})

# Ordering used when a list has to be shown worst-first.
SEVERITY_RANK = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0, "": 0}
# The severity labels, worst first, exactly as the prompt should offer them.
# Derived from ``SEVERITY_RANK`` so the scale and the ordering cannot drift —
# and so a severity the taxonomy does not grade to can never be emitted.
SEVERITY_SCALE: Tuple[str, ...] = tuple(
    label for label, _ in sorted(SEVERITY_RANK.items(), key=lambda item: -item[1]) if label
)


# ---------------------------------------------------------------------------
# The classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IssueClass:
    """One thing this tool reports, and the rules that decide it."""

    key: str                       # stable machine identifier
    label: str                     # human name, used in prose and prompts
    report_category: str           # category string on a verified finding
    definition: str                # what the class is, in one sentence
    qualifies: str                 # when something IS this class
    excludes: str                  # what looks like it but is not
    mechanisms: Tuple[str, ...]    # how it happens — the "hows"
    detection: str                 # "simulator", "inventory", or both
    note: str = ""                 # a caveat worth carrying into the prompt


CLASSES: Tuple[IssueClass, ...] = (
    IssueClass(
        key=PRIVILEGE_ESCALATION,
        label="Privilege Escalation",
        report_category=CATEGORY_PRIVESC,
        definition=("A principal gains an identity or permission set it did not previously "
                    "hold, where the destination is strictly more privileged than the source."),
        qualifies=("the source is not already administrative, the destination confers a "
                   "capability the source lacks, and iam:SimulatePrincipalPolicy confirms "
                   "the grant"),
        excludes=("admin escalating to anything; a destination whose power is a subset of "
                  "the source's; reaching the role the source already holds; a grant the "
                  "simulator denies"),
        mechanisms=(
            "editing IAM to promote itself — CreatePolicyVersion, Put*Policy, Attach*Policy, "
            "AddUserToGroup, CreateRole, UpdateAssumeRolePolicy — or minting credentials for a "
            "more privileged user with CreateAccessKey / CreateLoginProfile",
            "passing a role to a service that runs code as it — iam:PassRole with Lambda, EC2, "
            "CloudFormation, CodeBuild, SSM, Glue, SageMaker, ECS or EventBridge",
            "assuming a role more privileged than itself through sts:AssumeRole",
            "controlling a resource that executes as a privileged role — the resource-to-role join",
        ),
        detection="simulator",
    ),
    IssueClass(
        key=LATERAL_MOVEMENT,
        label="Lateral Movement",
        report_category=CATEGORY_PRIVESC,
        definition=("A principal reaches a different principal of comparable privilege — a "
                    "horizontal pivot that grants no material increase in power."),
        qualifies=("the pivot crosses a trust boundary, OU or blast-radius silo, or reaches "
                   "data the source's purpose does not include"),
        excludes=("peer-role assumption within one team or silo; access the source is already "
                  "entitled to"),
        mechanisms=(
            "assuming a peer role in another team, OU or account",
            "running code as a peer identity via iam:PassRole and a create action",
            "a resource policy that exposes another silo's data to this principal",
        ),
        detection="simulator",
        note=("Reported under Privilege Escalation until the report grows its own lateral "
              "section."),
    ),
    IssueClass(
        key=TRUST_ABUSE,
        label="Cross-Account Trust",
        report_category="Trust",
        definition="A trust relationship inside the target account admits a principal outside it.",
        qualifies=("the trust policy lives in the target account and names an external account, "
                   "the management-account root, \"*\", or a federated identity"),
        excludes="a misconfiguration in another account — that belongs to that account",
        mechanisms=(
            "a trust policy with Principal \"*\"",
            "trust to another account's root, which delegates the decision to that account",
            "trust without sts:ExternalId — the confused-deputy shape",
            "trust to the management-account root, admitting the whole organisation",
            "a federated trust missing aud or sub conditions",
            "a resource policy (S3, KMS, SNS, SQS) granting an external account",
        ),
        detection="inventory",
    ),
    IssueClass(
        key=CREDENTIAL_EXPOSURE,
        label="Credential and Secret Exposure",
        report_category=CATEGORY_DATA_ACCESS,
        definition=("Usable credentials or secret material are readable by a principal that "
                    "should not have them."),
        qualifies="a live credential or secret is reachable by a principal outside its purpose",
        excludes="rotation cadence, unused or aged keys, and hygiene — none are reported",
        mechanisms=(
            "long-lived active access keys an attacker could take and use",
            "secrets in resource configuration — Lambda environment variables, ECS task "
            "definitions, EC2 user-data, CloudFormation parameters and outputs, CodeBuild env, "
            "container images",
            "secret stores readable broadly — SSM SecureString and Secrets Manager",
            "KMS keys whose policy allows broad kms:Decrypt",
        ),
        detection="simulator",
    ),
    IssueClass(
        key=DATA_EXPOSURE,
        label="Data Exposure",
        report_category=CATEGORY_DATA_ACCESS,
        definition="Data is reachable from outside the boundary it was meant to stay in.",
        qualifies="a bucket, snapshot, image or file system is public or cross-account reachable",
        excludes="data the caller is entitled to read",
        mechanisms=(
            "public or cross-account S3 buckets and objects",
            "public or cross-account EBS and RDS snapshots",
            "AMIs shared with all accounts",
            "public EFS or FSx file systems",
        ),
        detection="inventory",
    ),
    IssueClass(
        key=NETWORK_EXPOSURE,
        label="Network Exposure",
        report_category="Network",
        definition=("A network path reaches a sensitive service from outside the trust boundary "
                    "it was meant to sit behind."),
        qualifies="an internet-reachable path terminates on a management or data service",
        excludes="an internet-facing service that is meant to be public, with auth in front of it",
        mechanisms=(
            "security-group ingress from 0.0.0.0/0 to management or data ports — 22, 3389, 3306, "
            "5432, 6379, 9200, 27017",
            "public load balancers, API Gateway or endpoints fronting internal services",
            "publicly accessible databases behind an open security group",
        ),
        detection="inventory",
    ),
    IssueClass(
        key=COMPUTE_MISCONFIGURATION,
        label="Compute Misconfiguration",
        report_category="Compute",
        definition=("A compute resource is configured so that reaching it yields the permissions "
                    "of the role it executes as."),
        qualifies="access to the resource carries an execution role of material privilege",
        excludes="an isolated resource whose role reaches nothing of value",
        mechanisms=(
            "an internet-reachable instance with an over-permissive instance profile",
            "a privileged container or one with a host mount",
            "an ECS task role with broad permissions",
            "a public ECR image carrying secrets",
            "an EKS control plane with a public endpoint and weak authentication",
        ),
        detection="inventory",
        note=("A feeder to Privilege Escalation and Credential/Secret Exposure rather than a "
              "standalone issue — report it joined to the escalation it enables."),
    ),
)

CLASSES_BY_KEY: Dict[str, IssueClass] = {issue.key: issue for issue in CLASSES}

# Deliberately not reported. Neither is exploitable on its own, and both used to
# pad the report with compliance boilerplate. Reintroduce one only with an exploit
# to back it.
EXCLUDED_CLASSES: Tuple[str, ...] = (
    "Detection and logging coverage (CloudTrail, Config, GuardDuty)",
    "Account hygiene (key rotation, unused credentials, root MFA)",
)


# ---------------------------------------------------------------------------
# Grading and lookup
# ---------------------------------------------------------------------------


def _lower(action: str) -> str:
    return (action or "").strip().lower()


def is_data_access(action: str) -> bool:
    """True for a primitive that reads or writes data rather than gaining power."""
    return _lower(action) in DATA_ACCESS_ACTIONS


def is_secret_access(action: str) -> bool:
    """True for a data-access primitive that reaches secrets or keys."""
    return _lower(action) in SECRET_ACTIONS


def is_probe_action(action: str) -> bool:
    """True when an action (or its ``svc:*`` form) belongs in the probe set."""
    a = _lower(action)
    if not a:
        return False
    if a == "*":
        return True
    if a in PRIVESC_ACTIONS:
        return True
    return (a.split(":", 1)[0] + ":*") in PRIVESC_ACTIONS


def severity_for(action: str, *, unscoped: bool = False) -> str:
    """Grade one verified grant.

    Grading is per action and, for data access, per resource scope: the same
    ``secretsmanager:GetSecretValue`` is a serious finding against ``*`` and a
    hardening item against the one secret the role actually needs. Callers that
    know the principal's identity have already set aside the ones whose privilege
    is expected (administrators and service-linked roles).
    """
    lowered = _lower(action)
    if lowered == "*":
        return "Critical"
    if lowered in DATA_ACCESS_ACTIONS:
        return "High" if (unscoped and lowered in HIGH_IMPACT_DATA) else "Medium"
    if lowered in CRITICAL_ACTIONS or lowered in HIGH_ACTIONS:
        return "High"
    if lowered.split(":", 1)[0] + ":*" in CRITICAL_ACTIONS:
        return "Critical"
    return "Medium"


def class_of(action: str) -> Optional[IssueClass]:
    """The class a simulator-confirmed grant belongs to, or ``None``."""
    lowered = _lower(action)
    if lowered in SECRET_ACTIONS:
        return CLASSES_BY_KEY[CREDENTIAL_EXPOSURE]
    if lowered in OBJECT_ACTIONS:
        return CLASSES_BY_KEY[DATA_EXPOSURE]
    if lowered in PRIVESC_ACTIONS:
        return CLASSES_BY_KEY[PRIVILEGE_ESCALATION]
    return None


def is_escalation(action: str) -> bool:
    """True for a primitive that *gains* power, as opposed to reading data.

    This is the taxonomy's test, not the probe set's — and the distinction is the
    point. :data:`PRIVESC_ACTIONS` is the ``SimulatePrincipalPolicy`` superset and
    deliberately carries the data-access primitives too, because they are worth
    confirming. So it is the wrong predicate for anything *labelled* escalation:
    a ``s3:GetObject`` statement, probed for confirmation, is a Data Exposure, and
    calling it privesc is how the digest taught the model to file data reads as
    escalation. ``*`` qualifies, since it grants everything.
    """
    return _lower(action) == "*" or class_of(action) is CLASSES_BY_KEY[PRIVILEGE_ESCALATION]


def category_for(action: str) -> str:
    """Which section of the report a verified grant belongs in."""
    issue = class_of(action)
    return issue.report_category if issue else CATEGORY_PRIVESC


# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------


def report_categories() -> Tuple[str, ...]:
    """The distinct report-category strings a finding may carry, in class order.

    A finding's ``category`` is the report section it belongs in, so this — not
    a hand-written list — is what the prompt offers the model. ``IAM``, ``Storage``,
    ``Secrets``, ``Logging`` and ``Other`` are therefore never choices: they were
    the old vocabulary, and they are how a logging gap became a "finding".
    """
    return tuple(dict.fromkeys(issue.report_category for issue in CLASSES))


def taxonomy_text(keys: Optional[Sequence[str]] = None) -> str:
    """The classes, rendered for the model: definition, test, and the hows.

    *keys* restricts the render to the named classes, in the order given; by
    default every reported class is rendered. The excluded-classes footer is
    always appended — it is the half of the taxonomy that stops the model
    padding the report with gaps.
    """
    selected = list(CLASSES) if keys is None else [CLASSES_BY_KEY[key] for key in keys]
    lines = ["Report only these classes, and none other:", ""]
    for index, issue in enumerate(selected, start=1):
        section = ("" if issue.report_category == issue.label
                   else f" [category: {issue.report_category}]")
        lines.append(f"{index}. {issue.label}{section} — {issue.definition}")
        lines.append(f"   Qualifies when: {issue.qualifies}.")
        lines.append(f"   Not this when: {issue.excludes}.")
        lines.append(f"   How it happens: " + "; ".join(issue.mechanisms) + ".")
        if issue.note:
            lines.append(f"   Note: {issue.note}")
        lines.append("")
    lines.append("Do not report, and do not list as gaps: "
                 + "; ".join(EXCLUDED_CLASSES) + ".")
    return "\n".join(lines).rstrip() + "\n"
