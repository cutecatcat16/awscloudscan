"""Static configuration: defaults, module lists, and environment lookups."""

from __future__ import annotations

import os
from typing import Dict, List

APP_NAME = "awscloudscan"

# ---------------------------------------------------------------------------
# CloudFox
# ---------------------------------------------------------------------------

CLOUDFOX_REPO = "BishopFox/cloudfox"
# Version pinned for reproducible auto-installs. A newer local install is fine —
# awscloudscan only *installs* when CloudFox is missing or below the minimum.
CLOUDFOX_PINNED_VERSION = "2.0.5"
CLOUDFOX_MIN_VERSION = (2, 0, 0)

# Modules that carry the most signal for a single-account privesc review.
#
# Deliberately absent: ``tags`` (generic tag inventory — thousands of rows, and no
# exploitation step consumes them) and ``route53`` (DNS records plus a loot file of
# A records to *scan*; CloudFox detects no takeover, and no class in the taxonomy
# rests on it). ``ram``, ``sns`` and ``sqs`` stay: RAM enumerates cross-account
# shares, and the SNS/SQS modules carry the resource policies that
# cross-account-trust findings are built from.
DEFAULT_MODULES: List[str] = [
    "inventory",
    "orgs",
    "principals",
    "permissions",
    "iam-simulator",
    "access-keys",
    "role-trusts",
    "resource-trusts",
    "workloads",
    "secrets",
    "buckets",
    "databases",
    "endpoints",
    "env-vars",
    "ecs-tasks",
    "lambda",
    "instances",
    "network-ports",
    "ram",
    "cloudformation",
    "sns",
    "sqs",
    "api-gw",
    "eks",
    "ecr",
    "filesystems",
    "codebuild",
    "elastic-network-interfaces",
]

# Modules that are slow / very verbose. Used to warn the user and to decide
# which artifacts are worth sending to the LLM in full.
HEAVY_MODULES = {
    "permissions",
    "iam-simulator",
    "network-ports",
    "instances",
    "ecs-tasks",
    "env-vars",
    "elastic-network-interfaces",
}

# ---------------------------------------------------------------------------
# PMapper
# ---------------------------------------------------------------------------

PMAPPER_PIP_PACKAGE = "principalmapper"

# CloudFox module -> the PMapper edge checker that sweeps the same AWS service.
# Used by --pmapper-skip-cloudfox-services so PMapper does not re-inventory a
# service CloudFox has already covered.
#
# IAM and STS are deliberately absent. PMapper's IAM edge checker is the core of
# the graph and consumes most of its runtime; CloudFox's IAM modules list policy
# rather than compute escalation edges, so the two are not interchangeable.
CLOUDFOX_MODULE_TO_PMAPPER_SERVICE: Dict[str, str] = {
    "instances": "ec2",
    "network-ports": "ec2",
    "elastic-network-interfaces": "ec2",
    "lambda": "lambda",
    "cloudformation": "cloudformation",
    "codebuild": "codebuild",
}
# Services that must never be auto-excluded, whatever the module list says.
PMAPPER_PROTECTED_SERVICES = ("iam", "sts")


def cloudfox_covered_pmapper_services(modules: List[str]) -> List[str]:
    """PMapper services already inventoried by the given CloudFox modules.

    Returns the deduplicated service names, excluding anything in
    :data:`PMAPPER_PROTECTED_SERVICES`. ``all-checks`` (CloudFox's "run every
    module") is expanded to :data:`DEFAULT_MODULES` before mapping.
    """
    if "all-checks" in modules:
        modules = DEFAULT_MODULES
    services: List[str] = []
    for module in modules:
        service = CLOUDFOX_MODULE_TO_PMAPPER_SERVICE.get(module)
        if service and service not in PMAPPER_PROTECTED_SERVICES and service not in services:
            services.append(service)
    return services

# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible chat-completions API)
# ---------------------------------------------------------------------------

DEEPSEEK_DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# DeepSeek-V4.1-Flash: 1M context, native vision, cheap and fast. awscloudscan pins
# thinking mode to the lowest effort ("low") since IAM review is a summarisation
# task that does not need a long chain of thought.
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_DEFAULT_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "low")
DEEPSEEK_DEFAULT_THINKING = os.getenv("DEEPSEEK_THINKING", "enabled")
# Valid values per https://api-docs.deepseek.com/guides/thinking_mode
DEEPSEEK_EFFORT_LEVELS = ("low", "high", "max")
DEEPSEEK_THINKING_VALUES = ("enabled", "disabled")
DEEPSEEK_KEY_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_KEY",
    "AWSSCLOUDSCAN_DEEPSEEK_API_KEY",
)
DEEPSEEK_TIMEOUT = int(os.getenv("AWSSCLOUDSCAN_DEEPSEEK_TIMEOUT", "300"))
DEEPSEEK_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Evidence budgets (keep LLM requests inside the context window)
# ---------------------------------------------------------------------------

# Total characters of evidence handed to a single LLM call.
#
# deepseek-flash has a 1M-token context (~4M chars). The old 180k default was
# ~5% of that — we were starving the model of the very evidence it needs to
# reason about escalation paths. 600k chars (~150k tokens, ~15% of the window)
# leaves plenty of headroom for the prompt, the schemas and long completions
# while giving the analyst look at nearly the whole account. Override with
# AWSSCLOUDSCAN_MAX_EVIDENCE_CHARS or --digest-budget.
MAX_EVIDENCE_CHARS = int(os.getenv("AWSSCLOUDSCAN_MAX_EVIDENCE_CHARS", "600000"))
# Rows sampled from each tabular artifact when building the LLM digest. Rows are
# signal-ranked before this cap is applied, so the cap keeps the *interesting*
# rows rather than the first N in file order.
MAX_ROWS_PER_TABLE = 120
# Cap on the size of any single free-text field preserved verbatim.
MAX_FIELD_CHARS = 4000

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DEFAULT_WORKDIR_NAME = "awscloudscan-output"


def config_file_path() -> str:
    """Return the path to awscloudscan's optional user config file."""
    base = os.getenv("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, APP_NAME, "config.json")
