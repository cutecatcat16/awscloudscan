"""Turn raw CloudFox / PMapper output on disk into structured evidence."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .runner import RunResult, pmapper_storage_root
from .util import human_bytes, info, read_json, warn

# CloudFox writes artifacts into these per-tool sub-directories.
_ARTIFACT_DIRS = ("csv", "json", "table", "loot")


@dataclass
class Evidence:
    account: str = ""
    profile: str = ""
    region: str = ""
    identity_arn: str = ""
    cloudfox_dir: Optional[str] = None

    csv_data: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    json_data: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, str] = field(default_factory=dict)
    loot: Dict[str, str] = field(default_factory=dict)

    pmapper_analysis: Optional[dict] = None
    pmapper_text: str = ""
    pmapper_graph: Dict[str, Any] = field(default_factory=dict)

    file_inventory: List[Tuple[str, int]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    # -- convenience accessors ------------------------------------------------

    def csv(self, module: str) -> List[Dict[str, str]]:
        return self.csv_data.get(module, [])

    def has_csv(self, module: str) -> bool:
        return module in self.csv_data

    def json(self, module: str) -> Any:
        return self.json_data.get(module)

    def org_accounts(self) -> List[Dict[str, str]]:
        return self.csv("org")

    def management_account(self) -> Optional[str]:
        for row in self.org_accounts():
            flag = (row.get("isManagementAccount?") or row.get("isManagementAccount") or "").strip().lower()
            if flag in ("true", "yes", "1"):
                return (row.get("ID") or row.get("Account") or "").strip() or None
        return None

    def member_accounts(self) -> List[str]:
        accounts: List[str] = []
        for row in self.org_accounts():
            acct = (row.get("ID") or "").strip()
            if acct and acct not in accounts:
                accounts.append(acct)
        return accounts


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        warn(f"could not read {path}: {exc}")
        return []


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def load_cloudfox_dir(path: str, evidence: Evidence) -> None:
    """Walk a CloudFox output directory and load every artifact it contains."""
    if not path or not os.path.isdir(path):
        evidence.missing.append(f"cloudfox output dir not found: {path}")
        return
    evidence.cloudfox_dir = path

    for root, _dirs, files in os.walk(path):
        parent = os.path.basename(root)
        for filename in sorted(files):
            full = os.path.join(root, filename)
            rel = os.path.relpath(full, path)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            evidence.file_inventory.append((rel, size))

            name, ext = os.path.splitext(filename)
            module = name
            if parent == "csv" and ext == ".csv":
                evidence.csv_data[module] = _read_csv(full)
            elif parent == "json" and ext == ".json":
                payload = read_json(full)
                if payload is not None:
                    evidence.json_data[module] = payload
            elif parent == "table":
                evidence.tables[module] = _read_text(full)
            elif parent == "loot":
                evidence.loot[module] = _read_text(full)


def load_pmapper(evidence: Evidence, result: RunResult) -> None:
    """Load PMapper analysis plus the raw graph nodes/edges, when available."""
    evidence.pmapper_analysis = result.pmapper_analysis
    evidence.pmapper_text = result.pmapper_analysis_text

    storage = result.pmapper_storage or pmapper_storage_root()
    if not evidence.account:
        return
    base = os.path.join(storage, evidence.account)
    graph_dir = os.path.join(base, "graph")
    if not os.path.isdir(graph_dir):
        evidence.missing.append(f"pmapper graph not found at {graph_dir}")
        return

    nodes = read_json(os.path.join(graph_dir, "nodes.json"))
    edges = read_json(os.path.join(graph_dir, "edges.json"))
    metadata = read_json(os.path.join(base, "metadata.json"))
    evidence.pmapper_graph = {
        "nodes": nodes if isinstance(nodes, list) else [],
        "edges": edges if isinstance(edges, list) else [],
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def collect(result: RunResult, workdir: str) -> Evidence:
    """Assemble an :class:`Evidence` object from a finished run."""
    identity = result.identity
    evidence = Evidence(
        account=identity.account if identity else "",
        profile=identity.profile if identity else "",
        region=identity.region if identity else "",
        identity_arn=identity.arn if identity else "",
    )
    load_cloudfox_dir(result.cloudfox_output_dir, evidence)

    # The CloudFox pmapper module writes into the same directory as the other
    # modules; only merge it when it points somewhere distinct, to avoid loading
    # the whole evidence tree a second time.
    if (result.cloudfox_pmapper_dir
            and os.path.abspath(result.cloudfox_pmapper_dir) != os.path.abspath(result.cloudfox_output_dir or "")
            and os.path.isdir(result.cloudfox_pmapper_dir)):
        shim = Evidence()
        load_cloudfox_dir(result.cloudfox_pmapper_dir, shim)
        for key, rows in shim.csv_data.items():
            evidence.csv_data.setdefault(f"pmapper-{key}", rows)
        for key, payload in shim.json_data.items():
            evidence.json_data.setdefault(f"pmapper-{key}", payload)
        for key, text in shim.tables.items():
            evidence.tables.setdefault(f"pmapper-{key}", text)

    load_pmapper(evidence, result)

    total = sum(size for _rel, size in evidence.file_inventory)
    info(f"collected {len(evidence.file_inventory)} CloudFox artifacts ({human_bytes(total)})")
    return evidence
