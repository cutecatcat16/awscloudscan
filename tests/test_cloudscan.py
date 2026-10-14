#!/usr/bin/env python3
"""Standalone tests for awscloudscan (run: python3 tests/test_cloudscan.py).

Covers the parts that are easy to get wrong and don't need live AWS access:
scope classification, DeepSeek response parsing, the two-stage analysis pipeline
(with a mocked client), and report rendering.
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudscan.collect import Evidence
from cloudscan.config import DEEPSEEK_DEFAULT_MODEL, DEEPSEEK_DEFAULT_REASONING_EFFORT
from cloudscan.deepseek import DeepSeekClient, DeepSeekError, parse_json_response
from cloudscan.prompts import Analysis, analyze
from cloudscan.report import render_report
from cloudscan.runner import Identity, RunLogEntry, RunResult
from cloudscan.scope import (ACCOUNT, EXTERNAL_ACCOUNT, MANAGEMENT_ACCOUNT,
                             ORGANIZATION, Scope, split_findings)
from cloudscan import config, issues, prompts, resource_edges, util, verify
from cloudscan import scope as verify_scope

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


TARGET = "136243808936"
MGMT = "676830062375"


def make_scope():
    evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1")
    evidence.csv_data["org"] = [
        {"Account": TARGET, "Name": "Mgmt Account", "ID": MGMT, "isManagementAccount?": "true"},
        {"Account": TARGET, "Name": "This account", "ID": TARGET, "isManagementAccount?": "false"},
    ]
    return Scope.from_evidence(evidence)


def test_scope():
    print("scope classification")
    scope = make_scope()
    check("management account parsed", scope.management_account == MGMT, scope.management_account)
    check("org members parsed", set(scope.org_members) == {MGMT, TARGET})

    # A role in this account trusting an external principal -> in scope.
    d = scope.classify_text(
        f"arn:aws:iam::{TARGET}:role/tlz_avm_automation trusts arn:aws:iam::232605596643:root")
    check("in-account role trusting external -> in scope", d.in_scope and d.category == ACCOUNT, d.category)

    # A finding purely about the management account -> out of scope.
    d = scope.classify_text(f"management account arn:aws:iam::{MGMT}:root has no MFA on root user")
    check("management account only -> excluded", (not d.in_scope) and d.category == MANAGEMENT_ACCOUNT, d.category)

    # Org-level construct with no account id -> out of scope.
    d = scope.classify_text("The organization's Service Control Policy does not enforce MFA for iam:CreateUser")
    check("org-level SCP -> excluded", (not d.in_scope) and d.category == ORGANIZATION, d.category)

    # Foreign account only -> out of scope.
    d = scope.classify_text("arn:aws:iam::389806457496:role/ConfigRole is over-permissioned")
    check("external account only -> excluded", (not d.in_scope) and d.category == EXTERNAL_ACCOUNT, d.category)

    # No account id -> treated as in-account.
    d = scope.classify_text("S3 bucket allows public read")
    check("no account refs -> in scope", d.in_scope and d.category == ACCOUNT, d.category)

    # LLM hint used when text is ambiguous.
    d = scope.classify_text("a vague issue", llm_hint=ORGANIZATION)
    check("llm hint honoured for vague text", (not d.in_scope) and d.category == ORGANIZATION, d.category)


def test_split_findings():
    print("finding scope split")
    scope = make_scope()
    findings = [
        {"id": "F1", "title": "Role trusts external", "evidence": f"arn:aws:iam::{TARGET}:role/x trusts arn:aws:iam::999988887777:root"},
        {"id": "F2", "title": "Mgmt root", "evidence": f"arn:aws:iam::{MGMT}:root"},
        {"id": "F3", "title": "Public bucket", "evidence": "bucket is public"},
        {"id": "F4", "title": "SCP gap", "evidence": "service control policy weak"},
    ]
    in_scope, out = split_findings(scope, findings)
    check("in-scope count", len(in_scope) == 2, [f["id"] for f in in_scope])
    check("out-of-scope count", len(out) == 2, [f["id"] for f in out])
    check("F1 in scope", any(f["id"] == "F1" for f in in_scope))
    check("F2 excluded", any(f["id"] == "F2" for f in out))
    check("decision annotated", all("_scope" in f for f in in_scope + out))


def test_json_parsing():
    print("deepseek response parsing")
    check("plain json", parse_json_response('{"a": 1}') == {"a": 1})
    check("fenced json", parse_json_response('```json\n{"a": 2}\n```') == {"a": 2})
    check("preamble json", parse_json_response('Here you go:\n{"a": 3}') == {"a": 3})
    try:
        parse_json_response("no json here")
        check("garbage raises", False)
    except DeepSeekError:
        check("garbage raises", True)


def test_deepseek_options():
    print("deepseek flash payload")
    captured = {}

    def fake_post(payload):
        captured.clear()
        captured.update(payload)
        return {"choices": [{"message": {"content": "{}"}}]}

    client = DeepSeekClient(api_key="sk-test")
    client._post = fake_post
    client.chat([{"role": "user", "content": "hi"}], json_mode=True)
    check("defaults to deepseek-flash", DEEPSEEK_DEFAULT_MODEL == "deepseek-flash")
    check("model sent is deepseek-flash", captured.get("model") == "deepseek-flash")
    check("default effort is low", captured.get("reasoning_effort") == DEEPSEEK_DEFAULT_REASONING_EFFORT == "low")
    check("thinking enabled", captured.get("thinking") == {"type": "enabled"})
    check("json mode set", captured.get("response_format") == {"type": "json_object"})
    check("temperature omitted while thinking", "temperature" not in captured)

    off = DeepSeekClient(api_key="sk-test", thinking="disabled")
    off._post = fake_post
    off.chat([{"role": "user", "content": "hi"}])
    check("thinking disabled sent", captured.get("thinking") == {"type": "disabled"})
    check("temperature restored when thinking off", captured.get("temperature") == 0.2)


def test_thinking_display():
    """--show-thinking: reasoning reaches stdout live and is saved beside the report."""
    print("thinking display: streamed capture, stdout routing, saved copy")
    import contextlib
    import io
    import tempfile
    from types import SimpleNamespace

    from cloudscan.cli import build_parser, save_thinking

    util.configure_logging()  # the display respects --quiet; make sure it is off

    def stream_events():
        # Every delta carries both keys — one null — exactly as the live API
        # sends them: the thinking arrives first, then the answer.
        return [
            {"choices": [{"delta": {"content": None, "reasoning_content": "Let me think. "}}]},
            {"choices": [{"delta": {"content": "", "reasoning_content": "Two hops to admin."}}]},
            # The answer deltas carry an explicit null reasoning field. A parser
            # that branches on *which* key is present would drop the answer here.
            {"choices": [{"delta": {"content": "{", "reasoning_content": None}}]},
            {"choices": [{"delta": {"content": '"ok": true}', "reasoning_content": None},
                         "finish_reason": "stop"}]},
            # Usage trails in an event that carries no choices at all.
            {"choices": [], "usage": {"completion_tokens": 30,
                                      "completion_tokens_details": {"reasoning_tokens": 20}}},
        ]

    client = DeepSeekClient(api_key="sk-test", show_thinking=True)
    client._iter_stream_events = lambda payload: iter(stream_events())
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = client.chat([{"role": "user", "content": "hi"}], stream=True,
                             label="stage 1 — escalation paths")

    check("streamed reasoning captured", result.reasoning == "Let me think. Two hops to admin.",
          result.reasoning)
    check("streamed answer captured despite null reasoning on its deltas",
          result.content == '{"ok": true}', result.content)
    check("usage read from the choices-less event", result.usage.get("completion_tokens") == 30)
    check("reasoning_tokens surfaced from usage", result.reasoning_tokens == 20)
    check("finish_reason captured", result.finish_reason == "stop")
    check("reasoning streamed to stdout", "Two hops to admin." in out.getvalue(), out.getvalue())
    check("stdout carries a labelled header", "stage 1 — escalation paths" in out.getvalue())
    check("reasoning is not duplicated onto stderr", "Two hops to admin." not in err.getvalue())
    check("the log line stays on stderr", "querying DeepSeek" in err.getvalue())
    check("last_result keeps this run's reasoning",
          client.last_result is not None and client.last_result.reasoning == "Let me think. Two hops to admin.")

    # A buffered (non-streaming) reply still honours the flag.
    buffered = DeepSeekClient(api_key="sk-test", show_thinking=True)
    buffered._post = lambda payload: {"choices": [
        {"message": {"content": "{}", "reasoning_content": "Buffered thought."}, "finish_reason": "stop"}]}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        buffered.chat([{"role": "user", "content": "hi"}])
    check("a buffered reply still shows its thinking", "Buffered thought." in out.getvalue(), out.getvalue())

    # Without the flag, stdout stays clean — but the reasoning is still captured.
    silent = DeepSeekClient(api_key="sk-test")
    silent._iter_stream_events = lambda payload: iter(stream_events())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        silent_result = silent.chat([{"role": "user", "content": "hi"}], stream=True)
    check("no stdout without --show-thinking", out.getvalue() == "", out.getvalue())
    check("reasoning still captured without the flag",
          silent_result.reasoning == "Let me think. Two hops to admin.")

    # Truncation is reported plainly, not as a misleading JSON parse error.
    truncated = DeepSeekClient(api_key="sk-test")
    truncated._post = lambda payload: {
        "choices": [{"message": {"content": '{"ok": true', "reasoning_content": "..."},
                     "finish_reason": "length"}],
        "usage": {"completion_tokens": 40,
                  "completion_tokens_details": {"reasoning_tokens": 35}}}
    try:
        truncated.chat_json([{"role": "user", "content": "hi"}], max_tokens=40)
        check("a truncated reply raises", False)
    except DeepSeekError as exc:
        message = str(exc)
        check("truncation names finish_reason", "finish_reason=length" in message, message)
        check("truncation names the token budget", "max_tokens=40" in message, message)
        check("truncation blames the thinking", "35 of those tokens went to thinking" in message, message)

    evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1")
    evidence.csv_data["org"] = [{"Account": TARGET, "Name": "This account",
                                 "ID": TARGET, "isManagementAccount?": "false"}]
    scope = make_scope()

    class ReasoningClient:
        """Returns no findings, but records the thinking behind each stage."""

        def __init__(self):
            self.calls = 0
            self.kwargs = []
            self.last_result = None

        def chat_json(self, messages, **kwargs):
            self.calls += 1
            self.kwargs.append(kwargs)
            self.last_result = SimpleNamespace(reasoning=f"thought {self.calls}")
            return {"escalation_paths": []} if self.calls == 1 else {"findings": []}

    rc = ReasoningClient()
    captured = analyze(rc, evidence, scope, stream_reasoning=True)
    check("streaming requested for both stages",
          all(k.get("stream") is True for k in rc.kwargs), rc.kwargs)
    check("each stage is labelled",
          "escalation" in (rc.kwargs[0].get("label") or "")
          and "findings" in (rc.kwargs[1].get("label") or ""), rc.kwargs)
    check("stage 1 thinking captured", captured.escalation_reasoning == "thought 1",
          captured.escalation_reasoning)
    check("stage 2 thinking captured", captured.findings_reasoning == "thought 2",
          captured.findings_reasoning)

    class FailingStage2(ReasoningClient):
        def chat_json(self, messages, **kwargs):
            self.calls += 1
            self.kwargs.append(kwargs)
            if self.calls == 2:
                raise DeepSeekError("stage 2 unavailable")
            self.last_result = SimpleNamespace(reasoning="only stage 1 thought")
            return {"escalation_paths": []}

    captured2 = analyze(FailingStage2(), evidence, scope, stream_reasoning=True)
    check("a failed stage keeps no borrowed thinking", captured2.findings_reasoning == "",
          captured2.findings_reasoning)
    check("the earlier stage's thinking survives a later failure",
          captured2.escalation_reasoning == "only stage 1 thought")

    with tempfile.TemporaryDirectory() as workdir:
        written = save_thinking(workdir, captured)
        check("both reasoning files are written",
              written == ["thinking-stage1-escalation.md", "thinking-stage2-findings.md"], written)
        saved = open(os.path.join(workdir, written[0]), encoding="utf-8").read()
        check("the saved copy holds the stage 1 thinking", "thought 1" in saved, saved)
    with tempfile.TemporaryDirectory() as workdir:
        check("a run without thinking writes no files", save_thinking(workdir, Analysis()) == [])

    parser = build_parser()
    check("--show-thinking is on by default",
          parser.parse_args(["-p", "prof", "-r", "us-east-1"]).show_thinking is True)
    check("--no-show-thinking turns it off",
          parser.parse_args(["-p", "prof", "-r", "us-east-1", "--no-show-thinking"]).show_thinking is False)
    check("--show-thinking is still accepted explicitly",
          parser.parse_args(["-p", "prof", "-r", "us-east-1", "--show-thinking"]).show_thinking is True)


class FakeClient:
    """Mimics DeepSeekClient.chat_json for pipeline testing."""

    def __init__(self, escalations, findings):
        self._escalations = escalations
        self._findings = findings
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages)
        # First call is escalations, second is findings.
        return self._escalations if len(self.calls) == 1 else self._findings


class BrokenClient:
    def chat_json(self, messages, **kwargs):
        raise DeepSeekError("simulated outage")


def test_analysis_pipeline():
    print("two-stage analysis pipeline")
    evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1", identity_arn=f"arn:aws:iam::{TARGET}:user/auditor")
    evidence.csv_data["org"] = [{"Account": TARGET, "Name": "This account", "ID": TARGET, "isManagementAccount?": "false"}]
    evidence.csv_data["workloads-admin"] = [{"Account": TARGET, "Role": f"arn:aws:iam::{TARGET}:role/admin", "Name": "w"}]
    scope = make_scope()

    escalations = {"summary": "Account is exposed.", "escalation_paths": [
        {"id": "ESC-01", "title": "PassRole->Lambda", "severity": "High",
         "from_principal": f"arn:aws:iam::{TARGET}:role/dev", "to_principal": f"arn:aws:iam::{TARGET}:role/admin",
         "hops": ["PassRole", "CreateFunction"], "primitives": ["iam:PassRole"], "confidence": "high"}]}
    findings = {"findings": [
        {"id": "CLOUDSCAN-01", "title": "Admin workload", "severity": "Critical", "scope": "account",
         "evidence": f"arn:aws:iam::{TARGET}:role/admin is admin"},
        {"id": "CLOUDSCAN-02", "title": "Mgmt acct issue", "severity": "Low", "scope": "management-account",
         "evidence": f"arn:aws:iam::{MGMT}:root"}]}

    client = FakeClient(escalations, findings)
    analysis = analyze(client, evidence, scope)
    check("two calls made", len(client.calls) == 2, len(client.calls))
    check("summary captured", analysis.summary == "Account is exposed.")
    check("escalation parsed", len(analysis.escalation_paths) == 1)
    check("used_ai set on success", analysis.used_ai is True)

    in_scope, out = split_findings(scope, analysis.findings)
    check("ai findings split correctly", len(in_scope) == 1 and len(out) == 1,
          (len(in_scope), len(out)))

    # Broken client must degrade to a warning, not crash.
    analysis2 = analyze(BrokenClient(), evidence, scope)
    check("broken client warns", len(analysis2.warnings) == 2, analysis2.warnings)
    check("broken client keeps summary set", bool(analysis2.summary))
    check("used_ai false when every call fails", analysis2.used_ai is False)

    # PMapper fallback when there is AI-no findings and pmapper data exists.
    evidence.pmapper_analysis = {"findings": [
        {"title": "IAM Principal Can Escalate Privileges", "severity": "High",
         "impact": "x", "description": "y", "recommendation": "z"}]}
    analysis3 = analyze(FakeClient({"escalation_paths": []}, {"findings": []}), evidence, scope)
    check("pmapper fallback used", any(f["id"].startswith("PMAPPER") for f in analysis3.findings),
          [f["id"] for f in analysis3.findings])


def test_report_render():
    print("report rendering")
    identity = Identity(account=TARGET, arn=f"arn:aws:iam::{TARGET}:user/auditor", user_id="AIDA", profile="test", region="ap-southeast-1")
    evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1", identity_arn=identity.arn)
    evidence.csv_data["org"] = [{"Account": TARGET, "Name": "This account", "ID": TARGET, "isManagementAccount?": "false"}]
    evidence.pmapper_text = "PMapper output"
    scope = make_scope()
    analysis = Analysis(
        summary="Summary text.",
        escalation_paths=[{"id": "ESC-01", "title": "Path", "severity": "High",
                           "from_principal": "a", "to_principal": "b", "hops": ["h1"], "primitives": ["iam:PassRole"],
                           "evidence": "e", "recommendation": "r", "confidence": "high"}],
        findings=[{"id": "CLOUDSCAN-01", "title": "T", "severity": "Critical", "category": "IAM",
                   "evidence": "e", "impact": "i", "recommendation": "r", "confidence": "high"}],
    )
    in_scope = [{"id": "CLOUDSCAN-01", "title": "T", "severity": "Critical", "category": "IAM",
                 "evidence": "e", "impact": "i", "recommendation": "r", "confidence": "high",
                 "_scope": {"label": "In-account", "reason": "x"}}]
    out = [{"id": "CLOUDSCAN-02", "title": "Mgmt", "severity": "Low", "_scope": {"label": "Parent", "reason": "org"}}]
    run = RunResult(identity=identity)
    run.log.append(RunLogEntry("cloudfox", "cloudfox aws ...", 0, 12.0))
    evidence.file_inventory.append(("csv/org.csv", 191))

    md = render_report(identity=identity, scope=scope, analysis=analysis,
                       in_scope_findings=in_scope, out_of_scope_findings=out,
                       evidence=evidence, run_result=run,
                       generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                       tool_versions={"cloudfox": "2.0.5", "pmapper": "1.1.5"},
                       thinking_files=["thinking-stage1-escalation.md"])
    for needle in ("# AWSCloudScan Report", "## Executive summary", "## Possible privilege-escalation paths",
                   "ESC-01", "CLOUDSCAN-01", "Appendix A", "Appendix B", "Appendix D", TARGET, MGMT):
        check(f"report contains {needle!r}", needle in md)
    check("out-of-scope item in appendix table", "CLOUDSCAN-02" in md)
    check("the report names the saved reasoning",
          "thinking-stage1-escalation.md" in md, md)


# ---------------------------------------------------------------------------
# verify.py — candidate selection, API simulation, verdicts and caching
# ---------------------------------------------------------------------------

REGION = "ap-southeast-1"
FSED_FN = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_CV_UAT"
FSED_ROLE = f"arn:aws:iam::{TARGET}:role/mfa-fsed-developer-customrole"


class FakeSimulator:
    """Stand-in for SimulatePrincipalPolicy, so verdicts can be tested offline."""

    def __init__(self, rules=None, *, default=None, context_rules=None,
                 fail_error=None, fail_times=0):
        self.rules = rules or {}
        self.default = default or {"decision": "allowed", "missing_context": [],
                                   "matched_statements": [{"p": 1}], "org_allowed": True}
        self.context_rules = context_rules or {}
        self.calls = []
        self.sleeps = []
        self.fail_error = fail_error
        self.fail_times = fail_times

    def describe(self):
        return "FakeSimulator"

    def simulate(self, arn, actions, resource, context=None):
        self.calls.append({"arn": arn, "actions": list(actions), "resource": resource,
                           "context": context})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_error
        mfa = None
        if context:
            values = (context[0].get("ContextKeyValues") or [None])
            mfa = str(values[0]).lower()
        out = {}
        for action in actions:
            entry = self.context_rules.get((arn, resource, action, mfa))
            if entry is None:
                entry = self.rules.get((arn, resource, action), self.default)
            out[action] = dict(entry, resource=resource)
        return out


def _row(principal, action, resource="*", effect="Allow", condition="No", name="", ptype="Role"):
    return {
        "Account": TARGET, "Type": ptype, "Name": name or principal,
        "Arn": f"arn:aws:iam::{TARGET}:role/{principal}", "Policy": "Managed",
        "Policy Name": "custompolicy", "Policy Arn": f"arn:aws:iam::{TARGET}:policy/custompolicy",
        "Effect": effect, "Action": action, "Resource": resource, "Condition": condition,
    }


def test_verify_candidates():
    print("verify: candidate selection")
    rows = [
        _row("r1", "lambda:UpdateFunctionCode", FSED_FN),
        _row("r1", "lambda:UpdateFunctionCode", FSED_FN),          # duplicate
        _row("r1", "iam:CreatePolicyVersion", "*", effect="Deny"),  # denied -> dropped
        _row("r1", "s3:ListBucket", "*"),                           # uninteresting -> dropped
        _row("r2", "iam:PassRole", ""),                             # empty resource -> "*"
        _row("r2", "sts:AssumeRole", "*", condition="Yes"),
    ]
    cands = verify.build_candidates(rows, account=TARGET)
    check("candidates deduplicated and filtered", len(cands) == 3, len(cands))
    keys = {c.key for c in cands}
    check("deny dropped", (f"arn:aws:iam::{TARGET}:role/r1", "iam:CreatePolicyVersion", "*") not in keys)
    check("uninteresting action dropped",
          not any(c.action == "s3:ListBucket" for c in cands))
    empty = [c for c in cands if c.action == "iam:PassRole"][0]
    check("empty resource defaults to '*'", empty.resource == "*", empty.resource)
    cond = [c for c in cands if c.action == "sts:AssumeRole"][0]
    check("condition flag parsed", cond.condition is True)
    check("resource preserved verbatim",
          any(c.resource == FSED_FN for c in cands))
    check("candidates sorted deterministically",
          cands == verify.build_candidates(list(reversed(rows)), account=TARGET))


def test_verify_grouping():
    print("verify: batching")
    cands = verify.build_candidates([
        _row("r1", "lambda:UpdateFunctionCode", FSED_FN),
        _row("r1", "lambda:CreateFunction", FSED_FN),
        _row("r1", "lambda:AddPermission", FSED_FN),
        _row("r2", "iam:PassRole", "*"),
    ])
    batches = verify.group_candidates(cands)
    check("one batch per (principal, resource)", len(batches) == 2, len(batches))
    sizes = sorted(len(b[2]) for b in batches)
    check("actions batched against a single resource", sizes == [1, 3], sizes)
    check("batch carries the principal arn", all(b[0].startswith("arn:aws:iam::") for b in batches))

    many = verify.build_candidates([
        _row("r3", f"lambda:UpdateFunctionCode", FSED_FN) for _ in range(1)
    ]) + [
        verify.Candidate(principal_arn=f"arn:aws:iam::{TARGET}:role/r3",
                         action=f"lambda:Action{i:03d}", resource=FSED_FN)
        for i in range(120)
    ]
    batches = verify.group_candidates(many, batch_size=50)
    check("batch size honoured", sorted(len(b[2]) for b in batches)[-1] == 50,
          sorted(len(b[2]) for b in batches))
    check("batch size capped at the API limit",
          all(len(b[2]) <= verify.MAX_ACTIONS_PER_CALL
              for b in verify.group_candidates(many, batch_size=9999)))


def test_verify_verdicts():
    print("verify: verdict mapping")
    cands = verify.build_candidates([
        _row("r1", "lambda:UpdateFunctionCode", FSED_FN),
        _row("r1", "iam:PassRole", "*"),
        _row("r1", "iam:CreateRole", "*"),
        _row("r1", "ec2:RunInstances", "*"),
    ])
    sim = FakeSimulator(rules={
        (f"arn:aws:iam::{TARGET}:role/r1", FSED_FN, "lambda:UpdateFunctionCode"):
            {"decision": "allowed", "missing_context": [], "matched_statements": [{"s": 1}],
             "org_allowed": True},
        (f"arn:aws:iam::{TARGET}:role/r1", "*", "iam:PassRole"):
            {"decision": "explicitDeny", "missing_context": [], "matched_statements": [],
             "org_allowed": None},
        (f"arn:aws:iam::{TARGET}:role/r1", "*", "iam:CreateRole"):
            {"decision": "implicitDeny", "missing_context": [], "matched_statements": [],
             "org_allowed": False},
        (f"arn:aws:iam::{TARGET}:role/r1", "*", "ec2:RunInstances"):
            {"decision": "allowed", "missing_context": ["aws:SourceVpc"],
             "matched_statements": [{"s": 2}], "org_allowed": True},
    })
    result = verify.verify(cands, sim, cache=None, mfa_probe=False)
    verdicts = {(d.candidate.action): d.verdict for d in result.decisions}
    check("allowed -> confirmed", verdicts["lambda:UpdateFunctionCode"] == verify.CONFIRMED,
          verdicts["lambda:UpdateFunctionCode"])
    check("explicitDeny -> denied", verdicts["iam:PassRole"] == verify.DENIED, verdicts["iam:PassRole"])
    check("implicitDeny -> not_allowed", verdicts["iam:CreateRole"] == verify.NOT_ALLOWED,
          verdicts["iam:CreateRole"])
    check("allowed + missing context -> conditional",
          verdicts["ec2:RunInstances"] == verify.CONDITIONAL, verdicts["ec2:RunInstances"])
    check("actionable() keeps confirmed + conditional", len(result.actionable()) == 2)
    check("api calls are batched, not per-candidate", result.api_calls == 2, result.api_calls)


def test_verify_resource_scoping():
    """The regression that started this: '*' and the named ARN disagree."""
    print("verify: resource scoping (the false-negative regression)")
    cands = verify.build_candidates([_row("mfa-fsed-developer-customrole",
                                          "lambda:UpdateFunctionCode", FSED_FN)])
    sim = FakeSimulator(rules={
        (FSED_ROLE, "*", "lambda:UpdateFunctionCode"):
            {"decision": "implicitDeny", "missing_context": [], "matched_statements": [],
             "org_allowed": False},
        (FSED_ROLE, FSED_FN, "lambda:UpdateFunctionCode"):
            {"decision": "allowed", "missing_context": [], "matched_statements": [{"s": 1}],
             "org_allowed": True},
    })
    result = verify.verify(cands, sim, cache=None, mfa_probe=False)
    check("named resource confirmed", result.decisions[0].verdict == verify.CONFIRMED,
          result.decisions[0].verdict)
    check("the API was asked about the named resource, never '*'",
          [c["resource"] for c in sim.calls] == [FSED_FN], [c["resource"] for c in sim.calls])
    check("org-allowed detail captured", result.decisions[0].org_allowed is True)


def test_verify_mfa_probe():
    print("verify: MFA condition probe")
    cands = verify.build_candidates([_row("r1", "sts:AssumeRole", "*")])
    arn = f"arn:aws:iam::{TARGET}:role/r1"
    rules = {(arn, "*", "sts:AssumeRole"):
             {"decision": "implicitDeny", "missing_context": ["aws:MultiFactorAuthPresent"],
              "matched_statements": [], "org_allowed": True}}
    context_rules = {
        (arn, "*", "sts:AssumeRole", "true"): {"decision": "allowed", "missing_context": [],
                                               "matched_statements": [{"s": 1}],
                                               "org_allowed": True},
        (arn, "*", "sts:AssumeRole", "false"): {"decision": "implicitDeny", "missing_context": [],
                                                "matched_statements": [], "org_allowed": True},
    }
    sim = FakeSimulator(rules=rules, context_rules=context_rules)
    result = verify.verify(cands, sim, cache=None, mfa_probe=True)
    decision = result.decisions[0]
    check("MFA-gated grant -> conditional", decision.verdict == verify.CONDITIONAL, decision.verdict)
    check("mfa_required set", decision.mfa_required is True)
    check("probe asked with and without MFA", len(sim.calls) == 3, len(sim.calls))

    context_rules[(arn, "*", "sts:AssumeRole", "false")] = {
        "decision": "allowed", "missing_context": [], "matched_statements": [{"s": 1}],
        "org_allowed": True}
    sim2 = FakeSimulator(rules=rules, context_rules=context_rules)
    result2 = verify.verify(cands, sim2, cache=None, mfa_probe=True)
    check("grant that ignores MFA -> confirmed",
          result2.decisions[0].verdict == verify.CONFIRMED, result2.decisions[0].verdict)
    check("mfa_required cleared", result2.decisions[0].mfa_required is False)


def test_verify_cache():
    print("verify: disk cache")
    arn = f"arn:aws:iam::{TARGET}:role/r1"
    a = verify._cache_key(arn, ["a:1", "a:2"], "*", None)
    b = verify._cache_key(arn, ["a:2", "a:1"], "*", None)
    c = verify._cache_key(arn, ["a:1", "a:2"], FSED_FN, None)
    d = verify._cache_key(arn, ["a:1", "a:2"], "*", verify._mfa_context(True))
    check("cache key is order independent", a == b)
    check("cache key depends on the resource", a != c)
    check("cache key depends on the context", a != d)

    import tempfile
    path = os.path.join(tempfile.mkdtemp(prefix="cloudscan-verify-"), "cache.json")
    cache = verify.SimulateCache(path)
    cache.put(a, {"x:1": {"decision": "allowed"}})
    cache.save()
    reloaded = verify.SimulateCache(path)
    check("cache survives a round trip", reloaded.get(a) is not None)
    check("cache records a hit", reloaded.hits == 1 and cache.dirty is False)

    cands = verify.build_candidates([_row("r1", "iam:PassRole", "*"),
                                     _row("r2", "iam:CreateRole", "*")])
    sim = FakeSimulator()
    first = verify.verify(cands, sim, cache=verify.SimulateCache(path), mfa_probe=False)
    check("first pass calls the API", first.api_calls == 2, first.api_calls)
    second = verify.verify(cands, sim, cache=verify.SimulateCache(path), mfa_probe=False)
    check("second pass is free", second.api_calls == 0, second.api_calls)
    check("second pass served from cache", second.cache_hits == 2, second.cache_hits)
    verdicts = lambda r: sorted((d.candidate.action, d.verdict) for d in r.decisions)
    check("cache does not change the verdict", verdicts(first) == verdicts(second))
    check("fresh decisions are not marked cached", not any(d.from_cache for d in first.decisions))
    check("replayed decisions are marked cached", all(d.from_cache for d in second.decisions))


def test_verify_degradation():
    print("verify: graceful degradation")
    cands = verify.build_candidates([_row("r1", "iam:PassRole", "*"),
                                     _row("r2", "iam:CreateRole", "*")])
    denied = verify.SimulateError("AccessDenied: not authorized", code="AccessDenied",
                                  permission=True)
    result = verify.verify(cands, FakeSimulator(fail_error=denied, fail_times=99), cache=None,
                           mfa_probe=False)
    check("AccessDenied -> degraded", result.degraded is True)
    check("AccessDenied -> nothing claimed", result.decisions == [], result.decisions)
    check("AccessDenied -> candidates marked skipped", result.skipped == len(cands), result.skipped)
    check("AccessDenied -> error recorded", bool(result.errors))

    fatal = verify.SimulateError("The config profile (x) could not be found",
                                 code="ProfileNotFound", fatal=True)
    result = verify.verify(cands, FakeSimulator(fail_error=fatal, fail_times=99), cache=None,
                           mfa_probe=False)
    check("bad profile -> degraded after one attempt", result.degraded is True)
    check("bad profile -> does not spam a warning per batch", len(result.errors) == 1,
          result.errors)

    throttled = verify.SimulateError("Throttling", code="Throttling", retryable=True)
    sim = FakeSimulator(fail_error=throttled, fail_times=2)
    sleeps = []
    result = verify.verify(cands, sim, cache=None, mfa_probe=False, sleep=sleeps.append)
    check("throttling is retried with backoff", sleeps == [1, 2], sleeps)
    check("retry recovers the run", result.api_calls == 2 and not result.degraded,
          (result.api_calls, result.degraded))


def test_verify_wildcards():
    print("verify: wildcard expansion")
    check("'*' expands to every primitive", len(verify.expand_action("*")) == len(verify._CONCRETE_PRIVESC_ACTIONS))
    check("'*' covers iam:PassRole", "iam:passrole" in verify.expand_action("*"))
    check("'iam:*' expands to the IAM primitives", "iam:passrole" in verify.expand_action("iam:*"))
    check("'s3:*' expands to data access", set(verify.expand_action("s3:*")) == {"s3:getobject", "s3:putobject"})
    check("'kms:*' expands to kms:Decrypt", verify.expand_action("kms:*") == ["kms:decrypt"])
    # The mixed-case pattern must resolve to the canonical lowercase spellings.
    # It covers both UpdateFunction actions because the escalation set includes
    # UpdateFunctionConfiguration alongside UpdateFunctionCode.
    check("prefix wildcard is matched case-insensitively",
          verify.expand_action("lambda:UpdateFunction*") ==
          ["lambda:updatefunctioncode", "lambda:updatefunctionconfiguration"])
    check("concrete action passes through unchanged",
          verify.expand_action("sts:AssumeRole") == ["sts:AssumeRole"])
    check("read-only prefix is not a candidate", verify.expand_action("ec2:Describe*") == [])
    check("read-only service prefix is not a candidate", verify.expand_action("iam:Get*") == [])
    check("organizations:List* is not a candidate (old false positive)",
          verify.expand_action("organizations:List*") == [])
    check("unprivileged action is not a candidate", verify.expand_action("s3:ListBucket") == [])
    check("is_candidate_action agrees with expand_action",
          verify.is_candidate_action("*") and not verify.is_candidate_action("ec2:Describe*"))

    # The crash this test guards: SimulatePrincipalPolicy rejects action names
    # shorter than three characters, and '*' is one character.
    check("no expanded action is shorter than the API minimum",
          all(len(a) >= 3 for a in verify.expand_action("*")))

    cands = verify.build_candidates([_row("agency_admin", "*", "*")])
    check("a global-* statement becomes concrete probes",
          len(cands) == len(verify._CONCRETE_PRIVESC_ACTIONS), len(cands))
    check("the wildcard is remembered for the report",
          all(c.source_action == "*" for c in cands))
    check("no candidate carries an unexpanded wildcard",
          not any("*" in c.action for c in cands))

    iam = verify.build_candidates([_row("backup_admin", "iam:*", "*")])
    check("'iam:*' is now detected (was silently skipped)",
          len(iam) == len(verify.expand_action("iam:*")) > 0, len(iam))
    check("concrete action keeps an empty source_action",
          verify.build_candidates([_row("r1", "sts:AssumeRole", "*")])[0].source_action == "")


def test_verify_findings():
    print("verify: findings and analysis merge")
    cands = verify.build_candidates([
        _row("r1", "iam:CreatePolicyVersion", "*"),
        _row("r1", "kms:Decrypt", "*"),
    ])
    sim = FakeSimulator()
    result = verify.verify(cands, sim, cache=None, mfa_probe=False)
    findings = result.to_findings()
    # The rollup: a principal's verified grants collapse into one finding,
    # worst first. Emitting one finding per (principal, action) is what turned
    # a single over-privileged role into 66 separate "issues".
    check("one finding per principal, not per action", len(findings) == 1, len(findings))
    finding = findings[0]
    check("the finding lists every verified action, worst first",
          finding["primitives"] == ["iam:CreatePolicyVersion", "kms:Decrypt"], finding["primitives"])
    check("the finding is titled by its worst grant",
          finding["title"].startswith("r1 can perform iam:CreatePolicyVersion"), finding["title"])
    check("admin-equivalent action is High/Critical",
          finding["severity"] in ("High", "Critical"), finding["severity"])
    check("the finding takes the category of its worst grant",
          finding["category"] == "Privilege Escalation", finding["category"])
    check("the grant count is recorded", finding["verification"]["grant_count"] == 2)
    check("finding records that it was verified", finding["verified"] is True)
    check("finding names the principal",
          finding["affected_principals"] == [f"arn:aws:iam::{TARGET}:role/r1"])
    check("finding carries the verification block",
          finding["verification"]["eval_decision"] == "allowed")
    check("finding carries a report identifier", finding["id"] == "VER-001", finding["id"])
    check("an unconditional allow is high confidence", finding["confidence"] == "high", finding["confidence"])

    # Escalation and data access are different problems and must not share a
    # severity: an unscoped kms:Decrypt reaches every key in the account (High),
    # a scoped one reaches a single key (Medium).
    scoped = verify.verify(
        verify.build_candidates([_row("r1", "kms:Decrypt", "arn:aws:kms:ap-southeast-1:123:key/abc")]),
        FakeSimulator(), cache=None, mfa_probe=False).to_findings()[0]
    check("scoped data access is Medium", scoped["severity"] == "Medium", scoped["severity"])
    check("data access is its own category", scoped["category"] == "Data Access", scoped["category"])
    unscoped = verify.verify(
        verify.build_candidates([_row("r1", "kms:Decrypt", "*")]),
        FakeSimulator(), cache=None, mfa_probe=False).to_findings()[0]
    check("unscoped data access escalates to High", unscoped["severity"] == "High", unscoped["severity"])

    arn = f"arn:aws:iam::{TARGET}:role/r1"
    gated = verify.verify(
        verify.build_candidates([_row("r1", "iam:CreatePolicyVersion", "*")]),
        FakeSimulator(rules={(arn, "*", "iam:CreatePolicyVersion"):
                             {"decision": "allowed", "missing_context": ["aws:SourceVpc"],
                              "matched_statements": [{"s": 1}], "org_allowed": True}}),
        cache=None, mfa_probe=False).to_findings()
    check("a condition-gated allow is medium confidence",
          [f["confidence"] for f in gated] == ["medium"], gated[0]["confidence"])

    wildcard = verify.verify(verify.build_candidates([_row("r1", "s3:*", "*")]), FakeSimulator(),
                             cache=None, mfa_probe=False).to_findings()
    check("wildcard origin is recorded in the finding",
          all(f["verification"]["source_actions"] == ["s3:*"] for f in wildcard), wildcard[0]["verification"])
    check("wildcard origin is quoted in the evidence",
          all("wildcard statement: s3:*" in f["evidence"] for f in wildcard))

    mixed = verify.verify(verify.build_candidates([
        _row("r1", "ssm:StartAutomationExecution", "*"),
        _row("r1", "ssm:*", "*"),
    ]), FakeSimulator(), cache=None, mfa_probe=False).to_findings()
    hits = [f for f in mixed if any(p.lower() == "ssm:startautomationexecution" for p in f["primitives"])]
    check("one action in two spellings is one finding", len(hits) == 1, len(hits))
    check("the operator's spelling is kept, not the wildcard expansion",
          "ssm:StartAutomationExecution" in hits[0]["primitives"]
          and "ssm:startautomationexecution" not in hits[0]["primitives"], hits[0]["primitives"])
    check("the wildcard origin survives the merge",
          hits[0]["verification"]["source_actions"] == ["ssm:*"], hits[0]["verification"])

    analysis = Analysis(summary="AI summary.", findings=[{"title": "Existing", "severity": "Low"}])
    added = verify.attach_to_analysis(analysis, result)
    check("verified findings merged", added == 1, added)
    check("verified findings lead the report", analysis.findings[0]["title"] != "Existing")
    check("ai findings preserved", any(f["title"] == "Existing" for f in analysis.findings))
    check("summary mentions verification", "AWS IAM evaluation engine" in analysis.summary)
    check("merge is idempotent", verify.attach_to_analysis(analysis, result) == 0)


def test_verify_batch_recovery():
    print("verify: batch-level rejection recovery")
    check("case variants collapse to one spelling",
          verify._dedupe_actions(["lambda:UpdateFunctionCode", "lambda:updatefunctioncode",
                                  "IAM:PassRole"]) == ["lambda:UpdateFunctionCode", "IAM:PassRole"],
          verify._dedupe_actions(["lambda:UpdateFunctionCode", "lambda:updatefunctioncode",
                                  "IAM:PassRole"]))

    cands = verify.build_candidates([
        _row("r1", "lambda:UpdateFunctionCode", FSED_FN),
        _row("r1", "lambda:CreateFunction", FSED_FN),
    ])
    sim = FakeSimulator(fail_times=1, fail_error=verify.SimulateError(
        "InvalidInput: require different authorization information", code="InvalidInput"))
    result = verify.verify(cands, sim, cache=None, mfa_probe=False)
    check("a rejected batch still yields verdicts",
          [d.verdict for d in result.decisions] == [verify.CONFIRMED, verify.CONFIRMED],
          [d.verdict for d in result.decisions])
    check("the rejected batch was retried in halves",
          sorted(len(c["actions"]) for c in sim.calls) == [1, 1, 2],
          [c["actions"] for c in sim.calls])
    check("the extra round-trips are counted",
          result.api_calls == 3 and not result.degraded, result.api_calls)


def test_cli_verification_flags():
    print("cli: verification flags")
    from cloudscan.cli import build_parser

    parser = build_parser()
    base = parser.parse_args(["-p", "prof", "-r", REGION])
    check("verification runs by default", base.verify is True)
    check("the MFA probe runs by default", base.verify_mfa_probe is True)
    check("cache path is left to the caller", base.verify_cache is None)

    off = parser.parse_args(["-p", "prof", "-r", REGION, "--no-verify"])
    check("--no-verify disables verification", off.verify is False)

    tuned = parser.parse_args(["-p", "prof", "-r", REGION,
                               "--verify-batch-size", "25", "--no-verify-mfa-probe"])
    check("--verify-batch-size is parsed", tuned.verify_batch_size == 25)
    check("--no-verify-mfa-probe disables the probe", tuned.verify_mfa_probe is False)


ATTACKER = f"arn:aws:iam::{TARGET}:role/attacker"


def _edge_evidence(*, lambda_rows=None, instance_rows=None, cfn_rows=None,
                   perms=None, principals=None):
    """An Evidence holding only what the join reads."""
    ev = Evidence(account=TARGET, profile="test", region=REGION)
    ev.csv_data["lambda"] = lambda_rows or []
    ev.csv_data["instances"] = instance_rows or []
    ev.csv_data["cloudformation"] = cfn_rows or []
    ev.csv_data["permissions"] = perms or []
    ev.csv_data["principals"] = principals or []
    return ev


def _lambda_row(name, role, arn=None):
    return {"Account": TARGET, "Region": REGION, "Name": name,
            "Arn": arn or f"arn:aws:lambda:{REGION}:{TARGET}:function:{name}",
            "Role": role, "IsAdminRole?": "No", "CanPrivEscToAdmin?": "No"}


def _cfn_row(name, role=""):
    return {"Account": TARGET, "Service": "cloudformation", "Region": REGION,
            "Name": name, "Role": role, "Parameters": "", "Outputs": ""}


def _edge_verify(rows, sim=None):
    """Run the same two stages cli.py does: candidates, then verdicts."""
    return verify.verify(verify.build_candidates(rows), sim or FakeSimulator(),
                         cache=None, mfa_probe=False)


def test_resource_edges_covers():
    print("resource edges: grant scoping")
    check("'*' covers everything", resource_edges._covers("*", FSED_FN))
    check("an exact match covers", resource_edges._covers(FSED_FN, FSED_FN))
    check("case does not matter", resource_edges._covers(FSED_FN.upper(), FSED_FN))
    check("a glob covers its family",
          resource_edges._covers(f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_*", FSED_FN))
    check("an unrelated ARN does not cover", not resource_edges._covers(
        f"arn:aws:lambda:{REGION}:{TARGET}:function:OTHER", FSED_FN))
    check("an empty pattern covers nothing", not resource_edges._covers("", FSED_FN))


def test_resource_edges_lambda():
    """The edge PMapper spent 2h52m failing to produce, derived in one join."""
    print("resource edges: lambda update-code -> execution role")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
        principals=[{"Account": TARGET, "Type": "Role", "Name": "attacker", "Arn": ATTACKER}],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN)]))
    check("one edge recovered", len(result.findings) == 1, len(result.findings))
    finding = result.findings[0]
    check("edge reaches the execution role", finding["to_principal"] == FSED_ROLE)
    check("edge starts from the granted principal", finding["from_principal"] == ATTACKER)
    check("edge records the shape", finding["edge"]["shape"] == "lambda-update-code")
    check("edge is marked verified", finding["verified"] is True)
    check("edge carries an EDGE id", finding["id"].startswith("EDGE-"))
    check("severity is taken from the target role's own privileges",
          finding["severity"] == "Critical", finding["severity"])
    check("the target role is disclosed in the impact",
          "Full use of" in finding["impact"], finding["impact"])
    check("the finding is labelled high confidence",
          finding["confidence"] == "high", finding["confidence"])
    check("the finding names the principal by its CloudFox label",
          "attacker can escalate" in finding["title"], finding["title"])
    check("the residual-risk caveat is stated",
          "Residual risk" in finding["evidence"])


def test_resource_edges_admin_sources_excluded():
    """An admin escalating is the baseline, not a finding — PMapper skips these too."""
    print("resource edges: administrative sources are excluded")
    admin = f"arn:aws:iam::{TARGET}:role/admin"
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
        principals=[{"Account": TARGET, "Type": "Role", "Name": "admin", "Arn": admin,
                     "IsAdminRole?": "YES", "CanPrivEscToAdmin?": "No"}],
    )
    rows = [_row("admin", "lambda:UpdateFunctionCode", FSED_FN)]
    check("the test principal ARN matches the admin flag", rows[0]["Arn"] == admin)
    result = resource_edges.derive_edges(ev, _edge_verify(rows))
    check("an admin principal originates no edge", result.findings == [])
    check("the omission is stated in the notes",
          any("administrative in their own right" in n for n in result.notes), result.notes)

    # The same grants from a non-admin principal do produce the edge.
    ev2 = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
        principals=[{"Account": TARGET, "Type": "Role", "Name": "admin", "Arn": admin,
                     "IsAdminRole?": "No"}],
    )
    result2 = resource_edges.derive_edges(ev2, _edge_verify(rows))
    check("a non-admin with the same grants does produce the edge",
          len(result2.findings) == 1, len(result2.findings))


def test_resource_edges_no_invented_grant():
    """A grant on another function must not become an edge to this role."""
    print("resource edges: no grant is invented")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode",
             f"arn:aws:lambda:{REGION}:{TARGET}:function:SOMETHING_ELSE")]))
    check("a grant on an unrelated function yields no edge", result.findings == [])
    check("the inventory still counted the function", result.resources == 1, result.resources)


def test_resource_edges_self_edge():
    print("resource edges: self-escalation is not an edge")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("mfa-fsed-developer-customrole", "lambda:UpdateFunctionCode", FSED_FN)]))
    check("a role reaching itself is skipped", result.findings == [])


def test_resource_edges_create_shape_accepts_any_scope():
    """Create-style actions let the attacker name the resource, so any scope works."""
    print("resource edges: create-style shapes accept a narrow scope")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
    )
    narrow = f"arn:aws:lambda:{REGION}:{TARGET}:function:SecretsManager*"
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "iam:PassRole", "*"),
        _row("attacker", "lambda:CreateFunction", narrow)]))
    check("a narrow CreateFunction still reaches the role",
          len(result.findings) == 1, len(result.findings))
    if result.findings:
        check("the shape is create-function",
              result.findings[0]["edge"]["shape"] == "lambda-create-function",
              result.findings[0]["edge"]["shape"])
        check("both grants are quoted",
              len(result.findings[0]["edge"]["grants"]) == 2)


def test_resource_edges_aggregation_and_notes():
    print("resource edges: aggregation and coverage notes")
    second = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_CV_SECOND"
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN),
                     _lambda_row("FSED_CV_SECOND", FSED_ROLE, second),
                     # A stack with no execution role: PMapper's checker skips these.
                     _cfn_row("some-stack", "")],
        perms=[_row("mfa-fsed-developer-customrole", "iam:AttachRolePolicy", "*")],
    )
    # A grant wide enough to cover both functions, which both run as the role.
    wildcard_fn = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_*"
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", wildcard_fn)]))
    check("two functions sharing a role collapse to one edge",
          len(result.findings) == 1, len(result.findings))
    check("both functions are listed as affected",
          len(result.findings[0]["edge"]["resources"]) == 2,
          result.findings[0]["edge"]["resources"])
    check("the report lists the role and both functions",
          len(result.findings[0]["affected_resources"]) == 3,
          result.findings[0]["affected_resources"])
    check("severity is High for a critical-primitive role",
          result.findings[0]["severity"] == "High", result.findings[0]["severity"])
    check("the empty-role stack produced no edge",
          all(f["edge"]["service"] != "CloudFormation" for f in result.findings))
    check("a note explains the CloudFormation gap",
          any("cloudformation" in n for n in result.notes), result.notes)


def test_resource_edges_bonus_grants_do_not_duplicate():
    """A shape whose needs subsume another's must not produce a second finding."""
    print("resource edges: bonus grants enrich without duplicating")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN),
        _row("attacker", "lambda:UpdateFunctionConfiguration", FSED_FN)]))
    check("code + config is one finding, not two",
          len(result.findings) == 1, len(result.findings))
    finding = result.findings[0]
    check("the required grant is the code rewrite",
          [g["action"] for g in finding["edge"]["grants"]] == ["lambda:UpdateFunctionCode"],
          finding["edge"]["grants"])
    check("the config grant is recorded as a bonus",
          [g["action"] for g in finding["edge"]["bonus_grants"]]
          == ["lambda:UpdateFunctionConfiguration"],
          finding["edge"]["bonus_grants"])
    check("the stronger capability is stated in the evidence",
          "configuration can be changed too" in finding["evidence"])
    check("primitives include the bonus action",
          "lambda:UpdateFunctionConfiguration" in finding["primitives"],
          finding["primitives"])


def test_resource_edges_code_only_has_no_bonus():
    print("resource edges: code-only escalation stands alone")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN)]))
    check("one edge", len(result.findings) == 1, len(result.findings))
    check("no bonus grants are claimed",
          result.findings[0]["edge"]["bonus_grants"] == [],
          result.findings[0]["edge"]["bonus_grants"])


def test_resource_edges_narrow_grant_aggregates_only_what_it_covers():
    """Aggregation must not claim a resource the grant cannot reach."""
    print("resource edges: aggregation respects the grant's scope")
    second = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_CV_SECOND"
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN),
                     _lambda_row("FSED_CV_SECOND", FSED_ROLE, second)],
        perms=[_row("mfa-fsed-developer-customrole", "iam:AttachRolePolicy", "*")],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN)]))
    check("one edge, naming only the function that was granted",
          len(result.findings) == 1
          and result.findings[0]["edge"]["resources"] == [FSED_FN],
          result.findings[0]["edge"]["resources"])


SECOND_ROLE = f"arn:aws:iam::{TARGET}:role/mfa-fsed-second-role"


def test_resource_edges_unbounded_collapses_per_principal_and_shape():
    """One wildcard grant must not become one finding per execution role."""
    print("resource edges: unbounded grants collapse per principal and shape")
    second = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_CV_SECOND"
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN),
                     _lambda_row("FSED_CV_SECOND", SECOND_ROLE, second)],
        # The worst role reached decides the finding's severity.
        perms=[_row("mfa-fsed-developer-customrole", "*", "*"),
               _row("mfa-fsed-second-role", "iam:AttachRolePolicy", "*")],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", "*")]))
    check("two roles reached by one wildcard produce one finding",
          len(result.findings) == 1, len(result.findings))
    finding = result.findings[0]
    check("the finding is marked unbounded", finding["edge"]["unbounded"] is True)
    check("both roles are listed on the edge record",
          {r["role"] for r in finding["edge"]["roles"]} == {FSED_ROLE, SECOND_ROLE},
          finding["edge"]["roles"])
    check("each role keeps its own resource list",
          {r["role"]: r["resources"] for r in finding["edge"]["roles"]}
          == {FSED_ROLE: [FSED_FN], SECOND_ROLE: [second]},
          finding["edge"]["roles"])
    check("the title says any execution role",
          "any Lambda execution role" in finding["title"], finding["title"])
    check("the roles replace the resource list as the affected set",
          sorted(finding["affected_resources"]) == sorted({FSED_ROLE, SECOND_ROLE}),
          finding["affected_resources"])
    check("the severity is the worst role reached",
          finding["severity"] == "Critical", finding["severity"])
    check("the evidence counts the roles reached",
          "Roles reached: 2" in finding["evidence"], finding["evidence"])
    check("the finding says the detail lives in the analysis JSON",
          "analysis JSON" in finding["evidence"] or "analysis JSON" in finding["description"])


def test_resource_edges_scoped_grants_stay_per_role():
    """A bounded grant must not be collapsed: each scope covers a different role."""
    print("resource edges: scoped grants keep one finding per role")
    second = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_CV_SECOND"
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN),
                     _lambda_row("FSED_CV_SECOND", SECOND_ROLE, second)],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN),
        _row("attacker", "lambda:UpdateFunctionCode", second)]))
    check("two scoped grants to two roles stay two findings",
          len(result.findings) == 2, len(result.findings))
    check("neither finding is unbounded",
          all(f["edge"]["unbounded"] is False for f in result.findings))
    check("each finding names its own role",
          sorted(f["to_principal"] for f in result.findings) == sorted({FSED_ROLE, SECOND_ROLE}),
          [f["to_principal"] for f in result.findings])


def test_resource_edges_unbounded_ignores_any_target_scope():
    """A create-style grant's scope never bounds which role the path reaches."""
    print("resource edges: unbounded detection ignores any-scope needs")
    second = f"arn:aws:lambda:{REGION}:{TARGET}:function:FSED_CV_SECOND"
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN),
                     _lambda_row("FSED_CV_SECOND", SECOND_ROLE, second)],
    )
    narrow = f"arn:aws:lambda:{REGION}:{TARGET}:function:SecretsManager*"
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "iam:PassRole", "*"),
        _row("attacker", "lambda:CreateFunction", narrow)]))
    check("passing any role collapses to one finding",
          len(result.findings) == 1, len(result.findings))
    check("both execution roles are listed",
          len(result.findings[0]["edge"]["roles"]) == 2,
          result.findings[0]["edge"]["roles"])


def test_resource_edges_role_listing_is_capped():
    """The collapsed finding stays readable; the full list lives in the JSON."""
    print("resource edges: the reached-role listing is capped")
    ordered = [(f"arn:aws:iam::{TARGET}:role/r{i}",
                {"severity": "High", "reached_primitive": "iam:AttachRolePolicy",
                 "resources": [f"arn:aws:lambda:{REGION}:{TARGET}:function:f{i}"]})
               for i in range(12)]
    text = resource_edges.order_roles_for_evidence(ordered, lambda a: a.rsplit("/", 1)[-1])
    check("the listing is capped", "and 4 more" in text, text)
    check("exactly the first eight roles are shown", text.count("resource(s)") == 8, text)


def test_resource_edges_conditional():
    """A context-gated grant is a medium-confidence edge, not a silent one."""
    print("resource edges: conditional grants are reported as conditional")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[],
    )
    sim = FakeSimulator(rules={
        (ATTACKER, FSED_FN, "lambda:UpdateFunctionCode"):
            {"decision": "allowed", "missing_context": ["aws:SourceVpc"],
             "matched_statements": [{"s": 1}], "org_allowed": True},
    })
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN)], sim))
    check("a context-gated grant still yields an edge",
          len(result.findings) == 1, len(result.findings))
    check("it is reported as medium confidence",
          result.findings[0]["confidence"] == "medium")
    check("the condition is stated in the evidence",
          "conditional" in result.findings[0]["evidence"])
    check("the severity falls back to Medium with no known primitive",
          result.findings[0]["severity"] == "Medium", result.findings[0]["severity"])


def test_resource_edges_attach():
    print("resource edges: merge into the analysis")
    ev = _edge_evidence(
        lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)],
        perms=[_row("mfa-fsed-developer-customrole", "*", "*")],
    )
    result = resource_edges.derive_edges(ev, _edge_verify([
        _row("attacker", "lambda:UpdateFunctionCode", FSED_FN)]))
    analysis = Analysis()
    analysis.summary = "AI summary."
    analysis.findings = [{"id": "VER-001", "title": "flat grant"}]
    added = resource_edges.attach_edges_to_analysis(analysis, result)
    check("one edge merged", added == 1, added)
    check("the edge leads the flat grants",
          analysis.findings[0]["id"].startswith("EDGE-"),
          [f["id"] for f in analysis.findings])
    check("the flat grant is preserved", len(analysis.findings) == 2)
    check("the summary records the replacement",
          "Resource-to-role join" in analysis.summary
          and "replace PMapper's" in analysis.summary, analysis.summary)
    check("merging twice does not duplicate",
          resource_edges.attach_edges_to_analysis(analysis, result) == 0)
    check("an empty result merges nothing",
          resource_edges.attach_edges_to_analysis(analysis, resource_edges.EdgeResult()) == 0)


def test_resource_edges_no_verdicts():
    print("resource edges: degrades honestly with no verdicts")
    ev = _edge_evidence(lambda_rows=[_lambda_row("FSED_CV_UAT", FSED_ROLE, FSED_FN)])
    empty = verify.VerifyResult(decisions=[], api_calls=0, cache_hits=0, errors=[], degraded=False)
    result = resource_edges.derive_edges(ev, empty)
    check("no verdicts -> no edges", result.findings == [])
    check("a note says why", any("no simulator verdicts" in n for n in result.notes),
          result.notes)


def test_report_naming():
    """A directory listing must say which report is current, and how complete it is."""
    print("cli: report naming and the latest pointer")
    import tempfile
    from cloudscan.cli import refresh_latest_report, report_stage

    check("a reviewed run ends in final_report",
          report_stage(ai_used=True, verified=True) == "final_report")
    check("review without verification is still final_report",
          report_stage(ai_used=True, verified=False) == "final_report")
    check("verification without review ends in verified_report",
          report_stage(ai_used=False, verified=True) == "verified_report")
    check("evidence alone ends in draft_report",
          report_stage(ai_used=False, verified=False) == "draft_report")
    check("every stage leaves _report at the end of the filename",
          all(stage.endswith("_report") for stage in (
              report_stage(ai_used=True, verified=True),
              report_stage(ai_used=False, verified=True),
              report_stage(ai_used=False, verified=False))))

    with tempfile.TemporaryDirectory() as workdir:
        # An immutable run artifact, which the pointer must never disturb.
        artifact = os.path.join(workdir, f"awscloudscan-{TARGET}-20260101-000000-verified_report.md")
        with open(artifact, "w", encoding="utf-8") as handle:
            handle.write("first run")
        first = refresh_latest_report(workdir, TARGET, "verified_report", "first run")
        check("the pointer is named for the account and the stage",
              os.path.basename(first) == f"awscloudscan-{TARGET}-latest-verified_report.md", first)
        check("the pointer carries the report",
              open(first, encoding="utf-8").read() == "first run")

        second = refresh_latest_report(workdir, TARGET, "final_report", "second run")
        pointers = [n for n in os.listdir(workdir) if "-latest-" in n]
        check("a later run moves the pointer to its own name",
              os.path.basename(second) == f"awscloudscan-{TARGET}-latest-final_report.md", second)
        check("only one pointer is left behind", pointers == [os.path.basename(second)], pointers)
        check("the pointer tracks the newest report",
              open(second, encoding="utf-8").read() == "second run")
        check("the timestamped run artifact survives", os.path.exists(artifact))
        check("a pointer from another account is left alone",
              os.path.basename(refresh_latest_report(workdir, "999999999999", "draft_report", "other"))
              == "awscloudscan-999999999999-latest-draft_report.md"
              and os.path.exists(second))


def test_cli_resource_edge_flags():
    print("cli: resource-edge flags")
    from cloudscan.cli import build_parser

    parser = build_parser()
    base = ["-p", "x", "-r", "us-east-1"]
    args = parser.parse_args(base)
    check("the join runs by default", args.resource_edges is True)
    check("PMapper's CloudFox-covered checkers are skipped by default",
          args.pmapper_skip_cloudfox_services is True)
    off = parser.parse_args(base + ["--no-resource-edges"])
    check("--no-resource-edges disables the join", off.resource_edges is False)
    back = parser.parse_args(base + ["--no-pmapper-skip-cloudfox-services"])
    check("the PMapper checkers can be restored",
          back.pmapper_skip_cloudfox_services is False)


def test_noise_fixes():
    """The three changes that stopped the report crying wolf."""
    print("noise fixes: suppression, identifier audit, path provenance")

    # -- suppression: administrators and service-linked roles are not findings.
    evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1")
    admin_arn = f"arn:aws:iam::{TARGET}:role/tlz_avm_automation"
    service_arn = (f"arn:aws:iam::{TARGET}:role/aws-service-role/autoscaling.amazonaws.com/"
                   "AWSServiceRoleForAutoScaling")
    plain_arn = f"arn:aws:iam::{TARGET}:role/awsbackup_admin"
    evidence.csv_data["principals"] = [
        {"Arn": admin_arn, "Name": "tlz_avm_automation", "IsAdminRole?": "YES"},
        {"Arn": service_arn, "Name": "AWSServiceRoleForAutoScaling", "IsAdminRole?": "NO"},
        {"Arn": plain_arn, "Name": "awsbackup_admin", "IsAdminRole?": "NO"},
    ]
    suppressed = verify.suppressed_principals(evidence)
    check("administrator suppressed", admin_arn in suppressed, suppressed)
    check("service-linked role suppressed", service_arn in suppressed, suppressed)
    check("ordinary role not suppressed", plain_arn not in suppressed, suppressed)

    cands = verify.build_candidates([
        _row("tlz_avm_automation", "iam:CreatePolicyVersion", "*"),
        _row("awsbackup_admin", "iam:CreatePolicyVersion", "*"),
    ])
    result = verify.verify(cands, FakeSimulator(), cache=None, mfa_probe=False)
    result.suppressed = suppressed
    findings = result.to_findings()
    principals = {p for f in findings for p in f["affected_principals"]}
    check("suppressed principal yields no finding", admin_arn not in principals, principals)
    check("the visible principal still yields a finding", plain_arn in principals, principals)
    check("suppressed grants are counted, not silently lost",
          result.summary().get("suppressed") and admin_arn in result.summary()["suppressed"],
          result.summary().get("suppressed"))
    check("summary reports the grant count set aside",
          len(result.suppressed_grants()) >= 1, len(result.suppressed_grants()))

    # -- identifier audit: the model truncates account ids from memory.
    known = {TARGET, MGMT}
    repair = verify_scope.repair_account_ids(f"the management account {MGMT[:-3]} owns the role", known)
    check("truncated id is repaired", repair.text == f"the management account {MGMT} owns the role", repair.text)
    check("the repair is recorded", repair.fixed == [(MGMT[:-3], MGMT)], repair.fixed)
    check("a correct id is left alone",
          verify_scope.repair_account_ids(f"account {MGMT}", known).fixed == [])
    check("a far-off id is not flagged",
          verify_scope.repair_account_ids("account 232605596643", known).suspicious == [],
          verify_scope.repair_account_ids("account 232605596643", known).suspicious)
    suspicious = verify_scope.repair_account_ids("account 999456789999", {"123456789012"})
    check("a close-but-wrong id is flagged, not rewritten",
          suspicious.suspicious == ["999456789999"] and suspicious.fixed == [],
          (suspicious.suspicious, suspicious.fixed))

    # -- path provenance: a path that starts abroad must say so.
    scope = make_scope()
    paths = [
        {"id": "ESC-01", "from_principal": f"arn:aws:iam::{MGMT}:root", "to_principal": "role/x"},
        {"id": "ESC-02", "from_principal": f"arn:aws:iam::{TARGET}:role/attacker", "to_principal": "role/y"},
    ]
    external = verify_scope.label_escalation_sources(scope, paths)
    check("external path counted", external == 1, external)
    check("external path labelled", paths[0]["_source_scope"]["external"] is True, paths[0]["_source_scope"])
    check("in-account path not labelled external", paths[1]["_source_scope"]["external"] is False)

    # -- and the report must surface all three.
    identity = Identity(account=TARGET, arn=f"arn:aws:iam::{TARGET}:user/auditor", user_id="AIDA",
                        profile="test", region="ap-southeast-1")
    analysis = Analysis(summary="s.", escalation_paths=paths, findings=[])
    md = render_report(identity=identity, scope=scope, analysis=analysis, in_scope_findings=[],
                       out_of_scope_findings=[], evidence=evidence, run_result=RunResult(identity=identity),
                       id_repairs=repair, external_paths=external,
                       generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    check("report states how many paths start abroad",
          "Escalation paths that start outside this account: 1" in md)
    check("report discloses the id correction", f"`{MGMT[:-3]}` → `{MGMT}`" in md)
    check("report marks the external path origin", "External (" in md)


def test_prompt_scope_fixes():
    """The prompt/digest changes that stopped data reads and admin baselines being reported."""
    print("prompt scope fixes: escalation test, caps, admin baseline, module drop")

    # -- Fix 1: "escalation" is the taxonomy's test, not the probe set's.
    check("wildcard action is escalation", issues.is_escalation("*"))
    for action in ("iam:CreatePolicyVersion", "iam:AttachRolePolicy", "lambda:CreateFunction",
                   "sts:AssumeRole", "ec2:RunInstances"):
        check(f"{action} is escalation", issues.is_escalation(action))
    for action in ("s3:GetObject", "s3:PutObject", "secretsmanager:GetSecretValue",
                   "ssm:GetParameter", "kms:Decrypt", "", "s3:ListBucket"):
        check(f"{action or '<empty>'} is not escalation", not issues.is_escalation(action))
    check("the probe set still carries the data primitives it must probe",
          all(issues.is_probe_action(a) for a in issues.DATA_ACCESS_ACTIONS),
          sorted(issues.DATA_ACCESS_ACTIONS))

    rows = [
        {"Type": "Role", "Name": "esc-role", "Policy Name": "p", "Effect": "Allow",
         "Action": "iam:CreatePolicyVersion", "Resource": "*", "Condition": ""},
        {"Type": "Role", "Name": "read-role", "Policy Name": "p", "Effect": "Allow",
         "Action": "s3:GetObject", "Resource": "arn:aws:s3:::b/*", "Condition": ""},
    ]
    summary = prompts._permissions_summary(rows, account=TARGET)
    check("escalation statement is rendered", "esc-role" in summary, summary)
    check("data-access statement is not labelled escalation", "read-role" not in summary, summary)
    check("the section says how many escalation grants it kept",
          "1 grant an escalation primitive" in summary, summary)

    # -- Fix 3: the caps, stated and enforced.
    util.configure_logging(quiet=True)
    try:
        items = [{"id": "a", "severity": "Low"}, {"id": "b", "severity": "Critical"},
                 {"id": "c", "severity": "Medium"}, {"id": "d", "severity": "Unknown"}]
        capped = prompts._worst_first(items, 2)
    finally:
        util.configure_logging()
    check("cap keeps the most severe, worst first", [i["id"] for i in capped] == ["b", "c"], capped)
    check("cap leaves a short list untouched", prompts._worst_first(items, 9) is items)

    evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1")
    evidence.csv_data["route53"] = [{"Account": TARGET, "Name": "x.", "Type": "A",
                                     "Value": "1.2.3.4", "PrivateZone": "False"}]
    scope = make_scope()
    escalation_prompt = prompts.build_escalation_messages("digest", evidence, scope)[-1]["content"]
    findings_prompt = prompts.build_findings_messages(
        "digest", {"escalation_paths": []}, evidence, scope)[-1]["content"]
    check("escalation stage states its cap",
          f"at most {prompts._MAX_ESCALATION_PATHS} paths" in escalation_prompt)
    check("findings stage states its cap",
          f"at most {prompts._MAX_FINDINGS} findings" in findings_prompt)
    check("findings stage asks for precision over recall",
          "Precision beats recall" in findings_prompt)
    check("the root-cause duplication licence is gone", "root cause" not in findings_prompt,
          findings_prompt)
    check("both stages state the admin baseline",
          "baseline" in escalation_prompt and "baseline" in findings_prompt)

    # -- Fix 4: the destination is not always an admin.
    check("to_principal describes both classes",
          "the principal or resource the path ends at" in prompts._ESCALATION_SCHEMA)
    check("to_principal no longer says 'admin/privileged'",
          "admin/privileged" not in prompts._ESCALATION_SCHEMA)

    # -- Fix 2: admin-bearing sections are labelled context.
    check("the digest keeps admin sections but calls them context",
          "never a finding" in prompts._node_summary(
              {"nodes": [{"arn": f"arn:aws:iam::{TARGET}:role/admin", "is_admin": True}]}))
    admin_evidence = Evidence(account=TARGET, profile="test", region="ap-southeast-1")
    admin_evidence.csv_data["workloads-admin"] = [
        {"Account": TARGET, "Type": "Role", "Name": "readonly", "Role": "ReadOnly"}]
    admin_evidence.csv_data["iam-simulator"] = [
        {"Account": TARGET, "Query": "AdministratorAccess", "Name": "tlz_avm_automation"}]
    admin_digest = prompts.build_evidence_digest(admin_evidence, scope)
    check("the iam-simulator admin section is labelled baseline",
          "principals that appear to be administrators (baseline; context, not findings)"
          in admin_digest, admin_digest)

    # -- Module drops: tags/route53 go, ram/sns/sqs stay.
    check("tags module dropped", "tags" not in config.DEFAULT_MODULES)
    check("route53 module dropped", "route53" not in config.DEFAULT_MODULES)
    check("ram, sns and sqs kept for cross-account trust evidence",
          {"ram", "sns", "sqs"} <= set(config.DEFAULT_MODULES))
    digest = prompts.build_evidence_digest(evidence, scope)
    check("a collected route53 artifact is not rendered", "CloudFox route53" not in digest)


def main():
    test_scope()
    test_split_findings()
    test_json_parsing()
    test_deepseek_options()
    test_thinking_display()
    test_analysis_pipeline()
    test_report_render()
    test_verify_candidates()
    test_verify_grouping()
    test_verify_verdicts()
    test_verify_resource_scoping()
    test_verify_mfa_probe()
    test_verify_cache()
    test_verify_degradation()
    test_verify_wildcards()
    test_verify_findings()
    test_verify_batch_recovery()
    test_cli_verification_flags()
    test_resource_edges_covers()
    test_resource_edges_lambda()
    test_resource_edges_admin_sources_excluded()
    test_resource_edges_no_invented_grant()
    test_resource_edges_self_edge()
    test_resource_edges_create_shape_accepts_any_scope()
    test_resource_edges_aggregation_and_notes()
    test_resource_edges_bonus_grants_do_not_duplicate()
    test_resource_edges_code_only_has_no_bonus()
    test_resource_edges_narrow_grant_aggregates_only_what_it_covers()
    test_resource_edges_unbounded_collapses_per_principal_and_shape()
    test_resource_edges_scoped_grants_stay_per_role()
    test_resource_edges_unbounded_ignores_any_target_scope()
    test_resource_edges_role_listing_is_capped()
    test_resource_edges_conditional()
    test_resource_edges_attach()
    test_resource_edges_no_verdicts()
    test_cli_resource_edge_flags()
    test_report_naming()
    test_noise_fixes()
    test_prompt_scope_fixes()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
