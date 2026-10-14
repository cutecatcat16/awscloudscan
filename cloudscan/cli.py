"""awscloudscan command-line interface and orchestration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import List, Optional

from . import __version__, config
from .collect import collect
from .deepseek import DeepSeekClient, DeepSeekError, resolve_api_key
from .deps import Dependencies, ensure_dependencies, print_dependency_report
from .prompts import analyze, build_evidence_digest
from .report import render_report
from .resource_edges import (attach_edges_to_analysis, derive_edges, log_edge_result)
from .runner import (RunResult, cloudfox_aws_bases, find_cloudfox_output, resolve_identity,
                     run_cloudfox, run_cloudfox_pmapper, run_pmapper)
from .scope import Scope, label_escalation_sources, repair_analysis, split_findings
from .util import configure_logging, error, info, ok, slugify, step, warn
from .verify import (CONDITIONAL, CONFIRMED, DEFAULT_BATCH_SIZE, DENIED, NOT_ALLOWED,
                     attach_to_analysis, verify_evidence)

BANNER = r"""
                    _             _
 __ ___ __ _____ __| |___ _  _ __| |___ __ __ _ _ _
/ _` \ V  V (_-</ _| / _ \ || / _` (_-</ _/ _` | ' \
\__,_|\_/\_//__/\__|_\___/\_,_\__,_/__/\__\__,_|_||_|
 CloudFox + PMapper wrapper with DeepSeek analysis
"""


# Stage tokens that go into the report filename, so that a plain directory
# listing says how complete a run is without opening anything.
_STAGE_FINAL = "final_report"        # the DeepSeek review completed
_STAGE_VERIFIED = "verified_report"  # the IAM simulator confirmed the findings, no review
_STAGE_DRAFT = "draft_report"        # evidence collected only


def report_stage(*, ai_used: bool, verified: bool) -> str:
    """Name the completeness of this run for the report filename."""
    if ai_used:
        return _STAGE_FINAL
    return _STAGE_VERIFIED if verified else _STAGE_DRAFT


def refresh_latest_report(workdir: str, account: str, stage: str, report: str) -> str:
    """Point ``awscloudscan-<account>-latest-<stage>.md`` at this run's report.

    The timestamped file is the immutable artifact of one run; this one answers
    "which report is current", so nobody has to compare timestamps. Exactly one
    pointer is kept — the previous one is removed rather than left behind looking
    current — and its stage suffix says whether the review stage ran. Timestamped
    run artifacts are never touched.
    """
    prefix = f"awscloudscan-{account}-latest-"
    pointer = os.path.join(workdir, f"{prefix}{stage}.md")
    with open(pointer, "w", encoding="utf-8") as handle:
        handle.write(report)
    for name in os.listdir(workdir):
        if name.startswith(prefix) and name.endswith(".md") and name != os.path.basename(pointer):
            try:
                os.remove(os.path.join(workdir, name))
            except OSError:
                pass
    return pointer


# Reasoning is saved beside the report under these names. It is written whenever
# a stage produced thinking — not only under --show-thinking — because the live
# stdout stream scrolls away, and the copy on disk is what makes a run
# reviewable after the fact.
_THINKING_FILES = (
    ("thinking-stage1-escalation.md", "escalation_reasoning"),
    ("thinking-stage2-findings.md", "findings_reasoning"),
)


def save_thinking(workdir: str, analysis) -> List[str]:
    """Write each stage's chain of thought beside the report; return the names written."""
    written: List[str] = []
    for name, attr in _THINKING_FILES:
        text = (getattr(analysis, attr, "") or "").strip()
        if not text:
            continue
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        written.append(name)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="awscloudscan",
        description="Run CloudFox + PMapper against an AWS account, have DeepSeek review the output, "
                    "derive privilege-escalation paths, and write an in-account Markdown report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  awscloudscan -p mfa-cloud-uat-auditor-user -r ap-southeast-1
  awscloudscan -p prod-admin -r us-east-1 --modules inventory,principals,permissions,workloads
  awscloudscan -p prod-admin -r us-east-1 --offline --cloudfox-output-dir ./cloudfox-output
  awscloudscan -p prod-admin -r us-east-1 --no-ai
""",
    )
    # Not marked required at the parser level so that --install-only can run
    # without an account; main() enforces them for a real scan.
    parser.add_argument("-p", "--profile", help="AWS CLI profile name to run under")
    parser.add_argument("-r", "--region", help="AWS region to operate in")

    out = parser.add_argument_group("output")
    out.add_argument("-o", "--output-dir", default=None,
                     help=f"working directory for tool output (default: ./{config.DEFAULT_WORKDIR_NAME})")
    out.add_argument("--report-path", default=None, help="explicit path for the Markdown report")
    out.add_argument("--include-out-of-scope", action="store_true",
                     help="also render parent-organisation/other-account items inline")
    out.add_argument("--exclude-account", action="append", default=[],
                     help="extra account id to treat as out of scope (repeatable)")

    tools = parser.add_argument_group("tool control")
    tools.add_argument("--modules", default=None,
                       help="comma-separated CloudFox modules (default: a curated set)")
    tools.add_argument("--all-modules", action="store_true",
                       help="run the full CloudFox 'all-checks' module set")
    tools.add_argument("--skip-cloudfox", action="store_true", help="do not run CloudFox")
    tools.add_argument("--skip-pmapper", action="store_true", help="do not run PMapper")
    tools.add_argument("--skip-pmapper-create", action="store_true",
                       help="reuse an existing PMapper graph instead of rebuilding it")
    tools.add_argument("--keep-orgs", action="store_true",
                       help="let PMapper include AWS Organizations/SCP data (default: --ignore-orgs)")
    tools.add_argument("--pmapper-all-regions", action="store_true",
                       help="let PMapper sweep every AWS region (default: only the -r region, "
                            "which avoids unreachable-region errors)")
    tools.add_argument("--pmapper-exclude-regions", default=None,
                       help="comma-separated AWS regions to exclude from the PMapper graph "
                            "(only used with --pmapper-all-regions)")
    tools.add_argument("--pmapper-skip-cloudfox-services",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="skip the PMapper edge checkers for services CloudFox already "
                            "inventoried (ec2, lambda, cloudformation, codebuild), because "
                            "awscloudscan's resource-to-role join recovers those edges instead "
                            "(default: on; --no-pmapper-skip-cloudfox-services to let PMapper "
                            "run them itself, which is very slow on large accounts)")
    tools.add_argument("--pmapper-exclude-services", default=None,
                       help="comma-separated PMapper services to skip when building the graph "
                            "(e.g. sagemaker,codebuild); trades edge coverage for speed")
    tools.add_argument("--pmapper-include-services", default=None,
                       help="comma-separated allow-list of PMapper services "
                            "(mutually exclusive with --pmapper-exclude-services)")
    tools.add_argument("--pmapper-debug", action="store_true",
                       help="run PMapper with --debug for verbose library-level output")
    tools.add_argument("--mfa-token", default=None, help="MFA token to pass to CloudFox")
    tools.add_argument("--offline", action="store_true",
                       help="do not run the tools; analyse existing output only")
    tools.add_argument("--cloudfox-output-dir", default=None,
                       help="explicit CloudFox output directory for --offline mode")

    ai = parser.add_argument_group("DeepSeek")
    ai.add_argument("--no-ai", action="store_true", help="skip the DeepSeek review entirely")
    ai.add_argument("--deepseek-api-key", default=None, help="DeepSeek API key (else DEEPSEEK_API_KEY)")
    ai.add_argument("--deepseek-model", default=None, help=f"model (default: {config.DEEPSEEK_DEFAULT_MODEL})")
    ai.add_argument("--deepseek-base-url", default=None, help=f"API base URL (default: {config.DEEPSEEK_DEFAULT_BASE_URL})")
    ai.add_argument("--deepseek-reasoning-effort", default=None, choices=list(config.DEEPSEEK_EFFORT_LEVELS),
                    help=f"thinking effort (default: {config.DEEPSEEK_DEFAULT_REASONING_EFFORT})")
    ai.add_argument("--deepseek-thinking", default=None, choices=list(config.DEEPSEEK_THINKING_VALUES),
                    help=f"thinking mode (default: {config.DEEPSEEK_DEFAULT_THINKING})")
    ai.add_argument("--show-thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="stream the model's reasoning to stdout as it is generated "
                         "(default: on; --no-show-thinking to quiet it — a copy is saved either way)")
    ai.add_argument("--digest-budget", type=int, default=None,
                    help=f"max characters of evidence sent to DeepSeek (default: {config.MAX_EVIDENCE_CHARS})")

    verify = parser.add_argument_group("verification")
    verify.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                        help="confirm escalation candidates against the AWS IAM policy simulator "
                             "(default: on; --no-verify to skip)")
    verify.add_argument("--verify-cache", default=None,
                        help="path to the verification cache (default: <workdir>/verify-cache.json)")
    verify.add_argument("--verify-batch-size", type=int, default=None,
                        help=f"actions per SimulatePrincipalPolicy call "
                             f"(default: {DEFAULT_BATCH_SIZE}, max 100)")
    verify.add_argument("--verify-mfa-probe", action=argparse.BooleanOptionalAction, default=True,
                        help="re-simulate with and without the MFA context key so that MFA-gated "
                             "grants are reported as conditional (default: on)")
    verify.add_argument("--resource-edges", action=argparse.BooleanOptionalAction, default=True,
                        help="join CloudFox's resource->execution-role inventory against the "
                             "verified grants to recover Lambda/EC2/CloudFormation escalation "
                             "edges, replacing PMapper's four service checkers (default: on; "
                             "requires --verify)")

    misc = parser.add_argument_group("miscellaneous")
    misc.add_argument("--allow-empty", action="store_true",
                      help="write a report even when nothing was collected (default: fail loudly)")
    misc.add_argument("--no-install", action="store_true",
                      help="never auto-install missing dependencies; fail instead")
    misc.add_argument("--install-only", action="store_true",
                      help="check/install dependencies and exit")
    misc.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    misc.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    misc.add_argument("--version", action="version", version=f"awscloudscan {__version__}")
    return parser


def _resolve_modules(args: argparse.Namespace) -> List[str]:
    if args.all_modules:
        return ["all-checks"]
    if args.modules:
        return [m.strip() for m in args.modules.split(",") if m.strip()]
    return list(config.DEFAULT_MODULES)


def _discover_cloudfox_dir(root: str, profile: str) -> Optional[str]:
    """Best-effort discovery of a CloudFox output directory under *root*.

    Handles both the CloudFox v2 layout (``<root>/cloudfox-output/aws/<name>``)
    and the older ``<root>/aws/<name>`` layout.
    """
    bases = cloudfox_aws_bases(root)
    if not bases and os.path.isdir(root):
        bases = [root]
    for base in bases:
        dirs = [os.path.join(base, e) for e in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, e))]
        if not dirs:
            continue
        # Prefer one that matches the profile name, else the most recent.
        for candidate in dirs:
            if profile and profile in os.path.basename(candidate):
                return candidate
        return max(dirs, key=os.path.getmtime)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    if not args.quiet:
        print(BANNER, file=sys.stderr)

    # -- dependencies ---------------------------------------------------------
    step("checking dependencies (cloudfox, pmapper, aws cli)")
    deps: Dependencies = ensure_dependencies(auto_install=not args.no_install)
    print_dependency_report(deps)

    if args.install_only:
        if deps.missing_required:
            error("required dependencies are still missing: " +
                  ", ".join(s.name for s in deps.missing_required))
            return 1
        ok("all required dependencies are available")
        return 0

    if not args.profile or not args.region:
        parser.error("the following arguments are required: -p/--profile, -r/--region")

    if deps.missing_required:
        error("missing required dependencies: " + ", ".join(s.name for s in deps.missing_required))
        if args.no_install:
            warn("re-run without --no-install to let awscloudscan install them")
        return 1

    # -- DeepSeek key preflight -----------------------------------------------
    # Fail before the (slow) tool runs rather than after, unless the user has
    # explicitly opted out of the AI review with --no-ai.
    if not args.no_ai and not resolve_api_key(args.deepseek_api_key):
        error("no DeepSeek API key found — the AI review cannot run without one.")
        error("Provide the key in any one of these ways:")
        error("    export DEEPSEEK_API_KEY=<your-key>")
        error("    awscloudscan -p <profile> -r <region> --deepseek-api-key <your-key>")
        error("    add {\"deepseek_api_key\": \"<your-key>\"} to " + config.config_file_path())
        error("Or re-run with --no-ai to skip the DeepSeek review entirely.")
        return 2

    tool_versions = {s.name: (s.version or "unknown") for s in deps.statuses}

    workdir = os.path.abspath(args.output_dir or config.DEFAULT_WORKDIR_NAME)
    os.makedirs(workdir, exist_ok=True)
    cloudfox_outdir = os.path.join(workdir, "cloudfox")
    result = RunResult(outdir=workdir)

    # -- tools ----------------------------------------------------------------
    if args.offline:
        info("offline mode: skipping tool execution")
        explicit = args.cloudfox_output_dir or _discover_cloudfox_dir(workdir, args.profile)
        if explicit:
            result.cloudfox_output_dir = os.path.abspath(explicit)
            info(f"using existing CloudFox output: {result.cloudfox_output_dir}")
        else:
            warn("no CloudFox output directory found; continuing with PMapper data only")
        try:
            result.identity = resolve_identity(args.profile, args.region, deps.aws)
        except SystemExit:
            warn("could not resolve identity offline; account will be inferred from evidence")
    else:
        step(f"resolving identity for profile '{args.profile}'")
        result.identity = resolve_identity(args.profile, args.region, deps.aws)
        ok(f"identity: {result.identity.arn} (account {result.identity.account})")

        modules = _resolve_modules(args)
        if not args.skip_cloudfox:
            run_cloudfox(deps.cloudfox, args.profile, args.region, cloudfox_outdir, modules, result,
                         mfa_token=args.mfa_token)
            find_cloudfox_output(result, cloudfox_outdir, args.profile)
            if not result.cloudfox_output_dir:
                result.cloudfox_output_dir = _discover_cloudfox_dir(cloudfox_outdir, args.profile)
        else:
            info("skipping CloudFox (--skip-cloudfox)")

        if not args.skip_pmapper:
            pmapper_exclude = [r.strip() for r in (args.pmapper_exclude_regions or "").split(",")
                               if r.strip()]
            # Default to the engagement region so PMapper never sweeps (and
            # chokes on) unrelated, possibly-unreachable regional endpoints.
            pmapper_include = [] if args.pmapper_all_regions else [args.region]

            include_services = [s.strip() for s in (args.pmapper_include_services or "").split(",")
                                if s.strip()]
            exclude_services = [s.strip() for s in (args.pmapper_exclude_services or "").split(",")
                                if s.strip()]
            if include_services and exclude_services:
                error("--pmapper-include-services and --pmapper-exclude-services are mutually exclusive")
                return 2
            if args.pmapper_skip_cloudfox_services:
                derived = config.cloudfox_covered_pmapper_services(modules)
                for service in derived:
                    if service not in exclude_services:
                        exclude_services.append(service)
                if derived:
                    info(f"pmapper: skipping the {', '.join(derived)} edge checkers (CloudFox "
                         f"inventoried these); awscloudscan's resource-to-role join covers them "
                         f"from the verified grants instead")
                else:
                    info("--pmapper-skip-cloudfox-services: no enabled CloudFox module maps to "
                         "a PMapper service; nothing to skip")

            run_pmapper(deps.pmapper, args.profile, args.region, result,
                        workdir=workdir, ignore_orgs=not args.keep_orgs,
                        skip_create=args.skip_pmapper_create,
                        include_regions=pmapper_include,
                        exclude_regions=pmapper_exclude,
                        include_services=include_services,
                        exclude_services=exclude_services,
                        debug=args.pmapper_debug)
            if result.cloudfox_output_dir:
                run_cloudfox_pmapper(deps.cloudfox, args.profile, args.region, cloudfox_outdir, result)
        else:
            info("skipping PMapper (--skip-pmapper)")

    # -- collect --------------------------------------------------------------
    step("collecting evidence")
    evidence = collect(result, workdir)
    if not evidence.account:
        # Infer from the CloudFox output directory name (e.g. profile-123456789012).
        name = os.path.basename(result.cloudfox_output_dir or "")
        for token in reversed(name.replace("_", "-").split("-")):
            if token.isdigit() and len(token) == 12:
                evidence.account = token
                break
    if not evidence.account:
        error("could not determine the target account id")
        return 1
    info(f"target account: {evidence.account}")

    # -- evidence guard -------------------------------------------------------
    # A run that collected nothing must not masquerade as a clean bill of health.
    has_pmapper = bool(evidence.pmapper_graph.get("nodes")) or bool(evidence.pmapper_analysis)
    if not evidence.file_inventory and not has_pmapper:
        error("no evidence was collected — CloudFox produced no artifacts and no PMapper graph was found.")
        error("Nothing can be reviewed, so no report was written. Likely causes:")
        error(f"    * CloudFox output is missing under {os.path.join(workdir, 'cloudfox')}")
        error("    * the profile has insufficient read permissions (verify with `aws sts get-caller-identity`)")
        error("    * CloudFox or PMapper failed before writing output (see the log above)")
        error("Re-run with --allow-empty to force an (empty) report anyway.")
        if not args.allow_empty:
            return 1

    scope = Scope.from_evidence(evidence, extra_excluded=args.exclude_account)
    info(f"scope: target={scope.target_account} management={scope.management_account or 'unknown'} "
         f"org_members={scope.org_members or 'none'}")

    # -- verification ---------------------------------------------------------
    # AWS, not CloudFox, is the authority on whether a permission is really
    # granted. CloudFox's permissions.csv carries no resource or condition
    # scoping, so it both over- and under-reports. Every candidate it suggests
    # is put to the IAM policy simulator, which evaluates the live policy set.
    verify_result = None
    if args.verify:
        step("verifying escalation candidates against the AWS IAM policy simulator")
        try:
            verify_result = verify_evidence(
                evidence,
                profile=args.profile,
                region=args.region,
                cache_path=args.verify_cache or os.path.join(workdir, "verify-cache.json"),
                batch_size=args.verify_batch_size or DEFAULT_BATCH_SIZE,
                mfa_probe=args.verify_mfa_probe,
            )
        except Exception as exc:  # a verifier failure must not sink the scan
            warn(f"verification failed: {exc}")
        if verify_result is not None:
            summary = verify_result.summary()
            counts = summary["counts"]
            if summary["checked"]:
                ok(f"verified {summary['checked']} candidate(s): "
                   f"{counts.get(CONFIRMED, 0)} confirmed, {counts.get(CONDITIONAL, 0)} conditional, "
                   f"{counts.get(NOT_ALLOWED, 0)} not allowed, {counts.get(DENIED, 0)} denied "
                   f"({summary['api_calls']} API call(s), {summary['cache_hits']} cached)")
            else:
                warn("no candidate could be verified")
            if verify_result.degraded:
                warn("verification was degraded: the simulate API became unavailable part-way "
                     f"through, so {verify_result.skipped} candidate(s) were not checked")
    else:
        info("skipping policy verification (--no-verify)")

    # -- AI analysis ----------------------------------------------------------
    analysis = None
    ai_model: Optional[str] = None
    ai_effort: Optional[str] = None
    ai_failed = False
    if args.no_ai:
        info("skipping DeepSeek review (--no-ai)")
    else:
        try:
            client = DeepSeekClient(
                api_key=args.deepseek_api_key,
                base_url=args.deepseek_base_url,
                model=args.deepseek_model,
                reasoning_effort=args.deepseek_reasoning_effort,
                thinking=args.deepseek_thinking,
                show_thinking=args.show_thinking,
            )
            ai_model = client.cfg.model
            ai_effort = client.cfg.reasoning_effort if (client.cfg.thinking or "").lower() == "enabled" else None
            if not client.configured:
                warn("no DeepSeek API key configured — proceeding without AI review")
                ai_failed = True
            else:
                step("running DeepSeek review")
                analysis = analyze(client, evidence, scope, digest_budget=args.digest_budget,
                                   stream_reasoning=args.show_thinking)
                if analysis.used_ai:
                    ok("DeepSeek review complete")
                else:
                    ai_failed = True
                    error("every DeepSeek request failed — no AI analysis was produced.")
                    error("Check the API key and network access to " + config.DEEPSEEK_DEFAULT_BASE_URL)
        except DeepSeekError as exc:
            warn(f"DeepSeek review failed: {exc}")
            ai_failed = True
            analysis = None

    if analysis is None:
        from .prompts import Analysis, pmapper_fallback_findings

        analysis = Analysis()
        analysis.findings = pmapper_fallback_findings(evidence)
        analysis.summary = ("AI review was not performed. This report contains the CloudFox/PMapper evidence, "
                            "the deterministic PMapper findings, and the raw tool output.")

    # -- merge verified findings ----------------------------------------------
    # Verified grants lead the report: they are the only findings backed by the
    # AWS evaluation engine rather than inference over CloudFox text.
    if verify_result is not None:
        added = attach_to_analysis(analysis, verify_result)
        info(f"verification: {added} verified finding(s) merged into the analysis")

    # -- resource-to-role join ------------------------------------------------
    # This is where PMapper's lambda/ec2/cloudformation/codebuild checkers would
    # have contributed. Recovering the edges here instead is what makes skipping
    # those checkers safe: it needs no policy evaluation at all, only the
    # inventory CloudFox already wrote and the verdicts the simulator just gave.
    edge_result = None
    if args.resource_edges:
        if verify_result is None:
            warn("resource-to-role join needs verification results; skipping (run without --no-verify)")
        else:
            edge_result = derive_edges(evidence, verify_result)
            log_edge_result(edge_result)
            edges_added = attach_edges_to_analysis(analysis, edge_result)
            if edges_added:
                info(f"resource-to-role join: {edges_added} edge(s) merged into the analysis")
    else:
        info("skipping the resource-to-role join (--no-resource-edges)")

    # -- identifier hygiene ---------------------------------------------------
    # The model writes account ids from memory. Correct them before the scope
    # split, because a fabricated id is also a scope input: a truncation can
    # decide whether a finding is read as in-account or foreign.
    repairs = repair_analysis(analysis, scope)
    for wrong, right in repairs.fixed:
        warn(f"AI text cited account {wrong}, which does not exist; corrected to {right}")
    for account in repairs.suspicious:
        warn(f"AI text cites account {account}, which was not seen in the evidence")

    # -- provenance of escalation paths ---------------------------------------
    external_paths = label_escalation_sources(scope, analysis.escalation_paths)
    if external_paths:
        info(f"escalation paths: {external_paths} start outside the target account "
             f"(the in-account role they trust is still the finding)")

    # -- scope filtering ------------------------------------------------------
    in_scope, out_of_scope = split_findings(scope, analysis.findings)
    info(f"findings: {len(in_scope)} in-account, {len(out_of_scope)} excluded")

    # -- report ---------------------------------------------------------------
    digest = build_evidence_digest(evidence, scope, budget=args.digest_budget)
    with open(os.path.join(workdir, "evidence-digest.md"), "w", encoding="utf-8") as handle:
        handle.write(digest)
    with open(os.path.join(workdir, "analysis.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "summary": analysis.summary,
            "escalation_paths": analysis.escalation_paths,
            "findings": analysis.findings,
            "in_scope_findings": in_scope,
            "out_of_scope_findings": out_of_scope,
            "warnings": analysis.warnings,
            # Recorded so the artifact says for itself whether the review stage
            # ran, rather than leaving it to be inferred from the report header.
            "used_ai": bool(analysis.used_ai),
            # What the model got wrong about account ids, and how many escalation
            # paths start in an account that was not assessed.
            "identifier_audit": {
                "corrected": [{"stated": wrong, "actual": right} for wrong, right in repairs.fixed],
                "suspicious": list(repairs.suspicious),
            },
            "external_escalation_paths": external_paths,
            "verification": verify_result.summary() if verify_result is not None else None,
            "resource_edges": edge_result.summary() if edge_result is not None else None,
        }, handle, indent=2, default=str)

    thinking_files = save_thinking(workdir, analysis)

    report = render_report(
        identity=result.identity,
        scope=scope,
        analysis=analysis,
        in_scope_findings=in_scope,
        out_of_scope_findings=out_of_scope,
        evidence=evidence,
        run_result=result,
        include_out_of_scope=args.include_out_of_scope,
        tool_versions=tool_versions,
        ai_model=ai_model,
        ai_effort=ai_effort,
        ai_used=bool(analysis and analysis.used_ai),
        id_repairs=repairs,
        external_paths=external_paths,
        thinking_files=thinking_files,
    )

    account_slug = slugify(evidence.account)
    stage = report_stage(ai_used=bool(analysis and analysis.used_ai),
                         verified=verify_result is not None)
    if args.report_path:
        report_path = os.path.abspath(args.report_path)
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        report_path = os.path.join(workdir, f"awscloudscan-{account_slug}-{stamp}-{stage}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    latest = refresh_latest_report(workdir, account_slug, stage, report)

    # -- summary --------------------------------------------------------------
    ok(f"report written to {report_path}")
    info(f"current report ({stage}): {latest}")
    support = ["evidence-digest.md", "analysis.json", "pmapper-analysis.json"] + thinking_files
    info(f"supporting files in {workdir}: {', '.join(support)}")
    critical = sum(1 for f in in_scope if str(f.get("severity", "")).lower() == "critical")
    high = sum(1 for f in in_scope if str(f.get("severity", "")).lower() == "high")
    if critical or high:
        warn(f"{critical} critical / {high} high severity in-account findings")
    if ai_failed:
        error("the DeepSeek review did not run — the report was written without AI analysis.")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
