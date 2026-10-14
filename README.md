# awscloudscan

A Python 3 wrapper around **CloudFox** and **PMapper** that runs both against a single
AWS account, has **DeepSeek** review the combined output, derives the possible
privilege-escalation paths, and writes a **Markdown report scoped to that account** —
excluding anything that belongs to a parent AWS Organization.

```bash
awscloudscan -p <profile-name> -r <region>
```

## What it does

1. **Checks dependencies** — and auto-installs anything missing (CloudFox, PMapper, the
   AWS CLI, the `requests` library), so it works on a fresh machine.
2. **Resolves the identity** behind `-p <profile>` (`sts:GetCallerIdentity`) to learn the
   account id and caller ARN.
3. **Runs CloudFox** — a curated set of modules by default (inventory, principals,
   permissions, iam-simulator, role-trusts, resource-trusts, workloads, secrets, buckets,
   databases, endpoints, env-vars, ECS/Lambda/EC2, network-ports, RAM, CloudFormation, …),
   or CloudFox's complete set with `--all-modules` (see [Full run](#full-run)). Output
   lands in `awscloudscan-output/cloudfox/aws/<profile>-<account>/`.
4. **Runs PMapper** — `graph create` (with `--ignore-orgs` by default, so SCP/org data is
   not mixed in) followed by `analysis`, plus CloudFox's own `pmapper` module to turn the
   graph into a privilege-escalation graph.
5. **Asks DeepSeek to review** the collected evidence in two stages: first the possible
   **escalation paths**, then the **findings**. Both stages are capped (20 paths, 40
   findings, worst first) and told that administrative principals are the baseline —
   context, not findings — because the tool's failure mode is volume, not omission. The
   evidence digest handed to
   the model always opens with an index of everything that was collected, and it is
   signal-ranked before any per-table cap is applied (wildcards, external/`*` trust
   principals, admin-looking grants and privesc actions rise to the top). If a section
   still cannot fit the budget it is named in an `### Analysis coverage` footer as
   `[dropped: <section>, N rows]` or `[truncated: <section>]`, so the model treats missing
   detail as *unknown* rather than clean.
6. **Verifies the candidates with AWS** — every candidate grant is put to
   `iam:SimulatePrincipalPolicy`, so a finding is backed by the IAM evaluation engine
   rather than inferred from policy text. (The permission is available on
   `IAMReadOnlyAccess`/`SecurityAudit`, so a read-only identity can still verify.) Verified
   grants are rolled up per principal and split into privilege-escalation and data-access
   findings — see [What counts as a finding](#what-counts-as-a-finding).
7. **Filters to the account boundary** — findings that concern only the management
   account, organisation-level constructs, or other accounts are moved to an appendix and
   kept out of the main results. Escalation paths that start outside the account are
   labelled, not dropped.
8. **Writes the report** as Markdown, plus machine-readable JSON and the raw evidence
   digest.

## Installation

Nothing needs to be installed ahead of time — `awscloudscan` provisions its own tools. To be
safe, install it up front:

```bash
cd cloudscan
python3 -m pip install --user principalmapper   # PMapper
# CloudFox is installed automatically from its GitHub release if missing.
```

To get the `awscloudscan` command on your PATH:

```bash
./install.sh
```

That symlinks `~/.local/bin/awscloudscan` to `run.py` in this checkout and verifies it by
running it from `/tmp`. **Re-run `./install.sh` after moving or renaming the checkout** —
the command works from any directory, but a symlink still has to point at the directory
the checkout is in now.

Note the installer deliberately does *not* use `pip install -e .`. An editable install
bakes the absolute source path into `site-packages`, so moving the checkout afterwards
leaves an `awscloudscan` command that fails with `ModuleNotFoundError: No module named
'cloudscan'` from every directory. The symlink carries no recorded path and cannot go
stale that way. `./install.sh` also removes any leftover `cloudscan` launcher from a
previous install, so exactly one command — `awscloudscan` — remains on your PATH.

Or run it in place without installing:

```bash
python3 run.py -p <profile> -r <region>
./awscloudscan.sh -p <profile> -r <region>
```

### DeepSeek API key

The AI review needs a DeepSeek key, supplied in any of these ways (first match wins):

```bash
export DEEPSEEK_API_KEY=sk-...          # recommended
awscloudscan -p prof -r region --deepseek-api-key sk-...
# or add {"deepseek_api_key": "sk-..."} to ~/.config/awscloudscan/config.json
```

If no key is present, `awscloudscan` **exits with status 2 before running any tools** and
prints how to supply one:

```
[-] no DeepSeek API key found — the AI review cannot run without one.
[-]     export DEEPSEEK_API_KEY=<your-key>
```

Pass `--no-ai` if you deliberately want to skip the AI review; that path needs no key.
The same failure applies if the key is rejected at runtime (e.g. a 401): the report is
still written, but the run exits 2 and the report records `DeepSeek review | not performed`.

Model and endpoint default to **`deepseek-flash`** (DeepSeek-V4.1-Flash — 1M context,
native vision, cheap and fast) and `https://api.deepseek.com`. Thinking mode is on at
**effort `low`**, since IAM review is a summarisation task. Override any of it:

```bash
awscloudscan -p prof -r region --deepseek-model deepseek-v4-pro --deepseek-reasoning-effort high
export DEEPSEEK_MODEL=deepseek-flash DEEPSEEK_REASONING_EFFORT=low DEEPSEEK_THINKING=enabled
```

Note that thinking mode ignores `temperature`, so awscloudscan omits it while thinking is
enabled. Valid efforts are `low`, `high`, `max`; thinking is `enabled` or `disabled`.

### Seeing what the model is thinking

Reasoning is **shown by default**. awscloudscan sends the model's reasoning to **stdout**,
while every log line goes to **stderr**. That split lets you watch the thinking scroll by
and still keep it separate from the run log:

```bash
awscloudscan -p prof -r region                    # reasoning streams to your terminal
awscloudscan -p prof -r region > think.txt        # ...or capture it on its own
awscloudscan -p prof -r region --no-show-thinking # quiet it
```

Each stage's chain of thought is streamed as it is generated and labelled with the stage
it belongs to. A copy is also written beside the report (`thinking-stage1-escalation.md`
and `thinking-stage2-findings.md`) whether or not you display it, so a run stays
reviewable after the fact — the live stream scrolls away, the file does not. `-q` quietens
the stream as well. Thinking itself is on by default; set `--deepseek-thinking disabled`
to stop it at the source.

## Usage

```
awscloudscan -p PROFILE -r REGION [options]
```

| Option | Description |
| --- | --- |
| `-p, --profile` | AWS CLI profile to run under (required) |
| `-r, --region` | AWS region to operate in (required); also the region PMapper is restricted to |
| `-o, --output-dir` | Working directory (default `./awscloudscan-output`); passed to CloudFox as `--outdir` |
| `--report-path` | Explicit path for the Markdown report |
| `--modules` | Comma-separated CloudFox modules (default: curated set) |
| `--all-modules` | **Full run.** Run CloudFox's complete `all-checks` set instead of the curated default. Provides `permissions`, so `iam:SimulatePrincipalPolicy` verification has candidates to check |
| `--skip-cloudfox` / `--skip-pmapper` | Skip one of the tools |
| `--skip-pmapper-create` | Reuse an existing PMapper graph |
| `--pmapper-all-regions` | Let PMapper sweep every region (default: only the `-r` region) |
| `--pmapper-exclude-regions` | Extra regions to exclude (used with `--pmapper-all-regions`) |
| `--pmapper-skip-cloudfox-services` | **On by default.** Skip PMapper's edge checkers for services CloudFox already inventoried (`ec2`, `lambda`, `cloudformation`, `codebuild`); the resource-to-role join covers them instead. Restore with `--no-pmapper-skip-cloudfox-services` |
| `--resource-edges` / `--no-resource-edges` | **On by default.** Join CloudFox's resource→execution-role inventory against the AWS-verified grants to recover Lambda/EC2/CloudFormation escalation edges. Requires `--verify` |
| `--pmapper-exclude-services` | Comma-separated PMapper services to skip (e.g. `sagemaker,codebuild`) |
| `--pmapper-include-services` | Comma-separated PMapper service allow-list (mutually exclusive with the exclude list) |
| `--pmapper-debug` | Run PMapper with `--debug` for verbose library-level output |
| `--allow-empty` | Write a report even when nothing was collected (default: fail) |
| `--keep-orgs` | Let PMapper include Organizations/SCP data (default: ignore) |
| `--mfa-token` | MFA token forwarded to CloudFox |
| `--offline` | Do not run the tools; analyse existing output only |
| `--cloudfox-output-dir` | CloudFox output directory to use with `--offline` |
| `--no-ai` | Skip the DeepSeek review |
| `--deepseek-api-key` / `--deepseek-model` / `--deepseek-base-url` | DeepSeek overrides |
| `--deepseek-reasoning-effort` | Thinking effort: `low`, `high`, `max` (default `low`) |
| `--deepseek-thinking` | Thinking mode: `enabled` or `disabled` (default `enabled`) |
| `--show-thinking` / `--no-show-thinking` | Stream the model's reasoning to stdout as it is generated (**on by default**); the saved copy is written either way |
| `--digest-budget` | Max characters of evidence sent to DeepSeek (default 600000) |
| `--exclude-account` | Extra account id to treat as out of scope (repeatable) |
| `--include-out-of-scope` | Also render excluded items inline |
| `--no-install` | Never auto-install; fail if a tool is missing |
| `--install-only` | Check/install dependencies and exit |
| `-v` / `-q` | Verbose / quiet logging |

### Examples

```bash
# Standard run
awscloudscan -p mfa-cloud-uat-auditor-user -r ap-southeast-1

# Full run — every module CloudFox's all-checks covers
awscloudscan -p prod-admin -r us-east-1 --all-modules

# Quick, IAM-focused run
awscloudscan -p prod-admin -r us-east-1 --modules inventory,principals,permissions,iam-simulator,role-trusts,workloads

# Re-analyse previous CloudFox output without touching AWS
awscloudscan -p prod-admin -r us-east-1 --offline --cloudfox-output-dir ./cloudfox-output/aws/prod-admin-123456789012

# No AI, deterministic report only
awscloudscan -p prod-admin -r us-east-1 --no-ai
```

### Full run

The default module set is a curated list, not everything CloudFox can do. For a full scan
of the account, pass `--all-modules`, which hands CloudFox its own `all-checks` set:

```bash
awscloudscan -p PROFILE -r REGION --all-modules
```

This is the run to use when you want the whole picture rather than a targeted pass.
`all-checks` includes `permissions`, the `permissions.csv` table that feeds the
`iam:SimulatePrincipalPolicy` verification step — without it every candidate grant stays
unverified. It also adds `tags` and `route53` on top of the curated default.

Two modules are **not** part of `all-checks`: `sns` and `resource-trusts`, and CloudFox
skips `outbound-assumed-roles` inside it. A full run therefore does not cover
cross-account SNS or resource-policy trust edges.

## Account scope: what is included and what is not

The report answers one question — *what is wrong inside this account?* A finding stays in
the main report when it references the **target account** (its principals, roles, policies,
or resources). It is moved to **Appendix A** when it concerns:

- the **management account** of the parent organisation, or
- **organisation-level** constructs (SCPs, org trails, `organizations:*`), or
- **other AWS accounts** (external trusts with no in-account foothold).

The classification is deterministic — every account id in a finding's text is matched
against the target account and the management account discovered from CloudFox's `org`
module. The model's own scope guess is only used as a tie-breaker when no account id is
present. A role **in this account** that trusts an external principal is kept as an
in-account finding, because the risky object lives in this account.

Two honesty checks back this claim up:

- **Escalation-path origins are labelled.** A path that starts at a principal outside
  this account — the management account's root, or a foreign account named in a trust
  policy — is still reported, because the role it reaches is here, but its row carries an
  **Origin** of `External (<account>)` rather than `In-account`. Without that label, "the
  management account escalates into this account" reads as a contradiction of the scope
  rule. The executive summary counts these paths separately.
- **Account ids the model invented are corrected.** A model writing an id from memory
  truncates it (on one engagement the summary said `676830375`; the real id was
  `676830062375`). Because an id is also a scope *input*, a truncation can decide whether
  a finding is read as in-account or foreign. Every id in the model's prose is checked
  against the ids actually seen in the evidence: a near miss is snapped to the real id and
  reported in the run log and the report, and a plausible-but-unknown id is flagged
  without being changed. Both are recorded under `identifier_audit` in `analysis.json`.

## What counts as a finding

Findings that come out of the verifier are **per principal**, not per action. If one role
can call 66 privileged actions, that is one over-privileged role to fix, not 66 issues —
so the grants are collapsed into a single finding led by the worst action, with the full
list kept under `verification.actions` in `analysis.json`. Without this the report is
weighted by report-writing, not by risk: on one engagement 102 of 141 principals held more
than one grant and a single role produced 66 findings.

The verifier also keeps two different problems apart, because merging them produced a wall
of false alarms:

| Category | Examples | Severity |
| --- | --- | --- |
| **Privilege Escalation** | `iam:CreatePolicyVersion`, `iam:PassRole`, `sts:AssumeRole`, `lambda:UpdateFunctionCode`, `*` | High, or Critical for `*` and admin-equivalent primitives |
| **Data Access** | `s3:GetObject`/`PutObject`, `kms:Decrypt`, `secretsmanager:GetSecretValue`, `ssm:GetParameter*` | Medium when scoped to a resource; **High** when allowed on `*`, where it reaches every secret or key in the account |

Reading one object is not privilege escalation, and it does not belong in the same
severity band as the ability to grant yourself admin. Only the unscoped forms are ranked
as a serious exposure.

Finally, **two classes of principal never originate a finding**, because their privilege is
the account's intended design rather than a weakness:

- **administrative principals** — CloudFox's `IsAdminRole?`. "An administrator can create
  roles" is a tautology. Four such roles held 259 of 764 findings on one engagement.
- **AWS service-linked roles** (`AWSServiceRoleFor*`, `/aws-service-role/…`) — created and
  assumed by an AWS service, not by a person; nothing an attacker holds can act as them.

Their grants are not silently dropped: the count and the reason reach `analysis.json` and
the run log, so a reviewer can see exactly what was set aside and why. On one engagement
this, the category split, and the per-principal rollup took **764 findings down to 145**.

## Output

Inside the working directory you get:

| File | Contents |
| --- | --- |
| `awscloudscan-<account>-<timestamp>-<stage>.md` | The Markdown report for one run (immutable artifact) |
| `awscloudscan-<account>-latest-<stage>.md` | Pointer to the current report |
| `analysis.json` | Structured AI analysis: summary, escalation paths, findings (split by scope), plus the identifier audit and suppression record |
| `evidence-digest.md` | Exactly what was sent to DeepSeek |
| `pmapper-analysis.json` / `.txt` | Raw PMapper analysis |
| `thinking-stage1-escalation.md` / `thinking-stage2-findings.md` | The model's chain of thought for each stage (written whenever thinking produced any) |
| `cloudfox/cloudfox-output/aws/<profile>-<account>/` | Raw CloudFox `csv`, `json`, `table`, `loot` output |

The report contains an executive summary, a ranked table of escalation paths (each with an
**Origin** column marking whether it starts inside or outside this account), the in-account
findings, and appendices for excluded items, PMapper's native findings, the collection log,
and the list of collected artifacts. The executive summary also states how many escalation
paths start abroad and how many account ids in the model's prose had to be corrected.

Every report filename ends in a **stage** that says how complete the run was, so you can
tell at a glance whether the DeepSeek review actually ran:

| Stage | Meaning |
| --- | --- |
| `final_report` | Evidence collected, findings verified with `iam:SimulatePrincipalPolicy`, and the DeepSeek review completed |
| `verified_report` | Findings verified, but the AI review was skipped (`--no-ai`) or did not run |
| `draft_report` | Evidence collected only — nothing verified |

A timestamped file is the immutable record of one run and is never overwritten. Alongside
it awscloudscan keeps exactly one `awscloudscan-<account>-latest-<stage>.md` pointer that always
names the current report, so nobody has to compare timestamps to find it. Writing a new
pointer removes the previous one — it is deleted rather than left behind looking current —
while the timestamped run artifacts are left untouched.

## How missing tools are handled

At startup awscloudscan verifies `python3`, `requests`, `cloudfox`, `pmapper`, and `aws`. For
anything missing it installs it:

| Tool | Strategy |
| --- | --- |
| CloudFox | Pinned GitHub release binary → `~/.local/bin/cloudfox`; falls back to `go install` |
| PMapper | `pip install principalmapper` |
| AWS CLI | `pip install awscli` (only when no `aws` is found) |
| requests | `pip install requests` |

PEP 668 ("externally managed environment") is handled by retrying with
`--break-system-packages`. A tool that installs but does not run is reported as
unsatisfied rather than crashing the run. Use `--no-install` to disable all of this, or
`--install-only` to just provision the tools and exit.

## Testing

```bash
python3 tests/test_cloudscan.py        # scope filter, JSON parsing, analysis pipeline, report
```

The suite mocks the DeepSeek client and needs no AWS access.

## Notes and limitations

- **Read-only by design.** awscloudscan only calls enumeration APIs; CloudFox and PMapper are
  themselves read-only tools. Run it with an identity you are authorised to use.
- CloudFox enumerates across all regions; `-r` sets the default region used for the run.
- PMapper is restricted to the `-r` region (its `global`/IAM data is always included), which
  avoids the `EndpointConnectionError` crashes that a full multi-region sweep triggers when
  some regional endpoints are unreachable. Use `--pmapper-all-regions` for the full sweep;
  awscloudscan will then auto-detect and exclude any unreachable region and retry.
- **PMapper's runtime is dominated by its Lambda edge checker.** That checker is cubic
  (`roles × principals × functions`) and performs a full local policy evaluation at the
  innermost step; its own source carries the comment *"re-examine if it is appropriate to
  skip the next checks, which can be O(n^2)"*. On an account with 201 principals and 75
  functions it is ~1M evaluations, and on this engagement it ran for **2 h 52 m without
  writing a graph at all**. The EC2, CloudFormation and CodeBuild checkers are merely
  quadratic.

  CloudFox already records which role every Lambda function, EC2 instance and
  CloudFormation stack executes as, and `verify.py` already knows — from
  `iam:SimulatePrincipalPolicy` — which principals may modify them. The
  **resource-to-role join** (`resource_edges.py`) puts those two halves together, so the
  four checkers are skipped by default (`--pmapper-skip-cloudfox-services`) and their
  output is recovered in milliseconds instead of hours. The join is stricter than PMapper
  in one place: PMapper supplies `iam:PassedToService` itself, so service-gated
  `iam:PassRole` grants come back as plain allows; here they are reported as conditional.
  It matches PMapper in another: principals CloudFox already rates administrative are not
  used as edge sources, just as PMapper skips `is_admin` nodes.

  When a path's grants are unbounded — allowed on `"*"` — the join emits **one finding per
  principal and shape**, not one per role: a single `lambda:UpdateFunctionCode` on `"*"`
  reaches every function's execution role, and reporting that once per role would weight
  the report's severity counts by one root cause. The collapsed finding lists the roles it
  reaches, worst first, and takes its severity from the worst of them; the full
  role-by-role breakdown stays on the finding's `edge.roles` record in `analysis.json`.
  Scoped grants keep one finding per role, because each scope covers a different one. On
  this account the join reports 16 findings covering 72 role-reaches.

  `iam` and `sts` are never excluded, and PMapper still runs for them. Use
  `--no-resource-edges` or `--no-pmapper-skip-cloudfox-services` to go back to PMapper's
  own checkers. Once a graph exists, later runs reuse it and skip creation entirely.
- If a run collects no evidence at all, awscloudscan exits non-zero instead of writing an empty
  report that looks like a clean bill of health. Override with `--allow-empty`.
- AI output is a starting point for human review — validate every finding and escalation
  path before remediation or disclosure.
- The report is scoped to the account; organisation-wide issues are listed separately for
  whoever owns the parent organisation.
