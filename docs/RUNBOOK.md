# Bourbaki Operator Runbook

Practical guide for launching long-running benchmarks against the
proposer-builder-reviewer loop, interpreting the JSON results, and
sharing numbers without re-introducing the v0.2.1 inflation problem.

The scripts referenced here live in
[`backend/scripts/`](../backend/scripts/README.md); the `just`
recipes are in [`justfile`](../justfile).

---

## Prerequisites

| Resource | Path | How to set up |
|---|---|---|
| Repo | `.` | `git clone https://github.com/0bserver07/bourbaki` |
| Backend deps | `backend/` | `cd backend && pip install -e ".[dev]"` |
| miniF2F | `.bourbaki/miniF2F-lean4/` | `git clone https://github.com/yangky11/miniF2F-lean4 .bourbaki/miniF2F-lean4` |
| PutnamBench | `.bourbaki/putnam-bench/` | `git clone https://github.com/trishullab/PutnamBench .bourbaki/putnam-bench` |
| Lean 4 + Mathlib | `.bourbaki/lean-project/` | `./scripts/setup-lean.sh` |
| `GLM_API_KEY` | env var or `.env` | `export GLM_API_KEY=...` or add to `.env` at the repo root |

The backend auto-loads `.env` from the repo root via Pydantic Settings
(`bourbaki.config.export_api_keys`). No separate `source .env` step is
needed — the standalone runner scripts pick it up automatically.

A `just preflight` recipe is planned for verifying the four resources
above before kicking off a long run; once landed it will check repo
checkouts, `lean --version`, `GLM_API_KEY`, and the
`.bourbaki/miniF2F-lean4/` files in one shot. See `just preflight`
once landed.

---

## Launching a benchmark

All entry points are in `backend/scripts/` and wrapped in `justfile`
recipes. The asyncio loop lives inside a single Python process, so
killing the shell that launched it does NOT kill the proof loop —
prefer `nohup` or `tmux`/`screen` for overnight runs.

### Smoke (~5 minutes)

```bash
just subset --ids mathd_algebra_10 --max-iter 8 --timeout 300
```

One easy problem (`mathd_algebra_10` is a `by norm_num` solve) at low
iteration cap. Useful for verifying the environment before committing
to a long run. If this fails, do not start a long run.

### 35-problem stratified subset (~3.5 hours)

```bash
just subset
# or with non-default flags:
python3 backend/scripts/run_minif2f_subset.py --max-iter 20 --timeout 600
```

The default IDs come from the 2026-05-09 run (preserved in
`.bourbaki/benchmarks/results/2026-05-09_2241_minif2f_valid.json`) —
this is the stratified sample across all miniF2F categories used
for A/B regression checks.

### Full 244-problem miniF2F valid split (~20 hours)

```bash
nohup just full-244 &
# or for the test split:
nohup just full-244-test &
# tail the run:
tail -f /tmp/bourbaki/run_full_244-*.log
```

The wall-time estimate assumes `--max-iter 20 --timeout 300 --pass-n 1`.
Pass-N>1 multiplies cost roughly linearly until the first verified pass;
see `--help` for flag combinations.

### A/B experiments

```bash
just ab-mathlib                       # mathlib_search tool A/B (~1.5h)
just ab-passn                         # Pass@N A/B (~4-9h depending on N)
```

Both save a control JSON and a treatment JSON to
`.bourbaki/benchmarks/results/` and print a per-problem comparison.

---

## System load matters

This is the lesson from [issue #19](https://github.com/0bserver07/bourbaki/issues/19).
The reviewer's `lean_prover` final-gate call has a 240s timeout
(bumped from 30s in commit `7b07c07`). Even at 240s, standalone
`lake env lean + import Mathlib` can time out on a busy machine —
the import alone usually needs 60-180s, and CPU pressure stretches
that. Symptom: `attempts > 0, verified = False` for problems you
know are solvable.

**Recommendation:** run benchmarks on a quiet/dedicated machine.
`uptime` load average < 4 is a soft rule of thumb. If you cannot
guarantee that, set `--timeout` higher (e.g. 600) and re-run
problems that fail with low attempt counts.

---

## Interpreting a result JSON

Result files land at `.bourbaki/benchmarks/results/YYYY-MM-DD_HHMM_<bench>.json`.
Use `view_result.py` to read them — it handles schema drift across
older runs:

```bash
just view                                          # latest
python3 backend/scripts/view_result.py             # latest (explicit)
python3 backend/scripts/view_result.py PATH        # specific file
```

The output shows:

- Headline: `<verified>/<total>` count and percentage.
- Per-source breakdown (mathd, algebra, aime, imo, induction, numbertheory, amc, unknown).
- Failed-problem list with `attempts` and terminal feedback kind.
- Wall time and timestamp.

The full schema and lower-level access live in
`backend/bourbaki/benchmarks/results_db.py`
(`list_results`, `load_result`, `latest_result`, `diff_results`,
`by_source`). Import it directly from a Python REPL for ad-hoc
analysis:

```python
from bourbaki.benchmarks import results_db
latest = results_db.latest_result(benchmark="minif2f")
print(results_db.by_source(latest))
```

### A/B comparison

```bash
just diff a.json b.json
# or:
python3 backend/scripts/view_result.py a.json --diff b.json
```

Prints a per-problem PASS/FAIL/regression table. A "regression"
means A solved but B did not — an honest warning that whatever changed
between the two runs has cost coverage on that problem.

---

## Sharing a result

Result JSONs are gitignored (`.bourbaki/` is excluded). Do not
`git add` them — the directory is gitignored for a reason
(some files are large; some contain sketchy intermediate state).

Recommended: post a summary as a GitHub issue comment.

```bash
gh issue comment 14 -F - <<'EOF'
Full 244 re-run done. Headline: 145/244 (59.4%), 0 false positives.

Per-source:
- mathd: 91/130 (70%)
- algebra: 12/18 (67%)
- ...

Wall time: 19h 04m, Pass@1. Result file (local-only):
.bourbaki/benchmarks/results/2026-05-22_0830_minif2f_valid.json
EOF
```

If a peer needs the raw JSON for deeper inspection, send it
directly (e.g. via DM or a temporary gist) rather than committing
it to git history.

When updating the headline number in
[`README.md`](../README.md) /
[`CHANGELOG.md`](../CHANGELOG.md) / [`ARCHITECTURE.md`](../ARCHITECTURE.md),
quote the result-file path in the same commit so the chain of
evidence stays linkable. Pattern:

> "62.9% (22/35) — see
> `.bourbaki/benchmarks/results/2026-05-09_2241_minif2f_valid.json`"

---

## When a run fails partway through

The runner saves the result JSON only at the end. If a 20-hour run
crashes at hour 18, the partial state is gone. Mitigations:

- Tee logs to `/tmp/bourbaki/<scriptname>-*.log` (every script does
  this automatically). You can still extract per-problem pass/fail
  from the log even if the JSON is missing.
- For the full 244 run, prefer to launch with `nohup` and a
  `tmux`/`screen` session so a shell hiccup doesn't kill the loop.
- If a single problem hangs the entire run, narrow the scope:
  rerun with `--ids problem_id` to confirm and patch the loop's
  per-problem timeout.

---

## See also

- [`backend/scripts/README.md`](../backend/scripts/README.md) — full
  per-script flag reference.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — loop internals.
- [`docs/REALITY_CHECK.md`](REALITY_CHECK.md) — honest history of
  the benchmark numbers and why we don't trust REPL-only detection.
- [`CHANGELOG.md`](../CHANGELOG.md) — release notes.
