# Bourbaki benchmark runner scripts

Standalone Python scripts for launching long-running benchmarks **outside**
of a bash-tool-managed session. The asyncio loop lives inside the single
Python process, so killing the launching shell does **not** kill the proof
loop.

Use these from the repo root:

```bash
python3 backend/scripts/<name>.py [flags]
```

All scripts:

- Read `GLM_API_KEY` from env (or `.env` at the repo root; the project
  auto-exports it via `bourbaki.config.export_api_keys`).
- Tee stdout to `/tmp/bourbaki/<scriptname>-YYYYMMDD-HHMMSS.log`.
- Print a banner with config + ETA + result path.
- Save the underlying `BenchmarkResult` JSON to
  `.bourbaki/benchmarks/results/` (the inner `run_minif2f` /
  `run_putnam` writes it).

## Prerequisites

| Resource | Path | How to set up |
|---|---|---|
| miniF2F checkout | `.bourbaki/miniF2F-lean4/` | `git clone https://github.com/yangky11/miniF2F-lean4 .bourbaki/miniF2F-lean4` |
| PutnamBench checkout | `.bourbaki/putnam-bench/` | `git clone https://github.com/trishullab/PutnamBench .bourbaki/putnam-bench` |
| Lean 4 + Mathlib | `.bourbaki/lean-project/` | `./scripts/setup-lean.sh` |
| `GLM_API_KEY` | env var or `.env` | `export GLM_API_KEY=...` or add to `.env` at repo root |

## Scripts

### `run_full_244.py`

Full miniF2F valid split (244 problems) through the
proposer-builder-reviewer loop.

```bash
python3 backend/scripts/run_full_244.py
python3 backend/scripts/run_full_244.py --split test
python3 backend/scripts/run_full_244.py --max-iter 30 --pass-n 4 --timeout 600
python3 backend/scripts/run_full_244.py --enable-mathlib-search
```

**Wall time:** roughly 20 hours for the valid split at defaults
(`--max-iter 20 --timeout 300 --pass-n 1`). `--split all` (488 problems)
doubles that.

### `run_minif2f_subset.py`

Arbitrary miniF2F subset. Defaults to the 35-problem stratified set used
in the May 9 A/B (IDs are pulled from
`.bourbaki/benchmarks/results/2026-03-19_1516_minif2f_valid.json`).

```bash
python3 backend/scripts/run_minif2f_subset.py                              # default 35
python3 backend/scripts/run_minif2f_subset.py --ids a,b,c
python3 backend/scripts/run_minif2f_subset.py --from-file ids.txt
python3 backend/scripts/run_minif2f_subset.py --pass-n 4
```

**Wall time:** ~2-3 hours for the 35-problem subset at defaults.

### `run_putnam_loop.py`

PutnamBench problems through the loop. Defaults to a 5-problem 2020-2023
dry-run subset. Answer-sorry problems are excluded by default (the loop
cannot fill answer placeholders).

```bash
python3 backend/scripts/run_putnam_loop.py                                 # 5-problem dry run
python3 backend/scripts/run_putnam_loop.py --year-range 2020-2023
python3 backend/scripts/run_putnam_loop.py --ids putnam_2020_a1,putnam_2020_b1
```

**Wall time:** ~20 min for the dry run, multi-hour for full years.

### `view_result.py`

Pretty-prints a benchmark result JSON. Defaults to the most recent file
in `.bourbaki/benchmarks/results/`.

```bash
python3 backend/scripts/view_result.py                       # latest
python3 backend/scripts/view_result.py --latest              # latest (explicit)
python3 backend/scripts/view_result.py path/to/file.json
python3 backend/scripts/view_result.py file_a.json --diff file_b.json
```

`--diff` shows per-problem A vs B comparison with PASS/FAIL/regression
columns.

### `ab_mathlib_search.py`

A/B for `loop_enable_mathlib_search` on an 8-problem hard subset (issue
\#17). Runs `run_minif2f` twice — once with the flag off (control), once
on (treatment) — and prints per-problem delta. Saves both JSONs to
`.bourbaki/benchmarks/results/`.

```bash
python3 backend/scripts/ab_mathlib_search.py
python3 backend/scripts/ab_mathlib_search.py --timeout 600 --max-iter 30
python3 backend/scripts/ab_mathlib_search.py --problems id1,id2,id3
```

**Wall time:** ~1.5 hours for the default 8 × 2 runs at 300s timeout.

### `ab_pass_at_n.py`

A/B for Pass@N on the 35-problem stratified subset (issue \#18). Runs
once with `pass_n=1` (control) and once with `pass_n=N` (treatment, default
N=4). Saves both JSONs and prints a per-problem A vs B table.

```bash
python3 backend/scripts/ab_pass_at_n.py
python3 backend/scripts/ab_pass_at_n.py --n 8
python3 backend/scripts/ab_pass_at_n.py --ids id1,id2 --n 4
```

**Wall time:** worst case ~`(1 + N) × n_problems × timeout`. For default
N=4, 35 problems, 300s: roughly 9 hours worst case; in practice closer to
4-6 hours since most successes happen on the first attempt.

## Shared helpers

`_runner_common.py` provides logging setup, banner formatting, env-var
checks, and the default 35-problem subset and 8-problem hard subset
constants. Not meant to be invoked directly.

## Quick-launch via `justfile`

A `justfile` at the repo root wraps the common entry points:

```bash
just full-244            # python3 backend/scripts/run_full_244.py
just full-244-test       # python3 backend/scripts/run_full_244.py --split test
just subset              # python3 backend/scripts/run_minif2f_subset.py
just ab-mathlib          # python3 backend/scripts/ab_mathlib_search.py
just ab-passn            # python3 backend/scripts/ab_pass_at_n.py
just putnam              # python3 backend/scripts/run_putnam_loop.py
just view                # python3 backend/scripts/view_result.py --latest
```

Install `just` (https://github.com/casey/just) and run `just --list` to
see all targets.
