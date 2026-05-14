# Bourbaki — common entry points for benchmark runs and dev tasks.
#
# Run `just --list` to see all recipes.
# All runners launch long-running asyncio loops in a single Python process,
# so killing the launching shell does NOT kill the proof loop.  Prefer
# `nohup just <target> &` (or screen/tmux) for overnight runs.

# Default: show the help-style listing.
default:
    @just --list

# ---------- miniF2F ----------

# Full miniF2F valid split (244 problems). ~20h at defaults.
full-244 *ARGS:
    python3 backend/scripts/run_full_244.py {{ARGS}}

# Full miniF2F test split (244 problems).
full-244-test *ARGS:
    python3 backend/scripts/run_full_244.py --split test {{ARGS}}

# 35-problem stratified subset (or pass --ids / --from-file).
subset *ARGS:
    python3 backend/scripts/run_minif2f_subset.py {{ARGS}}

# ---------- A/B experiments ----------

# A/B: loop_enable_mathlib_search on an 8-problem hard subset (#17).
ab-mathlib *ARGS:
    python3 backend/scripts/ab_mathlib_search.py {{ARGS}}

# A/B: pass_n=1 vs pass_n=N on the 35-problem subset (#18).
ab-passn *ARGS:
    python3 backend/scripts/ab_pass_at_n.py {{ARGS}}

# ---------- PutnamBench ----------

# PutnamBench through the loop. Defaults to 5-problem dry run.
putnam *ARGS:
    python3 backend/scripts/run_putnam_loop.py {{ARGS}}

# ---------- Inspection ----------

# Pretty-print the most recent result JSON.
view *ARGS:
    python3 backend/scripts/view_result.py {{ARGS}}

# Pretty-print result JSON A vs B: just diff a.json b.json
diff A B:
    python3 backend/scripts/view_result.py {{A}} --diff {{B}}

# ---------- Health checks ----------

# Pre-benchmark health check. Run before any long-running benchmark to
# catch a loaded system or a misconfigured z.ai key before the run
# silently regresses.  See issue #19 for the motivating regression.
preflight *ARGS:
    python3 backend/scripts/preflight.py {{ARGS}}

# ---------- Tests ----------

# Run the pytest suite.
test:
    cd backend && python3 -m pytest tests/ -x

# Run only the prover-loop tests.
test-loop:
    cd backend && python3 -m pytest tests/test_prover_loop.py tests/test_prover_proposer.py tests/test_prover_builder.py tests/test_prover_reviewer_memory.py tests/test_prover_state.py -x
