# Bourbaki — handoff for next session

**Date:** 2026-05-15. Master HEAD: `b04d2fe`. **160/160 tests pass. Pyright: 0 errors / 0 warnings** on `backend/bourbaki/`. Working tree clean. In sync with `origin/master`.

This file is the fastest way for the next agent to get on the same page as the previous one. It covers (1) where the project actually is, (2) what's broken, (3) what's flagged, (4) what's blocked on what, (5) what mistakes not to repeat, (6) where to look for everything.

If you only have time for one section, read **§3 (Open issues)** and **§5 (Lessons & don'ts)**.

---

## 1. Architecture in 60 seconds

The proving engine is a single async `while`-loop in `backend/bourbaki/prover/`:

```
ProverState → Proposer (GLM-5.1, structured output) → Builder (warm LeanREPLSession) →
              Reviewer (GLM-5.1, check_1 ∧ check_2) → lean_prover final gate → ProverResult
              ↑                                                       ↓ fail
              └────── Memory (3 strategies) ←───────── BuildFailed / ReviewRejected ──┘
```

- **Proposer** emits a full theorem replacement via Pydantic AI `output_type=ProverResult`. 90s per-call timeout.
- **Builder** sends `preamble + proposal.code` (with imports stripped) to a warm REPL session. Parses `sorries[*].goal`, scans for `axiom` / `apply?` / `exact?`. Returns typed feedback.
- **Reviewer** runs a second LLM with `output_type=ReviewDecision`. Approves only if `check_1 ∧ check_2`. On approval, invokes `lean_prover` once as ground-truth.
- **Memory** is one of `MemorylessMemory`, `PreviousKMemory`, `ExperienceMemory`.
- Max 50 iterations (8 for interactive). Loop-owned attempt counter, not `state.iteration`.

The HILBERT pipeline (`sketch → formalize → decompose → search_tree → stitch`) was deleted in Phase 3 (commit `2113629`, -6,576 / +118 lines). Only `autonomous/tactics.py` survived from the old `autonomous/` tree — its blocklist is still load-bearing in the builder.

Visuals: `assets/prover-loop.svg`, `assets/benchmark-history.svg` (both light-theme, GitHub-palette).

---

## 2. Verified results (lean_prover-gated, real numbers)

| Date | Subset | Verified | Notes |
|------|--------|---------:|-------|
| 2026-02-22 (audit) | full 244 | 6.2% (15/244) | the v0.2.1 91.8% / 94.3% claims were inflated ~15× by REPL false positives; retracted publicly |
| 2026-03-08 (v0.2.2) | full 244 | 25.8% (63/244) | + REPL pipe-recovery + tactic blocklist |
| 2026-04-01 | 10-problem | 50.0% (5/10) | HILBERT decomposer baseline |
| 2026-04-25 | 10-problem | **90.0% (9/10)** | proposer-builder-reviewer loop, 0 false positives |
| 2026-05-09 | 35-problem stratified | **62.9% (22/35)** † | same loop on a wider sample, 0 false positives |

**† The 22/35 is a lower bound.** See §3 #19 below.

PutnamBench (2020-2023, 5 problems): 0/5 was the May 9 dry run. The original `args_as_dict` blocker is fixed (commit `66cba4c` shim + Anthropic-compat routing restored); Putnam is just genuinely hard for glm-5.1 single-shot.

Per-source breakdown of the 22/35 (May 9): mathd 13/15 (87%), induction 2/2 (100%), numbertheory 2/2 (100%), algebra 3/5 (60%), amc 2/3, **aime 0/3 + imo 0/3 (model-capability ceiling)**, unknown 2/5.

---

## 3. Open GitHub issues — 7

**Tier-1 (blocks v0.3.0):**

- **#19** — Re-run 35-problem benchmark with 240s reviewer timeout fix. The reviewer's final `lean_prover` call was using a 30s default; standalone `lake env lean + import Mathlib` needs 60-180s. Fix landed in `7b07c07`. 22/35 May 9 result is a lower bound until this re-run completes. `[bug, benchmark, tier-1]`
- **#14** — Full 244-problem miniF2F valid + test run with the new loop. Gates on #19 (don't run #14 until #19's 35-problem number is solid). `[benchmark, tier-1]`
- **#15** — Tag v0.3.0 + the v0.2.0/v0.2.1 GitHub release notes are already retitled "RETRACTED (inflated numbers)" (done 2026-05-14); v0.3.0 tag itself waits on #14. `[process]`

**Tier-2 (post-release):**

- **#17** — Live A/B for `mathlib_search` proposer tool on the 8 hard-category problems that failed in May 9 run. Wired in commit `4ef9398`, off by default behind `enable_mathlib_search=True`. `[benchmark, infrastructure]`
- **#18** — Live A/B for Pass@N (N=4) on the 35-problem subset. Wired in commit `3222a07` behind `pass_n=N` flag. `[benchmark, scale]`
- **#20** — Live A/B for `prompts_v2.py` (a draft with shorter PROPOSER_SYSTEM_PROMPT, tactic shortlist, ≤2-sentence reasoning cap; -150 tokens/iteration). File exists at `backend/bourbaki/prover/prompts_v2.py` with 4 unit tests but never run live. Gates on #19. `[benchmark, infrastructure]`

**Meta:**

- **#11** — "Status: inflated results correction" — closes once #15 ships.

---

## 4. Flagged but not fixed (3 known footguns)

The timeout-audit agent (commit `8f7f753`) found three things it couldn't fix without violating its no-touch scope. Each needs an issue + a small follow-up PR.

1. **`ProverConfig.build_timeout=60.0` is declared but unwired.** `backend/bourbaki/prover/builder.py` calls `session.send_cmd(cmd)` with no `timeout=` kwarg → inherits the REPL's 120s default. The 60s field is misleading. Fix: either delete the field or wire it through `run_builder(state, session, timeout=cfg.build_timeout)`.

2. **`backend/bourbaki/server/routes/prove.py` `ProveRequest.timeout: int = 30`.** Same #19 class bug on the HTTP endpoint — a `POST /prove` request without `timeout=` will silently time out on any Mathlib-using proof. Bump to 240s.

3. **`backend/tests/test_prover_reviewer_memory.py` is fragile to missing `GLM_API_KEY`.** `reviewer._resolve_model_object` constructs `AnthropicProvider(...)` eagerly inside `run_reviewer`, raising `pydantic_ai.UserError` before `monkeypatch.setattr("pydantic_ai.Agent.run", ...)` can intercept. Passes on the user's machine because `.env` exists; CI without the secret would see 8 failures. Fix: monkeypatch `GLM_API_KEY` in a fixture, or refactor `_resolve_model_object` to defer provider construction.

Two cosmetic items the docs agent flagged but left:

4. **README's "Autonomous Mode" heading** (line 97). Body accurately describes the new loop; the heading + nav anchor still says "Autonomous Mode" from the pre-Phase-3 era. Renaming to "Proof Loop" or "Proof Search" touches the nav-strip anchor at line 14. User decision.

5. **GitHub repo description + topics** (visible via `gh repo view`) — still uses "autonomous agent" language. Marketing-level wording, not deprecated routes. User decision.

---

## 5. Lessons & don'ts (read these before you do anything)

### Environment

- **The Bash-tool-managed sessions kill long background processes.** `nohup ... </dev/null & disown` SOMETIMES survives the launching shell exit, but not reliably. Real long-running benchmarks need a separate terminal (tmux/screen/normal shell). The standalone runner scripts in `backend/scripts/` exist exactly to address this — run them from a user shell, not from inside an agent session.

- **System load matters.** On 2026-05-14 the machine had load average 17 and the REPL's `import Mathlib` timeout (300s) fired. The 35-problem run on 2026-05-09 worked because the system was quiet. **Always run `just preflight` before launching a benchmark** — it checks load avg + lake env + GLM round-trip.

- **Mathlib import time:** ~176s for full `import Mathlib`, ~47s for `Mathlib.Tactic` (on a quiet machine). The REPL pays this once per process; standalone `lake env lean` pays it every time.

- **The Lean toolchain is 4.28.0-rc1 (project pin) vs 4.22.0 (system default).** `lean --version` returns 4.22.0; `lake env lean --version` returns 4.28.0-rc1. The project's Mathlib was built against 4.28.0-rc1. `_run_lake_lean` in `tools/lean_prover.py` correctly uses `lake env lean` with `cwd=LEAN_PROJECT_DIR` — don't try to shortcut this.

### z.ai billing

- **z.ai has TWO API endpoints, with SEPARATE billing pools:**
  - `https://api.z.ai/api/anthropic` — Anthropic-compatible. **Where GLM-5/5.1 resource packages live for this account.** Use prefix `glm:`.
  - `https://api.z.ai/api/paas/v4/` — OpenAI-compatible. Different pool; this account doesn't have credits there. Use prefix `glm-oai:`.
- "Insufficient balance" error on OpenAI-compat with full funds visible in the dashboard ⇒ the funds are on the Anthropic-compat side. Don't reroute to OpenAI-compat to "fix" it.

### pydantic_ai upstream bug

- pydantic_ai 1.56's `ToolCallPart.args_as_dict` crashes with `TypeError: Expected bytes, bytearray or str` when z.ai's tool-use response feeds a non-string/non-dict (list, in practice) back as the `args`. **This is patched at import time by `backend/bourbaki/prover/_pydantic_ai_compat.py`** — a `_safe_args_as_dict` shim that coerces any input to a dict. Loaded automatically when you `import bourbaki.prover`. If upstream ships a fix, **delete the file + the import line in `__init__.py`**.

### Mistakes I made that you don't have to

- **Don't switch z.ai endpoints to work around an upstream bug.** I did this in `ddd6edc` and had to revert in `66cba4c` (the right fix is the shim above).
- **Don't write to the main worktree when using `isolation: worktree`.** Several earlier subagents committed directly to master instead of their assigned worktree. Solution: be explicit in agent prompts about cwd, and always cherry-pick from the agent's branch even if files appear to be on master already (they may be uncommitted).
- **Don't run 25 parallel agents.** I was asked once. 6-8 disjoint deliverables is the sweet spot. More than that causes merge conflict storms and overlap.
- **Don't trust pyright diagnostics during active edits.** The IDE LSP lags behind file writes; pyright on the CLI (`pyright backend/bourbaki/`) is the source of truth. Should be 0 errors / 0 warnings.
- **Don't give the REPL only 4 minutes to warm up.** On a busy system Mathlib import alone takes 3+ minutes. Use 600s+ outer timeouts when running 1-problem smokes.
- **Don't read result JSONs from `.bourbaki/benchmarks/results/` looking for verified counts** — that's where the gitignored runtime artifacts live. The committed historical numbers are in `CHANGELOG.md` and `docs/REALITY_CHECK.md`.

---

## 6. Where to find things

### Code

```
backend/bourbaki/
├── prover/                    ← the new architecture
│   ├── prover.py              ProverLoop + ProverConfig + routing
│   ├── proposer.py            run_proposer + _resolve_model_object
│   ├── builder.py             run_builder
│   ├── reviewer.py            run_reviewer + final lean_prover gate (240s timeout)
│   ├── memory.py              3 strategies + _resolve_model_object (duplicated)
│   ├── state.py               ProverState, FeedbackMessage, ProverResult, ReviewDecision
│   ├── feedback.py            10 typed feedback factories
│   ├── prompts.py             PROPOSER / REVIEWER / EXPERIENCE prompts (current)
│   ├── prompts_v2.py          DRAFT, off by default, tracked in #20
│   ├── _pydantic_ai_compat.py args_as_dict shim — patched at import time
│   └── __init__.py            imports the compat shim before anything else
├── tools/
│   ├── lean_prover.py         standalone Lean compile (default timeout NOW 240s)
│   ├── lean_repl.py           warm LeanREPLSession + REPLSessionPool
│   ├── proof_code_builder.py  assemble_standalone_proof(preamble, code)
│   └── mathlib_search.py      Loogle + LeanSearch + LeanExplore (proposer tool)
├── benchmarks/
│   ├── loader.py              MiniF2FProblem + load_minif2f_problems
│   ├── putnam_loader.py       PutnamProblem + load_putnam_problems
│   ├── minif2f.py             attempt_proof_loop / attempt_proof_pass_at_n / run_minif2f
│   ├── putnam.py              run_putnam (use_loop=True wired)
│   └── results_db.py          loader/aggregator for the saved JSONs
├── autonomous/
│   └── tactics.py             SURVIVED Phase 3 — blocklist still used by builder
└── server/routes/
    ├── query.py               POST /query (SSE)
    └── autonomous.py          STUB — all handlers return HTTP 410 Gone
```

### Scripts (run from a normal shell, not a bash-tool session)

```
backend/scripts/
├── preflight.py               just preflight       — system check before any benchmark
├── run_full_244.py            just full-244        — ~20 hour overnight
├── run_minif2f_subset.py      just subset          — default = 35-problem stratified
├── run_putnam_loop.py         just putnam          — PutnamBench through the loop
├── ab_mathlib_search.py       just ab-mathlib      — #17 A/B
├── ab_pass_at_n.py            just ab-passn        — #18 A/B
└── view_result.py             just view [--diff a b] — pretty-print result JSONs
```

Plus `justfile` at repo root maps each script to a `just <recipe>` entry point.

### Docs

| File | Status |
|------|--------|
| `README.md` | Current. Has Results table + Known Limitations section. |
| `ARCHITECTURE.md` | Current. Embeds both SVGs. Has dated history table. |
| `CHANGELOG.md` | Current. v0.3.0-pending entry is the source of truth for the architecture's history. |
| `docs/REALITY_CHECK.md` | Current. Leads with the 2026-05-14 timeout-bug retrospective. |
| `docs/RUNBOOK.md` | NEW — operator-facing guide for launching benchmarks. ~200 lines. |
| `docs/ASSESSMENT.md` | SUPERSEDED 2026-05-13. Body is decomposer-era; banner at top points to current docs. |
| `CLAUDE.md` | Current. Updated 2026-05-14 with prover/ layout + 410 Gone callouts on `/autonomous/*`. |
| `.bourbaki/plans/REFACTOR_PLAN.md` | COMPLETED. Historical record. |
| `.bourbaki/plans/proposer-builder-loop.md` | COMPLETED. Design doc, useful for understanding the why. |
| `.bourbaki/plans/refactor-audit.md` | COMPLETED. Keep/reuse/drop classification. |
| `.bourbaki/plans/archive/` | Obsolete plans (decomposer-blockers, lean-interact-eval). |

`.bourbaki/` itself is gitignored — local artifacts only. The benchmark result JSONs at `.bourbaki/benchmarks/results/` are not in git but you can find a per-file summary at `.bourbaki/benchmarks/results/INDEX.md`.

### Plans / design context

If you need to understand WHY the architecture is what it is, read `.bourbaki/plans/proposer-builder-loop.md` end-to-end. It's the design doc from before the refactor; everything it describes is now in the code.

The ax-prover reference implementation that inspired the loop is at `/Users/yadkonrad/dev_dev/year26/apr26/ax-prover-base/src/ax_prover/`.

---

## 7. Specific TODOs the next agent can pick up

In dependency order:

1. **`just preflight` on a quiet system + run `just subset` (the 35-problem benchmark).** This is the v0.3.0 critical path. Once it lands, file the result, update README's footnote on 62.9%, close #19.
2. **Decide on the 3 flagged footguns in §4.** Either three small PRs or one combined cleanup PR.
3. **Run `just ab-passn` and `just ab-mathlib`** in parallel-ish (they share GLM + REPL, so really sequentially) once #19 lands. Adds data to #17 and #18.
4. **`just full-244`** — the v0.3.0 release blocker. ~20 hours wall time on a quiet machine. Don't start before #19 confirms the 35-problem number with the timeout fix.
5. **Tag v0.3.0** — mechanical once #14's full-244 number is in hand. Closes #15 and #11.

Things deliberately NOT in scope right now:

- Rewriting the TUI's `/prove` to call `/query` with `use_loop=True`. The TUI currently prints a deprecation message; users go via the standalone scripts. Wiring it requires backend changes (`/query` doesn't accept `use_loop` yet) — flagged for later.
- Putnam-specific tuning. Putnam needs Pass@N and possibly `mathlib_search` to score > 0; those A/Bs come first.
- Prompts v2 swap. Keep `prompts_v2.py` as a draft until #20 confirms it's an improvement.

---

## 8. Recent commit history (most recent 20)

```
b04d2fe  fix: putnam.id (not .problem_id) in agent ad8d0fead's new WARN logs
389208d  test: codify timeout-alignment invariants across the prover chain
43ddf9c  feat: add scripts/preflight.py + `just preflight` recipe
8f7f753  fix: bump silent 30s timeout defaults + promote stall failures to WARN
4bb4fdb  docs: point prompts_v2 A/B at the new #20, not #17 (mathlib_search)
7ff4893  docs: add Known limitations section to README
faa0b49  docs: add operator runbook for long-running benchmark launches
f5d9a8d  docs: cover reviewer-timeout bug in REALITY_CHECK + ARCHITECTURE
b012f78  docs: log 240s reviewer-timeout fix + flag 62.9% as a lower bound
7b07c07  fix: bump reviewer's lean_prover timeout from 30s default to 240s
dcdfc57  docs: mark ASSESSMENT.md SUPERSEDED + refresh issue table
aa2dea7  docs: add RETRACTED header to REALITY_CHECK Feb-18 section
f573f2f  docs: add RETRACTED badges to CHANGELOG v0.2.0/v0.2.1
f68b2a1  docs: scrub deleted-module refs from ARCHITECTURE
0d75da1  docs: scrub /prove + autonomous/ references from README
8fd40c1  docs: scrub deleted-module references from CLAUDE.md
ae19e4c  chore: pyright cleanup across backend production code
c6a3398  chore: mark backend/scripts/*.py as executable
d5fae6a  feat: draft prompts_v2.py based on 35-problem run analysis (#17)
22621fd  docs(tui): deprecate /prove /pause /progress
```

For the deeper history of the refactor, `git log --oneline --since="2026-04-25"` is the full picture. Phase 1 scaffold was `49211ce`; Phase 3 deletes was `2113629`; the shim fix was `66cba4c`.

---

## 9. Quick verification commands

```bash
# Sanity check the codebase is healthy
cd backend && python3 -m pytest tests/ -q                       # 160+ pass
cd .. && pyright backend/bourbaki/                              # 0 errors

# Check current state vs origin
git status --short
git rev-list --count origin/master..HEAD                         # should be 0 unpushed

# Smoke the LLM routing (does NOT require Lean)
GLM_API_KEY=... python3 -c "
import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel
from bourbaki.prover.proposer import _resolve_model_object
class T(BaseModel): x: str
async def m():
    a = Agent(_resolve_model_object('glm:glm-5.1'), output_type=T, system_prompt='reply with x')
    print((await a.run('say hi')).output)
asyncio.run(m())
"
# Expected: < 10s, prints x='Hi' or similar

# Preflight before any benchmark
just preflight

# Quickest end-to-end smoke if the system is quiet
GLM_API_KEY=... python3 backend/scripts/run_minif2f_subset.py --ids mathd_algebra_10 --max-iter 4 --timeout 600
```

---

## 10. Hard rules the user has set (from MEMORY.md, do not violate)

- **NEVER push to remote without explicit user permission.** Ever. Even after multiple successful pushes in a session, treat each push as needing fresh OK.
- **NEVER commit personal/private content** to git. Code is fine. Meeting prep, references to named individuals, anything that's a "document about a person" is not.
- **NEVER report benchmark numbers without `lean_prover` verification.** REPL `goals=[]` does not mean proof is valid.
- **Always read code before claiming.** Verify with data first.
- **Distinguish code changes from personal documents.** Different treatment for each.

---

## 11. What state would mean "ready to ship v0.3.0"

All of the following:

- [ ] #19 has a re-run of the 35-problem subset with master at `b04d2fe` or later, on a quiet system. Result documented in README.
- [ ] #14 has a full 244-problem valid run with the new loop. Headline number in CHANGELOG.
- [ ] (Optional, recommended) #17 or #18 has a live A/B that shows the wired-but-untested flags work or doesn't help.
- [ ] CHANGELOG.md v0.3.0-pending → v0.3.0 — <date>.
- [ ] README.md headline updated with the full-244 number.
- [ ] `git tag -a v0.3.0` + `git push origin v0.3.0`.
- [ ] `gh release create v0.3.0` with the honest number.
- [ ] Close #11, #14, #15, #19.

That's the finish line. Everything else is post-release polish.

---

End of handoff. Good luck.
