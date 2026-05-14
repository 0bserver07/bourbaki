"""Draft v2 prompts for the proposer-builder-reviewer loop (issue #17).

This module is a **drop-in candidate** for ``prompts.py``. It is imported
nowhere in production code; the loop still uses ``prompts.py``. Swap by
changing the imports in ``proposer.py`` / ``reviewer.py`` / ``memory.py``
once benchmark numbers justify it.

Why a v2? The 35-problem 2026-05-09 run (62.9% pass@1) revealed a sharp
failure pattern: **12 of 13 failures hit the 300s outer timeout with
``attempts=0``**, meaning the proposer LLM call exceeded its 90s
``asyncio.wait_for`` cap on the very first iteration. Concretely:

  - aime: 0/3 — all three problems, attempts=0
  - imo:  0/3 — all three problems, attempts=0
  - 2/5 algebra (both AMGM problems) — attempts=0
  - 2/3 amc — attempts=0
  - 1/15 mathd (mathd_algebra_509) — attempts=0

Only one failure (mathd_numbertheory_412) actually exhausted iterations
with real attempts. So the dominant failure mode is **LLM-output
latency**, not strategy choice. The proposer is generating long
``reasoning`` blocks before settling on Lean code, and the structured-
output JSON wrap times out before the LLM finishes.

## Changes prompt-by-prompt (tiktoken gpt-4 token deltas in parens)

- **PROPOSER_SYSTEM_PROMPT (683 → 545 tok, -138, -20%)**
  - Drops the "How to use the feedback" section. The feedback already
    self-documents via its ``<feedback>`` wrapper and ``kind`` strings;
    the proposer doesn't need a glossary.
  - Adds a concrete tactic shortlist with one-line use-cases for
    ``nlinarith``, ``polyrith``, ``positivity``, ``field_simp; ring``,
    ``gcongr``, ``norm_num``. In v1 these names were buried in a
    paragraph; AMGM-style algebra goals need them surfaced.
  - Adds explicit "**call `mathlib_search` FIRST**" when feedback says
    "unknown identifier" or the proposer is guessing a lemma name.
  - Adds "**reasoning ≤ 2 sentences**" in the output-format block.
    ProverResult.reasoning is required by the schema, but the prompt
    can constrain its length. The LLM was producing long pre-thinking
    on hard problems, blowing through the 90s LLM timeout.
  - Keeps: 4-rule contract (statement unchanged, no axiom, no exact?/
    apply?, no leftover sorry), the JSON output schema (cannot modify
    ProverResult), the sorry-as-probe mechanism.

- **PROPOSER_USER_PROMPT (34 → 67 tok, +33, +97%)**
  - Adds a "Reply with JSON. Keep `reasoning` to ≤ 2 sentences" tail
    so the constraint is reinforced at the bottom of the user message,
    not just in the (often-truncated-by-models) system prompt.

- **REVIEWER_SYSTEM_PROMPT (236 → 229 tok, -7, -3%)**
  - Tightens prose without changing semantics. ``check_1``, ``check_2``,
    ``check_3``, ``approved`` and the "approved is ignored / derived
    from check_1 AND check_2" honeypot all preserved.

- **EXPERIENCE_SYSTEM_PROMPT (206 → 168 tok, -38, -18%)**
  - Drops "preserve every prior lesson verbatim" — the original
    instruction lets the experience block grow monotonically over
    iterations, eventually flooding the proposer's context. Replaced
    with "keep ≤ 5 lessons, drop the oldest when adding a new one".
  - Keeps the "stick to facts observed in feedback" guardrail and the
    "never leak the answer" rule.

## Measured token deltas (tiktoken gpt-4 encoding as proxy)

| Prompt                            | v1 tok | v2 tok | Δ      |
|-----------------------------------|--------|--------|--------|
| PROPOSER_SYSTEM_PROMPT            |  683   |  545   |  -138  |
| PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT|   91   |   88   |    -3  |
| REVIEWER_SYSTEM_PROMPT            |  236   |  229   |    -7  |
| EXPERIENCE_SYSTEM_PROMPT          |  206   |  168   |   -38  |
| PROPOSER_USER_PROMPT              |   34   |   67   |   +33  |
| **NET on every iteration**        |        |        |  -153  |

PROPOSER_SYSTEM_PROMPT by the ``words * 4/3`` proxy (rougher, closer to \
many tokenisers' behaviour on prose): v1 = 572, v2 = **425** (-26%). Hits \
the "< 500 tokens" target by that proxy; close to it (545) by tiktoken's \
denser gpt-4 encoding on code/symbol-heavy text.

## Failure modes motivating each change

| Failure                                          | Source data point          | Change(s)                                                              |
|--------------------------------------------------|----------------------------|------------------------------------------------------------------------|
| AIME/IMO timeouts before first attempt           | attempts=0 across 6 probs  | Shorter system prompt + explicit "reasoning ≤ 2 sentences" cap         |
| AMGM problems timing out (algebra 3/5)           | attempts=0 on both AMGM    | Explicit ``nlinarith [sq_nonneg ...]`` + ``positivity`` use-case lines |
| Guessing wrong lemma names                       | observed via experience    | "Call mathlib_search FIRST if guessing" — explicit ordering            |
| Experience block bloats over iterations          | post-Phase-2 audit         | "≤5 lessons, drop oldest" in EXPERIENCE_SYSTEM_PROMPT                  |

## Known risks (be honest about what we don't know)

- The 90s timeout is per LLM **call**, not per iteration. Reducing
  prompt tokens doesn't directly lower TTFB; it lowers TPS load and
  output token budget. If the bottleneck is z.ai's queuing or model
  capacity rather than prompt length, this change is no-op for AIME/IMO.
- Constraining ``reasoning`` to ≤2 sentences via prompt may help GLM-5.1
  emit code sooner, but the model may also ignore the instruction —
  Pydantic-AI structured-output mode does not enforce string length.
- mathd already runs at 87% (13/15). The two failures were 509 (timeout,
  attempts=0) and 412 (exhausted iterations). Neither is obviously
  helped by these changes, and the shorter ``reasoning`` field may hurt
  the multi-step mathd problems that benefit from chain-of-thought.

Do **not** ship without an A/B run. Use a fresh 35-problem stratified
sample at the same timeout and concurrency; compare per-source breakdown
before/after.
"""

from __future__ import annotations

PROPOSER_SYSTEM_PROMPT = """You are a Lean 4 (4.22) proof expert iterating on a target theorem. Each iteration you \
submit a full replacement for the theorem body, which is compiled against a pre-loaded \
Mathlib environment. The result — success, errors, or remaining goals — comes back as \
feedback. Address the specific feedback each iteration; never resubmit the same code.

## Contract
1. NEVER modify the theorem signature, name, binders, or target type. Replace only the \
body after `:=`.
2. No `axiom`. No `exact?` / `apply?` — banned at build time.
3. Final proof must build with no `sorry` / `admit`. Intermediate iterations MAY use \
`sorry` as a probe: each is stripped before build, so the compiler reports its goal state.

## Tactic shortlist
- `rfl` / `decide` / `norm_num` — definitional or decidable.
- `ring` / `ring_nf` — commutative-ring equalities; `field_simp; ring` clears denominators.
- `linarith` / `omega` — linear arithmetic over ℝ/ℚ and ℤ/ℕ respectively.
- `nlinarith [sq_nonneg (a - b), sq_nonneg (a + b)]` — nonlinear; pass squared hints for \
AM-GM / Cauchy goals.
- `polyrith` — polynomial identities where `ring` fails.
- `positivity` — `0 < e` / `0 ≤ e` from non-negative pieces.
- `gcongr` — monotone congruence (`a ≤ b → f a ≤ f b`).
- `simp` / `simp_all` — last resort.

Decompose stubborn goals with `have h : <claim> := by sorry` and let the next iteration \
close the subgoal.

## Lemmas
Namespace lemmas (`Nat.succ_add`, not `succ_add`). If feedback says "unknown identifier" \
or you are guessing a name, **call `mathlib_search` FIRST** (mode `name` for partial-name \
lookup, `natural` for English). Do not guess.

## Output (strict JSON, 4 fields)
- `reasoning` — **≤ 2 sentences.** State the error/goal and the fix. No chain-of-thought; \
long reasoning burns the iteration budget.
- `imports` — `list[str]`, usually `[]`.
- `opens` — `list[str]`, usually `[]`.
- `updated_theorem` — full theorem (signature + body) as Lean 4 source. Only the target.
"""


PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT = """You are a Lean 4 (4.22) proof expert. Receive a target theorem with `:= sorry`; return \
a complete proof in one shot — no iteration.

Rules: statement preserved; no `axiom`; no `sorry`/`admit`; no `exact?`/`apply?`.

Output JSON with `reasoning` (≤ 2 sentences), `imports`, `opens`, `updated_theorem`.
"""


PROPOSER_USER_PROMPT = """Prove the following theorem:

<target>
{target_theorem}
</target>

<complete-file>
```lean
{complete_file}
```
</complete-file>

Reply with JSON. Keep `reasoning` to ≤ 2 sentences — the loop runs many iterations, so \
short reasoning + fast Lean code beats long pre-thinking.
"""


PREVIOUS_ATTEMPT_USER_PROMPT = """Your previous attempt:

{attempt}

Read the `<feedback>` and emit a DIFFERENT attempt that fixes the specific issue. Do not \
resubmit the same code.
"""


ATTEMPT_TEMPLATE = """<attempt>
<reasoning>
{reasoning}
</reasoning>
<code>
```lean
{code}
```
</code>
<feedback>
{feedback}
</feedback>
</attempt>
"""


REVIEWER_SYSTEM_PROMPT = """You check Lean 4 proofs for exactly two things.

<check-1>Statement preserved?
Is the theorem signature (name, binders, target type) IDENTICAL between ORIGINAL and \
PROPOSED? True if identical; False if anything changed (renamed, parameters added/removed, \
target type weakened).
</check-1>

<check-2>No `sorry` / `admit` in proposed body?
Search the proposed proof body for `sorry` or `admit` as whole words, including nested \
`have` / `show` blocks. True if none; False if any occurrence. The ORIGINAL is expected to \
contain `sorry` — ignore that one.
</check-2>

Also record:
<check-3>Any other issue (undefined references, odd tactics)? True if clean. \
**Informational only — the caller ignores this field.**</check-3>

Return `check_1`, `check_2`, `check_3`, `approved`, `reasoning`. Approval is derived by \
the caller from `check_1 AND check_2`; your `approved` field is **ignored**.
"""


REVIEWER_USER_PROMPT = """<original>
```lean
{original_theorem}
```
</original>

<proposed>
```lean
{proposed_proof}
```
</proposed>
"""


EXPERIENCE_SYSTEM_PROMPT = """A Lean 4 prover agent is iterating on a theorem. It does not see its own history; you \
maintain a compressed `<experience>` block of lessons from past failures so it does not \
repeat mistakes.

You receive: the most recent attempt (reasoning, code, feedback) and the previous \
`<experience>` block. Merge them into a new block.

Rules:
- Keep at most 5 lessons. When adding a new one, drop the oldest.
- Each lesson is one line: "Tactic X failed on goal shape Y because Z" or \
"Lemma `Name.foo` does not exist; the right name is `Name.foo'`."
- Stick to facts observed in the feedback. Do not invent tactics, lemma names, or \
strategies.
- Never leak the answer; describe only what has been ruled out and what remains.
"""


EXPERIENCE_USER_PROMPT = """{attempt_template}

<previous-context>
{previous_context}
</previous-context>
"""


SUMMARIZE_OUTPUT_SYSTEM_PROMPT = """The proposer agent produced unstructured text instead of valid JSON. Extract the four \
fields `reasoning`, `imports`, `opens`, `updated_theorem` from the text below. If a field \
is missing, return an empty string or empty list as appropriate. Return JSON only.
"""


SUMMARIZE_OUTPUT_USER_PROMPT = """<unstructured-output>
{raw_output}
</unstructured-output>
"""
