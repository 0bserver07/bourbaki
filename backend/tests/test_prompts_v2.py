"""Smoke tests for the draft v2 prompts (issue #17).

These do not exercise the LLM — they guard the prompt text against
regressions that would defeat the v2 design goals:

- PROPOSER_SYSTEM_PROMPT stays small (don't let it bloat again).
- The tactic shortlist actually mentions the automation tactics the
  AMGM / nonlinear-algebra problems need.
- Reviewer keeps its two real checks plus the honeypot wording.
- v2 can be imported alongside v1 without name collisions in the
  shared ``bourbaki.prover`` namespace.
"""

from __future__ import annotations


def test_proposer_system_prompt_under_700_words():
    """Guard against future bloat. v2 ships at ~319 words (~545 tiktoken
    tokens / ~425 by the words*4/3 proxy). 700 words is a generous
    ceiling that still flags accidental copy-paste growth."""

    from bourbaki.prover import prompts_v2

    word_count = len(prompts_v2.PROPOSER_SYSTEM_PROMPT.split())
    assert word_count < 700, (
        f"PROPOSER_SYSTEM_PROMPT is {word_count} words; v2 target is < 700. "
        "If this fails, you have likely added a section back in."
    )


def test_proposer_system_prompt_mentions_key_tactics():
    """The Apr-25 / May-09 runs showed AMGM / nonlinear-algebra problems
    failing because the proposer wasn't reaching for nlinarith or
    polyrith. v2 surfaces these explicitly in the tactic shortlist;
    this test ensures nobody removes them."""

    from bourbaki.prover import prompts_v2

    prompt = prompts_v2.PROPOSER_SYSTEM_PROMPT
    for tactic in ("nlinarith", "polyrith", "positivity", "field_simp"):
        assert tactic in prompt, (
            f"PROPOSER_SYSTEM_PROMPT must mention `{tactic}`; AMGM goals "
            f"depend on it."
        )

    # mathlib_search must be present AND must be flagged as
    # "call FIRST when guessing" — not just a passing mention.
    assert "mathlib_search" in prompt, (
        "PROPOSER_SYSTEM_PROMPT must reference the mathlib_search tool."
    )
    assert "FIRST" in prompt, (
        "PROPOSER_SYSTEM_PROMPT must instruct the proposer to call "
        "mathlib_search FIRST before guessing a lemma name."
    )

    # Reasoning length cap — the whole point of v2 is to stop the LLM
    # from burning the 90s timeout on chain-of-thought.
    assert "2 sentences" in prompt or "≤ 2" in prompt, (
        "PROPOSER_SYSTEM_PROMPT must constrain `reasoning` length; the "
        "90s LLM timeout failures motivated this change."
    )


def test_reviewer_system_prompt_preserves_check_1_and_check_2():
    """The reviewer's two real checks (statement preserved, no sorry/
    admit) drive approval. ``check_3`` and ``approved`` are honeypots —
    documented as ignored by the caller. v2 must preserve all four
    field names so ``ReviewDecision`` parses correctly."""

    from bourbaki.prover import prompts_v2

    prompt = prompts_v2.REVIEWER_SYSTEM_PROMPT

    # The two checks that actually matter
    assert "check_1" in prompt
    assert "check_2" in prompt

    # The honeypots must remain in the schema-facing instructions
    assert "check_3" in prompt
    assert "approved" in prompt

    # The honeypot mechanism (caller derives approval from check_1 AND
    # check_2; ``approved`` field is ignored) must be explicit.
    assert "check_1 AND check_2" in prompt or "check_1 and check_2" in prompt.lower(), (
        "Reviewer prompt must state that approval = check_1 AND check_2."
    )
    assert "ignored" in prompt.lower(), (
        "Reviewer prompt must mark `approved` as ignored — losing this "
        "line would invite the model to self-approve buggy proofs."
    )

    # check_2 must still call out `sorry` AND `admit` — both are banned
    # in the final body.
    assert "sorry" in prompt
    assert "admit" in prompt


def test_prompts_v2_can_be_imported_alongside_v1():
    """The v2 module must not shadow v1 symbols when both are imported.
    They live as siblings in ``bourbaki.prover`` and have to coexist
    until a deliberate swap."""

    from bourbaki.prover import prompts as v1
    from bourbaki.prover import prompts_v2 as v2

    # Same surface — both modules must export the names the loop calls.
    expected = {
        "PROPOSER_SYSTEM_PROMPT",
        "PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT",
        "PROPOSER_USER_PROMPT",
        "PREVIOUS_ATTEMPT_USER_PROMPT",
        "ATTEMPT_TEMPLATE",
        "REVIEWER_SYSTEM_PROMPT",
        "REVIEWER_USER_PROMPT",
        "EXPERIENCE_SYSTEM_PROMPT",
        "EXPERIENCE_USER_PROMPT",
        "SUMMARIZE_OUTPUT_SYSTEM_PROMPT",
        "SUMMARIZE_OUTPUT_USER_PROMPT",
    }
    for name in expected:
        assert hasattr(v1, name), f"v1 is missing {name}"
        assert hasattr(v2, name), f"v2 is missing {name}"

    # And v2 should genuinely differ from v1 on the prompts we changed
    # (sanity check — guards against a copy/paste that accidentally
    # re-exports the v1 strings).
    assert v1.PROPOSER_SYSTEM_PROMPT != v2.PROPOSER_SYSTEM_PROMPT
    assert v1.EXPERIENCE_SYSTEM_PROMPT != v2.EXPERIENCE_SYSTEM_PROMPT
