# Building Bourbaki: A Math Agent That Can Check Its Own Work

> Originally published at [yad.codes/posts/building-bourbaki/](https://yad.codes/posts/building-bourbaki/) on February 9, 2026.

## Why Math, Why Now

I learn by building. The way I figure out if an idea makes sense is to sit down and construct the thing and see what happens. Reading papers helps. Thinking helps more. But actual construction is where I genuinely learn.

The Erdos Navigator project came from wanting to give a coding agent a structured environment for mathematical problems. It's a database of 1,179 Erdos problems with a REST API, CLI, and skills that tell the agent how to explore. It worked — the agent could find problems, check formalization status, understand what's been attempted. But it couldn't compute. It couldn't verify. It could discuss mathematics but couldn't execute mathematics.

That gap bothered me. So I built [Bourbaki](https://github.com/0bserver07/bourbaki).

If you're building agents for science, mathematics is promising ground. Not because agents find mathematics easy — they don't. But mathematics has something most domains lack: a way to check if you're right. A proof is correct or it isn't. Lean 4 can verify it mechanically.

This connects to ongoing thinking since the Erdos Navigator work. The pattern: LLM proposes, formal system verifies. That's fundamental across program synthesis, which initially drew me in. LLM-guided synthesis is really just this pattern applied to code: the model generates candidates, something else checks them.

## Z3 vs. Lean 4

Why Lean 4 instead of Z3? Z3 is an SMT solver answering "is this satisfiable?" It works in decidable theories: boolean logic, linear arithmetic, bit vectors, arrays. Perfect for program synthesis where checking specifications matters.

But mathematical proofs aren't constraint satisfaction. Proving square root of 2 is irrational requires constructing arguments from axioms, using previously proven lemmas and theorems. Lean 4 does this with Mathlib, a library of over 100,000 formalized mathematical results. You build on existing proofs using tactics like `ring`, `omega`, `norm_num` that automate reasoning categories. Z3 validates formulas; Lean validates proofs and helps find them.

For Bourbaki, I wanted both: SymPy for symbolic computation Z3 handles poorly (integration, series expansion, matrix operations), and Lean for multi-step proofs building on Mathlib. SymPy computes, Lean verifies.

## The Research Landscape

Two recent papers frame current significance:

**First Proof** (Abouzaid et al., 2026) published ten research-level math problems with encrypted answers — testing whether AI solves problems absent from training data. Not textbook exercises. Real research questions. The answers are locked until someone claims a solution. It's a benchmark that can't be gamed by memorization.

**Semi-Autonomous Mathematics Discovery with Gemini** (Feng et al., 2026) pointed Gemini at 700 open Erdos problems, resolving 13. Most of those solutions already existed. The problems were "open through obscurity rather than difficulty." The bottleneck wasn't reasoning — it was search.

## The State of Math + Proofs + LLMs

**Aristotle** (Harmonic, 2025) solved 5 of 6 International Mathematical Olympiad problems. Their system uses custom 200B+ parameter models fine-tuned with reinforcement learning, Monte Carlo Graph Search over Lean tactic space, and lemma decomposition — generating informal proofs, breaking them into lemmas, formalizing each, and revising on failure. Test-time training retrains on search traces during inference. This requires substantial cluster compute. It's a proof search engine.

**LeanDojo** (Caltech) builds research tooling: retrieval-augmented LLMs for Lean proof search, LeanCopilot as IDE assistant, LeanAgent for lifelong theorem-proving. Their goal: make formal verification more accessible by putting LLMs in the loop as proof assistants.

**Axiom Math AI** builds a reasoning engine, focusing on the deep structure of reasoning engines and discovery philosophy.

Then there's Bourbaki — built in a weekend with off-the-shelf LLMs, no custom training.

Lean 4 has become approachable. Tyler Josephson's "Lean for Scientists and Engineers" course teaches this outside pure mathematics frameworks. Kevin Buzzard's Xena Project has pushed Lean for working mathematicians. 2024 was the year it started clicking for the broader community. Mathlib keeps growing. The tooling keeps improving.

---

## Here is Bourbaki

Named after Nicolas Bourbaki — the collective pseudonym of mathematicians rewriting all mathematics from scratch using set theory.

Claude Code gives LLMs shell and dev tools for code. Bourbaki does the same for mathematics: it provides an LLM with a computer algebra system (SymPy), a proof assistant (Lean 4), and research APIs (OEIS, arXiv).

Ask a question in the TUI, the agent computes, verifies, researches, and streams the answer back. If it writes a proof, it formalizes it. If it makes a claim, it checks it.

**The Toolbox:**

| Tool | Function |
|------|----------|
| **Symbolic Compute** | Native SymPy: simplification, integration, solving, 30+ operations |
| **Lean Prover** | Lean 4 + Mathlib, machine-checked formal proofs |
| **Sequence Lookup** | OEIS: identify and explore integer sequences |
| **Paper Search** | arXiv: find relevant papers and results |
| **Web Search** | Exa: search for mathematical references |

**Architecture Stack:**

```
src/                          React + Ink TUI (display client)
├── components/               UI components (Input, AgentEventView, AnswerView)
├── hooks/                    useAgentRunner (SSE bridge), useModelSelection
└── skills/                   21 SKILL.md proof technique files

backend/bourbaki/             Python backend (owns all state)
├── agent/                    Pydantic AI agent, prompts, scratchpad, event mapper
├── tools/                    SymPy, Lean 4, OEIS, arXiv, Web Search, Skills
├── sessions/                 Persistence + context compaction
├── autonomous/               Long-running proof search with strategies
├── problems/                 13 classic problems database
└── server/routes/            FastAPI endpoints (query, sessions, skills, ...)
```

## The Loop

When you give it a problem:

1. You ask in the TUI
2. The backend agent reasons about approach
3. It calls tools: SymPy for computation, Lean for verification, OEIS/arXiv for lookup
4. Results feed back into the agent, which iterates if needed
5. A scratchpad enforces limits and deduplicates repeated calls
6. The final answer streams back to the TUI

**The Scratchpad**

Without it, the agent loops — calling the same tool with slightly rephrased queries. The scratchpad tracks every tool call per query and does two things: enforces hard limits (3 calls per tool by default) and detects duplicates using word-level Jaccard similarity at 0.7 threshold. If 70% of words overlap, it returns: "Similar query already sent to {tool_name}. Consider a different approach."

At the limit: "This is the last allowed call to {tool_name}." After that, the tool blocks for that query. The scratchpad formats usage summaries injected into iteration prompts: `search: 2/3 calls used (1 remaining)`.

**Skill Files**

These encode proof techniques as SKILL.md files with YAML frontmatter. 21 techniques across five categories:

- **Basic:** induction, strong induction, direct proof, contradiction, pigeonhole, counting
- **Analysis:** epsilon-delta, convergence tests, sequence limits, inequality chains
- **Geometry:** coordinate proof, synthetic construction, transformations
- **Algebra:** group homomorphisms, ring ideals, polynomials
- **Advanced:** extremal arguments, probabilistic method, conjecture exploration, formalization, proof explanation

Each file walks through a specific technique step-by-step, including which tools to use. They load into the system prompt as available techniques. The agent picks the technique based on the problem.

Skills load at three levels: built-in (`src/skills/`), user (`~/.bourbaki/skills/`), or project (`.bourbaki/skills/`).

**The Compute-Then-Verify Pattern**

SymPy proposes, Lean checks. The agent cannot say "it's obvious that..." and move on. If it claims an algebraic identity, it computes it. If it claims a proof step follows, it formalizes it.

```
❯ Prove that the sum of the first n integers equals n(n+1)/2

⏺ Thinking...
⏺ Symbolic Compute (expression=Sum(k, (k, 1, n)))
  ⎿ Computed result
⏺ Lean Prover
  ⎿ ✓ Verified in 2.3s

Proof by induction. Base case: n = 1, sum = 1 = 1·2/2. ✓
Inductive step: assume ∑_{k=1}^{n} k = n(n+1)/2.
Then ∑_{k=1}^{n+1} k = n(n+1)/2 + (n+1) = (n+1)(n+2)/2. ∎
```

The agent computed the symbolic sum with SymPy, then verified the full proof in Lean 4. Both happened inside the loop, not after. That's the difference between "the agent wrote a proof" and "the agent wrote and checked a proof."

**Lean Integration**

Simple by design. The agent generates Lean 4 code, Bourbaki writes it to a temp file, runs `lean` (or falls back to `lake env lean`), and parses output. The prover has a 30-second timeout. Response structure: `success`, `goals` (remaining obligations), `proofComplete`, `errors` (line, column, message, severity), `rawOutput`, and `duration` in milliseconds.

Three conditions for `proofComplete = true`: return code 0, no error-level messages, and no `sorry` in code. That last part matters. `sorry` is Lean's placeholder for skipped steps. Lean accepts the file but the proof is incomplete. Bourbaki checks explicitly. The agent can't sneak one past.

When Lean rejects something, errors are structured. The agent gets full output and can fix on the next iteration. Between that and scratchpad's 3-call limit on Lean, the agent has at most three attempts before working with what it has.

SymPy runs in-process (no subprocess overhead) covering: number theory (factorization, primality, Euler's totient, modular inverse), algebra (simplify, expand, solve), calculus (derivatives, integrals, limits, Taylor series, Fourier series, Laplace transforms), series (infinite sums and products), and linear algebra (determinants, eigenvalues, row reduction, characteristic polynomials).

SymPy is forgiving. Lean is not. A proof 95% correct fails exactly as hard as one 5% correct. The type checker doesn't grade on a curve.

An autonomous mode tries different strategies, backtracks when stuck, and remembers what worked and what didn't. It tracks dead ends. Start with `/prove <problem_id>` and let it run.

---

## Where This Leaves Me

I keep building environments for agents and finding the same thing. The agent's raw reasoning is fine. The bottleneck is always the environment. Drop it in a blank room and it hallucinates. Give it a database, tools, and structured techniques and it does real work.

The Erdos Navigator gave the agent a world of problems. The resume project gave it context about me. Bourbaki gives it math tools.

I'm not claiming Bourbaki solves research-level math. It doesn't. Aristotle can solve IMO problems autonomously. Bourbaki can help you work through a proof and check your steps. Those are different things. The First Proof benchmark exists because we don't have widely accessible systems for research-level math yet. But Bourbaki can handle mathematics that falls apart when LLMs attempt it from memory: symbolic computation, sequence identification, formalizing proofs the agent already knows how to write naturally.

The gap between "agent explains a proof" and "agent produces verified proof" is still large. Lean 4 is unforgiving. Translating mathematical intuition into formal logic is hard for humans and agents both. Aristotle solves formalization gaps with MCGS and custom RL training. LeanDojo solves it with retrieval-augmented search. I solve it by watching the agent fail and adding skill files teaching missing techniques. Different approaches, different budgets.

The scratchpad started as a quick fix for looping and became crucial. Without it, the agent was confident and repetitive. With it, the agent was forced to try new things.

I want to integrate the Erdos Navigator database so the agent can pull problems and work on them inside Bourbaki's compute-verify loop. The navigator provides problems, Bourbaki provides tools. I'm curious about LeanDojo's retrieval-augmented proof search — LeanCopilot could probably integrate into Bourbaki's tool chain, making Lean integration much deeper.

Or the obvious next step is something I haven't thought of yet. I keep building these things and ending up somewhere unplanned. That's the point. I'd rather apply and learn than theorize about why it works.

---

## Update: What Actually Compiles (March 2026)

The numbers in the changelog were wrong.

After building the benchmark runner, I reported 91.8% on miniF2F valid. That number came from the REPL, which said `goals=[]` after applying a tactic. I treated that as "proof complete." It wasn't. The REPL accepts tactics that produce well-typed terms locally but don't compile as standalone Lean files. `apply Set.mem_of_mem_filter` closes the goal in the REPL. It fails when you run `lean` on the file.

The verified number, where every solve compiles as a standalone Lean file through `lean_prover`, is **63/244 (25.8%)**.

That's the gap between "the REPL says it worked" and "Lean actually accepts the proof."

### The Pipe Bug

The verified number should have been higher than 63. During benchmark runs, a different bug was killing problems silently.

The Lean REPL communicates over stdin/stdout. Commands go in as JSON, responses come back as JSON, separated by blank lines. When a tactic times out (say `decide` on a large finite goal), Python's `asyncio.wait_for` cancels the read. But the REPL keeps writing its response. Those bytes sit in the pipe. The next command reads stale data from the previous response, gets a JSON parse error, and fails. Every command after that also fails.

Problem 47 times out. Problems 48 through 244 all fail with garbage data. It looks like they're hard problems. They're not. The session is broken.

The fix has two layers. When a read times out internally, drain the remaining output until you hit the blank-line separator, then continue. When a caller's `asyncio.wait_for` cancels the read externally, catch the `CancelledError`, drain, and re-raise. If draining itself times out (the tactic is hung and still computing), kill the session. It auto-restarts on the next call.

```python
async def _drain_stale_output(self, drain_timeout: float = 10.0) -> bool:
    """Read and discard remaining response to resync the pipe."""
    try:
        async def _drain():
            while True:
                raw = await self.proc.stdout.readline()
                if not raw:
                    return  # EOF
                if raw.decode().strip() == "":
                    return  # Blank line = end of stale response
        await asyncio.wait_for(_drain(), timeout=drain_timeout)
        return True
    except asyncio.TimeoutError:
        return False
```

The drain approach preserves all proof state IDs in the session. If you kill instead, every proof state from before the restart is invalid and the search tree is dead. Draining is almost free and keeps everything intact.

### What the 63 Solves Actually Are

All 63 are single-tactic proofs. `norm_num` solves 23, `omega` solves 13, `ring` gets 8, `linarith` gets 7. These are problems where Lean's built-in automation closes the goal in one step. The search tree finds multi-step proofs via REPL, but translating those back to standalone code that compiles is unsolved.

| Category | Verified | Rate |
|----------|----------|------|
| mathd | 54/130 | 42% |
| algebra | 5/18 | 28% |
| AMC/misc | 4/48 | 8% |
| aime | 0/12 | 0% |
| imo | 0/20 | 0% |
| induction | 0/8 | 0% |
| numbertheory | 0/8 | 0% |

Zero on IMO, AIME, induction, and number theory. Those require multi-step reasoning. The search tree explores tactic sequences through the REPL and sometimes finds them, but the proof can't be extracted as a standalone file. The REPL assigns proof state IDs that are session-local. A proof found as "apply tactic A to state 3, then tactic B to state 7" doesn't translate to a Lean file without reconstructing the full proof term.

### What I Learned

**Subprocess protocols break when you interrupt reads.** This is general to any line-delimited protocol over pipes. If you cancel a read mid-stream, the remaining bytes corrupt every subsequent read. You have to drain or restart. There's no third option.

**Don't trust interactive environments for verification.** The REPL is useful for exploration. It's fast (30ms per tactic vs 90s for a full Lean compilation). But "the REPL accepted it" and "Lean compiles it" are different claims. I conflated them for weeks.

**Corruption cascades look like hard problems.** When the pipe broke, the log showed problem after problem failing. The errors looked like tactic failures, not protocol corruption. Without logging the raw JSON parse errors, I couldn't tell the difference between "this problem is genuinely hard" and "the session is broken." I only caught it by noticing that a re-run of a single problem in isolation would succeed when the same problem failed in the benchmark sequence.

### Where This Goes

The pipe fix needs a benchmark re-run. Some of the 181 unsolved problems were probably failing due to session corruption, not because they're unsolvable. The honest gap to SOTA (HILBERT at 99.2%) is multi-step proofs. The search tree finds them through the REPL but can't export them. That's the next problem.

The original post said "I'm not claiming Bourbaki solves research-level math." That's still true, and the gap is wider than the REPL numbers suggested. The compute-then-verify pattern works. The verification part was just broken.

---

## References

### Research

- Abouzaid, M., Blumberg, A.J., Hairer, M., et al. (2026). [First Proof](https://arxiv.org/abs/2602.05192).
- Feng, T., et al. (2026). [Semi-Autonomous Mathematics Discovery with Gemini: A Case Study on the Erdos Problems](https://arxiv.org/abs/2601.22401).
- Achim, T., et al. (2025). [Aristotle: IMO-level Automated Theorem Proving](https://arxiv.org/abs/2510.01346). Harmonic.

### Math + LLM Tools

- [LeanDojo](https://leandojo.org/). Caltech. LLM-powered tools for Lean theorem proving.
- [Axiom Math AI](https://axiommath.ai/). Reasoning engine for mathematical discovery.

### Learning Lean

- Josephson, T. (2024). [Lean for Scientists and Engineers](https://www.youtube.com/playlist?list=PLX21uJ4UfpF43NExUcPcAEgnzV58x_26l). YouTube.
- Buzzard, K. (2024). [Lean in 2024](https://xenaproject.wordpress.com/2024/01/20/lean-in-2024/). Xena Project.

### Related Posts

- [building-erdos-navigator](https://yad.codes/posts/building-erdos-navigator/)
- [make-an-honest-resume-for-your-coding-agent](https://yad.codes/posts/make-an-honest-resume-for-your-coding-agent/)
- [coding-agents-made-me-better-programmer](https://yad.codes/posts/coding-agents-made-me-better-programmer/)
