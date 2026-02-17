# Full Roadmap Implementation Plan

**Date:** 2026-02-17
**Scope:** Phase A quick-wins + Phase 4 (semantic retrieval) + Phase 3 (autoformalize) + Phase 6 (multi-agent)
**Ordering:** Bottom-up (infrastructure → retrieval → autoformalize → multi-agent)
**Current baseline:** 88.9% miniF2F (217/244)

---

## Task 1: REPL stderr capture

**File:** `backend/bourbaki/tools/lean_repl.py`

**Steps:**
1. Write test: `test_lean_repl_stderr_captured` — start session, send malformed command, verify stderr content is captured (not DEVNULL)
2. Change line 75: `stderr=asyncio.subprocess.DEVNULL` → `stderr=asyncio.subprocess.PIPE`
3. Add `_stderr_buffer: list[str]` to `LeanREPLSession`
4. Add background task `_drain_stderr()` that reads stderr lines and logs them at `logger.debug` level, storing last 20 lines in buffer
5. Start `_drain_stderr` in `start()` method
6. Add `get_stderr_recent() -> list[str]` method to expose buffer
7. In `lean_tactic()`, attach stderr excerpt to error responses: `"stderr": session.get_stderr_recent()` when `success=False`
8. Verify: `pytest tests/test_repl_stderr.py`

---

## Task 2: Structured self-correction (2-round cap)

**Files:** `backend/bourbaki/agent/scratchpad.py`, `backend/bourbaki/agent/prompts.py`

**Steps:**
1. Write test: `test_correction_round_cap` — scratchpad tracks correction rounds per goal, blocks after 2
2. Add `_correction_rounds: dict[str, int]` to `Scratchpad` (key = goal hash)
3. Add `increment_correction_round(goal: str) -> int` method — returns current round count
4. Add `is_correction_exhausted(goal: str) -> bool` — True if rounds >= 2
5. Update `can_call_tool()` for lean_prover/lean_tactic: if `is_correction_exhausted` for the query, return blocked with "Maximum correction rounds reached. Change your proof strategy."
6. In `record_error()`, call `increment_correction_round()` with the code/tactic as key
7. Update self-correction protocol in `prompts.py`: add rule "After 2 correction rounds on the same goal, you MUST try a fundamentally different strategy"
8. Verify: `pytest tests/test_scratchpad.py tests/test_scoring.py`

---

## Task 3: REPL session pool

**File:** `backend/bourbaki/tools/lean_repl.py`

**Steps:**
1. Write test: `test_session_pool_acquires_and_releases` — pool creates sessions on demand, returns them after use
2. Add `REPLSessionPool` class:
   ```python
   class REPLSessionPool:
       def __init__(self, max_size: int = 4, full_mathlib: bool = False):
           self._pool: asyncio.Queue[LeanREPLSession]
           self._all_sessions: list[LeanREPLSession]
           self._max_size: int
           self._created: int = 0
       async def acquire(self) -> LeanREPLSession
       async def release(self, session: LeanREPLSession) -> None
       async def shutdown(self) -> None
   ```
3. Add `@asynccontextmanager async def pooled_session()` convenience wrapper
4. Update `get_session()` to optionally use pool: `get_session(pool: REPLSessionPool | None = None)`
5. Keep backward compatibility: existing singleton path unchanged when pool is None
6. Add `get_pool() -> REPLSessionPool` factory that creates a global pool
7. Verify: `pytest tests/test_repl_pool.py`

---

## Task 4: Semantic Mathlib retrieval via Moogle

**File:** `backend/bourbaki/tools/mathlib_search.py`

**Steps:**
1. Write test: `test_mathlib_search_semantic_mode` — mock httpx response, verify parse
2. Research Moogle API (https://www.moogle.ai/) — it's the successor to LeanSearch with semantic search
3. Add `MOOGLE_API = "https://www.moogle.ai/api/search"` constant
4. Add `_search_moogle(query, max_results, start) -> dict` function:
   - POST to Moogle API with `{"query": query, "isFind": false}`
   - Parse response: extract `name`, `module`, `type`, `doc` from results
   - Return same format as other search functions
5. Update `mathlib_search()`: add `mode="semantic"` branch → `_search_moogle()`
6. Add fallback: if Moogle fails, fall back to LeanSearch
7. Verify: `pytest tests/test_mathlib_search.py`

---

## Task 5: Goal-state-aware retrieval in search tree

**Files:** `backend/bourbaki/autonomous/search_tree.py`, `backend/bourbaki/autonomous/tactics.py`

**Steps:**
1. Write test: `test_generate_mathlib_queries_semantic` — verify semantic mode queries are generated
2. Update `generate_mathlib_queries()` in `tactics.py` to also return `("semantic", query)` tuples
3. Update `ProofSearchTree._search_mathlib()` to try semantic mode first, then fall back to type/natural
4. Add result deduplication by lemma name across modes
5. Verify: `pytest tests/test_search_tree.py tests/test_tactics.py`

---

## Task 6: Autoformalize tool

**File:** `backend/bourbaki/tools/autoformalize.py` (CREATE)

**Steps:**
1. Write test: `test_autoformalize_statement` and `test_autoformalize_proof_step`
2. Create `autoformalize.py` with:
   ```python
   async def autoformalize(
       input_text: str,
       mode: str = "statement",  # "statement" | "proof_step"
       context: str | None = None,
       model: str | None = None,
   ) -> dict[str, Any]
   ```
3. For `mode="statement"`: prompt LLM to convert NL theorem → Lean type signature, then verify with `lean_prover(mode="check")`
4. For `mode="proof_step"`: prompt LLM to convert NL step → Lean tactic, return with confidence
5. Add self-correction: if Lean check fails, retry with error feedback (1 retry)
6. Register tool in `core.py`: add `tool_autoformalize` with scratchpad tracking
7. Update `prompts.py`: add autoformalize to tool usage policy section
8. Verify: `pytest tests/test_autoformalize.py`

---

## Task 7: Sketch-first workflow in prompts

**Files:** `backend/bourbaki/agent/prompts.py`

**Steps:**
1. Update Mathematical Workflow in system prompt to add sketch-first step:
   ```
   ## Mathematical Workflow
   1. **Understand the problem**: Parse statement, identify domain
   2. **Gather evidence**: Test cases, patterns, OEIS
   3. **Plan proof sketch**: Outline strategy, identify key lemmas, list steps
   4. **Choose strategy**: Pick best approach from sketch
   5. **Execute proof**: Step-by-step with lean_tactic
   6. **Formalize**: Verify complete proof with lean_prover
   ```
2. Add proof planning prompt to self-correction: "Before retrying, outline your new strategy in 2-3 sentences"
3. Verify: import and inspect prompt string

---

## Task 8: Multi-agent role definitions

**File:** `backend/bourbaki/agent/roles.py` (CREATE)

**Steps:**
1. Write test: `test_role_definitions` — verify all roles have correct tool subsets
2. Create `roles.py` with:
   ```python
   @dataclass
   class AgentRole:
       name: str
       description: str
       tools: list[str]  # Tool names this role can use
       system_prompt_addendum: str  # Role-specific instructions

   STRATEGIST = AgentRole(...)
   SEARCHER = AgentRole(...)
   PROVER = AgentRole(...)
   VERIFIER = AgentRole(...)
   ```
3. Define tool subsets:
   - Strategist: `symbolic_compute`, `paper_search`, `skill_invoke`, `web_search`
   - Searcher: `mathlib_search`, `web_search`
   - Prover: `lean_tactic`, `mathlib_search`
   - Verifier: `lean_prover`
4. Add role-specific system prompt addendums (2-3 sentences each)
5. Verify: `pytest tests/test_roles.py`

---

## Task 9: Agent message protocol

**File:** `backend/bourbaki/agent/messages.py` (CREATE)

**Steps:**
1. Write test: `test_agent_message_creation` and `test_message_routing`
2. Create `messages.py` with:
   ```python
   @dataclass
   class AgentMessage:
       from_agent: str
       to_agent: str
       msg_type: str  # "strategy" | "lemma_list" | "proof_state" | "error" | "verified" | "subgoal"
       content: dict[str, Any]
       timestamp: float = field(default_factory=time.monotonic)

   class MessageBus:
       """Simple message routing for multi-agent coordination."""
       async def send(self, msg: AgentMessage) -> None
       async def receive(self, agent_name: str, timeout: float = 30) -> AgentMessage | None
       def get_history(self, agent_name: str) -> list[AgentMessage]
   ```
3. Verify: `pytest tests/test_messages.py`

---

## Task 10: Multi-agent coordinator

**File:** `backend/bourbaki/agent/coordinator.py` (CREATE)

**Steps:**
1. Write test: `test_coordinator_orchestrates_roles` — mock agents, verify message flow
2. Create `coordinator.py` with:
   ```python
   class ProofCoordinator:
       def __init__(self, model: str, pool: REPLSessionPool | None = None):
           self.bus = MessageBus()
           self.roles: dict[str, AgentRole]

       async def prove(self, theorem: str, timeout: float = 300) -> CoordinatorResult:
           # 1. Strategist generates proof sketch
           # 2. Searcher finds lemmas
           # 3. Prover attempts proof with lean_tactic
           # 4. On failure: loop back to strategist
           # 5. Verifier confirms with lean_prover
   ```
3. Implement orchestration loop:
   - Phase 1: Strategist generates sketch via `decompose_and_prove`
   - Phase 2: Searcher queries mathlib_search (semantic + type + natural) for each subgoal
   - Phase 3: Prover applies tactics guided by strategy + lemmas
   - Phase 4: Verifier checks complete proof
   - On failure: back to Phase 1 with insights
4. Add `CoordinatorResult` dataclass with proof_code, success, agent_stats
5. Verify: `pytest tests/test_coordinator.py`

---

## Task 11: Wire multi-agent into autonomous search

**File:** `backend/bourbaki/autonomous/search.py`

**Steps:**
1. Add `use_multi_agent: bool = False` to `AutonomousSearchConfig`
2. Add Phase 2.5 block after decomposition and before strategy rotation:
   ```python
   if self._config.use_multi_agent:
       coordinator = ProofCoordinator(model=settings.default_model)
       result = await coordinator.prove(theorem, timeout=...)
       if result.success: ...
   ```
3. Emit appropriate events for multi-agent progress
4. Verify: existing autonomous search tests still pass

---

## Task 12: Ensemble prover (parallel strategies)

**Files:** `backend/bourbaki/agent/coordinator.py`

**Steps:**
1. Write test: `test_ensemble_first_wins` — 3 parallel provers, first success returns
2. Add `ensemble_prove()` method to `ProofCoordinator`:
   - Launch 3 Prover agents in parallel with different strategies
   - Each gets a different tactic focus (automation-heavy, induction-heavy, mathlib-heavy)
   - Use `asyncio.wait(return_when=FIRST_COMPLETED)` — first success wins
   - Cancel remaining on success
3. Wire ensemble into coordinator's Phase 3
4. Verify: `pytest tests/test_coordinator.py`

---

## Task 13: Integration test for full pipeline

**File:** `backend/tests/test_multi_agent_integration.py` (CREATE)

**Steps:**
1. Create integration test with mocked LLM and REPL
2. Test: Coordinator receives theorem → Strategist plans → Searcher finds lemmas → Prover proves → Verifier confirms
3. Test: Failure path — Prover fails → back to Strategist → second attempt succeeds
4. Verify: `pytest tests/test_multi_agent_integration.py`

---

## Verification

After all tasks:
```bash
cd backend && python -m pytest tests/ -v
```

Expected: all tests pass, 0 failures.

---

## Dependencies

```
Task 1 (stderr)        → standalone
Task 2 (self-correction) → standalone
Task 3 (session pool)  → Task 1
Task 4 (semantic search) → standalone
Task 5 (goal-aware)    → Task 4
Task 6 (autoformalize) → Task 4
Task 7 (sketch-first)  → standalone
Task 8 (roles)         → standalone
Task 9 (messages)      → standalone
Task 10 (coordinator)  → Tasks 3, 6, 8, 9
Task 11 (wire search)  → Task 10
Task 12 (ensemble)     → Tasks 3, 10
Task 13 (integration)  → Tasks 10, 11, 12
```

Parallelizable batches:
- **Batch 1:** Tasks 1, 2, 4, 7, 8, 9 (all independent)
- **Batch 2:** Tasks 3, 5, 6 (depend on Batch 1)
- **Batch 3:** Task 10 (depends on Batch 2)
- **Batch 4:** Tasks 11, 12 (depend on Task 10)
- **Batch 5:** Task 13 (integration)
