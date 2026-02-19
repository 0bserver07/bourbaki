"""Verify miniF2F benchmark results with lean_prover (whole-file compilation).

Takes existing REPL-reported results and re-checks each solved problem
by compiling the full Lean file. This catches false positives where the
REPL reports "no remaining goals" but the proof is actually invalid.

Usage:
    python -m bourbaki.benchmarks.verify_minif2f \
        --results .bourbaki/benchmarks/results/2026-02-18_0045_minif2f_enhanced_test.json \
        --timeout 150
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / ".bourbaki" / "benchmarks" / "results"


async def verify_results(
    results_path: str,
    timeout: int = 150,
    max_problems: int | None = None,
) -> None:
    """Verify solved problems from existing benchmark results.

    Args:
        results_path: Path to the JSON results file.
        timeout: Timeout in seconds for lean_prover per problem.
        max_problems: Optional limit on number of problems to verify.
    """
    from bourbaki.tools.lean_prover import lean_prover

    with open(results_path) as f:
        data = json.load(f)

    # Find solved problems with proof code
    results_key = "results" if "results" in data else "problems"
    all_results = data.get(results_key, [])

    solved = []
    for r in all_results:
        is_solved = r.get("solved", False)
        code = r.get("proof_code") or r.get("proof") or ""
        if is_solved and code:
            solved.append(r)

    logger.info("Found %d solved problems with proof code out of %d total", len(solved), len(all_results))

    if max_problems:
        solved = solved[:max_problems]

    total = len(solved)
    logger.info("Verifying %d solved problems from %s", total, results_path)
    logger.info("Timeout: %ds per problem", timeout)

    verified = 0
    failed = 0
    errors = []
    start = time.monotonic()

    for i, r in enumerate(solved):
        pid = r.get("id") or r.get("problem_id") or f"problem_{i}"
        code = r.get("proof_code") or r.get("proof", "")

        # Ensure import Mathlib is present
        if "import Mathlib" not in code:
            code = "import Mathlib\n\n" + code

        logger.info("[%d/%d] Verifying %s...", i + 1, total, pid)

        try:
            result = await asyncio.wait_for(
                lean_prover(code=code, mode="check", timeout=timeout),
                timeout=timeout + 10,
            )

            if result.get("proofComplete"):
                verified += 1
                logger.info("  VERIFIED (%ds)", result.get("duration", 0) // 1000)
            else:
                failed += 1
                err_list = result.get("errors") or []
                err_msg = ""
                if err_list:
                    first = err_list[0]
                    err_msg = first.get("message", str(first)) if isinstance(first, dict) else str(first)
                logger.info("  REJECTED: %s", err_msg[:100])
                errors.append({"id": pid, "error": err_msg[:200]})

        except asyncio.TimeoutError:
            failed += 1
            logger.info("  TIMEOUT (%ds)", timeout)
            errors.append({"id": pid, "error": f"Timeout ({timeout}s)"})

        except Exception as e:
            failed += 1
            logger.info("  ERROR: %s", e)
            errors.append({"id": pid, "error": str(e)[:200]})

    elapsed = time.monotonic() - start

    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info("Total verified:   %d/%d (%.1f%%)", verified, total, verified / total * 100 if total else 0)
    logger.info("Rejected:         %d", failed)
    logger.info("Time:             %.0fs (%.1fh)", elapsed, elapsed / 3600)
    logger.info("=" * 60)

    # Save verification results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    split = data.get("split", data.get("benchmark", "unknown"))
    out_path = RESULTS_DIR / f"{ts}_minif2f_verified_{split}.json"

    verify_data = {
        "source_results": str(results_path),
        "total_solved_claimed": total,
        "verified": verified,
        "rejected": failed,
        "verified_rate": verified / total if total else 0,
        "verification_timeout": timeout,
        "total_time_seconds": round(elapsed, 1),
        "timestamp": ts,
        "errors": errors,
    }

    with open(out_path, "w") as f:
        json.dump(verify_data, f, indent=2)

    logger.info("Results saved to %s", out_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify miniF2F results with lean_prover")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--timeout", type=int, default=150, help="Timeout per verification (default: 150s)")
    parser.add_argument("--max", type=int, default=None, help="Max problems to verify")
    args = parser.parse_args()

    asyncio.run(verify_results(
        results_path=args.results,
        timeout=args.timeout,
        max_problems=args.max,
    ))


if __name__ == "__main__":
    main()
