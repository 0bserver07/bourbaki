#!/usr/bin/env bash
# Run miniF2F benchmark suite against Bourbaki.
#
# Prerequisites:
#   - miniF2F-lean4 cloned to .bourbaki/miniF2F-lean4
#   - Lean 4 + Mathlib installed (.bourbaki/lean-project/)
#   - Python backend dependencies installed
#
# Usage:
#   ./scripts/run-benchmark.sh                  # Run validation split
#   ./scripts/run-benchmark.sh --split test      # Run test split
#   ./scripts/run-benchmark.sh --split all       # Run both splits
#   ./scripts/run-benchmark.sh --source aime     # Filter by source
#   ./scripts/run-benchmark.sh --quick           # Run 20 easy problems

set -euo pipefail

MINIF2F_DIR=".bourbaki/miniF2F-lean4"
SPLIT="valid"
SOURCE=""
QUICK=false

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --split) SPLIT="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --quick) QUICK=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Clone miniF2F if not present
if [ ! -d "$MINIF2F_DIR" ]; then
  echo "Cloning miniF2F-lean4..."
  git clone https://github.com/yangky11/miniF2F-lean4 "$MINIF2F_DIR"
fi

# Build the Python command
CMD="python -c \"
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.insert(0, 'backend')
from bourbaki.benchmarks.minif2f import run_minif2f

async def main():
    result = await run_minif2f(
        split='${SPLIT}',
        source_filter='${SOURCE}' if '${SOURCE}' else None,
        timeout=120,
    )
    print()
    print(f'=== miniF2F {\"${SPLIT}\"} Results ===')
    print(f'Solved: {result.solved}/{result.total} ({result.pass_rate*100:.1f}%)')
    print(f'Time: {result.total_time_seconds:.0f}s')
    for src, counts in sorted(result.by_source.items()):
        pct = counts[\"solved\"]/counts[\"total\"]*100 if counts[\"total\"] else 0
        print(f'  {src}: {counts[\"solved\"]}/{counts[\"total\"]} ({pct:.0f}%)')

asyncio.run(main())
\""

echo "Running miniF2F benchmark (split=$SPLIT)..."
echo ""
eval "$CMD"
