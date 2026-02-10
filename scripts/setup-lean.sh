#!/usr/bin/env bash
# Setup a Lean 4 project with Mathlib for Bourbaki proof verification.
#
# This creates a Lake project at .bourbaki/lean-project/ with Mathlib
# as a dependency. After running, Bourbaki can use the full range of
# Mathlib tactics (norm_num, ring, linarith, simp, omega, etc.).
#
# Requirements:
#   - Lean 4 (elan): https://leanprover.github.io/lean4/doc/setup.html
#   - ~4GB disk space for Mathlib + cache
#
# Usage:
#   ./scripts/setup-lean.sh

set -euo pipefail

LEAN_PROJECT_DIR=".bourbaki/lean-project"

# Check prerequisites
if ! command -v lean &>/dev/null && ! command -v elan &>/dev/null; then
  echo "Error: Lean 4 is not installed."
  echo "Install via: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
  exit 1
fi

if ! command -v lake &>/dev/null; then
  echo "Error: lake (Lean build tool) not found in PATH."
  exit 1
fi

# Create project directory
if [ -d "$LEAN_PROJECT_DIR" ]; then
  echo "Lean project already exists at $LEAN_PROJECT_DIR"
  echo "To rebuild, remove it first: rm -rf $LEAN_PROJECT_DIR"
  exit 0
fi

echo "Creating Lean 4 project with Mathlib at $LEAN_PROJECT_DIR..."
mkdir -p "$LEAN_PROJECT_DIR"
cd "$LEAN_PROJECT_DIR"

# Initialize Lake project
lake init bourbaki math

echo "Fetching Mathlib cache (this downloads prebuilt .olean files)..."
lake exe cache get

echo "Building project..."
lake build

echo ""
echo "Done! Mathlib is ready."
echo "Bourbaki will auto-detect Mathlib on next startup."
echo ""
echo "To verify, run: lean --run .bourbaki/lean-project/lakefile.lean"
