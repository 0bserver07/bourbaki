#!/usr/bin/env bash
# Setup a Lean 4 project with Mathlib for Bourbaki proof verification.
#
# This creates a Lake project at .bourbaki/lean-project/ with Mathlib
# as a dependency. After running, Bourbaki can use the full range of
# Mathlib tactics (norm_num, ring, linarith, simp, omega, etc.).
#
# Requirements:
#   - elan (Lean toolchain manager): https://leanprover.github.io/lean4/doc/setup.html
#   - ~4GB disk space for Mathlib + cache
#
# Usage:
#   ./scripts/setup-lean.sh

set -euo pipefail

LEAN_PROJECT_DIR=".bourbaki/lean-project"

# Check prerequisites
if ! command -v elan &>/dev/null; then
  echo "Error: elan (Lean toolchain manager) is not installed."
  echo "Install via: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
  exit 1
fi

# Create project directory
if [ -d "$LEAN_PROJECT_DIR" ]; then
  echo "Lean project already exists at $LEAN_PROJECT_DIR"
  echo "To rebuild, remove it first: rm -rf $LEAN_PROJECT_DIR"
  exit 0
fi

# Step 1: Figure out what toolchain Mathlib needs by checking its lean-toolchain
# file from the repo, then install that toolchain BEFORE running lake init.
echo "Checking latest Mathlib toolchain requirement..."
MATHLIB_TOOLCHAIN=$(curl -sSfL https://raw.githubusercontent.com/leanprover-community/mathlib4/master/lean-toolchain)
echo "Mathlib requires: $MATHLIB_TOOLCHAIN"

echo "Installing toolchain (this may download ~200MB)..."
elan toolchain install "$MATHLIB_TOOLCHAIN"

# Step 2: Create the project directory and write lean-toolchain so elan
# uses the correct version for all lake commands in this directory.
echo "Creating Lean 4 project with Mathlib at $LEAN_PROJECT_DIR..."
mkdir -p "$LEAN_PROJECT_DIR"
cd "$LEAN_PROJECT_DIR"

echo "$MATHLIB_TOOLCHAIN" > lean-toolchain

# Step 3: Initialize Lake project with math template using the correct toolchain.
# elan reads lean-toolchain from cwd and dispatches to the right lake binary.
lake init bourbaki math

echo "Fetching Mathlib cache (this downloads prebuilt .olean files)..."
lake exe cache get

echo "Building project..."
lake build

# Step 5: Build lean4-repl for tactic-by-tactic interaction
echo "Setting up lean4-repl for interactive tactic mode..."
REPL_DIR=".lake/repl"
if [ ! -d "$REPL_DIR" ]; then
  git clone https://github.com/leanprover-community/repl "$REPL_DIR"
  cd "$REPL_DIR" && lake build && cd ../..
  echo "lean4-repl built successfully."
else
  echo "lean4-repl already exists at $REPL_DIR"
fi

echo ""
echo "Done! Mathlib and lean4-repl are ready."
echo "Bourbaki will auto-detect Mathlib on next startup."
echo ""
echo "To verify: cd $LEAN_PROJECT_DIR && lake env lean --version"
