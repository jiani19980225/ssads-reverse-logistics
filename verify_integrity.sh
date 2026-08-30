#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments"
PYTHON_BIN="${PYTHON_BIN:-$EXPERIMENT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

echo "--- STARTING REVERSE LOGISTICS INTEGRITY AUDIT ---"
echo "[1/5] Environment"
"$PYTHON_BIN" --version

echo "[2/5] Syntax"
"$PYTHON_BIN" -m compileall -q \
  "$EXPERIMENT_DIR/src" "$EXPERIMENT_DIR/scripts" "$EXPERIMENT_DIR/tests"

echo "[3/5] Static analysis"
(
  cd "$EXPERIMENT_DIR"
  "$PYTHON_BIN" -m ruff check src scripts tests
  "$PYTHON_BIN" -m mypy src scripts tests
)

echo "[4/5] Unit and adversarial tests"
(
  cd "$EXPERIMENT_DIR"
  "$PYTHON_BIN" -m pytest tests -q
)

echo "[5/5] One-seed end-to-end experiment"
(
  cd "$EXPERIMENT_DIR"
  "$PYTHON_BIN" scripts/run_summary.py --seeds 0 >/dev/null
)

echo "--- INTEGRITY AUDIT PASSED ---"
