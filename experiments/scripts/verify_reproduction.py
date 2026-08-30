"""Re-run every reported experiment and diff it against the committed reference.

The manuscript's numbers come from two kinds of artifact:

  * ``outputs/reviewer_experiments/*.csv`` -- written to disk by
    ``run_reviewer_experiments.py``.
  * ``outputs/reference/*.txt`` -- the console output of the four scripts that
    only print (``run_summary.py`` in three extractor modes,
    ``run_calibration.py``, ``run_ablation.py``, ``run_diagnostics.py``).
    Table II and Table III come from this second group, so they are captured
    here rather than left for a reader to compare by eye.

Both groups are deterministic under the pinned dependencies, so reproduction is
a byte comparison rather than a judgement call. Exit status is 0 when every
artifact matches and 1 otherwise, which makes this usable in CI.

Usage:
    python scripts/verify_reproduction.py            # verify against reference
    python scripts/verify_reproduction.py --update   # regenerate the reference
"""

from __future__ import annotations

import argparse
import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REFERENCE = BASE / "outputs" / "reference"
COMMITTED_CSVS = BASE / "outputs" / "reviewer_experiments"

# name -> argv for scripts whose result is stdout only.
STDOUT_RUNS: dict[str, list[str]] = {
    "summary_keyword": ["run_summary.py", "--seeds", "0-29"],
    "summary_phrase": ["run_summary.py", "--seeds", "0-29", "--extractor", "strong"],
    "summary_deepseek": ["run_summary.py", "--seeds", "0-29", "--extractor", "llm"],
    "calibration": ["run_calibration.py", "--seeds", "0-29", "--llm"],
    "ablation": ["run_ablation.py", "--seeds", "0-29"],
    "diagnostics": ["run_diagnostics.py", "--seeds", "0-29"],
}

REVIEWER_ARGS = ["run_reviewer_experiments.py", "--seeds", "0-29", "--include-llm"]


def run(argv: list[str]) -> str:
    """Run a script with the current interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, str(BASE / "scripts" / argv[0]), *argv[1:]],
        cwd=BASE,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{argv[0]} exited {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def verify_stdout(update: bool) -> list[str]:
    failures = []
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for name, argv in STDOUT_RUNS.items():
        produced = run(argv)
        target = REFERENCE / f"{name}.txt"
        if update:
            target.write_text(produced, encoding="utf-8")
            print(f"  wrote {target.relative_to(BASE)}")
            continue
        if not target.exists():
            failures.append(f"{name}: no committed reference")
            continue
        if produced != target.read_text(encoding="utf-8"):
            failures.append(f"{name}: output differs from committed reference")
        else:
            print(f"  ok   {name}")
    return failures


def verify_csvs(update: bool) -> list[str]:
    if update:
        run(REVIEWER_ARGS)
        print(f"  refreshed {COMMITTED_CSVS.relative_to(BASE)}")
        return []
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        run([*REVIEWER_ARGS, "--out-dir", tmp])
        for committed in sorted(COMMITTED_CSVS.glob("*.csv")):
            fresh = Path(tmp) / committed.name
            if not fresh.exists():
                failures.append(f"{committed.name}: not regenerated")
            elif not filecmp.cmp(committed, fresh, shallow=False):
                failures.append(f"{committed.name}: differs from committed copy")
            else:
                print(f"  ok   {committed.name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate the committed reference instead of checking against it.",
    )
    args = parser.parse_args()

    print("[1/2] console-output experiments (Tables II and III, ablation, diagnostics)")
    failures = verify_stdout(args.update)
    print("[2/2] reviewer audit tables")
    failures += verify_csvs(args.update)

    if args.update:
        print("\nReference regenerated. Commit the result.")
        return
    if failures:
        print("\nREPRODUCTION FAILED:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("\nREPRODUCTION OK: every reported artifact matches the committed reference.")


if __name__ == "__main__":
    main()
