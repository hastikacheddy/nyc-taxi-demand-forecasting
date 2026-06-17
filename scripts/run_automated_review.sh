#!/usr/bin/env bash
#
# Run the full local quality + security review (same checks as CI).
#   pip install -r requirements-dev.txt
#   bash scripts/run_automated_review.sh
set -uo pipefail
fail=0

run() { echo; echo "== $1 =="; shift; "$@" || fail=1; }

run "flake8 (lint)"                 flake8 src/ tests/ dags/ testing/
run "bandit (security, medium+)"    bandit -r src dags -ll
run "semgrep (architecture rules)"  semgrep --error --config .semgrep/ src dags
run "radon (complexity report)"     radon cc src dags -s -a

echo
if [[ "$fail" -eq 0 ]]; then echo "✅ all checks passed"; else echo "❌ some checks failed"; fi
exit "$fail"
