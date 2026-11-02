#!/bin/bash
#
# Local CI for msoaxon: what the Linux jobs in .github/workflows/test.yml
# check on Python 3.12, as one script (`bash tests/ci.sh`). GitHub does not
# run it: the workflows spread these checks over a Python 3.11-3.14 matrix,
# macOS, Windows and the Rust `fast` builds, which one script on one
# machine cannot stand in for (dev-toolchain#34). Keep the two in step.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Same ruff as .pre-commit-config.yaml and the lint job.
uvx ruff@0.16.10 format --check .
uvx ruff@0.16.10 check .
# The 3.12 row of the pytest job: the whole suite, slow tests included.
uv sync --locked --python 3.12
MPLBACKEND=Agg uv run --python 3.12 pytest -q
