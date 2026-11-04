#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../kev/.venv/bin/python" "${SCRIPT_DIR}/evaluate_test.py" "$@"
