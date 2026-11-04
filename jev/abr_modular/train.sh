#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEV_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VARIANT="${1:?Usage: bash train.sh baseline|cost|regret|both [--resume] [other trainer options]}"
shift
case "${VARIANT}" in
  baseline) COST=0; REGRET=0 ;;
  cost) COST=1; REGRET=0 ;;
  regret) COST=0; REGRET=1 ;;
  both) COST=1; REGRET=1 ;;
  *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac
export HF_HOME="${JEV_DIR}/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "${JEV_DIR}/kev/.venv/bin/python" "${SCRIPT_DIR}/train.py" \
  --candidate-cost "${COST}" --regret-loss "${REGRET}" "$@"
