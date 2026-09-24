#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$repo_dir/.cache/matplotlib"
# The runner's own --config default predates the move; it is left unedited so
# its source hash still matches results/ablation_d20/protocol.json.
exec "$repo_dir/.venv/bin/python" scripts/run_selection_ablation.py \
  --config src/expt/config/ablation_d20.json "$@"
