#!/usr/bin/env bash
# Run from any directory; install .venv using the server instructions first.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_path="$repo_dir/.venv/bin/python"
snakemake_path="$repo_dir/.venv/bin/snakemake"
if [[ ! -x "$python_path" || ! -x "$snakemake_path" ]]; then
  echo "Create .venv and install experiments-requirements.lock first (see docs/group_lingam_server.md)." >&2
  exit 1
fi
export MPLBACKEND=Agg
export MPLCONFIGDIR="$repo_dir/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
# A fresh source cache avoids using stale cached scripts after git pull.
task_cache=$(mktemp -d "${TMPDIR:-/tmp}/coarsening-snakemake.XXXXXX")
"$snakemake_path" results/group_lingam/full/sample_sizes.pdf \
  --snakefile "$repo_dir/src/expt/workflow/Snakefile" \
  --directory "$repo_dir/src/expt/workflow" \
  --configfile "$repo_dir/config/group_lingam_full.yaml" \
  --runtime-source-cache-path "$task_cache" \
  --cores 1 --resources benchmark_slot=1 --rerun-incomplete "$@"
