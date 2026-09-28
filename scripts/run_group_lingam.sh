#!/usr/bin/env bash
# GroupLiNGAM vs ours over sample size. Needs a Python 3.11 .venv at the repo
# root with src/expt/config/group_lingam_requirements.txt installed (see README).
# Resumable: completed fits are cached as JSON; rerun this script after an interruption.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_path="$repo_dir/.venv/bin/python"
snakemake_path="$repo_dir/.venv/bin/snakemake"
config="$repo_dir/src/expt/config/group_lingam.yaml"
if [[ ! -x "$python_path" || ! -x "$snakemake_path" ]]; then
  echo "Create .venv and install src/expt/config/group_lingam_requirements.txt first (see README)." >&2
  exit 1
fi
export MPLBACKEND=Agg
export MPLCONFIGDIR="$repo_dir/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
# Concurrent single-threaded fits, declared in the config (default 1).
slots=$("$python_path" -c "import sys, yaml; print(int(yaml.safe_load(open(sys.argv[1]))['group_lingam'].get('parallel_fits', 1)))" "$config")
# A fresh source cache avoids using stale cached scripts after git pull.
task_cache=$(mktemp -d "${TMPDIR:-/tmp}/coarsening-snakemake.XXXXXX")
"$snakemake_path" results/group_lingam_sample_sizes.pdf \
  --snakefile "$repo_dir/src/expt/workflow/Snakefile" \
  --directory "$repo_dir/src/expt/workflow" \
  --configfile "$config" \
  --runtime-source-cache-path "$task_cache" \
  --cores "$slots" --resources benchmark_slot="$slots" --rerun-incomplete "$@"
