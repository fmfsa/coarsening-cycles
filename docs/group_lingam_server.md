# Full GroupLiNGAM experiment: GitHub and server instructions

This runs both methods from scratch on the server. Do not mix laptop timings
with server timings. It does not require saved laptop results, R, or the paper
source. No commit, push, or full experimental run has been performed by Codex.

## Push this revision from the laptop

The exact file list is `docs/github_push_files.txt`. It includes the benchmark,
workflow, tests, plotting updates, environment lock, and small saved reference
CSVs needed by the other paper-figure targets. Existing unchanged library and
generator files are already tracked in the repository.
The repository also tracks some older easy-regime results; the full target
does not use those files.

It excludes `.DS_Store`, `.venv/`, `.cache/`, `.claude/`, generated `results/`,
`output/`, and logs. The copied `AGENTS.md` contains paper-planning context and
is not needed to run the experiments; it is not in this staging manifest.

From `/Users/fmfsa/Repositories/coarsening-cycles`, review and run:

```bash
git switch -c codex/paper-experiments
git add --pathspec-from-file=docs/github_push_files.txt
git diff --cached --stat
git diff --cached --check
git commit -m "Add paper revision plots and paired GroupLiNGAM benchmark"
git push -u origin codex/paper-experiments
```

If using a different branch name, substitute it in both laptop and server
commands. These commands are provided for the user; nothing is automatically
committed or pushed. Review any previously staged files before committing.

## Install on the server

Use Python 3.11. The older `environment.yml` targets Python 3.13 and is not
the environment for this benchmark: lingam 1.13.0 requires SciPy <=1.13.1.

```bash
git clone --branch codex/paper-experiments git@github.com:fmfsa/coarsening-cycles.git
cd coarsening-cycles
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r experiments-requirements.lock
.venv/bin/pytest -q
bash scripts/run_group_lingam_full.sh --dry-run
```

For an existing clean server clone, use `git fetch origin`,
`git switch codex/paper-experiments`, and
`git pull --ff-only origin codex/paper-experiments` instead of cloning.
The lock records the laptop's installed package versions. Installation on
Linux has not been executed here; retain any server-specific installation
changes and the resolved package versions with the run.

## Grid and budgets

The versioned configuration is `config/group_lingam_full.yaml`:

- d=10, κ=4, density=0.5, independent Laplace noise.
- Stable and unstable regimes, seeds 0–9.
- n=100, 500, 1000, 2000, 5000, 10000.
- Both methods see the same dataset per regime/n/seed: 120 datasets, 240 fits.
- GroupLiNGAM 1.13.0, alpha=0.01, native edge estimation.
- Ours uses magnitude-based Hungarian matching, thresholds 0.1,
  FastICA max_iter=10000 and tol=1e-6, with the dataset seed.

| n | Maximum fitting time per method/seed |
|---:|---:|
| 100 | 10 minutes |
| 500 | 10 minutes |
| 1000 | 30 minutes |
| 2000 | 2 hours |
| 5000 | 6 hours |
| 10000 | 24 hours |

These are declared ceilings, not predicted runtimes. The completed laptop
GroupLiNGAM fit at n=1000 took about 316 s; n=2000 and larger have not yet
been measured. The exact kernel-based implementation may take days for the
full sweep. It builds dense n-by-n matrices, so use a server allocation with
ample RAM (32 GiB or more is a prudent starting allocation, not a measured
requirement). A larger budget does not guarantee completion.

Change limits in the YAML **before** launching, if required. Both methods
receive the same limit for a given n. Keep the final configuration with the
results. After starting, do not change the data, code, environment, or budgets
and silently merge the resulting timings into the same run.

## Run and resume

On a dedicated server or within an appropriate scheduler allocation:

```bash
mkdir -p logs
cp config/group_lingam_full.yaml logs/group_lingam_full_config.yaml
git rev-parse HEAD > logs/group_lingam_full_commit.txt
.venv/bin/python -m pip freeze > logs/group_lingam_full_environment.txt
nohup bash scripts/run_group_lingam_full.sh > logs/group_lingam_full.log 2>&1 < /dev/null &
echo $! > logs/group_lingam_full.pid
```

On a managed cluster, use the same `bash scripts/run_group_lingam_full.sh`
command inside its batch scheduler instead of running on a login node.
The launcher sets one numerical thread and runs one job at a time, avoiding
competing timed fits and data generation. Do not increase parallelism for a
timing figure without changing and documenting that protocol for both methods.

Inspect progress with `tail -f logs/group_lingam_full.log`. Per-fit stdout,
errors, and convergence warnings are retained under
`src/expt/workflow/results/group_lingam/full/logs/`.

Rerun the same launcher to resume after interruption: completed JSONs are
cached and incomplete jobs are rerun. Timeouts are completed recorded
attempts, so a normal resume does not retry them. A larger-budget retry is a
separate protocol change; preserve the previous result and document it.

## Outputs to retrieve

All are under `src/expt/workflow/results/group_lingam/full/`:

- `sample_sizes.pdf`: ARI, true-partition-projected cluster-DAG F1, and **full
  fitting** time (not the ablation's candidate-selection-only time).
- `sample_sizes_summary.csv` and `sample_sizes_table.md`: accuracy, exact
  recovery, timing, completion counts, and declared budgets.
- `metrics.csv`: one record per method/regime/n/seed.
- `regime=*/n=*/seed=*/method=*.json`: groups, known edges, source/data hashes,
  versions, wall/CPU timing, sampled peak RSS, and status for every attempt.
- `logs/`: per-fit logs, including returned-estimate convergence warnings.

Retrieve the entire full-results directory and the launch metadata in `logs/`.
Keep the generated dataset files if exact binary replay is needed; their paths
are under `src/expt/workflow/results/synth/`. Results are gitignored, so use
file transfer or an artifact archive, not `git push`, to retrieve them.
