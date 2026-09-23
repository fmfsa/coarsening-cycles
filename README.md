# Coarsening Linear Non-Gaussian Causal Models with Cycles

Code accompanying the paper. Recovers the **condensation** $G^{\mathrm{sc}}$ (SCC partition + inter-SCC edges) of a linear non-Gaussian cyclic SCM from observational data, using ICA-LiNG-D + Tarjan.

The library lives in `src/repare_cycle/`; experiments are orchestrated by Snakemake under `src/expt/workflow/`.

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate repare_cycle
pip install -e .
```

### pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 (tested on 3.13).

### R + disjointCycles (only for the scalability comparison)

The scalability and disjoint-micro figures compare against the [disjointCycles](https://github.com/ysamwang/disjointCycles) R package. To run those targets you need R (≥ 4.0) on `PATH`:

```r
install.packages(c("pak", "jsonlite"))
pak::pak("ysamwang/disjointCycles")
```

Without R, the d=10 main grid and threshold-sensitivity sweep still run; only `scalability_disjointcycles.pdf` and `disjoint_micro.pdf` will fail.

## Reproducing the paper figures

The figures come out of one Snakemake pipeline. The historical Fig. 5 data
are included; that figure is replotted without running simulations.

| Paper figure | File produced |
|---|---|
| Fig. 3 — paper grid through n=100,000 | `src/expt/workflow/results/paper_fig3.pdf` |
| Fig. 4 — scalability vs disjointCycles (`scalability_disjointcycles.pdf`) | `src/expt/workflow/results/scalability_disjointcycles.pdf` |
| Fig. 5 — sample complexity (saved results) | `src/expt/workflow/results/sample_complexity.pdf` |
| Fig. 6 (App. C.1) — strict disjoint-cycles micro experiment | `src/expt/workflow/results/disjoint_micro.pdf` |
| Fig. 7 (App. C.2) — threshold sensitivity | `src/expt/workflow/results/synth_threshold.pdf` |

To reproduce all of them:

```bash
cd src/expt/workflow
snakemake --cores all
```

The full sweep takes a few hours on a CPU workstation (most time goes into the d=100 cells of the scalability sweep). Intermediate fits are cached on disk; reruns are fast.

### Reproducing one figure at a time

```bash
cd src/expt/workflow

# Fig. 3 (main synthetic results)
snakemake results/paper_fig3.pdf --cores all

# Fig. 4 (scalability)
snakemake results/scalability_disjointcycles.pdf --cores all

# Fig. 5 (replot the saved CSV; no simulations)
snakemake results/sample_complexity.pdf --cores 1

# Fig. 6 (disjoint-cycles micro)
snakemake results/disjoint_micro.pdf --cores all

# Fig. 7 (threshold sensitivity)
snakemake results/synth_threshold.pdf --cores all
```

### Smoke test (a single fit)

```bash
snakemake "results/synth/regime=hard/d=10/num_cycles=3/density=0.3/samp_size=1000/seed=0/method=lacerda/metrics.csv" --cores 1
```

### Dry run

```bash
snakemake -n
```

### Additional experiments

Three additional experiments that are not part
of `snakemake --cores all` (the default target builds the five paper figures
listed above plus legacy main-grid panels); build them explicitly:

```bash
cd src/expt/workflow

# Candidate-selection ablation: every branch shares one FastICA estimate per
# dataset (verified via stored W hashes), so differences isolate selection.
snakemake results/synth_ablation.csv results/synth_ablation_scal.csv results/ablation_selection.pdf --cores all

# Split/merge failure-mode diagnostics (default tau, plus the threshold sweep)
snakemake results/split_merge.pdf results/threshold_split_merge.pdf --cores all

# Whole-SCC hard-intervention experiment
snakemake results/intervention_effects.csv results/intervention_effects.pdf --cores all
```

## Experiment grids

The grids share one set of generate/fit/evaluate scripts. Defaults are in `src/expt/workflow/rules/synth.smk`, `rules/threshold.smk`, `rules/ablation.smk`, and `rules/intervention.smk`.

**Main d=10 grid** (Fig. 3) — `rules/synth.smk`:

| Parameter | Default |
|---|---|
| Methods | `["lacerda", "lacerda_rnd"]` |
| Regimes | `["hard", "unstable"]` |
| Number of nodes | `[10]` |
| Non-trivial SCCs | `[3, 4, 5]` |
| Edge densities | `[0.3, 0.5, 0.8]` |
| Sample sizes | `[50, 100, 500, 1000, 5000]` |
| Seeds | `range(10)` |
| Noise | symmetric Laplace |

**Scalability grid** (Fig. 4) — `rules/synth.smk`:

| Parameter | Default |
|---|---|
| Methods | `["lacerda", "disjointcycles"]` |
| Regime | `"hard"` |
| Density | `0.5` |
| Non-trivial SCCs | `10` (fixed) |
| Number of nodes | `[20, 50, 100]` |
| Sample sizes | `[50, 100, 500, 1000, 5000, 10000, 50000, 100000]` |
| Seeds | `range(10)` |
| Noise | skewed exponential (so disjointCycles' third-moment tests carry signal) |

**Strict disjoint-cycles micro grid** (Fig. 6) — `rules/synth.smk`. Each SCC is a single Hamilton cycle (no intra-SCC chords), so all simple cycles are pairwise vertex-disjoint by construction.

| Parameter | Default |
|---|---|
| Methods | `["lacerda", "disjointcycles"]` |
| d, κ | `20`, `5` |
| Density | `0.5` (inter-block only) |
| Sample sizes | `[100, 500, 1000, 5000, 10000, 50000, 100000]` |
| Seeds | `range(10)` |
| Noise | skewed exponential |

**Threshold-sensitivity grid** (Fig. 7) — `rules/threshold.smk`:

| Parameter | Default |
|---|---|
| Method | `lacerda` (first-stable) |
| Regime | `"hard"` |
| d, κ, λ | `10`, `4`, `0.5` |
| Sample sizes | `[500, 5000, 50000]` |
| Thresholds τ | `[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]` |
| Seeds | `range(10)` |

**Candidate-selection ablation** — `rules/ablation.smk`:

| Parameter | Default |
|---|---|
| Branches | `abl_first_stable`, `abl_random_r{0..4}` (uniform pick after enumeration), `abl_hungarian`, `abl_arbitrary_r{0..4}` (unconstrained control); `abl_randhun_r{0..4}` (random-cost matching) is produced for provenance |
| Main grid | the d=10 grid above with sample sizes extended by `10000` |
| Cost arm | (d, κ) ∈ {(5, 2), (10, 5), (20, 10)}, both regimes, n ∈ {1000, 10000, 50000} |
| Enumeration safeguards | 10,000-candidate cap and a 60 s wall-clock budget; truncation is recorded in `enumeration_cap_hit` / `enumeration_timed_out`, never silent |

**Whole-SCC intervention grid** — `rules/intervention.smk`. Clamps the largest
non-trivial SCC (with ≥ 1 downstream SCC) at `μ̂ + σ̂` and compares the analytic
post-intervention means of the generating SEM against block-level least-squares
propagation through the true and the recovered condensation:

| Parameter | Default |
|---|---|
| Regimes | `["hard", "unstable"]` |
| d, κ, λ | `10`, `4`, `0.5` |
| Sample sizes | `[100, 500, 1000, 5000, 50000]` |
| Seeds | `range(20)` |

## Pipeline stages

Each stage is a Snakemake rule wrapping a script in `src/expt/workflow/scripts/`:

1. **`generate_synth.py`** — Draws a random cyclic graph + samples observational data. Output: `dataset.npz` with `weights`, `scc_labels`, `scc_sizes`, `obs`, and the noise parameters (`noise_means`, `noise_vars`, `noise_dist`) used by the intervention experiment.
2. **`fit.py`** — Runs ICA-LiNG-D (`run_lingd` from `src.repare_cycle.lingd`): FastICA → permutation search → candidate $\hat B$ → Tarjan. Output: `model.pkl`.
3. **`evaluate_synth.py`** — Computes ARI, cluster-DAG F1, variable-level F1, inter-SCC F1, SHD against ground truth. Output: `metrics.csv`.
4. **`collect.py`** — Concatenates all per-cell `metrics.csv` into a single CSV per grid.
5. **`plot_*.py`** — One plotting script per paper figure.

The R baseline is invoked from `disjointcycles_fit.R` (called by `fit.py` when `method=disjointcycles`).

The additional experiments add their own fit scripts on top of the same
generate/evaluate/collect stages: `ablation_fit.py` (one job per dataset, one
`model.pkl` per selection branch, all consuming the identical FastICA estimate),
`ablation_randhun_fit.py`, `ablation_arbitrary_fit.py`, and
`intervention_run.py` (analytic `do(X_S = c)` truth vs block-regression
estimates, with strict dataset-level structural-recovery accounting).

## Regimes

- **`hard`** — weights in [0.5, 0.95], rescaled to ρ(B) ≈ 0.9. Stable; the first-stable filter recovers the true $B$.
- **`unstable`** — same weight pool rescaled to ρ(B) = 1.5. The true $B$ violates the stability condition, so the first-stable filter necessarily picks a cycle-reversed twin. The condensation is identical (Theorem 1).

## Pick strategies (`fit.py`)

- **`first_stable`** — first candidate with ρ(B̂) < 1. Used by the d=10 main grid.
- **`random_admissible`** — uniform draw from the full distributional equivalence class. Used by `lacerda_rnd` to expose the variable-level identifiability gap in Fig. 3 row 3.
- **`hungarian_any`** — single admissible permutation via the Hungarian algorithm on costs −log|Ŵᵢⱼ|. Used by the scalability and disjoint-micro grids; O(d³) and sufficient because Theorem 1 makes the condensation invariant across the equivalence class.

`run_lingd` is a thin wrapper over a staged API (`fit_ica_unmixing` →
`enumerate_admissible_candidates` → `choose_enumerated_candidate`, or
`hungarian_candidate`), so experiments can hold the ICA estimate fixed while
varying only selection. Two further candidate functions exist for the
ablation: `random_matching_candidate` (random-cost assignment, admissible but
not uniform, no enumeration) and `arbitrary_permutation_candidate` (uniform
row permutation, admissibility measured but not enforced). Enumeration is
guarded by an O(d³) existence pre-check and a wall-clock budget; candidate
counts, truncation flags and per-stage runtimes are recorded on every fit.

## Metrics

- **ARI (SCC partition)** — Adjusted Rand Index of the predicted vs true SCC labels.
- **Cluster-DAG F1** — predicted variable-level edges projected onto the true SCC partition, compared to the true cluster-DAG. The identification target of Theorem 1.
- **Variable-level F1** — F1 over all directed edges. Not identifiable in general; the gap to cluster-DAG F1 is the unidentifiable intra-SCC structure.
- **Inter-SCC F1** — variable-pair F1 restricted to pairs crossing SCC boundaries. Diagnostic.
- **Split/merge rates** (`repare_cycle.metrics`) — pairwise failure-mode diagnostics: a split pair is two variables of one true SCC placed in different estimated clusters, a merge pair two variables of different true SCCs placed together. Rates are normalised by the eligible pair counts; each run is classified exact / split-only / merge-only / mixed.
- **Selection metadata** — `n_candidates_enumerated`, `n_candidates_returned`, `enumeration_cap_hit`, `enumeration_timed_out`, `ica_runtime_sec`, `selection_runtime_sec` (NaN for fits predating the staged pipeline).

## Project layout

```
src/
  repare_cycle/             # Library
    graph.py                # Generators, CyclicLinearSEM, Tarjan, scc_partition
    lingd.py                # ICA-LiNG-D, staged: ICA / enumeration / selection
    metrics.py              # Split/merge partition diagnostics
    intervention.py         # Whole-SCC hard interventions + block regression
    examples.py             # Hard-coded SEMs from the literature (e.g. Lacerda 2008)
  expt/workflow/            # Snakemake pipeline
    Snakefile               # Top-level: paper figures and legacy panels
    rules/synth.smk         # Main + scalability + disjoint-micro grids
    rules/threshold.smk     # Threshold-sensitivity sweep
    rules/ablation.smk      # Candidate-selection ablation (+ cost arm)
    rules/intervention.smk  # Whole-SCC intervention experiment
    scripts/                # generate / fit / evaluate / collect / plot_*
    results/                # Outputs (created on first run)
tests/
  test_repare_cycle.py
  test_lingd_refactor.py    # Staged API ≡ run_lingd; safeguard behaviour
  test_metrics.py           # Split/merge diagnostics
  test_intervention.py      # Convention check, Monte Carlo, block regression
```

## Tests

```bash
pytest tests/
```

## September paper revision: P3.8–9 and P4.10

Use Python 3.11 for the benchmark: `lingam==1.13.0` constrains SciPy to
<=1.13.1, which has no Python 3.13 wheel. From the repository root:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e . --group dev --group experiments
source .venv/bin/activate
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd src/expt/workflow

# Fig. 5: uses ../../../data/reference/sample_complexity.csv verbatim.
snakemake results/sample_complexity.pdf --cores 1

# Fig. 3: explicit paper grid to n=100,000; original smaller grids unchanged.
snakemake results/paper_fig3.pdf --cores 4

# Ablation: label corrections from saved results, no new fits.
snakemake results/paper_ablation_selection.pdf --cores 1

# Fig. 6-style ablation comparison: ARI, cluster-DAG F1, and selection time.
# Reuses saved timings; also exports per-cell data, medians, and a caption.
snakemake results/ablation_accuracy_runtime.pdf --cores 1

# Quick paired benchmark: 18 datasets, 36 fits, 120 s budget per fit.
# Run timed comparisons on an otherwise idle machine.
snakemake results/group_lingam/pilot/comparison.pdf --cores 1 --resources benchmark_slot=1

# Runtime feasibility: seed 0, n=1000 and 2000, both regimes, 3600 s/fit.
# This is a runtime check, not a 10-seed accuracy comparison.
snakemake results/group_lingam/feasibility/sample_sizes.pdf --cores 1 --resources benchmark_slot=1

# Subsequent feasibility check at n=5000: seed 0, 7200 s/fit.
snakemake results/group_lingam/large_probe/sample_sizes.pdf --cores 1 --resources benchmark_slot=1

# Full server benchmark: 10 seeds; six sizes through n=10,000.
# Limits by sample size are declared in config/group_lingam_full.yaml.
cd ../../..
bash scripts/run_group_lingam_full.sh --dry-run
# Launch later, on the server:
# bash scripts/run_group_lingam_full.sh
```

The benchmark also produces `metrics.csv`, `summary.csv`, `report.md`, and
one JSON per attempt with data/code hashes, versions, groups, known edges,
wall/CPU times, memory, and failure status. A completed timeout is cached;
delete that specific JSON (or force its rule) to retry it deliberately.
Do not run other experiments concurrently with timed comparisons.

`sample_sizes.pdf` is the Fig. 6-style layout (ARI, cluster-DAG F₁, runtime;
stable/unstable rows). Its `sample_sizes_table.md` reports completion counts
and exact recovery separately. The original pilot `comparison.pdf` remains
available. A one-seed feasibility figure has no estimated uncertainty band.
The full server grid includes n=2000 and uses longer, explicit per-size time
limits. See [server setup and run instructions](docs/group_lingam_server.md)
and [the exact GitHub staging manifest](docs/github_push_files.txt).

`scripts/reuse_group_lingam.py` can initialize a new profile with compatible
completed fits. It verifies observation and implementation hashes, package
versions, machine platform, and fit settings; it never reuses timeouts or
overwrites source results. Reused JSONs retain their source path and hash.

Fig. 3's paper target uses only first-stable fits, with 10 seeds for both
regimes, all three densities and SCC counts, and sample sizes
50, 100, 500, 1000, 5000, 10000, 50000, 100000. It does not extend the
ablation grid. It reuses the 900 saved first-stable cells through n=5,000
and runs only the 540 missing high-n cells. The output CSV marks historical
versus new results; original low-n environment/logs are unavailable. The
main-grid target above retains its smaller legacy range and can regenerate
the low-n fits independently.

The Fig. 5 reference CSV is versioned and checksummed in `data/reference/`.
The legacy `results/sample_complexity.csv` target exports this saved CSV.
The optional `results/sample_complexity_rerun.csv` target runs the historical
generator separately; it is not a dependency of any figure and does not
overwrite the reference data. The original generator's failure handling is
historical and has not been revised or used for this figure update.

See `docs/paper_revision_handoff.md` for paper wording, limitations, and results.
`experiments-requirements.lock` records the installed Python 3.11 environment;
from the root, `uv pip install --python .venv/bin/python -r experiments-requirements.lock`
recreates its package versions.
