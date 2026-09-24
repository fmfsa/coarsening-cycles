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

All figures from the paper come out of one Snakemake pipeline:

| Paper figure | File produced |
|---|---|
| Fig. 3 — main synthetic results (`synth_main_combined.pdf`) | `src/expt/workflow/results/synth_main_combined.pdf` |
| Fig. 4 — scalability vs disjointCycles (`scalability_disjointcycles.pdf`) | `src/expt/workflow/results/scalability_disjointcycles.pdf` |
| Fig. 5 (App. C.1) — strict disjoint-cycles micro experiment | `src/expt/workflow/results/disjoint_micro.pdf` |
| Fig. 6 (App. C.2) — threshold sensitivity | `src/expt/workflow/results/synth_threshold.pdf` |
| App. B.6 — sample-complexity corroboration of Prop. 3 | `src/expt/workflow/results/sample_complexity.pdf` |

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
snakemake results/synth_main_combined.pdf --cores all

# Fig. 4 (scalability)
snakemake results/scalability_disjointcycles.pdf --cores all

# Fig. 5 (disjoint-cycles micro)
snakemake results/disjoint_micro.pdf --cores all

# Fig. 6 (threshold sensitivity)
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
of `snakemake --cores all` (the default target still builds exactly the four
paper figures); build them explicitly:

```bash
cd src/expt/workflow

# Original candidate-selection grid and scalability diagnostic.
# For the paper's final d=20, 30-seed table, see "d=20 candidate-selection
# ablation" below.
snakemake results/synth_ablation.csv results/synth_ablation_scal.csv results/ablation_selection.pdf --cores all

# Split/merge failure-mode diagnostics (default tau, plus the threshold sweep)
snakemake results/split_merge.pdf results/threshold_split_merge.pdf --cores all

# Whole-SCC hard-intervention experiment
snakemake results/intervention_effects.csv results/intervention_effects.pdf --cores all

# Sample-complexity corroboration of Prop. 3 (App. B.6; ICLR Fig. 5)
snakemake results/sample_complexity.pdf --cores all
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

**Strict disjoint-cycles micro grid** (Fig. 5) — `rules/synth.smk`. Each SCC is a single Hamilton cycle (no intra-SCC chords), so all simple cycles are pairwise vertex-disjoint by construction.

| Parameter | Default |
|---|---|
| Methods | `["lacerda", "disjointcycles"]` |
| d, κ | `20`, `5` |
| Density | `0.5` (inter-block only) |
| Sample sizes | `[100, 500, 1000, 5000, 10000, 50000, 100000]` |
| Seeds | `range(10)` |
| Noise | skewed exponential |

**Threshold-sensitivity grid** (Fig. 6) — `rules/threshold.smk`:

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
  expt/
    config/ablation_d20.json  # d=20 selection-ablation configuration
    workflow/               # Snakemake pipeline
      Snakefile             # Top-level: lists the four paper figures as targets
      rules/synth.smk       # Main + scalability + disjoint-micro grids
      rules/threshold.smk   # Threshold-sensitivity sweep
      rules/ablation.smk    # Candidate-selection ablation (+ cost arm)
      rules/intervention.smk  # Whole-SCC intervention experiment
      scripts/              # generate / fit / evaluate / collect / plot_*
      results/              # Outputs; paper data are committed, the rest is gitignored
scripts/                    # d=20 ablation runner/report; figure replotting
tests/
  test_repare_cycle.py
  test_lingd.py             # Candidate selection, enumeration limits, timing
  test_selection_ablation.py # Exact recovery, seed aggregation, shared budget
  test_metrics.py           # Split/merge diagnostics
  test_intervention.py      # Convention check, Monte Carlo, block regression
```

## Tests

```bash
pytest tests/
```

## Paper figures from saved data

`scripts/regenerate_paper_figures.py` replots figures from committed CSVs
without launching any fits (missing data raise an explicit error). Output goes
to `output/pdf/`.

```bash
.venv/bin/python scripts/regenerate_paper_figures.py              # Figs 3 and 5
.venv/bin/python scripts/regenerate_paper_figures.py --figures 4 7 8
```

| Fig. | Data (under `src/expt/workflow/results/`) | Output |
|---|---|---|
| 3 | `paper_fig3.csv.gz` | `fig3_main.pdf` |
| 4 | `synth_scalability.csv` | `scalability_disjointcycles.pdf` |
| 5 | `sample_complexity.csv` | `fig5_sample_complexity.pdf` |
| 7 | `synth_disjoint.csv` | `disjoint_micro.pdf` |
| 8 | `synth_threshold.csv` | `synth_threshold.pdf` |

Use `--fig{N}-data PATH` to point at data stored elsewhere.
`paper_fig3.csv.gz` is a 1,440-row snapshot (first-stable; 2 regimes × 8 sample
sizes × κ ∈ {3,4,5} × density ∈ {0.3,0.5,0.8} × 10 seeds), combining 900
historical cells at n ≤ 5000 with 540 later extension cells. Absolute runtimes
are not comparable across those two batches.

## d=20 candidate-selection ablation (paper table)

A standalone runner, outside Snakemake, with separate ICA and selection timers.
Configuration: `src/expt/config/ablation_d20.json`. Measurements:
`src/expt/workflow/results/ablation_d20/`. It was run on Python 3.11; besides
the package it needs `pandas matplotlib seaborn pytest`.

```bash
bash scripts/run_ablation_d20.sh                         # fits; single-threaded BLAS
.venv/bin/python scripts/report_selection_ablation.py   # table -> output/pdf/ablation_d20_table*
.venv/bin/python scripts/compare_ablation_selection.py --input src/expt/workflow/results/ablation_d20
```

The report and the paired comparison read saved measurements only. The runner
resumes a results directory only if configuration, source hashes and
environment all match `protocol.json`; the committed copy has its `machine`
field blanked, so collect fresh timings in a new directory
(`run_ablation_d20.sh --output DIR`, then `report_selection_ablation.py --input DIR --output FILE.pdf`).

**Setup.** d=20, κ=5 nontrivial SCCs, density 0.5, Laplace noise; stable and
unstable (spectral radius rescaled to 1.5) regimes; n ∈ {10,000, 50,000};
seeds 0–29 (120 datasets). One FastICA estimate per dataset (max_iter=10,000,
tol=1e-6, random_state=0) is shared by all four methods. Thresholds start at
0.1; if no candidate is found the W threshold is halved down to 0.01. Enumeration
has no candidate-count cap and a 60 s total budget per dataset, shared across
threshold rounds.

**Methods.** Enumeration + first-stable (falls back to the first unstable
candidate), enumeration + uniform random (samples the returned, possibly
timeout-truncated, list), ours (Hungarian assignment; no enumeration), and a
control of arbitrary row permutations without admissibility constraints. The
two randomized methods use five draws per dataset.

**Measurement.** ICA time covers fitting and retries after imports. Selection
time covers enumeration and stability checks; the shared enumeration cost is
charged to both enumeration methods. Data generation, scoring and I/O are
excluded, and branch order rotates across seeds. Draws are averaged within a
dataset before aggregating over seeds; ARI and exact recovery (SCC partition
*and* condensation edges correct) are means, times are medians. A missing
estimate has missing, not zero, accuracy. Bold selection times in the table mark
a median paired selection/ICA ratio above one.

**Files.** `protocol.json` (config, source hashes, environment, threads);
per-dataset `result.json` and `estimate.npz` (ICA estimate and ground truth);
`draws.csv`, `cells.csv`, `summary.csv` (draw-, dataset- and aggregate-level);
`verification.json` (run checks: timeouts, missing estimates, largest candidate
list, control admissibility).
