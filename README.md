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

All four figures from the paper come out of one Snakemake pipeline:

| Paper figure | File produced |
|---|---|
| Fig. 3 — main synthetic results (`synth_main_combined.pdf`) | `src/expt/workflow/results/synth_main_combined.pdf` |
| Fig. 4 — scalability vs disjointCycles (`scalability_disjointcycles.pdf`) | `src/expt/workflow/results/scalability_disjointcycles.pdf` |
| Fig. 5 (App. C.1) — strict disjoint-cycles micro experiment | `src/expt/workflow/results/disjoint_micro.pdf` |
| Fig. 6 (App. C.2) — threshold sensitivity | `src/expt/workflow/results/synth_threshold.pdf` |

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

## Experiment grids

Three independent grids share one set of generate/fit/evaluate scripts. Defaults are in `src/expt/workflow/rules/synth.smk` and `rules/threshold.smk`.

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

## Pipeline stages

Each stage is a Snakemake rule wrapping a script in `src/expt/workflow/scripts/`:

1. **`generate_synth.py`** — Draws a random cyclic graph + samples observational data. Output: `dataset.npz` with `weights`, `scc_labels`, `scc_sizes`, `obs`.
2. **`fit.py`** — Runs ICA-LiNG-D (`run_lingd` from `src.repare_cycle.lingd`): FastICA → permutation search → candidate $\hat B$ → Tarjan. Output: `model.pkl`.
3. **`evaluate_synth.py`** — Computes ARI, cluster-DAG F1, variable-level F1, inter-SCC F1, SHD against ground truth. Output: `metrics.csv`.
4. **`collect.py`** — Concatenates all per-cell `metrics.csv` into a single CSV per grid.
5. **`plot_*.py`** — One plotting script per paper figure.

The R baseline is invoked from `disjointcycles_fit.R` (called by `fit.py` when `method=disjointcycles`).

## Regimes

- **`hard`** — weights in [0.5, 0.95], rescaled to ρ(B) ≈ 0.9. Stable; the first-stable filter recovers the true $B$.
- **`unstable`** — same weight pool rescaled to ρ(B) = 1.5. The true $B$ violates the stability condition, so the first-stable filter necessarily picks a cycle-reversed twin. The condensation is identical (Theorem 1).

## Pick strategies (`fit.py`)

- **`first_stable`** — first candidate with ρ(B̂) < 1. Used by the d=10 main grid.
- **`random_admissible`** — uniform draw from the full distributional equivalence class. Used by `lacerda_rnd` to expose the variable-level identifiability gap in Fig. 3 row 3.
- **`hungarian_any`** — single admissible permutation via the Hungarian algorithm. Used by the scalability and disjoint-micro grids; O(d³) and sufficient because Theorem 1 makes the condensation invariant across the equivalence class.

## Metrics

- **ARI (SCC partition)** — Adjusted Rand Index of the predicted vs true SCC labels.
- **Cluster-DAG F1** — predicted variable-level edges projected onto the true SCC partition, compared to the true cluster-DAG. The identification target of Theorem 1.
- **Variable-level F1** — F1 over all directed edges. Not identifiable in general; the gap to cluster-DAG F1 is the unidentifiable intra-SCC structure.
- **Inter-SCC F1** — variable-pair F1 restricted to pairs crossing SCC boundaries. Diagnostic.

## Project layout

```
src/
  repare_cycle/             # Library
    graph.py                # Generators, CyclicLinearSEM, Tarjan, scc_partition
    lingd.py                # ICA-LiNG-D (run_lingd)
    examples.py             # Hard-coded SEMs from the literature (e.g. Lacerda 2008)
  expt/workflow/            # Snakemake pipeline
    Snakefile               # Top-level: lists the four paper figures as targets
    rules/synth.smk         # Main + scalability + disjoint-micro grids
    rules/threshold.smk     # Threshold-sensitivity sweep
    scripts/                # generate / fit / evaluate / collect / plot_*
    results/                # Outputs (created on first run)
tests/
  test_repare_cycle.py
```

## Tests

```bash
pytest tests/
```
