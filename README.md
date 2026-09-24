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

### TVB (only for the brain-simulator pilot)

The TVB linear-equilibrium pilot (`snakemake tvb_all`) integrates [The Virtual Brain](https://www.thevirtualbrain.org) as a numerical check of the equilibrium model. Install the optional extra:

```bash
pip install -e ".[tvb]"    # pins tvb-library==2.10.0, tvb-data==3.0.0
```

Core installation is unchanged, and nothing else imports TVB. Without the extra, everything above still runs — including the pilot's generation/fit/evaluation stages, which are TVB-free; only the `tvb_validate`/`tvb_preflight` stages (and hence `tvb_all` and `tvb_pilot.pdf`) need it, and the TVB tests skip automatically.

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

### Additional experiments

Three additional experiments that are not part
of `snakemake --cores all` (the default target still builds exactly the four
paper figures); build them explicitly:

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

### TVB experiments

```bash
cd src/expt/workflow
snakemake tvb_all --cores all      # Experiment 1 (linear); needs `pip install -e ".[tvb]"`
snakemake tvb_rww_all --cores all  # Experiment 2 (Reduced Wong-Wang); needs Exp. 1 datasets + TVB
```

One-cell smoke test (TVB-free — generation, fitting, and evaluation don't import TVB):

```bash
snakemake "results/tvb/seed=0/samp_size=100/method=lacerda/metrics.csv" --cores 1
```

Dry run: `snakemake -n tvb_all`.

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

**TVB Experiment 1 grid** (`tvb_all`) — `rules/tvb.smk`:

| Parameter | Default |
|---|---|
| Method | `lacerda` (hungarian_any) |
| d, κ, λ (inter & intra) | `20`, `5`, `0.3` |
| Spectral radius | rescaled to `0.9` |
| Weight magnitudes | `[0.5, 0.95]` before rescaling |
| Inputs | standardized Laplace (mean 0, variance 1) |
| Sample sizes | prefixes `[100, 500, 2000, 5000]` of one 5000-sample dataset |
| Seeds | `range(5)` |

**TVB Experiment 2 grid** (`tvb_rww_all`) — `rules/tvb_rww.smk`:

| Parameter | Default |
|---|---|
| Model | Reduced Wong-Wang, one population per node |
| Graphs | same five benchmark graphs as Experiment 1 |
| Regime | calibrated (G, I_o); unique stable fixed point for every seed |
| Input amplitudes | Laplace scale `{small: 0.0005, medium: 0.002, large: 0.005}` nA (≈2%/9%/23% nonlinear), truncated at ±3 scales |
| Observations | every one integrated by TVB to `max_i |dS_i| ≤ 1e-7` |
| Sample sizes | prefixes `[100, 500, 2000, 5000]` of one 5000-sample dataset per (seed, amplitude) |
| Fits | LiNG-D unchanged, per-fit standardized, hungarian_any |
| Held-out validation | ± cluster stimulations per true SCC at tol 1e-10 |

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

The TVB experiments add their own generation/evaluation/validation/preflight/plot scripts (`generate_tvb.py`, `evaluate_tvb.py`, `validate_tvb.py`, `intervention_descendants_tvb.py`, `preflight_tvb.py`, `plot_tvb*.py`, plus the Wong-Wang stages `calibrate/generate/verify/stimulate_rww_tvb.py` and `score_rww_descendants.py`) while reusing `fit.py` (with opt-in `sample_limit`/`standardize` params) and `collect.py` unchanged.

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
- **Variable-level F1** — F1 over all directed edges of the *selected LiNG-D representative* (`model.full_adj_ij`), an intermediate object. Not identifiable in general; the gap to cluster-DAG F1 is the unidentifiable intra-SCC structure. A diagnostic, never algorithm performance.
- **Inter-SCC F1** — variable-pair F1 restricted to pairs crossing SCC boundaries. Diagnostic.
- **Split/merge rates** (`repare_cycle.metrics`) — pairwise failure-mode diagnostics: a split pair is two variables of one true SCC placed in different estimated clusters, a merge pair two variables of different true SCCs placed together. Rates are normalised by the eligible pair counts; each run is classified exact / split-only / merge-only / mixed.
- **Selection metadata** — `n_candidates_enumerated`, `n_candidates_returned`, `enumeration_cap_hit`, `enumeration_timed_out`, `ica_runtime_sec`, `selection_runtime_sec` (NaN for fits predating the staged pipeline).
- **Cross-SCC descendant F1** (TVB pilot) — transitive closures of the true and estimated condensations, expanded to ordered variable pairs and scored only on pairs from *different true SCCs* — so no alignment between estimated and true cluster labels is needed. Over-merging predicts both directions between two true SCCs and costs precision.
- **Direct C-DAG F1** (TVB pilot) — exact-match comparison of the recovered C-DAG (`model.dag`) against the true condensation: an edge counts only when both endpoint clusters equal true SCCs exactly. Unlike the projected cluster-DAG F1 (which projects variable edges onto the *true* partition and can be nonzero even when the estimate collapses to a single cluster with no C-DAG edges), this evaluates the object the pipeline actually outputs. `partition_exact` flags an exact SCC-partition match.
- **Intervention-validated descendant F1** (TVB pilot, seed 0) — for each soft intervention, the variables the recovered C-DAG predicts to respond (stimulated clusters + their C-DAG descendants) scored against the variables that actually respond in the TVB steady state. An interventional validation of the recovered abstraction, not just of the numerics.

## Experiment 1: linear TVB compatibility pilot

A focused study (`snakemake tvb_all`) showing that condensation recovery works on equilibrium data from a linear model implemented in TVB, with numerical TVB validation: the fitted observations are computed analytically from the shared equilibrium equation (`generate_tvb.py`), and TVB integrates a representative subset (32 observational episodes) plus every soft intervention to confirm the two agree. In this linear system the intervention validation is largely a numerical confirmation of graph reachability; the nonlinear stress test where TVB generates *every* observation is Experiment 2 below. Outputs land in `src/expt/workflow/results/`: `tvb_results.csv` (recovery metrics for the 20-fit grid, including the direct C-DAG comparison), `tvb_equilibrium_validation.csv` (per-episode TVB-vs-exact errors), `tvb_intervention_descendants.csv` (C-DAG response predictions vs actual TVB stimulation responses), `tvb_default_preflight.csv` (default-connectome SCC statistics), the four-panel `tvb_pilot.pdf`, and the topology figure `tvb_topology.pdf` (true cyclic variable graph, true C-DAG, and recovered C-DAG, each SCC drawn as a colored cluster).

**Equilibrium derivation.** The bridge (`src/repare_cycle/tvb_bridge.py`, internal — not part of the public API) runs `models.Linear(gamma=-1)` with linear coupling `a=1`, zero conduction delays, deterministic Heun integration, and a constant regional stimulus $e$:

$$\dot x = -x + W_{\mathrm{tvb}}\,x + e \;\Rightarrow\; x^\ast = (I - W_{\mathrm{tvb}})^{-1} e,$$

which is exactly the cyclic-LiNG equilibrium $x = e\,(I - W)^{-1}$ the library samples from ([`graph.py`](src/repare_cycle/graph.py)). Each pilot observation is one such equilibrium for one standardized-Laplace input row; `tvb_validate` integrates 32 observational episodes and one soft intervention per true SCC (shift $1/\sqrt{|\mathrm{SCC}|}$ on each member) for seed 0, requiring residual ≤ 1e-7 (within 250 ms — the slowest seed-0 mode crosses that threshold just after 200 ms) and relative error ≤ 1e-5 against the exact solution. All 45 episodes converge with relative error ≤ 6e-7.

**Float32 history caveat.** Stock TVB 2.10 stores the simulator's history buffer and per-connection weights in float32 (`tvb/simulator/history.py`), so the coupling term — and hence the simulated fixed point — carries a ~1e-6 relative bias that no amount of integration time removes. The bridge swaps in a float64 `SparseHistory` subclass (identical logic, wider dtype) after `Simulator.configure()`; without it, the equilibrium residual plateaus around 1e-6 instead of reaching 1e-7.

**Matrix conventions.** Repository truth is `weights[src, dst]`; TVB connectivity is `weights[target, source]` ("to, from"). Conversion is a transpose in both directions (`to_tvb_orientation` / `from_tvb_orientation`), which preserves the SCC partition.

**What the algorithm outputs.** The scientific output is `model.dag`, the recovered C-DAG (cluster nodes = variable sets, guaranteed acyclic). The variable-level matrix `model.full_adj_ij` is a *selected LiNG-D representative* — one member of the distributional equivalence class chosen by `hungarian_any`, used only as an intermediate to find the SCCs. Other compatible variable graphs differ (especially inside SCCs) while sharing the same identifiable coarsening, so variable-level scores are diagnostics, never algorithm performance, and the topology figure deliberately draws only the true variable graph and the two C-DAGs.

## Experiment 2: Reduced Wong-Wang (nonlinear)

`snakemake tvb_rww_all` tests whether condensation recovery survives nonlinear neural-mass dynamics. The same five 20-region multi-SCC graphs get one Reduced Wong-Wang population per node (`tvb.simulator.models.ReducedWongWang`: nonlinear input-to-rate transfer, recurrence, saturation, S ∈ [0,1]), so the true SCC partition and C-DAG remain known. Unlike Experiment 1, **TVB generates every observation**: each episode applies an independent truncated-Laplace shift to the per-region external drive `I_o` and integrates (deterministic Heun, zero delays, no noise) until `max_i |dS_i/dt| ≤ 1e-7`; the settled state is the observation. Independent episodes are batched block-diagonally into one TVB simulation for speed — numerically identical to one-at-a-time integration.

- **Calibration** (`tvb_rww_calibrate`, TVB-free): scans global coupling G and baseline drive `I_o` around TVB's standard parameters; a candidate qualifies only if dense multi-start Newton finds exactly one equilibrium, stable with margin ≥ 0.002/ms, for every seed graph — also under uniform worst-case input shifts at the truncation corner. Among qualifiers the winner maximizes a *linearized-oracle identifiability score*: the SCC ARI that the unchanged LiNG-D pipeline reaches on standardized samples of the candidate's exactly-linearized equilibrium map. The nonlinear dynamics reshape effective coupling magnitudes (they span orders of magnitude in near-critical regimes, so no fixed threshold separates them), and a regime must be identifiable in the locally-linear limit for the amplitude ladder to measure what *nonlinearity* destroys rather than a threshold mismatch. The scan CSV documents how knife-edged this is — neighboring regimes flip between oracle ARI ≈ 0 and ≈ 0.85.
- **Ground-truth verification** (`tvb_rww_verify`, TVB-free): at the chosen regime, the equilibrium Jacobian is stable and its cross-region support equals the intended graph's support for every seed — so the intended SCC partition/C-DAG is genuinely the local ground truth of the nonlinear system.
- **Difficulty ladder**: Laplace scale ∈ {0.0005, 0.002, 0.005} nA (truncated at ±3 scales), calibrated against the measured linearization error of the exact equilibrium map — the near-critical margin amplifies inputs ~60×/nA, so these correspond to ≈2%, ≈9%, and ≈23% nonlinear response. Small ≈ locally linear; large probes saturation. 5 seeds × 3 amplitudes × nested prefixes [100, 500, 2000, 5000]; fits run the unchanged LiNG-D pipeline on per-fit **standardized** settled states (deviation from plain centering, disclosed: the dynamics also reshape per-node scales, and standardization is a positive-diagonal similarity transform `B → D B D⁻¹` that provably preserves support, SCCs, and the condensation).
- **Held-out perturbation validation** (`tvb_rww_stimulate` + `score_rww_descendants`): positive AND negative cluster stimulations per true SCC, integrated to steady state at tol 1e-10 (non-descendant responses are exactly zero in the dynamics, so the floor is integration drift), scored against the descendants predicted by the recovered C-DAG. Never shown to the algorithm.
- **Outputs**: `tvb_rww_results.csv`, `tvb_rww_calibration.csv`, `tvb_rww_verification.csv`, `tvb_rww_descendants.csv`, and the comparison figure `tvb_rww.pdf` (linear-vs-RWW recovery, recovery vs amplitude, held-out descendant validation, true-vs-recovered C-DAG topology).

**Findings.** Nonlinear neural-mass dynamics narrow, but do not close, the recovery window. (i) *Sample hunger*: where the linear pilot recovers exactly at n=2000, Wong-Wang data yields nothing below n=2000 and reaches its recovery only at n=5000. (ii) *Partial recovery under mild nonlinearity*: at ≈2-9% nonlinear response, median SCC ARI reaches 0.79-0.91 and direct C-DAG F1 0.42-0.62 — clusters mostly right with boundary errors (e.g. one 2-cycle split into singletons), never a fully exact partition. Mild nonlinearity (9%) slightly *helps* over 2%, consistent with it adding non-Gaussian signal for ICA. (iii) *Saturation kills it*: at ≈23% nonlinearity, recovery collapses (ARI ≈ 0). (iv) *Held-out perturbations agree*: C-DAG-predicted descendants match actual nonlinear TVB stimulation responses at F1 ≈ 0.92 for the recovered models vs ≈ 0.67 for collapsed ones. (v) *The effective SEM is the bottleneck, not TVB or noise*: even on exactly-linearized standardized samples, the oracle tops out at ARI ≈ 0.85 in the best regime, and neighboring regimes flip to ≈ 0 — nonlinear dynamics reshape effective coupling magnitudes across orders of magnitude, which fixed-threshold LiNG-D tolerates only in a narrow operating window (see `tvb_rww_calibration.csv`).

## Experiment 3 (preflight only): the anatomical connectome

`tvb_preflight` reports the SCC structure of TVB's default 76-region connectome under |weight| thresholding — symmetry, reciprocity, SCC counts, largest-SCC fraction. It is *descriptive only*: the near-symmetric weights collapse into one giant SCC at low thresholds, where condensation recovery is vacuous. That realism gap is exactly why the recovery claims rest on the controlled multi-SCC benchmarks of Experiments 1-2, and recovery-quality curves are reported as scientific results, not enforced as pipeline gates.

## Project layout

```
src/
  repare_cycle/             # Library
    graph.py                # Generators, CyclicLinearSEM, Tarjan, scc_partition
    lingd.py                # ICA-LiNG-D, staged: ICA / enumeration / selection
    metrics.py              # Split/merge partition diagnostics
    intervention.py         # Whole-SCC hard interventions + block regression
    examples.py             # Hard-coded SEMs from the literature (e.g. Lacerda 2008)
    tvb_bridge.py           # Internal TVB bridge (optional `.[tvb]` extra; not public API)
  expt/workflow/            # Snakemake pipeline
    Snakefile               # Top-level: lists the four paper figures as targets
    rules/synth.smk         # Main + scalability + disjoint-micro grids
    rules/threshold.smk     # Threshold-sensitivity sweep
    rules/ablation.smk      # Candidate-selection ablation (+ cost arm)
    rules/intervention.smk  # Whole-SCC intervention experiment
    rules/tvb.smk           # TVB Experiment 1: linear compatibility pilot (`tvb_all`)
    rules/tvb_rww.smk       # TVB Experiment 2: Reduced Wong-Wang (`tvb_rww_all`)
    scripts/                # generate / fit / evaluate / collect / plot_*
    results/                # Outputs (created on first run)
tests/
  test_repare_cycle.py
  test_lingd_refactor.py    # Staged API ≡ run_lingd; safeguard behaviour
  test_metrics.py           # Split/merge diagnostics
  test_intervention.py      # Convention check, Monte Carlo, block regression
  test_tvb_bridge.py        # Pure bridge tests + TVB tests (auto-skip without `.[tvb]`)
```

## Tests

```bash
pytest tests/
```

The TVB integration tests in `tests/test_tvb_bridge.py` skip automatically when the `.[tvb]` extra is not installed; everything else (including the pure bridge math) runs without it.
