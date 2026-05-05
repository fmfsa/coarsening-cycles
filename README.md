# Coarsening Causal Models with Cycles

Recursive partition refinement for causal discovery in directed graphs with cycles.

## Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate repare_cycle
pip install -e .
```

### Option B: pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -e . && pip install snakemake seaborn pandas matplotlib
```

Requires Python >= 3.9 (tested on 3.13).

### R + disjointCycles (only for the scalability comparison)

The scalability figure compares against the [disjointCycles](https://github.com/ysamwang/disjointCycles) R package. To run those cells you need R (>= 4.0) on `PATH` and the package installed:

```r
install.packages(c("pak", "jsonlite"))
pak::pak("ysamwang/disjointCycles")
```

If you skip this step, the d=10 main grid still runs; only `results/scalability_disjointcycles.pdf` will fail.

## Running experiments

All experiments are orchestrated with [Snakemake](https://snakemake.readthedocs.io/) from the workflow directory:

```bash
cd src/expt/workflow
```

### Run the full grid (all regimes, densities, sample sizes)

```bash
snakemake --cores all
```

This produces one PDF per (regime, density) combination in `src/expt/workflow/results/`:
- `synth_main_regime=hard_density=0.3.pdf`
- `synth_main_regime=hard_density=0.5.pdf`
- `synth_main_regime=hard_density=0.8.pdf`
- `synth_main_regime=unstable_density=0.3.pdf`
- `synth_main_regime=unstable_density=0.5.pdf`
- `synth_main_regime=unstable_density=0.8.pdf`

And a collected CSV: `results/synth_results.csv`.

### Run a single slice (faster for testing)

```bash
# Only the hard regime at density 0.3
snakemake results/synth_main_regime=hard_density=0.3.pdf --cores all

# Only a specific fit (useful for debugging)
snakemake "results/synth/regime=hard/d=10/num_cycles=3/density=0.3/samp_size=1000/seed=0/method=lacerda/metrics.csv" --cores 1
```

### Dry run (see what would execute without running)

```bash
snakemake -n
```

### Force re-run of everything

```bash
snakemake --cores all --forceall
```

## Experiment parameters

Edit `src/expt/workflow/rules/synth.smk` to change:

| Parameter | Variable | Default |
|-----------|----------|---------|
| Methods compared | `synth_methods` | `["lacerda", "lacerda_rnd"]` |
| Regimes | `synth_regimes` | `["hard", "unstable"]` |
| Number of nodes | `synth_d` | `[10]` |
| Number of SCCs (cycles) | `synth_num_cycles` | `[3, 4, 5]` |
| Edge densities | `synth_densities` | `[0.3, 0.5, 0.8]` |
| Sample sizes | `synth_samp_sizes` | `[50, 100, 500, 1000, 5000]` |
| Random seeds | `synth_seeds` | `range(10)` |

The scalability sweep against `disjointCycles` uses an independent grid (`scal_d`, `scal_num_cycles`, `scal_samp_sizes`) defined in the same file; building `results/scalability_disjointcycles.pdf` triggers it.

### Adding a new method

1. Add the method name to `synth_methods` in `rules/synth.smk`.
2. Add the corresponding `pick_strategy` logic in the `synth_fit` rule's `params` block.
3. If the method needs a different fitting script, create a new rule pointing to a new script.

### Adding a new regime

1. Add the regime name to `synth_regimes` in `rules/synth.smk`.
2. Add the corresponding config to `REGIME_CONFIG` in `scripts/generate_synth.py`.

## Pipeline stages

The pipeline has four stages (each is a Snakemake rule):

1. **`synth_generate`** — Creates a random cyclic graph + samples observational data. Output: `dataset.npz` containing `weights`, `scc_labels`, `scc_sizes`, `obs`.

2. **`synth_fit`** — Runs ICA-LiNG-D (FastICA + N-rooks permutation search), picks a B-hat candidate, runs Tarjan to get the condensation. Output: `model.pkl`.

3. **`synth_evaluate`** — Compares the estimated condensation to ground truth. Metrics: ARI (partition), cluster-DAG F1, variable-level F1, inter-SCC F1, SHD. Output: `metrics.csv`.

4. **`synth_collect`** + **`synth_plot`** — Aggregates all metrics into one CSV and produces the figure PDFs.

## Regimes explained

- **`hard`**: Weights in [0.5, 0.95], rescaled so ρ(B) < 1 (close to stability boundary, typically ρ ≈ 0.9). The stability filter can still find the true B, but estimation is challenging at small n.

- **`unstable`**: Same weight pool but rescaled to target ρ(B) = 1.5. The true B violates the stability condition, so the first-stable filter necessarily picks a different equivalence-class member (the cycle-reversed twin). This is the key test: even with the "wrong" variable-level graph, the condensation is correct (Theorem 1).

## Methods explained

- **`lacerda`** (first-stable): Picks the first B-hat with ρ < 1. For sparse cyclic SEMs under ρ(B) < 1, this uniquely recovers the true variable-level graph.

- **`lacerda_rnd`** (random-admissible): Draws uniformly from all admissible permutations (stable + unstable). Deliberately samples the full distributional equivalence class. Demonstrates that cluster-DAG F1 → 1 regardless of which member is picked, while variable-level F1 saturates below 1.

## Key metrics

- **ARI (SCC partition)**: Adjusted Rand Index between predicted and true SCC labels. Measures partition quality.
- **Cluster-DAG F1**: Predicted variable edges projected onto the true SCC partition, compared to true cluster-DAG edges. This is the identification target of Theorem 1.
- **Variable-level F1**: F1 over all directed edges. Not identifiable in general — the gap between this and cluster-DAG F1 is the unidentifiable intra-SCC structure.
- **Inter-SCC F1**: Variable-pair-level F1 restricted to pairs crossing SCC boundaries. A diagnostic metric.

## Project structure

```
src/
  repare_cycle/          # Library code
    graph.py             # Graph generators, CyclicLinearSEM
    lingd.py             # ICA-LiNG-D implementation
    examples.py          # Worked examples
  expt/workflow/         # Snakemake experiment pipeline
    Snakefile            # Top-level orchestration
    rules/synth.smk      # Synthetic experiment rules + parameters
    scripts/             # Pipeline stage scripts
    results/             # Output (figures, CSV, intermediate files)
paper/
  main.tex              # Paper source
  figures/              # Figure files for the paper
tests/
  test_repare_cycle.py  # Unit tests
```

## Running tests

```bash
pytest tests/
```

## Compiling the paper

```bash
cd paper
pdflatex main.tex
bibtex main        # if using .bib (currently bibliography is inline)
pdflatex main.tex
pdflatex main.tex
```
