# Coarsening Linear Non-Gaussian Causal Models with Cycles

Code for the paper [Coarsening Linear Non-Gaussian Causal Models with Cycles](https://arxiv.org/abs/2605.10163) (NeurIPS 2026).

Given observational data from a linear non-Gaussian causal model with feedback loops, we recover its **condensation**: the partition of variables into strongly connected components (SCCs) and the DAG between them. The condensation is the same for every model in the observational equivalence class, so it is identifiable, and we learn it in worst-case cubic time (ICA, then a Hungarian assignment, then Tarjan's algorithm).

## Installation

```bash
conda env create -f environment.yml
conda activate repare_cycle
pip install -e .
```

Or with pip only (Python ≥ 3.9):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The comparison against [disjointCycles](https://github.com/ysamwang/disjointCycles) (Figs. 4 and 7) additionally needs R ≥ 4.0:

```r
install.packages(c("pak", "jsonlite"))
pak::pak("ysamwang/disjointCycles")
```

## Reproducing the paper

### Replot from saved data

The data behind the figures are committed, so replotting takes seconds and runs no experiments. Figures are written to `output/pdf/`.

```bash
python scripts/regenerate_paper_figures.py --figures 3 4 5 7 8 grouplingam
```

### Rerun the experiments

The experiments are a Snakemake pipeline. The full run takes a few hours on a CPU workstation; fits are cached, so reruns are fast.

```bash
cd src/expt/workflow
snakemake --cores all
```

To build a single figure, pass its target, e.g. `snakemake results/synth_threshold.pdf --cores all`:

| Figure | Target (under `src/expt/workflow/`) |
|---|---|
| 3: main synthetic results | `results/synth_main_combined.pdf` |
| 4: scalability vs disjointCycles | `results/scalability_disjointcycles.pdf` |
| 5: sample complexity | `results/sample_complexity.pdf` |
| 7: strict disjoint cycles | `results/disjoint_micro.pdf` |
| 8: threshold sensitivity | `results/synth_threshold.pdf` |

Experiment grids (dimensions, densities, sample sizes, seeds) are defined at the top of each file in `src/expt/workflow/rules/`.

### Other experiments

These are not part of the default target:

```bash
# Split/merge failure modes
snakemake results/split_merge.pdf results/threshold_split_merge.pdf --cores all

# Whole-SCC hard interventions
snakemake results/intervention_effects.pdf --cores all
```

Two experiments run outside Snakemake, from the repository root. Their runners expect a Python 3.11 environment in `.venv`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r src/expt/config/group_lingam_requirements.txt -e .

# d=20 candidate-selection ablation (paper table); config in src/expt/config/ablation_d20.json
bash scripts/run_ablation_d20.sh
.venv/bin/python scripts/report_selection_ablation.py

# GroupLiNGAM comparison; takes several days, resumable
bash scripts/run_group_lingam.sh
```

## Repository layout

```
src/repare_cycle/     Library
  lingd.py            ICA-LiNG-D: ICA, candidate enumeration, selection
  graph.py            Graph generators, CyclicLinearSEM, Tarjan
  metrics.py          Split/merge diagnostics
  intervention.py     Whole-SCC hard interventions
  benchmark.py        GroupLiNGAM adapter
src/expt/workflow/    Snakemake pipeline (rules/, scripts/, results/)
src/expt/config/      Configs for the ablation and GroupLiNGAM runs
scripts/              Standalone runners and figure replotting
tests/                Run with `pytest tests/`
```

## Citation

```bibtex
@inproceedings{madaleno2026coarsening,
  title         = {Coarsening Linear Non-Gaussian Causal Models with Cycles},
  author        = {Madaleno, Francisco and Pereira, Francisco C. and Markham, Alex},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2605.10163},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```
