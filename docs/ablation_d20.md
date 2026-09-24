# Candidate-selection ablation

The paper table uses one study: `config/ablation_d20.json`, with measurements
in `src/expt/workflow/results/ablation_d20/`.

## Run and reproduce the table

Use Python 3.11 and the repository's dependencies. For a fresh pip environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e . pandas matplotlib seaborn pytest
```

From the repository root:

```bash
bash scripts/run_ablation_d20.sh
.venv/bin/python scripts/report_selection_ablation.py
```

The shell launcher fixes numerical libraries to one thread. The runner resumes
completed cells only when configuration, measured source files, and environment
match. The committed `protocol.json` has its `machine` field blanked, so the
runner will not resume into `results/ablation_d20/`; the report still reads it.
To collect fresh timings, use a separate directory:

```bash
bash scripts/run_ablation_d20.sh --output src/expt/workflow/results/ablation_d20_server
.venv/bin/python scripts/report_selection_ablation.py --input src/expt/workflow/results/ablation_d20_server --output output/pdf/ablation_d20_server_table.pdf
```

The report reads saved measurements without fitting models. It exports PDF, PNG,
LaTeX, diagnostics, and separate explanatory text as `output/pdf/ablation_d20_table*`.
Results and generated reports are gitignored unless explicitly included.

## Experimental setup

- d=20, κ=5 nontrivial SCCs, density 0.5, and Laplace noise.
- Stable and unstable regimes; n=10,000 and 50,000; seeds 0–29:
  120 datasets in total.
- SCCs contain Hamilton cycles plus additional within-SCC edges with probability
  0.5. Each leftover node joins an SCC with probability 0.5 or remains a singleton.
  The ordering between parts includes both SCCs and singletons.
- Signed weight magnitudes are initially uniform in [0.5, 0.95]. Stable graphs
  have spectral radius below one; unstable graphs are rescaled to 1.5.
- One FastICA estimate per dataset is shared by all four selection methods:
  max_iter=10,000, tolerance=1e-6, random_state=0, with the existing retry behavior.
- W and B thresholds start at 0.1. If no candidate is found, the W threshold
  is halved using the existing 0.01 stopping floor; the B threshold stays fixed.
- Enumeration has **no candidate-count cap** and a **60-second total budget**
  per dataset, shared across threshold rounds.

The four methods are enumeration + first-stable, enumeration + uniform random,
ours (Hungarian assignment), and arbitrary uniform row permutations without
admissibility constraints. Both randomized methods use five draws per dataset.
First-stable selects after enumeration; it falls back to the first unstable
candidate when no stable candidate was found. Uniform random samples the
returned list, which may be incomplete after a timeout. Ours does not enumerate.

## Measurements and interpretation

ICA timing includes fitting/retries after dependency imports. Selection timing
includes enumeration and stability checks for the enumeration methods. The same
shared enumeration cost is charged to each of those two methods. Data generation,
scoring, and file operations are excluded. Branch order rotates across seeds.

Random draws are averaged within each dataset before aggregation across seeds.
ARI and exact recovery are means; stage times are medians. Exact recovery requires
both the SCC partition and condensation edges to match. Missing estimates have
missing accuracy, not zero scores; accuracy is conditional on returned estimates,
while timings include every attempt. Bold selection times indicate a median
paired selection/ICA ratio above one.

The completed study has 120 dataset attempts, 16 enumeration timeouts, and no
ICA convergence warnings. Seed 20 in the unstable regime returned no enumeration
candidate at either sample size, so each unstable enumeration accuracy estimate
uses 29/30 seeds. Ours and the control returned estimates for all datasets.
The largest candidate list contains 30,336 candidates. Selection exceeded ICA
on 115/120 datasets for the enumeration methods. None of the 600 control draws
was admissible or recovered the condensation.

`protocol.json` records configuration, source hashes, environment, and thread
settings. Each dataset stores `result.json` and `estimate.npz` (ICA estimate and
ground truth). `draws.csv`, `cells.csv`, and `summary.csv` provide draw-level,
dataset-level, and aggregate measurements; `verification.json` records checks.

Optional paired comparisons can be reproduced without new fits:

```bash
.venv/bin/python scripts/compare_ablation_selection.py --input src/expt/workflow/results/ablation_d20
```

This generates dataset-level bootstrap contrasts, including a diagnostic subset
excluding enumeration timeouts. These descriptive intervals do not establish
equivalence merely by including zero.
