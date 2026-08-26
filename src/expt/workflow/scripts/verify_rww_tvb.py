"""Verify the Wong-Wang ground truth per seed (Experiment 2; TVB-free).

For the calibrated regime and every benchmark graph: solve the baseline
fixed point, confirm local stability (spectral abscissa of the equilibrium
Jacobian), and confirm that the effective cross-region support of the
Jacobian equals the intended graph's support — i.e. the SCC partition and
C-DAG of the intended graph really are the ground truth of the nonlinear
system's local dynamics.
"""

import json

import numpy as np
import pandas as pd

from repare_cycle.tvb_bridge import rww_equilibria, rww_jacobian

with open(snakemake.input.chosen) as fh:
    chosen = json.load(fh)
g, io = float(chosen["g"]), float(chosen["io"])

rows = []
for seed, path in enumerate(snakemake.input.datasets):
    data = np.load(path, allow_pickle=True)
    weights = data["weights"]
    d = weights.shape[0]
    eq = rww_equilibria(weights, np.full(d, io), g)
    state = eq.states[np.argmax(eq.margins)]
    margin = float(np.max(eq.margins))

    jac = rww_jacobian(state, np.full(d, io), weights, g)
    offdiag = jac.copy()
    np.fill_diagonal(offdiag, 0.0)
    support_true = weights.T != 0.0  # J[i, j] != 0 iff W[j, i] != 0
    support_match = bool(np.array_equal(offdiag != 0.0, support_true))
    coupling_gains = np.abs(offdiag[support_true])

    ok = eq.unique_stable and margin > 0.0 and support_match
    rows.append({
        "seed": seed,
        "g": g,
        "io": io,
        "n_equilibria": eq.n_equilibria,
        "unique_stable": bool(eq.unique_stable),
        "s_min": float(state.min()),
        "s_max": float(state.max()),
        "stability_margin": margin,
        "support_match": support_match,
        "min_coupling_gain": (float(coupling_gains.min())
                              if coupling_gains.size else float("nan")),
        "max_coupling_gain": (float(coupling_gains.max())
                              if coupling_gains.size else float("nan")),
    })
    print(f"seed {seed}: {'ok' if ok else 'PROBLEM'}, "
          f"n_equilibria={eq.n_equilibria}, margin={margin:.4f}/ms, "
          f"support_match={support_match}")

pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
