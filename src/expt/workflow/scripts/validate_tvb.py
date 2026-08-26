"""Numerically validate the TVB bridge against exact equilibria (seed 0).

Integrates TVB's linear model (gamma=-1, linear coupling a=1, zero delays,
deterministic Heun, constant regional stimulus) for:

  * the first `n_episodes` observational input rows E[k] — expected
    equilibrium is the stored observation X[k]; and
  * one soft intervention per true SCC — stimulus delta_e with
    1/sqrt(|SCC|) on each member; the baseline (zero input) equilibrium is
    exactly zero, so the intervened equilibrium IS the steady-state
    response, expected to equal the stored delta_e (I - W)^{-1}.

Emits results/tvb_equilibrium_validation.csv in long format — one row per
(episode, coordinate) — so the exact-vs-TVB panel of the pilot figure plots
straight from this file. Episode-level diagnostics (convergence, terminal
residual, settling time, error norms) repeat across an episode's rows.

Requires the `.[tvb]` extra. Acceptance (checked downstream, not gated
here): every episode converges, terminal residual <= 1e-7, relative
equilibrium/response error <= 1e-5.
"""

import numpy as np
import pandas as pd

from repare_cycle.tvb_bridge import simulate_to_equilibrium

n_episodes = int(snakemake.params.n_episodes)
dt = float(snakemake.params.dt)
chunk_ms = float(snakemake.params.chunk_ms)
max_ms = float(snakemake.params.max_ms)
tol = float(snakemake.params.tol)

data = np.load(snakemake.input.data, allow_pickle=True)
weights = data["weights"]
d = weights.shape[0]

episodes = [
    ("observational", k, float("nan"), float("nan"),
     data["inputs"][k], data["obs"][k])
    for k in range(n_episodes)
] + [
    ("intervention", j,
     int(data["intervention_scc_labels"][j]),
     int(data["intervention_scc_sizes"][j]),
     data["intervention_inputs"][j], data["intervention_responses"][j])
    for j in range(data["intervention_inputs"].shape[0])
]

rows = []
for episode_type, episode_id, scc_label, scc_size, stim, x_exact in episodes:
    result = simulate_to_equilibrium(
        weights, stim, dt=dt, chunk_ms=chunk_ms, max_ms=max_ms, tol=tol
    )
    err = result.x - x_exact
    abs_err_linf = float(np.max(np.abs(err)))
    abs_err_l2 = float(np.linalg.norm(err))
    rel_err_l2 = abs_err_l2 / max(float(np.linalg.norm(x_exact)), 1e-12)
    for coord in range(d):
        rows.append({
            "seed": 0,
            "episode_type": episode_type,
            "episode_id": episode_id,
            "scc_label": scc_label,
            "scc_size": scc_size,
            "coord": coord,
            "x_exact": x_exact[coord],
            "x_tvb": result.x[coord],
            "converged": result.converged,
            "terminal_residual": result.terminal_residual,
            "settling_time_ms": result.settling_time_ms,
            "sim_time_ms": result.sim_time_ms,
            "n_steps": result.n_steps,
            "abs_err_linf": abs_err_linf,
            "abs_err_l2": abs_err_l2,
            "rel_err_l2": rel_err_l2,
            "runtime_sec": result.runtime_sec,
            "dt": dt,
            "tol": tol,
        })
    status = "ok" if result.converged else "DID NOT CONVERGE"
    print(
        f"{episode_type} #{episode_id}: {status}, "
        f"residual={result.terminal_residual:.2e}, "
        f"settling={result.settling_time_ms:.2f} ms, "
        f"rel_err={rel_err_l2:.2e}"
    )

pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
