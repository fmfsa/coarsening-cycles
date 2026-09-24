"""Held-out Wong-Wang cluster stimulations with TVB (Experiment 2).

For seed 0 and one amplitude level: apply a POSITIVE and a NEGATIVE
external-drive shift of stim_scale/sqrt(|SCC|) to each true SCC's members,
integrate the nonlinear network to its new steady state, and record the
per-region response S* - S*_baseline. These perturbations are never given
to the fitting algorithm — they exist purely as held-out validation of the
descendants the recovered C-DAG predicts (scored downstream by
score_rww_descendants.py).

A tighter tolerance than the observation stage is used deliberately:
non-descendant responses are exactly zero in the dynamics, so the
measurable floor is set by integration drift (~residual/margin), and
tol=1e-10 keeps that floor well below the response threshold.
"""

import json

import numpy as np
import pandas as pd

from repare_cycle.tvb_bridge import rww_stable_fixed_point, simulate_rww_ensemble

amp = str(snakemake.wildcards.amp)
sigma = float(snakemake.params.sigma)
stim_scale = float(snakemake.params.stim_scale)
dt = float(snakemake.params.dt)
chunk_ms = float(snakemake.params.chunk_ms)
max_ms = float(snakemake.params.max_ms)
tol = float(snakemake.params.tol)

base = np.load(snakemake.input.data, allow_pickle=True)
weights = base["weights"]
scc_labels = base["scc_labels"]
d = weights.shape[0]
with open(snakemake.input.chosen) as fh:
    chosen = json.load(fh)
g, io = float(chosen["g"]), float(chosen["io"])

# Baseline settled by TVB itself at the tight tolerance, so responses are
# differences between two states produced by the same integrator.
newton = rww_stable_fixed_point(weights, np.full(d, io), g)
baseline = simulate_rww_ensemble(
    weights, np.full((1, d), io), g=g, dt=dt, chunk_ms=chunk_ms,
    max_ms=max_ms, tol=tol, init=newton.state, batch_size=1,
)
assert bool(baseline.converged[0]), "baseline failed to settle"
s_base = baseline.states[0]

labels = sorted(set(scc_labels.tolist()))
episodes = []  # (scc_label, scc_size, sign, delta_io)
for label in labels:
    members = np.flatnonzero(scc_labels == label)
    shift = stim_scale * sigma / np.sqrt(members.size)
    for sign in (+1, -1):
        delta = np.zeros(d)
        delta[members] = sign * shift
        episodes.append((label, members.size, sign, delta))

input_rows = np.array([io + delta for *_ , delta in episodes])
result = simulate_rww_ensemble(
    weights, input_rows, g=g, dt=dt, chunk_ms=chunk_ms, max_ms=max_ms,
    tol=tol, init=s_base, batch_size=len(episodes),
)

rows = []
for k, (label, size, sign, _) in enumerate(episodes):
    response = result.states[k] - s_base
    for coord in range(d):
        rows.append({
            "seed": 0,
            "amp": amp,
            "scc_label": int(label),
            "scc_size": int(size),
            "sign": sign,
            "coord": coord,
            "s_baseline": s_base[coord],
            "s_stimulated": result.states[k, coord],
            "response": response[coord],
            "converged": bool(result.converged[k]),
            "residual": float(result.residuals[k]),
            "sim_time_ms": float(result.sim_time_ms[k]),
            "stim_shift": stim_scale * sigma / np.sqrt(size),
        })

n_conv = int(result.converged.sum())
print(f"amp={amp}: {n_conv}/{len(episodes)} stimulations converged, "
      f"max residual {result.residuals.max():.2e}")
pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
