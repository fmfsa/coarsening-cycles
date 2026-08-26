"""Generate Wong-Wang equilibrium observations with TVB (Experiment 2).

Unlike Experiment 1, every observation here comes out of TVB itself: one
Reduced Wong-Wang population per node of the benchmark graph, deterministic
Heun, zero delays, no process noise. For each episode an independent
truncated-Laplace shift of the per-region external drive I_o is applied
and the network is integrated until max_i |dS_i/dt| <= tol; the settled
S* is the observation. Episodes are batched block-diagonally
(simulate_rww_ensemble) for speed — numerically identical to one-at-a-time
integration.

Non-convergent episodes are retried with a 4x time budget, then replaced
by fresh input draws (up to 3 rounds); the counts are stored and printed,
and generation fails loudly only if convergence still cannot be reached.

The npz keeps the fit-pipeline keys (`weights`, `scc_labels`, `scc_sizes`,
`obs`) so fit.py/evaluate_tvb.py work unchanged; `obs` holds the RAW
settled states — fit.py centers per prefix via its `center` param.
"""

import json

import numpy as np
from repare_cycle.tvb_bridge import rww_stable_fixed_point, simulate_rww_ensemble

seed = int(snakemake.wildcards.seed)
amp = str(snakemake.wildcards.amp)
amp_id = int(snakemake.params.amp_id)
sigma = float(snakemake.params.sigma)          # Laplace scale b
trunc_k = float(snakemake.params.trunc_k)      # truncate at +/- k*b
n_obs = int(snakemake.params.n_obs)
dt = float(snakemake.params.dt)
chunk_ms = float(snakemake.params.chunk_ms)
max_ms = float(snakemake.params.max_ms)
tol = float(snakemake.params.tol)
batch_size = int(snakemake.params.batch_size)

base = np.load(snakemake.input.data, allow_pickle=True)
weights = base["weights"]
d = weights.shape[0]
with open(snakemake.input.chosen) as fh:
    chosen = json.load(fh)
g, io = float(chosen["g"]), float(chosen["io"])

# Baseline fixed point (warm start only — observations come from TVB).
# Raises if the calibrated regime is not single-stable-fixed-point.
baseline = rww_stable_fixed_point(weights, np.full(d, io), g)

rng = np.random.default_rng([seed, amp_id, 271828])


def _draw_shifts(n):
    raw = rng.laplace(0.0, sigma, size=(n, d))
    return np.clip(raw, -trunc_k * sigma, trunc_k * sigma)


shifts = _draw_shifts(n_obs)
result = simulate_rww_ensemble(
    weights, io + shifts, g=g, dt=dt, chunk_ms=chunk_ms, max_ms=max_ms,
    tol=tol, init=baseline.state, batch_size=batch_size,
)
states = result.states
converged = result.converged
residuals = result.residuals
n_retried = n_replaced = 0

# Retry stragglers with a 4x budget, warm-started from where they stopped.
if not converged.all():
    idx = np.flatnonzero(~converged)
    n_retried = idx.size
    retry = simulate_rww_ensemble(
        weights, io + shifts[idx], g=g, dt=dt, chunk_ms=chunk_ms,
        max_ms=4 * max_ms, tol=tol, init=states[idx], batch_size=batch_size,
    )
    states[idx], converged[idx], residuals[idx] = (
        retry.states, retry.converged, retry.residuals)

# Replace episodes that still refuse to settle with fresh draws.
for _ in range(3):
    if converged.all():
        break
    idx = np.flatnonzero(~converged)
    n_replaced += idx.size
    shifts[idx] = _draw_shifts(idx.size)
    redo = simulate_rww_ensemble(
        weights, io + shifts[idx], g=g, dt=dt, chunk_ms=chunk_ms,
        max_ms=4 * max_ms, tol=tol, init=baseline.state,
        batch_size=batch_size,
    )
    states[idx], converged[idx], residuals[idx] = (
        redo.states, redo.converged, redo.residuals)

if not converged.all():
    raise RuntimeError(
        f"seed={seed} amp={amp}: {int((~converged).sum())}/{n_obs} episodes "
        f"failed to reach |dS|<= {tol} even after retries "
        f"(worst residual {residuals.max():.2e})."
    )

print(f"seed={seed} amp={amp}: {n_obs} episodes converged "
      f"(retried {n_retried}, replaced {n_replaced}), "
      f"max residual {residuals.max():.2e}, "
      f"TVB wall time {result.runtime_sec:.1f}s")

np.savez(
    snakemake.output[0],
    weights=weights,
    scc_labels=base["scc_labels"],
    scc_sizes=base["scc_sizes"],
    obs=states,                       # raw settled S*; fit.py centers
    inputs=shifts,                    # per-episode I_o shifts
    io_baseline=io,
    g=g,
    S_baseline=baseline.state,
    amp=amp,
    amp_sigma=sigma,
    trunc_k=trunc_k,
    tol=tol,
    spectral_radius=float(base["spectral_radius"]),
    residuals=residuals,
    sim_time_ms=result.sim_time_ms,
    n_retried=n_retried,
    n_replaced=n_replaced,
    orientation=str(base["orientation"]),
)
