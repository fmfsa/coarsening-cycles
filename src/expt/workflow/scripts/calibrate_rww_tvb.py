"""Calibrate a stable Reduced Wong-Wang regime (Experiment 2; TVB-free).

Scans global coupling G and regional baseline drive I_o around TVB's
standard parameters. For every candidate and every benchmark graph,
`rww_equilibria` enumerates equilibria by dense multi-start damped Newton
— Newton converges to saddles as readily as to attractors, so the starts
double as a (heuristic) multistability probe. A candidate qualifies when,
for EVERY seed graph:

  * exactly ONE equilibrium is found (no coexisting attractors OR saddles),
  * it is stable with margin >= `margin_floor`, and
  * both remain true under uniform worst-case input shifts
    ±`robust_shift` (the truncation corner of the largest episode
    amplitude).

Among qualifying candidates, the winner maximizes a LINEARIZED-ORACLE
identifiability score: LiNG-D (the unchanged pipeline settings, tau=0.1,
hungarian_any) run on samples of the candidate's exactly-linearized
equilibrium map, standardized — the SCC-partition ARI it reaches in the
locally-linear limit. Nonlinear dynamics reshape the effective coupling
magnitudes (b_col = C/Lambda spans orders of magnitude in near-critical
regimes), so a regime must first be identifiable in the linear limit for
the amplitude ladder to measure what NONLINEARITY destroys, rather than
measuring a threshold/criticality mismatch. Ties broken by stability
margin. Writes the full scan CSV and the chosen parameters as JSON for
the downstream generation, verification, and stimulation stages.
"""

import json

import networkx as nx
import numpy as np
import pandas as pd

from repare_cycle.lingd import run_lingd
from repare_cycle.tvb_bridge import rww_equilibria, rww_linearization
from sklearn.metrics import adjusted_rand_score

g_grid = [float(g) for g in snakemake.params.g_grid]
io_grid = [float(io) for io in snakemake.params.io_grid]
margin_floor = float(snakemake.params.margin_floor)
robust_shift = float(snakemake.params.robust_shift)
oracle_threshold = float(snakemake.params.oracle_threshold)
oracle_n = int(snakemake.params.oracle_n)


def _oracle_ari(weights, state, io_vector, g, labels_true, seed):
    """SCC-partition ARI of LiNG-D on exactly-linearized samples.

    The linear map's input scale cancels under standardization, so the
    oracle depends only on the regime's effective mixing geometry.
    """
    d = weights.shape[0]
    lin = rww_linearization(state, io_vector, weights, g)
    rng = np.random.default_rng([seed, 424242])
    deltas = np.clip(rng.laplace(0.0, 1.0, size=(oracle_n, d)), -3.0, 3.0)
    xs = deltas @ lin.mixing.T
    xs = xs - xs.mean(axis=0)
    xs = xs / xs.std(axis=0)
    result = run_lingd(xs, threshold_b=oracle_threshold,
                       threshold_w=oracle_threshold,
                       pick_strategy="hungarian_any")
    graph = nx.DiGraph((np.abs(result["B_chosen"].T) > 0).astype(int))
    est = np.zeros(d, dtype=int)
    for k, scc in enumerate(nx.strongly_connected_components(graph)):
        est[list(scc)] = k
    return float(adjusted_rand_score(labels_true, est))


def _probe(weights, io_vector, g):
    """One candidate probe: strict single-stable-equilibrium check."""
    eq = rww_equilibria(weights, io_vector, g)
    margin = float(eq.margins[0]) if eq.n_equilibria == 1 else (
        float(np.min(eq.margins)) if eq.n_equilibria else float("nan")
    )
    ok = eq.unique_stable and margin >= margin_floor
    state = eq.states[0] if eq.n_equilibria else None
    return {"n_equilibria": eq.n_equilibria, "ok": ok,
            "margin": margin, "state": state}


seed_weights = {}
seed_labels = {}
for path in snakemake.input:
    data = np.load(path, allow_pickle=True)
    seed = len(seed_weights)
    seed_weights[seed] = data["weights"]
    seed_labels[seed] = data["scc_labels"]

rows = []
probes = {}
candidates = []  # (g, io, worst_margin)
for g in g_grid:
    for io in io_grid:
        candidate_ok = True
        worst_margin = np.inf
        for seed, weights in seed_weights.items():
            d = weights.shape[0]
            base = _probe(weights, np.full(d, io), g)
            lo = _probe(weights, np.full(d, io - robust_shift), g)
            hi = _probe(weights, np.full(d, io + robust_shift), g)
            seed_ok = base["ok"] and lo["ok"] and hi["ok"]
            candidate_ok &= seed_ok
            worst_margin = min(worst_margin, base["margin"])
            probes[(g, io, seed)] = base
            rows.append({
                "g": g, "io": io, "seed": seed,
                "n_equilibria": base["n_equilibria"],
                "margin": base["margin"],
                "n_equilibria_minus_shift": lo["n_equilibria"],
                "n_equilibria_plus_shift": hi["n_equilibria"],
                "margin_minus_shift": lo["margin"],
                "margin_plus_shift": hi["margin"],
                "seed_ok": seed_ok,
                "oracle_ari": float("nan"),
                "s_min": (float(base["state"].min())
                          if base["state"] is not None else float("nan")),
                "s_max": (float(base["state"].max())
                          if base["state"] is not None else float("nan")),
            })
        if candidate_ok:
            candidates.append((g, io, worst_margin))

if not candidates:
    pd.DataFrame(rows).to_csv(snakemake.output.scan, index=False)
    raise RuntimeError(
        "No (G, I_o) candidate passed calibration — every scanned regime is "
        "multistable, unstable, or fragile under the worst-case input "
        f"shift ±{robust_shift}. See {snakemake.output.scan} for the scan."
    )

# Linearized-oracle identifiability for the qualifying candidates.
scored = []
row_index = {(r["g"], r["io"], r["seed"]): i for i, r in enumerate(rows)}
for g, io, worst_margin in candidates:
    aris = []
    for seed, weights in seed_weights.items():
        d = weights.shape[0]
        ari = _oracle_ari(weights, probes[(g, io, seed)]["state"],
                          np.full(d, io), g, seed_labels[seed], seed)
        rows[row_index[(g, io, seed)]]["oracle_ari"] = ari
        aris.append(ari)
    scored.append((float(np.mean(aris)), worst_margin, g, io))
    print(f"candidate G={g}, I_o={io}: margin={worst_margin:.4f}/ms, "
          f"linearized-oracle ARI={np.mean(aris):.3f}")

pd.DataFrame(rows).to_csv(snakemake.output.scan, index=False)

oracle_star, margin_star, g_star, io_star = max(scored)
chosen = {
    "g": g_star,
    "io": io_star,
    "worst_margin": margin_star,
    "oracle_ari": oracle_star,
    "margin_floor": margin_floor,
    "robust_shift": robust_shift,
    "oracle_threshold": oracle_threshold,
    "criteria": "unique stable fixed point for all seeds, also under "
                "uniform +/-robust_shift input shifts; among qualifiers, "
                "maximize LiNG-D SCC ARI on the exactly-linearized "
                "standardized equilibrium map (identifiability in the "
                "locally-linear limit), tie-broken by margin",
}
with open(snakemake.output.chosen, "w") as fh:
    json.dump(chosen, fh, indent=2)
print(f"chosen regime: G={g_star}, I_o={io_star}, "
      f"worst margin {margin_star:.4f}/ms, "
      f"linearized-oracle ARI {oracle_star:.3f}")
