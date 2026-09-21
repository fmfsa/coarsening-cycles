"""Empirical corroboration of Proposition 4 (App. A.X).

Fix one data-generating SCM and sweep n widely with many seeds. At each
(n, seed) we
  1. sample n iid observations,
  2. run lacerda (FastICA + N-rooks + Tarjan) at threshold τ = β_min/2
     with the first-stable picker — for sparse cyclic graphs the unique
     stable equivalence-class member coincides with the data-generating
     B, so this targets supp(B) directly,
  3. record exact-support match (variable-level F1 = 1) and the
     condensation match (ARI = 1 AND cluster-DAG F1 = 1).

If the proposition is empirically tight, on a log-log plot:
  - P[support-recovery error] vs n should decay with slope ≈ −2
    (the 1/n² rate from the proposition);
  - P[condensation-recovery error] inherits the same rate by Step 3.

Output: results/sample_complexity.csv with columns
    n, seed, support_match, condensation_match, var_fscore, ari, cluster_f1, beta_min
"""

import os
import time

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from repare_cycle.graph import random_cyclic_graph, CyclicLinearSEM
from repare_cycle.lingd import run_lingd

# ──────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ──────────────────────────────────────────────────────────────────────────
GRAPH_SEED = 42                # fixes the data-generating B once
D          = 10
NUM_CYCLES = 4                 # κ = 4, matching the main-grid setup
DENSITY    = 0.5
WEIGHT_RNG = (0.5, 0.95)       # main-grid weight regime
NOISE      = "laplace"

SAMPLE_SIZES = [100, 200, 300, 400, 500, 700, 1000, 2000, 5000, 10000]
N_SEEDS      = 300

OUT_CSV = "results/sample_complexity.csv"


def main() -> None:
    # When invoked as a Snakemake `script:`, write to the rule's declared
    # output. When run standalone, fall back to OUT_CSV.
    out_path = OUT_CSV
    try:
        out_path = snakemake.output[0]   # type: ignore[name-defined]
    except NameError:
        pass

    # 1) Fix the data-generating graph once.
    graph, B_true = random_cyclic_graph(
        d=D, num_cycles=NUM_CYCLES, density=DENSITY, seed=GRAPH_SEED,
        weight_range=WEIGHT_RNG, target_rho=None,
    )
    nz = np.abs(B_true[B_true != 0])
    beta_min = float(nz.min())
    n_edges  = int((B_true != 0).sum())
    true_sccs = list(nx.strongly_connected_components(graph))
    n_sccs    = len(true_sccs)
    print(f"[setup] d={D}, β_min={beta_min:.4f}, "
          f"#edges={n_edges}, #SCCs={n_sccs}, "
          f"max SCC size={max(len(s) for s in true_sccs)}")

    # True variable-level adjacency (i→j). random_cyclic_graph returns
    # weights[i, j] for edge i→j, so no transpose is needed.
    true_adj_ij = (B_true != 0).astype(int)
    np.fill_diagonal(true_adj_ij, 0)
    scc_lab_true = np.zeros(D, dtype=int)
    for i, s in enumerate(true_sccs):
        for v in s:
            scc_lab_true[v] = i
    # True cluster-DAG edges (across SCC labels).
    true_cluster_edges = set()
    for u, v in zip(*np.nonzero(true_adj_ij)):
        su, sv = int(scc_lab_true[u]), int(scc_lab_true[v])
        if su != sv:
            true_cluster_edges.add((su, sv))

    tau = beta_min / 2.0

    # 2) Sweep.
    rows: list[dict] = []
    t_start = time.perf_counter()
    for n in SAMPLE_SIZES:
        n_runs = 0
        n_supp_ok = 0
        n_cond_ok = 0
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1_000_003 * seed + 17)
            sem = CyclicLinearSEM(
                B_true, means=(0.0, 0.0), variances=(1.0, 1.0),
                rng=rng, noise_dist=NOISE, allow_unstable=False,
            )
            X = sem.sample(n)
            try:
                r = run_lingd(
                    X, threshold_b=tau, threshold_w=tau,
                    ica_max_iter=10_000, ica_tolerance=1e-6,
                    pick_strategy="first_stable",
                    random_state=seed,
                )
            except Exception:
                continue
            B_hat = r["B_chosen"]
            # Variable-level adjacency (i→j) from B̂. Tetrad column convention:
            # B[i,j] != 0 ⇔ edge j→i, so transpose for i→j layout.
            pred_adj_ij = (np.abs(B_hat.T) > 0).astype(int)
            np.fill_diagonal(pred_adj_ij, 0)

            # Variable-level F1 + Hamming distance (smooth proxy for the
            # entry-wise support-recovery error). The proposition predicts
            # E[hamming] ≲ d² K / n², which is smoother on log-log than the
            # binary "exact match" indicator.
            tp = int(((pred_adj_ij == 1) & (true_adj_ij == 1)).sum())
            fp = int(((pred_adj_ij == 1) & (true_adj_ij == 0)).sum())
            fn = int(((pred_adj_ij == 0) & (true_adj_ij == 1)).sum())
            prec = tp / (tp + fp) if tp + fp > 0 else 1.0
            rec  = tp / (tp + fn) if tp + fn > 0 else 1.0
            var_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            hamming = int((pred_adj_ij != true_adj_ij).sum())

            # SCC partition of pred_adj_ij.
            dg = nx.DiGraph()
            dg.add_nodes_from(range(D))
            for i, j in zip(*np.nonzero(pred_adj_ij)):
                if i != j:
                    dg.add_edge(int(i), int(j))
            est_sccs = list(nx.strongly_connected_components(dg))
            est_lab = np.zeros(D, dtype=int)
            for k, s in enumerate(est_sccs):
                for v in s:
                    est_lab[v] = k
            if len(set(est_lab)) == 1 and len(set(scc_lab_true)) == 1:
                ari = 1.0
            else:
                ari = float(adjusted_rand_score(scc_lab_true, est_lab))

            # Cluster-DAG F1 against the TRUE partition.
            pred_cluster_edges = set()
            for u, v in zip(*np.nonzero(pred_adj_ij)):
                su, sv = int(scc_lab_true[u]), int(scc_lab_true[v])
                if su != sv:
                    pred_cluster_edges.add((su, sv))
            tp_c = len(pred_cluster_edges & true_cluster_edges)
            n_p  = len(pred_cluster_edges)
            n_t  = len(true_cluster_edges)
            if n_p == 0 and n_t == 0:
                cluster_f1 = 1.0
            elif n_p == 0 or n_t == 0:
                cluster_f1 = 0.0
            else:
                p_c = tp_c / n_p
                r_c = tp_c / n_t
                cluster_f1 = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0

            supp_match = int(var_f1 >= 0.999)
            cond_match = int(ari >= 0.999 and cluster_f1 >= 0.999)
            rows.append(dict(
                n=n, seed=seed,
                support_match=supp_match,
                condensation_match=cond_match,
                hamming=hamming,
                var_fscore=var_f1, ari=ari, cluster_f1=cluster_f1,
                beta_min=beta_min,
            ))
            n_runs    += 1
            n_supp_ok += supp_match
            n_cond_ok += cond_match
        if n_runs > 0:
            print(f"[n={n:6d}] runs={n_runs:3d}/{N_SEEDS}  "
                  f"P[supp]≈{n_supp_ok/n_runs:.3f}  "
                  f"P[cond]≈{n_cond_ok/n_runs:.3f}  "
                  f"elapsed={time.perf_counter()-t_start:.0f}s")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__" or "snakemake" in globals():
    main()
