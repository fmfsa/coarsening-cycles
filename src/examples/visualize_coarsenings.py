"""Example figure: true cyclic DG with multiple SCCs vs OICA-estimated coarsening.

Three rows — 5, 10, 20 nodes — each with 2–4 non-trivial SCCs.
Three columns:
  Col 1 — True DG (nodes coloured by SCC)
  Col 2 — True condensation DAG (ground truth coarsening)
  Col 3 — OICA-estimated partition DAG

Tarjan's role in the ALGORITHM:
  oica_scc_partition() calls nx.strongly_connected_components (= Kosaraju's,
  equivalent to Tarjan's) on the *estimated* support graph of the ICA mixing
  matrix.  Tarjan's is therefore a core algorithmic step, not only used for
  evaluation.

Run:
    PYTHONPATH=src python src/examples/visualize_coarsenings.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from repare_cycle.graph import CyclicLinearSEM, random_multi_scc_graph
from repare_cycle.repare_cycle import PartitionDgModelOICA

PALETTE = list(plt.cm.tab10.colors)


def scc_colour_map(sccs):
    cmap = {}
    for i, scc in enumerate(sccs):
        for v in scc:
            cmap[v] = PALETTE[i % len(PALETTE)]
    return cmap


def draw_true_dg(ax, graph, sccs, title):
    node_list = sorted(graph.nodes())
    cmap = scc_colour_map(sccs)
    colors = [cmap[v] for v in node_list]
    pos = nx.spring_layout(graph, seed=7, k=1.8)
    nx.draw_networkx(
        graph, pos=pos, ax=ax, nodelist=node_list,
        node_color=colors, node_size=500,
        font_size=9, font_color="white", font_weight="bold",
        arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.1",
    )
    patches = [
        mpatches.Patch(
            color=PALETTE[i % len(PALETTE)],
            label=f"SCC {i}: {{{','.join(str(v) for v in sorted(s))}}}",
        )
        for i, s in enumerate(sccs)
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower left", framealpha=0.75)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_partition_dag(ax, dag, sccs, title):
    if dag.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return

    cmap = scc_colour_map(sccs)
    node_list = list(dag.nodes)
    colors = [cmap.get(sorted(p)[0], "lightgrey") for p in node_list]
    labels = {p: "{" + ",".join(str(v) for v in sorted(p)) + "}" for p in node_list}

    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(dag, prog="dot")
    except Exception:
        pos = nx.spring_layout(dag, seed=3)

    nx.draw_networkx(
        dag, pos=pos, ax=ax, labels=labels,
        node_color=colors, node_size=900,
        font_size=8, arrows=True, arrowsize=12,
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def build_true_condensation(graph):
    cond = nx.condensation(graph)
    scc_map = {k: frozenset(cond.nodes[k]["members"]) for k in cond.nodes}
    dag = nx.DiGraph()
    for k in cond.nodes:
        dag.add_node(tuple(sorted(scc_map[k])))
    for u, v in cond.edges:
        dag.add_edge(tuple(sorted(scc_map[u])), tuple(sorted(scc_map[v])))
    return dag


# ---------------------------------------------------------------------------
# Configurations — all use random_multi_scc_graph to guarantee multiple SCCs
# (n_sccs non-trivial SCCs of given sizes)
# ---------------------------------------------------------------------------
CONFIGS = [
    # (n_sccs, scc_sizes,        inter_density, n_samples, seed, label)
    (3, [2, 2, 2],             0.5, 3_000,  0, "n=6  (3 SCCs of size 2)"),
    (3, [3, 3, 2],             0.4, 8_000,  1, "n=8  (2 SCCs of size 3, 1 of size 2)"),
    (4, [4, 3, 3, 2],          0.4, 20_000, 2, "n=12 (SCCs of size 4,3,3,2)"),
]

fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(16, 5.5 * len(CONFIGS)))
fig.suptitle(
    "OICA:  True cyclic DG  →  Ground-truth condensation  →  Estimated coarsening\n"
    "(Tarjan's SCC is a core step of the ALGORITHM — run on the ICA-estimated support graph)",
    fontsize=11, y=1.01,
)

for row, (n_sccs, scc_sizes, inter_d, n_samp, seed, label) in enumerate(CONFIGS):
    print(f"\n[{label}]")
    graph, weights = random_multi_scc_graph(
        n_sccs=n_sccs, scc_sizes=scc_sizes,
        inter_density=inter_d, intra_extra_edges=1, seed=seed,
    )
    sccs = sorted(nx.strongly_connected_components(graph), key=lambda s: min(s))
    n = sum(scc_sizes)
    print(f"  True SCCs: {[sorted(s) for s in sccs]}")

    model_sem = CyclicLinearSEM(
        weights, means=(0.0, 0.0), variances=(0.8, 1.2),
        rng=np.random.default_rng(seed), noise_dist="laplace",
    )
    obs = model_sem.sample(n_samp)

    print(f"  Running OICA ({n_samp:,} samples)...")
    oica_model = PartitionDgModelOICA()
    oica_model.fit(obs, beta=0.01)
    est_sccs = oica_model.oica_sccs

    print(f"  OICA SCCs (before CI edge test): {sorted([sorted(s) for s in est_sccs])}")
    print(f"  Final partition nodes: {sorted([sorted(p) for p in oica_model.dag.nodes])}")

    true_cond = build_true_condensation(graph)

    draw_true_dg(
        axes[row][0], graph, sccs,
        f"True DG — {label}\n"
        f"{n_sccs} non-trivial SCCs",
    )
    draw_partition_dag(
        axes[row][1], true_cond, sccs,
        "True condensation (ground truth)\n"
        f"[Tarjan's on true graph → {n_sccs} blocks]",
    )
    draw_partition_dag(
        axes[row][2], oica_model.dag, sccs,
        f"OICA estimate ({n_samp:,} obs. samples)\n"
        f"[Tarjan's on ICA support → {len(est_sccs)} blocks found]",
    )

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "coarsenings_example.pdf")
plt.savefig(out, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
