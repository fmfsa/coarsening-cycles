"""Self-contained example: RePaRe-Cycle on a small directed graph with cycles.

Run from the repo root:
    python -c "import sys; sys.path.insert(0, 'src')" src/examples/example.py
Or more simply:
    PYTHONPATH=src python src/examples/example.py

Does NOT require the package to be installed (uses sys.path insert).
Does NOT require dcor (uses assume="gaussian").

The figure shows three panels:
  Left  — True DG, nodes coloured by SCC (same colour = same SCC)
  Centre — True interventional coarsening (partition DAG, ground truth)
  Right  — Discovered partition DAG from RePaRe-Cycle
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from repare_cycle.graph import CyclicLinearSEM, scc_partition
from repare_cycle.repare_cycle import PartitionDgModelIvn, _get_totally_ordered_partition

# ---------------------------------------------------------------------------
# 1. Build a hand-crafted DG with known SCC structure
#    Nodes:  0, 1  ← 2-cycle (SCC A)
#            2     ← singleton, receives from {0,1}   (SCC B)
#            3, 4  ← 2-cycle downstream of 2          (SCC C)
#            5     ← singleton, receives from {3,4}   (SCC D)
#
#    Condensation DAG:  A → B → C → D
# ---------------------------------------------------------------------------
n = 6
# weights[i, j] = weight of edge i → j
W = np.zeros((n, n))
# SCC A: 2-cycle 0 ↔ 1
W[0, 1] = 0.4
W[1, 0] = 0.4
# SCC A → SCC B
W[0, 2] = 0.5
W[1, 2] = 0.3
# SCC B → SCC C: 2-cycle 3 ↔ 4
W[2, 3] = 0.4
W[2, 4] = 0.3
# SCC C internal cycle
W[3, 4] = 0.35
W[4, 3] = 0.35
# SCC C → SCC D
W[3, 5] = 0.4
W[4, 5] = 0.3

# Check stability
sr = np.max(np.abs(np.linalg.eigvals(W)))
if sr >= 1.0:
    W = W / (sr + 0.1)
    print(f"Scaled weights (spectral radius was {sr:.3f})")

true_dg = nx.DiGraph(W.astype(bool))
true_sccs = scc_partition(true_dg)
print(f"True SCCs: {[sorted(s) for s in true_sccs]}")

# ---------------------------------------------------------------------------
# 2. Generate data from the cyclic SEM
# ---------------------------------------------------------------------------
n_samples = 3000
model_sem = CyclicLinearSEM(W, means=(0.0, 0.0), variances=(0.8, 1.2),
                             rng=np.random.default_rng(42))

obs = model_sem.sample(n_samples)

# Soft interventions: one per node (targets = each node individually)
targets = [(0,), (2,), (3,), (5,)]
ivn_data = {}
for i, (t,) in enumerate(targets):
    ivn_data[str(i)] = (
        model_sem.sample(n_samples, shift_interventions={t: (6.0, 0.5)}),
        {t},
        "soft",
    )

data_dict = {"obs": (obs, set(), "obs")}
data_dict.update(ivn_data)

# ---------------------------------------------------------------------------
# 3. Compute ground-truth interventional coarsening
# ---------------------------------------------------------------------------
target_des_masks = {}
for i, (t,) in enumerate(targets):
    mask = np.zeros(n, dtype=bool)
    mask[list(nx.descendants(true_dg, t))] = True
    mask[t] = True
    target_des_masks[str(i)] = mask

true_partition = list(_get_totally_ordered_partition(target_des_masks))
print(f"True interventional coarsening: {[sorted(p) for p in true_partition]}")

# Build ground-truth partition DAG
def _is_adj_dg(pa, ch, dg):
    return any(dg.has_edge(u, v) for u in pa for v in ch)

true_partition_dag = nx.DiGraph()
for p in true_partition:
    true_partition_dag.add_node(tuple(sorted(p)))
node_list = list(true_partition_dag.nodes)
for i, pa in enumerate(node_list):
    for ch in node_list[i + 1:]:
        if _is_adj_dg(set(pa), set(ch), true_dg):
            true_partition_dag.add_edge(pa, ch)

# ---------------------------------------------------------------------------
# 4. Run RePaRe-Cycle
# ---------------------------------------------------------------------------
fit_model = PartitionDgModelIvn()
fit_model.fit(data_dict, alpha=0.01, beta=0.01, assume="gaussian")
est_dag = fit_model.dag

print(f"Estimated partition: {[sorted(p) for p in est_dag.nodes]}")
print(f"Estimated edges: {list(est_dag.edges)}")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
# Colour nodes by SCC membership
scc_list = [frozenset(s) for s in true_sccs]
palette = plt.cm.tab10.colors
node_color_map = {}
for i, scc in enumerate(scc_list):
    for node in scc:
        node_color_map[node] = palette[i % len(palette)]
node_colors = [node_color_map[v] for v in sorted(true_dg.nodes)]


def _label_for_partition_node(part):
    return "{" + ",".join(str(v) for v in sorted(part)) + "}"


def _partition_dag_colors(partition_dag, true_scc_list, palette):
    """Colour each partition block by the SCC(s) it contains (first SCC colour)."""
    colors = []
    for part in partition_dag.nodes:
        part_set = set(part)
        for i, scc in enumerate(true_scc_list):
            if part_set & scc:
                colors.append(palette[i % len(palette)])
                break
        else:
            colors.append("lightgrey")
    return colors


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel 1: True DG ---
ax = axes[0]
ax.set_title("True DG (nodes coloured by SCC)", fontsize=11)
pos = nx.spring_layout(true_dg, seed=1)
nx.draw_networkx(
    true_dg, pos=pos, ax=ax,
    node_color=node_colors, node_size=600,
    arrows=True, arrowsize=15,
    font_color="white", font_weight="bold",
)
legend_patches = [
    mpatches.Patch(color=palette[i], label=f"SCC {i}: {{{','.join(str(v) for v in sorted(scc))}}}")
    for i, scc in enumerate(scc_list)
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower left")
ax.axis("off")

# --- Panel 2: True partition DAG ---
ax = axes[1]
ax.set_title("True interventional coarsening G^I", fontsize=11)
true_labels = {node: _label_for_partition_node(node) for node in true_partition_dag.nodes}
true_colors = _partition_dag_colors(true_partition_dag, scc_list, palette)
pos2 = nx.spring_layout(true_partition_dag, seed=2)
nx.draw_networkx(
    true_partition_dag, pos=pos2, ax=ax,
    labels=true_labels,
    node_color=true_colors, node_size=1200,
    arrows=True, arrowsize=15,
    font_size=9,
)
ax.axis("off")

# --- Panel 3: Estimated partition DAG ---
ax = axes[2]
ax.set_title("Discovered partition DAG (RePaRe-Cycle)", fontsize=11)
est_labels = {node: _label_for_partition_node(node) for node in est_dag.nodes}
est_colors = _partition_dag_colors(est_dag, scc_list, palette)
pos3 = nx.spring_layout(est_dag, seed=3)
nx.draw_networkx(
    est_dag, pos=pos3, ax=ax,
    labels=est_labels,
    node_color=est_colors, node_size=1200,
    arrows=True, arrowsize=15,
    font_size=9,
)
ax.axis("off")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "example_output.pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"\nFigure saved to: {out_path}")
plt.show()
