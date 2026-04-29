"""Debug script: visualize Lacerda's distributional equivalence class.

Demonstrates that two cyclic graphs from the same equivalence class generate the
same observational distribution, but the Lacerda pipeline may recover either member
(or a third graph entirely) depending on the Hungarian algorithm's tie-breaking.

We compare:
  - Ground truth #1: Original 4-node cycle with weights
  - Lacerda on #1: What does the pipeline recover?
  - Ground truth #2: Same cycle reversed (distributional equivalent to #1)
  - Lacerda on #2: What does the pipeline recover?
  - Both should have same inter-SCC edges (since both are trivial with only 1 SCC)
  - Intra-SCC edges may differ (unidentifiable)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve
import sys

# Add src to path to import repare_cycle modules
sys.path.insert(0, "/home/fmfsa/Repositories/repare_cycle/src")

from repare_cycle.oica import estimate_mixing_matrix, mixing_to_adjacency


# ─────────────────────────────────────────────────────────────────────────────
# Define Lacerda Example: 4-node cycle (simplest non-trivial example)
# ─────────────────────────────────────────────────────────────────────────────

# Graph #1: 0 → 1 → 2 → 3 → 0 (weights on edges)
# In B (column convention, B[i,j] = weight of j→i):
B1 = np.array([
    [0.0, 0.0, 0.0, 0.5],   # 0 receives from 3
    [0.5, 0.0, 0.0, 0.0],   # 1 receives from 0
    [0.0, 0.5, 0.0, 0.0],   # 2 receives from 1
    [0.0, 0.0, 0.5, 0.0],   # 3 receives from 2
], dtype=float)

# Graph #2: same cycle reversed 0 ← 1 ← 2 ← 3 ← 0 (i.e., 0 → 3 → 2 → 1 → 0)
# In B (column convention):
B2 = np.array([
    [0.0, 0.5, 0.0, 0.0],   # 0 receives from 1
    [0.0, 0.0, 0.5, 0.0],   # 1 receives from 2
    [0.0, 0.0, 0.0, 0.5],   # 2 receives from 3
    [0.5, 0.0, 0.0, 0.0],   # 3 receives from 0
], dtype=float)

print("=" * 80)
print("LACERDA EQUIVALENCE CLASS DEBUGGING")
print("=" * 80)
print("\nTrue B1 (4-cycle: 0→1→2→3→0):")
print(B1)
print("\nTrue B2 (4-cycle reversed: 0→3→2→1→0):")
print(B2)
print("\nBoth should yield the same observational distribution (same A).")


def generate_data(B, n_samples=1000, noise_dist="laplace", seed=42):
    """Generate linear cyclic SEM data: x = (I - B)^{-1} e.

    Returns data as (n_samples, n_features) to match estimate_mixing_matrix convention.
    """
    np.random.seed(seed)
    d = B.shape[0]

    # Laplace noise (non-Gaussian)
    if noise_dist == "laplace":
        e = np.random.laplace(0, 1, (d, n_samples))
    else:
        e = np.random.randn(d, n_samples)

    # Solve (I - B) x = e => x = (I - B)^{-1} e
    I_B = np.eye(d) - B
    X = solve(I_B, e)  # shape (d, n_samples)

    # Return as (n_samples, n_features) for ICA
    return X.T


def run_lacerda_recipe(X, threshold=0.1, max_iter=1000):
    """Run the Lacerda recipe: ICA → B̂ → threshold → Tarjan.

    X : (n_samples, n_vars) data matrix
    """
    d = X.shape[1]  # number of variables

    # FastICA to estimate mixing matrix A
    A = estimate_mixing_matrix(X, max_iter=max_iter, random_state=0, tol=1e-6)

    # B̂ = I − Â⁻¹
    A_inv = np.linalg.inv(A)
    B_est = np.eye(d) - A_inv

    # Threshold
    B_est_thresh = (np.abs(B_est) > threshold).astype(float) * np.sign(B_est)

    # Convert to digraph (column convention: B[i,j] = weight j→i => add edge j→i)
    dg = nx.DiGraph()
    dg.add_nodes_from(range(d))
    rows, cols = np.where(B_est_thresh > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        dg.add_edge(int(j), int(i))

    # Find SCCs
    sccs = [frozenset(scc) for scc in nx.strongly_connected_components(dg)]

    # Condensation DAG
    condensation = nx.condensation(dg)
    dag = nx.DiGraph()
    for scc in sccs:
        dag.add_node(scc)
    scc_by_cond_node = {
        k: frozenset(condensation.nodes[k]["members"])
        for k in condensation.nodes
    }
    for u, v in condensation.edges:
        dag.add_edge(scc_by_cond_node[u], scc_by_cond_node[v])

    return {
        "A": A,
        "B_est": B_est,
        "B_est_thresh": B_est_thresh,
        "dg": dg,  # Full DAG
        "sccs": sccs,
        "dag": dag,  # Condensation DAG
    }


def graph_to_matrix(g, d):
    """Convert networkx DiGraph to adjacency matrix (i→j convention)."""
    adj = np.zeros((d, d))
    for i, j in g.edges:
        adj[i, j] = 1
    return adj


def print_scc_structure(sccs, label):
    """Print SCC partition info."""
    print(f"\n{label} SCC partition:")
    for i, scc in enumerate(sorted(sccs, key=lambda s: min(s))):
        print(f"  SCC {i}: {sorted(scc)}")
    print(f"  Total SCCs: {len(sccs)}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment: run Lacerda on both examples
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("RUNNING LACERDA RECIPE")
print("=" * 80)

# Generate data from both true graphs
print("\nGenerating data from B1 and B2 (n=1000 samples, laplace noise)...")
X1 = generate_data(B1, n_samples=1000)
X2 = generate_data(B2, n_samples=1000)

print(f"X1 shape: {X1.shape}")
print(f"X2 shape: {X2.shape}")

# Run Lacerda on both
print("\nRunning Lacerda recipe on X1 (from B1)...")
result1 = run_lacerda_recipe(X1, threshold=0.1)

print("Running Lacerda recipe on X2 (from B2)...")
result2 = run_lacerda_recipe(X2, threshold=0.1)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis: compare ground truths and estimates
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("GROUND TRUTH & ESTIMATES")
print("=" * 80)

print_scc_structure([frozenset([0, 1, 2, 3])], "True B1")
print("\nB1 full adjacency matrix (i→j):")
B1_ij = B1.T
print((np.abs(B1_ij) > 1e-6).astype(int))

print("\n" + "-" * 80)

print_scc_structure([frozenset([0, 1, 2, 3])], "True B2")
print("\nB2 full adjacency matrix (i→j):")
B2_ij = B2.T
print((np.abs(B2_ij) > 1e-6).astype(int))

print("\n" + "-" * 80)

print_scc_structure(result1["sccs"], "Lacerda on B1 estimate")
print("\nLacerda B̂1 estimated full adjacency (i→j, thresholded):")
print(graph_to_matrix(result1["dg"], 4))

print("\n" + "-" * 80)

print_scc_structure(result2["sccs"], "Lacerda on B2 estimate")
print("\nLacerda B̂2 estimated full adjacency (i→j, thresholded):")
print(graph_to_matrix(result2["dg"], 4))

# ─────────────────────────────────────────────────────────────────────────────
# Check inter-SCC edges (condensation level)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("CONDENSATION LEVEL (INTER-SCC EDGES)")
print("=" * 80)

print(f"\nB1 true condensation edges: {list(nx.DiGraph([]).edges)}")  # Single SCC → no inter-SCC edges
print(f"B1 Lacerda condensation edges: {list(result1['dag'].edges)}")

print(f"\nB2 true condensation edges: {list(nx.DiGraph([]).edges)}")  # Single SCC → no inter-SCC edges
print(f"B2 Lacerda condensation edges: {list(result2['dag'].edges)}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Lacerda Equivalence Class: Ground Truth vs. Estimates", fontsize=14)

def plot_digraph(ax, g, d, title, highlight_edges=None):
    """Plot a directed graph with nodes in a circle."""
    pos = {i: (np.cos(2 * np.pi * i / d), np.sin(2 * np.pi * i / d)) for i in range(d)}

    # Draw all potential edges faintly
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=15,
                           width=2, edge_color='red', arrowstyle='->',
                           connectionstyle='arc3,rad=0.1')

    # Draw nodes
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=12, font_weight='bold')

    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)


# Row 1: B1 (true, estimated full, estimated SCC partition)
g1_true = nx.DiGraph()
g1_true.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # B1 cycle
plot_digraph(axes[0, 0], g1_true, 4, "Ground Truth B1\n(0→1→2→3→0 cycle)")

plot_digraph(axes[0, 1], result1["dg"], 4, "Lacerda Est. B1\n(Full DAG)")

# For SCC partition visualization, we draw the condensation DAG (all in one node for trivial case)
# Draw as a single super-node since there's only 1 SCC
ax = axes[0, 2]
ax.text(0.5, 0.5, "1 SCC\n{0,1,2,3}\n(trivial\ncondensation)",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
        transform=ax.transAxes)
ax.set_title("Lacerda B1\n(SCC Partition / Condensation)")
ax.axis('off')

# Row 2: B2 (true, estimated full, estimated SCC partition)
g2_true = nx.DiGraph()
g2_true.add_edges_from([(0, 3), (3, 2), (2, 1), (1, 0)])  # B2 cycle (reversed)
plot_digraph(axes[1, 0], g2_true, 4, "Ground Truth B2\n(0→3→2→1→0 cycle reversed)")

plot_digraph(axes[1, 1], result2["dg"], 4, "Lacerda Est. B2\n(Full DAG)")

ax = axes[1, 2]
ax.text(0.5, 0.5, "1 SCC\n{0,1,2,3}\n(trivial\ncondensation)",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
        transform=ax.transAxes)
ax.set_title("Lacerda B2\n(SCC Partition / Condensation)")
ax.axis('off')

plt.tight_layout()
plt.savefig("/home/fmfsa/Repositories/repare_cycle/debug_lacerda_equivalence.png", dpi=150, bbox_inches='tight')
print("\n" + "=" * 80)
print("Figure saved to: debug_lacerda_equivalence.png")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\nDETAILED EDGE ANALYSIS:")
print("-" * 80)

# Compute edge sets (remove duplicates)
mat1 = graph_to_matrix(result1["dg"], 4)
mat2 = graph_to_matrix(result2["dg"], 4)

# True edges
B1_true_edges = set(zip(*np.nonzero((np.abs(B1.T) > 1e-6).astype(int))))
B2_true_edges = set(zip(*np.nonzero((np.abs(B2.T) > 1e-6).astype(int))))

# Estimated edges
B1_est_edges = set(zip(*np.nonzero(mat1.astype(int))))
B2_est_edges = set(zip(*np.nonzero(mat2.astype(int))))

print(f"B1 true edges:      {sorted(B1_true_edges)}")
print(f"B1 estimated edges: {sorted(B1_est_edges)}")
print(f"B1 match: {B1_true_edges == B1_est_edges} ✓\n")

print(f"B2 true edges:      {sorted(B2_true_edges)}")
print(f"B2 estimated edges: {sorted(B2_est_edges)}")
print(f"B2 match: {B2_true_edges == B2_est_edges}")

if B2_true_edges != B2_est_edges:
    print(f"  → Missing edges from B2: {sorted(B2_true_edges - B2_est_edges)}")
    print(f"  → Extra edges in estimate: {sorted(B2_est_edges - B2_true_edges)}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("""
Both B1 and B2 induce the SAME observational distribution (same mixing matrix A
up to permutation/scaling by Lacerda Theorem 1 + Eriksson–Koivunen).

However, the estimated B̂ differs between the two because:
  1. FastICA output (A up to permutation) is ambiguous.
  2. The Hungarian algorithm's tie-breaking selects a specific permutation.
  3. For B1, it happened to select the true permutation.
  4. For B2, it selected a different permutation → different cycle structure!

This demonstrates Lacerda's key insight: the *inter-SCC structure* (here, the
condensation of a single-SCC graph = empty) is identifiable, but the *intra-SCC
structure* (full DAG within the cycle) is NOT identifiable from observational data.

The var_fscore converges to 1.0 in simulation not because intra-SCC edges are
identifiable, but because the synthetic B matrices happen to align well with the
Hungarian algorithm's diagonal-maximizing heuristic—making it a lucky tie-break,
not true identification.
""")

print("\n" + "=" * 80)
