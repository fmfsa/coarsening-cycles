"""Shared drawing helpers for the graph-topology figures.

Used by plot_tvb_topology.py (Experiment 1) and plot_tvb_rww.py
(Experiment 2). Layouts are driven by the TRUE condensation's topological
generations so the feedback clusters and their causal ordering read
left-to-right; recovered C-DAGs are laid out by their own generations.
"""

import networkx as nx
import numpy as np
import seaborn as sns

ROSE = "#AE6B91"
SINGLETON_GRAY = "0.80"


def layered_positions(cdag, y_gap=1.8):
    """Cluster centers layered by topological generation of a C-DAG."""
    pos = {}
    for gx, gen in enumerate(nx.topological_generations(cdag)):
        clusters = sorted(gen, key=min)
        for i, cluster in enumerate(clusters):
            pos[cluster] = (float(gx), y_gap * ((len(clusters) - 1) / 2.0 - i))
    return pos


def member_positions(cluster_pos, radius=0.30):
    """Variables on a small circle around their cluster's center."""
    pos = {}
    for cluster, (cx, cy) in cluster_pos.items():
        members = sorted(cluster)
        if len(members) == 1:
            pos[members[0]] = (cx, cy)
            continue
        for k, v in enumerate(members):
            angle = 2 * np.pi * k / len(members) + np.pi / 2
            pos[v] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
    return pos


def scc_coloring(true_cdag):
    """(cluster_face, scc_of_var): distinct colors for nontrivial true SCCs,
    gray for singletons; every variable mapped to its true cluster."""
    nontrivial = sorted((c for c in true_cdag.nodes if len(c) > 1), key=min)
    palette = sns.color_palette("flare", len(nontrivial))
    cluster_face = {c: SINGLETON_GRAY for c in true_cdag.nodes}
    cluster_face.update(dict(zip(nontrivial, palette)))
    scc_of_var = {v: c for c in true_cdag.nodes for v in c}
    return cluster_face, scc_of_var


def draw_var_graph(ax, adj, var_pos, scc_of_var, cluster_face, title):
    """Directed variable graph, nodes colored by their TRUE SCC."""
    d = adj.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    rows, cols = np.where(adj > 0)
    graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
    intra = [(u, v) for u, v in graph.edges if scc_of_var[u] == scc_of_var[v]]
    inter = [(u, v) for u, v in graph.edges if scc_of_var[u] != scc_of_var[v]]
    common = dict(ax=ax, arrows=True, arrowstyle="-|>", arrowsize=9,
                  node_size=260, connectionstyle="arc3,rad=0.10")
    nx.draw_networkx_edges(
        graph, var_pos, edgelist=intra, width=1.8, alpha=0.85,
        edge_color=[cluster_face[scc_of_var[u]] for u, _ in intra], **common)
    nx.draw_networkx_edges(graph, var_pos, edgelist=inter, width=1.0,
                           alpha=0.45, edge_color="0.35", **common)
    node_face = [cluster_face[scc_of_var[v]] for v in range(d)]
    nx.draw_networkx_nodes(graph, var_pos, ax=ax, node_size=260,
                           node_color=node_face, edgecolors="white",
                           linewidths=1.0)
    nx.draw_networkx_labels(graph, var_pos, ax=ax, font_size=8)
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_aspect("equal")


def draw_cdag(ax, cdag, cluster_pos, cluster_face, title):
    """C-DAG with each cluster as one node; clusters matching a true SCC
    inherit its color, mismatched clusters are outlined in rose."""
    sizes = [520 + 430 * (len(c) - 1) for c in cdag.nodes]
    faces, edge_cols, styles = [], [], []
    for c in cdag.nodes:
        if c in cluster_face:              # exact match with a true cluster
            faces.append(cluster_face[c])
            edge_cols.append("white")
            styles.append("solid")
        else:                              # mismatched cluster
            faces.append("#F3E8EE")
            edge_cols.append(ROSE)
            styles.append("dashed")
    nx.draw_networkx_edges(cdag, cluster_pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=12, width=1.6,
                           edge_color="0.35", node_size=max(sizes),
                           connectionstyle="arc3,rad=0.08")
    nodes = nx.draw_networkx_nodes(cdag, cluster_pos, ax=ax, node_size=sizes,
                                   node_color=faces, edgecolors=edge_cols,
                                   linewidths=1.6)
    nodes.set_linestyle(styles)
    for c in cdag.nodes:
        colored = c in cluster_face and cluster_face[c] != SINGLETON_GRAY
        ax.annotate(",".join(str(v) for v in sorted(c)), cluster_pos[c],
                    ha="center", va="center", fontsize=8,
                    color="white" if colored else "black")
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_aspect("equal")
