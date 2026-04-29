"""Graph utilities for directed graphs (DGs) with cycles.


Includes:
- Tarjan's strongly connected components algorithm (iterative, avoids recursion limits)
- Random directed graph generation (Erdős-Rényi and scale-free with back-edges)
- Cyclic linear structural equation model (SEwC) sampling

Background
----------
For a linear cyclic SCM (also called SEwC - Structural Equations with Cycles):

    X = B X + N  (column-vector convention, B[i,j] = weight of edge i→j)

The solution is:

    X = (I - B)^{-1} N

which requires (I - B) to be invertible. A sufficient condition is that the
spectral radius ρ(B) < 1. Under soft (shift) interventions on node k:

    X^I = (I - B)^{-1} (N + δ_k e_k)

so intervention effects propagate through the entire graph, including back
through cycles. This is why hard (do) interventions are problematic in cyclic
SEMs: setting X_k = c (a hard intervention) breaks the cyclic mechanism and
can render the system inconsistent.

Tarjan's Algorithm
------------------
Tarjan's algorithm finds all strongly connected components (SCCs) of a directed
graph in O(V + E) time. Each SCC is a maximal set of nodes that are mutually
reachable via directed paths. The SCCs form the nodes of the condensation DAG,
which is the finest valid DAG-coarsening of the original DG.

For the intervention-determined approach, nodes within the same SCC always
share the same intervened-ancestor signature (since they are mutual ancestors
of each other), so they are automatically grouped together by the RePaRe
algorithm without any modification. The SCC floor ensures no SCC is ever split.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def tarjan_scc(graph: nx.DiGraph) -> list[list]:
    """Compute strongly connected components using Tarjan's algorithm.

    Iterative implementation to avoid Python's recursion limit on large graphs.
    Returns SCCs in reverse topological order of the condensation DAG (i.e.,
    the first SCC has no outgoing edges to later SCCs).

    Parameters
    ----------
    graph : nx.DiGraph

    Returns
    -------
    sccs : list of lists, each list is one SCC (node labels from the graph)

    Notes
    -----
    NetworkX provides nx.strongly_connected_components and nx.condensation as
    built-in alternatives. This explicit implementation is provided for
    educational purposes and to make the algorithm's role in the theoretical
    framework transparent.
    """
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj = {
        node_to_idx[u]: [node_to_idx[v] for v in graph.successors(u)]
        for u in nodes
    }

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = set()
    sccs = []

    # Iterative Tarjan using an explicit call stack
    # Each frame: (node_idx, iterator_over_neighbors, is_new_frame)
    for start in range(len(nodes)):
        if start in index:
            continue
        call_stack = [(start, iter(adj[start]), True)]
        while call_stack:
            v, neighbors, is_new = call_stack[-1]
            if is_new:
                index[v] = lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack.add(v)
                call_stack[-1] = (v, neighbors, False)

            advanced = False
            for w in neighbors:
                if w not in index:
                    call_stack.append((w, iter(adj[w]), True))
                    advanced = True
                    break
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if not advanced:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])

                if lowlink[v] == index[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(nodes[w])
                        if w == v:
                            break
                    sccs.append(scc)

    return sccs


def scc_partition(graph: nx.DiGraph) -> list[frozenset]:
    """Return the partition of nodes by strongly connected component.

    Uses NetworkX's built-in implementation (Kosaraju's algorithm).

    Parameters
    ----------
    graph : nx.DiGraph

    Returns
    -------
    partition : list of frozensets, one per SCC
    """
    return [frozenset(scc) for scc in nx.strongly_connected_components(graph)]


def random_directed_graph(
    n: int,
    density: float,
    graph_type: str = "er",
    seed: int = 0,
    weight_range: tuple[float, float] = (0.25, 0.75),
) -> tuple[nx.DiGraph, np.ndarray]:
    """Generate a random directed graph and associated weight matrix.

    For ER graphs, uses Erdős-Rényi random directed graph (each directed edge
    included independently with probability `density`), which naturally produces
    cycles. For SF graphs, uses a Barabási-Albert undirected base then orients
    edges randomly and adds back-edges to create cycles.

    Weights are scaled so that the spectral radius ρ(B) < 1, guaranteeing
    stability of the cyclic linear SEM (I - B) invertible.

    Parameters
    ----------
    n : int
        Number of nodes.
    density : float
        Edge probability (ER) or density parameter (SF).
    graph_type : {"er", "sf"}
    seed : int
    weight_range : (w_min, w_max)
        Absolute value range for edge weights before sign randomization.

    Returns
    -------
    graph : nx.DiGraph
    weights : np.ndarray, shape (n, n)
        weights[i, j] is the weight of directed edge i → j (0 if no edge).
    """
    rng = np.random.default_rng(seed)

    if graph_type == "er":
        graph = nx.erdos_renyi_graph(n, density, seed=seed, directed=True)
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))

    elif graph_type == "sf":
        m_param = max(1, int(round(max(density * (n - 1) / 2, 1))))
        base_graph = nx.barabasi_albert_graph(n, m_param, seed=seed)
        order = rng.permutation(n)
        rank = {node: idx for idx, node in enumerate(order)}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        for u, v in base_graph.edges():
            src, dst = (u, v) if rank[u] < rank[v] else (v, u)
            graph.add_edge(src, dst)
        # Add back-edges with probability density/2 to introduce cycles
        for u, v in list(graph.edges()):
            if rng.random() < density / 2 and not graph.has_edge(v, u):
                graph.add_edge(v, u)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type!r}. Use 'er' or 'sf'.")

    # Build weight matrix
    w_min, w_max = weight_range
    weights = np.zeros((n, n))
    for u, v in graph.edges():
        w = rng.uniform(w_min, w_max)
        if rng.random() < 0.5:
            w = -w
        weights[u, v] = w

    # Scale weights to ensure spectral radius < 1 (stability of cyclic SEM)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
    if spectral_radius >= 1.0:
        weights = weights / (spectral_radius + 0.1)

    # Rebuild graph from weights (in case scaling zeroed any edges)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    rows, cols = np.nonzero(weights)
    for i, j in zip(rows, cols):
        graph.add_edge(int(i), int(j))

    return graph, weights


def random_multi_scc_graph(
    n_sccs: int,
    scc_sizes: list[int] | int,
    n_singletons: int = 0,
    inter_density: float = 0.4,
    intra_extra_edges: int = 1,
    seed: int = 0,
    weight_range: tuple[float, float] = (0.2, 0.6),
) -> tuple[nx.DiGraph, np.ndarray]:
    """Generate a directed graph with a *controlled* number of non-trivial SCCs.

    Each SCC is constructed as a directed cycle (the minimal structure that
    forces mutual reachability) plus ``intra_extra_edges`` random extra directed
    edges within the SCC (to create richer cyclic topology).  SCCs are then
    connected by random directed edges that respect a DAG ordering of the SCCs
    (so no inter-SCC cycles are introduced), with probability ``inter_density``
    for each possible cross-SCC edge.

    Optionally, ``n_singletons`` isolated nodes (trivial SCCs of size 1) are
    added and integrated into the DAG structure between SCCs.

    Parameters
    ----------
    n_sccs : int
        Number of non-trivial SCCs (each will have size >= 2).
    scc_sizes : int or list[int]
        Size of each SCC.  If a single int, all SCCs have that size.
    n_singletons : int
        Number of singleton nodes (trivial SCCs) to add.  These are placed
        randomly in the DAG ordering between SCCs.
    inter_density : float
        Probability of a directed edge between two different SCCs (only added
        in the causal direction to preserve acyclicity at the SCC level).
    intra_extra_edges : int
        Number of extra random directed edges to add *within* each SCC
        (beyond the minimal cycle).
    seed : int
    weight_range : (w_min, w_max)
        Absolute-value range for edge weights.

    Returns
    -------
    graph : nx.DiGraph
    weights : np.ndarray, shape (n, n)
        weights[i, j] = weight of edge i -> j (0 if no edge).
    """
    rng = np.random.default_rng(seed)

    if isinstance(scc_sizes, int):
        scc_sizes = [scc_sizes] * n_sccs

    if len(scc_sizes) != n_sccs:
        raise ValueError("len(scc_sizes) must equal n_sccs")

    n = sum(scc_sizes) + n_singletons
    w_min, w_max = weight_range
    weights = np.zeros((n, n))

    # Build node index ranges for each non-trivial SCC
    scc_starts = [sum(scc_sizes[:i]) for i in range(n_sccs)]
    scc_nodes = [
        list(range(scc_starts[k], scc_starts[k] + scc_sizes[k]))
        for k in range(n_sccs)
    ]

    # Singleton nodes occupy the remaining indices
    singleton_start = sum(scc_sizes)
    singleton_nodes = list(range(singleton_start, singleton_start + n_singletons))

    # Intra-SCC: directed cycle + extra edges
    for nodes in scc_nodes:
        m = len(nodes)
        # Directed cycle: 0->1->2->...->0
        for i in range(m):
            u, v = nodes[i], nodes[(i + 1) % m]
            w = rng.uniform(w_min, w_max)
            if rng.random() < 0.5:
                w = -w
            weights[u, v] = w
        # Extra random intra-SCC edges (avoid duplicates and self-loops)
        all_intra = [(u, v) for u in nodes for v in nodes if u != v
                     and weights[u, v] == 0]
        rng.shuffle(all_intra)
        for u, v in all_intra[:intra_extra_edges]:
            w = rng.uniform(w_min, w_max)
            if rng.random() < 0.5:
                w = -w
            weights[u, v] = w

    # Build a random DAG ordering over all blocks (SCCs + singletons).
    # Each block is either a non-trivial SCC or a singleton node.
    all_blocks = list(range(n_sccs + n_singletons))
    block_order = list(rng.permutation(all_blocks))

    def _block_nodes(block_idx: int) -> list[int]:
        if block_idx < n_sccs:
            return scc_nodes[block_idx]
        return [singleton_nodes[block_idx - n_sccs]]

    # Inter-block edges: directed from earlier blocks to later blocks only
    for ki in range(len(block_order)):
        for kj in range(ki + 1, len(block_order)):
            src_block = block_order[ki]
            dst_block = block_order[kj]
            for u in _block_nodes(src_block):
                for v in _block_nodes(dst_block):
                    if rng.random() < inter_density:
                        w = rng.uniform(w_min, w_max)
                        if rng.random() < 0.5:
                            w = -w
                        weights[u, v] = w

    # Scale weights to ensure spectral radius < 1
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
    if spectral_radius >= 1.0:
        weights = weights / (spectral_radius + 0.1)

    # Build graph from weights
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    rows, cols = np.nonzero(weights)
    for i, j in zip(rows, cols):
        graph.add_edge(int(i), int(j))

    return graph, weights


def random_cyclic_graph(
    d: int,
    num_cycles: int,
    density: float = 0.5,
    seed: int = 0,
    weight_range: tuple[float, float] = (0.25, 0.75),
) -> tuple[nx.DiGraph, np.ndarray]:
    """Generate a random directed graph on *d* nodes with exactly *num_cycles*
    non-trivial SCCs.

    This is the primary generator for the synthetic experiments.  It mirrors
    the experimental setup of the original RePaRe paper (which varies the
    number of interventions ι ∈ {2, 5, 8} on d = 10 DAG nodes) but instead
    varies the number of cycles κ ∈ {0, 1, 2, 3, ...}.

    Construction
    ------------
    1. Randomly partition *d* nodes into *num_cycles* groups of size ≥ 2
       (the non-trivial SCCs) plus remaining singletons.  Group sizes are
       drawn uniformly at random subject to the constraint that each group
       has at least 2 members.
    2. Within each group a directed Hamilton cycle is created (guaranteeing
       mutual reachability) plus extra random intra-SCC directed edges
       controlled by *density*.
    3. Between blocks (SCCs and singletons) DAG-respecting directed edges
       are added with probability *density*.

    When *num_cycles* = 0 the result is a random DAG (no cycles at all).

    Parameters
    ----------
    d : int
        Total number of nodes.
    num_cycles : int
        Number of non-trivial SCCs (cycles).  Each SCC will have ≥ 2 nodes.
    density : float
        Controls both intra-SCC extra edges and inter-block edge probability.
    seed : int
    weight_range : (w_min, w_max)
        Absolute-value range for edge weights before sign randomisation.

    Returns
    -------
    graph : nx.DiGraph
    weights : np.ndarray, shape (d, d)
    """
    rng = np.random.default_rng(seed)

    if num_cycles < 0:
        raise ValueError("num_cycles must be >= 0")
    if 2 * num_cycles > d:
        raise ValueError(
            f"Cannot fit {num_cycles} cycles (each >= 2 nodes) into {d} nodes"
        )

    w_min, w_max = weight_range
    weights = np.zeros((d, d))

    # ── 1. Partition nodes into SCC groups + singletons ──────────────
    if num_cycles == 0:
        # Pure DAG — all nodes are singletons
        scc_sizes: list[int] = []
        n_singletons = d
    else:
        # Start each SCC at minimum size 2
        scc_sizes = [2] * num_cycles
        budget = d - 2 * num_cycles  # nodes left to allocate

        # Randomly grow SCCs with some of the budget; rest → singletons
        # Each leftover node has ~50% chance of joining a random SCC
        for _ in range(budget):
            if rng.random() < 0.5 and num_cycles > 0:
                idx = rng.integers(num_cycles)
                scc_sizes[idx] += 1
            # else: stays as singleton (handled below)

        n_singletons = d - sum(scc_sizes)

    # Shuffle the sizes so large/small SCCs aren't always in the same position
    rng.shuffle(scc_sizes)

    # Assign node indices to each SCC
    scc_starts = [sum(scc_sizes[:i]) for i in range(num_cycles)]
    scc_nodes = [
        list(range(scc_starts[k], scc_starts[k] + scc_sizes[k]))
        for k in range(num_cycles)
    ]
    singleton_start = sum(scc_sizes)
    singleton_nodes = list(range(singleton_start, singleton_start + n_singletons))

    # ── 2. Intra-SCC edges: directed cycle + density-controlled extras ──
    for nodes in scc_nodes:
        m = len(nodes)
        # Directed Hamilton cycle
        for i in range(m):
            u, v = nodes[i], nodes[(i + 1) % m]
            w = rng.uniform(w_min, w_max) * rng.choice([-1, 1])
            weights[u, v] = w
        # Extra intra-SCC edges controlled by density
        for u in nodes:
            for v in nodes:
                if u != v and weights[u, v] == 0 and rng.random() < density:
                    weights[u, v] = rng.uniform(w_min, w_max) * rng.choice([-1, 1])

    # ── 3. Inter-block DAG edges ─────────────────────────────────────
    all_blocks = list(range(num_cycles + n_singletons))
    block_order = list(rng.permutation(all_blocks))

    def _block_nodes(block_idx: int) -> list[int]:
        if block_idx < num_cycles:
            return scc_nodes[block_idx]
        return [singleton_nodes[block_idx - num_cycles]]

    for ki in range(len(block_order)):
        for kj in range(ki + 1, len(block_order)):
            for u in _block_nodes(block_order[ki]):
                for v in _block_nodes(block_order[kj]):
                    if rng.random() < density:
                        weights[u, v] = (
                            rng.uniform(w_min, w_max) * rng.choice([-1, 1])
                        )

    # ── 4. Stability: spectral radius < 1 ────────────────────────────
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
    if spectral_radius >= 1.0:
        weights = weights / (spectral_radius + 0.1)

    # ── 5. Build NetworkX graph ───────────────────────────────────────
    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    rows, cols = np.nonzero(weights)
    for i, j in zip(rows, cols):
        graph.add_edge(int(i), int(j))

    return graph, weights


class CyclicLinearSEM:
    """Linear cyclic structural equation model (SEwC).

    Model (column-vector convention):
        X = B X + N
        X = (I - B)^{-1} N

    where B[i, j] is the weight of directed edge i → j, so X_j depends on X_i
    via weight B[i, j].

    Soft (shift) interventions add to the noise of the target node:
        X^I = (I - B)^{-1} (N + shift_I)

    This is the only type of intervention that is well-defined for cyclic SEMs:
    hard (do) interventions break the cyclic mechanism and can lead to
    inconsistencies.

    Stability condition: spectral radius ρ(B) < 1 ensures (I - B) is invertible
    and the power-series expansion (I - B)^{-1} = I + B + B² + ... converges.
    """

    # Supported non-Gaussian distributions for ICA identifiability.
    # Gaussian is also supported but cannot be used with the OICA approach.
    _NOISE_DISTS = frozenset({"gaussian", "laplace", "uniform", "exponential"})

    def __init__(
        self,
        weights: np.ndarray,
        means: tuple[float, float] = (-2.0, 2.0),
        variances: tuple[float, float] = (0.5, 2.0),
        rng: np.random.Generator | None = None,
        noise_dist: str = "laplace",
    ):
        """
        Parameters
        ----------
        weights : np.ndarray, shape (n, n)
            Weight matrix. weights[i, j] = weight of edge i → j.
        means : (lo, hi)
            Range for sampling per-node noise means uniformly.
        variances : (lo, hi)
            Range for sampling per-node noise variances uniformly.
        rng : np.random.Generator, optional
        noise_dist : {"laplace", "gaussian", "uniform", "exponential"}
            Marginal distribution for each node's noise term.
            "laplace" (default) is non-Gaussian and works well with FastICA.
            "gaussian" is supported for the intervention-determined approach
            but ICA cannot identify Gaussian mixing matrices.
        """
        if noise_dist not in self._NOISE_DISTS:
            raise ValueError(
                f"Unknown noise_dist '{noise_dist}'. "
                f"Choose from {sorted(self._NOISE_DISTS)}."
            )
        self.noise_dist = noise_dist
        self.weights = np.asarray(weights, dtype=float)
        self.n = self.weights.shape[0]
        self.rng = rng if rng is not None else np.random.default_rng(0)

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(self.weights))))
        if spectral_radius >= 1.0:
            raise ValueError(
                f"Spectral radius {spectral_radius:.4f} >= 1. "
                "The cyclic SEM is unstable. Use random_directed_graph() which "
                "automatically scales weights to ensure stability."
            )

        # Precompute (I - B)^{-1} for efficient sampling
        self._solution_matrix = np.linalg.inv(np.eye(self.n) - self.weights)

        # Per-node noise parameters (sampled once, fixed for the model)
        self.noise_means = self.rng.uniform(*means, size=self.n)
        self.noise_vars = self.rng.uniform(*variances, size=self.n)

    def _sample_noise(self, n_samples: int) -> np.ndarray:
        """Sample the n × d noise matrix using the configured distribution."""
        noise = np.empty((n_samples, self.n))
        for j in range(self.n):
            mu = self.noise_means[j]
            sigma = float(np.sqrt(self.noise_vars[j]))
            if self.noise_dist == "gaussian":
                noise[:, j] = self.rng.normal(mu, sigma, size=n_samples)
            elif self.noise_dist == "laplace":
                # Laplace(mu, b) has variance 2b^2; set b = sigma/sqrt(2)
                noise[:, j] = self.rng.laplace(mu, sigma / np.sqrt(2), size=n_samples)
            elif self.noise_dist == "uniform":
                # Uniform(mu - a, mu + a) has variance a^2/3; set a = sigma*sqrt(3)
                a = sigma * np.sqrt(3)
                noise[:, j] = self.rng.uniform(mu - a, mu + a, size=n_samples)
            elif self.noise_dist == "exponential":
                # Exponential(lambda) shifted to mean mu: scale = sigma, shift so mean = mu
                noise[:, j] = self.rng.exponential(sigma, size=n_samples) + mu - sigma
        return noise

    def sample(
        self,
        n_samples: int,
        shift_interventions: dict[int, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Sample from the cyclic SEM.

        Parameters
        ----------
        n_samples : int
        shift_interventions : dict mapping node index → (shift_mean, shift_var)
            Soft interventions that add Gaussian shifts to the noise of the target.
            The effect propagates through all directed paths including cycles.

        Returns
        -------
        data : np.ndarray, shape (n_samples, n_nodes)
        """
        noise = self._sample_noise(n_samples)

        if shift_interventions:
            for node, (shift_mean, shift_var) in shift_interventions.items():
                noise[:, node] += self.rng.normal(
                    shift_mean, np.sqrt(shift_var), size=n_samples
                )

        # X (n_samples × n) = N (n_samples × n) @ (I - B)^{-1}
        return noise @ self._solution_matrix
