"""Recursive Partition Refinement for directed graphs with cycles (RePaRe-Cycle).

Extension of the RePaRe algorithm (Coarsening Causal DAG Models) to directed
graphs (DGs) that may contain cycles.

Two approaches are implemented:

1. **Intervention-determined** (``PartitionDgModelIvn``):
   Uses soft interventional data with unknown targets to identify SCCs.
   Nodes in the same SCC always share identical intervention-descendant
   patterns and are automatically grouped together.

2. **OICA-based** (``PartitionDgModelOICA``):
   Identifies the finest DAG-coarsening (SCC partition) from observational
   data alone using FastICA (Lacerda et al., UAI 2008; Dai et al., ICLR 2026).
   The key idea: A = (I-B)^{-1} is identifiable up to column permutation and
   scaling; support(A) = transitive closure of G; SCCs(transitive closure) = SCCs(G).
   Soft interventions can then improve edge accuracy but are not needed for
   partition identifiability.

Hard (do) interventions are rejected in both models because they break
the cyclic SEM mechanism, potentially making the system inconsistent.
"""

from __future__ import annotations

from collections import deque
import itertools

import networkx as nx
import numpy as np
from scipy.stats import chi2, chi2_contingency, ks_2samp

from .utils import SimpleCanCorr


# ---------------------------------------------------------------------------
# Shared base: partition refinement loop + CI edge tests
# ---------------------------------------------------------------------------

class _PartitionDgModelBase:
    """Internal base class providing the partition refinement loop and CI tests.

    Subclasses must populate the following attributes before calling
    _run_repare_loop():
        self.obs          — (n_samples, n_vars) observational data
        self.obs_type     — str, typically "obs"
        self.data_dict    — dict of key → (n_samples, n_vars) arrays (ivn envs)
        self.env_targets  — dict of key → set of target node indices
        self.env_types    — dict of key → str ("soft" only)
        self.beta         — float, significance level for edge tests
        self.assume       — {None, "gaussian", "discrete"}
        self.partition    — deque of sets (causally ordered blocks)
        self.dag          — nx.DiGraph (initialized with one node = all vars)
        self.refinable    — deque of sets (blocks to still refine)
        self.edge_tests   — list (for logging test outcomes)
    """

    def _refine(self):
        to_refine = self.refinable.pop()
        u = self.partition.popleft()
        v = to_refine - u
        if self.partition and v != self.partition[-1]:
            self.refinable.append(v)
        return to_refine, u, v

    def _recurse(self):
        to_refine, u, v = self._refine()
        if not u or not v:
            return
        self.dag.add_node(tuple(u))
        self.dag.add_node(tuple(v))
        if self._is_adj(u, v, self.dag.predecessors(tuple(to_refine))):
            self.dag.add_edge(tuple(u), tuple(v))
        for pa in self.dag.predecessors(tuple(to_refine)):
            for ch in (u, v):
                conditioning_set = (
                    c for c in self.dag.predecessors(tuple(to_refine)) if c != pa
                )
                if ch == v:
                    conditioning_set = itertools.chain(conditioning_set, (u,))
                if self._is_adj(set(pa), ch, conditioning_set):
                    self.dag.add_edge(pa, tuple(ch))
        for pa in (u, v):
            for ch in self.dag.successors(tuple(to_refine)):
                conditioning_set = (
                    c for c in self.dag.predecessors(tuple(to_refine)) if c != pa
                )
                if self._is_adj(pa, set(ch), conditioning_set):
                    self.dag.add_edge(tuple(pa), ch)
        self.dag.remove_node(tuple(to_refine))

    def _is_adj(self, pa, ch, conditioning_set=None) -> bool:
        """IsEdgeTest (Algorithm 4): test edge between partition blocks.

        For cyclic DGs with soft-only interventions, all environments are
        used for the independence test (no environments need to be skipped).
        """
        z_indices = []
        if conditioning_set:
            for part in conditioning_set:
                z_indices.extend(part)
        z_indices = sorted(set(z_indices))

        if self.assume is None:

            def p_val(x, y, z=None):  # z unsupported by distance correlation
                import dcor
                return dcor.independence.distance_correlation_t_test(x, y).pvalue

        elif self.assume == "gaussian":

            def p_val(x, y, z=None):
                try:
                    cca = SimpleCanCorr(x, y, Z=z)
                    return cca.wilks_lambda_test().loc[0, "Pr > F"]
                except (ValueError, np.linalg.LinAlgError):
                    return 1.0

        elif self.assume == "hsic":

            def p_val(x, y, z=None):  # z unsupported by HSIC
                from lingam.hsic import hsic_test_gamma
                _, p = hsic_test_gamma(x, y)
                return float(p)

        elif self.assume == "discrete":
            p_val = 1  # placeholder
        else:
            raise ValueError(f"Unsupported assumption '{self.assume}'")

        pa_indices = sorted(pa)
        ch_indices = sorted(ch)

        datasets = [(self.obs, set(), getattr(self, "obs_type", "obs"))] + [
            (env, self.env_targets.get(key, set()), self.env_types.get(key, "soft"))
            for key, env in self.data_dict.items()
        ]

        p_values = []
        for env, targets, env_type in datasets:
            x = env[:, pa_indices]
            y = env[:, ch_indices]
            if x.ndim == 1:
                x = x[:, None]
            if y.ndim == 1:
                y = y[:, None]
            z = env[:, z_indices] if z_indices else None
            p_values.append(p_val(x, y, z=z))

        if not p_values:
            self.edge_tests.append(
                {"pa": tuple(pa_indices), "ch": tuple(ch_indices),
                 "p_value": None, "decision": False, "reason": "insufficient_data"}
            )
            return False

        return float(max(p_values)) <= self.beta

    def _run_repare_loop(self):
        """Execute the partition refinement loop (Algorithm 1: RePaRe)."""
        n = self.obs.shape[1]
        self.dag = nx.DiGraph()
        self.dag.add_node(tuple(range(n)))
        self.refinable = deque([set(range(n))])
        self.edge_tests = []
        while self.refinable:
            self._recurse()

    def expand_coarsened_dag(self, fully_connected: bool = False) -> np.ndarray:
        """Expand the coarse partition DAG into a full variable-level adjacency matrix.

        Parameters
        ----------
        fully_connected : bool
            When True, each partition block is internally fully connected.

        Returns
        -------
        adjacency : np.ndarray, shape (n, n), dtype int
        """
        if not hasattr(self, "obs"):
            raise ValueError("Model must be fit before calling expand_coarsened_dag.")
        n = self.obs.shape[1]
        adjacency = np.zeros((n, n), dtype=int)

        if fully_connected:
            for part in self.dag.nodes:
                ordered = sorted(part)
                for idx, src in enumerate(ordered):
                    for dst in ordered[idx + 1:]:
                        adjacency[src, dst] = 1

        for src_part, dst_part in self.dag.edges:
            for src in src_part:
                for dst in dst_part:
                    adjacency[src, dst] = 1

        return adjacency


# ---------------------------------------------------------------------------
# Intervention-determined approach
# ---------------------------------------------------------------------------

class PartitionDgModelIvn(_PartitionDgModelBase):
    """Partition-DAG model using soft interventions to identify SCCs.

    Learns an interventional coarsening G^I of an underlying DG using soft
    interventional data with unknown targets. Hard (do) interventions raise
    a ValueError because they break the cyclic SEM mechanism.

    Algorithm
    ---------
    1. RefineAux: for each soft intervention I, compute M[v,I] = 1 iff
       variable v's distribution changes (two-sample test on each column).
    2. RefineTest: use M to build a totally ordered partition (Algorithm 3).
       Nodes with identical rows in M end up in the same block — this
       automatically preserves the SCC structure.
    3. RePaRe loop: recursively split blocks and test edges via CCA/Wilks' Λ.
    """

    def __init__(self, rng: np.random.Generator = np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng
        self.score = 0.0
        self.fit_metadata = {}

    def fit(
        self,
        data_dict: dict,
        alpha: float = 0.05,
        beta: float = 0.05,
        assume: str | None = "gaussian",
        refine_test: str = "ks",
    ) -> "PartitionDgModelIvn":
        """Fit from interventional data.

        Parameters
        ----------
        data_dict : dict
            "obs" key → observational data (array or (data, set(), "obs")).
            Other keys → soft interventional environments:
            (data, targets, "soft") tuples. Hard interventions raise ValueError.
        alpha : significance level for RefineAux (detecting affected variables).
        beta : significance level for IsEdgeTest (edge testing).
        assume : {"gaussian", None, "discrete"} distributional assumption.
        refine_test : {"ks", "energy"} two-sample test for RefineAux.
        """
        self.assume = assume
        self.refine_test = (refine_test or "ks").lower()
        data_dict = data_dict.copy()
        self.beta = beta
        obs = data_dict.pop("obs")

        obs_array, _, obs_type = _split_env(obs)
        self.obs = obs_array
        self.obs_type = obs_type

        processed_envs, env_targets, env_types = {}, {}, {}
        for key, env in data_dict.items():
            env_array, targets, env_type = _split_env(env)
            _require_soft(key, env_type)
            processed_envs[key] = env_array
            env_targets[key] = targets
            env_types[key] = env_type

        self.data_dict = processed_envs
        self.env_targets = env_targets
        self.env_types = env_types

        # --- RefineAux: build intervention-descendant indicator matrix M ---
        p_val_fn = _build_refine_pval_fn(self.obs, assume, refine_test)
        partition_masks = {
            idx: (np.asarray(p_val_fn(post_ivn)) < alpha)
            for idx, post_ivn in self.data_dict.items()
        }

        # --- RefineTest: totally ordered partition from M ---
        self.partition = _get_totally_ordered_partition(partition_masks)

        # --- RePaRe loop ---
        self._run_repare_loop()

        self.score = float("nan")
        self.fit_metadata = dict(
            alpha=alpha, beta=beta, score=self.score,
            num_parts=self.dag.number_of_nodes(),
            num_edges=self.dag.number_of_edges(),
        )
        return self


# ---------------------------------------------------------------------------
# OICA-based approach
# ---------------------------------------------------------------------------

class PartitionDgModelOICA(_PartitionDgModelBase):
    """Partition-DAG model using ICA to identify SCCs from observational data.

    Identifies the finest valid DAG-coarsening (= SCC partition) purely from
    observational data via FastICA, without requiring any interventions.
    Optionally, soft interventional data can be provided to improve edge
    accuracy in the CI-test phase.

    Algorithm
    ---------
    1. OICA phase (observational): estimate mixing matrix A = (I-B)^{-1} via
       FastICA. The support of A is the adjacency of the transitive closure of
       G, so running Tarjan's on support(A) recovers the SCC partition exactly.
       (Identifiable up to ICA ambiguity when A has no proportional columns.)
    2. Order SCCs via topological sort of the estimated condensation DAG.
    3. RePaRe loop: use CI tests to determine edges between SCC blocks.
       Soft interventional data (if provided) helps here but is not required.

    References
    ----------
    Lacerda et al. (2008). Discovering cyclic causal models by ICA. UAI.
    Dai, Albrecht, Spirtes, Zhang (2026). Distributional Equivalence in
        Linear Non-Gaussian Latent-Variable Cyclic Causal Models. ICLR.

    Attributes
    ----------
    dag : nx.DiGraph  — learned partition DAG.
    oica_mixing : np.ndarray  — estimated mixing matrix (I-B)^{-1}.
    oica_sccs : list[frozenset]  — SCC partition from ICA.
    """

    def __init__(self, rng: np.random.Generator = np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng
        self.fit_metadata = {}

    def fit(
        self,
        obs: np.ndarray,
        ivn_data: dict | None = None,
        beta: float = 0.05,
        assume: str | None = "gaussian",
        threshold: float = 0.1,
        max_iter: int = 10_000,
        random_state: int = 0,
        partition: "list[set[int]] | None" = None,
    ) -> "PartitionDgModelOICA":
        """Fit using OICA for SCC identification + optional CI-test edge refinement.

        Parameters
        ----------
        obs : (n_samples, n_vars) observational data array.
        ivn_data : optional dict of soft interventional environments,
            {key: (array, targets_set, "soft"), ...}. Hard interventions rejected.
        beta : significance level for IsEdgeTest.
        assume : {"gaussian", "hsic", None, "discrete"} for CI tests.
            "hsic" uses lingam.hsic.hsic_test_gamma (non-parametric; recommended
            when the cyclic SEM noise is non-Gaussian, e.g. Laplace).
        threshold : |A[i,j]| threshold for declaring a non-zero entry.
        max_iter : max FastICA iterations.
        random_state : FastICA seed.
        partition : optional precomputed partition (list of sets in topological
            order, sources first). If provided, the OICA partition step is
            skipped and the given partition is used for the RePaRe loop. Used
            by ``lacerda_repare`` to combine Lacerda's partition with RePaRe's
            CI-based edge tests.
        """
        from .oica import estimate_mixing_matrix

        self.obs = np.asarray(obs)
        self.obs_type = "obs"
        self.beta = beta
        self.assume = assume
        n = self.obs.shape[1]

        if partition is None:
            # --- Step 1: OICA — estimate mixing matrix and SCC partition ---
            self.oica_mixing = estimate_mixing_matrix(
                self.obs, max_iter=max_iter, random_state=random_state
            )

            # Support of A: entry [i,j] > threshold → j can reach i → edge j→i
            support = np.abs(self.oica_mixing) > threshold
            np.fill_diagonal(support, 0)
            support_graph = nx.DiGraph()
            support_graph.add_nodes_from(range(n))
            rows, cols = np.where(support)
            for i, j in zip(rows.tolist(), cols.tolist()):
                support_graph.add_edge(j, i)  # j→i (j reaches i)

            # Tarjan's SCCs on the support graph (= transitive closure of G)
            self.oica_sccs = [
                frozenset(scc) for scc in nx.strongly_connected_components(support_graph)
            ]

            # --- Step 2: build condensation and topological order ---
            condensation = nx.condensation(support_graph)
            scc_by_cond_node = {
                k: frozenset(condensation.nodes[k]["members"])
                for k in condensation.nodes
            }
            topo_order = list(nx.topological_sort(condensation))
            self.partition = deque(set(scc_by_cond_node[k]) for k in topo_order)
        else:
            # Caller supplied the partition (e.g. derived from Lacerda's B̂).
            # Must be in topological order (sources first).
            self.partition = deque(set(p) for p in partition)
            self.oica_mixing = None
            self.oica_sccs = [frozenset(p) for p in partition]

        # --- Step 3: process optional interventional data ---
        self.data_dict = {}
        self.env_targets = {}
        self.env_types = {}
        if ivn_data:
            for key, env in ivn_data.items():
                env_array, targets, env_type = _split_env(env)
                _require_soft(key, env_type)
                self.data_dict[key] = env_array
                self.env_targets[key] = targets
                self.env_types[key] = env_type

        # --- Step 4: RePaRe loop (edge testing via CI tests) ---
        self._run_repare_loop()

        self.fit_metadata = dict(
            beta=beta,
            threshold=threshold,
            num_sccs=len(self.oica_sccs),
            num_parts=self.dag.number_of_nodes(),
            num_edges=self.dag.number_of_edges(),
            partition_source=("oica" if partition is None else "external"),
        )
        return self


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _split_env(env):
    """Parse an environment spec into (data_array, target_set, env_type)."""
    env_type = None
    if isinstance(env, tuple):
        if len(env) == 2:
            data, targets = env
        elif len(env) == 3:
            data, targets, env_type = env
        else:
            raise ValueError(
                "Environment tuple must be (data, targets) or (data, targets, type)"
            )
    elif isinstance(env, dict):
        if "data" not in env:
            raise ValueError("Environment dict must contain a 'data' key")
        data = env["data"]
        targets = env.get("targets", ())
        env_type = env.get("type", env.get("intervention_type"))
    else:
        data = env
        targets = ()
    data_array = np.asarray(data)
    target_set = set(targets)
    if env_type is None:
        env_type = "obs" if len(target_set) == 0 else "soft"
    return data_array, target_set, env_type


def _require_soft(key, env_type):
    """Raise ValueError if the environment is a hard intervention."""
    if env_type in {"hard", "do"}:
        raise ValueError(
            f"Environment '{key}' is a hard intervention (type='{env_type}'). "
            "Hard interventions break cyclic SEM mechanisms and are not "
            "supported in RePaRe-Cycle. Use soft (shift) interventions only."
        )


def _build_refine_pval_fn(obs_data, assume, refine_test):
    """Return a function p_val(post_ivn) → array of p-values per variable."""

    if assume is None:

        def p_val(post_ivn):
            baseline = np.asarray(obs_data)
            comparison = np.asarray(post_ivn)
            if baseline.ndim == 1:
                baseline = baseline[:, None]
            if comparison.ndim == 1:
                comparison = comparison[:, None]
            num_feats = baseline.shape[1]
            pvals = np.empty(num_feats)
            test = (refine_test or "ks").lower()
            for j in range(num_feats):
                if test == "ks":
                    stat = ks_2samp(
                        baseline[:, j], comparison[:, j],
                        alternative="two-sided", mode="auto",
                    )
                    pvals[j] = stat.pvalue
                elif test == "energy":
                    import dcor
                    x, y = baseline[:, j], comparison[:, j]
                    result = dcor.homogeneity.energy_test(
                        x[:, None], y[:, None], num_resamples=199
                    )
                    pvals[j] = result.pvalue
                else:
                    raise ValueError(f"Unknown refine_test '{refine_test}'")
            return pvals

    elif assume == "gaussian":

        def p_val(post_ivn):
            num_feats = obs_data.shape[1]
            pvals = np.empty(num_feats)
            for j in range(num_feats):
                x, y = obs_data[:, j], post_ivn[:, j]

                def ll(d):
                    m = d.size
                    mu = d.mean()
                    s2 = max(float(np.mean((d - mu) ** 2)), 1e-12)
                    return -0.5 * m * (np.log(2 * np.pi * s2) + 1)

                LR = 2.0 * (ll(x) + ll(y) - ll(np.concatenate([x, y])))
                pvals[j] = chi2.sf(LR, df=2)
            return pvals

    elif assume == "discrete":

        def p_val(post_ivn):
            num_feats = obs_data.shape[1]
            pvals = np.empty(num_feats)
            for j in range(num_feats):
                x, y = obs_data[:, j], post_ivn[:, j]
                vals = np.concatenate([x, y])
                uniq, inv = np.unique(vals, return_inverse=True)
                V = uniq.size
                table = np.zeros((2, V), dtype=int)
                for idx in inv[: x.size]:
                    table[0, idx] += 1
                for idx in inv[x.size:]:
                    table[1, idx] += 1
                _, pval, _, _ = chi2_contingency(table, correction=False)
                pvals[j] = float(pval)
            return pvals

    else:
        raise ValueError(f"Unsupported assumption '{assume}'")

    return p_val


def _get_totally_ordered_partition(ivn_biadj: dict) -> deque:
    """Build a totally ordered partition from intervention-descendant indicators.

    Implements Algorithm 3 (RefineTest). Nodes in the same SCC always share
    identical rows in the M matrix, so they end up in the same block.

    Parameters
    ----------
    ivn_biadj : dict  — {key: bool array of length n_vars}
        M[v, I] = True iff variable v changed under intervention I.

    Returns
    -------
    partition : deque of sets, causally ordered (upstream blocks first).
    """
    num_atoms = len(next(iter(ivn_biadj.values())))
    partition = [list(range(num_atoms))]
    while ivn_biadj:
        idx, ivned = ivn_biadj.popitem()
        new_partition = []
        for part in partition:
            first_part = [atom for atom in part if not ivned[atom]]
            second_part = [atom for atom in part if ivned[atom]]
            if first_part and second_part:
                new_partition.append(first_part)
                new_partition.append(second_part)
            else:
                new_partition.append(part)
        partition = new_partition
    return deque(set(part) for part in partition)
