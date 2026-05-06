"""Named ground-truth SEMs from the literature.

We hard-code published examples (rather than random graphs) so that
algorithm output can be checked against an analytically known target.
The first entry, ``lacerda_example_1``, reproduces the 5-variable SEM
of Lacerda et al. (UAI 2008), p. 1–2 (Eq. 2, Fig. 1).

All ``B`` matrices use the column convention used everywhere else in
``repare_cycle``::

    B[i, j] != 0   ⇔   edge   j  →  i

i.e. ``B[i, j]`` is the coefficient with which ``x_j`` enters the
structural equation for ``x_i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class NamedExample:
    """A named ground-truth linear cyclic SEM.

    Attributes
    ----------
    name : str
        Short identifier (used as a Snakemake wildcard / dict key).
    B : np.ndarray
        ``(d, d)`` weight matrix in column convention
        (``B[i, j] != 0  ⇔  j → i``).
    description : str
        One-line citation / structural summary.
    """

    name: str
    B: np.ndarray
    description: str

    @property
    def d(self) -> int:
        return self.B.shape[0]

    @property
    def true_dg(self) -> nx.DiGraph:
        """The directed graph induced by ``supp(B)``, in i→j convention."""
        g = nx.DiGraph()
        g.add_nodes_from(range(self.d))
        # B[i, j] != 0 means edge j → i
        for i, j in zip(*np.nonzero(self.B)):
            g.add_edge(int(j), int(i))
        return g

    @property
    def true_sccs(self) -> list[frozenset[int]]:
        return [frozenset(int(v) for v in s)
                for s in nx.strongly_connected_components(self.true_dg)]

    @property
    def true_condensation(self) -> nx.DiGraph:
        """The condensation DAG over the SCC partition."""
        cond = nx.condensation(self.true_dg)
        # Rebuild with frozenset(members) as node IDs to match downstream
        # consumers (e.g. fit.py).
        out = nx.DiGraph()
        members = {k: frozenset(int(v) for v in cond.nodes[k]["members"])
                   for k in cond.nodes}
        for s in members.values():
            out.add_node(s)
        for u, v in cond.edges:
            out.add_edge(members[u], members[v])
        return out


def _build_lacerda_example_1() -> NamedExample:
    """Lacerda et al. (UAI 2008), Eq. 2 / Fig. 1.

    The SEM is::

        x1 = e1
        x2 = 1.2 x1 - 0.3 x4 + e2
        x3 = 2 x2          + e3
        x4 =      - x3     + e4
        x5 = 3 x2          + e5

    yielding a 3-cycle ``x2 → x3 → x4 → x2`` inside SCC ``{x2, x3, x4}``.
    The full edge list (i → j convention) is::

        x1 → x2   (weight  1.2)
        x4 → x2   (weight -0.3)
        x2 → x3   (weight  2.0)
        x3 → x4   (weight -1.0)
        x2 → x5   (weight  3.0)

    The SCC partition is ``{x1}, {x2, x3, x4}, {x5}``. The cluster
    DAG (the identifiable target under the condensation-identifiability
    theorem) has edges ``{x1} → {x2,x3,x4}`` (via x1 → x2) and
    ``{x2,x3,x4} → {x5}`` (via x2 → x5) — two cluster edges total.
    """
    d = 5  # x1=0, x2=1, x3=2, x4=3, x5=4
    B = np.zeros((d, d), dtype=np.float64)
    # Column convention: B[i, j] = coefficient of x_j → x_i
    B[1, 0] = 1.2     # x1 → x2
    B[1, 3] = -0.3    # x4 → x2
    B[2, 1] = 2.0     # x2 → x3
    B[3, 2] = -1.0    # x3 → x4
    B[4, 1] = 3.0     # x2 → x5
    return NamedExample(
        name="lacerda_example_1",
        B=B,
        description=(
            "Lacerda et al. UAI 2008, Eq. 2 / Fig. 1 — 5-variable SEM with "
            "SCC partition {x1}, {x2,x3,x4}, {x5}; cycle x2→x3→x4→x2."
        ),
    )


EXAMPLES: dict[str, NamedExample] = {
    e.name: e for e in [
        _build_lacerda_example_1(),
    ]
}


def get(name: str) -> NamedExample:
    """Look up a named example by its identifier."""
    if name not in EXAMPLES:
        raise KeyError(
            f"Unknown example {name!r}. Known: {sorted(EXAMPLES)}"
        )
    return EXAMPLES[name]
