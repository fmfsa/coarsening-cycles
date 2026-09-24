"""SCC preflight of TVB's default 76-region connectome (descriptive only).

Loads the default connectivity shipped with tvb-data, converts it to
repository orientation (weights[src, dst] = TVB weights.T), and reports,
at each quantile of the nonzero-|weight| distribution, the SCC structure
of the thresholded directed graph: symmetry error, retained edges,
directed reciprocity, SCC counts, largest-SCC fraction, and condensation
edge count.

The connectome is never oriented or modified, and nothing here claims
causal recovery on it — near-symmetric weights collapse into one giant
SCC at low thresholds, which is exactly why the pilot's recovery claims
rest on the controlled synthetic benchmark instead.

Requires the `.[tvb]` extra (tvb-library + tvb-data).
"""

import pandas as pd

from repare_cycle.tvb_bridge import (
    connectome_threshold_stats,
    from_tvb_orientation,
    load_default_connectivity,
)

conn = load_default_connectivity()
weights = from_tvb_orientation(conn.weights)

rows = connectome_threshold_stats(weights)
for row in rows:
    row["n_regions"] = int(conn.number_of_regions)
    row["undirected_flag"] = int(bool(conn.undirected))

pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
print(
    f"Preflight of {conn.number_of_regions}-region default connectome: "
    f"largest-SCC fraction {rows[0]['largest_scc_fraction']:.2f} at q=0, "
    f"{rows[-1]['largest_scc_fraction']:.2f} at q={rows[-1]['quantile']}"
)
