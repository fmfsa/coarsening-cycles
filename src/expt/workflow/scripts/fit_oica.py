"""Fit a PartitionDgModelOICA to a synthetic dataset.

Uses FastICA on observational data to identify the SCC partition (finest
valid DAG-coarsening), then applies CI tests to determine edges between
SCC blocks. Soft interventional data is optionally used in the CI-test phase.
"""

import pickle
import time

import numpy as np
from repare_cycle.repare_cycle import PartitionDgModelOICA

beta = float(snakemake.params.beta)
assume_param = snakemake.params.assume
assume = None if assume_param in (None, "None", "none") else assume_param
threshold = float(getattr(snakemake.params, "threshold", 0.1))
max_iter = int(getattr(snakemake.params, "max_iter", 10_000))
random_state = int(getattr(snakemake.params, "random_state", 0))

data = np.load(snakemake.input.data, allow_pickle=True)

obs = data["obs"]
targets = data["targets"]

# Load soft interventional data (same datasets used by the IVN model)
ivn_data = {}
for idx, target in enumerate(targets):
    key = str(idx)
    if key in data:
        tgt = set(np.atleast_1d(target).astype(int))
        ivn_data[key] = (data[key], tgt, "soft")

model = PartitionDgModelOICA()
start = time.perf_counter()
model.fit(
    obs,
    ivn_data=ivn_data if ivn_data else None,
    beta=beta,
    assume=assume,
    threshold=threshold,
    max_iter=max_iter,
    random_state=random_state,
)
model.fit_runtime_sec = time.perf_counter() - start

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
