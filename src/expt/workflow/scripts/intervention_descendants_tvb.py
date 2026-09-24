"""Validate recovered C-DAG descendant predictions against TVB stimulation.

For seed 0, the pipeline already integrated one soft intervention per true
SCC with TVB (validate_tvb.py). This script closes the loop: for every
fitted model (one per sample size) and every intervention, it compares

  * the variables that ACTUALLY responded in the TVB steady state
    (|x_tvb| above `resp_threshold` — descendant responses are O(0.1+),
    non-descendants are zero up to the 1e-7 integration residual), against
  * the variables the recovered C-DAG PREDICTS to respond: every estimated
    cluster intersecting the stimulated SCC plus everything reachable from
    those clusters in `model.dag`.

The result is a per-(sample size, intervention) precision/recall/F1 of the
C-DAG's causal-response predictions — an interventional validation of the
recovered abstraction, not just of the numerics. TVB itself is not
imported; the TVB responses are read from the validation CSV.
"""

import pickle

import numpy as np
import pandas as pd

from repare_cycle.tvb_bridge import predicted_descendant_variables

resp_threshold = float(snakemake.params.resp_threshold)
samp_sizes = [int(n) for n in snakemake.params.samp_sizes]

data = np.load(snakemake.input.data, allow_pickle=True)
scc_labels = data["scc_labels"]
intervention_scc_labels = data["intervention_scc_labels"]
intervention_scc_sizes = data["intervention_scc_sizes"]

validation = pd.read_csv(snakemake.input.validation)
ivn = validation[validation["episode_type"] == "intervention"]

rows = []
for samp_size, model_path in zip(samp_sizes, snakemake.input.models):
    model = pickle.load(open(model_path, "rb"))
    for j, scc_label in enumerate(intervention_scc_labels):
        stimulated = frozenset(np.where(scc_labels == scc_label)[0].tolist())
        episode = ivn[ivn["episode_id"] == j]
        responded = frozenset(
            episode.loc[episode["x_tvb"].abs() > resp_threshold, "coord"].tolist()
        )
        predicted = predicted_descendant_variables(model.dag, stimulated)

        tp = len(predicted & responded)
        n_pred = len(predicted)
        n_resp = len(responded)
        precision = tp / n_pred if n_pred else 1.0
        recall = tp / n_resp if n_resp else 1.0
        fscore = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        rows.append({
            "seed": 0,
            "samp_size": samp_size,
            "scc_label": int(scc_label),
            "scc_size": int(intervention_scc_sizes[j]),
            "n_responders_tvb": n_resp,
            "n_predicted": n_pred,
            "ivn_desc_precision": precision,
            "ivn_desc_recall": recall,
            "ivn_desc_fscore": fscore,
            "resp_threshold": resp_threshold,
        })

pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
