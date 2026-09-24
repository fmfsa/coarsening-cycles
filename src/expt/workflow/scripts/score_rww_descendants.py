"""Score recovered C-DAG descendants against Wong-Wang stimulations.

TVB-free. For every amplitude level, sample size, true SCC, and
stimulation sign: compare the variables the recovered C-DAG (`model.dag`,
fitted on that amplitude's observational data only) predicts to respond
against the variables whose held-out nonlinear TVB response exceeds
`resp_threshold`. Held-out perturbation validation of C-DAG descendants —
the stimulations are never seen by the fitting algorithm.
"""

import pickle

import numpy as np
import pandas as pd

from repare_cycle.tvb_bridge import predicted_descendant_variables

resp_threshold = float(snakemake.params.resp_threshold)
amps = [str(a) for a in snakemake.params.amps]
samp_sizes = [int(n) for n in snakemake.params.samp_sizes]

data = np.load(snakemake.input.data, allow_pickle=True)
scc_labels = data["scc_labels"]

stimulation = pd.concat(
    [pd.read_csv(p) for p in snakemake.input.stimulations], ignore_index=True
)

# Models arrive amp-major: [amp0×n0, amp0×n1, ..., amp1×n0, ...]
model_paths = iter(snakemake.input.models)
models = {(a, n): pickle.load(open(next(model_paths), "rb"))
          for a in amps for n in samp_sizes}

rows = []
for amp in amps:
    amp_stims = stimulation[stimulation["amp"] == amp]
    for (label, sign), episode in amp_stims.groupby(["scc_label", "sign"]):
        stimulated = frozenset(np.flatnonzero(scc_labels == label).tolist())
        responded = frozenset(
            episode.loc[episode["response"].abs() > resp_threshold,
                        "coord"].tolist()
        )
        for n in samp_sizes:
            predicted = predicted_descendant_variables(
                models[(amp, n)].dag, stimulated
            )
            tp = len(predicted & responded)
            n_pred, n_resp = len(predicted), len(responded)
            precision = tp / n_pred if n_pred else 1.0
            recall = tp / n_resp if n_resp else 1.0
            fscore = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            rows.append({
                "seed": 0,
                "amp": amp,
                "samp_size": n,
                "scc_label": int(label),
                "scc_size": len(stimulated),
                "sign": int(sign),
                "n_responders_tvb": n_resp,
                "n_predicted": n_pred,
                "ivn_desc_precision": precision,
                "ivn_desc_recall": recall,
                "ivn_desc_fscore": fscore,
                "resp_threshold": resp_threshold,
            })

pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
