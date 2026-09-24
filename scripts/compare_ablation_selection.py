"""Paired, dataset-level bootstrap contrasts for the selection ablation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(source):
    cells = pd.read_csv(source / "cells.csv")
    rng = np.random.default_rng(0)
    records = []
    for (regime, n), group in cells.groupby(["regime", "n"]):
        first = group[group.method == "enum_first_stable"].set_index("seed")
        for left, right in [("enum_uniform_random", "enum_first_stable"),
                            ("ours", "enum_first_stable"), ("enum_uniform_random", "ours")]:
            a = group[group.method == left].set_index("seed")
            b = group[group.method == right].set_index("seed")
            for subset in ["all", "complete_enumeration"]:
                seeds = a.index.intersection(b.index)
                if subset == "complete_enumeration":
                    seeds = seeds.intersection(first.index[~(first.cap_hit | first.timed_out)])
                for metric, scale in [("ari", 1), ("exact", 100)]:
                    paired = pd.concat([a.loc[seeds, metric], b.loc[seeds, metric]], axis=1).dropna()
                    difference = scale * (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
                    if len(difference):
                        boot = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
                        lower, upper = np.quantile(boot, [.025, .975])
                        mean = difference.mean()
                    else:
                        mean = lower = upper = float("nan")
                    records.append(dict(regime=regime, n=n, left=left, right=right, subset=subset,
                                        metric="exact_percentage_points" if metric == "exact" else "ari",
                                        paired_seeds=len(difference), mean_difference=mean,
                                        ci95_low=lower, ci95_high=upper))
    result = pd.DataFrame(records)
    result.to_csv(source / "paired_contrasts.csv", index=False)
    (source / "paired_contrasts_notes.md").write_text(
        "Differences are left minus right. Exact-recovery differences are percentage points. "
        "Five random draws are already averaged within each dataset in cells.csv. "
        "Percentile 95% bootstrap intervals resample paired dataset/seed differences within "
        "each regime and sample size (10,000 resamples; RNG seed 0). The complete-enumeration subset "
        "excludes datasets whose enumeration hit a candidate cap or time budget. "
        "Intervals are descriptive, without multiple-comparison correction; an interval "
        "containing zero does not establish equivalence.\n")
    print(result[(result.left == "enum_uniform_random") &
                 (result.right == "enum_first_stable") & (result.subset == "all")].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    compare(parser.parse_args().input)
