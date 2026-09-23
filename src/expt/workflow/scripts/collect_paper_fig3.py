"""Reuse the saved low-n first-stable results and append only missing cells."""
from itertools import product
import pandas as pd

saved = pd.read_csv(snakemake.input.saved)
saved = saved[(saved.method == "lacerda") & saved.samp_size.isin([50, 100, 500, 1000, 5000])].copy()
saved["source_kind"] = "historical_saved"
fresh = pd.concat([pd.read_csv(p) for p in snakemake.input.fresh], ignore_index=True)
fresh["source_kind"] = "new_run"
df = pd.concat([saved, fresh], ignore_index=True)
keys = ["regime", "num_cycles", "density", "samp_size", "seed"]
expected = set(product(["hard", "unstable"], [3, 4, 5], [.3, .5, .8],
                       [50, 100, 500, 1000, 5000, 10000, 50000, 100000], range(10)))
actual = set(df[keys].itertuples(index=False, name=None))
if df.duplicated(keys).any() or actual != expected or not (df.method == "lacerda").all() or not (df.d == 10).all():
    raise ValueError("Fig. 3 must have exactly 1,440 unique first-stable cells in the declared grid")
df.sort_values(keys).to_csv(snakemake.output[0], index=False)
