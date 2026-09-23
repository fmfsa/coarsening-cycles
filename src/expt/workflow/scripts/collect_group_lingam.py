"""Collect every scheduled attempt and verify method pairing."""
import json
import pandas as pd

rows = [json.loads(open(path).read()) for path in snakemake.input]
df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows])
keys = ["regime", "samp_size", "seed"]
if df.duplicated(keys+["method"]).any():
    raise ValueError("Duplicate benchmark attempt")
for _, pair in df.groupby(keys):
    if set(pair.method) != {"hungarian", "group_lingam"} or pair.dataset_sha256.nunique() != 1:
        raise ValueError("Unpaired benchmark datasets")
df.sort_values(keys+["method"]).to_csv(snakemake.output[0], index=False)
