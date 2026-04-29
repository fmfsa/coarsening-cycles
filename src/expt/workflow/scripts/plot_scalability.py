"""Scalability plots: OICA performance vs number of nodes and SCC structure.

Produces three figures:
  1. ARI vs sample size, hue = num_nodes  (how does problem size affect recovery?)
  2. ARI vs num_nodes at largest sample size  (asymptotic scaling)
  3. ARI vs fraction of nodes in non-trivial SCCs  (does more cyclic = harder?)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.8)

df = pd.read_csv(snakemake.input[0])
cyclic = df[df.has_cycles == 1]
max_samp = cyclic["samp_size"].max()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Panel 1: ARI vs sample size, hue = num_nodes ──────────────────────────
ax = axes[0]
sns.lineplot(
    data=cyclic, x="samp_size", y="ari_scc",
    hue="num_nodes", style="num_nodes", markers=True,
    estimator="median", errorbar="ci", ax=ax,
)
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel("sample size (n)")
ax.set_ylabel("ARI (SCC/Tarjan) ↑")
ax.set_title("Scaling with # nodes")
ax.legend(title="# nodes")

# ── Panel 2: ARI vs num_nodes at largest n ────────────────────────────────
ax = axes[1]
large_n = cyclic[cyclic.samp_size == max_samp]
sns.boxplot(data=large_n, x="num_nodes", y="ari_scc", ax=ax)
ax.set_ylim(0, 1)
ax.set_xlabel("number of nodes")
ax.set_ylabel("ARI (SCC/Tarjan) ↑")
ax.set_title(f"Asymptotic ARI  (n={max_samp:,} samples)")

# ── Panel 3: ARI vs fraction in non-trivial SCCs ──────────────────────────
ax = axes[2]
# Bin frac_nontrivial and plot median ARI per bin
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
cyclic = cyclic.copy()
cyclic["frac_bin"] = pd.cut(cyclic["frac_nontrivial"], bins=bins, labels=labels)

# Use only large sample sizes for cleaner signal
large_n_all = cyclic[cyclic.samp_size >= cyclic.samp_size.quantile(0.75)]
sns.boxplot(data=large_n_all, x="frac_bin", y="ari_scc", ax=ax)
ax.set_ylim(0, 1)
ax.set_xlabel("fraction of nodes in non-trivial SCCs")
ax.set_ylabel("ARI (SCC/Tarjan) ↑")
ax.set_title("Dependence on cyclic structure")
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(snakemake.output[0], bbox_inches="tight", pad_inches=0.02)
plt.close()
