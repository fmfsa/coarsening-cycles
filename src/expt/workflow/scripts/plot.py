"""Main result plots: OICA SCC recovery vs sample size and density."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=2.3)

df = pd.read_csv(snakemake.input[0])

# Only graphs with at least one non-trivial SCC
cyclic = df[df.has_cycles == 1]

# ARI vs sample size (hue = density)
plt.figure()
sns.lineplot(
    data=cyclic, x="samp_size", y="ari_scc",
    hue="density", style="density", markers=True,
    estimator="median", errorbar="ci",
)
plt.xscale("log")
plt.ylim(0, 1)
plt.ylabel("ARI (SCC/Tarjan) ↑")
plt.xlabel("sample size (n)")
plt.legend(title="density")
plt.tight_layout()
plt.savefig(snakemake.output["ari"], bbox_inches="tight", pad_inches=0.02)
plt.close()

# F-score vs sample size (hue = density)
plt.figure()
sns.lineplot(
    data=cyclic, x="samp_size", y="fscore",
    hue="density", style="density", markers=True,
    estimator="median", errorbar="ci",
)
plt.xscale("log")
plt.ylim(0, 1)
plt.ylabel("F-score ↑")
plt.xlabel("sample size (n)")
plt.legend(title="density")
plt.tight_layout()
plt.savefig(snakemake.output["fscore"], bbox_inches="tight", pad_inches=0.02)
plt.close()
