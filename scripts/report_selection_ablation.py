"""Render a paper table from a completed selection benchmark; no model fitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"enum_first_stable": "enum. + first-stable", "enum_uniform_random": "enum. + uniform random",
          "ours": "ours", "arbitrary_control": "arbitrary control"}


def build_report(source, output):
    summary = pd.read_csv(source / "summary.csv")
    cfg = json.loads((source / "protocol.json").read_text())["protocol"]["config"]
    expected = {(r, n, m) for r in cfg["regimes"] for n in cfg["sample_sizes"] for m in LABELS}
    actual = set(zip(summary.regime, summary.n, summary.method))
    if actual != expected or not (summary.seeds == cfg["seeds"]).all():
        raise ValueError("The table requires the complete configured grid; finish/resume the benchmark first.")
    sizes = sorted(summary.n.unique())
    measures = [("ari", "ARI"), ("exact_pct", "Exact (%)"), ("ica_sec", "ICA time (s)"),
                ("selection_sec", "Selection time")]
    headers = ["Regime", "Selection"] + [f"n={n:,}" for _, _ in measures for n in sizes]
    rows, bold = [], []
    for regime in cfg["regimes"]:
        for method, label in LABELS.items():
            row = ["Stable" if regime == "hard" else "Unstable", label]
            for metric, _ in measures:
                for n in sizes:
                    cell = summary[(summary.regime == regime) & (summary.method == method) & (summary.n == n)].iloc[0]
                    value = cell[metric]
                    text = (f"{value:.3f}" if metric == "ari" else f"{value:.1f}".rstrip("0").rstrip(".") if metric == "exact_pct"
                            else f"{value * 1000:.3f} ms" if metric == "selection_sec" and value < .001
                            else f"{value:.3f} s" if metric == "selection_sec" else f"{value:.3f}")
                    if metric == "selection_sec" and cell.selection_over_ica > 1:
                        bold.append((len(rows) + 1, len(row)))
                    row.append(text)
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis("off")
    widths = [.075, .19] + [.735 / (len(measures) * len(sizes))] * (len(measures) * len(sizes))
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                     cellLoc="center", loc="center", bbox=[0, .01, 1, .84])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ""
        if c <= 1:
            cell.set_text_props(ha="left")
        if r == 0:
            cell.visible_edges = "B"
        if r in [4, 8]:
            cell.visible_edges = "B"
        if (r, c) in bold:
            cell.set_text_props(weight="bold", color="#2C1E3D")
            # Text weight, rather than color alone, denotes the measured criterion.
    x = sum(widths[:2])
    for _, title in measures:
        width = sum(widths[2:2 + len(sizes)])
        ax.text(x + width / 2, .93, title, transform=ax.transAxes, ha="center", fontsize=13)
        ax.plot([x + .005, x + width - .005], [.90, .90], transform=ax.transAxes, color="black", lw=.6)
        x += width
    ax.plot([0, 1], [.99, .99], transform=ax.transAxes, color="black", lw=1)
    enum_summary = summary[summary.method == "enum_first_stable"]
    missing = summary[summary.completed < summary.seeds]
    coverage_notes = []
    for (regime, completed, seeds), group in missing.groupby(["regime", "completed", "seeds"]):
        methods = set(group.method)
        ns = sorted(group.n.unique())
        if methods == {"enum_first_stable", "enum_uniform_random"} and len(group) == 2 * len(ns):
            who = "Both enumeration rules"
        else:
            who = ", ".join(LABELS[m] for m in LABELS if m in methods)
        coverage_notes.append(
            f"{who} returned estimates for {int(completed)} of {int(seeds)} seeds "
            f"in the {'stable' if regime == 'hard' else 'unstable'} regime at n="
            + ", ".join(f"{n:,}" for n in ns) + "."
        )
    coverage = " ".join(coverage_notes)
    if coverage:
        coverage += " Accuracy is conditional on returned estimates; timing includes every attempt."
    scope = "in total across threshold rounds" if cfg.get("enumeration_budget_scope") == "total" else "per threshold round"
    enumeration_limit = (
        f"Enumeration has no candidate-count cap and a {cfg['enumeration_budget_sec']:g}-second limit {scope}."
        if cfg["max_perms"] is None else
        f"Enumeration is capped at {cfg['max_perms']:,} candidates and {cfg['enumeration_budget_sec']:g} seconds {scope}."
    )
    notes = (
        f"d={cfg['d']}, κ={cfg['num_cycles']}, density={cfg['density']}, {cfg['noise_dist']} noise; "
        f"{cfg['seeds']} seeds per regime/n; {cfg['random_draws']} random draws averaged within each dataset.\n"
        "ARI and exact recovery: means across seeds. Exact means the SCC partition AND condensation edges are correct. "
        "Stage times: medians across seeds.\n"
        "Bold: median paired selection/ICA ratio > 1. Enumeration includes stability checks; "
        "first-stable falls back to an unstable candidate when no stable candidate exists.\n"
        + enumeration_limit + " "
        + f"{int(enum_summary.timeout_count.sum())}/{int(enum_summary.seeds.sum())} cells hit the time budget. "
        "Accuracy is conditional on returned estimates; completion and per-setting truncation counts accompany this table."
        + ("\n" + coverage if coverage else "")
    )
    fig.subplots_adjust(left=.015, right=.985, top=.97, bottom=.03)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    caption = (
        rf"Candidate-selection ablation at $d = {cfg['d']}$ with $\kappa = {cfg['num_cycles']}$ non-trivial SCCs "
        rf"({cfg['seeds']} seeds per cell). "
        "Exact denotes exact recovery of both the SCC partition and the condensation. "
        "ARI and exact recovery are averaged across seeds; stage times are medians. "
        "Bold selection times indicate a median paired selection-to-ICA time ratio greater than one."
        + (" " + coverage if coverage else "")
    )
    tex = [r"\begin{table}[t]", r"  \centering", "  \\caption{\n    " + caption + "\n  }",
           r"  \label{tab:ablation}", r"  \small", r"  \setlength{\tabcolsep}{4.5pt}",
           r"  \resizebox{\linewidth}{!}{%",
           r"  \begin{tabular}{ll" + "c" * (len(headers) - 2) + "}", r"    \toprule",
           " & " + " & " + " & ".join(r"\multicolumn{" + str(len(sizes)) + "}{c}{" + title.replace("%", r"\%") + "}" for _, title in measures) + r" \\",
           " ".join(r"\cmidrule(lr){" + str(3 + i * len(sizes)) + "-" + str(2 + (i + 1) * len(sizes)) + "}" for i in range(len(measures))),
           " & ".join(headers[:2] + [{10000: r"$n{=}10^4$", 50000: r"$n{=}5{\cdot}10^4$"}.get(n, rf"$n{{=}}{n}$") for _ in measures for n in sizes]) + r" \\", r"\midrule"]
    for i, row in enumerate(rows, 1):
        values = [v.replace("enum. ", r"enum.\ ").replace(" ms", r"\,ms").replace(" s", r"\,s") if j != 1 else v.replace("enum. ", r"enum.\ ") for j, v in enumerate(row)]
        tex.append(" & ".join(r"\textbf{" + v + "}" if (i, j) in bold else v for j, v in enumerate(values)) + r" \\")
        if i == 4:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"  \end{tabular}", "  }", r"\end{table}"]
    output.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    diagnostics = ["# Selection-stage benchmark", "", notes.replace("\n", "\n\n"), "",
                   enumeration_limit + " Truncated candidate sets are not the full equivalence class.", "",
                   "| Regime | n | Method | Completed | Cap hits | Budget hits | Selection > ICA | Median total (s) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        diagnostics.append(f"| {r.regime} | {r.n} | {LABELS[r.method]} | {r.completed}/{r.seeds} | {r.cap_count} | {r.timeout_count} | {r.dominance_count}/{r.seeds} | {r.total_sec:.5f} |")
    output.with_suffix(".md").write_text("\n".join(diagnostics) + "\n")
    explanation = (
        rf"We use $d={cfg['d']}$ variables, $\kappa={cfg['num_cycles']}$ non-trivial SCCs, "
        rf"density ${cfg['density']}$, and {cfg['noise_dist'].capitalize()} noise, with {cfg['seeds']} seeds per regime and sample size. "
        "All four selection rules share one fitted ICA estimate per dataset. "
        rf"For each randomized rule, {cfg['random_draws']} draws are averaged within each dataset before aggregation across seeds. "
        "ICA is timed after dependency imports; selection timing excludes scoring and file operations. "
        "Enumeration-based rules include candidate enumeration and stability checks; ours uses a single "
        "cubic-time assignment, while the arbitrary control ignores admissibility. "
        "First-stable falls back to an unstable candidate when the enumerated list contains no stable candidate.\n\n"
        + enumeration_limit + " "
        + f"{int(enum_summary.timeout_count.sum())} of {int(enum_summary.seeds.sum())} datasets reached the time budget. "
        "Uniform random selection samples the enumerated list, which need not be the full class when truncated. "
        f"For the enumeration-based rules, selection exceeded ICA time in {int(enum_summary.dominance_count.sum())} "
        f"of {int(enum_summary.seeds.sum())} datasets. These results demonstrate that enumeration can dominate "
        "the measured computation in this setting under the stated enumeration limits.\n"
        + ("\n" + coverage + "\n" if coverage else "")
    )
    output.with_name(output.stem + "_text.md").write_text(explanation)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "src/expt/workflow/results/ablation_d20")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/ablation_d20_table.pdf")
    args = parser.parse_args()
    build_report(args.input, args.output)
