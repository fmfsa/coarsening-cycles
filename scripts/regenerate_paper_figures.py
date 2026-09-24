"""Replot selected paper figures from saved data; never launches experiments."""
import argparse
from pathlib import Path
import runpy
from types import SimpleNamespace
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--figures", nargs="+", choices=["3", "4", "5", "7", "8"], default=["3", "5"])
parser.add_argument("--fig3-data", type=Path, default=ROOT / "data/reference/paper_fig3.csv.gz")
parser.add_argument("--fig5-data", type=Path, default=ROOT / "src/expt/workflow/results/sample_complexity.csv")
parser.add_argument("--fig4-data", type=Path, default=ROOT / "src/expt/workflow/results/synth_scalability.csv")
parser.add_argument("--fig7-data", type=Path, default=ROOT / "src/expt/workflow/results/synth_disjoint.csv")
parser.add_argument("--fig8-data", type=Path, default=ROOT / "data/reference/synth_threshold.csv.gz")
args = parser.parse_args()
out = ROOT / "output/pdf"
out.mkdir(parents=True, exist_ok=True)
targets = {
    "3": (args.fig3_data, "plot_synth_combined.py", "fig3_main.pdf"),
    "4": (args.fig4_data, "plot_scalability.py", "scalability_disjointcycles.pdf"),
    "5": (args.fig5_data, "plot_sample_complexity.py", "fig5_sample_complexity.pdf"),
    "7": (args.fig7_data, "plot_disjoint.py", "disjoint_micro.pdf"),
    "8": (args.fig8_data, "plot_threshold.py", "synth_threshold.pdf"),
}
for figure in args.figures:
    source, _, _ = targets[figure]
    if not source.is_file():
        raise FileNotFoundError(f"Saved data missing: {source}. Supply --fig{figure}-data PATH.")
for figure in args.figures:
    source, script, name = targets[figure]
    target = out / name
    runpy.run_path(str(ROOT / "src/expt/workflow/scripts" / script),
                   init_globals={"snakemake": SimpleNamespace(input=[str(source)], output=[str(target)])})
    print(target)
