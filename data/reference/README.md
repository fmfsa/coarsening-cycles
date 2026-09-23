# Historical Fig. 5 results

`sample_complexity.csv` is the user-supplied saved result, copied without changes
from the `coarsening_cycles` results directory on 2026-09-23.

SHA-256: `bc2019a892b1d8f45b45ef56c3d179ef9342d5fb65878fd45fe07febd1635d2b`.

It contains 3,000 rows: 300 distinct seeds at each n in
100, 200, 300, 400, 500, 700, 1000, 2000, 5000, 10000.
Every row reports beta_min = 0.5138680255555728.
The historical first-stable run's raw environment/logs are not supplied;
the saved numerical results, rather than a new simulation, are the provenance
for the regenerated figure. No Hungarian results are present in this CSV.

The existing plotting rule reads this file directly. Fresh simulation is an
explicit separate target (`results/sample_complexity_rerun.csv`) and does not
overwrite the historical result. Do not replace this reference with new runs.

`synth_results.csv.gz` and `synth_ablation.csv.gz` preserve saved main-grid
and ablation tables found in the earlier paper-review worktree. They contain
1,800 and 18,360 rows, respectively; gzip compression is lossless. Uncompressed
checksums are in `synthetic_provenance.json`. The ablation replot preserves
the historical figure's first-stable, Hungarian, and random-matching branches.
The main-grid saved table stops at n=5,000 and does not substantiate the
paper's n=100,000 endpoint; the dedicated paper target covers that extension.
