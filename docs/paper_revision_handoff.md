# P3.8–9 / P4.10 paper handoff

Code and artifacts only. The May preprint is a reference; the current ICLR
manuscript is not in this checkout and has not been edited. No claims below
imply changes to the frozen NeurIPS source or its shared figure directory.

## Fig. 5: saved results, no new simulation

The supplied CSV is preserved in `data/reference/sample_complexity.csv`.
It contains 300 distinct seeds for each of 10 sample sizes, totaling 3,000
rows. Its SHA-256 is recorded beside it. The regenerated figure keeps the
original curves, confidence intervals and slope-fitting convention; only the
reference label changes from Proposition 4 to Proposition 3. Zero empirical
Hamming errors are masked on the logarithmic axis rather than clipped to an
arbitrary positive floor; the saved observations are unchanged.

Verified from the saved data:

- beta_min = 0.5138680255555728.
- Historical Hamming slope = -4.00568268683941 (plotted as -4.01).
- Log-log failure-probability slope over n=200–1000 = -2.9056195350277556.
- At each tested n=2000, 5000, 10000, all 300 rows have exact support recovery.
- Exact condensation recovery is already 300/300 at n=1000; it is distinct
  from the support recovery probability shown in the historical figure.

Suggested caption correction: “The log-log slope of the failure probability
1 − Pr[supp(B̂_n) = supp(B)] over n ∈ [200, 1000] is −2.9.”

Suggested setup correction: replace the claimed sweep [10², 10⁵] with
[10², 10⁴], and list the actual sample sizes if space permits. The grid is
denser near the transition, rather than strictly log-spaced.

Suggested recovery wording: “All 300 recorded runs recover the exact support
at each tested sample size n ≥ 2000.” This is empirical, not a guarantee that
the population failure probability is zero. The CSV verifies the numerical
claims but does not reconstruct the original runtime environment or missing
logs. No Hungarian experiment has been added to Fig. 5.

The fixed population graph was checked during planning: 32 admissible
candidates, one stable candidate equal to the generating graph. This check
is specific to this model, not a general uniqueness claim for overlapping
cycles. The theoretical P1 issues remain outside this implementation.

## Fig. 3 and ablation labels

The plotting scripts now use mathematical F₁ and “variable-level graph”.
A separate reproduction target extends the main grid to n=100,000 without
expanding ablations. Per-fit outputs record failed fits and enumeration
limits, so a regenerated figure need not be mistaken for unlimited enumeration.
The paper target combines 900 saved first-stable cells through n=5,000 with
540 new cells at n=10,000, 50,000, 100,000. The generator and estimator in the
source worktree match the current code; the fitting wrapper differs only in
new diagnostic fields and explicitly forwarding the unchanged seed 0.
`source_kind` distinguishes the two sources in the combined CSV. Original
environment details and convergence logs for saved cells are unavailable;
do not combine their runtime values with the new benchmark's timings.

All 540 new fits completed without recorded fit failures, enumeration caps,
or enumeration timeouts. At n=10,000, exact condensation recovery occurred
in 89/90 stable and 86/90 unstable cells; at n=50,000 and 100,000, it occurred
in 90/90 cells in each regime. The median partition ARI and cluster-DAG F₁
are both 1 from n=5,000 onward in both regimes. Variable-level F₁ has median
1 in the stable regime, but remains about 0.77–0.78 in the unstable regime
over these large sample sizes. These are aggregate summaries over the
κ/density/seed grid, not guarantees for every cell. The new execution log
contains no convergence warnings; the environment is pinned in
`experiments-requirements.lock`.

The ablation replot uses its complete saved CSV and preserves the historical
figure's three branches (first-stable, Hungarian, random-matching). It does
not change which branches are reported in the ICLR manuscript's separate
ablation table.

The additional `results/ablation_accuracy_runtime.pdf` target presents the
same three selection branches in the Fig. 6 format: partition ARI,
cluster-DAG F₁, and candidate-selection time against sample size, with stable/unstable
rows. It uses saved `selection_runtime_sec` (excluding ICA, including enumeration
for first-stable); five
random-matching draws are averaged within each dataset before aggregation.
The companion summary CSV retains ICA and total fit time separately.
This is a replot of historical measurements, not a new controlled timing
experiment. The original accuracy-only ablation figure is preserved.

The runtime comparison uses Fig. 3's deep-wine, rose, and pale-rose colors
for ours, first-stable, and random-matching, respectively. "Ours" is a shorter
display label for the same Hungarian estimator; no method or measurement
was removed. First-stable is slower in all 1,080 paired saved cells, which
have identical ICA timings. At n=1,000, pooling regimes and graph settings,
median selection times are 9.367 ms (first-stable) versus 0.049 ms (ours),
while median total fit times are 1.643 s versus 1.605 s. The large shared
ICA cost explains the small difference in total time. Enumeration counts
fall from a median of 10,000 candidates at n=50/100 to 32 at n=5,000/10,000.
The 10,000-candidate cap is reached in all 180 cells at each of n=50 and
100, in 58/180 at n=500, and 5/180 at n=1,000; no cap is hit at n=5,000 or
10,000. Interpret these as capped-enumeration timings, not unrestricted
worst-case search costs. Missing historical timeout fields are not evidence
that all of those runs finished without hitting a time budget.

The selection-time panels use logarithmic y-axes (now explicitly labeled
"log scale"). Both ours and random-matching solve the same linear assignment
problem; random-matching additionally initializes a random generator and
builds random costs on each call. That overhead is a plausible contributor
to its larger saved microsecond-level time, not an established asymptotic
speed disadvantage. For finite-sample accuracy, ours minimizes the sum of
negative log magnitudes of matched W entries; random-matching ignores these
magnitudes in its objective. Because subsequent normalization divides by
matched entries, selection can affect error amplification and thresholded
edges. This is a mechanism consistent with the observed gap, not a causal
conclusion established by the ablation alone.

Server preparation is documented in `docs/group_lingam_server.md`, with an
exact 33-file staging list in `docs/github_push_files.txt`. The full target
now covers n=100, 500, 1000, 2000, 5000, 10000 with ten seeds in both regimes
(120 datasets, 240 fits), using explicit per-size limits from
`config/group_lingam_full.yaml`. A clean-copy dry run schedules all 120
generations and 240 fits; all 65 tests pass. Linux dependency installation
has not been tested here. No full run, commit, or push was performed.

Both the main Fig. 3 recipe and the historical Fig. 5 experiment use
first-stable enumeration. The new benchmark below uses Algorithm 1's
Hungarian path; label these procedures explicitly when discussing runtime.

## GroupLiNGAM pilot protocol

Compare `lingam.GroupLiNGAM(alpha=0.01)` in lingam 1.13.0 with our Hungarian
estimator at d=10, κ=4, density 0.5, Laplace noise, both stable and unstable
regimes. Use n=100,500,1000 and seeds 0–2, with identical data for both
methods. The pilot has 36 fits and a 120-second fit limit. No subsampling,
true groups, oracle tuning, or replacement of unsuccessful fits is allowed.

The package uses pairwise HSIC tests combined by Fisher's method and native
adaptive-lasso edge estimation. This is the named package implementation,
not a reproduction of the original 2010 mutual-information-based numerical
experiments. Source: https://lingam.readthedocs.io/en/latest/_modules/lingam/group_lingam.html
and https://arxiv.org/abs/1006.5041.

Measure fitting wall time and CPU time separately from worker setup/data-loading
and evaluation time; lazy initialization inside the method call is included.
Run one timed fit at a time with one numerical thread.
Record timeouts as censored timings, not ordinary 120-second completed fits.
Peak RSS is sampled every 50 ms, includes imported libraries, and is not
isolated algorithm workspace.

Use native output groups and preserve unknown within-group coefficients.
Report ARI, known-edge projection onto the true partition (explicitly an
oracle diagnostic), and exact recovery of variable memberships plus the
condensation's edges. Do not score GroupLiNGAM on within-group micro edges.
Report attempted/completed/timed-out counts with every accuracy summary.
The all-attempt success fraction is operational exact recovery **within the
fit budget**, not uncensored statistical accuracy. Scikit-learn can return an
ICA estimate with a convergence warning; those returned estimates are retained
and scored, with the warnings preserved in the pilot execution log.

The Appendix C.1 format is appropriate, but its d=20 disjoint cycles and
skewed noise were tailored to the Drton baseline. This comparison stays in
the main cyclic model class. It tests finite-sample behavior and measured
cost at d=10, not dimensional complexity or universal method superiority.

The generated pilot `report.md` and `summary.csv` supply the measured results.
Three seeds are exploratory; use the separately documented 10-seed full
target for publication-sized evidence after reviewing the pilot.

## Observed pilot results (2026-09-23)

All 36 scheduled attempts are retained and dataset hashes match within every
method pair. Ours returned models for 18/18 attempts; GroupLiNGAM completed
12/18 and hit the 120-second limit on all six n=1000 attempts. Its n=1000
accuracy is therefore unknown, not zero.

At n=500 in the stable regime, GroupLiNGAM's partition ARIs were
1, 1, 0.2574 (median 1), versus 0, 0.0476, 0 (median 0) for ours. Exact
condensation recovery was 1/3 versus 0/3. Median fitting wall times were
71.219 s versus 0.113 s; median fitting CPU times were 71.208 s versus
0.071 s. At n=500 in the unstable regime, median ARI was 0.2574 versus 0,
and neither method exactly recovered the condensation in any of the three
seeds. The full table is in the generated pilot report.

Four Hungarian fits emitted FastICA convergence warnings, all at n=100
(stable seeds 1,2; unstable seeds 1,2). Returned estimates were scored rather
than discarded. The execution log and `convergence_warnings.json` retain
the affected run identifiers. No inference about convergence is made from
the archived Fig. 3 cells, whose original logs are unavailable.

Paper implication: this is an accuracy–computation tradeoff, not evidence
that ours dominates GroupLiNGAM. The baseline can recover partitions better
at small n, while our fitting cost is much lower in this pilot. Three seeds
and censored high-n fits do not support a definitive ranking. Before a
publication comparison, prioritize completing a modest n=1000 paired study
with more seeds and a larger declared time budget; do not automatically
launch the much more expensive n=10,000 full target. The full target is
implemented but was not run.

## Follow-up design after pilot review

The pilot PDF is diagnostic, not the proposed paper figure. Its three seeds,
short n-grid, and all-baseline timeouts at n=1000 cannot show the larger-sample
comparison. At n=100, GroupLiNGAM returns ten singleton groups in all six
cells, whereas Hungarian returns one group containing all ten variables.
Both have ARI zero for opposite partition errors. At n=500 the stable
GroupLiNGAM partition is exactly correct in two of three seeds. These
observations motivate extending the sample-size sweep, not changing metrics
or hyperparameters to favor either method.

Use Fig. 6's presentation: columns for partition ARI, cluster-DAG F₁
(explicitly the true-partition projection), and fitting wall time on a log
axis; rows for stable and unstable regimes. Use consistent method colors,
10 paired seeds per sample size, and median curves with uncertainty across
seeds. Put exact recovery, completion counts, timeouts, and convergence
warnings in a companion table. Missing accuracy stays missing; runtime
limits receive lower-bound markers, never zero accuracy points.

The desired shared grid is n=100, 500, 1000, 2000, 5000, 10000, at the
existing d=10, κ=4, density=0.5 and Laplace-noise setup. First measure one
declared seed per regime at n=1000 and 2000 with a sufficient budget, then
assess n=5000 before expanding to ten seeds and n=10000. This is a feasibility
stage, not selection of favorable seeds; the final grid and budgets should
be fixed from runtime/memory measurements before the larger accuracy sweep.
The first feasibility stage was started and then deferred at the user's request;
see the status below. The ten-seed expanded sweep has not been launched.

The unmodified GroupLiNGAM implementation evaluates all nonempty proper
subsets (1022 at d=10) and repeatedly forms n-by-n HSIC kernel matrices.
Thus larger n is expensive even at fixed d. Do not assume the 10^5 endpoint
of the Drton comparison is feasible, silently subsample GroupLiNGAM, or
present an approximate independence test as the original implementation.
Preserve original pilot results if rerunning timed-out cells with a larger
budget, and reuse completed compatible fits.

Placement: the candidate-selection plot supports the ICLR revision's
Appendix C.3 ablation discussion (`tab:ablation`, per the supplied AGENTS.md).
It is an optional companion plot, not a replacement for Fig. 3 or Fig. 6,
and its random-matching branch is not automatically interchangeable with
branches in the manuscript table. The GroupLiNGAM figure belongs in a
dedicated appendix comparison with a pointer from the related-work discussion.
Fig. 6 supplies its presentation format, not its specialized d=20 disjoint-cycle
generator: this comparison uses d=10 and Hungarian. The current code also uses
Hungarian in its strict disjoint-cycle comparison; Fig. 3 uses first-stable.

### Feasibility status and layout preview

The user deferred further computation until later on 2026-09-23 and requested
a visual preview. The feasibility workflow is stopped. Stable seed 0 at n=1000
completed with GroupLiNGAM in 315.969 s (CPU 314.446 s; sampled peak RSS
320.1 MiB), with ARI=1, projected F₁=1, and exact condensation recovery.
The unstable n=1000 GroupLiNGAM attempt was interrupted on request and is
unscored, not an algorithm failure or a completed timeout. No n=2000 or n=5000
feasibility fit started. Completed JSONs, the execution log, and `status.json`
are preserved under `results/group_lingam/feasibility/`.

`output/pdf/group_lingam_layout_preview.pdf` uses the three-panel layout with
a shaded, unmeasured region for n=2000, 5000, 10000. Its plotted data combine
the original three-seed pilot with the one completed longer-budget baseline
fit, replacing only that cell's earlier timeout. Other timeout markers retain
their actual 120-second bounds. This preview mixes budgets and is explicitly
not the final ten-seed comparison. The original pilot files remain unchanged.
The companion table in `results/group_lingam/preview/sample_sizes_table.md`
records completion counts and the mixed budgets.

The subsequently requested current-results figure is
`output/pdf/group_lingam_current.pdf`. It uses only n=100, 500, 1000 and
matches the rendered Fig. 6's shared rose color (solid ours, dashed baseline), typography, and axis labels; it has
no planned region. Its snapshot data, companion table, and caption are in
`results/group_lingam/current/`. It uses the same mixed-budget observations
described above. No new fits were run for this styling update.
