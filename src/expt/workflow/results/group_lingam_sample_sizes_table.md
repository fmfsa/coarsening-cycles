# GroupLiNGAM sample-size comparison: companion table

The two methods receive identical observations per regime, n, and seed. d=10, 4 non-trivial SCCs, density 0.5, Laplace noise; one numerical thread per fit. Fits ran one at a time for n <= 2000; the n=5000 fits ran 4 at a time, so n=5000 GroupLiNGAM times are inflated by contention relative to the serial points. GroupLiNGAM 1.13.0: alpha=0.01, native edge estimation. Ours: Hungarian selection, threshold 0.1. Fit wall/CPU times exclude worker setup, data loading, and evaluation; they include lazy initialization inside the method call.

ARI and F1 summarize completed fits only. F1 projects known predicted edges onto the true partition; it is an oracle diagnostic, not end-to-end condensation accuracy. Exact recovery requires both variable memberships and condensation edges. Timeouts are unknown accuracy, and unsuccessful operational exact recovery within the declared budget. Completed-only summaries may be biased if other fits time out. Bootstrap intervals are descriptive with few seeds; no interval is estimated from a single seed.

| Regime | n | Method | Completed/attempts | Timeouts | Errors | ARI median | Oracle F1 median | Exact within budget/attempts | Wall s median | CPU s median | Budget s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stable | 100 | GroupLiNGAM | 10/10 | 0 | 0 | 0 | 0.64 | 0/10 | 6.29 | 6.29 | 600 |
| stable | 100 | ours | 10/10 | 0 | 0 | 0 | 0.639 | 0/10 | 0.284 | 0.284 | 600 |
| stable | 500 | GroupLiNGAM | 10/10 | 0 | 0 | 0.543 | 0.838 | 1/10 | 79 | 79 | 600 |
| stable | 500 | ours | 10/10 | 0 | 0 | 0 | 0.766 | 0/10 | 0.0187 | 0.0187 | 600 |
| stable | 1000 | GroupLiNGAM | 10/10 | 0 | 0 | 0.479 | 0.834 | 2/10 | 284 | 284 | 1800 |
| stable | 1000 | ours | 10/10 | 0 | 0 | 0.309 | 0.857 | 0/10 | 0.0182 | 0.0182 | 1800 |
| stable | 2000 | GroupLiNGAM | 10/10 | 0 | 0 | 0.814 | 0.974 | 3/10 | 1.64e+03 | 1.64e+03 | 7200 |
| stable | 2000 | ours | 10/10 | 0 | 0 | 1 | 1 | 6/10 | 0.0189 | 0.0189 | 7200 |
| stable | 5000 | GroupLiNGAM | 10/10 | 0 | 0 | 0.836 | 1 | 3/10 | 4.11e+04 | 4.11e+04 | 86400 |
| stable | 5000 | ours | 10/10 | 0 | 0 | 1 | 1 | 10/10 | 0.022 | 0.022 | 86400 |
| unstable | 100 | GroupLiNGAM | 10/10 | 0 | 0 | 0 | 0.578 | 0/10 | 6.35 | 6.35 | 600 |
| unstable | 100 | ours | 10/10 | 0 | 0 | 0 | 0.631 | 0/10 | 0.506 | 0.506 | 600 |
| unstable | 500 | GroupLiNGAM | 10/10 | 0 | 0 | 0.378 | 0.693 | 0/10 | 74.3 | 74.3 | 600 |
| unstable | 500 | ours | 10/10 | 0 | 0 | 0 | 0.721 | 0/10 | 0.0181 | 0.0181 | 600 |
| unstable | 1000 | GroupLiNGAM | 10/10 | 0 | 0 | 0.458 | 0.774 | 2/10 | 286 | 286 | 1800 |
| unstable | 1000 | ours | 10/10 | 0 | 0 | 0.129 | 0.795 | 0/10 | 0.0184 | 0.0184 | 1800 |
| unstable | 2000 | GroupLiNGAM | 10/10 | 0 | 0 | 0.598 | 0.893 | 2/10 | 1.63e+03 | 1.63e+03 | 7200 |
| unstable | 2000 | ours | 10/10 | 0 | 0 | 0.608 | 0.935 | 1/10 | 0.0187 | 0.0186 | 7200 |
| unstable | 5000 | GroupLiNGAM | 10/10 | 0 | 0 | 0.691 | 0.921 | 3/10 | 4.06e+04 | 4.05e+04 | 86400 |
| unstable | 5000 | ours | 10/10 | 0 | 0 | 1 | 1 | 9/10 | 0.0216 | 0.0216 | 86400 |

Peak RSS and interval endpoints are provided in the companion CSV. This fixed-d experiment does not establish dimension scaling or a universal method ranking.
