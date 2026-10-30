# Acceptance criteria

Sections D, C, V, H and E were committed in the repository's first commit, together with the project skeleton.
Sections N, W, P, B and T were added during the build, before the code they test. Section R was added in a
review round, after the first results. Criteria were not loosened after results were seen. Two criterion texts were
corrected; both corrections are marked in place. Every row is scored mechanically by `expkit acceptance` from
`results/*.json` and `build/*` (see "Results" at the bottom).

Conventions: α = 0.05 two-sided everywhere unless stated. "Rep" = one independent re-randomization.
"Simulation" means a Monte-Carlo procedure on top of real data (re-randomizing real units, or
injecting a known effect); it is always labeled as such in outputs.

## D. Data

| ID | Criterion |
|----|-----------|
| D1 | Criteo Uplift v2.1 loads from the Hugging Face Hub (`criteo/criteo-uplift`) with exactly 13,979,592 rows, and the treatment ratio, visit rate and conversion rate match the published key figures (0.85, 0.046992, 0.00292) to 4 decimals. |
| D2 | ASOS Digital Experiments Dataset loads from OSF (`osf.io/64jsb`) with 24,153 rows, 78 experiments and 99 treatments (experiment x variant), as stated in its datasheet. |
| D3 | Hillstrom MineThatData email RCT loads with 64,000 rows. |
| D4 | Total data on disk for this repo (raw + intermediates) is <= 3.5 GB. |

## C. Correctness of each estimator (external referee = scipy / statsmodels / published tables)

| ID | Criterion |
|----|-----------|
| C1 | Welch t-test computed from SQL sufficient statistics (n, Σy, Σy²) matches `scipy.stats.ttest_ind(equal_var=False)` on real Criteo units: |Δ estimate|, |Δ SE|, |Δ p| all <= 1e-6 (relative for estimate/SE). |
| C2 | Delta-method ratio metric SE matches `statsmodels` OLS with cluster-robust (by randomization unit) SE on the expanded event-level data to <= 1e-6 relative (after the documented small-sample factor); and reduces exactly to Welch when every denominator = 1. |
| C3 | CUPED (pooled θ, Deng et al. 2013) from sufficient statistics equals Welch on the explicitly adjusted unit values `y - θ'(x - x̄)` with θ from `statsmodels` OLS, to <= 1e-6. Its point estimate agrees with ANCOVA `y ~ T + X` (statsmodels, HC1) to within 1% relative on real Criteo data (asymptotic equivalence, not algebraic identity). |
| C4 | SRM chi-square statistic and p-value match `scipy.stats.chisquare` to <= 1e-9. |
| C5 | Benjamini–Hochberg adjusted p-values and reject flags match `statsmodels.stats.multitest.multipletests(method="fdr_bh")` exactly (<= 1e-12). |
| C6 | Lan–DeMets O'Brien–Fleming boundaries for K=5 equally spaced looks, two-sided α=0.05: the cumulative crossing probabilities of our boundaries, computed by an **independent** implementation (`gsdesign` 0.1.5, Python port of the R gsDesign2 recursive-integration code), equal the OBF spending function α(t_k) to <= 1e-6 absolute; the classical Jennison & Turnbull (2000, Table 2.3) constants for K=5 (Pocock 2.413, O'Brien–Fleming 2.040) give crossing probability 0.050 ± 0.0005 in our integrator; Monte-Carlo type-I error on 200,000 simulated Brownian paths is in [4.7%, 5.3%]. |
|    | *Correction note (2026-10-29, during build):* the original C6 listed "published" K=5 boundaries 4.877/3.357/2.680/2.290/2.031 without a source. They could not be verified and are inconsistent with the spending function itself (look-1 boundary must equal z₀.₉₇₅/√0.2 = 4.383 by definition). The referee was replaced by the verifiable one above before the result was recorded. |
| C7 | mSPRT statistic equals the Gaussian-mixture likelihood ratio computed by numerical integration to <= 1e-6 relative; under a simulated i.i.d. Gaussian null monitored after every batch (≥ 500 looks), the always-valid FPR is <= 5%. |
| C8 | Bootstrap (Poisson) CI for a difference in means agrees with the Welch CI width within 5% on a large real sample, and is the engine's fallback when n is small or the metric is flagged non-normal. |
| C9 | Property-based tests (hypothesis) pass: sign flips when arms are swapped, location/scale equivariance, p-value in [0,1], CUPED with θ=0 equals Welch, BH monotonicity. |

## V. Validation on real randomized data

| ID | Criterion |
|----|-----------|
| V1 | A/A on the 2,096,937 real Criteo control units, >= 2,000 re-randomizations (50/50): false-positive rate in **[4%, 6%]** for each of: Welch t-test (visit, conversion), CUPED (visit, conversion), CUPAC (visit, conversion), delta-method ratio (conversion per visit). |
| V2 | Same A/A: p-values are uniform — Kolmogorov–Smirnov p-value > 0.01 for every method/metric in V1. |
| V3 | Bootstrap A/A: >= 500 reps on real control-unit subsamples: FPR in [3.5%, 6.5%] (wider band because fewer reps; binomial SE ≈ 1 pp). |
| V4 | Variance reduction on the real Criteo experiment (cross-fitted, treatment-blind covariates): **CUPAC >= 30% on visit, >= 15% on conversion**, and CUPAC >= CUPED on both. Report the equivalent sample-size saving (= variance-reduction %) and runtime saving at fixed traffic. |
| V5 | Power with injected effects (simulation on real control units): empirical power of the Welch test within ±3 pp of the analytic power formula at every effect size; CUPAC reaches 80% power at a smaller minimum detectable effect than the raw t-test. |
| V6 | Peeking on ASOS (simulation anchored on the 396 real ASOS cumulative trajectories: real per-snapshot sample sizes, means and variances; null imposed by drawing both arms from the same real per-period distribution): naive daily peeking FPR >= 2x nominal (>= 10%); **mSPRT FPR <= 5%**; O'Brien–Fleming (Lan–DeMets) FPR in [3%, 5.5%]. |
|    | *Correction note (review round):* this is a **model-based** check (a Brownian information-time null on the real look schedules), not a validation on real data. OBF ≈ 5% holds by construction under that model. The simulation now uses variance-based information on the relative scale. The real-trajectory evidence is in P1 and R-section evidence. |
| V7 | Peeking on real Criteo control units in random arrival order (A/A, >= 2,000 reps, >= 20 looks): naive FPR >= 10%; mSPRT FPR <= 5%; OBF FPR in [4%, 6%]. |
| V8 | Real ASOS experiments (no ground truth; descriptive): report the share of trajectories declared significant by naive peeking, fixed-horizon at the last snapshot, mSPRT, and OBF, plus the SRM rate (assuming equal allocation). No pass threshold — reported honestly. |

## H. Heterogeneous effects and targeting (Criteo real treatment)

| ID | Criterion |
|----|-----------|
| H1 | Randomization balance: a LightGBM classifier predicting treatment from f0–f11 has holdout AUC <= 0.51. |
| H2 | T-learner and X-learner (LightGBM) have a positive normalized Qini coefficient on the holdout for visit, with a 95% bootstrap CI excluding 0; and beat a random ranking. |
| H3 | The repo's own Qini curve matches `causalml.metrics.get_qini` on the same holdout to <= 1e-6. |
| H4 | Targeting answer: targeting the **top 20%** by predicted uplift retains **>= 50%** of the incremental visits (and the table reports incremental conversions for top 10/20/30/50%) with bootstrap CIs. |

## E. Engineering

| ID | Criterion |
|----|-----------|
| E1 | `expkit analyze --config experiments/criteo.yaml` writes an HTML readout containing: SRM banner, metric table with CIs, raw vs CUPED/CUPAC side by side, guardrail section, sequential-boundary chart, ship/no-ship recommendation with the rule that fired. A test parses the HTML for each section. |
| E2 | A config with a simulated SRM bug (labeled) produces a readout whose recommendation is "BLOCKED — SRM" regardless of metric results. |
| E3 | `make all` runs clean (exit 0) end to end. |
| E4 | `ruff check` and `ruff format --check` clean; `mypy src` clean. |
| E5 | `pytest` all pass; line coverage of `src/expkit/stats` >= 90%. |
| E6 | Dockerfile and `.github/workflows/ci.yml` exist (CI not run remotely by rule; Docker build only if the daemon is available). |
| E7 | `make all` finishes in < 45 minutes on a 10-core workstation. |

## Additional criteria (added 2026-10-29 during the build, before the corresponding code)

Known facts at the time of writing (from the raw-data profile only): in Criteo, control has
exposure = 0 for all 2,096,937 units and 428,212 of 11,882,655 treated units were exposed (3.6%),
i.e. one-sided non-compliance. Hillstrom `spend` is zero-inflated (~1% buyers, max $499).

### N. Non-compliance / triggered analysis (Criteo `treatment` = assignment, `exposure` = receipt)

| ID | Criterion |
|----|-----------|
| N1 | ITT and CACE/LATE (Wald = ITT_Y / ITT_D, assignment as instrument) for visit and conversion, with SEs. Our Wald/2SLS estimate and robust SE match `linearmodels` IV2SLS (robust) to <= 1e-6 relative. |
| N2 | Covariate-adjusted 2SLS (f0–f11 as exogenous controls) reported; its SE is smaller than the unadjusted Wald SE. |
| N3 | Exclusion restriction / one-sided non-compliance discussed and probed with a real-data check: exposure is identically 0 in control (one-sided), and a falsification test comparing treated-never-exposed vs control units within strata of predicted exposure propensity (ghost-ads-style counterfactual trigger predicted from features) is reported with its numbers and caveats. |
| N4 | Dilution: report the required sample size for the ITT test vs a triggered (complier-only / predicted-trigger) analysis at the observed effect, using the real compliance rate and real variances; the ratio is reported as a number. Triggered-analysis estimate on predicted-trigger strata is diluted back to ITT and agrees with the full ITT within its 95% CI. |

### W. Winner's curse and empirical-Bayes shrinkage (ASOS portfolio)

| ID | Criterion |
|----|-----------|
| W1 | Fit a normal-normal EB prior (MLE with heterogeneous SEs) and a nonparametric NPMLE prior (grid EM, Koenker–Mizera / Efron g-modeling) per ASOS metric, on final relative lifts of the 99 treatments. NPMLE log-likelihood >= normal log-likelihood (it nests it). |
| W2 | EB validated by simulation (labeled): draw true effects from the fitted prior and noise from the real SEs; 95% posterior intervals cover the true effect in [93%, 97%] of draws; EB posterior-mean MSE < raw-estimate MSE. |
| W3 | Report: (a) average overstatement of "winners" (p < 0.05, positive) = observed lift vs. posterior mean, in %; (b) estimated portfolio true win rate P(θ > 0); (c) estimated FDR among declared winners (mean posterior P(θ <= 0)). Real numbers, no threshold. |
| W4 | Fat tails (Azevedo et al. 2020): report excess kurtosis of the deconvolved prior and compare the normal vs NPMLE posterior for the top winners. |

### P. Real peeking evidence on ASOS trajectories (no ground truth; uses real snapshots)

| ID | Criterion |
|----|-----------|
| P1 | On the real trajectories (metric 1..4, 99 treatments): share where naive daily peeking ever declared a significant winner/loser that the final fixed-horizon analysis does not support (not significant or opposite sign) is reported; same for mSPRT and OBF. Expected ordering: naive > OBF and naive > mSPRT. |
| P2 | Power/speed trade-off: among trajectories the final analysis calls significant, report each method's detection rate and mean/median stopping time (days) vs. the fixed horizon. |

### B. Decision economics

| ID | Criterion |
|----|-----------|
| B1 | Bayesian expected loss E[max(−θ,0)] (ship) and E[max(θ,0)] (don't ship) under a normal posterior matches Monte-Carlo (10^6 draws) to <= 1e-3 relative. |
| B2 | The scorecard shows an expected-loss panel with the threshold ("cost of being wrong") and states whether the Bayesian rule agrees with the p-value rule. On the ASOS portfolio, the count of decisions where the two rules disagree (both directions) is reported. |

### T. Heavy tails

| ID | Criterion |
|----|-----------|
| T1 | On the heaviest-tailed real metric available (Hillstrom `spend`, real RCT), report for caps at p90/p95/p99/p99.9 of buyer spend and no cap: point estimate, SE, variance reduction %, bias vs. the uncapped ATE, and MSE; plus empirical power from real-data subsamples (simulation). If capping does not help on this metric, say so. |

---

## R. Review-round criteria (added after the first results; scored the same way)

| ID | Criterion |
|----|-----------|
| R1 | A failed covariate-balance check stops the scorecard (rule R1b, HOLD) instead of switching estimators; the Criteo lift is reported as a sensitivity interval over negative-sampling corrections, next to crude and Mantel–Haenszel odds ratios. |
| R2 | A synthetic test shows that label-dependent negative sampling creates feature imbalance in a randomized experiment, that X-adjustment does not recover the true lift, and that the Mantel–Haenszel OR and re-weighting by 1/r do. |
| R3 | A/A at the real 85/15 split, 10,000 reps: FPR in [4%, 6%] for every V1 method/metric, KS p > 0.01. |
| R4 | Lan–DeMets boundaries keep MC type-I error in [4.7%, 5.3%] on a schedule with near-duplicate looks (regression test for the integrator). |
| R5 | Every rendered doc (README, memo, appendix, brief) equals its template rendered from `results/*.json` (drift test). |

## Results

### Notes on the FAIL rows

- **V4 (conversion CUPAC ≥ 15%)**: misses. The first CUPAC model reached 7.6%. It was re-tuned once with more
  regularization and still misses. Visits pass.
- **N2 (2SLS with f0–f11 has a smaller SE)**: true for visits, false for conversions. Linear raw-feature controls add
  noise to the rare-event IV. The CUPAC covariate does reduce the SE for both outcomes.
- **N4 (diluted top-k predicted-trigger estimate agrees with the full ITT)**: misses. The top-k cut drops users who
  still carry exposures, so the diluted estimate is biased low. The unbiased use, stratification over predicted-exposure
  deciles, is reported in the appendix.
- **E7 (< 45 min)**: the timing in the table is wall clock on a host shared with other workloads.
- **H1** passes its pre-set threshold (AUC ≤ 0.51), but the threshold ignored n: the same AUC is z ≈ 33. The
  imbalance is real and drives criterion R1.

<!-- results-table -->

Generated by `expkit acceptance` on 2026-10-29 16:34 — 42 PASS, 3 FAIL, 5 INFO. INFO = descriptive criterion with no pass threshold (pre-registered as such).

| ID | Status | Evidence (from `results/*.json`, produced by `make all`) |
|----|--------|----------|
| D1 | **PASS** | rows=13,979,592 treat=0.85 visit=0.046992 conv=0.002917 |
| D2 | **PASS** | rows=24,153 experiments=78 treatments=99 |
| D3 | **PASS** | rows=64,000 |
| D4 | **PASS** | data/ = 603 MB |
| C1 | **PASS** | max diff vs scipy over visit/conversion (13.98M rows) = 3.9e-16 |
| C2 | **PASS** | real data: delta SE vs statsmodels cluster-robust rel diff = 8.4e-15 (+ unit test exact reduction to Welch) |
| C3 | **PASS** | vs statsmodels-θ Welch: stat 6.6e-09, est 4.6e-09; vs ANCOVA HC1: est rel diff 0.04%, SE 0.21% |
| C4 | **PASS** | chi2/p abs diff = 0.0e+00 |
| C5 | **PASS** | max abs diff = 2.8e-17 (+ unit test incl. reject flags) |
| C6 | **PASS** | test_obf_boundaries_vs_independent_gsdesign=PASS, test_classical_constants_jennison_turnbull=PASS, test_obf_monte_carlo_type1=PASS |
| C7 | **PASS** | test_msprt_lambda_matches_numerical_integration=PASS, test_msprt_pvalues_monotone_and_valid=PASS |
| C8 | **PASS** | test_bootstrap_agrees_with_welch=PASS |
| C9 | **PASS** | test_welch_swap_flips_sign=PASS, test_welch_location_scale=PASS, test_cuped_theta_zero_equals_welch=PASS, test_bh_properties=PASS, test_msprt_properties=PASS |
| V1 | **PASS** | 10,000 reps on 2,096,937 units; FPR visit=4.80%, conversion=5.02%, visit_cuped=4.64%, conversion_cuped=4.82%, visit_cupac=4.67%, conversion_cupac=4.83%, conversion_per_visit_delta=4.84% |
| V2 | **PASS** | min KS p = 0.226 (visit_cuped) |
| V3 | **PASS** | 1000 reps × n=20,000, B=500: visit_bootstrap=4.9%, conversion_bootstrap=4.9% |
| V4 | **FAIL** | real experiment VR: visit CUPED 25.2% / CUPAC 30.8%; conversion CUPED 10.7% / CUPAC 12.9%; A/A empirical VR visit CUPAC 30.2%, conversion CUPAC 13.4% |
| V5 | **PASS** | max /empirical − analytic/ (Welch) = 1.2 pp; interpolated 80%-power MDE, visit: raw 1.97%, CUPED 1.74%, CUPAC 1.68% |
| V6 | **PASS** | MODEL-BASED (Brownian null on real look schedules, variance-based information; OBF ~5% by construction): naive 24.9%, mSPRT 1.22%, OBF 5.06%, fixed 5.05% |
| V7 | **PASS** | 2,000 reps × 24 looks: naive 27.0%, mSPRT 0.85%, OBF 5.60% |
| V8 | **INFO** | trajectories 396 -> 381 (valid variance, >=2 daily looks) -> 361 (>=3 looks). Declared significant: naive 218, fixed-horizon 105, mSPRT 100, OBF 136; SRM flagged 11/99 (equal-split assumption) |
| H1 | **PASS** | C2ST AUC = 0.50997 ± 0.00030 (z = 33 vs 0.5). Passes the pre-set 0.51 threshold, but is overwhelmingly significant: see docs (balance finding) |
| H2 | **PASS** | Qini area above random / total increment (non-standard; visit, holdout): T 0.352 [0.335, 0.370], X 0.371 [0.352, 0.388], random 0.003; standard Qini coefficient X 0.091 |
| H3 | **PASS** | n=300,000: max abs diff 0.0e+00, rel 0.0e+00 |
| H4 | **PASS** | top 20% by visit X-learner keeps 91.0% of IPW incremental visits (90% CI 88.6%-93.8%) on 78.5% of exposures (efficiency 1.16x); conversion X-learner keeps 87.4% of incremental conversions on 69.1% (response model 95.0% on 73.5%) |
| N1 | **PASS** | visit ITT 0.01034 (compliance 3.60%) → CACE 0.2870 ± 0.0040; max rel diff vs linearmodels IV2SLS = 1.8e-10 |
| N2 | **FAIL** | SE ratio 2SLS(f0–f11)/Wald: visit 0.959, conversion 1.042; 2SLS(CUPAC)/Wald: visit 0.870, conversion 0.944 |
| N3 | **PASS** | exposed users in control = 0; sharp IV inequalities (Kitagawa 2015) by predicted-exposure decile: 0/40 violated; retargeting caveat recorded |
| N4 | **FAIL** | ideal-trigger sample-size ratio (85/15-weighted): visit 8.1x, conversion 2.3x; top-k predicted-trigger diluted estimate within full-ITT CI at top 10/20/30/50%: [False, False, False, False] (the unbiased stratified use is reported separately) |
| W1 | **PASS** | fitted; NPMLE logL >= normal logL holds by construction (it nests the normal), so it is a sanity check, not evidence of non-normality; metric_1: 356.7 vs 406.3; metric_2: 314.9 vs 357.1; metric_3: 313.4 vs 347.6; metric_4: 313.9 vs 339.8 |
| W2 | **PASS** | self-consistent (truth from normal prior) coverage metric_1=95.0%, metric_2=95.0%, metric_3=94.8%, metric_4=94.8%; EB MSE < raw MSE: True. Misspecified (truth from NPMLE prior), coverage among winners: metric_1=85%, metric_2=84%, metric_3=84%, metric_4=85% |
| W3 | **INFO** | metric_1: 16 winners, small (<1%) winners overstated x1.34 (median, NPMLE), FDR normal 0.6% vs NPMLE 4.4%, P(true lift>+0.1%) 23% [12%, 39%]; metric_2: 12 winners, small (<1%) winners overstated x1.48 (median, NPMLE), FDR normal 0.3% vs NPMLE 4.5%, P(true lift>+0.1%) 19% [12%, 43%]; metric_3: 12 winners, small (<1%) winners overstated x1.85 (median, NPMLE), FDR normal 0.7% vs NPMLE 1.2%, P(true lift>+0.1%) 28% [15%, 64%]; metric_4: 16 winners, small (<1%) winners overstated x1.81 (median, NPMLE), FDR normal 0.8% vs NPMLE 2.6%, P(true lift>+0.1%) 43% [30%, 58%] |
| W4 | **INFO** | metric_1 NPMLE excess kurtosis 8.2 [4.2, 16.0], metric_2 NPMLE excess kurtosis 6.8 [3.0, 12.1], metric_3 NPMLE excess kurtosis 6.9 [3.3, 13.9], metric_4 NPMLE excess kurtosis 5.1 [2.0, 10.4] |
| P1 | **PASS** | real trajectories (n=361): naive declared 218, not supported by final 116, sign reversals 33 vs MODEL expectation (NPMLE effects) 162 / 60 / 8; OBF 38 unsupported, 4 reversals; mSPRT 32, 9 |
| P2 | **INFO** | detection among final-significant: naive 97%, obf 93%, msprt 65%; on the 68 trajectories both detect, median stop day mSPRT 11.5 vs OBF 13 (horizon median 45) |
| B1 | **PASS** | test_expected_loss_vs_monte_carlo=PASS |
| B2 | **PASS** | panel shown when informative (ASOS d53f0e: framework HOLD, expected-loss rule ship). ASOS metric 1 under the NPMLE truth model: p-value rule ships 16 (expected harmful 0.7), fixed eps=0.1% loss rule ships 66 (14 with negative posterior mean; expected harmful 25.1) |
| T1 | **INFO** | lowest MSE at cap p95 (VR 14%, bias -3.6%); p90: VR 34%, bias -11.2%, power 73%→82%; caps ≥p99 do nothing (source tops out at $499) |
| E1 | **PASS** | test_criteo_readout_has_all_sections=PASS |
| E2 | **PASS** | test_srm_bug_blocks=PASS |
| E3 | **PASS** | make all exit 0 in 25.1 min |
| E4 | **PASS** | ruff check exit 0, ruff format --check exit 0, mypy src exit 0 |
| E5 | **PASS** | 42 tests, 0 failed; coverage of expkit/stats = 92.1% |
| E6 | **PASS** | Dockerfile + .github/workflows/ci.yml included, not run (no remote; Docker daemon unavailable) |
| E7 | **PASS** | 25.1 min wall clock (shared host) |
| R1 | **PASS** | scorecard HOLD — balance check failed [R1b]; conversion post-stratified lift over negative-sampling corrections k=1..20: +46.8% to +59.8% (visit +17.4% to +25.7%); odds ratios crude/MH: conversion 1.596/1.474, visit 1.285/1.261 |
| R2 | **PASS** | test_negative_sampling_creates_imbalance_that_adjustment_cannot_fix=PASS |
| R3 | **PASS** | 10,000 A/A reps at 85/15: FPR 4.63%-5.38%; min KS p 0.043 |
| R4 | **PASS** | test_obf_type1_with_near_duplicate_looks=PASS |
| R5 | **PASS** | test_rendered_docs_match_results=PASS |
