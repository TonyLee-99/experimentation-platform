# Acceptance criteria

Written on 2026-10-29, **before** any engine, validation or report code existed.
The only facts known at that point were the raw row counts from the data download.
Every criterion is measurable. At the end of the build, each line gets PASS / FAIL with the command
that produced the evidence and an excerpt of its output (see "Results" at the bottom).

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
| E7 | `make all` finishes in < 45 minutes on the 10-core workstation. |

## Raised-bar additions (added 2026-10-29 before building)

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

## Results

<!-- results-table -->
