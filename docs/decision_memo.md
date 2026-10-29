# Decision memo: keep the display campaign running, and stop reading the raw lift

**To:** Advertiser PM / VP Growth · **From:** Product Data Science · **Data:** Criteo ad-incrementality test (13.98M
users, randomized 85% eligible / 15% held out), analyzed with `expkit analyze --config experiments/criteo.yaml`
(readout: `reports/criteo_readout.html`).

## Recommendation: SHIP (keep the campaign on), and plan budget on +17% visits, not +27%

| | Raw dashboard number | Number to plan on (feature-adjusted) | 95% CI |
|---|---|---|---|
| Site visits | +27.1% | **+16.6%** (CUPAC) | +16.0% to +17.2% |
| Conversions | +59.4% | **+46.0%** (CUPAC) | +42.8% to +49.2% |
| Conversions per visit (guardrail, must not fall > 2%) | +25.5% | passes | +21.5% to +29.4% |

The decision follows our standard rules: no sample-ratio mismatch (split exactly 85.000/15.000), the guardrail is
within its margin, and the primary metric is significant after multiple-testing correction (rule R4). The Bayesian
view agrees: the expected loss from shipping is effectively zero.

## Why the dashboard number is wrong

The two groups were not identical before the campaign started. This dataset pools several advertisers' tests that
were re-sampled to an 85% treatment share, and the treated group ended up with slightly more high-intent users
(86.1% treated among the top 10% most likely visitors vs ~84.9% elsewhere). A raw comparison credits the campaign
with visits those users would have made anyway. Three independent corrections agree on **+16.6% to +18.3%**, so
the raw +27.1% overstates the effect by about 60%. The headline split check could not catch this; our scorecard now
runs a pre-period balance check on every experiment and flags it.

## What one ad impression is worth

Only **3.6%** of eligible users were actually shown an ad. Among them, an exposure raised the visit probability by
about **+18 to +29 percentage points** (instrumental-variable estimates, adjusted vs unadjusted), from roughly 13% to
41%. Use this per-exposure value, not the diluted per-user lift, when comparing against the cost per impression.

## Where to spend (targeting)

PENDING

## Risks and what I would do next

1. **Data integrity first.** Ask engineering why assignment correlates with user features. If it comes from the
   re-sampling, our adjusted numbers stand. If assignment itself is leaky, re-run with clean randomization.
2. **Log counterfactual exposure (ghost ads)** in the control group. Analyzing only users who would have seen an ad
   would detect the same visit effect with **7× fewer users**.
3. **Validate the targeting policy prospectively** with a holdout before shifting budget.
4. Caveats: Criteo sub-sampled this data non-uniformly for privacy, so absolute incrementality is not their real
   business number. There is no cost data, so "spend" here means share of impressions.
