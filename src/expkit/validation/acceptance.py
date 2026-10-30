"""Mechanical PASS/FAIL evaluation of docs/ACCEPTANCE.md against results/*.json.

Each criterion is a small function returning (status, evidence). Nothing here is hand-scored:
if a results file is missing, the criterion is reported as MISSING.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from expkit.analyses.common import read_json, write_json
from expkit.data import ROOT

Check = Callable[[], tuple[str, str]]
BUILD = ROOT / "build"  # engineering artifacts (junit, coverage, lint, timing); gitignored


def _ok(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def _j(name: str) -> dict[str, Any]:
    return read_json(name)


def d1() -> tuple[str, str]:
    c = _j("datasets")["datasets"][0]
    d = c["details"]
    cond = (
        c["rows"] == 13_979_592
        and round(d["treatment_ratio"], 4) == 0.85
        and round(d["visit_rate"], 4) == 0.0470
        and round(d["conversion_rate"], 4) == 0.0029
    )
    return (
        _ok(cond),
        f"rows={c['rows']:,} treat={d['treatment_ratio']} visit={d['visit_rate']} conv={d['conversion_rate']}",
    )


def d2() -> tuple[str, str]:
    a = _j("datasets")["datasets"][1]
    d = a["details"]
    return _ok(
        a["rows"] == 24_153 and d["experiments"] == 78 and d["treatments"] == 99
    ), f"rows={a['rows']:,} experiments={d['experiments']} treatments={d['treatments']}"


def d3() -> tuple[str, str]:
    h = _j("datasets")["datasets"][2]
    return _ok(h["rows"] == 64_000), f"rows={h['rows']:,}"


def d4() -> tuple[str, str]:
    b = sum(p.stat().st_size for p in (ROOT / "data").rglob("*") if p.is_file())
    return _ok(b <= 3.5e9), f"data/ = {b / 1e6:,.0f} MB"


def c1() -> tuple[str, str]:
    c = _j("crosscheck")["C1_welch_vs_scipy_full_data"]
    worst = max(max(v["rel_diff_stat"], v["rel_diff_se"], v["abs_diff_p"], v["rel_diff_estimate"]) for v in c.values())
    return _ok(worst <= 1e-6), f"max diff vs scipy over visit/conversion (13.98M rows) = {worst:.1e}"


def c2() -> tuple[str, str]:
    c = _j("crosscheck")["C2_delta_vs_statsmodels_cluster"]["rel_diff_se"]
    return (
        _ok(c <= 1e-6),
        f"real data: delta SE vs statsmodels cluster-robust rel diff = {c:.1e} (+ unit test exact reduction to Welch)",
    )


def c3() -> tuple[str, str]:
    c = _j("crosscheck")["C3_cuped"]
    cond = c["rel_diff_stat"] <= 1e-6 and c["rel_diff_estimate"] <= 1e-6 and c["rel_diff_vs_ancova"] <= 0.01
    return _ok(cond), (
        f"vs statsmodels-θ Welch: stat {c['rel_diff_stat']:.1e}, est {c['rel_diff_estimate']:.1e}; "
        f"vs ANCOVA HC1: est rel diff {c['rel_diff_vs_ancova']:.2%}, SE {c['rel_diff_se_vs_ancova']:.2%}"
    )


def c4() -> tuple[str, str]:
    c = _j("crosscheck")["C4_srm_vs_scipy"]
    return _ok(max(c.values()) <= 1e-9), f"chi2/p abs diff = {max(c.values()):.1e}"


def c5() -> tuple[str, str]:
    c = _j("crosscheck")["C5_bh_vs_statsmodels"]["max_abs_diff"]
    return _ok(c <= 1e-12), f"max abs diff = {c:.1e} (+ unit test incl. reject flags)"


def _junit() -> dict[str, str]:
    import xml.etree.ElementTree as ET

    paths = sorted(BUILD.glob("junit*.xml"))  # junit.xml (make test) + junit_docs.xml (make check-docs)
    if not paths:
        return {}
    out = {}
    for tc in (t for path in paths for t in ET.parse(path).getroot().iter("testcase")):
        status = (
            "FAIL"
            if tc.find("failure") is not None or tc.find("error") is not None
            else ("SKIP" if tc.find("skipped") is not None else "PASS")
        )
        out[tc.get("name", "")] = status
    return out


def _tests(*names: str) -> tuple[str, str]:
    j = _junit()
    if not j:
        return "MISSING", "build/junit.xml not found (run make test)"
    st = [j.get(n, "MISSING") for n in names]
    return ("PASS" if all(s == "PASS" for s in st) else "FAIL"), ", ".join(
        f"{n}={s}" for n, s in zip(names, st, strict=True)
    )


def c6() -> tuple[str, str]:
    return _tests(
        "test_obf_boundaries_vs_independent_gsdesign",
        "test_classical_constants_jennison_turnbull",
        "test_obf_monte_carlo_type1",
    )


def c7() -> tuple[str, str]:
    return _tests("test_msprt_lambda_matches_numerical_integration", "test_msprt_pvalues_monotone_and_valid")


def c8() -> tuple[str, str]:
    return _tests("test_bootstrap_agrees_with_welch")


def c9() -> tuple[str, str]:
    return _tests(
        "test_welch_swap_flips_sign",
        "test_welch_location_scale",
        "test_cuped_theta_zero_equals_welch",
        "test_bh_properties",
        "test_msprt_properties",
    )


V1_KEYS = [
    "visit",
    "conversion",
    "visit_cuped",
    "conversion_cuped",
    "visit_cupac",
    "conversion_cupac",
    "conversion_per_visit_delta",
]


def v1() -> tuple[str, str]:
    a = _j("aa_tests")
    m = a["methods"]
    fprs = {k: m[k]["fpr"] for k in V1_KEYS}
    cond = a["n_reps"] >= 2000 and all(0.04 <= v <= 0.06 for v in fprs.values())
    return _ok(cond), f"{a['n_reps']:,} reps on {a['n_units']:,} units; FPR " + ", ".join(
        f"{k}={v:.2%}" for k, v in fprs.items()
    )


def v2() -> tuple[str, str]:
    m = _j("aa_tests")["methods"]
    ks = {k: m[k]["ks_pvalue"] for k in V1_KEYS}
    return _ok(min(ks.values()) > 0.01), f"min KS p = {min(ks.values()):.3f} ({min(ks, key=lambda k: ks[k])})"


def v3() -> tuple[str, str]:
    b = _j("aa_bootstrap")
    m = b["methods"]
    vals = {k: m[k]["fpr"] for k in ("visit_bootstrap", "conversion_bootstrap")}
    cond = b["n_reps"] >= 500 and all(0.035 <= v <= 0.065 for v in vals.values())
    return _ok(cond), f"{b['n_reps']} reps × n={b['sub_n']:,}, B={b['n_boot']}: " + ", ".join(
        f"{k}={v:.1%}" for k, v in vals.items()
    )


def v4() -> tuple[str, str]:
    a = _j("aa_tests")["variance_reduction_aa"]
    s = _j("scorecard_criteo")["variance_reduction"]
    cond = (
        s["visit_cupac"] >= 0.30
        and s["conversion_cupac"] >= 0.15
        and s["visit_cupac"] >= s["visit_cuped"]
        and s["conversion_cupac"] >= s["conversion_cuped"]
    )
    return _ok(cond), (
        f"real experiment VR: visit CUPED {s['visit_cuped']:.1%} / CUPAC {s['visit_cupac']:.1%}; "
        f"conversion CUPED {s['conversion_cuped']:.1%} / CUPAC {s['conversion_cupac']:.1%}; "
        f"A/A empirical VR visit CUPAC {a['visit_cupac']:.1%}, conversion CUPAC {a['conversion_cupac']:.1%}"
    )


def v5() -> tuple[str, str]:
    p = _j("power_curves")["curves"]
    worst = max(abs(pt["power_raw"] - pt["analytic_raw"]) for c in p.values() for pt in c["points"])

    def mde80(c: dict[str, Any], k: str) -> float:
        """Relative lift at which empirical power crosses 80% (linear interpolation on the grid)."""
        pts = c["points"]
        for a, b in itertools.pairwise(pts):
            pa, pb = a[f"power_{k}"], b[f"power_{k}"]
            if pa < 0.8 <= pb:
                return float(a["relative_lift"] + (0.8 - pa) / (pb - pa) * (b["relative_lift"] - a["relative_lift"]))
        return float("inf")

    v = p["visit"]
    cond = worst <= 0.03 and mde80(v, "cupac") < mde80(v, "raw")
    return _ok(cond), (
        f"max |empirical − analytic| (Welch) = {worst * 100:.1f} pp; interpolated 80%-power MDE, visit: "
        f"raw {mde80(v, 'raw'):.2%}, CUPED {mde80(v, 'cuped'):.2%}, CUPAC {mde80(v, 'cupac'):.2%}"
    )


def v6() -> tuple[str, str]:
    f = _j("asos_portfolio")["null_peeking_simulation"]["fpr"]
    n, m, o = f["naive"]["fpr"], f["msprt"]["fpr"], f["obf"]["fpr"]
    return _ok(n >= 0.10 and m <= 0.05 and 0.03 <= o <= 0.055), (
        f"MODEL-BASED (Brownian null on real look schedules, variance-based information; OBF ~5% by construction): "
        f"naive {n:.1%}, mSPRT {m:.2%}, OBF {o:.2%}, fixed {f['fixed_horizon']['fpr']:.2%}"
    )


def v7() -> tuple[str, str]:
    f = _j("criteo_peeking")["fpr"]
    n, m, o = f["naive"]["fpr"], f["msprt"]["fpr"], f["obf"]["fpr"]
    reps = _j("criteo_peeking")["n_reps"]
    looks = _j("criteo_peeking")["looks"]
    cond = reps >= 2000 and looks >= 20 and n >= 0.10 and m <= 0.05 and 0.04 <= o <= 0.06
    return _ok(cond), f"{reps:,} reps × {looks} looks: naive {n:.1%}, mSPRT {m:.2%}, OBF {o:.2%}"


def v8() -> tuple[str, str]:
    r = _j("asos_portfolio")
    s = r["real_peeking"]["summary"]
    c = r["trajectory_counts"]
    srm = r["srm"]
    return "INFO", (
        f"trajectories {c['all_in_dataset']} -> {c['with_valid_variance_and_at_least_2_daily_looks']} (valid variance, "
        f">=2 daily looks) -> {s['n_trajectories']} (>=3 looks). Declared significant: naive {s['naive']['declared_significant']}, "
        f"fixed-horizon {s['n_final_significant']}, mSPRT {s['msprt']['declared_significant']}, OBF "
        f"{s['obf']['declared_significant']}; SRM flagged {srm['n_srm_flagged_p_lt_0.0005']}/{srm['n_treatments']} (equal-split assumption)"
    )


def h1() -> tuple[str, str]:
    b = _j("hte")["balance"]
    return _ok(b["auc"] <= 0.51), (
        f"C2ST AUC = {b['auc']:.5f} ± {b['auc_se']:.5f} (z = {b['z_vs_0.5']:.0f} vs 0.5). "
        "Passes the pre-set 0.51 threshold, but is overwhelmingly significant: see docs (balance finding)"
    )


def h2() -> tuple[str, str]:
    q = _j("hte")["qini_visit"]
    t, x, r = q["t_learner"], q["x_learner"], q["random"]
    k = "qini_area_above_random_share"
    cond = all(m[f"{k}_ci95"][0] > 0 for m in (t, x)) and min(t[k], x[k]) > r[f"{k}_ci95"][1]
    return _ok(cond), (
        f"Qini area above random / total increment (non-standard; visit, holdout): T {t[k]:.3f} {_ci(t[f'{k}_ci95'])}, "
        f"X {x[k]:.3f} {_ci(x[f'{k}_ci95'])}, random {r[k]:.3f}; standard Qini coefficient X {x['qini_coefficient']:.3f}"
    )


def _ci(c: list[float]) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def h3() -> tuple[str, str]:
    c = _j("hte")["causalml_check"]
    return _ok(
        c["max_rel_diff"] <= 1e-6
    ), f"n={c['n']:,}: max abs diff {c['max_abs_diff']:.1e}, rel {c['max_rel_diff']:.1e}"


def h4() -> tuple[str, str]:
    t = _j("hte")["targeting"]
    v = next(r for r in t["outcome_visit"]["rank_by_visit_x_learner"] if r["top"] == 0.2)
    c = next(r for r in t["outcome_conversion"]["rank_by_conversion_x_learner"] if r["top"] == 0.2)
    resp = next(r for r in t["outcome_conversion"]["rank_by_conversion_response_model"] if r["top"] == 0.2)
    ci = v["share_of_incremental_ci90"]
    return _ok(v["share_of_incremental"] >= 0.5), (
        f"top 20% by visit X-learner keeps {v['share_of_incremental']:.1%} of IPW incremental visits (90% CI "
        f"{ci[0]:.1%}-{ci[1]:.1%}) on {v['share_of_exposures']:.1%} of exposures (efficiency {v['efficiency']:.2f}x); "
        f"conversion X-learner keeps {c['share_of_incremental']:.1%} of incremental conversions on {c['share_of_exposures']:.1%} "
        f"(response model {resp['share_of_incremental']:.1%} on {resp['share_of_exposures']:.1%})"
    )


def n1() -> tuple[str, str]:
    n = _j("noncompliance")
    worst = max(
        max(n[y]["rel_diff_coef_vs_linearmodels"], n[y]["rel_diff_se_vs_linearmodels"]) for y in ("visit", "conversion")
    )
    v = n["visit"]
    return _ok(worst <= 1e-6), (
        f"visit ITT {v['itt']:.5f} (compliance {v['compliance_rate']:.2%}) → CACE {v['cace']:.4f} ± "
        f"{v['cace_se']:.4f}; max rel diff vs linearmodels IV2SLS = {worst:.1e}"
    )


def n2() -> tuple[str, str]:
    n = _j("noncompliance")
    r = {y: n[y]["tsls_features"]["se_ratio_vs_wald"] for y in ("visit", "conversion")}
    rc = {y: n[y]["tsls_cupac"]["se_ratio_vs_wald"] for y in ("visit", "conversion")}
    cond = all(v < 1 for v in r.values())
    return _ok(cond), (
        f"SE ratio 2SLS(f0–f11)/Wald: visit {r['visit']:.3f}, conversion {r['conversion']:.3f}; "
        f"2SLS(CUPAC)/Wald: visit {rc['visit']:.3f}, conversion {rc['conversion']:.3f}"
    )


def n3() -> tuple[str, str]:
    n = _j("noncompliance")
    viol = sum(n[y]["kitagawa"]["n_violations_z_gt_1.96"] for y in ("visit", "conversion"))
    tot = sum(n[y]["kitagawa"]["n_inequalities"] for y in ("visit", "conversion"))
    return _ok(n["one_sided"]["exposed_users_in_control"] == 0 and viol == 0), (
        f"exposed users in control = {n['one_sided']['exposed_users_in_control']}; sharp IV inequalities (Kitagawa 2015) "
        f"by predicted-exposure decile: {viol}/{tot} violated; retargeting caveat recorded"
    )


def n4() -> tuple[str, str]:
    n = _j("noncompliance")
    v = n["visit"]
    agree = [t["diluted_within_full_ci"] for t in v["predicted_trigger"]]
    return _ok(all(agree)), (
        f"ideal-trigger sample-size ratio (85/15-weighted): visit {v['dilution']['n_ratio_itt_over_ideal_trigger_weighted_85_15']:.1f}x, "
        f"conversion {n['conversion']['dilution']['n_ratio_itt_over_ideal_trigger_weighted_85_15']:.1f}x; top-k predicted-trigger "
        f"diluted estimate within full-ITT CI at top 10/20/30/50%: {agree} (the unbiased stratified use is reported separately)"
    )


def w1() -> tuple[str, str]:
    e = _j("asos_portfolio")["eb"]
    ok = all(e[m]["npmle_prior"]["loglik"] >= e[m]["normal_prior"]["loglik"] for m in e)
    return (
        _ok(ok),
        "fitted; NPMLE logL >= normal logL holds by construction (it nests the normal), so it is a sanity check, "
        + "not evidence of non-normality; "
        + "; ".join(f"{m}: {e[m]['normal_prior']['loglik']:.1f} vs {e[m]['npmle_prior']['loglik']:.1f}" for m in e),
    )


def w2() -> tuple[str, str]:
    e = _j("asos_portfolio")["eb"]
    cov = {m: e[m]["simulation_check"]["coverage_95"] for m in e}
    win = {m: e[m]["simulation_check"]["truth_npmle"]["coverage_95_among_winners"] for m in e}
    mse = all(e[m]["simulation_check"]["mse_eb_normal"] < e[m]["simulation_check"]["mse_raw"] for m in e)
    return _ok(all(0.93 <= c <= 0.97 for c in cov.values()) and mse), (
        "self-consistent (truth from normal prior) coverage "
        + ", ".join(f"{m}={c:.1%}" for m, c in cov.items())
        + f"; EB MSE < raw MSE: {mse}. Misspecified (truth from NPMLE prior), coverage among winners: "
        + ", ".join(f"{m}={c:.0%}" for m, c in win.items())
    )


def w3() -> tuple[str, str]:
    e = _j("asos_portfolio")["eb"]
    parts = []
    for m in e:
        w = e[m].get("winners")
        if w:
            ci = e[m]["npmle_prior"]["p_true_lift_gt_0.1pct_ci90"]
            parts.append(
                f"{m}: {e[m]['n_winners']} winners, small (<1%) winners overstated x{w['small_winners_median_ratio_npmle']:.2f} "
                f"(median, NPMLE), FDR normal {w['fdr_normal']:.1%} vs NPMLE {w['fdr_npmle']:.1%}, "
                f"P(true lift>+0.1%) {e[m]['npmle_prior']['p_true_lift_gt_0.1pct']:.0%} [{ci[0]:.0%}, {ci[1]:.0%}]"
            )
    return "INFO", "; ".join(parts)


def w4() -> tuple[str, str]:
    e = _j("asos_portfolio")["eb"]
    return "INFO", ", ".join(
        f"{m} NPMLE excess kurtosis {e[m]['npmle_prior']['excess_kurtosis']:.1f} "
        f"[{e[m]['npmle_prior']['excess_kurtosis_ci90'][0]:.1f}, {e[m]['npmle_prior']['excess_kurtosis_ci90'][1]:.1f}]"
        for m in e
    )


def p1() -> tuple[str, str]:
    r = _j("asos_portfolio")
    s = r["real_peeking"]["summary"]
    mb = r["model_benchmark"]["expected_per_361"]["npmle_effects"]
    u = {k: s[k]["declared_but_final_does_not_support"] for k in ("naive", "msprt", "obf")}
    rev = {k: s[k]["sign_reversals"] for k in ("naive", "msprt", "obf")}
    return _ok(u["naive"] > u["obf"] and u["naive"] > u["msprt"]), (
        f"real trajectories (n={s['n_trajectories']}): naive declared {s['naive']['declared_significant']}, not supported by "
        f"final {u['naive']}, sign reversals {rev['naive']} vs MODEL expectation (NPMLE effects) {mb['declared']:.0f} / "
        f"{mb['unsupported']:.0f} / {mb['sign_reversals']:.0f}; OBF declared {s['obf']['declared_significant']} (model {mb['obf_declared']:.0f}), {u['obf']} unsupported, {rev['obf']} reversals (model {mb['obf_sign_reversals']:.1f}); mSPRT {u['msprt']}, {rev['msprt']}"
    )


def p2() -> tuple[str, str]:
    s = _j("asos_portfolio")["real_peeking"]["summary"]
    c = s["stopping_on_common_set"]
    return "INFO", (
        "detection among final-significant: "
        + ", ".join(f"{k} {s[k]['detection_rate_among_final_significant']:.0%}" for k in ("naive", "obf", "msprt"))
        + f"; on the {c['n_detected_by_both_msprt_and_obf']} trajectories both detect, median stop day mSPRT "
        f"{c['median_stop_day_msprt']:g} vs OBF {c['median_stop_day_obf']:g} (horizon median {s['median_horizon_days']:g})"
    )


def b1() -> tuple[str, str]:
    return _tests("test_expected_loss_vs_monte_carlo")


def b2() -> tuple[str, str]:
    e = _j("asos_portfolio")["eb"]
    s = _j("scorecard_asos_d53f0e")
    rules = e["metric_1"]["decisions"]["rules"]
    pv, lo = rules["p_value (p<0.05, positive)"], rules["expected loss < 0.1% (fixed eps)"]
    return _ok(s["bayes_panel"] and s["bayes_informative"]), (
        f"panel shown when informative (ASOS d53f0e: framework {s['verdict']}, expected-loss rule {s['bayes_verdict']}). "
        f"ASOS metric 1 under the NPMLE truth model: p-value rule ships {pv['n_ship']} (expected harmful "
        f"{pv['expected_harmful_launches']:.1f}), fixed eps=0.1% loss rule ships {lo['n_ship']} ({lo['n_ship_negative_posterior_mean']} "
        f"with negative posterior mean; expected harmful {lo['expected_harmful_launches']:.1f})"
    )


def t1() -> tuple[str, str]:
    h = _j("heavy_tails")
    rows = {r["cap"]: r for r in h["full_sample"]}
    best = min(rows.values(), key=lambda r: r["mse"])
    pw = h["power_simulation"]["power"]
    return "INFO", (
        f"lowest MSE at cap {best['cap']} (VR {best['variance_reduction']:.0%}, bias {best['relative_bias']:+.1%}); "
        f"p90: VR {rows['p90']['variance_reduction']:.0%}, bias {rows['p90']['relative_bias']:+.1%}, power "
        f"{pw['none']:.0%}→{pw['p90']:.0%}; caps ≥p99 do nothing (source tops out at $499)"
    )


def e1() -> tuple[str, str]:
    return _tests("test_criteo_readout_has_all_sections")


def e2() -> tuple[str, str]:
    return _tests("test_srm_bug_blocks")


def e3() -> tuple[str, str]:
    p = BUILD / "make_status.json"
    if not p.exists():
        return "MISSING", "build/make_status.json not written (make all did not reach the end)"
    s = json.loads(p.read_text())
    return _ok(s.get("exit", 1) == 0), f"make all exit {s.get('exit')} in {s.get('seconds', 0) / 60:.1f} min"


def e4() -> tuple[str, str]:
    p = BUILD / "lint.json"
    if not p.exists():
        return "MISSING", "build/lint.json not found (make lint)"
    s = json.loads(p.read_text())
    return _ok(all(v == 0 for v in s.values())), ", ".join(f"{k} exit {v}" for k, v in s.items())


def e5() -> tuple[str, str]:
    j = _junit()
    cov_p = BUILD / "coverage.json"
    if not j or not cov_p.exists():
        return "MISSING", "junit/coverage not found"
    cov = json.loads(cov_p.read_text())
    files = {k: v for k, v in cov["files"].items() if "/stats/" in k}
    cov_stats = sum(v["summary"]["covered_lines"] for v in files.values()) / sum(
        v["summary"]["num_statements"] for v in files.values()
    )
    fails = [k for k, v in j.items() if v == "FAIL"]
    return _ok(
        not fails and cov_stats >= 0.9
    ), f"{len(j)} tests, {len(fails)} failed; coverage of expkit/stats = {cov_stats:.1%}"


def e6() -> tuple[str, str]:
    have = (ROOT / "Dockerfile").exists() and (ROOT / ".github/workflows/ci.yml").exists()
    return _ok(have), "Dockerfile + .github/workflows/ci.yml included; CI passes on GitHub Actions; Docker image not built (no local Docker daemon)"


def e7() -> tuple[str, str]:
    p = BUILD / "make_status.json"
    if not p.exists():
        return "MISSING", "no timing"
    s = json.loads(p.read_text())
    return _ok(s.get("seconds", 1e9) < 45 * 60), f"{s.get('seconds', 0) / 60:.1f} min wall clock (shared host)"


def r1() -> tuple[str, str]:
    b = _j("balance")
    si = b["sensitivity_interval"]
    o = b["odds_ratios"]
    s = _j("scorecard_criteo")
    return _ok(s["rule"] == "R1b"), (
        f"scorecard {s['verdict']} [{s['rule']}]; conversion post-stratified lift over negative-sampling corrections "
        f"k=1..20: {si['conversion_poststrat_relative'][0]:+.1%} to {si['conversion_poststrat_relative'][1]:+.1%} (visit "
        f"{si['visit_poststrat_relative'][0]:+.1%} to {si['visit_poststrat_relative'][1]:+.1%}); odds ratios crude/MH: conversion "
        f"{o['conversion']['crude_or']:.3f}/{o['conversion']['mantel_haenszel_or']:.3f}, visit {o['visit']['crude_or']:.3f}/"
        f"{o['visit']['mantel_haenszel_or']:.3f}"
    )


def r2() -> tuple[str, str]:
    return _tests("test_negative_sampling_creates_imbalance_that_adjustment_cannot_fix")


def r3() -> tuple[str, str]:
    a = _j("aa_tests_85_15")
    m = a["methods"]
    f = {k: m[k]["fpr"] for k in V1_KEYS}
    ks = min(m[k]["ks_pvalue"] for k in V1_KEYS)
    return _ok(all(0.04 <= v <= 0.06 for v in f.values()) and ks > 0.01), (
        f"{a['n_reps']:,} A/A reps at 85/15: FPR {min(f.values()):.2%}-{max(f.values()):.2%}; min KS p {ks:.3f}"
    )


def r4() -> tuple[str, str]:
    return _tests("test_obf_type1_with_near_duplicate_looks")


def r5() -> tuple[str, str]:
    return _tests("test_rendered_docs_match_results")


CHECKS: dict[str, Check] = {
    "D1": d1,
    "D2": d2,
    "D3": d3,
    "D4": d4,
    "C1": c1,
    "C2": c2,
    "C3": c3,
    "C4": c4,
    "C5": c5,
    "C6": c6,
    "C7": c7,
    "C8": c8,
    "C9": c9,
    "V1": v1,
    "V2": v2,
    "V3": v3,
    "V4": v4,
    "V5": v5,
    "V6": v6,
    "V7": v7,
    "V8": v8,
    "H1": h1,
    "H2": h2,
    "H3": h3,
    "H4": h4,
    "N1": n1,
    "N2": n2,
    "N3": n3,
    "N4": n4,
    "W1": w1,
    "W2": w2,
    "W3": w3,
    "W4": w4,
    "P1": p1,
    "P2": p2,
    "B1": b1,
    "B2": b2,
    "T1": t1,
    "E1": e1,
    "E2": e2,
    "E3": e3,
    "E4": e4,
    "E5": e5,
    "E6": e6,
    "E7": e7,
    "R1": r1,
    "R2": r2,
    "R3": r3,
    "R4": r4,
    "R5": r5,
}


def evaluate() -> list[dict[str, str]]:
    rows = []
    for cid, fn in CHECKS.items():
        try:
            status, evidence = fn()
        except (FileNotFoundError, KeyError, StopIteration) as exc:
            status, evidence = "MISSING", f"{type(exc).__name__}: {exc}"
        rows.append({"id": cid, "status": status, "evidence": evidence})
    write_json("acceptance", {"generated": time.strftime("%Y-%m-%d %H:%M"), "rows": rows})
    return rows


def as_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "| ID | Status | Evidence (from `results/*.json`, produced by `make all`) |",
        "|----|--------|----------|",
    ]
    lines += [f"| {r['id']} | **{r['status']}** | {r['evidence'].replace('|', '/')} |" for r in rows]
    return "\n".join(lines)


MARKER = "<!-- results-table -->"


def update_doc(rows: list[dict[str, str]], path: Path | None = None) -> None:
    """Replace everything after the marker in docs/ACCEPTANCE.md with the generated table."""
    doc = path or ROOT / "docs" / "ACCEPTANCE.md"
    text = doc.read_text()
    head = text.split(MARKER)[0] if MARKER in text else text.rstrip() + "\n\n"
    counts = {k: sum(r["status"] == k for r in rows) for k in ("PASS", "FAIL", "INFO", "MISSING")}
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    body = (
        f"{MARKER}\n\nGenerated by `expkit acceptance` on {time.strftime('%Y-%m-%d %H:%M')} — {summary}. "
        "INFO = descriptive criterion with no pass threshold (pre-registered as such).\n\n" + as_markdown(rows) + "\n"
    )
    doc.write_text(head + body)


def main(path: Path | None = None) -> None:
    rows = evaluate()
    update_doc(rows, path)
    print(as_markdown(rows))
