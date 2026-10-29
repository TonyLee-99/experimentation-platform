"""End-to-end scorecard on the real Criteo data (E1, E2). Skipped when the data is not downloaded."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from expkit.analyses.common import CUPAC_PARQUET
from expkit.config import ExperimentConfig
from expkit.data import CRITEO_PARQUET, ROOT
from expkit.decision import GuardrailResult, decide
from expkit.stats.types import Estimate

needs_data = pytest.mark.skipif(
    not (CRITEO_PARQUET.exists() and CUPAC_PARQUET.exists()), reason="real data not present"
)


def _est(delta: float, lo: float, hi: float, p: float) -> Estimate:
    return Estimate("m", "welch", 1.0, 1.0 + delta, delta, 0.01, lo, hi, p, delta, lo, hi, 100, 100)


def test_decision_rules_order() -> None:
    good = _est(0.05, 0.02, 0.08, 1e-4)
    g_fail = GuardrailResult("g", _est(-0.1, -0.15, -0.05, 1e-6), 0.02)
    g_unsure = GuardrailResult("g", _est(-0.01, -0.04, 0.02, 0.4), 0.02)
    g_ok = GuardrailResult("g", _est(0.0, -0.01, 0.01, 0.9), 0.02)
    assert decide(True, 1e-9, [g_ok], good, 1e-4, 0.05).rule == "R1"
    assert decide(False, 0.5, [g_fail], good, 1e-4, 0.05).rule == "R2"
    assert decide(False, 0.5, [g_unsure], good, 1e-4, 0.05).rule == "R3"
    assert decide(False, 0.5, [g_ok], good, 1e-4, 0.05).verdict == "SHIP"
    assert decide(False, 0.5, [g_ok], _est(-0.05, -0.08, -0.02, 1e-4), 1e-4, 0.05).rule == "R5"
    assert decide(False, 0.5, [g_ok], _est(0.0, -0.02, 0.02, 0.9), 0.9, 0.05).rule == "R6"


def _render(name: str) -> tuple[str, str]:
    from expkit.report.scorecard import analyze, render

    cfg = ExperimentConfig.from_yaml(ROOT / "experiments" / name)
    r = analyze(cfg)
    path = render(r, Path(cfg.output).with_name("test_" + Path(cfg.output).name))
    return r.decision.verdict, Path(path).read_text()


@needs_data
@pytest.mark.data
def test_criteo_readout_has_all_sections() -> None:
    verdict, html = _render("criteo.yaml")
    for section in ("srm", "balance", "decision", "metrics", "guardrails", "bayes", "sequential"):
        assert f'id="{section}"' in html
    assert "No sample ratio mismatch" in html
    assert re.search(r"cuped\[cupac_visit\]", html)
    assert "data:image/png;base64," in html
    assert verdict in {"SHIP", "NO-SHIP", "HOLD"}


@needs_data
@pytest.mark.data
def test_srm_bug_blocks() -> None:
    verdict, html = _render("criteo_srm_bug.yaml")
    assert verdict == "BLOCKED — SRM"
    assert "SRM DETECTED" in html and "SIMULATION" in html


def test_bootstrap_fallback_on_small_skewed_sample(tmp_path: Path) -> None:
    """A tiny, very skewed experiment must be routed to the bootstrap by the scorecard."""
    import numpy as np
    import pandas as pd

    from expkit.metrics.compiler import run
    from expkit.metrics.spec import MetricCatalog
    from expkit.report.scorecard import _bootstrap_metric, _clt_check

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {"uid": np.arange(n), "arm": rng.integers(0, 2, n), "rev": rng.pareto(1.5, n) * (rng.random(n) < 0.1)}
    )
    path = tmp_path / "u.parquet"
    df.to_parquet(path)
    cat = MetricCatalog.model_validate(
        {
            "source": {"table": str(path), "unit": "uid", "variant": "arm"},
            "metrics": [{"name": "rev", "type": "mean", "expr": "rev"}],
        }
    )
    vs = run(cat, ["rev"])
    clt = _clt_check(cat, "rev", vs[0], vs[1])
    assert not clt["ok"]
    est = _bootstrap_metric(cat, "rev", None, 0.05, 0, 1)
    assert est.method == "bootstrap" and est.ci_low < est.delta < est.ci_high
