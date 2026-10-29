"""Docs drift guard: every number in README / docs comes from results/*.json via docs/templates."""

from __future__ import annotations

import pytest

from expkit.data import ROOT


def test_rendered_docs_match_results() -> None:
    if not (ROOT / "results" / "scorecard_asos_d53f0e.json").exists():
        pytest.skip("results not generated")
    from expkit.docs import render_all

    for path, text in render_all(write=False).items():
        assert path.exists(), f"{path} missing: run `expkit docs`"
        assert path.read_text() == text, f"{path.name} drifted from results/*.json: run `expkit docs`"
