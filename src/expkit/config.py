"""Experiment config (one YAML per experiment), validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Guardrail(BaseModel):
    metric: str
    max_relative_drop: float = Field(gt=0, description="non-inferiority margin, e.g. 0.02 = -2%")


class Sequential(BaseModel):
    show: bool = True  # False when the dataset has no timestamps (a simulated order would be decoration)
    looks: int = 20
    arrival_order: Literal["random", "real"] = "random"
    seed: int = 7
    tau_relative: float = 0.10  # mSPRT mixing sd as a share of the control mean


class SummarySource(BaseModel):
    """Experiments published only as per-arm (n, mean, variance) snapshots, e.g. the ASOS dataset."""

    experiment_id: str
    variant_id: int
    metrics: dict[str, int]  # metric name -> ASOS metric_id


class Economics(BaseModel):
    """Break-even view. Prices are ASSUMPTIONS (the dataset has no cost data) and are labeled as such."""

    exposure_metric: str = "exposure"
    value_per_conversion: list[float] = Field(default_factory=lambda: [20.0, 40.0, 80.0])
    cost_per_exposed_user: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])


class Decision(BaseModel):
    alpha: float = 0.05
    loss_threshold_relative: float = 0.001  # "cost of being wrong": 0.1% of the control mean
    prior_sd_relative: float = 0.10  # normal prior on the true lift, centered at 0
    balance_p_threshold: float = 0.001


class SimulatedBug(BaseModel):
    """Labeled SIMULATION: drop a share of one arm's units matching a condition (logging bug)."""

    variant: int
    drop_share: float
    condition: str = "TRUE"


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    metrics_catalog: str = ""
    source_type: Literal["warehouse", "summary"] = "warehouse"
    summary: SummarySource | None = None
    control: int | str = 0
    treatment: int | str = 1
    expected_split: dict[str, float]
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    guardrails: list[Guardrail] = Field(default_factory=list)
    variance_reduction: dict[str, str] = Field(default_factory=dict)  # metric -> covariate set
    balance_covariate_set: str | None = None
    sequential: Sequential = Field(default_factory=Sequential)
    decision: Decision = Field(default_factory=Decision)
    simulated_bug: SimulatedBug | None = None
    compare_covariate_sets: dict[str, list[str]] = Field(default_factory=dict)  # metric -> extra sets to report VR for
    diagnostics: list[str] = Field(default_factory=list)  # shown, never decisive (e.g. mix-shift ratios)
    post_stratify: dict[str, str] = Field(default_factory=dict)  # metric -> prognostic score column (100 bins)
    sensitivity_results: str | None = None  # results/<name>.json with a negative-sampling sensitivity grid
    economics: Economics | None = None
    pm_note: str = ""
    output: str = "reports/readout.html"
    results_json: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))
