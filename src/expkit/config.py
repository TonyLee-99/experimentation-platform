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
    looks: int = 20
    arrival_order: Literal["random"] = "random"  # Criteo has no timestamps: simulated arrival order
    seed: int = 7
    tau_relative: float = 0.10  # mSPRT mixing sd as a share of the control mean


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
    metrics_catalog: str
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
    output: str = "reports/readout.html"
    results_json: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))
