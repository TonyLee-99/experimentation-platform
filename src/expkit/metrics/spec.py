"""YAML metric definitions (the "semantic layer").

A metric is defined once, centrally, and every experiment reuses it — the same idea as
Eppo's metric definitions, Statsig's metric catalog or Netflix XP's metric repository. This
prevents each analyst from re-implementing "conversion rate" slightly differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Join(BaseModel):
    table: str
    key: str  # join column (not "on": YAML 1.1 parses on as a boolean)


class Source(BaseModel):
    table: str
    unit: str
    variant: str
    grain: Literal["unit", "event"] = "unit"
    joins: list[Join] = Field(default_factory=list)


class Metric(BaseModel):
    name: str
    type: Literal["mean", "ratio"]
    description: str = ""
    expr: str | None = None  # mean metrics: per-row SQL expression, summed per unit
    numerator: str | None = None  # ratio metrics
    denominator: str | None = None
    higher_is_better: bool = True

    @model_validator(mode="after")
    def _check(self) -> Metric:
        if self.type == "mean" and not self.expr:
            raise ValueError(f"mean metric {self.name} needs 'expr'")
        if self.type == "ratio" and not (self.numerator and self.denominator):
            raise ValueError(f"ratio metric {self.name} needs numerator and denominator")
        return self


class MetricCatalog(BaseModel):
    source: Source
    metrics: list[Metric]
    covariate_sets: dict[str, list[str]] = Field(default_factory=dict)

    def get(self, name: str) -> Metric:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(f"unknown metric {name!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> MetricCatalog:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))
