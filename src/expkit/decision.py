"""The explicit ship / no-ship decision framework used by every scorecard.

Rules are evaluated in order; the first one that fires decides. Writing them down is the point:
the same evidence must always lead to the same call, whoever the analyst is (cf. Kohavi et al.
2020 ch. 3 and 21; Microsoft ExP "Experimentation Scorecard"; Spotify's "Risk-aware product
decisions in A/B tests with multiple metrics", Schultzberg et al. 2024).

R1  SRM (p < 0.0005)                                  -> BLOCKED: data not trustworthy.
R2  Any guardrail significantly worse than its margin -> NO-SHIP: guardrail harm.
R3  Any guardrail not proven within its margin        -> HOLD: extend / investigate.
R4  Primary metric: BH-adjusted significant, positive -> SHIP.
R5  Primary metric: BH-adjusted significant, negative -> NO-SHIP.
R6  Otherwise                                         -> NO-SHIP: no evidence of improvement.

Warnings that do not change the rule but change *which estimate is primary*:
W1  Covariate imbalance (balance p < threshold)       -> raw estimates are biased; the
    regression-adjusted (CUPED/CUPAC) estimate is the decision estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from expkit.stats.types import Estimate


@dataclass
class GuardrailResult:
    metric: str
    estimate: Estimate
    margin: float

    @property
    def status(self) -> str:
        if self.estimate.rel_ci_high < -self.margin:
            return "fail"
        if self.estimate.rel_ci_low < -self.margin:
            return "inconclusive"
        return "pass"


@dataclass
class Decision:
    verdict: str
    rule: str
    reason: str
    warnings: list[str] = field(default_factory=list)


def decide(
    srm_blocked: bool,
    srm_p: float,
    guardrails: list[GuardrailResult],
    primary: Estimate,
    primary_q: float,
    alpha: float,
    balance_warning: str | None = None,
) -> Decision:
    warnings = [balance_warning] if balance_warning else []
    if srm_blocked:
        return Decision(
            "BLOCKED — SRM",
            "R1",
            f"Sample ratio mismatch (p = {srm_p:.2g} < 0.0005). Do not interpret any metric; "
            "debug assignment/logging first.",
            warnings,
        )
    failed = [g for g in guardrails if g.status == "fail"]
    if failed:
        g = failed[0]
        return Decision(
            "NO-SHIP",
            "R2",
            f"Guardrail {g.metric} is worse than its -{g.margin:.1%} margin "
            f"(95% CI of relative change upper bound {g.estimate.rel_ci_high:+.2%}).",
            warnings,
        )
    unsure = [g for g in guardrails if g.status == "inconclusive"]
    if unsure:
        g = unsure[0]
        return Decision(
            "HOLD",
            "R3",
            f"Guardrail {g.metric} not proven within -{g.margin:.1%} "
            f"(CI lower bound {g.estimate.rel_ci_low:+.2%}). Extend or investigate.",
            warnings,
        )
    if primary_q < alpha and primary.delta > 0:
        return Decision(
            "SHIP",
            "R4",
            f"Primary metric {primary.metric} up {primary.rel_lift:+.2%} "
            f"(95% CI {primary.rel_ci_low:+.2%} to {primary.rel_ci_high:+.2%}, BH q = {primary_q:.2g}); "
            "all guardrails within margin.",
            warnings,
        )
    if primary_q < alpha and primary.delta < 0:
        return Decision(
            "NO-SHIP",
            "R5",
            f"Primary metric {primary.metric} significantly down ({primary.rel_lift:+.2%}).",
            warnings,
        )
    return Decision(
        "NO-SHIP",
        "R6",
        f"No significant change in {primary.metric} (BH q = {primary_q:.2g}).",
        warnings,
    )
