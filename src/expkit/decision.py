"""The explicit ship / no-ship decision framework used by every scorecard.

Rules are evaluated in order; the first one that fires decides. Writing them down is the point:
the same evidence must always lead to the same call, whoever the analyst is (cf. Kohavi et al.
2020 ch. 3 and 21; Spotify's "Risk-aware product decisions in A/B tests with multiple metrics",
Schultzberg et al. 2024).

R1  SRM (p < 0.0005)                                   -> BLOCKED: data not trustworthy.
R1b Pre-treatment covariate balance fails (p < 0.001)  -> HOLD: assignment not trustworthy; the
    effect *size* is not identified without knowing why. Investigate before acting on magnitude.
R2  Any guardrail significantly worse than its margin  -> NO-SHIP: guardrail harm.
R3  Any guardrail not proven within its margin         -> HOLD: extend / investigate.
R4  Primary metric: BH-adjusted significant, positive  -> SHIP.
R5  Primary metric: BH-adjusted significant, negative  -> NO-SHIP.
R6  Otherwise                                          -> NO-SHIP: no evidence of improvement.

A failed balance check is treated like SRM, not as a warning: switching to a regression-adjusted
estimate silently assumes selection on observables, which is exactly what a balance failure
puts in doubt. The readout still shows what R4–R6 would say under each estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from expkit.stats.types import Estimate


def fmt_p(p: float) -> str:
    """p-values that underflow double precision are shown as an upper bound, never as 0."""
    return "<1e-300" if p <= 1e-300 else f"{p:.2g}"


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


def primary_rule(primary: Estimate, primary_q: float, alpha: float) -> Decision:
    """R4–R6 only: what the primary metric alone says (used for the per-estimator robustness rows)."""
    if primary_q < alpha and primary.delta > 0:
        return Decision(
            "SHIP",
            "R4",
            f"Primary metric {primary.metric} up {primary.rel_lift:+.2%} (95% CI {primary.rel_ci_low:+.2%} to "
            f"{primary.rel_ci_high:+.2%}, BH q = {fmt_p(primary_q)}).",
        )
    if primary_q < alpha and primary.delta < 0:
        return Decision(
            "NO-SHIP", "R5", f"Primary metric {primary.metric} significantly down ({primary.rel_lift:+.2%})."
        )
    return Decision(
        "NO-SHIP",
        "R6",
        f"No significant change in {primary.metric}: {primary.rel_lift:+.2%} (95% CI {primary.rel_ci_low:+.2%} to "
        f"{primary.rel_ci_high:+.2%}, BH q = {fmt_p(primary_q)}).",
    )


def decide(
    srm_blocked: bool,
    srm_p: float,
    guardrails: list[GuardrailResult],
    primary: Estimate,
    primary_q: float,
    alpha: float,
    balance_failed: bool = False,
    balance_reason: str = "",
) -> Decision:
    if srm_blocked:
        return Decision(
            "BLOCKED — SRM",
            "R1",
            f"Sample ratio mismatch (p = {fmt_p(srm_p)} < 0.0005). Do not interpret any metric; "
            "debug assignment/logging first.",
        )
    if balance_failed:
        return Decision("HOLD — balance check failed", "R1b", balance_reason)
    failed = [g for g in guardrails if g.status == "fail"]
    if failed:
        g = failed[0]
        return Decision(
            "NO-SHIP",
            "R2",
            f"Guardrail {g.metric} is worse than its -{g.margin:.1%} margin "
            f"(95% CI of relative change upper bound {g.estimate.rel_ci_high:+.2%}).",
        )
    unsure = [g for g in guardrails if g.status == "inconclusive"]
    if unsure:
        g = unsure[0]
        return Decision(
            "HOLD",
            "R3",
            f"Guardrail {g.metric} not proven within -{g.margin:.1%} "
            f"(CI lower bound {g.estimate.rel_ci_low:+.2%}). Extend or investigate.",
        )
    d = primary_rule(primary, primary_q, alpha)
    if d.rule == "R4":
        d.reason += " All guardrails within margin."
    return d
