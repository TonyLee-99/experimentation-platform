"""Sample Ratio Mismatch (SRM) check.

If the observed split of units differs from the configured split by more than chance, the
assignment or logging is broken and *no metric result can be trusted* (Kohavi, Tang & Xu 2020,
"Trustworthy Online Controlled Experiments", ch. 21; Fabijan et al. 2019, "Diagnosing Sample
Ratio Mismatch in Online Controlled Experiments", KDD). Microsoft ExP uses a very strict
p-value threshold (0.0005) because the test is run on every experiment and false alarms are
costly; we use the same default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

SRM_P_THRESHOLD = 0.0005


@dataclass(frozen=True)
class SRMResult:
    observed: list[float]
    expected_share: list[float]
    chi2: float
    p_value: float
    threshold: float

    @property
    def blocked(self) -> bool:
        return self.p_value < self.threshold

    @property
    def observed_share(self) -> list[float]:
        tot = sum(self.observed)
        return [o / tot for o in self.observed]


def srm_check(
    counts: list[float] | np.ndarray,
    expected_share: list[float] | np.ndarray,
    threshold: float = SRM_P_THRESHOLD,
) -> SRMResult:
    """Pearson chi-square goodness-of-fit of observed unit counts vs. the configured split."""
    obs = np.asarray(counts, dtype=np.float64)
    share = np.asarray(expected_share, dtype=np.float64)
    share = share / share.sum()
    exp = obs.sum() * share
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=obs.size - 1))
    return SRMResult(obs.tolist(), share.tolist(), chi2, p, threshold)
