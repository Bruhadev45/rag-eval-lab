"""Paired statistics for comparing two configurations on the same queries.

Every comparison here is *paired*: both configurations answered the same 120
queries, so the unit of analysis is the per-query difference. Paired tests are
far more sensitive than comparing two independent means, and they are the
honest choice when the query set is fixed.

Two complementary procedures:

- **Paired bootstrap** for a confidence interval on the mean difference.
- **Permutation (sign-flip) test** for a p-value under the null that the two
  configurations are exchangeable on each query.

Both are exact-in-spirit, distribution-free, and seeded, so every run of this
repository produces identical numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing configuration B against configuration A.

    Attributes:
        metric: Which metric was compared.
        mean_a: Mean of A over the queries.
        mean_b: Mean of B over the queries.
        delta: ``mean_b - mean_a`` -- positive means B is better.
        ci_low: Lower end of the 95% bootstrap CI on ``delta``.
        ci_high: Upper end of the 95% bootstrap CI on ``delta``.
        p_value: Two-sided permutation p-value for ``delta != 0``.
        n_queries: Number of paired queries.
    """

    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_queries: int

    @property
    def significant(self) -> bool:
        """True when the 95% CI excludes zero and p < 0.05."""
        return self.p_value < 0.05 and (self.ci_low > 0 or self.ci_high < 0)


def paired_comparison(
    a: list[float],
    b: list[float],
    *,
    metric: str = "",
    n_resamples: int = 10_000,
    seed: int = 20260831,
) -> PairedComparison:
    """Compare per-query metric values ``b`` against ``a``.

    Args:
        a: Metric values for the baseline configuration, one per query.
        b: Metric values for the comparison configuration, same query order.
        metric: Name recorded on the result, for reporting.
        n_resamples: Bootstrap and permutation iterations.
        seed: RNG seed. Fixed by default so results reproduce exactly.

    Raises:
        ValueError: If the lists differ in length or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"paired lists differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compare empty result lists")

    diffs = [bi - ai for ai, bi in zip(a, b)]
    observed = mean(diffs)
    n = len(diffs)
    rng = random.Random(seed)

    # Paired bootstrap: resample queries with replacement.
    boot_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    ci_low = boot_means[int(0.025 * n_resamples)]
    ci_high = boot_means[int(0.975 * n_resamples)]

    # Sign-flip permutation: under the null, each per-query difference is
    # equally likely to have either sign.
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        flipped = mean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(flipped) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
    # Add-one smoothing keeps the p-value honest at the resolution we ran.
    p_value = (at_least_as_extreme + 1) / (n_resamples + 1)

    return PairedComparison(
        metric=metric,
        mean_a=mean(a),
        mean_b=mean(b),
        delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_queries=n,
    )
