"""Exploratory within-recording statistics; no frame-wise independence claim.

Continuous bouts contribute one median each. Bout labels are permuted under an
exchangeability null, and all eligible feature tests enter BY FDR correction.
Clustering and testing these same data remains exploratory/post-selection.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import math

import numpy as np

from .core import InputError

RESAMPLES = 9999
SEED = 20261102
MIN_BOUTS = 5


def mean_sd(samples):
    values = np.asarray(samples, dtype=float)
    if not len(values):
        return None, None
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def balanced_mean_sd(groups):
    """Descriptive weighted sample SD, giving each nonempty group equal mass."""
    groups = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if not groups:
        return None, None
    mean = float(np.mean([group.mean() for group in groups]))
    numerator = float(np.mean([np.mean((group - mean) ** 2) for group in groups]))
    sum_squared_weights = sum(1 / len(group) for group in groups) / len(groups) ** 2
    denominator = 1 - sum_squared_weights
    return mean, float(np.sqrt(max(0., numerator / denominator))) if denominator > 0 else None


def fdr_by(pvalues):
    """Benjamini–Yekutieli adjusted p-values for an arbitrary dependent family."""
    values = np.asarray(pvalues, dtype=float)
    if not len(values):
        return np.empty(0)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('p-values must be finite values from 0 to 1.')
    order = np.argsort(values, kind='stable')
    count = len(values)
    harmonic = float(np.sum(1 / np.arange(1, count + 1)))
    adjusted = values[order] * count * harmonic / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty(count)
    result[order] = adjusted
    return result


def _weights(counts):
    return np.concatenate([np.full(counts[0], 1 / counts[0]),
                           *[np.full(n, -1 / ((len(counts)-1) * n)) for n in counts[1:]]])


@lru_cache(maxsize=2)
def _plan(counts, resamples):
    """Small plans are shared across features; large plans stream in batches."""
    weights = _weights(counts)
    count = len(weights)
    if len(np.unique(weights)) == 2 and math.comb(count, counts[0]) <= resamples:
        rows = math.comb(count, counts[0])
        plan = np.full((rows, count), weights[-1])
        for index, selected in enumerate(combinations(range(count), counts[0])):
            plan[index, list(selected)] = weights[0]
        return plan, True
    rng = np.random.default_rng(SEED)
    plan = np.asarray([rng.permutation(weights) for _ in range(resamples)])
    plan.setflags(write=False)
    return plan, False


def permutation_test(current, other_groups, resamples=RESAMPLES, cancel=None):
    """Two-sided difference of bout means, with equal weight per rest group."""
    if cancel is not None and cancel.is_set():
        raise InputError('Statistical analysis cancelled.')
    if not isinstance(resamples, int) or resamples <= 0:
        raise ValueError('A positive integer resample count is required.')
    groups = [np.asarray(current, dtype=float), *[np.asarray(g, dtype=float) for g in other_groups]]
    if len(groups) < 2 or any(not len(group) for group in groups):
        raise ValueError('Nonempty groups are required.')
    values = np.concatenate(groups)
    if not np.isfinite(values).all():
        raise ValueError('Bout values must be finite.')
    counts = tuple(map(len, groups))
    weights = _weights(counts)
    observed = float(np.dot(weights, values))
    tolerance = 1e-12 * max(1., float(np.max(np.abs(values))))
    if float(np.ptp(values)) <= tolerance:
        return dict(p_value=1., bout_effect=observed, permutations=1, exact_test=True)
    if len(values) <= 256:
        plan, exact = _plan(counts, resamples)
        extreme = 0
        for start in range(0, len(plan), 128):
            if cancel is not None and cancel.is_set():
                raise InputError('Statistical analysis cancelled.')
            null = np.einsum('ij,j->i', plan[start:start+128], values)
            extreme += int((np.abs(null) >= abs(observed) - tolerance).sum())
        size = len(plan)
    else:
        exact = False
        rng = np.random.default_rng(SEED)
        extreme, size = 0, resamples
        for start in range(0, resamples, 128):
            if cancel is not None and cancel.is_set():
                raise InputError('Statistical analysis cancelled.')
            plan = np.asarray([rng.permutation(weights) for _ in range(min(128, resamples-start))])
            null = np.einsum('ij,j->i', plan, values)
            extreme += int((np.abs(null) >= abs(observed) - tolerance).sum())
    pvalue = extreme / size if exact else (extreme + 1) / (size + 1)
    return dict(p_value=float(pvalue), bout_effect=observed, permutations=size, exact_test=exact)


STATISTICAL_METHOD = (
    'Exploratory within-recording association tests, not independent validation of clusters or population-level inference. '
    'SD is the sample SD of valid observations; the rest SD uses equal group weights with the weighted sample-variance correction. '
    'Medians are reported separately from means and SD. Bouts are continuous same-group source segments, not montage segments; '
    'missing observations never create extra bouts. Each bout contributes one median only when >=3 valid observations/pairs and '
    '>=50% within-bout valid coverage are available. Tests require >=5 valid current bouts and >=5 rest bouts in total, '
    'with at least one valid bout in every reliably tracked comparison group. The statistic is mean(current bout medians) minus '
    'the mean of the per-comparison-group means of bout medians. Bout group assignments are permuted with group bout counts fixed. '
    'Two-sided absolute-statistic test, exact when a two-weight assignment family has <=9999 allocations; otherwise 9999 random '
    'permutations with seed 20261102 and (extreme+1)/(9999+1) correction. All eligible feature p-values, including non-highlighted '
    'features, receive Benjamini–Yekutieli FDR correction within this selected group/settings run. q<=0.05 is marked exploratory '
    'significance. Correlated bouts, temporal confounding, post-selection from pose-derived clusters, repeated settings choices, '
    'and tests for other selected groups are not resolved by this correction. Independent recordings/animals are required for '
    'confirmatory biological inference.'
)
