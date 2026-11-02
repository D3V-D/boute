"""Descriptive cross-group contrasts and reliability-gated pose similarity.

Groups, rather than their durations, receive equal weight in comparisons.
Contrast and similarity scores are descriptive. Bout-level significance is
reported separately; correlated landmarks do not constitute replication.
"""
from __future__ import annotations

import math
import numpy as np

from .core import InputError
from .statistics import mean_sd, balanced_mean_sd, permutation_test, fdr_by, MIN_BOUTS, STATISTICAL_METHOD


class GroupComparisons:
    def __init__(self, groups, continuity, group_id, cancel=None):
        self.groups = np.asarray(groups)
        self.ids = sorted(set(map(int, groups)) | {int(group_id)})
        self.target = self.ids.index(int(group_id))
        self.frames = [np.flatnonzero(self.groups == group) for group in self.ids]
        self.motion = [np.flatnonzero(continuity & (self.groups[1:] == group)) for group in self.ids]
        self.cancel = cancel
        self.features = []
        self.candidates = []
        self.statistics = []
        self.bouts = [[] for _ in self.ids]
        if len(self.groups):
            starts = np.r_[0, np.flatnonzero(~continuity) + 1]
            ends = np.r_[starts[1:], len(self.groups)]
            for start, end in zip(starts, ends):
                self.bouts[self.ids.index(int(self.groups[start]))].append(np.arange(start, end))

    def add(self, key, family, subject, kind, values, usable, observation_valid,
            unit, resolution, motion=True):
        """Aggregate one physical feature using the same eligibility rules per group."""
        if self.cancel is not None and self.cancel.is_set():
            raise InputError('Group comparison cancelled.')
        samples, medians, coverages = [], [], []
        indices = self.motion if motion else self.frames
        for frame_indices, value_indices in zip(self.frames, indices):
            coverage = float(observation_valid[frame_indices].mean()) if len(frame_indices) else 0.
            good = usable[value_indices] & np.isfinite(values[value_indices])
            sample = values[value_indices][good]
            minimum = max(5, math.ceil(len(value_indices) * (.25 if motion else .5)))
            reliable = coverage >= .5 and len(sample) >= minimum
            samples.append(sample if reliable else np.empty(0))
            medians.append(float(np.median(sample)) if reliable else float('nan'))
            coverages.append(coverage)
        medians = np.asarray(medians)
        self.features.append(dict(key=key, family=family, values=medians, resolution=resolution))
        current = samples[self.target]
        others = [i for i in range(len(self.ids)) if i != self.target and len(samples[i])]
        bouts = []
        eligible_statistics = bool(len(current) and others)
        for group_index, runs in enumerate(self.bouts):
            medians_for_group = []
            if not eligible_statistics or not len(samples[group_index]):
                bouts.append(np.empty(0))
                continue
            for run in runs:
                positions = run[:-1] if motion else run
                if not len(positions) or observation_valid[run].mean() < .5:
                    continue
                good = usable[positions] & np.isfinite(values[positions])
                if good.sum() >= max(3, math.ceil(len(positions) * .5)):
                    medians_for_group.append(float(np.median(values[positions][good])))
            bouts.append(np.asarray(medians_for_group))
        current_mean, current_sd = mean_sd(current)
        rest_mean, rest_sd = balanced_mean_sd([samples[i] for i in others])
        current_bout_mean, current_bout_sd = mean_sd(bouts[self.target])
        rest_bout_mean, rest_bout_sd = balanced_mean_sd([bouts[i] for i in others])
        statistic = dict(feature=key, subject=subject, unit=unit,
                         group_mean=current_mean, group_sd=current_sd, other_groups_mean=rest_mean, other_groups_sd=rest_sd,
                         group_bout_mean=current_bout_mean, group_bout_sd=current_bout_sd,
                         other_groups_bout_mean=rest_bout_mean, other_groups_bout_sd=rest_bout_sd,
                         group_bouts=len(bouts[self.target]) if eligible_statistics else None,
                         comparison_bouts=sum(len(bouts[i]) for i in others) if eligible_statistics else None,
                         p_value=None, q_value=None, significant=None, bout_effect=None, permutations=0, exact_test=False,
                         test_status='Too few bouts' if eligible_statistics else 'No comparison data' if not others else 'Insufficient tracking')
        if (len(current) and others and len(bouts[self.target]) >= MIN_BOUTS and
                all(len(bouts[i]) for i in others) and sum(len(bouts[i]) for i in others) >= MIN_BOUTS):
            statistic.update(permutation_test(bouts[self.target], [bouts[i] for i in others], cancel=self.cancel))
            statistic['test_status'] = 'Exploratory bout permutation'
        self.statistics.append(statistic)
        others = [i for i in range(len(self.ids)) if i != self.target and len(samples[i])]
        if not len(current) or not others:
            return
        baseline = float(np.median(medians[others]))
        value = float(medians[self.target])
        change = value - baseline
        # Conservative descriptive contrast floors, expressed in physical units.
        # These avoid promoting floating-point residue and tiny relative changes.
        if abs(change) < resolution:
            return
        if motion and max(value, baseline) < 1.5 * max(min(value, baseline), resolution):
            return
        ordered = np.sort(current)
        probabilities = []
        for index in others:
            other = samples[index]
            left = np.searchsorted(ordered, other, side='left')
            right = np.searchsorted(ordered, other, side='right')
            # Probability that a current observation exceeds an observation from
            # this comparison group, with half credit for ties.
            probabilities.append(float(np.mean(1 - (left + right) / (2 * len(ordered)))))
        effect = 2 * float(np.mean(probabilities)) - 1
        if abs(effect) < .35 or effect * change <= 0:
            return
        group_values = medians[np.isfinite(medians)]
        scale = max(resolution, float(np.percentile(group_values, 75) - np.percentile(group_values, 25)))
        strength = abs(effect) * math.log1p(abs(change) / scale)
        high = change > 0
        if kind == 'body_motion':
            title = ('Unusually active: ' if high else 'Unusually stable: ') + subject
            measure = 'Body-relative speed'
            category = 'movement' if high else 'stability'
        elif kind == 'reference_motion':
            title = ('More image movement: ' if high else 'Less image movement: ') + subject
            measure = 'Reference landmark image speed'
            category = 'image movement' if high else 'image stability'
        elif kind == 'pair_motion':
            title = ('More relative movement: ' if high else 'More stable relationship: ') + subject
            measure = 'Change in landmark distance'
            category = 'relationship movement' if high else 'relationship stability'
        elif kind == 'pair_posture':
            title = ('Farther apart: ' if high else 'Closer together: ') + subject
            measure = 'Typical landmark distance'
            category = 'posture'
        elif kind == 'angle_motion':
            title = ('More joint-angle movement: ' if high else 'More stable joint angle: ') + subject
            measure = 'Angular speed'
            category = 'angle movement' if high else 'angle stability'
        else:
            title = ('Larger joint angle: ' if high else 'Smaller joint angle: ') + subject
            measure = 'Typical joint angle'
            category = 'angle posture'
        description = f'{measure}: {value:.2f} {unit}, compared with {baseline:.2f} in other groups.'
        if motion and baseline > 1e-8:
            description += f' {value / baseline:.2f}× their typical level.'
        self.candidates.append(dict(title=title, description=description, category=category,
                                    subject=subject, feature=key, strength=float(strength),
                                    rank_effect=float(effect), group_value=value,
                                    other_groups_value=baseline, unit=unit,
                                    valid_samples=len(current), coverage_percent=coverages[self.target] * 100,
                                    comparison_samples=sum(len(samples[i]) for i in others),
                                    comparison_groups=len(others), sample_kind='motion pairs' if motion else 'observations'))

    def finish(self):
        tested = [row for row in self.statistics if row['p_value'] is not None]
        for row, q in zip(tested, fdr_by([row['p_value'] for row in tested])):
            row['q_value'] = float(q)
            row['significant'] = bool(q <= .05)
        statistics = {row['feature']: row for row in self.statistics}
        selected, used_subjects, used_categories = [], set(), set()
        candidates = sorted(self.candidates, key=lambda item: (-item['strength'], item['feature']))
        # Prefer different findings, then fill remaining places without repeating
        # the same landmark/pair. A pair's posture and speed can be correlated.
        for diverse in (True, False):
            for candidate in candidates:
                if len(selected) >= 4:
                    break
                if candidate['subject'] in used_subjects:
                    continue
                if diverse and candidate['category'] in used_categories:
                    continue
                item = dict(candidate, **{key: value for key, value in statistics[candidate['feature']].items() if key not in ('feature', 'subject', 'unit')})
                item['direction_consistent'] = None if item['bout_effect'] is None else bool(item['bout_effect'] * (item['group_value'] - item['other_groups_value']) > 0)
                item['supports_insight'] = bool(item['significant'] and item['direction_consistent'])
                selected.append(item)
                used_subjects.add(candidate['subject'])
                used_categories.add(candidate['category'])
        similarities = []
        if self.features:
            matrix = np.stack([feature['values'] for feature in self.features], axis=1)
            feature_families = np.asarray([feature['family'] for feature in self.features])
            eligible = np.isfinite(matrix).sum(axis=0) >= 2
            current_available = np.isfinite(matrix[self.target]) & eligible
            current_features = int(current_available.sum())
            scales = np.ones(len(self.features))
            for i, feature in enumerate(self.features):
                values = matrix[:, i][np.isfinite(matrix[:, i])]
                if len(values) >= 2:
                    scales[i] = max(feature['resolution'], float(np.percentile(values, 75) - np.percentile(values, 25)))
            current_families = len(set(feature_families[current_available]))
            for index, group in enumerate(self.ids):
                if index == self.target:
                    continue
                common = current_available & np.isfinite(matrix[index])
                shared = int(common.sum())
                families = sorted(set(feature_families[common]))
                if shared < 3 or shared < current_features * .5 or len(families) < min(2, current_families):
                    continue
                normalized = (matrix[self.target] - matrix[index]) / scales
                family_errors = [float(np.mean(normalized[common & (feature_families == family)] ** 2)) for family in families]
                distance = float(np.sqrt(np.mean(family_errors)))
                similarities.append(dict(group_id=group, distance=distance, similarity_score=100 / (1 + distance),
                                          shared_features=shared, available_features=current_features,
                                          feature_coverage_percent=shared / max(1, current_features) * 100,
                                          feature_families=', '.join(families)))
        similarities.sort(key=lambda item: (item['distance'], item['group_id']))
        for rank, row in enumerate(similarities, 1):
            row['rank'] = rank
        return dict(insights=selected, similarities=similarities, feature_statistics=self.statistics,
                    statistical_method=STATISTICAL_METHOD, tested_features=len(tested),
                    insight_status=('No clear differences met the coverage and contrast requirements at this confidence threshold.' if not selected and len(self.ids) > 1
                                    else 'No other groups are available for comparison.' if not selected else ''),
                    similarity_status=('No other groups are available for comparison.' if len(self.ids) < 2 else 'Not enough shared reliable features to rank other groups.') if not similarities else '')


COMPARISON_METHOD = (
    'Group medians give each group equal weight, independent of duration. Features require >=50% valid observations and '
    '>=5 valid samples; motion features also require >=25% eligible continuous pairs. Contrasts require absolute group-balanced '
    'rank effect >=0.35, a median difference >=0.1 px/s (motion), 1 px (distance), 1 deg/s (angular speed), or 5 deg (angle), '
    'and >=1.5-fold contrast for motion. Strength=abs(rank effect)*log1p(abs(median difference)/max(across-group median IQR, unit floor)). '
    'Up to four different subjects are shown, preferring different contrast categories. '
    'Similarity uses per-group medians for landmark body speed/reference image speed, pair distance, pair distance-change speed, '
    'and the selected angle/angle speed. Each feature is divided by its across-group IQR, floored at its contrast unit floor. '
    'Distance=sqrt(mean over shared feature families of mean squared standardized differences within that family)). '
    'Families have equal weight: movement, posture, relationships, and selected joint angle. At least three shared features, '
    '50% of the current group reliable features, and two families (when available) are required. '
    'Similarity score=100/(1+distance), with 100 indicating identical shared features. Scores are descriptive, not probabilities '
    'or significance tests; correlated frames/landmarks are not independent replicates. These metrics describe the same pose data used to form the groups.'
)
