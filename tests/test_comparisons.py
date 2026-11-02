import math
import threading
import unittest

import numpy as np

from boute.comparisons import GroupComparisons
from boute.core import InputError


class ComparisonTests(unittest.TestCase):
    def builder(self, lengths=(8, 8, 8), target=0):
        groups = np.repeat(np.arange(len(lengths)), lengths)
        return GroupComparisons(groups, groups[1:] == groups[:-1], target)

    def add(self, builder, levels, key='feature', family='movement', subject='Leg', kind='body_motion', resolution=.1, motion=True, excluded=()):
        groups = builder.groups[1:] if motion else builder.groups
        values = np.asarray(levels, dtype=float)[groups]
        valid = ~np.isin(groups, excluded)
        observations = ~np.isin(builder.groups, excluded)
        builder.add(key, family, subject, kind, values, valid, observations, 'px/s' if motion else 'px', resolution, motion=motion)

    def test_activity_and_stability_are_relative_to_other_groups(self):
        builder = self.builder()
        self.add(builder, [0, 8, 8], key='leg', subject='Leg')
        self.add(builder, [20, 2, 2], key='antenna', subject='Antenna')
        result = builder.finish()
        by_subject = {row['subject']: row for row in result['insights']}
        self.assertIn('Unusually stable', by_subject['Leg']['title'])
        self.assertIn('Unusually active', by_subject['Antenna']['title'])
        self.assertEqual(by_subject['Leg']['other_groups_value'], 8)
        self.assertEqual(by_subject['Leg']['comparison_groups'], 2)
        self.assertEqual(by_subject['Leg']['valid_samples'], 7)
        self.assertEqual(by_subject['Leg']['rank_effect'], -1)

    def test_pair_relationship_and_posture_contrasts(self):
        builder = self.builder()
        self.add(builder, [0, 5, 5], family='relationships', kind='pair_motion', subject='Leg ↔ Thorax')
        self.add(builder, [30, 10, 10], key='posture', family='posture', kind='pair_posture', subject='Head ↔ Thorax', motion=False, resolution=1.)
        insights = builder.finish()['insights']
        self.assertTrue(any('More stable relationship' in row['title'] for row in insights))
        self.assertTrue(any('Farther apart' in row['title'] for row in insights))

    def test_group_duration_does_not_dominate_baseline(self):
        builder = self.builder((8, 100, 8))
        self.add(builder, [15, 1, 9])
        insight = builder.finish()['insights'][0]
        self.assertEqual(insight['other_groups_value'], 5)
        self.assertEqual(insight['comparison_groups'], 2)

    def test_identical_groups_do_not_get_false_unique_insights(self):
        builder = self.builder()
        self.add(builder, [0, 0, 0])
        self.assertFalse(builder.finish()['insights'])
        self.add(builder, [2, 2, 2], key='same')
        self.assertFalse(builder.finish()['insights'])

    def test_small_or_overlapping_differences_are_not_promoted(self):
        builder = self.builder((8, 8))
        values = np.zeros(15)
        values[builder.motion[0]] = [0, 0, 0, 3, 100, 100, 100]
        values[builder.motion[1]] = [0, 0, 0, 1, 100, 100, 100]
        builder.add('overlap', 'movement', 'Leg', 'body_motion', values, np.ones(15, bool), np.ones(16, bool), 'px/s', .1)
        self.assertFalse(builder.finish()['insights'])
        builder = self.builder()
        self.add(builder, [.00001, 0, 0])
        self.assertFalse(builder.finish()['insights'])

    def test_low_coverage_and_short_groups_cannot_generate_contrasts(self):
        builder = self.builder((4, 8, 8))
        self.add(builder, [0, 10, 10])
        self.assertFalse(builder.finish()['insights'])
        builder = self.builder()
        self.add(builder, [0, 10, 10], excluded=(0,))
        result = builder.finish()
        self.assertFalse(result['insights'])
        self.assertFalse(result['similarities'])

    def test_similarity_distance_has_known_value_and_identical_score_is_100(self):
        builder = self.builder()
        for i, family in enumerate(('movement', 'relationships', 'posture')):
            self.add(builder, [1, 1, 10], key=str(i), family=family, subject=str(i))
        result = builder.finish()['similarities']
        self.assertEqual([row['group_id'] for row in result], [1, 2])
        self.assertEqual(result[0]['distance'], 0)
        self.assertEqual(result[0]['similarity_score'], 100)
        # Across-group IQR of [1,1,10] is 4.5. Difference 9 / 4.5 = 2.
        self.assertAlmostEqual(result[1]['distance'], 2)
        self.assertAlmostEqual(result[1]['similarity_score'], 100/3)
        self.assertEqual(result[1]['shared_features'], 3)

    def test_families_receive_equal_weight_despite_more_pair_features(self):
        distances = []
        for extra in (0, 100):
            builder = self.builder((8, 8))
            for i in range(2 + extra):
                self.add(builder, [1, 2], key=f'm{i}', family='movement', subject=str(i))
            self.add(builder, [10, 10], key='posture', family='posture', subject='Pair', motion=False, resolution=1)
            distances.append(builder.finish()['similarities'][0]['distance'])
        self.assertAlmostEqual(distances[0], math.sqrt(2))
        self.assertAlmostEqual(distances[0], distances[1])

    def test_similarity_omits_missing_data_instead_of_filling_with_zeros(self):
        builder = self.builder()
        for i, family in enumerate(('movement', 'posture', 'relationships')):
            self.add(builder, [1, 1, 10], key=str(i), family=family, excluded=(1,))
        rows = builder.finish()['similarities']
        self.assertEqual([row['group_id'] for row in rows], [2])
        self.assertEqual(rows[0]['feature_coverage_percent'], 100)

    def test_similarity_requires_three_shared_features_and_family_coverage(self):
        builder = self.builder()
        self.add(builder, [1, 2, 3])
        self.assertFalse(builder.finish()['similarities'])
        for i in range(3):
            self.add(builder, [1, 2, 3], key=f'm{i}', subject=str(i))
        # A second available family is required to be shared when it exists.
        self.add(builder, [1, 2, 3], key='p', family='posture', subject='Pair', excluded=(1,))
        self.assertEqual([row['group_id'] for row in builder.finish()['similarities']], [2])

    def test_contrast_subjects_are_not_repeated_and_limit_is_four(self):
        builder = self.builder()
        for i in range(10):
            self.add(builder, [0, 10+i, 10+i], key=str(i), subject=f'Part {i//2}')
        insights = builder.finish()['insights']
        self.assertEqual(len(insights), 4)
        self.assertEqual(len({row['subject'] for row in insights}), 4)

    def test_no_comparison_groups_and_cancellation(self):
        builder = self.builder((8,))
        self.add(builder, [1])
        result = builder.finish()
        self.assertFalse(result['insights'])
        self.assertFalse(result['similarities'])
        self.assertIn('No other groups', result['insight_status'])
        builder.cancel = threading.Event()
        builder.cancel.set()
        with self.assertRaises(InputError):
            self.add(builder, [1])
