import math
import threading
import unittest

import numpy as np

from boute.comparisons import GroupComparisons
from boute.core import InputError
from boute.statistics import mean_sd, balanced_mean_sd, fdr_by, permutation_test


class StatisticsTests(unittest.TestCase):
    def builder(self, repeats=6, bout_frames=8):
        groups = np.tile(np.repeat([0, 1], bout_frames), repeats)
        return GroupComparisons(groups, groups[1:] == groups[:-1], 0)

    def add(self, builder, key='motion', different=True, motion=True, missing=None):
        source = builder.groups[1:] if motion else builder.groups
        values = np.where(source == 0, 0., 10. if different else 0.)
        usable = np.ones(len(source), bool) if missing is None else missing
        observations = np.ones(len(builder.groups), bool)
        builder.add(key, 'movement', key, 'body_motion', values, usable, observations, 'px/s', .1, motion=motion)

    def test_sample_sd_is_not_population_sd(self):
        mean, sd = mean_sd([1, 2, 3])
        self.assertEqual(mean, 2)
        self.assertEqual(sd, 1)
        self.assertEqual(mean_sd([]), (None, None))
        self.assertEqual(mean_sd([1]), (1, None))

    def test_comparison_sd_gives_groups_equal_weight(self):
        mean, sd = balanced_mean_sd([np.array([1., 1.]), np.array([3., 3., 3.])])
        self.assertEqual(mean, 2)
        self.assertAlmostEqual(sd, math.sqrt(24/19))
        self.assertEqual(balanced_mean_sd([]), (None, None))

    def test_exact_permutation_matches_hand_counted_allocations(self):
        result = permutation_test([0, 0], [[1, 1]])
        # Two of six choices put equal observations together at either extreme.
        self.assertEqual(result['permutations'], 6)
        self.assertTrue(result['exact_test'])
        self.assertAlmostEqual(result['p_value'], 1/3)
        result = permutation_test([0] * 5, [[1] * 5])
        self.assertEqual(result['permutations'], 252)
        self.assertAlmostEqual(result['p_value'], 2/252)

    def test_identical_bouts_give_p_one(self):
        result = permutation_test([2] * 6, [[2] * 6])
        self.assertEqual(result['p_value'], 1)

    def test_monte_carlo_is_reproducible_and_never_returns_zero(self):
        a = permutation_test(np.arange(20), [np.arange(20) + 100], resamples=199)
        b = permutation_test(np.arange(20), [np.arange(20) + 100], resamples=199)
        self.assertEqual(a, b)
        self.assertFalse(a['exact_test'])
        self.assertEqual(a['permutations'], 199)
        self.assertGreaterEqual(a['p_value'], 1/200)

    def test_large_bout_plans_stream_without_changing_reproducibility(self):
        result = permutation_test(np.arange(150), [np.arange(150) + 500], resamples=129)
        self.assertEqual(result['permutations'], 129)
        self.assertEqual(result['p_value'], 1/130)
        self.assertFalse(result['exact_test'])

    def test_weighted_comparison_statistic_uses_group_bout_means(self):
        result = permutation_test([10] * 5, [[0] * 5, [4] * 10], resamples=19)
        self.assertAlmostEqual(result['bout_effect'], 8)

    def test_by_known_values_and_order(self):
        expected = [1/24, 1/12, 1/12, 1/60]
        np.testing.assert_allclose(fdr_by([.01, .04, .03, .002]), expected)
        self.assertEqual(fdr_by([]).size, 0)
        np.testing.assert_equal(fdr_by([1, 1]), [1, 1])
        with self.assertRaises(ValueError):
            fdr_by([float('nan')])

    def test_frame_count_does_not_create_bout_replication(self):
        groups = np.repeat([0, 1], 1000)
        builder = GroupComparisons(groups, groups[1:] == groups[:-1], 0)
        self.add(builder)
        report = builder.finish()
        insight = report['insights'][0]
        self.assertEqual(insight['group_bouts'], 1)
        self.assertEqual(insight['comparison_bouts'], 1)
        self.assertIsNone(insight['p_value'])
        self.assertIsNone(insight['q_value'])
        self.assertEqual(report['tested_features'], 0)
        self.assertEqual(insight['group_sd'], 0)

    def test_valid_bouts_receive_p_and_q(self):
        builder = self.builder()
        self.add(builder)
        report = builder.finish()
        insight = report['insights'][0]
        self.assertEqual(insight['group_bouts'], 6)
        self.assertEqual(insight['comparison_bouts'], 6)
        self.assertAlmostEqual(insight['p_value'], 2/924)
        self.assertAlmostEqual(insight['q_value'], insight['p_value'])
        self.assertTrue(insight['significant'])

    def test_correction_includes_non_highlighted_eligible_features(self):
        builder = self.builder()
        self.add(builder, 'different')
        self.add(builder, 'constant', different=False)
        report = builder.finish()
        self.assertEqual(report['tested_features'], 2)
        self.assertEqual(len(report['insights']), 1)
        insight = report['insights'][0]
        self.assertAlmostEqual(insight['q_value'], insight['p_value'] * 3)
        self.assertEqual(report['feature_statistics'][1]['p_value'], 1)

    def test_bout_effect_direction_is_not_confused_with_frame_median_direction(self):
        groups = np.r_[np.zeros(200, int), np.ones(6, int), np.tile(np.repeat([0, 1], 6), 7)]
        builder = GroupComparisons(groups, groups[1:] == groups[:-1], 0)
        values = np.where(groups == 0, 20., 5.)
        values[:200] = 0
        builder.add('posture', 'posture', 'Pair', 'pair_posture', values,
                    np.ones(len(groups), bool), np.ones(len(groups), bool), 'px', 1., motion=False)
        insight = builder.finish()['insights'][0]
        self.assertEqual(insight['group_value'], 0)
        self.assertGreater(insight['bout_effect'], 0)
        self.assertFalse(insight['direction_consistent'])
        self.assertFalse(insight['supports_insight'])

    def test_missing_tracking_does_not_split_one_bout_into_many(self):
        groups = np.repeat([0, 1], 100)
        builder = GroupComparisons(groups, groups[1:] == groups[:-1], 0)
        usable = np.arange(200) % 2 == 0
        self.add(builder, motion=False, missing=usable)
        stat = builder.finish()['feature_statistics'][0]
        self.assertEqual(stat['group_bouts'], 1)
        self.assertEqual(stat['comparison_bouts'], 1)
        self.assertIsNone(stat['p_value'])

    def test_bouts_need_enough_valid_observations(self):
        builder = self.builder(bout_frames=3)
        self.add(builder)
        stat = builder.finish()['feature_statistics'][0]
        self.assertEqual(stat['group_bouts'], 0)  # Only two motion pairs per bout.
        self.assertIsNone(stat['p_value'])

    def test_invalid_inputs_and_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InputError):
            permutation_test([0] * 5, [[1] * 5], cancel=cancel)
        for current, others in (([], [[1]]), ([1], [[]]), ([float('nan')], [[1]])):
            with self.subTest(current=current), self.assertRaises(ValueError):
                permutation_test(current, others)
        with self.assertRaises(ValueError):
            permutation_test([0], [[1]], resamples=0)
