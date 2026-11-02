import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np

from boute.core import InputError, VideoInfo, generate, inspect_video, merge_results
from boute.pose import (PoseData, read_pose, align_pose, save_pose, load_pose, valid_points,
                        analyze_pose, draw_pose, montage_indices, export_overlay, export_report)


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def pose(self, positions, confidence=None):
        xy = np.asarray(positions, dtype=float)
        return PoseData(tuple(f'p{i}' for i in range(xy.shape[1])), xy,
                        np.ones(xy.shape[:2]) if confidence is None else np.asarray(confidence, dtype=float), 'source.csv', 'test alignment')

    def csv(self, text):
        path = self.root / 'pose.csv'
        path.write_text(text, encoding='utf-8')
        return path

    def test_combined_csv_headers_do_not_become_observations(self):
        path = self.csv(',B-SOiD labels,scorer,model,model,model,model,model,model\n'
                        '0,21,bodyparts,Head,Head,Head,Leg,Leg,Leg\n'
                        '1,21,coords,x,y,likelihood,x,y,likelihood\n'
                        '2,21,0,10,20,.9,30,40,.8\n3,3,1,11,21,.7,,bad,.4\n')
        pose = read_pose(path)
        self.assertEqual(pose.names, ('Head', 'Leg'))
        self.assertEqual(pose.xy.shape, (2, 2, 2))
        np.testing.assert_allclose(pose.xy[0], [[10, 20], [30, 40]])
        self.assertTrue(np.isnan(pose.xy[1, 1]).all())
        np.testing.assert_allclose(pose.confidence, [[.9, .8], [.7, .4]])

    def test_standard_and_multi_animal_dlc_csv(self):
        pose = read_pose(self.csv('scorer,m,m,m\nbodyparts,nose,nose,nose\ncoords,x,y,likelihood\n0,5,6,.9\n'))
        self.assertEqual(pose.names, ('nose',))
        pose = read_pose(self.csv('scorer,m,m,m,m,m,m\nindividuals,fly1,fly1,fly1,fly2,fly2,fly2\n'
                                  'bodyparts,nose,nose,nose,nose,nose,nose\ncoords,x,y,likelihood,x,y,likelihood\n0,5,6,.9,7,8,.7\n'))
        self.assertEqual(pose.names, ('fly1 / nose', 'fly2 / nose'))

    def test_no_pose_and_ambiguous_headers(self):
        self.assertIsNone(read_pose(self.csv('B-SOiD labels\n1\n2\n')))
        with self.assertRaises(InputError):
            read_pose(self.csv('scorer,m,m,m,m,m,m\nbodyparts,n,n,n,n,n,n\ncoords,x,y,likelihood,x,y,likelihood\n0,1,2,1,1,2,1\n'))
        with self.assertRaises(InputError):
            read_pose(self.csv('scorer,m,m\nbodyparts,n,n\ncoords,x,y\n0,1,2\n'))

    def test_sparse_or_duplicated_frame_indices_are_rejected(self):
        header = 'scorer,m,m,m\nbodyparts,n,n,n\ncoords,x,y,likelihood\n'
        for rows in ('0,1,2,1\n2,2,3,1\n', '0,1,2,1\n0,2,3,1\n', 'frame.png,1,2,1\n'):
            with self.subTest(rows=rows), self.assertRaises(InputError):
                read_pose(self.csv(header + rows))

    def test_alignment_rate_offset_and_portable_archive(self):
        pose = self.pose(np.arange(40).reshape(20, 1, 2))
        aligned = align_pose(pose, VideoInfo(5, 5, 100, 100), 10, .5)
        np.testing.assert_array_equal(aligned.xy, pose.xy[[5, 7, 9, 11, 13]])
        save_pose(self.root, aligned)
        loaded = load_pose(self.root)
        self.assertEqual(loaded.names, aligned.names)
        self.assertEqual(loaded.alignment, aligned.alignment)
        np.testing.assert_equal(loaded.confidence, aligned.confidence)
        with self.assertRaises(InputError):
            align_pose(pose, VideoInfo(10, 10, 100, 100), 10, 2)

    def test_motion_never_crosses_group_boundaries_or_montage_jumps(self):
        positions = [[[x, 20], [x+10, 20]] for x in [10, 11, 80, 100, 101]]
        report = analyze_pose(self.pose(positions), np.array([0, 0, 1, 0, 0]), VideoInfo(10, 5, 200, 100), 0, smoothing=1)
        self.assertEqual(report['eligible_motion_pairs'], 2)
        self.assertEqual(report['body_parts'][1]['median_speed_px_s'], 10)
        self.assertEqual(report['body_parts'][1]['body_speed_px_s'], 0)
        self.assertEqual(report['pairs'][0]['median_distance_change_px_s'], 0)

    def test_lower_rate_pose_samples_are_not_counted_as_false_stability(self):
        raw = self.pose([[[10, 10], [20, 20]], [[11, 10], [21, 20]], [[12, 10], [22, 20]]])
        info = VideoInfo(4, 9, 100, 100)
        aligned = align_pose(raw, info, 1, 0)
        report = analyze_pose(aligned, np.zeros(9), info, 0, smoothing=1)
        self.assertEqual(report['frame_count'], 9)
        self.assertEqual(report['pose_observations'], 3)
        self.assertEqual(report['eligible_motion_pairs'], 2)
        self.assertEqual(report['body_parts'][0]['median_speed_px_s'], 1)
        # A group boundary hidden between low-rate observations still breaks continuity.
        groups = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0])
        report = analyze_pose(aligned, groups, info, 0, smoothing=1)
        self.assertEqual(report['eligible_motion_pairs'], 1)
        self.assertEqual(report['body_parts'][0]['median_speed_px_s'], 1)

    def test_missing_and_low_confidence_are_not_reported_as_stability(self):
        pose = self.pose([[[10, 10], [20, 20]], [[11, 10], [21, 20]], [[12, 10], [22, 20]]], [[1, .1], [1, np.nan], [1, .2]])
        report = analyze_pose(pose, np.zeros(3), VideoInfo(10, 3, 100, 100), 0, smoothing=1)
        part = report['body_parts'][1]
        self.assertEqual(part['coverage_percent'], 0)
        self.assertEqual(part['valid_motion_pairs'], 0)
        self.assertIsNone(part['median_speed_px_s'])
        self.assertIsNone(part['body_speed_px_s'])
        self.assertIsNone(report['pairs'][0]['median_distance_change_px_s'])

    def test_invalid_confidence_and_offscreen_points_are_excluded(self):
        pose = self.pose([[[10, 10]], [[-1, 0]], [[5, 5]], [[np.nan, 2]], [[10, 10]]], [[1], [1], [1.2], [1], [-.1]])
        np.testing.assert_array_equal(valid_points(pose, .8, VideoInfo(10, 5, 100, 100))[:, 0], [True, False, False, False, False])
        with self.assertRaises(InputError):
            valid_points(pose, float('nan'), VideoInfo(10, 5, 100, 100))

    def test_axis_removes_rigid_rotation_and_translation(self):
        positions = []
        for theta in [0, .3, .6, .9]:
            origin = np.array([50+theta, 50])
            direction = np.array([np.cos(theta), np.sin(theta)])
            positions.append([origin, origin+10*direction, origin+20*direction])
        pose = self.pose(positions)
        report = analyze_pose(pose, np.zeros(4), VideoInfo(10, 4, 100, 100), 0, reference=0, axis=1, smoothing=1)
        self.assertGreater(report['body_parts'][2]['median_speed_px_s'], 10)
        self.assertAlmostEqual(report['body_parts'][2]['body_speed_px_s'], 0, places=8)
        with self.assertRaises(InputError):
            analyze_pose(pose, np.zeros(4), VideoInfo(10, 4, 100, 100), 0, reference=0, axis=0)

    def test_comparison_baseline_uses_other_groups_continuous_motion(self):
        positions = [[[10, 10], [x, 20]] for x in [20, 21, 22, 30, 32, 34]]
        report = analyze_pose(self.pose(positions), np.array([0, 0, 0, 1, 1, 1]), VideoInfo(1, 6, 100, 100), 0, smoothing=1)
        self.assertAlmostEqual(report['body_parts'][1]['speed_ratio_vs_others'], .5)
        self.assertAlmostEqual(report['body_parts'][1]['body_speed_ratio_vs_others'], .5)

    def test_median_smoothing_does_not_cross_missing_samples_or_group_runs(self):
        pose = self.pose([[[x, 20]] for x in [10, 11, 90, 12, 13]])
        report = analyze_pose(pose, np.array([0, 0, 1, 0, 0]), VideoInfo(1, 5, 100, 100), 0, smoothing=3)
        self.assertEqual(report['body_parts'][0]['median_speed_px_s'], 1)
        pose = self.pose([[[x, 20]] for x in [10, 90, 10, 10, 10]])
        report = analyze_pose(pose, np.zeros(5), VideoInfo(1, 5, 100, 100), 0, smoothing=3)
        self.assertEqual(report['body_parts'][0]['median_speed_px_s'], 0)

    def test_joint_angle_and_pair_distance_change(self):
        pose = self.pose([[[20, 10], [10, 10], [10, 20]], [[21, 10], [10, 10], [10, 21]], [[22, 10], [10, 10], [10, 22]]])
        report = analyze_pose(pose, np.zeros(3), VideoInfo(10, 3, 100, 100), 0, smoothing=1, angle=(0, 1, 2))
        self.assertEqual(report['angles'][0]['median_angle_deg'], 90)
        self.assertEqual(report['angles'][0]['median_angular_speed_deg_s'], 0)
        self.assertEqual(report['pairs'][0]['median_distance_change_px_s'], 10)
        with self.assertRaises(InputError):
            analyze_pose(pose, np.zeros(3), VideoInfo(10, 3, 100, 100), 0, angle=(0, 0, 1))

    def test_single_frame_group_no_motion_and_cancel(self):
        pose = self.pose([[[10, 10]], [[20, 20]], [[30, 30]]])
        report = analyze_pose(pose, np.array([0, 1, 0]), VideoInfo(10, 3, 100, 100), 1)
        self.assertEqual(report['eligible_motion_pairs'], 0)
        self.assertIsNone(report['body_parts'][0]['median_speed_px_s'])
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InputError):
            analyze_pose(pose, np.zeros(3), VideoInfo(10, 3, 100, 100), 0, cancel=cancel)

    def test_overlay_selects_parts_confidence_and_source_frame_without_mutating(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pose = self.pose([[[20, 20], [50, 50]], [[30, 30], [60, 60]]], [[1, .1], [1, 1]])
        rendered = draw_pose(frame, pose, 0, [0, 1], .8)
        self.assertTrue(rendered[20, 20].any())
        self.assertFalse(rendered[50, 50].any())
        self.assertFalse(frame.any())
        later = draw_pose(frame, pose, 1, [1], .8)
        self.assertTrue(later[60, 60].any())
        self.assertFalse(later[30, 30].any())
        np.testing.assert_equal(draw_pose(frame, pose, 0, [], .8), frame)

    def make_results(self):
        video = self.root / 'video.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100, 100))
        for _ in range(6):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()
        labels = np.array([0, 1, 0, 1, 0, 1])
        path = self.csv('B-SOiD labels\n' + '\n'.join(map(str, labels)))
        info = inspect_video(video)
        output = generate(video, path, [], labels, labels, info, self.root / 'out', 10, 0, lambda *_: None, threading.Event())
        return output, labels, info

    def test_merge_preserves_aligned_pose_and_reindexes_montage(self):
        output, labels, info = self.make_results()
        pose = self.pose([[[20+i*5, 20]] for i in range(6)])
        save_pose(output, pose)
        merged = merge_results(output, [0, 1], info, lambda *_: None, threading.Event())
        np.testing.assert_equal(load_pose(merged).xy, pose.xy)
        np.testing.assert_equal(montage_indices(output, '0'), [0, 2, 4])
        np.testing.assert_equal(montage_indices(merged, '0'), list(range(6)))
        report = analyze_pose(load_pose(merged), np.zeros(6), info, 0, smoothing=1)
        self.assertEqual(report['eligible_motion_pairs'], 5)

    def test_annotated_export_alignment_and_cancel_cleanup(self):
        output, _, _ = self.make_results()
        pose = self.pose([[[20+i*5, 20]] for i in range(6)])
        indices = montage_indices(output, '0')
        source = output / 'group_0.mp4'
        original_bytes = source.read_bytes()
        target = self.root / 'annotated.mp4'
        export_overlay(source, target, pose, indices, [0], .8, False, lambda *_: None, threading.Event())
        self.assertEqual(inspect_video(target).frames, 3)
        cap = cv2.VideoCapture(str(target))
        for x in (20, 30, 40):
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertGreater(frame[20, x].max(), 50)
        cap.release()
        self.assertEqual(source.read_bytes(), original_bytes)
        cancel = threading.Event()
        cancel.set()
        retained = target.read_bytes()
        with self.assertRaises(InputError):
            export_overlay(source, target, pose, indices, [0], .8, False, lambda *_: None, cancel)
        self.assertEqual(target.read_bytes(), retained)
        self.assertFalse(list(self.root.glob('tmp*.mp4')))

    def test_report_contains_settings_units_coverage_and_valid_samples(self):
        pose = self.pose([[[10, 10]], [[11, 10]]])
        report = analyze_pose(pose, np.zeros(2), VideoInfo(10, 2, 100, 100), 0)
        output = export_report(self.root, report, 'Possible grooming')
        settings = json.loads((output / 'analysis_settings.json').read_text())
        self.assertEqual(settings['group_name'], 'Possible grooming')
        self.assertIn('pixels', settings['units'])
        self.assertEqual(settings['eligible_motion_pairs'], 1)
        with (output / 'body_parts.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['coverage_percent'], '100.0')
        self.assertEqual(row['valid_motion_pairs'], '1')
