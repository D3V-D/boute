import csv
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from boute.core import InputError, generate, inspect_video, merge_results, rename_group
from boute.timeline import read_jumps


class IncrementalMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(20):
            writer.write(np.full((48, 64, 3), 15 + index * 10, dtype=np.uint8))
        writer.release()
        self.mapped = np.array([3, 21, 10, 21] * 5)
        labels = np.append(self.mapped, [0, 99])
        self.csv = self.root / 'labels.csv'
        self.csv.write_text('B-SOiD labels\n' + '\n'.join(map(str, labels)))
        self.info = inspect_video(self.video)
        self.previous = generate(self.video, self.csv, [], labels, self.mapped, self.info,
                                 self.root / 'out', 10, 0, lambda *_: None, threading.Event())

    def merge(self, selected, previous=None, cancel=None):
        return merge_results(previous or self.previous, selected, self.info, lambda *_: None,
                             cancel or threading.Event())

    def rows(self, output, name):
        with (output / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_only_selected_videos_decoded_one_video_encoded_and_others_copied(self):
        # Merging must work even after the original inputs become unavailable.
        self.video.unlink()
        self.csv.unlink()
        capture_factory = cv2.VideoCapture
        writer_factory = cv2.VideoWriter
        captures = []

        def capture(path):
            cap = Mock(wraps=capture_factory(path))
            captures.append((Path(path), cap))
            return cap

        with patch('boute.core.cv2.VideoCapture', side_effect=capture), patch(
            'boute.core.cv2.VideoWriter', wraps=writer_factory
        ) as writers:
            output = self.merge([3, 21])
        self.assertEqual(writers.call_count, 1)
        self.assertEqual(Path(writers.call_args.args[0]), output / 'group_3.mp4')
        self.assertEqual({path for path, cap in captures if cap.read.call_count},
                         {self.previous / 'group_3.mp4', self.previous / 'group_21.mp4'})
        self.assertEqual(sum(cap.read.call_count for _, cap in captures), 15)
        self.assertNotIn(self.previous / 'group_10.mp4', {path for path, _ in captures})
        self.assertEqual((output / 'group_10.mp4').read_bytes(),
                         (self.previous / 'group_10.mp4').read_bytes())
        self.assertTrue((self.previous / 'group_21.mp4').exists())
        self.assertFalse((output / 'group_21.mp4').exists())
        manifest = self.rows(output, 'frame_manifest.csv')
        self.assertEqual([int(r['original_video_frame']) for r in manifest], list(range(20)))
        merged = [r for r in manifest if r['group_id'] == '3']
        self.assertEqual([int(r['exported_group_frame']) for r in merged], list(range(15)))
        indices = np.flatnonzero(self.mapped != 10)
        cap = cv2.VideoCapture(str(output / 'group_3.mp4'))
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame.mean())
        cap.release()
        np.testing.assert_allclose(decoded, 15 + indices * 10, atol=8)
        summary = {r['group_id']: r for r in self.rows(output, 'group_summary.csv')}
        self.assertEqual(int(summary['3']['frame_count']), 15)
        self.assertEqual(float(summary['3']['percentage']), 75)
        self.assertEqual(float(summary['3']['duration_seconds']), 1.5)
        self.assertEqual(len(read_jumps(output / 'frame_manifest.csv')['3']), 5)
        self.assertEqual((output / 'source_metadata.txt').read_bytes(),
                         (self.previous / 'source_metadata.txt').read_bytes())
        self.assertTrue((output / 'source_csvs' / 'labels.csv').exists())

    def test_repeated_merge_with_zero_frame_target_and_provenance(self):
        first = self.merge([3, 21])
        second = self.merge([0, 3], first)
        final = self.merge([0, 10], second)
        self.assertEqual(inspect_video(final / 'group_0.mp4').frames, 20)
        mapping = {int(r['original_group_id']): int(r['merged_group_id'])
                   for r in self.rows(final, 'group_mapping.csv')}
        self.assertEqual(mapping, {0: 0, 3: 0, 10: 0, 21: 0, 99: 99})
        self.assertFalse(read_jumps(final / 'frame_manifest.csv'))
        self.assertEqual([r['exported_group_frame'] for r in self.rows(final, 'frame_manifest.csv')],
                         list(map(str, range(20))))

    def test_names_follow_unchanged_and_merged_groups(self):
        rename_group(self.previous, '3', 'Grooming?')
        rename_group(self.previous, '21', 'Possible grooming')
        rename_group(self.previous, '10', 'Walking')
        output = self.merge([3, 21])
        names = {r['group_id']: r['group_name'] for r in self.rows(output, 'group_summary.csv')}
        self.assertEqual(names['3'], 'Grooming?')
        self.assertEqual(names['10'], 'Walking')
        self.assertNotIn('21', names)
        rename_group(output, '3', 'Grooming')
        original = {r['group_id']: r['group_name'] for r in self.rows(self.previous, 'group_summary.csv')}
        self.assertEqual(original['3'], 'Grooming?')
        final = self.merge([3, 10], output)
        names = {r['group_id']: r['group_name'] for r in self.rows(final, 'group_summary.csv')}
        self.assertEqual(names['3'], 'Grooming')

    def test_empty_groups_merge_without_encoder(self):
        with patch('boute.core.cv2.VideoWriter') as writer:
            output = self.merge([0, 99])
        writer.assert_not_called()
        summary = {r['group_id']: r for r in self.rows(output, 'group_summary.csv')}
        self.assertEqual(summary['0']['frame_count'], '0')
        self.assertNotIn('99', summary)
        self.assertFalse((output / 'group_0.mp4').exists())

    def test_cancel_and_decode_failure_leave_previous_results_and_no_partial_run(self):
        existing = set(self.previous.parent.iterdir())
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(InputError, 'cancelled'):
            self.merge([3, 21], cancel=cancel)
        self.assertEqual(set(self.previous.parent.iterdir()), existing)
        capture_factory = cv2.VideoCapture
        def failing_capture(path):
            cap = Mock(wraps=capture_factory(path))
            cap.read.return_value = (False, None)
            return cap
        with patch('boute.core.cv2.VideoCapture', side_effect=failing_capture):
            with self.assertRaisesRegex(InputError, 'Could not decode'):
                self.merge([3, 21])
        self.assertEqual(set(self.previous.parent.iterdir()), existing)
        self.assertTrue((self.previous / 'group_3.mp4').exists())

    def test_invalid_selection_does_not_create_run(self):
        existing = set(self.previous.parent.iterdir())
        for selected in ([3], [3, 3], [3, 100]):
            with self.subTest(selected=selected), self.assertRaises(InputError):
                self.merge(selected)
        self.assertEqual(set(self.previous.parent.iterdir()), existing)
