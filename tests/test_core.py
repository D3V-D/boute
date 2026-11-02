import csv
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from boute.core import AlignmentRequired, InputError, VideoInfo, align, classify, filename_rate, generate, inspect_video, merge_groups, read_labels


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv(self, text, name='labels_pose_20Hz.csv'):
        path = self.root / name
        path.write_text(text)
        return path

    def test_dlc_headers(self):
        path = self.csv('scorer,B-SOiD labels,DLC_model,DLC_model\nbodyparts,,head,head\ncoords,,x,y\n0,3,4,5\n1,21,6,7\n2,-1,8,9\n')
        np.testing.assert_array_equal(read_labels(path), [3, 21, -1])
        self.assertEqual(filename_rate(path), 20)

    def test_streamlit_numeric_labels_on_metadata_rows(self):
        path = self.csv(',B-SOiD labels,scorer,DLC_model\n0,25,bodyparts,head\n1,25,coords,x\n2,3,0,42\n3,21,1,43\n')
        np.testing.assert_array_equal(read_labels(path), [3, 21])

    def test_invalid_labels(self):
        for text in ['a,b\n1,2', 'B-SOiD labels\n3\n\n', 'B-SOiD labels\n\n3', 'B-SOiD labels\n1.5', 'B-SOiD labels\nnan', 'B-SOiD labels\n']:
            with self.subTest(text=text), self.assertRaises(InputError):
                read_labels(self.csv(text))

    def test_alignment(self):
        labels = np.arange(40)
        info = VideoInfo(10, 20, 64, 48)
        np.testing.assert_array_equal(align(labels, info, 20), np.arange(0, 40, 2))
        np.testing.assert_array_equal(align(labels, VideoInfo(10, 10, 64, 48), 20, 1, True), np.arange(20, 40, 2))
        with self.assertRaises(AlignmentRequired):
            align(labels, info, None)
        with self.assertRaises(AlignmentRequired):
            align(labels, info, 440)
        with self.assertRaises(InputError):
            align(labels, info, 20, 1, True)
        np.testing.assert_array_equal(align(labels, VideoInfo(10, 40, 64, 48), None), labels)

    def test_alignment_uses_last_frame_sample_not_container_end(self):
        # All ten frames have labels: the last sample is at 0.9 s, before 0.95 s.
        labels = np.arange(19)
        info = VideoInfo(10, 10, 64, 48)
        np.testing.assert_array_equal(align(labels, info, 20), np.arange(0, 19, 2))
        with self.assertRaises(InputError):
            align(labels, VideoInfo(10, 11, 64, 48), 20, manual=True)

    def test_nonfinite_container_properties_are_input_errors_and_release_video(self):
        for values in ([float('nan'), 20, 64, 48], [10, float('inf'), 64, 48],
                       [10, 20, float('nan'), 48], [10, 0.5, 64, 48]):
            cap = Mock()
            cap.get.side_effect = values
            with self.subTest(values=values), patch('boute.core.cv2.VideoCapture', return_value=cap):
                with self.assertRaises(InputError):
                    inspect_video(Path('invalid.mp4'))
                cap.release.assert_called_once()

    def test_intake_replacement(self):
        a = self.csv('B-SOiD labels\n1', 'a.csv')
        b = self.csv('B-SOiD labels\n2', 'b.csv')
        files = classify([str(a)])
        files.update(classify([str(b)]))
        self.assertEqual(files['labels'], b)
        with self.assertRaises(InputError):
            classify([str(a), str(b)])
        with self.assertRaises(InputError):
            inspect_video(a)

    def test_intake_reports_full_missing_path_and_folders(self):
        missing = self.root / 'build' / 'labels.csv'
        with self.assertRaises(InputError) as error:
            classify([str(missing)])
        self.assertIn(str(missing), str(error.exception))
        with self.assertRaises(InputError) as error:
            classify([str(self.root)])
        self.assertIn('rather than a folder', str(error.exception))
        self.assertIn(str(self.root), str(error.exception))

    def test_summary_csv_explains_which_results_file_to_choose(self):
        for name in ('example_bout_lengths_440Hz.csv', 'example_transitions_mat_440Hz.csv'):
            path = self.csv('summary', name)
            with self.subTest(name=name), self.assertRaisesRegex(InputError, 'labels_pose'):
                classify([str(path)])

    def make_video(self):
        path = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(20):
            writer.write(np.full((48, 64, 3), 15 + index * 10, dtype=np.uint8))
        writer.release()
        return path

    def test_exact_export_order_and_manifest(self):
        video = self.make_video()
        labels = np.array([3, 21, 21, 10] * 5)
        path = self.csv('B-SOiD labels\n' + '\n'.join(map(str, labels)))
        info = inspect_video(video)
        output = generate(video, path, [], labels, labels, info, self.root / 'out', 10, 0, lambda *_: None, threading.Event())
        total = 0
        for group in np.unique(labels):
            cap = cv2.VideoCapture(str(output / f'group_{group}.mp4'))
            decoded = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(frame.mean())
            cap.release()
            expected = 15 + np.flatnonzero(labels == group) * 10
            self.assertEqual(len(decoded), len(expected))
            np.testing.assert_allclose(decoded, expected, atol=5)
            total += len(decoded)
        self.assertEqual(total, 20)
        with (output / 'frame_manifest.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(r['original_video_frame']) for r in rows], list(range(20)))
        self.assertEqual([int(r['group_id']) for r in rows], labels.tolist())
        self.assertEqual(float(rows[10]['original_video_timestamp']), 1)
        self.assertTrue((output / 'source_csvs' / path.name).exists())

    def test_merge_validation_and_repeated_merges(self):
        labels = np.array([-1, 3, 21, 99])
        mapped = np.array([3, 21, -1, 3])
        for selected in ([3], [3, 3], [3, 200]):
            with self.subTest(selected=selected), self.assertRaises(InputError):
                merge_groups(labels, mapped, selected)
        merged, frames, target = merge_groups(labels, mapped, [3, 21, 99])
        self.assertEqual(target, 3)
        np.testing.assert_array_equal(merged, [-1, 3, 3, 3])
        np.testing.assert_array_equal(frames, [3, 3, -1, 3])
        np.testing.assert_array_equal(labels, [-1, 3, 21, 99])
        np.testing.assert_array_equal(mapped, [3, 21, -1, 3])
        merged, frames, target = merge_groups(merged, frames, [-1, 3])
        np.testing.assert_array_equal(frames, [-1] * 4)

    def test_merged_export_chronology_summary_and_provenance(self):
        from boute.timeline import read_jumps
        video = self.make_video()
        mapped = np.array([3, 21, 10, 21] * 5)
        labels = np.append(mapped, 99)  # Label-only groups can also be merged.
        path = self.csv('B-SOiD labels\n' + '\n'.join(map(str, labels)))
        merged, frames, target = merge_groups(labels, mapped, [3, 21, 99])
        info = inspect_video(video)
        output = generate(video, path, [], merged, frames, info, self.root / 'out', 10, 0,
                          lambda *_: None, threading.Event(), {3: 3, 21: 3, 99: 3, 10: 10})
        self.assertFalse((output / 'group_21.mp4').exists())
        with (output / 'group_summary.csv').open() as stream:
            summary = {r['group_id']: r for r in csv.DictReader(stream)}
        self.assertEqual(set(summary), {'3', '10'})
        self.assertEqual(int(summary['3']['frame_count']), 15)
        self.assertEqual(float(summary['3']['percentage']), 75)
        self.assertEqual(float(summary['3']['duration_seconds']), 1.5)
        with (output / 'frame_manifest.csv').open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['group_id'] == '3']
        indices = np.flatnonzero(frames == target)
        self.assertEqual([int(r['original_video_frame']) for r in rows], indices.tolist())
        self.assertEqual([int(r['exported_group_frame']) for r in rows], list(range(15)))
        cap = cv2.VideoCapture(str(output / 'group_3.mp4'))
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame.mean())
        cap.release()
        np.testing.assert_allclose(decoded, 15 + indices * 10, atol=5)
        self.assertEqual(len(read_jumps(output / 'frame_manifest.csv')['3']), 5)
        with (output / 'group_mapping.csv').open() as stream:
            mapping = {int(r['original_group_id']): int(r['merged_group_id']) for r in csv.DictReader(stream)}
        self.assertEqual(mapping, {3: 3, 21: 3, 99: 3, 10: 10})
        np.testing.assert_array_equal(read_labels(path), labels)

    def test_export_supports_unicode_paths_and_long_multibyte_names(self):
        video = self.make_video().rename(self.root / ('猫' * 70 + '.avi'))
        path = self.csv('B-SOiD labels\n3', 'résultats.csv')
        labels = np.array([3])
        output = generate(video, path, [], labels, np.full(20, 3), inspect_video(video),
                          self.root / 'out', 10, 0, lambda *_: None, threading.Event())
        self.assertLessEqual(len(output.name.encode('utf-8')), 120)
        self.assertEqual(inspect_video(output / 'group_3.mp4').frames, 20)
        self.assertIn(video.name, (output / 'source_metadata.txt').read_text(encoding='utf-8'))
        self.assertTrue((output / 'source_csvs' / path.name).exists())

    def test_cancel_removes_partial_run(self):
        video = self.make_video()
        path = self.csv('B-SOiD labels\n3')
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InputError):
            generate(video, path, [], np.array([3]), np.full(20, 3), inspect_video(video), self.root / 'out', 10, 0, lambda *_: None, cancel)
        self.assertEqual(list((self.root / 'out').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
