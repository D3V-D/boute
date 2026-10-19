import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from boute.timeline import FrameJump, JumpMarkers, read_jumps


class TimelineTests(unittest.TestCase):
    def test_manifest_gaps_are_per_group_and_use_output_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / 'frame_manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['group_id', 'original_video_frame', 'exported_group_frame'])
                writer.writerows([
                    [-1, 0, 0], [-1, 1, 1], [21, 2, 0], [21, 3, 1],
                    [-1, 4, 2], [21, 5, 2], [21, 6, 3], [-1, 7, 3],
                ])
            self.assertEqual(read_jumps(manifest), {
                '-1': [FrameJump(2, 1, 4), FrameJump(3, 4, 7)],
                '21': [FrameJump(2, 3, 5)],
            })

    def test_tooltip_distinguishes_distance_from_skipped_frames(self):
        description = FrameJump(1, 10, 20).description
        self.assertIn('10 → 20', description)
        self.assertIn('+10 frames', description)
        self.assertIn('9 frames skipped', description)

    def test_hover_uses_nearest_marker_with_pixel_tolerance(self):
        jumps = [FrameJump(20, 10, 30), FrameJump(80, 40, 60)]
        strip = SimpleNamespace(jumps=jumps, positions=[20, 80], frame_count=101,
                                winfo_width=lambda: 501)
        self.assertEqual(JumpMarkers.nearest(strip, 104), jumps[0])
        self.assertEqual(JumpMarkers.nearest(strip, 400), jumps[1])
        self.assertIsNone(JumpMarkers.nearest(strip, 250))
        strip.jumps = []
        self.assertIsNone(JumpMarkers.nearest(strip, 100))
