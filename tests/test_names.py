import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from boute.core import InputError, group_export_filename, rename_group


class GroupNameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.summary = self.output / 'group_summary.csv'
        # Legacy results do not yet have a name column.
        self.summary.write_text('group_id,frame_count,percentage,duration_seconds\n3,15,75,1.5\n21,5,25,0.5\n')

    def rows(self):
        with self.summary.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_unicode_names_and_csv_quotes_roundtrip_preserving_ids_and_counts(self):
        name = '  Grooming, "maybe" — 猫  '
        cleaned = rename_group(self.output, '3', name)
        self.assertEqual(cleaned, name.strip())
        rename_group(self.output, '21', 'Walking')
        rows = self.rows()
        self.assertEqual(rows[0], {'group_id': '3', 'frame_count': '15', 'percentage': '75',
                                   'duration_seconds': '1.5', 'group_name': name.strip()})
        self.assertEqual(rows[1]['group_name'], 'Walking')
        rename_group(self.output, '3', '')
        self.assertEqual(self.rows()[0]['group_name'], '')
        self.assertEqual(self.rows()[1]['group_name'], 'Walking')

    def test_invalid_names_and_unknown_groups_leave_summary_unchanged(self):
        original = self.summary.read_bytes()
        for name in ('x' * 81, 'multi\nline', 'bad\x00name', 'bad\tname'):
            with self.subTest(name=name), self.assertRaises(InputError):
                rename_group(self.output, '3', name)
        with self.assertRaises(InputError):
            rename_group(self.output, '404', 'Missing')
        self.assertEqual(self.summary.read_bytes(), original)

    def test_failed_atomic_replace_retains_summary_and_cleans_temp_file(self):
        original = self.summary.read_bytes()
        with patch('boute.core.Path.replace', side_effect=PermissionError('Read only')):
            with self.assertRaises(PermissionError):
                rename_group(self.output, '3', 'Grooming')
        self.assertEqual(self.summary.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [self.summary])

    def test_export_filenames_are_portable_and_keep_id(self):
        for name, expected in [('', 'group_3.mp4'), ('Grooming?', 'group_3_Grooming.mp4'),
                               ('../walk: fly*', 'group_3_walk_fly.mp4')]:
            with self.subTest(name=name):
                self.assertEqual(group_export_filename({'group_id': '3', 'group_name': name}), expected)
        filename = group_export_filename({'group_id': '21', 'group_name': '猫' * 80})
        self.assertLess(len(filename.encode('utf-8')), 120)
        self.assertTrue(filename.startswith('group_21_'))
