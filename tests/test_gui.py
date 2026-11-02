"""Headless regression checks for platform failures and GUI state handling."""
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from boute.gui import App


class GuiTests(unittest.TestCase):
    def test_pose_results_ignore_stale_groups_and_inactive_viewers(self):
        from boute.pose_view import PoseViewerMixin
        app = SimpleNamespace(viewer_active=False, pose_request=2, output=Path('/results'),
                              pose_cancel=Mock(), pose_status=Mock(), show_pose_report=Mock())
        PoseViewerMixin.handle_pose_event(app, 'pose_report', (2, app.output, {}, None))
        app.show_pose_report.assert_not_called()
        app.viewer_active = True
        app.pose_cancel.is_set.return_value = False
        PoseViewerMixin.handle_pose_event(app, 'pose_report', (1, app.output, {}, None))
        app.show_pose_report.assert_not_called()
        PoseViewerMixin.handle_pose_event(app, 'pose_report', (2, app.output, {'group_id': 3}, None))
        app.show_pose_report.assert_called_once_with({'group_id': 3})

    def test_renaming_updates_display_without_reopening_video(self):
        group = {'group_id': '3', 'percentage': '75', 'group_name': ''}
        entry = Mock()
        entry.get.return_value = 'Grooming?'
        app = SimpleNamespace(output=Path('/results'), selected=group, name_entry=entry,
                              group_buttons={'3': Mock()}, group_title=Mock(), name_hint=Mock())
        with patch('boute.gui.rename_group', return_value='Grooming?') as save:
            App.save_group_name(app)
        save.assert_called_once_with(Path('/results'), '3', 'Grooming?')
        self.assertEqual(group['group_name'], 'Grooming?')
        app.group_title.configure.assert_called_once_with(text='Grooming? (Group 3)')
        self.assertIn('Grooming?', app.group_buttons['3'].configure.call_args.kwargs['text'])

    def test_failed_rename_retains_old_name(self):
        group = {'group_id': '3', 'group_name': 'Walking'}
        app = SimpleNamespace(output=Path('/results'), selected=group, name_entry=Mock())
        with patch('boute.gui.rename_group', side_effect=PermissionError('Read only')), patch(
            'boute.gui.messagebox.showerror'
        ) as dialog:
            App.save_group_name(app)
        self.assertEqual(group['group_name'], 'Walking')
        dialog.assert_called_once()

    def test_named_export_suggests_safe_filename_and_keeps_source_id(self):
        app = SimpleNamespace(output=Path('/results'), selected={'group_id': '3', 'group_name': 'Grooming?'})
        with patch('boute.gui.filedialog.asksaveasfilename', return_value='/exports/grooming.mp4') as dialog, patch(
            'boute.gui.shutil.copy2'
        ) as copy:
            App.export(app)
        self.assertEqual(dialog.call_args.kwargs['initialfile'], 'group_3_Grooming.mp4')
        copy.assert_called_once_with(Path('/results/group_3.mp4'), '/exports/grooming.mp4')

    def test_mac_open_panels_omit_native_filters_and_keep_multiple_selection(self):
        for key in ('video', 'labels'):
            app = SimpleNamespace(accept=Mock())
            with self.subTest(key=key), patch('boute.gui.sys.platform', 'darwin'), patch(
                'boute.gui.filedialog.askopenfilenames', return_value=('/input/file',)
            ) as dialog:
                App.pick(app, key)
            self.assertNotIn('filetypes', dialog.call_args.kwargs)
            self.assertIs(dialog.call_args.kwargs['parent'], app)
            app.accept.assert_called_once_with(['/input/file'])

    def test_other_platforms_keep_open_dialog_filters(self):
        for platform in ('win32', 'linux'):
            app = SimpleNamespace(accept=Mock())
            with self.subTest(platform=platform), patch('boute.gui.sys.platform', platform), patch(
                'boute.gui.filedialog.askopenfilenames', return_value=()
            ) as dialog:
                App.pick(app, 'video')
            self.assertIn('*.mov', dialog.call_args.kwargs['filetypes'][0][1])
            app.accept.assert_not_called()

    def test_mac_save_panel_omits_filters_but_preserves_mp4_default(self):
        app = SimpleNamespace(output=Path('/results'), selected={'group_id': '3'})
        with patch('boute.gui.sys.platform', 'darwin'), patch(
            'boute.gui.filedialog.asksaveasfilename', return_value='/exports/combined.mp4'
        ) as dialog, patch('boute.gui.shutil.copy2') as copy:
            App.export(app)
        self.assertNotIn('filetypes', dialog.call_args.kwargs)
        self.assertEqual(dialog.call_args.kwargs['defaultextension'], '.mp4')
        self.assertEqual(dialog.call_args.kwargs['initialfile'], 'group_3.mp4')
        copy.assert_called_once()

    def test_mac_unfiltered_intake_still_rejects_unsupported_files(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'notes.txt'
            path.touch()
            app = SimpleNamespace(files={}, refresh_inputs=Mock())
            with patch('boute.gui.sys.platform', 'darwin'), patch('boute.gui.messagebox.showerror') as error:
                App.accept(app, [str(path)])
            error.assert_called_once()
            self.assertEqual(app.files, {})
            app.refresh_inputs.assert_not_called()

    def test_missing_native_dnd_library_keeps_file_picker_app_available(self):
        with patch('boute.gui.TkinterDnD.Tk.__init__', side_effect=RuntimeError('Unable to load tkdnd library.')), patch.multiple(
            App, title=Mock(), geometry=Mock(), minsize=Mock(), configure=Mock(),
            protocol=Mock(), upload=Mock(), after=Mock(),
        ):
            app = App()
            self.assertFalse(app.dnd_enabled)
            app.upload.assert_called_once()

    def test_closed_or_closing_app_does_not_process_worker_dialogs(self):
        events = queue.Queue()
        events.put(('error', 'Analysis cancelled.'))
        app = SimpleNamespace(closed=False, closing=True, events=events, after=Mock())
        with patch('boute.gui.messagebox.showerror') as dialog:
            App.poll(app)
        dialog.assert_not_called()
        self.assertEqual(events.qsize(), 1)
        app.after.assert_not_called()

    def test_close_marks_closing_before_waiting_for_worker(self):
        app = SimpleNamespace(closed=False, closing=False, cancel=Mock(), pose_cancel=Mock(), background_jobs=[], stop_preview=Mock(),
                              worker=Mock(), after=Mock(), close=Mock(), destroy=Mock())
        app.worker.is_alive.return_value = True
        App.close(app)
        self.assertTrue(app.closing)
        app.cancel.set.assert_called_once()
        app.destroy.assert_not_called()
        app.after.assert_called_once_with(60, app.close)

    def test_failed_preview_clears_previous_group_image(self):
        app = SimpleNamespace(stop_preview=Mock(), selected=None, group_buttons={},
                              group_title=Mock(), details=Mock(), play_button=Mock(), name_entry=Mock(), name_hint=Mock(),
                              jump_markers=Mock(), timeline=Mock(), export_button=Mock(),
                              preview=Mock(), group_jumps={}, output=Path('/results'),
                              update_idletasks=Mock(), show_frame=Mock(return_value=False))
        with patch('boute.gui.cv2.VideoCapture'):
            App.select_group(app, {'group_id': '3', 'frame_count': '10', 'duration_seconds': '1'})
        self.assertEqual(app.stop_preview.call_count, 2)
        self.assertEqual(app.preview.configure.call_args.kwargs['image'], '')
        app.play_button.configure.assert_called_with(state='disabled')

    def test_play_without_capture_does_nothing(self):
        app = SimpleNamespace(cap=None)
        App.toggle(app)

    def test_export_cannot_overwrite_another_group_video(self):
        app = SimpleNamespace(output=Path('/results'), selected={'group_id': '3'})
        with patch('boute.gui.filedialog.asksaveasfilename', return_value='/results/group_21.mp4'), patch(
            'boute.gui.messagebox.showerror'
        ) as dialog, patch('boute.gui.shutil.copy2') as copy:
            App.export(app)
        dialog.assert_called_once()
        copy.assert_not_called()

    def test_export_to_external_file_still_works(self):
        app = SimpleNamespace(output=Path('/results'), selected={'group_id': '3'})
        with patch('boute.gui.filedialog.asksaveasfilename', return_value='/exports/combined.mp4'), patch(
            'boute.gui.shutil.copy2'
        ) as copy:
            App.export(app)
        copy.assert_called_once_with(Path('/results/group_3.mp4'), '/exports/combined.mp4')
