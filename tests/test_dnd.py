from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from boute.core import InputError
from boute.dnd import uri_paths
from boute.gui import App


class DropTests(unittest.TestCase):
    def test_file_uri_list(self):
        self.assertEqual(uri_paths('# Files\r\nfile:///tmp/my%20video.mp4\r\n'
                                   'file://localhost/tmp/caf%C3%A9.csv\r\n'),
                         ['/tmp/my video.mp4', '/tmp/café.csv'])
        for data in ('', 'build', 'https://example.com/a.csv', 'file://remote/tmp/a.csv'):
            with self.subTest(data=data), self.assertRaises(InputError):
                uri_paths(data)

    def test_linux_recovers_original_selection_when_drop_data_is_build(self):
        with TemporaryDirectory() as temp:
            video = Path(temp) / 'my video.mp4'
            labels = Path(temp) / 'café.csv'
            video.touch()
            labels.touch()
            app = SimpleNamespace(
                tk=tk.Tcl(), files={}, highlight=Mock(), refresh_inputs=Mock(),
                selection_get=Mock(return_value=video.as_uri() + '\r\n' + labels.as_uri()),
            )
            with patch('boute.gui.sys.platform', 'linux'):
                self.assertEqual(App.drop(app, SimpleNamespace(data='build'), []), 'copy')
            self.assertEqual(app.files, {'video': video, 'labels': labels})
            app.selection_get.assert_called_once_with(selection='XdndSelection', type='text/uri-list')

    def test_valid_drop_does_not_request_selection(self):
        with TemporaryDirectory() as temp:
            video = Path(temp) / 'my video.mp4'
            video.touch()
            interp = tk.Tcl()
            data = interp.call('list', str(video))
            app = SimpleNamespace(tk=interp, files={}, highlight=Mock(),
                                  refresh_inputs=Mock(), selection_get=Mock())
            self.assertEqual(App.drop(app, SimpleNamespace(data=data), []), 'copy')
            self.assertEqual(app.files, {'video': video})
            app.selection_get.assert_not_called()

    def test_failed_selection_refuses_drop_without_changing_inputs(self):
        app = SimpleNamespace(tk=tk.Tcl(), files={'labels': Path('/existing.csv')},
                              highlight=Mock(), refresh_inputs=Mock(),
                              selection_get=Mock(side_effect=tk.TclError('No selection')))
        with patch('boute.gui.sys.platform', 'linux'), patch('boute.gui.messagebox.showerror') as showerror:
            self.assertEqual(App.drop(app, SimpleNamespace(data='build'), []), 'refuse_drop')
        self.assertEqual(app.files, {'labels': Path('/existing.csv')})
        app.refresh_inputs.assert_not_called()
        showerror.assert_called_once()


if __name__ == '__main__':
    unittest.main()
