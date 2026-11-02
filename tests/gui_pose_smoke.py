"""Exercise auto pose detection, async analysis, overlays, and merge integration."""
from pathlib import Path
import csv
import json
import customtkinter as ctk
import sys
import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boute.gui import App
from boute.pose import load_pose


def wait_for(app, condition, timeout=20):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert condition(), 'Desktop operation timed out'


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    video = root / 'test.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    for _ in range(30):
        writer.write(np.full((120, 160, 3), 20, dtype=np.uint8))
    writer.release()
    labels = root / 'test_labels_pose_10Hz.csv'
    with labels.open('w', newline='') as stream:
        rows = csv.writer(stream)
        rows.writerow(['', 'B-SOiD labels', 'scorer', *(['model'] * 9)])
        rows.writerow([0, 0, 'bodyparts', 'Thorax', 'Thorax', 'Thorax', 'Leg', 'Leg', 'Leg', 'Head', 'Head', 'Head'])
        rows.writerow([1, 0, 'coords', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood'])
        for i in range(31):
            rows.writerow([i+2, i // 15, i, 40, 40, 1, 60+i, 70, .9, 60, 40, 1])
    app = App()
    errors = []
    app.report_callback_exception = lambda *args: errors.append(args)
    app.update()
    app.accept([str(video), str(labels)])
    with patch('boute.gui.Path.home', return_value=root):
        app.start(10, 0, True)
        wait_for(app, lambda: app.pose_report is not None)
    assert not errors, errors
    assert app.pose_data.names == ('Thorax', 'Leg', 'Head')
    assert (app.output / 'pose_data.npz').exists()
    assert app.pose_report['eligible_motion_pairs'] == 14
    app.geometry('1100x700')
    app.update()
    assert len(app.parts_table.get_children()) == 3
    assert app.preview.winfo_width() >= 200
    assert app.preview.winfo_height() >= 100
    position = app.cap.get(cv2.CAP_PROP_POS_FRAMES)
    original_frame = app.current_frame.copy()
    app.show_dots.set(True)
    app.overlay_changed()
    app.show_part_labels.set(True)
    app.overlay_changed()
    assert app.cap.get(cv2.CAP_PROP_POS_FRAMES) == position
    np.testing.assert_equal(app.current_frame, original_frame)
    app.pose_threshold.set(.95)
    app.pose_settings_changed()
    app.request_pose_analysis()
    wait_for(app, lambda: app.pose_report is not None)
    assert app.pose_report['body_parts'][1]['valid_frames'] == 0
    app.pose_threshold.set(.8)
    app.pose_settings_changed()
    app.select_group(app.groups[1])
    wait_for(app, lambda: app.pose_report is not None)
    assert app.pose_report['group_id'] == 1
    app.axis_menu.set('Head')
    app.pose_settings_changed()
    app.request_pose_analysis()
    wait_for(app, lambda: app.pose_report is not None)
    assert app.pose_report['settings']['axis'] == 'Head'
    reports = root / 'reports'
    reports.mkdir()
    with patch('boute.pose_view.filedialog.askdirectory', return_value=str(reports)), patch('boute.pose_view.messagebox.showinfo'):
        app.save_pose_report()
    report_dir = next(reports.iterdir())
    assert json.loads((report_dir / 'analysis_settings.json').read_text())['axis'] == 'Head'
    assert (report_dir / 'angles.csv').exists()
    annotated = root / 'annotated.mp4'
    with patch('boute.pose_view.filedialog.asksaveasfilename', return_value=str(annotated)), patch('boute.pose_view.messagebox.showinfo'):
        app.save_pose_overlay()
        wait_for(app, lambda: not app.pose_exporting)
    assert annotated.exists()
    # Separate DLC import uses an explicit alignment confirmation.
    with patch('boute.pose_view.filedialog.askopenfilename', return_value=str(labels)):
        app.import_pose()
        wait_for(app, lambda: any(isinstance(child, ctk.CTkToplevel) for child in app.winfo_children()))
    dialog = next(child for child in app.winfo_children() if isinstance(child, ctk.CTkToplevel))
    confirm = next(child for child in dialog.winfo_children() if isinstance(child, ctk.CTkButton))
    confirm.invoke()
    wait_for(app, lambda: app.pose_report is not None)
    assert load_pose(app.output).xy.shape == (30, 3, 2)
    app.select_group(app.groups[2])  # Label-only group has no video frames.
    app.update()
    assert app.current_frame is None
    assert app.frame_count == 0
    app.select_group(app.groups[1])
    wait_for(app, lambda: app.pose_report is not None)
    app.show_dots.set(True)
    app.overlay_changed()
    app.name_entry.insert(0, 'Walking')
    app.save_group_name()
    for variable in app.group_checks.values():
        variable.set(True)
    app.merge_selected()
    wait_for(app, lambda: not app.merging and app.pose_report is not None)
    assert app.pose_report['eligible_motion_pairs'] == 29
    assert len(app.pose_data.xy) == 30
    assert (app.output / 'pose_data.npz').exists()
    assert app.show_dots.get()
    app.new_analysis()
    app.update()
    app.close()
    wait_for(app, lambda: app.closed)
    assert not errors, errors
    print('PASS: pose auto-import, separate CSV alignment, tables, confidence, body axis, analysis/annotated exports, live overlay redraw, empty groups, small-window layout, merged continuity, preferences, shutdown')
