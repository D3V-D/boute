"""Actual desktop cards with SDs, bout p/q values, tabs, and Methods dialog."""
from pathlib import Path
import csv
import sys
import tempfile
import time
from unittest.mock import patch

import customtkinter as ctk
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boute.gui import App


def wait_for(app, condition, timeout=20):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert condition(), 'Desktop operation timed out'


def label_texts(parent):
    text = []
    for child in parent.winfo_children():
        if isinstance(child, ctk.CTkLabel):
            text.append(child.cget('text'))
        text.extend(label_texts(child))
    return text


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    video = root / 'test.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    for _ in range(96):
        writer.write(np.full((120, 160, 3), 20, dtype=np.uint8))
    writer.release()
    labels = root / 'test_labels_pose_10Hz.csv'
    with labels.open('w', newline='') as stream:
        rows = csv.writer(stream)
        rows.writerow(['', 'B-SOiD labels', 'scorer', *(['model'] * 9)])
        rows.writerow([0, 0, 'bodyparts', 'Thorax', 'Thorax', 'Thorax', 'Leg', 'Leg', 'Leg', 'Head', 'Head', 'Head'])
        rows.writerow([1, 0, 'coords', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood'])
        for i in range(96):
            group = (i // 8) % 2
            x = 60 if group == 0 else 70 + (i % 8) * 2
            rows.writerow([i+2, group, i, 40, 40, 1, x, 70, 1, 60, 40, 1])
    app = App()
    errors = []
    app.report_callback_exception = lambda *args: errors.append(args)
    app.accept([str(video), str(labels)])
    with patch('boute.gui.Path.home', return_value=root):
        app.start(10, 0, True)
        wait_for(app, lambda: app.pose_report is not None)
    app.update()
    assert not errors, errors
    assert app.pose_report['insights']
    assert app.pose_report['tested_features'] > 0
    assert any(row['q_value'] is not None for row in app.pose_report['insights'])
    assert len(app.insight_cards.winfo_children()) == len(app.pose_report['insights'])
    labels_on_cards = label_texts(app.insight_cards)
    assert any(text.startswith('SD ') for text in labels_on_cards)
    assert any(' · q ' in text for text in labels_on_cards)
    assert any('bouts' in text for text in labels_on_cards)
    assert app.pose_tabs.get() == 'Insights'
    app.pose_tabs.set('Data')
    app.update()
    assert app.parts_table.winfo_ismapped()
    app.pose_tabs.set('Settings')
    app.update()
    assert app.reference_menu.winfo_ismapped()
    app.pose_threshold.set(.9)
    app.pose_settings_changed()
    assert not app.insight_cards.winfo_children()
    assert app.pose_report is None
    app.request_pose_analysis()
    wait_for(app, lambda: app.pose_report is not None)
    app.pose_tabs.set('Insights')
    app.show_analysis_methods()
    app.update()
    dialog = next(child for child in app.winfo_children() if isinstance(child, ctk.CTkToplevel))
    box = next(child for child in dialog.winfo_children() if isinstance(child, ctk.CTkTextbox))
    assert 'Benjamini' in box.get('1.0', 'end')
    dialog.destroy()
    app.geometry('1100x700')
    app.update()
    assert app.report_button.winfo_ismapped()
    assert app.overlay_export_button.winfo_ismapped()
    app.close()
    wait_for(app, lambda: app.closed)
    assert not errors, errors
    print('PASS: compact SD/p/q cards, bout evidence, Insights/Data/Settings, Methods, stale-state clearing, small layout, exports, shutdown')
