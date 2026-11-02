"""Programmatic desktop smoke test; requires a live display, not a native DnD test."""
from pathlib import Path
import sys
import tempfile
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from boute.core import generate, inspect_video
from boute.gui import App

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    video = root / 'test.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    for i in range(20): writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    csv = root / 'labels.csv'
    csv.write_text('B-SOiD labels\n' + '\n'.join(['3', '21'] * 10))
    labels = np.array([3,21] * 10)
    output = generate(video, csv, [], labels, labels, inspect_video(video), root / 'out', 10, 0, lambda *_: None, threading.Event())
    app = App()
    app.update()
    app.accept([str(video), str(csv)])
    app.update()
    assert app.continue_button.winfo_ismapped()
    app.analysis = (labels, labels, inspect_video(video), 10, 0, {3: 3, 21: 21})
    app.output = output
    app.viewer()
    app.update()
    assert app.cap.isOpened()
    capture = app.cap
    app.name_entry.insert(0, 'Grooming?')
    app.save_group_name()
    assert app.cap is capture
    assert app.selected['group_name'] == 'Grooming?'
    assert app.group_title.cget('text') == 'Grooming? (Group 3)'
    app.viewer()  # Reload names from the persisted summary.
    app.update()
    assert app.name_entry.get() == 'Grooming?'
    app.seek(5)
    app.toggle()
    for _ in range(5): app.update()
    app.toggle()
    app.select_group(app.groups[1])
    app.update()
    assert app.merge_button.cget('state') == 'disabled'
    app.group_checks['3'].set(True)
    app.update_merge_selection()
    assert app.merge_button.cget('state') == 'disabled'
    app.group_checks['21'].set(True)
    app.update_merge_selection()
    assert app.merge_button.cget('state') == 'normal'
    app.merge_selected()
    deadline = time.monotonic() + 15
    while app.merging and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert not app.merging
    assert len(app.groups) == 1
    assert app.selected['group_id'] == '3'
    assert app.frame_count == 20
    assert app.selected['group_name'] == 'Grooming?'
    assert app.name_entry.get() == 'Grooming?'
    assert not app.group_jumps
    assert app.cap.isOpened()
    assert app.output != output and output.exists()
    assert (app.output / 'group_mapping.csv').exists()
    app.new_analysis()
    app.update()
    assert not app.files
    app.close()
    print('PASS: upload, replacement intake, viewer, preview decode, seek, playback scheduling, group selection, renaming and persistence, merge controls, merged playback, new analysis, close')
