"""Programmatic desktop smoke test; requires a live display, not a native DnD test."""
from pathlib import Path
import sys
import tempfile
import threading
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
    app.output = output
    app.viewer()
    app.update()
    assert app.cap.isOpened()
    app.seek(5)
    app.toggle()
    for _ in range(5): app.update()
    app.toggle()
    app.select_group(app.groups[1])
    app.update()
    app.new_analysis()
    app.update()
    assert not app.files
    app.close()
    print('PASS: upload, replacement intake, viewer, preview decode, seek, playback scheduling, group selection, new analysis, close')
