from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk


@dataclass(frozen=True)
class FrameJump:
    output_frame: int
    previous_source: int
    next_source: int

    @property
    def description(self) -> str:
        distance = self.next_source - self.previous_source
        return (f'Source frame {self.previous_source:,} → {self.next_source:,}\n'
                f'Jump: +{distance:,} frames · {distance - 1:,} frames skipped')


def read_jumps(manifest: Path) -> dict[str, list[FrameJump]]:
    """Find gaps between successive source frames within each montage."""
    jumps: dict[str, list[FrameJump]] = {}
    previous: dict[str, int] = {}
    with manifest.open(newline='') as stream:
        for row in csv.DictReader(stream):
            group = row['group_id']
            source = int(row['original_video_frame'])
            if group in previous and source > previous[group] + 1:
                jumps.setdefault(group, []).append(FrameJump(
                    int(row['exported_group_frame']), previous[group], source))
            previous[group] = source
    return jumps


class JumpMarkers(tk.Canvas):
    """A marker strip sharing the slider's full-width frame coordinates."""

    def __init__(self, parent, command):
        super().__init__(parent, height=14, bg='#ffffff', highlightthickness=0)
        self.command = command
        self.jumps = []
        self.positions = []
        self.frame_count = 0
        self.tooltip = None
        self.hovered = None
        self.bind('<Configure>', self.redraw)
        self.bind('<Motion>', self.hover)
        self.bind('<Leave>', lambda event: self.hide_tooltip())
        self.bind('<Button-1>', self.click)
        self.bind('<Destroy>', lambda event: self.hide_tooltip())

    def set_data(self, jumps, frame_count):
        self.hide_tooltip()
        self.jumps = jumps
        self.frame_count = frame_count
        self.positions = [jump.output_frame for jump in jumps]
        self.redraw()

    def redraw(self, event=None):
        self.hide_tooltip()
        self.delete('all')
        width = max(1, self.winfo_width() - 1)
        # Dense jumps can share a pixel; retain every jump for hover lookup.
        pixels = {round(j.output_frame / max(1, self.frame_count - 1) * width)
                  for j in self.jumps}
        for x in sorted(pixels):
            self.create_line(x, 2, x, 13, fill='#ba651b', width=2)

    def nearest(self, x):
        if not self.jumps:
            return None
        width = max(1, self.winfo_width() - 1)
        last = max(1, self.frame_count - 1)
        index = bisect_left(self.positions, x / width * last)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.jumps)]
        nearest = min(candidates, key=lambda i: abs(self.positions[i] / last * width - x))
        if abs(self.positions[nearest] / last * width - x) <= 6:
            return self.jumps[nearest]
        return None

    def hover(self, event):
        jump = self.nearest(event.x)
        if jump == self.hovered:
            return
        self.hide_tooltip()
        if jump is None:
            return
        self.hovered = jump
        self.tooltip = tk.Toplevel(self)
        self.tooltip.withdraw()
        self.tooltip.overrideredirect(True)
        tk.Label(self.tooltip, text=jump.description, justify='left',
                 bg='#252525', fg='white', padx=10, pady=7,
                 font=('Arial', 11)).pack()
        self.tooltip.update_idletasks()
        x = max(0, min(event.x_root + 12,
                       self.winfo_screenwidth() - self.tooltip.winfo_reqwidth()))
        y = max(0, self.winfo_rooty() - self.tooltip.winfo_reqheight() - 6)
        self.tooltip.geometry(f'+{x}+{y}')
        self.tooltip.deiconify()

    def hide_tooltip(self):
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None
        self.hovered = None

    def click(self, event):
        jump = self.nearest(event.x)
        self.hide_tooltip()
        if jump is not None:
            self.command(jump.output_frame)
