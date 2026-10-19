from __future__ import annotations

import csv
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from PIL import Image, ImageTk

from .core import AlignmentRequired, InputError, align, classify, filename_rate, generate, inspect_video, read_labels
from .dnd import uri_paths
from .timeline import JumpMarkers, read_jumps

WHITE = '#ffffff'
INK = '#252525'
MUTED = '#777777'
LIGHT = '#f4f4f4'


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title('Boute')
        self.geometry('1120x740')
        self.minsize(820, 560)
        self.configure(bg=WHITE)
        ctk.set_appearance_mode('light')
        self.files = {}
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.cap = None
        self.playing = False
        self.preview_job = None
        self.closed = False
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.upload()
        self.after(60, self.poll)

    def button(self, parent, text, command, **kwargs):
        return ctk.CTkButton(parent, text=text, command=command, fg_color=LIGHT,
                             hover_color='#e9e9e9', text_color=INK, corner_radius=6,
                             height=34, font=('Arial', 13), **kwargs)

    def label(self, parent, text, size=14, color=INK, **kwargs):
        return tk.Label(parent, text=text, bg=WHITE, fg=color, font=('Arial', size), **kwargs)

    def clear(self):
        self.stop_preview()
        for child in self.winfo_children():
            child.destroy()

    def upload(self):
        self.clear()
        self.zones = []
        body = tk.Frame(self, bg=WHITE)
        body.pack(fill='both', expand=True)
        body.rowconfigure(0, weight=1)
        for index, (key, title, hint) in enumerate([
            ('video', 'Choose original video', 'Click to browse, or drop a video here.'),
            ('labels', 'Choose B-SOiD results CSV', 'Look for “labels_pose” in the filename.\nClick to browse, or drop the CSV here.'),
        ]):
            body.columnconfigure(index, weight=1, uniform='uploads')
            zone = tk.Frame(body, bg=WHITE)
            zone.grid(row=0, column=index, sticky='nsew')
            center = tk.Frame(zone, bg=WHITE)
            center.place(relx=.5, rely=.5, anchor='center', relwidth=.94)
            icon = tk.Canvas(center, width=56, height=56, bg=WHITE, highlightthickness=0)
            icon.pack(pady=(0, 18))
            # Lucide file-up icon geometry (ISC license, see THIRD_PARTY_NOTICES.md).
            s = 2
            for points in [(14,2,6,2,4,4,4,20,6,22,18,22,20,20,20,8,14,2), (14,2,14,8,20,8), (12,18,12,12), (9,15,12,12,15,15)]:
                icon.create_line(*[v*s+4 for v in points], fill='#888888', width=1.5, joinstyle='round', capstyle='round')
            text = self.label(center, title, 13, MUTED, wraplength=280)
            text.pack()
            help_text = self.label(center, hint, 12, MUTED, wraplength=340)
            help_text.pack(pady=(12, 0))
            widgets = [zone, center, icon, text, help_text]
            for widget in widgets:
                widget.bind('<Button-1>', lambda event, k=key: self.pick(k))
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<DropEnter>>', lambda event, w=widgets: self.highlight(w, LIGHT))
                widget.dnd_bind('<<DropLeave>>', lambda event, w=widgets: self.highlight(w, WHITE))
                widget.dnd_bind('<<Drop>>', lambda event, w=widgets: self.drop(event, w))
            self.zones.append((key, text))
        footer = tk.Frame(self, bg=WHITE, height=90)
        footer.pack(fill='x')
        footer.pack_propagate(False)
        self.continue_button = self.button(footer, 'Continue →', self.start, width=130)
        self.refresh_inputs()

    def highlight(self, widgets, color):
        for widget in widgets:
            widget.configure(bg=color)
        return 'copy'

    def drop(self, event, widgets):
        self.highlight(widgets, WHITE)
        try:
            paths = list(self.tk.splitlist(event.data))
            if sys.platform.startswith('linux') and (not paths or any(not Path(p).is_file() for p in paths)):
                # Read the original XDND selection while the source still owns it.
                # Some Linux/TkDND combinations deliver unrelated Tcl data here.
                try:
                    raw = self.selection_get(selection='XdndSelection', type='text/uri-list')
                    paths = uri_paths(raw)
                except (tk.TclError, InputError, ValueError):
                    pass  # Retain the original path for a useful intake error.
            updates = classify(paths)
            self.files.update(updates)
            self.refresh_inputs()
        except (InputError, tk.TclError) as error:
            messagebox.showerror('Could not add files', str(error), parent=self)
            return 'refuse_drop'
        return 'copy'

    def pick(self, key):
        types = [('Video', '*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.mpeg *.mpg')] if key == 'video' else [('B-SOiD results (labels_pose)', '*labels_pose*.csv'), ('All CSV files', '*.csv')]
        paths = filedialog.askopenfilenames(parent=self, title='Choose original video' if key == 'video' else 'Choose B-SOiD results CSV (labels_pose)', filetypes=types)
        if paths:
            self.accept(list(paths))

    def accept(self, paths):
        try:
            updates = classify(paths)
            self.files.update(updates)
            self.refresh_inputs()
        except InputError as error:
            messagebox.showerror('Could not add files', str(error), parent=self)

    def refresh_inputs(self):
        for key, label in self.zones:
            if key in self.files:
                label.configure(text=f'{"Video" if key == "video" else "Results CSV"}: {self.files[key].name}')
        if 'video' in self.files and 'labels' in self.files:
            self.continue_button.place(relx=.5, rely=.35, anchor='center')

    def start(self, rate=None, offset=0, manual=False):
        if not manual:
            video_fly = re.search(r'fly[_ -]*(\d+)', self.files['video'].stem, re.I)
            label_fly = re.search(r'fly[_ -]*(\d+)', self.files['labels'].stem, re.I)
            if video_fly and label_fly and video_fly[1] != label_fly[1]:
                if not messagebox.askyesno('Check recording identity',
                    f'The video filename identifies Fly {video_fly[1]}, but the labels identify Fly {label_fly[1]}. Matching durations do not verify recording identity.\n\nHave you verified that these files describe the same recording despite their names?', parent=self):
                    return
        self.clear()
        self.cancel.clear()
        center = tk.Frame(self, bg=WHITE)
        center.place(relx=.5, rely=.5, anchor='center')
        self.status = self.label(center, 'Reading classifications…', 14, MUTED)
        self.status.pack(pady=(0, 20))
        self.progress = ctk.CTkProgressBar(center, width=300, height=3, progress_color='#666666', fg_color=LIGHT)
        self.progress.pack()
        self.progress.set(0)
        self.percent = self.label(center, '', 12, MUTED)
        self.percent.pack(pady=12)
        def work():
            try:
                labels = read_labels(self.files['labels'])
                info = inspect_video(self.files['video'])
                actual_rate = rate if manual else filename_rate(self.files['labels'])
                mapped = align(labels, info, actual_rate, offset, manual)
                output = generate(self.files['video'], self.files['labels'], [],
                                  labels, mapped, info, Path.home() / 'Boute_Exports', actual_rate, offset,
                                  lambda p, s: self.events.put(('progress', (p, s))), self.cancel)
                self.events.put(('done', output))
            except AlignmentRequired as error:
                self.events.put(('alignment', (str(error), filename_rate(self.files['labels']))))
            except Exception as error:
                self.events.put(('error', str(error)))
        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'progress':
                    self.progress.set(data[0])
                    self.status.configure(text=data[1])
                    self.percent.configure(text=f'{data[0]:.0%}')
                elif kind == 'done':
                    self.output = data
                    self.viewer()
                elif kind == 'alignment':
                    self.alignment_dialog(*data)
                elif kind == 'error':
                    messagebox.showerror('Analysis could not finish', data, parent=self)
                    self.upload()
        except queue.Empty:
            pass
        self.after(60, self.poll)

    def alignment_dialog(self, reason, rate):
        self.upload()
        dialog = ctk.CTkToplevel(self)
        dialog.title('Confirm alignment')
        dialog.geometry('520x430')
        dialog.configure(fg_color=WHITE)
        dialog.transient(self)
        dialog.grab_set()
        ctk.CTkLabel(dialog, text=reason, wraplength=450, justify='left').pack(padx=30, pady=(30, 20))
        ctk.CTkLabel(dialog, text='Label samples per second').pack()
        fps = ctk.CTkEntry(dialog, width=200)
        fps.pack(pady=6)
        if rate:
            fps.insert(0, str(rate))
        ctk.CTkLabel(dialog, text='Video start in label timeline (seconds)').pack()
        offset = ctk.CTkEntry(dialog, width=200)
        offset.insert(0, '0')
        offset.pack(pady=6)
        ctk.CTkLabel(dialog, text='Matching durations cannot prove these files describe the same fly or experiment. Confirm their identity before continuing.', wraplength=440, text_color=MUTED).pack(pady=18)
        def confirm():
            try:
                r, o = float(fps.get()), float(offset.get())
                import math
                if not math.isfinite(r) or not math.isfinite(o) or r <= 0 or o < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror('Check alignment', 'Enter a positive label FPS and a nonnegative offset.', parent=dialog)
                return
            dialog.destroy()
            self.start(r, o, True)
        self.button(dialog, 'Confirm and continue', confirm).pack()

    def viewer(self):
        self.clear()
        with (self.output / 'group_summary.csv').open() as stream:
            self.groups = list(csv.DictReader(stream))
        self.group_jumps = read_jumps(self.output / 'frame_manifest.csv')
        sidebar = ctk.CTkScrollableFrame(self, width=210, fg_color=WHITE, corner_radius=0)
        sidebar.pack(side='left', fill='y', padx=(20, 12), pady=26)
        self.group_buttons = {}
        for group in self.groups:
            gid = group['group_id']
            b = self.button(sidebar, f'Group {gid}     {float(group["percentage"]):.1f}%', lambda g=group: self.select_group(g), width=190, anchor='w')
            b.configure(fg_color=WHITE)
            b.pack(pady=3)
            self.group_buttons[gid] = b
        main = tk.Frame(self, bg=WHITE)
        main.pack(side='left', fill='both', expand=True, padx=(20, 36), pady=32)
        heading = tk.Frame(main, bg=WHITE)
        heading.pack(fill='x')
        self.group_title = self.label(heading, '', 22)
        self.group_title.pack(side='left')
        self.button(heading, 'New analysis', self.new_analysis, width=110).pack(side='right')
        self.details = self.label(main, '', 12, MUTED)
        self.details.pack(anchor='w', pady=(8, 18))
        self.preview = tk.Label(main, bg=WHITE)
        self.preview.pack(fill='both', expand=True)
        timeline_area = tk.Frame(main, bg=WHITE)
        timeline_area.pack(fill='x', pady=18)
        self.jump_markers = JumpMarkers(timeline_area, command=self.seek)
        self.jump_markers.pack(fill='x')
        self.timeline = ctk.CTkSlider(timeline_area, from_=0, to=1, number_of_steps=1, command=self.seek, progress_color='#888888', button_color='#555555', button_hover_color='#333333', fg_color=LIGHT)
        self.timeline.pack(fill='x')
        self.label(timeline_area, 'Orange ticks mark frame jumps · Hover for details, click to seek.', 11, MUTED).pack(anchor='w', pady=(5, 0))
        controls = tk.Frame(main, bg=WHITE)
        controls.pack(fill='x')
        self.play_button = self.button(controls, 'Play', self.toggle, width=70)
        self.play_button.pack(side='left')
        self.speed = ctk.CTkOptionMenu(controls, values=['0.25×', '0.5×', '1×', '2×'], width=85, fg_color=LIGHT, button_color=LIGHT, button_hover_color='#e9e9e9', text_color=INK)
        self.speed.set('1×')
        self.speed.pack(side='left', padx=12)
        self.export_button = self.button(controls, 'Export MP4', self.export, width=110)
        self.export_button.pack(side='right')
        self.button(controls, 'Open folder', self.open_folder, width=110).pack(side='right', padx=10)
        self.label(main, 'Frame montages · Disconnected moments may appear continuous.', 11, MUTED).pack(anchor='w', pady=(22, 4))
        self.label(main, 'Verify that the video and labels describe the same fly and experiment.', 11, MUTED).pack(anchor='w')
        self.select_group(self.groups[0])

    def stop_preview(self):
        self.playing = False
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def select_group(self, group):
        self.stop_preview()
        self.selected = group
        for gid, button in self.group_buttons.items():
            button.configure(fg_color=LIGHT if gid == group['group_id'] else WHITE)
        self.group_title.configure(text=f'Group {group["group_id"]}')
        self.details.configure(text=f'{int(group["frame_count"]):,} frames · {float(group["duration_seconds"]):.1f} seconds')
        self.play_button.configure(text='Play')
        self.frame_count = int(group['frame_count'])
        self.jump_markers.set_data(self.group_jumps.get(group['group_id'], []), self.frame_count)
        self.timeline.configure(to=max(1, self.frame_count - 1), number_of_steps=max(1, self.frame_count - 1))
        self.timeline.set(0)
        state = 'normal' if self.frame_count else 'disabled'
        self.export_button.configure(state=state)
        self.play_button.configure(state=state)
        if not self.frame_count:
            self.preview.configure(image='', text='No source video frames map to this group.', fg=MUTED)
            return
        self.cap = cv2.VideoCapture(str(self.output / f'group_{group["group_id"]}.mp4'))
        self.preview_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.update_idletasks()
        self.show_frame()

    def show_frame(self):
        if self.cap is None:
            return False
        ok, frame = self.cap.read()
        if not ok:
            return False
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(100, self.preview.winfo_width()), max(100, self.preview.winfo_height())), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text='')
        self.timeline.set(max(0, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
        return True

    def toggle(self):
        self.playing = not self.playing
        self.play_button.configure(text='Pause' if self.playing else 'Play')
        if self.playing:
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.frame_count:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.tick()
        elif self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None

    def tick(self):
        self.preview_job = None
        if not self.playing:
            return
        started = time.monotonic()
        if not self.show_frame():
            self.playing = False
            self.play_button.configure(text='Play')
            return
        delay = 1 / (self.preview_fps * float(self.speed.get().rstrip('×')))
        self.preview_job = self.after(max(1, round((delay - (time.monotonic() - started)) * 1000)), self.tick)

    def seek(self, value):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(value)))
            self.show_frame()

    def export(self):
        source = self.output / f'group_{self.selected["group_id"]}.mp4'
        target = filedialog.asksaveasfilename(parent=self, initialfile=source.name, defaultextension='.mp4', filetypes=[('MP4 video', '*.mp4')])
        if target and Path(target).resolve() != source.resolve():
            try:
                shutil.copy2(source, target)
            except OSError as error:
                messagebox.showerror('Could not export', str(error), parent=self)

    def open_folder(self):
        try:
            if sys.platform == 'win32':
                os.startfile(self.output)
            else:
                subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(self.output)])
        except OSError as error:
            messagebox.showerror('Could not open folder', f'{error}\n\n{self.output}', parent=self)

    def new_analysis(self):
        self.files = {}
        self.upload()

    def close(self):
        self.cancel.set()
        self.stop_preview()
        if self.worker and self.worker.is_alive():
            self.after(60, self.close)
            return
        self.closed = True
        self.destroy()


def main():
    App().mainloop()
