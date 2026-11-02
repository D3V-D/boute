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
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from PIL import Image, ImageTk

from .core import AlignmentRequired, InputError, align, classify, filename_rate, generate, inspect_video, merge_groups, merge_results, read_labels, rename_group, group_export_filename
from .dnd import uri_paths
from .timeline import JumpMarkers, read_jumps
from .pose import read_pose, align_pose, save_pose, draw_pose
from .pose_view import PoseViewerMixin

WHITE = '#ffffff'
INK = '#18181b'
MUTED = '#71717a'
LIGHT = '#f4f4f5'
BORDER = '#e4e4e7'
FONT = 'Segoe UI' if sys.platform == 'win32' else 'Helvetica Neue' if sys.platform == 'darwin' else 'DejaVu Sans'


def group_button_text(group):
    text = f'Group {group["group_id"]}     {float(group["percentage"]):.1f}%'
    name = group.get('group_name', '')
    if name:
        text += '\n' + (name[:20] + '…' if len(name) > 21 else name)
    return text


def group_heading(group):
    name = group.get('group_name', '')
    return f'{name} (Group {group["group_id"]})' if name else f'Group {group["group_id"]}'


def dialog_filters(filetypes):
    # Some macOS Tk builds abort inside setAllowedFileTypes. This native
    # exception cannot be caught in Python, so omit filters before opening
    # either panel. File intake still validates the selected paths.
    return {} if sys.platform == 'darwin' else {'filetypes': filetypes}


class App(PoseViewerMixin, TkinterDnD.Tk):
    def __init__(self):
        self.dnd_enabled = True
        try:
            super().__init__()
        except RuntimeError:
            # TkinterDnD creates the Tk root before loading its native library.
            # Keep the file pickers usable when that platform library cannot load.
            self.dnd_enabled = False
        self.title('Boute')
        self.geometry('1380x880')
        self.minsize(1100, 700)
        self.configure(bg=WHITE)
        ctk.set_appearance_mode('light')
        self.viewer_active = False
        self.pose_exporting = False
        self.pose_data = None
        self.pose_output = None
        self.pose_preferences = {}
        self.pose_message = ''
        self.pose_report = None
        self.pose_request = 0
        self.pose_cancel = threading.Event()
        self.background_jobs = []
        self.current_frame = None
        self.current_frame_number = 0
        self.files = {}
        self.analysis = None
        self.merging = False
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.cap = None
        self.playing = False
        self.preview_job = None
        self.closed = False
        self.closing = False
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.upload()
        self.after(60, self.poll)

    def button(self, parent, text, command, primary=False, **kwargs):
        options = dict(fg_color=INK if primary else WHITE,
                       hover_color='#27272a' if primary else LIGHT,
                       text_color=WHITE if primary else INK, corner_radius=6,
                       border_width=0 if primary else 1, border_color=BORDER,
                       height=34, font=(FONT, 12))
        options.update(kwargs)
        return ctk.CTkButton(parent, text=text, command=command, **options)

    def label(self, parent, text, size=14, color=INK, **kwargs):
        return tk.Label(parent, text=text, bg=WHITE, fg=color, font=(FONT, size), **kwargs)

    def clear(self):
        self.viewer_active = False
        self.pose_cancel.set()
        self.pose_request += 1
        self.current_frame = None
        self.stop_preview()
        for child in self.winfo_children():
            child.destroy()

    def upload(self):
        self.clear()
        self.zones = []
        header = tk.Frame(self, bg=WHITE)
        header.pack(fill='x', padx=40, pady=(36, 12))
        self.label(header, 'Boute', 28).pack(anchor='w')
        self.label(header, 'From behavioral clusters to research observations.', 13, MUTED).pack(anchor='w', pady=(8, 0))
        self.label(header, 'Load a recording and B-SOiD labels. Pose coordinates in labels_pose are detected automatically.', 12, MUTED).pack(anchor='w', pady=(8, 0))
        body = tk.Frame(self, bg=WHITE)
        body.pack(fill='both', expand=True)
        body.rowconfigure(0, weight=1)
        for index, (key, title, hint) in enumerate([
            ('video', 'Choose original video', 'Click to browse, or drop a video here.'),
            ('labels', 'Choose B-SOiD results CSV', 'Look for “labels_pose” in the filename.\nClick to browse, or drop the CSV here.'),
        ]):
            body.columnconfigure(index, weight=1, uniform='uploads')
            zone = tk.Frame(body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
            zone.grid(row=0, column=index, sticky='nsew', padx=(32 if index == 0 else 12, 12 if index == 0 else 32), pady=16)
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
            if not self.dnd_enabled:
                hint = 'Click to browse for a file. Drag and drop is unavailable on this system.'
            help_text = self.label(center, hint, 12, MUTED, wraplength=340)
            help_text.pack(pady=(12, 0))
            widgets = [zone, center, icon, text, help_text]
            for widget in widgets:
                widget.bind('<Button-1>', lambda event, k=key: self.pick(k))
                if self.dnd_enabled:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind('<<DropEnter>>', lambda event, w=widgets: self.highlight(w, LIGHT))
                    widget.dnd_bind('<<DropLeave>>', lambda event, w=widgets: self.highlight(w, WHITE))
                    widget.dnd_bind('<<Drop>>', lambda event, w=widgets: self.drop(event, w))
            self.zones.append((key, text))
        footer = tk.Frame(self, bg=WHITE, height=90)
        footer.pack(fill='x')
        footer.pack_propagate(False)
        self.continue_button = self.button(footer, 'Analyze recording →', self.start, width=180, primary=True)
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
        types = [('Video', ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.m4v', '*.webm', '*.mpeg', '*.mpg')), ('All files', '*')] if key == 'video' else [('B-SOiD results (labels_pose)', '*labels_pose*.csv'), ('All CSV files', '*.csv')]
        paths = filedialog.askopenfilenames(parent=self, title='Choose original video' if key == 'video' else 'Choose B-SOiD results CSV (labels_pose)', **dialog_filters(types))
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
        self.analysis = None
        self.merging = False
        self.processing_screen()
        def work():
            try:
                labels = read_labels(self.files['labels'])
                info = inspect_video(self.files['video'])
                actual_rate = rate if manual else filename_rate(self.files['labels'])
                mapped = align(labels, info, actual_rate, offset, manual)
                output = generate(self.files['video'], self.files['labels'], [],
                                  labels, mapped, info, Path.home() / 'Boute_Exports', actual_rate, offset,
                                  lambda p, s: self.events.put(('progress', (p, s))), self.cancel)
                pose_notice = None
                self.events.put(('progress', (1, 'Checking pose coordinates…')))
                try:
                    raw_pose = read_pose(self.files['labels'])
                    if raw_pose is not None:
                        if len(raw_pose.xy) != len(labels):
                            raise InputError('Pose rows do not match the behavioral labels. Load the pose CSV separately.')
                        save_pose(output, align_pose(raw_pose, info, actual_rate, offset))
                except (InputError, ValueError, OSError) as error:
                    pose_notice = f'Video groups are ready. Pose import needs attention: {error}'
                analysis = (labels, mapped, info, actual_rate, offset, {int(g): int(g) for g in set(labels)})
                self.events.put(('done', (output, analysis, None)))
                if pose_notice:
                    self.events.put(('pose_notice', pose_notice))
            except AlignmentRequired as error:
                self.events.put(('alignment', (str(error), filename_rate(self.files['labels']))))
            except Exception as error:
                self.events.put(('error', str(error)))
        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()

    def processing_screen(self):
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

    def poll(self):
        if self.closed or self.closing:
            return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'progress':
                    self.progress.set(data[0])
                    self.status.configure(text=data[1])
                    self.percent.configure(text=f'{data[0]:.0%}')
                elif kind == 'done':
                    self.output, self.analysis, selected_id = data
                    self.merging = False
                    self.viewer(selected_id)
                elif kind == 'pose_notice':
                    self.pose_status.configure(text=data)
                elif kind.startswith('pose_'):
                    self.handle_pose_event(kind, data)
                elif kind == 'alignment':
                    self.alignment_dialog(*data)
                elif kind == 'error':
                    messagebox.showerror('Analysis could not finish', data, parent=self)
                    if self.merging:
                        self.merging = False
                        self.viewer()
                    else:
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

    def viewer(self, selected_id=None):
        self.clear()
        with (self.output / 'group_summary.csv').open(encoding='utf-8', newline='') as stream:
            self.groups = list(csv.DictReader(stream))
        self.group_jumps = read_jumps(self.output / 'frame_manifest.csv')
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Boute.Treeview', background=WHITE, fieldbackground=WHITE, foreground=INK,
                        rowheight=28, borderwidth=0, font=(FONT, 10))
        style.configure('Boute.Treeview.Heading', background=LIGHT, foreground=MUTED,
                        font=(FONT, 10), relief='flat', padding=6)
        style.map('Boute.Treeview', background=[('selected', '#e4e4e7')], foreground=[('selected', INK)])
        top = tk.Frame(self, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill='x')
        self.label(top, 'Boute  /  Behavior workspace', 14).pack(side='left', padx=24, pady=16)
        self.button(top, 'New analysis', self.new_analysis, width=110).pack(side='right', padx=20, pady=10)
        shell = tk.PanedWindow(self, orient='horizontal', bg=LIGHT, sashwidth=6, borderwidth=0)
        shell.pack(fill='both', expand=True)
        rail = tk.Frame(shell, bg=WHITE)
        shell.add(rail, minsize=195, width=215)
        self.label(rail, 'Groups', 17).pack(anchor='w', padx=16, pady=(20, 4))
        self.label(rail, 'Select groups to combine', 11, MUTED).pack(anchor='w', padx=16, pady=(0, 12))
        self.merge_button = self.button(rail, 'Merge selected', self.merge_selected, primary=True)
        self.merge_button.configure(state='disabled')
        self.merge_button.pack(fill='x', padx=14, pady=(0, 6))
        self.label(rail, 'Keeps the lowest selected ID.', 10, MUTED).pack(anchor='w', padx=16, pady=(0, 10))
        sidebar = ctk.CTkScrollableFrame(rail, width=180, fg_color=WHITE, corner_radius=0)
        sidebar.pack(fill='both', expand=True, padx=6, pady=(0, 12))
        self.group_checks, self.group_buttons = {}, {}
        for group in self.groups:
            gid = group['group_id']
            row = tk.Frame(sidebar, bg=WHITE)
            row.pack(fill='x', pady=3)
            checked = tk.BooleanVar(value=False)
            self.group_checks[gid] = checked
            ctk.CTkCheckBox(row, text='', width=22, variable=checked, fg_color=INK,
                            command=self.update_merge_selection, checkbox_width=16, checkbox_height=16).pack(side='left')
            button = self.button(row, group_button_text(group), lambda g=group: self.select_group(g), width=145,
                                 anchor='w', border_width=0, height=52 if group.get('group_name') else 36)
            button.pack(side='left', fill='x', expand=True)
            self.group_buttons[gid] = button
        main = tk.Frame(shell, bg=WHITE)
        shell.add(main, minsize=435, stretch='always')
        content = tk.Frame(main, bg=WHITE)
        content.pack(fill='both', expand=True, padx=20, pady=20)
        self.group_title = self.label(content, '', 21, wraplength=400, justify='left')
        self.group_title.pack(anchor='w', fill='x')
        self.group_title.bind('<Configure>', lambda event: self.group_title.configure(wraplength=max(200, event.width-2)))
        self.details = self.label(content, '', 11, MUTED)
        self.details.pack(anchor='w', pady=(6, 16))
        name_row = tk.Frame(content, bg=WHITE)
        name_row.pack(fill='x', pady=(0, 4))
        self.name_entry = ctk.CTkEntry(name_row, placeholder_text='Behavior name (e.g. grooming?)',
                                      fg_color=WHITE, border_color=BORDER, text_color=INK, height=34, corner_radius=6)
        self.name_entry.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.name_entry.bind('<Return>', lambda event: self.save_group_name())
        self.button(name_row, 'Save name', self.save_group_name, width=95).pack(side='right')
        self.name_hint = self.label(content, 'Optional researcher label · Blank restores the group ID.', 10, MUTED)
        self.name_hint.pack(anchor='w', pady=(0, 14))
        video_card = tk.Frame(content, bg=LIGHT, highlightbackground=BORDER, highlightthickness=1)
        video_card.pack(fill='both', expand=True)
        self.preview = tk.Label(video_card, bg=LIGHT)
        self.preview.pack(fill='both', expand=True, padx=6, pady=6)
        self.preview.bind('<Configure>', lambda event: self.render_current_frame())
        timeline_area = tk.Frame(content, bg=WHITE)
        timeline_area.pack(fill='x', pady=(12, 10))
        self.jump_markers = JumpMarkers(timeline_area, command=self.seek)
        self.jump_markers.pack(fill='x')
        self.timeline = ctk.CTkSlider(timeline_area, from_=0, to=1, number_of_steps=1, command=self.seek,
                                     progress_color=INK, button_color=INK, button_hover_color='#3f3f46', fg_color=BORDER)
        self.timeline.pack(fill='x')
        self.label(timeline_area, 'Orange ticks = source-frame gaps · Hover for details', 10, MUTED).pack(anchor='w', pady=(5, 0))
        controls = tk.Frame(content, bg=WHITE)
        controls.pack(fill='x')
        self.play_button = self.button(controls, 'Play', self.toggle, width=58, primary=True)
        self.play_button.pack(side='left')
        self.speed = ctk.CTkOptionMenu(controls, values=['0.25×', '0.5×', '1×', '2×'], width=64,
                                      fg_color=LIGHT, button_color=BORDER, button_hover_color='#d4d4d8', text_color=INK)
        self.speed.set('1×')
        self.speed.pack(side='left', padx=4)
        self.export_button = self.button(controls, 'Export MP4', self.export, width=90)
        self.export_button.pack(side='right')
        self.button(controls, 'Folder', self.open_folder, width=54).pack(side='right', padx=3)
        self.label(content, 'Frame montage · Verify the recording and label identity.', 10, MUTED).pack(anchor='w', pady=(14, 0))
        research = tk.Frame(shell, bg=WHITE)
        shell.add(research, minsize=330, width=385)
        self.build_pose_panel(research)
        self.viewer_active = True
        self.select_group(next((g for g in self.groups if g['group_id'] == str(selected_id)), self.groups[0]))

    def save_group_name(self):
        try:
            name = rename_group(self.output, self.selected['group_id'], self.name_entry.get())
        except (InputError, OSError, UnicodeError) as error:
            messagebox.showerror('Could not save name', str(error), parent=self)
            return
        self.selected['group_name'] = name
        button = self.group_buttons[self.selected['group_id']]
        button.configure(text=group_button_text(self.selected), height=52 if name else 34)
        self.group_title.configure(text=group_heading(self.selected))
        self.name_hint.configure(text='Name saved.' if name else 'Name cleared.')

    def update_merge_selection(self):
        count = sum(var.get() for var in self.group_checks.values())
        self.merge_button.configure(text=f'Merge selected ({count})' if count else 'Merge selected',
                                    state='normal' if count >= 2 else 'disabled')

    def merge_selected(self):
        selected = [int(gid) for gid, var in self.group_checks.items() if var.get()]
        if len(selected) < 2 or self.analysis is None:
            return
        labels, mapped, info, rate, offset, mapping = self.analysis
        labels, mapped, target = merge_groups(labels, mapped, selected)
        mapping = {original: target if current in selected else current
                   for original, current in mapping.items()}
        self.processing_screen()
        self.merging = True
        self.status.configure(text='Merging selected groups…')

        def work():
            try:
                output = merge_results(self.output, selected, info,
                                       lambda p, s: self.events.put(('progress', (p, s))), self.cancel)
                self.events.put(('done', (output, (labels, mapped, info, rate, offset, mapping), target)))
            except Exception as error:
                self.events.put(('error', str(error)))
        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()

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
        self.current_frame = None
        self.current_frame_number = 0
        if hasattr(self, 'pose_status'):
            self.prepare_group_pose()
        for gid, button in self.group_buttons.items():
            button.configure(fg_color=LIGHT if gid == group['group_id'] else WHITE)
        self.group_title.configure(text=group_heading(group))
        self.name_entry.delete(0, 'end')
        self.name_entry.insert(0, group.get('group_name', ''))
        self.name_hint.configure(text='Optional researcher label · Blank restores the group ID.')
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
        if not self.show_frame():
            self.stop_preview()
            self.play_button.configure(state='disabled')
            self.preview.configure(image='', text='Could not read this group video. Check that the output file is available.', fg=MUTED)

    def show_frame(self):
        if self.cap is None:
            return False
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.current_frame = frame
        self.current_frame_number = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.render_current_frame()
        self.timeline.set(self.current_frame_number)
        return True

    def render_current_frame(self):
        if self.current_frame is None:
            return
        frame = self.current_frame
        if self.pose_data is not None and hasattr(self, 'show_dots') and self.show_dots.get():
            if self.current_frame_number < len(self.current_pose_indices):
                frame = draw_pose(frame, self.pose_data, int(self.current_pose_indices[self.current_frame_number]),
                                  self.selected_pose_parts(), self.pose_threshold.get(), self.show_part_labels.get())
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(1, self.preview.winfo_width()-4), max(1, self.preview.winfo_height()-4)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text='')

    def toggle(self):
        if self.cap is None:
            return
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
        target = filedialog.asksaveasfilename(parent=self, initialfile=group_export_filename(self.selected), defaultextension='.mp4', **dialog_filters([('MP4 video', '*.mp4')]))
        if target and Path(target).resolve() != source.resolve():
            if Path(target).resolve().parent == self.output.resolve():
                messagebox.showerror('Could not export', 'Choose a location outside this results folder to preserve its group videos.', parent=self)
                return
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
        if self.closed:
            return
        self.closing = True
        self.pose_cancel.set()
        self.cancel.set()
        self.stop_preview()
        if (self.worker and self.worker.is_alive()) or any(job.is_alive() for job in self.background_jobs):
            self.after(60, self.close)
            return
        self.closed = True
        self.destroy()


def main():
    App().mainloop()
