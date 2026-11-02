"""Desktop pose controls and analysis panels, using the application's event queue."""
from __future__ import annotations

import math
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import numpy as np

from .core import InputError, filename_rate, group_export_filename
from .pose import (read_pose, align_pose, save_pose, load_pose, analyze_pose,
                   montage_indices, draw_pose, export_overlay, export_report)

BG, INK, MUTED, BORDER, SOFT = '#ffffff', '#18181b', '#71717a', '#e4e4e7', '#f4f4f5'


def number(value, suffix=''):
    return '—' if value is None or not math.isfinite(value) else f'{value:.2f}{suffix}'


class PoseViewerMixin:
    def start_background(self, action):
        self.background_jobs = [job for job in self.background_jobs if job.is_alive()]
        thread = threading.Thread(target=action, daemon=False)
        self.background_jobs.append(thread)
        thread.start()

    def build_pose_panel(self, parent):
        self.pose_report = None
        self.pose_indices_cache = {}
        if getattr(self, 'pose_output', None) != self.output:
            self.pose_output = self.output
            try:
                self.pose_data = load_pose(self.output)
                self.pose_message = ''
            except (OSError, ValueError, KeyError) as error:
                self.pose_data = None
                self.pose_message = f'Pose data unavailable: {error}'
        pane = ctk.CTkScrollableFrame(parent, width=355, fg_color=BG, corner_radius=0)
        pane.pack(fill='both', expand=True, padx=12, pady=12)
        ctk.CTkLabel(pane, text='Movement insights', text_color=INK, font=('Arial', 19, 'bold'), anchor='w').pack(fill='x')
        ctk.CTkLabel(pane, text='Evidence for your next annotation', text_color=MUTED, font=('Arial', 12), anchor='w').pack(fill='x', pady=(0, 12))
        self.pose_status = ctk.CTkLabel(pane, text='', wraplength=320, justify='left', text_color=MUTED, anchor='w')
        self.pose_status.pack(fill='x', pady=(0, 8))
        insight_card = ctk.CTkFrame(pane, fg_color=SOFT, corner_radius=8, border_width=1, border_color=BORDER)
        insight_card.pack(fill='x', pady=(0, 12))
        ctk.CTkLabel(insight_card, text='Insights for this group', text_color=INK,
                     font=('Arial', 15, 'bold'), anchor='w').pack(fill='x', padx=12, pady=(12, 6))
        self.group_insights_text = ctk.CTkLabel(insight_card, text='Load pose data to find distinctive movement, stability, and relationships.',
                                               wraplength=285, justify='left', anchor='w', text_color=INK, font=('Arial', 12))
        self.group_insights_text.pack(fill='x', padx=12, pady=(0, 10))
        ctk.CTkLabel(insight_card, text='Compared with equally weighted groups in this recording. Descriptive differences, not behavior labels.',
                     wraplength=285, justify='left', text_color=MUTED, font=('Arial', 10)).pack(fill='x', padx=12, pady=(0, 12))
        ctk.CTkLabel(pane, text='Similar groups', text_color=INK, font=('Arial', 15, 'bold'), anchor='w').pack(fill='x')
        self.similarity_note = ctk.CTkLabel(pane, text='Load pose data to compare groups.', wraplength=310,
                                           justify='left', text_color=MUTED, font=('Arial', 10), anchor='w')
        self.similarity_note.pack(fill='x', pady=(2, 8))
        self.similarity_table = self.pose_table(pane, [('rank', '#', 30), ('group', 'Group', 110),
                                                     ('distance', 'Dist. ↓', 72), ('score', 'Score', 65),
                                                     ('shared', 'Shared features', 105)], height=4)
        self.similarity_table.bind('<<TreeviewSelect>>', self.similarity_selected)
        ctk.CTkLabel(pane, text='Lower distance = more similar. Families receive equal weight; features are scaled by across-group IQR. Score = 100 / (1 + distance), not a probability. Select a row to inspect that group.',
                     wraplength=310, justify='left', text_color=MUTED, font=('Arial', 10)).pack(fill='x', pady=(4, 12))
        self.button(pane, 'Load / replace pose CSV', self.import_pose).pack(fill='x', pady=(0, 12))
        if self.pose_data is None:
            self.pose_status.configure(text=self.pose_message or 'No pose data yet. Use a DLC CSV with body-part x, y, and likelihood columns. Combined labels_pose files are loaded automatically.')
            return
        pose = self.pose_data
        if len(pose.xy) != self.analysis[2].frames:
            self.pose_data = None
            self.pose_status.configure(text='Pose frame count does not match the video. Load the pose CSV again to align it.')
            return
        self.pose_status.configure(text=f'{len(pose.names)} landmarks · {len(pose.xy):,} aligned frames\nConfidence-gated · No interpolation')
        prefs = self.pose_preferences
        self.show_dots = tk.BooleanVar(value=prefs.get('dots', False))
        self.show_part_labels = tk.BooleanVar(value=prefs.get('labels', False))
        switches = ctk.CTkFrame(pane, fg_color=SOFT, corner_radius=8)
        switches.pack(fill='x', pady=(0, 10))
        ctk.CTkSwitch(switches, text='Show dots', variable=self.show_dots, command=self.overlay_changed,
                       progress_color=INK, font=('Arial', 12)).pack(anchor='w', padx=12, pady=(12, 8))
        ctk.CTkCheckBox(switches, text='Show landmark names', variable=self.show_part_labels,
                       command=self.overlay_changed, fg_color=INK, checkbox_width=16, checkbox_height=16,
                       font=('Arial', 12)).pack(anchor='w', padx=12, pady=(0, 12))
        confidence_row = ctk.CTkFrame(pane, fg_color='transparent')
        confidence_row.pack(fill='x')
        self.confidence_label = ctk.CTkLabel(confidence_row, text='Confidence', text_color=INK)
        self.confidence_label.pack(side='left')
        self.pose_threshold = tk.DoubleVar(value=prefs.get('threshold', .8))
        self.confidence_value = ctk.CTkLabel(confidence_row, text=f'{self.pose_threshold.get():.2f}', text_color=MUTED)
        self.confidence_value.pack(side='right')
        ctk.CTkSlider(pane, from_=0, to=1, number_of_steps=100, variable=self.pose_threshold,
                      command=lambda value: self.pose_settings_changed(), button_color=INK,
                      progress_color=INK, fg_color=BORDER).pack(fill='x', pady=(0, 10))
        self.part_options = {}
        chooser = ctk.CTkScrollableFrame(pane, height=100, fg_color=SOFT, corner_radius=8)
        chooser.pack(fill='x', pady=(0, 10))
        for name in pose.names:
            variable = tk.BooleanVar(value=name in prefs.get('parts', pose.names))
            self.part_options[name] = variable
            ctk.CTkCheckBox(chooser, text=name, variable=variable, command=self.overlay_changed,
                            fg_color=INK, checkbox_width=15, checkbox_height=15, font=('Arial', 11)).pack(anchor='w', padx=4, pady=3)
        ctk.CTkLabel(pane, text='Body reference', anchor='w', text_color=INK).pack(fill='x')
        reference = prefs.get('reference')
        if reference not in pose.names:
            usable = np.isfinite(pose.confidence) & (pose.confidence >= .8) & (pose.confidence <= 1)
            reference = pose.names[int(usable.sum(axis=0).argmax())]
        self.reference_menu = self.pose_menu(pane, list(pose.names), reference)
        ctk.CTkLabel(pane, text='Body axis (optional rotation correction)', anchor='w', text_color=MUTED, font=('Arial', 11)).pack(fill='x')
        self.axis_menu = self.pose_menu(pane, ['Translation only', *pose.names], prefs.get('axis', 'Translation only'))
        ctk.CTkLabel(pane, text='Median smoothing (unique pose observations)', anchor='w', text_color=MUTED, font=('Arial', 11)).pack(fill='x')
        self.smoothing_menu = self.pose_menu(pane, ['Off', '3 samples', '5 samples'], prefs.get('smoothing', '3 samples'))
        self.button(pane, 'Update analysis', self.request_pose_analysis, primary=True).pack(fill='x', pady=(4, 12))
        tabs = ctk.CTkTabview(pane, height=470, fg_color=SOFT, segmented_button_fg_color=SOFT,
                             segmented_button_selected_color=BG, segmented_button_selected_hover_color=BG,
                             segmented_button_unselected_color=SOFT, text_color=INK, corner_radius=8)
        tabs.pack(fill='x')
        for title in ('Summary', 'Parts', 'Relations'):
            tabs.add(title)
        self.insight_text = ctk.CTkLabel(tabs.tab('Summary'), text='Select a group to analyze.', wraplength=285,
                                        justify='left', text_color=INK, anchor='nw', font=('Arial', 13))
        self.insight_text.pack(fill='x', padx=8, pady=12)
        self.motion_chart = tk.Canvas(tabs.tab('Summary'), bg=SOFT, height=200, highlightthickness=0)
        self.motion_chart.pack(fill='x', padx=8)
        ctk.CTkLabel(tabs.tab('Summary'), text='2D pixels / second · Rankings require ≥50% valid observations and ≥25% of eligible body-motion pairs (minimum 3). Descriptive hints, not behavior diagnoses.',
                     wraplength=285, justify='left', text_color=MUTED, font=('Arial', 11)).pack(fill='x', padx=8, pady=12)
        self.parts_table = self.pose_table(tabs.tab('Parts'), [('part', 'Landmark', 135), ('camera', 'Image px/s', 85),
                                                             ('body', 'Body px/s', 85), ('coverage', 'Valid %', 70),
                                                             ('ratio', 'Body vs rest', 90), ('range', 'Body range px', 100),
                                                             ('pairs', 'Body pairs', 85)])
        ctk.CTkLabel(tabs.tab('Parts'), text='Body speed removes reference motion; an axis also removes rotation. “vs rest” compares this landmark with other groups. — means insufficient data.',
                     wraplength=285, justify='left', text_color=MUTED, font=('Arial', 11)).pack(fill='x', pady=8)
        self.pairs_table = self.pose_table(tabs.tab('Relations'), [('pair', 'Landmark pair', 205), ('change', 'Change px/s', 100),
                                                                 ('range', 'Range px', 80), ('pairs', 'Valid pairs', 85)], height=5)
        ctk.CTkLabel(tabs.tab('Relations'), text='Joint angle: A → vertex → C', anchor='w', text_color=INK).pack(fill='x', pady=(10, 0))
        self.angle_menus = []
        for i in range(3):
            self.angle_menus.append(self.pose_menu(tabs.tab('Relations'), list(pose.names), prefs.get(f'angle{i}', pose.names[min(i, len(pose.names)-1)])))
        self.angle_text = ctk.CTkLabel(tabs.tab('Relations'), text='Choose three distinct points, then update.', wraplength=285,
                                      justify='left', text_color=MUTED, font=('Arial', 11))
        self.angle_text.pack(fill='x', pady=8)
        self.report_button = self.button(pane, 'Export analysis tables', self.save_pose_report)
        self.report_button.configure(state='disabled')
        self.report_button.pack(fill='x', pady=(14, 6))
        self.overlay_export_button = self.button(pane, 'Export MP4 with dots', self.save_pose_overlay)
        self.overlay_export_button.configure(state='disabled' if self.pose_exporting else 'normal')
        self.overlay_export_button.pack(fill='x')
        ctk.CTkLabel(pane, text='Overlays use original image coordinates. Cropped / resized pose data must be converted before import. Exported tables include confidence, sample counts, and analysis settings.',
                     wraplength=320, justify='left', text_color=MUTED, font=('Arial', 11)).pack(fill='x', pady=10)

    def pose_menu(self, parent, values, value):
        menu = ctk.CTkOptionMenu(parent, values=values, command=lambda value: self.pose_settings_changed(),
                                fg_color=BG, button_color=BORDER, button_hover_color='#d4d4d8',
                                text_color=INK, dropdown_fg_color=BG, dropdown_text_color=INK, height=30)
        menu.set(value if value in values else values[0])
        menu.pack(fill='x', pady=(0, 8))
        return menu

    def pose_table(self, parent, columns, height=10):
        box = tk.Frame(parent, bg=SOFT)
        box.pack(fill='both', expand=True)
        table = ttk.Treeview(box, columns=[c[0] for c in columns], show='headings', height=height, style='Boute.Treeview')
        for key, title, width in columns:
            table.heading(key, text=title)
            table.column(key, width=width, minwidth=min(60, width), stretch=False)
        vertical = ttk.Scrollbar(box, orient='vertical', command=table.yview)
        horizontal = ttk.Scrollbar(box, orient='horizontal', command=table.xview)
        table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        table.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        return table

    def remember_pose_preferences(self):
        if self.pose_data is None or not hasattr(self, 'reference_menu'):
            return
        self.pose_preferences = dict(dots=self.show_dots.get(), labels=self.show_part_labels.get(),
                                     threshold=self.pose_threshold.get(), parts=[name for name, v in self.part_options.items() if v.get()],
                                     reference=self.reference_menu.get(), axis=self.axis_menu.get(), smoothing=self.smoothing_menu.get(),
                                     **{f'angle{i}': menu.get() for i, menu in enumerate(self.angle_menus)})

    def overlay_changed(self):
        self.remember_pose_preferences()
        self.render_current_frame()

    def pose_settings_changed(self):
        self.pose_report = None
        self.pose_cancel.set()
        if hasattr(self, 'report_button'):
            self.report_button.configure(state='disabled')
        self.confidence_value.configure(text=f'{self.pose_threshold.get():.2f}')
        self.pose_status.configure(text='Settings changed · Update analysis to refresh measurements.')
        self.clear_group_comparisons('Settings changed. Update analysis to compare this group again.')
        self.overlay_changed()

    def selected_pose_parts(self):
        return [i for i, name in enumerate(self.pose_data.names) if self.part_options[name].get()]

    def prepare_group_pose(self):
        self.clear_group_comparisons('Comparing this group with the rest of the recording…' if self.pose_data is not None else 'Load pose data to find distinctive movement, stability, and relationships.')
        self.pose_report = None
        self.pose_cancel.set()
        self.pose_request += 1
        if self.pose_data is not None:
            gid = self.selected['group_id']
            if gid not in self.pose_indices_cache:
                self.pose_indices_cache[gid] = montage_indices(self.output, gid)
            self.current_pose_indices = self.pose_indices_cache[gid]
            self.request_pose_analysis()

    def request_pose_analysis(self):
        if self.pose_data is None or self.analysis is None:
            return
        self.pose_cancel.set()
        self.pose_cancel = threading.Event()
        cancel = self.pose_cancel
        self.pose_request += 1
        token, output, gid = self.pose_request, self.output, int(self.selected['group_id'])
        self.pose_report = None
        self.report_button.configure(state='disabled')
        try:
            names = self.pose_data.names
            reference = names.index(self.reference_menu.get())
            axis = None if self.axis_menu.get() == 'Translation only' else names.index(self.axis_menu.get())
            angle = tuple(names.index(menu.get()) for menu in self.angle_menus)
            if len(set(angle)) < 3:
                angle = None
            smoothing = {'Off': 1, '3 samples': 3, '5 samples': 5}[self.smoothing_menu.get()]
            threshold = self.pose_threshold.get()
        except (ValueError, KeyError) as error:
            self.pose_status.configure(text=str(error))
            return
        self.remember_pose_preferences()
        self.clear_group_comparisons('Comparing this group with the rest of the recording…')
        pose, groups, info = self.pose_data, self.analysis[1], self.analysis[2]
        self.pose_status.configure(text='Analyzing continuous source segments…')
        def work():
            try:
                report = analyze_pose(pose, groups, info, gid, threshold, reference, axis, smoothing, angle, cancel)
                self.events.put(('pose_report', (token, output, report, None)))
            except Exception as error:
                self.events.put(('pose_report', (token, output, None, str(error))))
        self.start_background(work)

    def handle_pose_event(self, kind, data):
        if kind != 'pose_exported' and not self.viewer_active:
            return
        if kind == 'pose_report':
            token, output, report, error = data
            if token != self.pose_request or output != self.output or self.pose_cancel.is_set():
                return
            if error:
                self.pose_status.configure(text=error)
                self.clear_group_comparisons('Comparison unavailable. Check the analysis settings.')
            else:
                self.pose_report = report
                self.show_pose_report(report)
        elif kind == 'pose_imported':
            output, pose, error = data
            if output != self.output:
                return
            if error:
                self.pose_status.configure(text=error)
                self.clear_group_comparisons('Comparison unavailable. Check the analysis settings.')
            else:
                self.pose_alignment_dialog(pose)
        elif kind == 'pose_exported':
            self.pose_exporting = False
            if self.viewer_active and self.pose_data is not None:
                self.overlay_export_button.configure(state='normal')
            path, error = data
            messagebox.showerror('Export failed', error, parent=self) if error else messagebox.showinfo('Export complete', f'Saved to:\n{path}', parent=self)

    def clear_group_comparisons(self, message):
        self.group_insights_text.configure(text=message)
        self.similarity_note.configure(text='Comparison pending.' if self.pose_data is not None else 'Pose data is required for similarity.')
        self.similarity_table.delete(*self.similarity_table.get_children())

    def show_group_comparisons(self, report):
        descriptions = []
        for item in report['insights']:
            evidence = (f'{item["valid_samples"]:,} valid {item["sample_kind"]} · '
                        f'{item["coverage_percent"]:.0f}% coverage · {item["comparison_groups"]} comparison groups')
            descriptions.append(f'{item["title"]}\n{item["description"]}\n{evidence}')
        self.group_insights_text.configure(text='\n\n'.join(descriptions) if descriptions else report['insight_status'])
        self.similarity_table.delete(*self.similarity_table.get_children())
        names = {int(group['group_id']): group.get('group_name', '') for group in self.groups}
        for row in report['similarities']:
            row['group_name'] = names.get(row['group_id'], '')
            name = f'Group {row["group_id"]}' + (f' · {row["group_name"]}' if row['group_name'] else '')
            self.similarity_table.insert('', 'end', iid=str(row['group_id']), values=(row['rank'], name,
                                         f'{row["distance"]:.3f}', number(row['similarity_score']),
                                         f'{row["shared_features"]}/{row["available_features"]}'))
        self.similarity_note.configure(text=f'{len(report["similarities"])} groups meet the shared-data requirements.' if report['similarities'] else report['similarity_status'])

    def similarity_selected(self, event=None):
        selected = self.similarity_table.selection()
        if selected:
            group = next((group for group in self.groups if group['group_id'] == selected[0]), None)
            if group is not None:
                self.select_group(group)

    def show_pose_report(self, report):
        self.show_group_comparisons(report)
        self.report_button.configure(state='normal')
        self.pose_status.configure(text=f'{report["frame_count"]:,} video frames · {report["pose_observations"]:,} pose observations\n{report["eligible_motion_pairs"]:,} continuous observation pairs\nConfidence ≥ {report["settings"]["confidence"]:.2f} · Video-aligned sampling')
        parts = report['body_parts']
        eligible = [p for p in parts if p['body_speed_px_s'] is not None and p['body_motion_pairs'] >= max(3, report['eligible_motion_pairs'] * .25) and p['coverage_percent'] >= 50 and p['body_part'] != report['settings']['reference']]
        ordered = sorted(eligible, key=lambda p: p['body_speed_px_s'], reverse=True)
        if ordered:
            active, stable = ordered[0], ordered[-1]
            distinction = sorted((p for p in eligible if p['body_speed_ratio_vs_others'] is not None), key=lambda p: p['body_speed_ratio_vs_others'], reverse=True)
            hint = (f'Most active relative to body\n{active["body_part"]} · {number(active["body_speed_px_s"], " px/s")}\n\n'
                    f'Most stable relative to body\n{stable["body_part"]} · {number(stable["body_speed_px_s"], " px/s")}')
            if distinction:
                part = distinction[0]
                hint += f'\n\nHighest speed ratio vs other groups\n{part["body_part"]} · {number(part["body_speed_ratio_vs_others"], "×")}'
        else:
            hint = 'Not enough reliable adjacent observations to rank body-relative motion. Try a better-tracked reference or adjust confidence.'
        self.insight_text.configure(text=hint)
        chart = self.motion_chart
        chart.delete('all')
        for index, part in enumerate(ordered[:6]):
            y = index * 31 + 10
            chart.create_text(0, y, text=part['body_part'][:25], fill=MUTED, anchor='nw', font=('Arial', 10))
            width = 200 * part['body_speed_px_s'] / max(1e-8, ordered[0]['body_speed_px_s'])
            chart.create_rectangle(0, y+16, max(1, width), y+23, fill=INK, outline='')
            chart.create_text(210, y+15, text=number(part['body_speed_px_s']), fill=INK, anchor='nw', font=('Arial', 10))
        self.parts_table.delete(*self.parts_table.get_children())
        for part in sorted(parts, key=lambda p: -(p['body_speed_px_s'] if p['body_speed_px_s'] is not None else -1)):
            self.parts_table.insert('', 'end', values=(part['body_part'], number(part['median_speed_px_s']), number(part['body_speed_px_s']),
                                                       number(part['coverage_percent']), number(part['body_speed_ratio_vs_others'], '×'),
                                                       number(part['body_range_px']), part['body_motion_pairs']))
        self.pairs_table.delete(*self.pairs_table.get_children())
        for pair in sorted(report['pairs'], key=lambda p: -(p['median_distance_change_px_s'] if p['median_distance_change_px_s'] is not None else -1)):
            self.pairs_table.insert('', 'end', values=(pair['body_part_a'] + ' ↔ ' + pair['body_part_b'], number(pair['median_distance_change_px_s']), number(pair['distance_range_px']), pair['valid_motion_pairs']))
        if report['angles']:
            angle = report['angles'][0]
            self.angle_text.configure(text=f'Median {number(angle["median_angle_deg"], "°")} · Range {number(angle["angle_range_deg"], "°")}\nAngular speed {number(angle["median_angular_speed_deg_s"], "°/s")} · {angle["valid_frames"]} valid frames')
        else:
            self.angle_text.configure(text='Choose three distinct landmarks and update analysis.')

    def import_pose(self):
        from .gui import dialog_filters
        path = filedialog.askopenfilename(parent=self, title='Load DeepLabCut pose CSV', **dialog_filters([('Pose CSV', '*.csv'), ('All files', '*')]))
        if not path:
            return
        output = self.output
        self.pose_status.configure(text='Reading pose CSV…')
        def work():
            try:
                pose = read_pose(Path(path))
                if pose is None:
                    raise InputError('No DLC bodyparts / coords headers found in this CSV.')
                self.events.put(('pose_imported', (output, pose, None)))
            except Exception as error:
                self.events.put(('pose_imported', (output, None, str(error))))
        self.start_background(work)

    def pose_alignment_dialog(self, pose):
        info = self.analysis[2]
        dialog = ctk.CTkToplevel(self)
        dialog.title('Align pose coordinates')
        dialog.geometry('500x420')
        dialog.configure(fg_color=BG)
        dialog.transient(self)
        dialog.grab_set()
        ctk.CTkLabel(dialog, text='Align landmarks with the original video', font=('Arial', 17, 'bold'), text_color=INK).pack(pady=(24, 12))
        ctk.CTkLabel(dialog, text=f'{len(pose.xy):,} pose samples · {info.frames:,} video frames\nUse matching recording, original image dimensions and crop.', wraplength=440, text_color=MUTED).pack(pady=(0, 12))
        ctk.CTkLabel(dialog, text='Pose samples per second').pack()
        rate = ctk.CTkEntry(dialog)
        rate.pack(pady=6)
        inferred = filename_rate(Path(pose.source))
        if inferred is None and len(pose.xy) == info.frames:
            inferred = info.fps
        if inferred is None and len(pose.xy) == len(self.analysis[0]):
            inferred = self.analysis[3]
        if inferred is not None:
            rate.insert(0, str(inferred))
        ctk.CTkLabel(dialog, text='Video start in pose timeline (seconds)').pack()
        offset = ctk.CTkEntry(dialog)
        offset.insert(0, str(self.analysis[4] if len(pose.xy) == len(self.analysis[0]) else 0))
        offset.pack(pady=6)
        def confirm():
            try:
                aligned = align_pose(pose, info, float(rate.get()), float(offset.get()))
                save_pose(self.output, aligned)
            except (InputError, ValueError, OSError) as error:
                messagebox.showerror('Could not align pose', str(error), parent=dialog)
                return
            dialog.destroy()
            self.pose_data = aligned
            self.pose_output = self.output
            self.pose_preferences = {}
            self.viewer(self.selected['group_id'])
        self.button(dialog, 'Use these coordinates', confirm, primary=True).pack(pady=18)

    def save_pose_report(self):
        if self.pose_report is None:
            return
        root = filedialog.askdirectory(parent=self, title='Choose a folder for analysis tables')
        if root:
            try:
                output = export_report(Path(root), self.pose_report, self.selected.get('group_name', ''))
                messagebox.showinfo('Analysis exported', f'Saved tables and settings to:\n{output}', parent=self)
            except (OSError, ValueError) as error:
                messagebox.showerror('Could not export analysis', str(error), parent=self)

    def save_pose_overlay(self):
        if self.pose_data is None or not self.frame_count or self.pose_exporting:
            return
        from .gui import dialog_filters
        filename = Path(group_export_filename(self.selected)).stem + '_pose.mp4'
        target = filedialog.asksaveasfilename(parent=self, initialfile=filename, defaultextension='.mp4', title='Export annotated MP4', **dialog_filters([('MP4 video', '*.mp4')]))
        if not target:
            return
        target = Path(target)
        if target.resolve().parent == self.output.resolve():
            messagebox.showerror('Choose another folder', 'Export outside this results folder to preserve its videos.', parent=self)
            return
        self.pose_exporting = True
        self.overlay_export_button.configure(state='disabled')
        self.pose_status.configure(text='Exporting annotated MP4… You can continue reviewing.')
        pose, parts, threshold, labels = self.pose_data, self.selected_pose_parts(), self.pose_threshold.get(), self.show_part_labels.get()
        source = self.output / f'group_{self.selected["group_id"]}.mp4'
        indices = self.current_pose_indices.copy()
        def work():
            try:
                export_overlay(source, target, pose, indices, parts, threshold, labels, lambda *_: None, self.cancel)
                self.events.put(('pose_exported', (str(target), None)))
            except Exception as error:
                self.events.put(('pose_exported', (None, str(error))))
        self.start_background(work)
