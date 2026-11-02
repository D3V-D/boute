from __future__ import annotations

import csv
import math
import re
import shutil
import threading
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class InputError(ValueError):
    pass


class AlignmentRequired(InputError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def read_labels(path: Path) -> np.ndarray:
    values: list[int] = []
    column = None
    scorer_column = None
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if column is None:
                matches = [i for i, cell in enumerate(row) if cell.strip().casefold() == 'b-soid labels']
                if matches:
                    if len(matches) != 1:
                        raise InputError('The CSV contains multiple B-SOiD labels columns.')
                    column = matches[0]
                    scorer_column = next((i for i, cell in enumerate(row) if cell.strip().casefold() == 'scorer'), None)
                elif line > 100:
                    break
                continue
            if not values and scorer_column is not None and len(row) > scorer_column and row[scorer_column].strip().casefold() in {'bodyparts', 'coords'}:
                continue
            value = row[column].strip() if len(row) > column else ''
            # DLC's additional multi-index headers occupy only the initial rows.
            if not values and line <= 10 and value.lower() in {'', 'coords', 'bodyparts', 'scorer', 'likelihood', 'x', 'y'} and any(cell.strip().lower() in {'bodyparts', 'coords', 'scorer'} for cell in row):
                continue
            try:
                numeric = float(value)
                if not math.isfinite(numeric) or not numeric.is_integer() or abs(numeric) > 2**53:
                    raise ValueError
                values.append(int(numeric))
            except ValueError:
                raise InputError(f'Invalid or missing behavioral label at CSV row {line}: {value!r}.') from None
    if column is None:
        raise InputError('No column named “B-SOiD labels” was found in the CSV.')
    if not values:
        raise InputError('The labels CSV contains no classified frames.')
    return np.asarray(values, dtype=np.int64)


def inspect_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        values = [float(cap.get(prop)) for prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                                                   cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        if not cap.isOpened() or any(not math.isfinite(value) or value <= 0 for value in values):
            raise InputError('Cannot read this video. Try a valid MP4, MOV, AVI, or MKV file.')
        info = VideoInfo(values[0], *(int(value) for value in values[1:]))
        if min(info.frames, info.width, info.height) <= 0:
            raise InputError('The video reports invalid frame counts or dimensions.')
        return info
    finally:
        cap.release()


def filename_rate(path: Path) -> float | None:
    match = re.search(r'(\d+(?:\.\d+)?)\s*Hz', path.name, re.I)
    return float(match[1]) if match else None


def align(labels: np.ndarray, info: VideoInfo, rate: float | None, offset: float = 0,
          manual: bool = False) -> np.ndarray:
    if not math.isfinite(offset) or offset < 0:
        raise InputError('Starting offset must be a finite, nonnegative number of seconds.')
    if rate is None:
        if len(labels) == info.frames and offset == 0:
            return labels.copy()
        raise AlignmentRequired('Label sampling rate is unknown. Confirm the label FPS and video starting offset.')
    if not math.isfinite(rate) or rate <= 0:
        raise InputError('Label FPS must be a positive, finite number.')
    duration = len(labels) / rate
    tolerance = max(1 / rate, 1 / info.fps)
    if not manual and (abs(duration - info.duration) > tolerance or offset != 0):
        raise AlignmentRequired(f'Video duration: {info.duration:.3f} s. Labels: {duration:.3f} s at {rate:g} Hz. Confirm sampling rate and segment offset.')
    indices = np.floor((np.arange(info.frames, dtype=np.float64) / info.fps + offset) * rate + 1e-8).astype(np.int64)
    if indices[-1] >= len(labels):
        raise InputError('The video extends beyond the available labels at this FPS and offset.')
    return labels[indices]


def classify(paths: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in paths:
        path = Path(value)
        if not path.is_file():
            if path.is_dir():
                raise InputError(f'Please choose a video or CSV file rather than a folder:\n{path.absolute()}')
            raise InputError(
                f'Cannot find or access this file:\n{path.absolute()}\n\n'
                'Try clicking the drop area and choosing the original file from disk.'
            )
        if path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mpeg', '.mpg'}:
            key = 'video'
        elif path.suffix.lower() == '.csv':
            if any(k in path.name.lower() for k in ('bout_lengths', 'transitions_mat')):
                raise InputError('This is a summary CSV, which Boute does not need. Choose the CSV with “labels_pose” in its filename; it contains the behavioral group for each frame.')
            key = 'labels'
        else:
            raise InputError(f'Unsupported file: {path.name}')
        if key in result:
            raise InputError(f'Please select only one {key.replace("_", " ")} file at a time.')
        result[key] = path
    return result


def merge_groups(labels: np.ndarray, mapped: np.ndarray,
                 selected: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine selected current groups under their lowest ID without changing inputs."""
    groups = set(selected)
    if len(groups) < 2:
        raise InputError('Select at least two groups to merge.')
    if not groups.issubset(set(map(int, np.unique(labels)))):
        raise InputError('One or more selected groups are no longer available.')
    target = min(groups)
    merged_labels, merged_mapped = labels.copy(), mapped.copy()
    merged_labels[np.isin(labels, list(groups))] = target
    merged_mapped[np.isin(mapped, list(groups))] = target
    return merged_labels, merged_mapped, target


def rename_group(output: Path, group_id: str, name: str) -> str:
    """Atomically persist a researcher label without touching group IDs or video."""
    name = name.strip()
    if len(name) > 80 or any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise InputError('Use a name of up to 80 characters on a single line.')
    summary = output / 'group_summary.csv'
    with summary.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    matching = next((row for row in rows if row['group_id'] == str(group_id)), None)
    if matching is None:
        raise InputError('This group is no longer available.')
    if 'group_name' not in fields:
        fields.append('group_name')
    matching['group_name'] = name
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=output, prefix='.group_summary_', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        # Close the temporary file before replacing it, including on Windows.
        temporary.replace(summary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return name


def group_export_filename(group: dict[str, str]) -> str:
    """Suggest a descriptive export filename safe on Windows, macOS, and Linux."""
    name = re.sub(r'[^\w.-]+', '_', group.get('group_name', '')).strip('._')
    name = name.encode('utf-8')[:90].decode('utf-8', errors='ignore').rstrip('._')
    return f'group_{group["group_id"]}' + (f'_{name}' if name else '') + '.mp4'


def generate(video: Path, csv_path: Path, optional: list[Path], labels: np.ndarray,
             mapped: np.ndarray, info: VideoInfo, root: Path, rate: float | None,
             offset: float, progress: Callable[[float, str], None], cancel: threading.Event,
             group_mapping: dict[int, int] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # Filesystem component limits count bytes on Linux/macOS, not Unicode characters.
    stem = re.sub(r'[^\w.-]+', '_', video.stem).encode('utf-8')[:90].decode('utf-8', errors='ignore')
    name = stem + '_' + datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')
    output = root / name
    output.mkdir()  # Never replace a previous run.
    writers = {}
    counts = {int(g): 0 for g in np.unique(labels)}
    cap = cv2.VideoCapture(str(video))
    complete = False
    try:
        if info.width % 2 or info.height % 2:
            raise InputError('This video has odd pixel dimensions, which MP4 encoding would crop. Convert it to even dimensions before processing.')
        for group in np.unique(mapped):
            writer = cv2.VideoWriter(str(output / f'group_{group}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), info.fps, (info.width, info.height))
            if not writer.isOpened():
                writer.release()
                raise InputError('The MP4 encoder could not start. Check available disk space and OpenCV installation.')
            writers[int(group)] = writer
        with (output / 'frame_manifest.csv').open('w', encoding='utf-8', newline='') as stream:
            manifest = csv.writer(stream)
            manifest.writerow(['group_id', 'original_video_frame', 'original_video_timestamp', 'exported_group_frame'])
            for index, group_value in enumerate(mapped):
                if cancel.is_set():
                    raise InputError('Analysis cancelled.')
                ok, frame = cap.read()
                if not ok:
                    raise InputError(f'Video decoding stopped at frame {index}; expected {info.frames} frames. No partial results were retained.')
                group = int(group_value)
                writers[group].write(frame)
                manifest.writerow([group, index, f'{index / info.fps:.9f}', counts[group]])
                counts[group] += 1
                if index % 50 == 0:
                    progress((index + 1) / info.frames, f'Generating Group {group}…')
            if cap.read()[0]:
                raise InputError('The video contains more frames than reported by its container. Remux the recording and retry.')
        for writer in writers.values():
            writer.release()
        writers.clear()
        for group, count in counts.items():
            if count:
                exported = inspect_video(output / f'group_{group}.mp4')
                if exported.frames != count:
                    raise InputError(f'Output verification failed for Group {group}. Check disk space.')
        with (output / 'group_summary.csv').open('w', encoding='utf-8', newline='') as stream:
            summary = csv.writer(stream)
            summary.writerow(['group_id', 'frame_count', 'percentage', 'duration_seconds', 'group_name'])
            for group, count in sorted(counts.items()):
                summary.writerow([group, count, count / info.frames * 100, count / info.fps, ''])
        sources = output / 'source_csvs'
        sources.mkdir()
        for source in [csv_path, *optional]:
            shutil.copy2(source, sources / source.name)
        (output / 'source_metadata.txt').write_text(
            f'Boute\nVideo: {video.resolve()}\nLabels: {csv_path.resolve()}\nVideo FPS: {info.fps}\n'
            f'Video frames: {info.frames}\nDimensions: {info.width} x {info.height}\nLabel count: {len(labels)}\n'
            f'Label FPS: {rate if rate is not None else "direct frame correspondence"}\nOffset seconds: {offset}\n'
            'Frame indices are zero-based. Offset is video start time in the label timeline.\n'
            'Mapping uses floor((video_frame / video_fps + offset) * label_fps).\n'
            'MP4s are silent frame montages, not continuous behavioral bouts. Disconnected movements may appear continuous.\n'
            'Matching durations cannot establish the same fly or experiment. Input identity must be verified by the researcher.\n', encoding='utf-8')
        if group_mapping is not None:
            with (output / 'group_mapping.csv').open('w', encoding='utf-8', newline='') as stream:
                mapping = csv.writer(stream)
                mapping.writerow(['original_group_id', 'merged_group_id'])
                mapping.writerows(sorted(group_mapping.items()))
        complete = True
        progress(1, 'Ready')
        return output
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()
        if not complete:
            shutil.rmtree(output)


def merge_results(previous: Path, selected: list[int], info: VideoInfo,
                  progress: Callable[[float, str], None], cancel: threading.Event) -> Path:
    """Interleave only selected montages, copying all other videos byte for byte."""
    groups = set(selected)
    with (previous / 'group_summary.csv').open(encoding='utf-8', newline='') as stream:
        summary = {int(row['group_id']): row for row in csv.DictReader(stream)}
    if len(groups) < 2 or not groups.issubset(summary):
        raise InputError('Select at least two available groups to merge.')
    target = min(groups)
    selected_frames = sum(int(summary[group]['frame_count']) for group in groups)
    stem = previous.name.encode('utf-8')[:90].decode('utf-8', errors='ignore')
    output = previous.parent / (stem + '_merge_' + datetime.now().strftime('%Y-%m-%d_%H%M%S_%f'))
    output.mkdir()
    captures = {}
    writer = None
    complete = False
    consumed = {group: 0 for group in groups}
    counts = {group: 0 for group in summary if group not in groups}
    counts[target] = 0

    def check_cancel():
        if cancel.is_set():
            raise InputError('Merge cancelled.')

    try:
        check_cancel()
        for group, row in summary.items():
            check_cancel()
            if not int(row['frame_count']):
                continue
            source = previous / f'group_{group}.mp4'
            if group not in groups:
                progress(0, f'Copying unchanged Group {group}…')
                shutil.copy2(source, output / source.name)
                continue
            exported = inspect_video(source)
            if (exported.frames != int(row['frame_count']) or
                    (exported.width, exported.height) != (info.width, info.height) or
                    not math.isclose(exported.fps, info.fps, rel_tol=1e-4)):
                raise InputError(f'Group {group} no longer matches its results metadata.')
            captures[group] = cv2.VideoCapture(str(source))
        if selected_frames:
            writer = cv2.VideoWriter(str(output / f'group_{target}.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), info.fps,
                                     (info.width, info.height))
            if not writer.isOpened():
                raise InputError('The MP4 encoder could not start. Check available disk space and OpenCV installation.')
        with (previous / 'frame_manifest.csv').open(encoding='utf-8', newline='') as source, \
                (output / 'frame_manifest.csv').open('w', encoding='utf-8', newline='') as dest:
            manifest = csv.DictReader(source)
            rewritten = csv.DictWriter(dest, fieldnames=manifest.fieldnames)
            rewritten.writeheader()
            last_source = -1
            for row in manifest:
                check_cancel()
                group = int(row['group_id'])
                source_index = int(row['original_video_frame'])
                if source_index <= last_source or group not in summary:
                    raise InputError('The frame manifest is inconsistent; the previous results were retained.')
                last_source = source_index
                if group in groups:
                    # Each selected decoder advances sequentially. The manifest supplies
                    # the chronological interleave, including previously merged groups.
                    if group not in captures or int(row['exported_group_frame']) != consumed[group]:
                        raise InputError(f'The frame manifest does not match Group {group}.')
                    ok, frame = captures[group].read()
                    if not ok:
                        raise InputError(f'Could not decode Group {group} at frame {consumed[group]}.')
                    writer.write(frame)
                    consumed[group] += 1
                    row['group_id'] = str(target)
                    row['exported_group_frame'] = str(counts[target])
                    counts[target] += 1
                    if counts[target] % 50 == 0 or counts[target] == selected_frames:
                        progress(counts[target] / max(1, selected_frames), f'Merging into Group {target}…')
                else:
                    counts[group] += 1
                rewritten.writerow(row)
        for group, row in summary.items():
            count = consumed[group] if group in groups else counts[group]
            if count != int(row['frame_count']):
                raise InputError(f'The frame manifest does not match the frame count for Group {group}.')
        if writer is not None:
            writer.release()
            writer = None
            if inspect_video(output / f'group_{target}.mp4').frames != selected_frames:
                raise InputError('Merged video verification failed. Check disk space.')
        check_cancel()
        total = sum(counts.values())
        with (output / 'group_summary.csv').open('w', encoding='utf-8', newline='') as stream:
            summary_writer = csv.writer(stream)
            summary_writer.writerow(['group_id', 'frame_count', 'percentage', 'duration_seconds', 'group_name'])
            for group, count in sorted(counts.items()):
                summary_writer.writerow([group, count, count / max(1, total) * 100, count / info.fps,
                                         summary[group].get('group_name', '')])
        mapping = {group: group for group in summary}
        mapping_path = previous / 'group_mapping.csv'
        if mapping_path.exists():
            with mapping_path.open(encoding='utf-8', newline='') as stream:
                mapping = {int(row['original_group_id']): int(row['merged_group_id'])
                           for row in csv.DictReader(stream)}
        with (output / 'group_mapping.csv').open('w', encoding='utf-8', newline='') as stream:
            mapping_writer = csv.writer(stream)
            mapping_writer.writerow(['original_group_id', 'merged_group_id'])
            mapping_writer.writerows((original, target if current in groups else current)
                                    for original, current in sorted(mapping.items()))
        shutil.copy2(previous / 'source_metadata.txt', output / 'source_metadata.txt')
        shutil.copytree(previous / 'source_csvs', output / 'source_csvs')
        check_cancel()
        complete = True
        progress(1, 'Ready')
        return output
    finally:
        for capture in captures.values():
            capture.release()
        if writer is not None:
            writer.release()
        if not complete:
            shutil.rmtree(output)
