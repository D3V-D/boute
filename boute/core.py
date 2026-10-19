from __future__ import annotations

import csv
import math
import re
import shutil
import threading
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
        info = VideoInfo(float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not cap.isOpened() or not math.isfinite(info.fps) or info.fps <= 0 or min(info.frames, info.width, info.height) <= 0:
            raise InputError('Cannot read this video. Try a valid MP4, MOV, AVI, or MKV file.')
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
    if offset + info.duration > duration + 1e-8:
        raise InputError('The video extends beyond the available labels at this FPS and offset.')
    indices = np.floor((np.arange(info.frames, dtype=np.float64) / info.fps + offset) * rate + 1e-8).astype(np.int64)
    if indices[-1] >= len(labels):
        raise InputError('Alignment refers to labels outside the CSV.')
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


def generate(video: Path, csv_path: Path, optional: list[Path], labels: np.ndarray,
             mapped: np.ndarray, info: VideoInfo, root: Path, rate: float | None,
             offset: float, progress: Callable[[float, str], None], cancel: threading.Event) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^\w.-]+', '_', video.stem)[:90] + '_' + datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')
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
        with (output / 'frame_manifest.csv').open('w', newline='') as stream:
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
        with (output / 'group_summary.csv').open('w', newline='') as stream:
            summary = csv.writer(stream)
            summary.writerow(['group_id', 'frame_count', 'percentage', 'duration_seconds'])
            for group, count in sorted(counts.items()):
                summary.writerow([group, count, count / info.frames * 100, count / info.fps])
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
        complete = True
        progress(1, 'Ready')
        return output
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()
        if not complete:
            shutil.rmtree(output)
