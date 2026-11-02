"""DeepLabCut import, confidence-aware kinematics, and optional pose rendering.

All measurements use original video time, never adjacent montage frames.
Distances are 2D image pixels; these are descriptive cluster summaries.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
import threading

import cv2
import numpy as np

from .core import InputError, VideoInfo, align, inspect_video
from .comparisons import GroupComparisons, COMPARISON_METHOD


@dataclass
class PoseData:
    names: tuple[str, ...]
    xy: np.ndarray
    confidence: np.ndarray
    source: str = ''
    alignment: str = ''
    sample_indices: np.ndarray | None = None


def read_pose(path: Path) -> PoseData | None:
    """Read DLC CSVs or B-SOiD labels_pose CSVs, including multi-animal headers.

    Absent pose headers return None. Missing/non-numeric observations remain NaN;
    they are excluded from both rendering and analysis rather than interpolated.
    """
    bodyparts = individuals = None
    chunks, pending = [], []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        for line, row in enumerate(reader):
            lowered = [cell.strip().lower() for cell in row]
            if 'individuals' in lowered:
                individuals = row
            if 'bodyparts' in lowered:
                bodyparts = row
            if 'coords' in lowered and bodyparts is not None:
                coordinates = row
                break
            if line >= 99:
                return None
        else:
            return None
        index_column = next(i for i, value in enumerate(coordinates) if value.strip().lower() == 'coords')
        columns = []
        names = []
        for col, coord in enumerate(coordinates):
            if coord.strip().lower() != 'x':
                continue
            if col + 2 >= len(coordinates) or [c.strip().lower() for c in coordinates[col:col+3]] != ['x', 'y', 'likelihood']:
                raise InputError('Pose CSV requires x, y, and likelihood columns for each body part.')
            name = bodyparts[col].strip() if col < len(bodyparts) else ''
            if individuals is not None and col < len(individuals):
                name = individuals[col].strip() + ' / ' + name
            if not name or name in names:
                raise InputError('Pose CSV has missing or duplicate body-part names. Select one scorer/model per CSV.')
            names.append(name)
            columns.extend((col, col+1, col+2))
        if not names:
            raise InputError('No x/y body-part coordinates were found in the pose CSV.')
        for sample_index, row in enumerate(reader):
            try:
                frame_index = float(row[index_column])
            except (ValueError, IndexError):
                raise InputError('Pose CSV must contain a complete, zero-based numeric video-frame index.') from None
            if not math.isfinite(frame_index) or frame_index != sample_index:
                raise InputError('Pose frame indices are missing, duplicated, or out of order. Use the complete DLC video export; sparse frame tables cannot be aligned safely.')
            values = []
            for column in columns:
                try:
                    values.append(float(row[column]))
                except (ValueError, IndexError):
                    values.append(float('nan'))
            pending.append(values)
            if len(pending) >= 10000:
                chunks.append(np.asarray(pending, dtype=np.float32))
                pending = []
        if pending:
            chunks.append(np.asarray(pending, dtype=np.float32))
    if not chunks:
        raise InputError('The pose CSV contains no observations.')
    values = np.concatenate(chunks).reshape(-1, len(names), 3)
    return PoseData(tuple(names), values[:, :, :2].copy(), values[:, :, 2].copy(), str(path.resolve()))


def align_pose(pose: PoseData, info: VideoInfo, rate: float | None, offset: float) -> PoseData:
    indices = align(np.arange(len(pose.xy)), info, rate, offset, manual=True)
    return PoseData(pose.names, pose.xy[indices].copy(), pose.confidence[indices].copy(),
                    pose.source, f'Pose FPS: {rate if rate is not None else info.fps}; offset: {offset} s; video FPS: {info.fps}',
                    indices if pose.sample_indices is None else pose.sample_indices[indices])


def save_pose(output: Path, pose: PoseData):
    # No Python objects/pickle; files remain portable across OSes.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, names=np.asarray(pose.names), xy=pose.xy, confidence=pose.confidence,
                     source=np.asarray(pose.source), alignment=np.asarray(pose.alignment),
                     sample_indices=np.arange(len(pose.xy)) if pose.sample_indices is None else pose.sample_indices)
        temporary.replace(output / 'pose_data.npz')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_pose(output: Path) -> PoseData | None:
    path = output / 'pose_data.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return PoseData(tuple(map(str, data['names'])), data['xy'].copy(), data['confidence'].copy(),
                        str(data['source']), str(data['alignment']),
                        data['sample_indices'].copy() if 'sample_indices' in data else None)


def valid_points(pose: PoseData, threshold: float, info: VideoInfo) -> np.ndarray:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise InputError('Confidence must be between 0 and 1.')
    xy = pose.xy
    return (np.isfinite(xy).all(axis=2) & np.isfinite(pose.confidence) &
            (pose.confidence >= threshold) & (pose.confidence <= 1) &
            (xy[:, :, 0] >= 0) & (xy[:, :, 0] < info.width) &
            (xy[:, :, 1] >= 0) & (xy[:, :, 1] < info.height))


def montage_indices(output: Path, group_id: str) -> np.ndarray:
    indices = []
    with (output / 'frame_manifest.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['group_id'] == str(group_id):
                indices.append(int(row['original_video_frame']))
    return np.asarray(indices, dtype=np.int64)


def _value(values, mask, operation=np.median):
    usable = values[mask & np.isfinite(values)]
    return float(operation(usable)) if len(usable) else None


def _range(xy, mask):
    usable = xy[mask & np.isfinite(xy).all(axis=1)]
    if not len(usable):
        return None
    return float(np.linalg.norm(np.percentile(usable, 95, axis=0) - np.percentile(usable, 5, axis=0)))


def analyze_pose(pose: PoseData, groups: np.ndarray, info: VideoInfo, group_id: int,
                 threshold: float = .8, reference: int = 0, axis: int | None = None,
                 smoothing: int = 3, angle: tuple[int, int, int] | None = None,
                 cancel: threading.Event | None = None) -> dict:
    if len(pose.xy) != info.frames or len(groups) != info.frames:
        raise InputError('Pose, group labels, and video must have the same aligned frame count.')
    if smoothing not in (1, 3, 5):
        raise InputError('Smoothing must use 1, 3, or 5 video frames.')
    if not 0 <= reference < len(pose.names) or (axis is not None and (axis == reference or not 0 <= axis < len(pose.names))):
        raise InputError('Choose distinct, available body reference and axis landmarks.')
    original_frame_count = int((groups == group_id).sum())
    original_runs = np.cumsum(np.r_[True, groups[1:] != groups[:-1]])
    source_frames = np.arange(info.frames)
    if pose.sample_indices is not None:
        if len(pose.sample_indices) != info.frames or np.any(np.diff(pose.sample_indices) < 0):
            raise InputError('Pose sample indices must be chronological and match the video.')
        # Repeated sample-and-hold overlays are not repeated measurements.
        source_frames = np.r_[0, np.flatnonzero(np.diff(pose.sample_indices) != 0) + 1]
    valid = valid_points(pose, threshold, info)[source_frames]
    xy = pose.xy[source_frames].astype(np.float64, copy=True)
    xy[~valid] = np.nan
    groups = groups[source_frames]
    selected = groups == group_id
    continuity = original_runs[source_frames[1:]] == original_runs[source_frames[:-1]]
    group_pairs = selected[1:] & selected[:-1] & continuity
    time_delta = np.diff(source_frames) / info.fps
    other_pairs = continuity & ~selected[1:]
    if smoothing > 1 and len(xy) >= smoothing:
        from numpy.lib.stride_tricks import sliding_window_view
        runs = np.cumsum(np.r_[True, ~continuity])
        same_run = runs[smoothing-1:] == runs[:1-smoothing]
        half = smoothing // 2
        for part in range(len(pose.names)):
            if cancel is not None and cancel.is_set():
                raise InputError('Pose analysis cancelled.')
            windows = sliding_window_view(xy[:, part, :], smoothing, axis=0)
            good = sliding_window_view(valid[:, part], smoothing).all(axis=1) & same_run
            medians = np.median(windows, axis=2)
            centers = np.flatnonzero(good) + half
            xy[centers, part] = medians[good]
    relative = xy - xy[:, reference:reference+1]
    body_valid = valid & valid[:, reference:reference+1]
    if axis is not None:
        vector = xy[:, axis] - xy[:, reference]
        length = np.linalg.norm(vector, axis=1)
        good_axis = valid[:, axis] & (length > 1e-8)
        direction = vector / np.where(length > 1e-8, length, 1)[:, None]
        x = (relative * direction[:, None, :]).sum(axis=2)
        y = relative[:, :, 1] * direction[:, None, 0] - relative[:, :, 0] * direction[:, None, 1]
        relative = np.stack((x, y), axis=2)
        body_valid &= good_axis[:, None]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=2) / time_delta[:, None]
    body_speed = np.linalg.norm(np.diff(relative, axis=0), axis=2) / time_delta[:, None]
    comparisons = GroupComparisons(groups, continuity, group_id, cancel)
    parts = []
    for index, name in enumerate(pose.names):
        if cancel is not None and cancel.is_set():
            raise InputError('Pose analysis cancelled.')
        paired = valid[1:, index] & valid[:-1, index]
        body_paired = body_valid[1:, index] & body_valid[:-1, index]
        median = _value(speed[:, index], paired & group_pairs)
        baseline = _value(speed[:, index], paired & other_pairs)
        body_median = _value(body_speed[:, index], body_paired & group_pairs)
        body_baseline = _value(body_speed[:, index], body_paired & other_pairs)
        if index != reference:
            comparisons.add(f'body_speed:{name}', 'movement', name, 'body_motion',
                            body_speed[:, index], body_paired, body_valid[:, index], 'px/s', .1)
        else:
            comparisons.add(f'reference_speed:{name}', 'movement', name + ' (reference)', 'reference_motion',
                            speed[:, index], paired, valid[:, index], 'px/s', .1)
        parts.append(dict(body_part=name, valid_frames=int((valid[:, index] & selected).sum()),
                          coverage_percent=float((valid[:, index] & selected).sum() / max(1, selected.sum()) * 100),
                          valid_motion_pairs=int((paired & group_pairs).sum()),
                          body_motion_pairs=int((body_paired & group_pairs).sum()),
                          median_speed_px_s=median, p90_speed_px_s=_value(speed[:, index], paired & group_pairs, lambda v: np.percentile(v, 90)),
                          body_speed_px_s=body_median, movement_range_px=_range(xy[:, index], valid[:, index] & selected),
                          body_range_px=_range(relative[:, index], body_valid[:, index] & selected),
                          other_groups_speed_px_s=baseline, other_groups_body_speed_px_s=body_baseline,
                          speed_ratio_vs_others=median / baseline if median is not None and baseline is not None and baseline > 1e-8 else None,
                          body_speed_ratio_vs_others=body_median / body_baseline if body_median is not None and body_baseline is not None and body_baseline > 1e-8 else None))
    pairs = []
    for a, b in itertools.combinations(range(len(pose.names)), 2):
        if cancel is not None and cancel.is_set():
            raise InputError('Pose analysis cancelled.')
        good = valid[:, a] & valid[:, b]
        distance = np.linalg.norm(xy[:, a] - xy[:, b], axis=1)
        available_pairs = good[1:] & good[:-1]
        paired = available_pairs & group_pairs
        subject = pose.names[a] + ' ↔ ' + pose.names[b]
        comparisons.add(f'pair_motion:{a}:{b}', 'relationships', subject, 'pair_motion',
                        np.abs(np.diff(distance)) / time_delta, available_pairs, good, 'px/s', .1)
        comparisons.add(f'pair_distance:{a}:{b}', 'posture', subject, 'pair_posture',
                        distance, good, good, 'px', 1., motion=False)
        samples = distance[good & selected]
        pairs.append(dict(body_part_a=pose.names[a], body_part_b=pose.names[b],
                          valid_frames=int(len(samples)), valid_motion_pairs=int(paired.sum()),
                          median_distance_px=float(np.median(samples)) if len(samples) else None,
                          distance_range_px=float(np.percentile(samples, 90) - np.percentile(samples, 10)) if len(samples) else None,
                          median_distance_change_px_s=_value(np.abs(np.diff(distance)) / time_delta, paired)))
    angles = []
    if angle is not None:
        a, b, c = angle
        if len(set(angle)) != 3 or any(i < 0 or i >= len(pose.names) for i in angle):
            raise InputError('Select three distinct landmarks for a joint angle (middle landmark is the vertex).')
        u, v = xy[:, a] - xy[:, b], xy[:, c] - xy[:, b]
        denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        good = valid[:, a] & valid[:, b] & valid[:, c] & (denominator > 1e-8)
        degrees = np.degrees(np.arccos(np.clip((u * v).sum(axis=1) / np.where(denominator > 1e-8, denominator, 1), -1, 1)))
        subject = f'{pose.names[a]} → {pose.names[b]} → {pose.names[c]}'
        comparisons.add(f'angle_speed:{a}:{b}:{c}', 'joint angle', subject, 'angle_motion',
                        np.abs(np.diff(degrees)) / time_delta, good[1:] & good[:-1], good, '°/s', 1.)
        comparisons.add(f'angle:{a}:{b}:{c}', 'joint angle', subject, 'angle_posture',
                        degrees, good, good, '°', 5., motion=False)
        samples = degrees[good & selected]
        angles.append(dict(landmark_a=pose.names[a], vertex=pose.names[b], landmark_c=pose.names[c],
                           valid_frames=int(len(samples)), valid_motion_pairs=int((good[1:] & good[:-1] & group_pairs).sum()), median_angle_deg=float(np.median(samples)) if len(samples) else None,
                           angle_range_deg=float(np.percentile(samples, 90)-np.percentile(samples, 10)) if len(samples) else None,
                           median_angular_speed_deg_s=_value(np.abs(np.diff(degrees)) / time_delta, good[1:] & good[:-1] & group_pairs)))
    return dict(group_id=int(group_id), frame_count=original_frame_count, pose_observations=int(selected.sum()), eligible_motion_pairs=int(group_pairs.sum()),
                settings=dict(confidence=threshold, smoothing_observations=smoothing, reference=pose.names[reference],
                              axis=pose.names[axis] if axis is not None else None, fps=info.fps, video_width=info.width, video_height=info.height,
                              source=pose.source, alignment=pose.alignment, comparison_method=COMPARISON_METHOD,
                              units='2D pixels and seconds; video-aligned sampling with repeated pose samples counted once',
                              method='Repeated held pose samples counted once; time deltas use original video timestamps. Median filter only within fully valid, continuous same-group observation windows. Motion excludes group boundaries and missing observations. Body coordinates subtract reference; optional axis also removes rotation. Ranges: body-part p95-p5 diagonal, pair/angle p90-p10. No missing-point interpolation. No physical-scale calibration.'),
                body_parts=parts, pairs=pairs, angles=angles, **comparisons.finish())


COLORS = ((235, 99, 37), (94, 197, 34), (246, 130, 139), (255, 174, 20), (72, 180, 244), (200, 100, 200))  # BGR


def draw_pose(frame: np.ndarray, pose: PoseData, source_index: int, parts: list[int], threshold: float, labels: bool = False) -> np.ndarray:
    result = frame.copy()
    if not 0 <= source_index < len(pose.xy):
        return result
    height, width = frame.shape[:2]
    for index in parts:
        xy, confidence = pose.xy[source_index, index], pose.confidence[source_index, index]
        if not np.isfinite(xy).all() or not np.isfinite(confidence) or not threshold <= confidence <= 1:
            continue
        x, y = map(float, xy)
        if not (0 <= x < width and 0 <= y < height):
            continue
        point = (int(round(x)), int(round(y)))
        color = COLORS[index % len(COLORS)]
        cv2.circle(result, point, 5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(result, point, 3, color, -1, cv2.LINE_AA)
        if labels:
            cv2.putText(result, pose.names[index], (point[0]+7, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, pose.names[index], (point[0]+7, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    return result


def export_overlay(source: Path, target: Path, pose: PoseData, indices: np.ndarray, parts: list[int],
                   threshold: float, labels: bool, progress, cancel: threading.Event):
    if source.resolve() == target.resolve() or target.suffix.lower() != '.mp4':
        raise InputError('Choose a separate MP4 file for the annotated export.')
    info = inspect_video(source)
    if len(indices) != info.frames or (len(indices) and (indices.min() < 0 or indices.max() >= len(pose.xy))):
        raise InputError('The pose data does not match this montage.')
    valid_points(pose, threshold, info)
    cap = cv2.VideoCapture(str(source))
    writer = None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.mp4', delete=False) as stream:
            temporary = Path(stream.name)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), info.fps, (info.width, info.height))
        if not writer.isOpened():
            raise InputError('The annotated MP4 encoder could not start.')
        for frame_number, original in enumerate(indices):
            if cancel.is_set():
                raise InputError('Annotated export cancelled.')
            ok, frame = cap.read()
            if not ok:
                raise InputError('The group video could not be fully decoded.')
            writer.write(draw_pose(frame, pose, int(original), parts, threshold, labels))
            if frame_number % 50 == 0:
                progress((frame_number+1)/len(indices), 'Exporting annotated MP4…')
        writer.release()
        writer = None
        if inspect_video(temporary).frames != info.frames:
            raise InputError('Annotated MP4 verification failed.')
        if cancel.is_set():
            raise InputError('Annotated export cancelled.')
        temporary.replace(target)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_report(root: Path, report: dict, group_name: str) -> Path:
    output = root / f'group_{report["group_id"]}_analysis_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir()
    try:
        settings = dict(report['settings'], group_id=report['group_id'], group_name=group_name,
                        frame_count=report['frame_count'], eligible_motion_pairs=report['eligible_motion_pairs'], pose_observations=report['pose_observations'],
                        insight_status=report['insight_status'], similarity_status=report['similarity_status'],
                        statistical_method=report['statistical_method'], tested_features=report['tested_features'])
        (output / 'analysis_settings.json').write_text(json.dumps(settings, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        for name in ('body_parts', 'pairs', 'angles', 'insights', 'similarities', 'feature_statistics'):
            rows = report[name]
            if not rows:
                continue
            with (output / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return output
    except Exception:
        import shutil
        shutil.rmtree(output)
        raise
