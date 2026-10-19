# Boute

A minimal, local desktop utility for turning B-SOiD behavioral labels into per-group video montages.

## Open Boute

- **Windows:** double-click `start_boute_windows.bat`. The setup script offers a per-user Python installation through winget when needed.
- **macOS:** open `start_boute_mac.command`. If macOS blocks an unsigned downloaded file, use Finder’s Open action. Install Python 3.10+ from python.org if prompted; that distribution includes Tk.
- **Linux:** launch `start_boute_linux.sh` from a terminal, or choose “Run in terminal” in a file manager. Python 3.10+, Tk, venv, and a graphical desktop are required. On Debian/Ubuntu these are supplied by `python3`, `python3-tk`, and `python3-venv`; OpenCV may also require `libgl1`.

The launcher creates a project-local `.venv` and installs dependencies on first launch or when requirements change. First setup requires internet access. Subsequent use is offline. Older `start_bsoid_studio_*` filenames are included as compatibility launchers; the application is named Boute throughout.

The macOS and Linux scripts do not silently install operating-system packages. On macOS the launcher offers the official Python download page when Python is absent. Linux installation depends on the distribution and is explained in the launcher. Launchers require a compatible Python on PATH (Windows also checks `py`).

## Workflow

1. Click **Choose original video** and **Choose B-SOiD results CSV**. For the CSV, look for `labels_pose` anywhere in the filename; this file contains the behavioral group for each frame. The picker initially filters for that name; use **All CSV files** if it was renamed. `bout_lengths` and `transitions_mat` CSVs are summaries and are not needed. You can also drop the video and results CSV together where native dropping works. Click either section to replace its input. Starting a new analysis clears all inputs.
2. Choose **Continue**. If sampling rate or duration is ambiguous, confirm label samples per second and the video’s starting offset within the label timeline.
3. Select a group, play its montage, scrub, choose playback speed, or export an MP4. Orange ticks above the scrubber mark gaps in the original video frames. Hover over a tick to see the source frame numbers, jump distance, and number of skipped frames; click it to seek to the first frame after the jump. **Open folder** reveals the complete run.

Outputs go to `Boute_Exports` in your home directory. Every run receives a unique timestamped directory and never replaces a previous run. Failed or cancelled runs remove their partial outputs. Closing the window cancels processing and waits for video handles to be released.

## Scientific interpretation

**These outputs are frame montages, not continuous behavioral bouts.** Frames are concatenated in source order, so disconnected movements may appear continuous. Audio is not exported. Bout-preserving playback is not implemented.

Boute searches for the `B-SOiD labels` column, supports DLC metadata headers, rejects invalid data rows, and stores only labels rather than pose columns. Negative and nonconsecutive integer labels are retained as separate groups.

When a filename includes a frequency such as `440Hz`, video timestamps map to label timestamps using `floor((frame / video_fps + offset) * label_fps)`. Equal row and frame counts alone do not override a conflicting filename frequency. If no frequency is available and counts match exactly, Boute uses direct frame indexing. Otherwise an explicit alignment dialog is required. Default alignment requires duration agreement within one sampling interval; all mappings must remain inside the available labels. Manual offsets support a video segment from a longer label sequence.

**Matching durations or counts cannot prove that the files describe the same fly or experiment.** Researchers must verify recording identity and temporal correspondence. Boute assumes a constant frame rate and zero-order-held labels. Variable-frame-rate recordings must be converted to a known constant rate before analysis. MP4 encoding uses OpenCV’s MPEG-4 Part 2 encoder (`mp4v`); compatibility depends on installed platform codecs. Odd pixel dimensions are rejected to avoid silent encoder cropping.

Groups present in the labels but absent from the aligned video are listed with zero frames and no MP4. Original dimensions and FPS are preserved. Videos decode sequentially with bounded frame memory; label mappings remain in memory and one encoder is open per represented group.

Each run contains:

- `group_<ID>.mp4` for each represented group.
- `frame_manifest.csv`: group, original frame index, timestamp, and output frame index. All indices are zero-based.
- `group_summary.csv`: frame counts, percentages of source video frames, and montage durations.
- `source_metadata.txt`: input paths, rates, offset, mapping rule, and interpretation notes.
- `source_csvs/`: a copy of the original results CSV.

## Development and validation

Run `.venv/bin/python -m unittest discover -s tests -v` (Windows: `.venv\Scripts\python.exe -m unittest discover -s tests -v`).

Tests cover realistic DLC multi-row headers, invalid/missing labels, nonconsecutive groups, direct and timestamp alignment, segment offsets, incompatible durations, input replacement, cancellation cleanup, and synthetic video export. Synthetic frames have known brightness values so decoding the exported videos verifies chronological identity as well as frame counts. Lossy codec comparisons use a small tolerance.

Desktop checks on each target OS should cover actual OS drag-enter/leave/drop, multi-file dropping, file replacement, playback and scrubbing, folder opening, native save dialogs, shutdown during processing, and double-click setup. Headless or programmatic widget checks do not substitute for native drag-and-drop testing. See `VALIDATION.md` for checks performed in this build environment.

## Structure

- `boute/core.py`: intake, CSV parsing, alignment, sequential export, manifests.
- `boute/gui.py`: minimal interface, queued worker updates, native drop events, playback.
- `bootstrap.py`: dependency-aware virtual environment setup.
- `tests/test_core.py`: executable processing tests.

Icon attribution is in `THIRD_PARTY_NOTICES.md`.
