# Boute 🪰🤸

Small desktop tool that takes a behavior video and its B-SOiD labels and makes one montage video per behavior group.

**Note:** a montage is just all the frames from one group stitched together in order. It is not a set of continuous bouts, so unrelated movements can look like one smooth clip. No audio is exported.

## Running it

- **Windows:** double-click `start_boute_windows.bat`. If Python is missing, it offers to install it through winget.
- **macOS:** open `start_boute_mac.command`. If macOS blocks it, right-click and choose Open. You'll need Python 3.10+ from python.org.
- **Linux:** run `start_boute_linux.sh` from a terminal. You need Python 3.10+, Tk, venv, and a desktop session. On Debian/Ubuntu that's `python3 python3-tk python3-venv`, and possibly `libgl1` for OpenCV.

The first launch makes a `.venv` in the project folder and installs dependencies, so you need internet once. After that it works offline.

## Using it

1. Choose the original video and the B-SOiD results CSV. The CSV is the one with `labels_pose` in the filename (the `bout_lengths` and `transitions_mat` files aren't needed). If you renamed it, switch the picker to "All CSV files". You can also drag both files in at once (experimental).
2. Hit **Continue**. If Boute can't figure out the label rate or timing on its own, it will ask for the label samples per second and where the video starts within the label timeline.
3. Pick a group, play it, scrub, change speed, or export an MP4.

Orange ticks above the scrubber mark gaps in the original video frames. Hover for the source frame numbers and how many frames were skipped, or click to jump past the gap.

## Output

Everything goes to `~/Boute_Exports/`, in a new timestamped folder per run. Nothing gets overwritten, and cancelled or failed runs clean up after themselves.

Each run folder has:

- `group_<ID>.mp4` for each group that shows up in the video
- `frame_manifest.csv` with group, original frame index, timestamp, and output frame index (all zero-based)
- `group_summary.csv` with frame counts, percent of video, and montage duration
- `source_metadata.txt` with input paths, rates, offset, and the mapping rule used
- `source_csvs/` with a copy of your results CSV

Groups that are in the labels but not in the video are listed with zero frames and get no MP4.

## How labels line up with video

- If the filename has a frequency like `440Hz`, Boute maps frames to labels with `floor((frame / video_fps + offset) * label_fps)`.
- With no frequency in the name and matching row/frame counts, it uses direct frame indexing.
- Otherwise you'll get an alignment dialog. Durations have to agree within one sampling interval, unless you set a manual offset (useful when the video is a segment of a longer recording).

## Limitations

- Matching counts or durations don't prove the files are from the same fly or session. Check that yourself.
- Constant frame rate only. Convert variable-frame-rate videos first.
- Export uses OpenCV's `mp4v` encoder, so playback depends on your system's codecs. Videos with odd pixel dimensions are rejected.
- Original resolution and FPS are kept.

## Tests

```
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, use `.venv\Scripts\python.exe` instead. Drag-and-drop and the native dialogs haven't been automated, so check those by hand on each OS. See `VALIDATION.md` for what was checked.

## Layout

- `boute/core.py`: CSV parsing, alignment, export
- `boute/gui.py`: the interface and playback
- `bootstrap.py`: sets up the venv
- `tests/test_core.py`: tests

Icon credits are in `THIRD_PARTY_NOTICES.md`.
