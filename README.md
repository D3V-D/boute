# Boute 🪰🤸

Small desktop tool that takes a behavior video and its B-SOiD labels and makes one montage video per behavior group.

**Note:** a montage is just all the frames from one group stitched together in order. It is not a set of continuous bouts, so unrelated movements can look like one smooth clip. No audio is exported.

## Running it

- **Windows:** double-click `start_boute_windows.bat`. If Python is missing, it offers to install it through winget.
- **macOS:** open `start_boute_mac.command`. If macOS blocks it, right-click and choose Open. You'll need Python 3.10+ from python.org. Open and save dialogs intentionally show all files to avoid a native Tk file-filter crash on some Macs, including the Anaconda setup reported by a user. Boute still validates video/CSV selections and defaults exports to MP4; no manual code patch is needed.
- **Linux:** run `start_boute_linux.sh` from a terminal. You need Python 3.10+, Tk, venv, and a desktop session. On Debian/Ubuntu that's `python3 python3-tk python3-venv`, and possibly `libgl1` for OpenCV.

The first launch makes a `.venv` in the project folder and installs dependencies, so you need internet once. After that it works offline.

## Using it

1. Choose the original video and the B-SOiD results CSV. The CSV is the one with `labels_pose` in the filename (the `bout_lengths` and `transitions_mat` files aren't needed). If you renamed it, switch the picker to "All CSV files". You can also drag both files in at once (experimental). If the native drag-and-drop library cannot load on your system, file pickers remain available.
2. Hit **Continue**. If Boute can't figure out the label rate or timing on its own, it will ask for the label samples per second and where the video starts within the label timeline.
3. Pick a group, play it, scrub, change speed, or export an MP4. Enter an optional behavior name (for example, “grooming?”) and click **Save name** or press Enter. Names appear in the viewer and sidebar, are saved in `group_summary.csv`, and are included in suggested export filenames. Clear the entry and save to restore the group ID as its display name. Renaming does not regenerate videos.
4. To combine behaviors, check two or more groups in the sidebar and click **Merge selected**. The combined group keeps the lowest selected ID and plays its frames in original video order. You can merge it again with other groups. Group names carry forward; a combined group keeps the lowest selected ID’s name, which you can edit after merging. Each merge creates a new run folder with updated MP4s, frame manifest, summary, and `group_mapping.csv` tracing original IDs to merged IDs. Only the selected group videos are decoded and combined, in source-frame order; other group videos are copied without re-encoding. The original recording is not reopened. Merging re-encodes the selected videos, so repeated merges may slightly reduce image quality. Earlier run folders and the input CSV stay unchanged.

Orange ticks above the scrubber mark gaps in the original video frames. Hover for the source frame numbers and how many frames were skipped, or click to jump past the gap.

## Output

Everything goes to `~/Boute_Exports/`, in a new timestamped folder per run. Nothing gets overwritten, and cancelled or failed runs clean up after themselves.

Each run folder has:

- `group_<ID>.mp4` for each group that shows up in the video
- `frame_manifest.csv` with group, original frame index, timestamp, and output frame index (all zero-based)
- `group_summary.csv` with group IDs, researcher names, frame counts, percent of video, and montage duration
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

On Windows, use `.venv\Scripts\python.exe` instead. Drag-and-drop and the native dialogs haven't been automated, so check those by hand on each OS.

## Layout

- `boute/core.py`: CSV parsing, alignment, export
- `boute/gui.py`: the interface and playback
- `bootstrap.py`: sets up the venv
- `tests/test_core.py`: tests

Icon credits are in `THIRD_PARTY_NOTICES.md`.

## Pose overlays and movement insights

Boute automatically reads body-part coordinates from combined B-SOiD `labels_pose` CSVs. For separate DeepLabCut exports, choose **Load / replace pose CSV** in the viewer and confirm the sampling rate and video offset. Supported CSVs have DLC `bodyparts` and `coords` header rows with `x`, `y`, and `likelihood` for each landmark and a complete, zero-based numeric frame index. Sparse frame tables and duplicate indices are rejected rather than silently treated as consecutive samples. Multi-animal CSVs retain individual names. No DeepLabCut installation or GPU is needed; Boute uses existing predictions.

Use the same recording and the original image dimensions/crop. Cropped or rescaled coordinates must first be converted to the original video's coordinate system. Alignment maps pose samples using the video FPS, pose sampling rate, and offset. Measurements use video-aligned observations; a faster pose export does not increase the temporal resolution of these summaries. If pose sampling is slower than the video, repeated held samples are counted once, and speeds use the elapsed source-video time between observations. This prevents repeated coordinates from creating false stability. Any intervening group boundary still breaks motion continuity.

- **Show dots** redraws the current preview immediately, without decoding another frame or rebuilding videos. Select which landmarks to show and optionally display their names. Original videos remain unannotated.
- **Confidence** defaults to 0.80. Nonfinite, low-confidence, out-of-bounds, or invalid-confidence observations are excluded. Missing observations are not filled. Coverage and usable adjacent-sample counts are reported so failed tracking is not mistaken for stability.
- **Body reference** subtracts the selected landmark's position to separate whole-animal translation from relative motion. Optionally choose a different landmark for the **body axis** to remove in-plane rotation as well. Choose reliable landmarks on the same animal. Invalid or coincident references/axes make body-relative measurements unavailable for those frames.
- **Median smoothing** defaults to three distinct pose observations and can be disabled or changed to five. It only operates inside fully valid, continuous same-group windows. It reduces tracking jitter but cannot establish whether every measured movement is biological.
- **Summary** ranks body-relative motion and highlights the largest speed ratio against other groups for the same landmark. Ranking requires at least 50% valid observations, 25% of eligible body-motion pairs, and at least three such pairs; the reference landmark is excluded. Hints support human annotation and do not automatically identify behaviors.
- **Parts** reports median and 90th-percentile image speed, body-relative speed, coverage, movement range, and comparisons against the same landmark in other groups. Ranges are the diagonal of the coordinate-wise 5th-to-95th percentile bounds, not distance traveled. Tables scroll horizontally to expose additional columns.
- **Relations** ranks landmark pairs by median absolute change in distance. It also measures a selected three-landmark angle, with the middle landmark as the vertex. Pair and angle ranges use the 10th-to-90th percentiles. These are 2D image measurements, not 3D joint reconstructions.

Motion is calculated between successive distinct pose observations on the original video timeline only when the intervening video frames all belong to the group and both observations are valid. Boute never measures speed across montage jumps, missing samples, or group boundaries. Comparisons against other groups use their own continuous segments. Merged groups are analyzed using their updated labels, so adjacent source frames combined by a merge can form a continuous segment.

Click **Update analysis** after changing confidence, reference, axis, smoothing, or angle landmarks. **Export analysis tables** writes body-part, pair, and (when selected) angle CSVs plus JSON settings, units, source, sample counts, and the researcher-supplied group name. **Export MP4 with dots** creates a separate annotated video with the selected landmarks and confidence threshold. Plain **Export MP4** still exports the unannotated montage.

Aligned pose data is saved in `pose_data.npz` inside the results folder and carried forward when groups are merged. Earlier results stay intact. Pose processing runs in background workers so playback and navigation remain usable. The native desktop interface uses a neutral palette, bordered controls, a resizable three-panel workspace, tabs, and compact tables inspired by shadcn/ui; it does not require a browser or Node.js.
