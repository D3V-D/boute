# Boute validation

## Current intake changes

- The upload screen now has two inputs: original video and B-SOiD results CSV, with a `labels_pose` filename hint and matching file-picker filter. Summary CSV inputs were removed because they only archived files and did not affect analysis.
- All 16 automated tests pass, including summary CSV guidance, simulated Linux drop fallback, per-group jump positions, hover lookup, and skipped-frame wording. These do not validate native OS dragging.
- User reports that Linux clicking/browsing works but native dropping still produces an invalid `build` path, even with the selection fallback. Linux native dropping remains unresolved. Windows native dropping is unverified.
- The desktop smoke test below predates the two-input screen; the updated screen has not received a new visual check.

## Passed in this Linux environment

- Seven executable processing tests (`.venv/bin/python -m unittest discover -s tests -v`).
- DLC multi-row parsing, including the actual Streamlit layout where metadata rows also carry numeric labels.
- Invalid and missing labels, missing label column, invalid video, input replacement, and duplicate-type rejection.
- Direct indexing, unequal-rate timestamp mapping, segment offsets, and incompatible duration rejection.
- Synthetic MP4 generation: each exported frame checked against its expected source brightness and chronological position; total exported frames equal the input count; manifest timestamps and identifiers checked.
- Cancellation removes incomplete output directories.
- Live-display programmatic smoke test (`.venv/bin/python tests/gui_smoke.py`): app initialization, three upload sections, file intake, Continue visibility, results widgets, preview decoding, seek, playback scheduling, group switching, new analysis, and shutdown without widget errors.
- Supplied Fly 8 Sucrose labels: 52,800 valid labels and 16 unique groups after excluding two DLC metadata rows. No pose coordinate table retained in memory.
- Supplied Fly 10 Cocaine video: all 52,800 frames decoded sequentially without error. Metadata: 52,800 frames, 440 FPS, 720 × 540, 120 seconds.

## Input identity

The original Fly 8 CSV and Fly 10 video were independent examples and were not combined. On October 18, 2026, the user supplied the matching Fly 10 Cocaine `labels_pose` CSV. It contains 52,800 valid labels at 440 Hz, matching the Fly 10 video's 52,800 frames at 440 FPS (120 seconds), and represents 28 groups. The app explicitly asks for recording-identity confirmation when filenames identify different fly numbers.

## Matching Fly 10 export — October 18, 2026

- Generated all 28 group MP4s with automatic alignment and no offset or sampling-rate override.
- Decoded all 52,800 exported frames; each group's decoded count matches its source label count.
- Verified that the manifest covers every original frame exactly once, in source order, with the correct group.
- Independently derived gaps from the aligned labels and verified all 2,521 jump markers, including their montage positions and before/after source frame numbers.
- Outputs and `validation_report.json` are in `validation_outputs/Fly_10_Cocaine_TRAINING_2026-10-18_211741_840860/`.
- These checks verify export and marker data. Native hover appearance and playback interaction still need visual review.

## Still requires manual/platform testing

- Actual OS drag-enter, drag-leave, and file drop interactions. Native tkinterdnd2 handlers are installed; programmatic smoke tests do not validate OS drag behavior.
- Human visual review of layout, real-time playback at high frame rates, speed accuracy, and prolonged use.
- Native save dialog and folder-opening actions.
- Windows and macOS setup, double-click launch, and platform codec availability.
- First-time Python installation through Windows winget and missing-Python paths on other operating systems.

Preview renders decoded frames on the GUI thread and may run below the requested rate for high-FPS or high-resolution footage. Exported files retain the source FPS regardless of preview throughput. MP4 is MPEG-4 Part 2 rather than H.264; target-system codec testing remains necessary. Bout-preserving playback is not implemented.
