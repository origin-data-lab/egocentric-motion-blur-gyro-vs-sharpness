# Methodology — Field Note #06

All values were re-derived from the source recordings for this release (end-to-end re-run) and match the 120 values published in the first version of Field Note #06.

## Inputs (not public)
| Recording | Files | Frames |
|---|---|---:|
| stereo_rig_left | left camera MP4 (1600×1200, 30 fps, H.264), per-frame exposure timestamps CSV, IMU CSV (501.7 Hz) | 3,600 |
| phone_knife_prep | MP4 (1920×1080, 30 fps), native IMU CSV (410.6 Hz), keystep annotation JSON, video–IMU offset JSON | 1,979 |
| phone_seasoning | same | 1,978 |

## Stereo rig
1. Centre crop (rows 300–900, cols 400–1200) of each decoded frame: Laplacian variance; FAST corners (threshold 20).
2. Mean gyro vector over each frame’s exposure window [start, start + duration], device timestamps, camera–IMU offset −1.1 ms (estimated in Field Note #03).
3. Frame-to-frame image motion from optical flow on 400×300 frames (Shi–Tomasi 400 features, pyramidal LK 21×21, 3 levels, RANSAC partial affine 2 px), scaled ×4.
4. Least-squares 3×2 map from integrated inter-frame gyro rotation to image motion: R² = 0.9743.
5. Predicted blur = |map · mean gyro during exposure| × exposure time.
6. Relative sharpness = sharpness / median sharpness of frames with predicted blur < 0.35 px within ±90 frames (≥ 6 frames required).

## Phones
1. Centre crop rows 270–810, cols 480–1440; same metrics.
2. Frame times from container PTS; IMU time shifted by the measured video–IMU offset (+0.188 s, +0.200 s; estimated); rotation rate = |mean gyro| within ±16.7 ms.
3. Relative sharpness against frames below 5 °/s within ±90 frames. Exposure time is not recorded, so results are reported against rotation rate only.
4. Non-task spans from the human keystep annotation (Field Note #05).

## Evaluation
* Spearman ρ between predicted blur and relative sharpness.
* Binned medians and IQR.
* Gyro-only flag: AUC for frames with relative sharpness < 0.6; precision/recall at fixed thresholds.

## Status of inputs taken from other notes
* 8.42 px/° at the image centre — verified in Field Note #04.
* Camera–IMU offset −1.1 ms (rig), +188/+200 ms (phones) — estimated.
* Phone ≈17–22 px/° — estimated from video–gyro fitting.

## Limitations
Camera-motion blur only (hands/tools not modelled); Laplacian/FAST on H.264 are proxies; three recordings with different devices, tasks and operators; rolling shutter and translation not modelled; phone exposure unknown; 113 phone frames lack a reference.
