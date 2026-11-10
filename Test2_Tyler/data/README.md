# Tyler's Phase 2A raw sensor recordings

These are newly executed simulated object-search trials, separate from `Test2/data`.

- `sample/demo_001/`: 69 synchronized bundles from the successful detector-driven search/inspection pilot.
- `sample/timeout_001/`: deliberately bounded no-completion trial, ending with `task_timeout` (see its summary for actual counts).
- `sample/timeout_002/`: corrected floating-point deadline check, six bundles ending at 1.0 simulated second; first timeout remains preserved.

Each trial is an exclusive output directory. `metadata.json` records full configuration, model checksum, software baseline, controller and clock conventions; `summary.json` records actual outcome. `sha256.json` fingerprints the recorded files. Do not edit a completed trial's raw data.

```text
trial/
  metadata.json, summary.json, sha256.json
  rgb/00000.png ...             lossless 640×480 Go2-camera RGB
  depth/00000.npz ...           depth_m, K, T_world_cam, timestamp
  lidar/00000.npz ...           ranges, 3D points, T_world_lidar, scan geometry/range limits, timestamp
  states.csv                   pose/quaternion/velocity, IMU gyro/acceleration, all sensor times,
                               wall elapsed time, requested/applied commands, contacts, task state/reason
  events.jsonl                 YOLO detections, target estimates, decisions, safety diagnostics/contacts
  inspection/                  RGB, annotated image, observation JSON only on successful completion
  demo.gif, trajectory.png      derived visual evidence
```

All modalities are sampled at the same unchanged physics state, nominally every 0.2 simulated seconds. Acquisition/inference are synchronous; wall inference latency is logged separately and is not simulated hardware latency. Each recorded applied command is the decision for the following control interval. The terminal row has a zero command.

Depth is **optical-z metres**, float32; infinity means no return. The shared sensor raycasts depth at 160×120 and upsamples it to the RGB shape. LiDAR is the shared MID-360-style model, with 720 planar rays and a rotating 3D cloud. Poses and IMU are simulated. Target coordinates in decisions are backprojected visible-depth estimates, not hidden scene object coordinates. The complete scene config remains in metadata for reproduction, outside the policy observation interface.

The checker in `q2_data_simulation/replay.py` verifies file hashes, frame links, sensor timestamps, intervals, valid-depth/LiDAR statistics and task replay; `--repeat-yolo` also reruns the actual model on every saved RGB image. It does not start MuJoCo.

No physical Go2 data, independently labeled detection benchmark, high-level RL rollout dataset or trained Tyler PPO checkpoint is claimed here.

## Final submission data delivery

Phase 2B and Phase 2C recordings extend this historical format. Phase 2C adds RGB-registered rendered depth and keeps collision depth in `legacy_depth/`; these are distinct sensor models. All streams remain simulation-only. PPO checkpoints now exist under `results/`, separately from these actual-MuJoCo recordings.

To reduce Git's active-file count while delivering all data, raw `rgb/`, `depth/`, `legacy_depth/` and `lidar/` directories are ignored only after complete lossless copies are packed into `recording_archives/`. No local recordings were deleted. The [archive manifest](../recording_archives/manifest.json) includes finalized and interrupted raw trials. From the repository root run `.venv/bin/python -B Test2_Tyler/package_recordings.py restore`; hashes are checked and conflicting files are never overwritten. Metadata, summaries, state/event tables, figures, GIFs and inspection images remain browsable in their original paths. The [final report](../report/ENGR6010_Second_Test_Final_Report.md) supersedes historical statements above about unavailable RL checkpoints and depth calibration.
