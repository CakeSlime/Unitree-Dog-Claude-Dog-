# Phase 2A Implementation Report

**Autonomous Indoor Object Search and Inspection Using the Unitree Go2**  
Student: Tyler. Course: ENGR 6010, Second Test. Date: 9 November 2026.

## Outcome

A separate working `Test2_Tyler/` application now performs **actual YOLO chair detection on rendered Go2 camera images, search/confirmation, bounded approach, stopped inspection capture, and synchronized sensor recording**. One bounded simulation trial completed successfully. Deliberate timeout trials failed and stopped; a floating-point deadline issue was corrected and verified. All sixteen current unit/contract tests passed. Actual-model replay reproduced detections and task decisions across every frame of the successful recording.

This completes the initial Phase 2A working-demonstration pipeline. It does not establish general indoor autonomy, detector accuracy over held-out data, physical Go2 operation, or completion of Q3 training. Full professor-rubric compliance remains conditional on the original brief and any physical-dog requirement.

No physical robot was commanded. No PPO training, group checkpoint loading, Git commit, push or remote branch modification occurred. All 1,468 files under Connor/group `Test2/` retained the same paths and SHA-256 values after implementation.

## 1. Implementation and separation

| Component | Implementation |
|---|---|
| Workspace | Separate configs, adapters, Q1/Q2/Q3 modules, tests, data/sample, results and report directories |
| Simulator | Shared `SimSession`, torque-driven Go2 with the shared scripted gait; no forensic PPO policy |
| Scene adapter | Shared house-185 mesh assets; Tyler's chair placement/lighting in a generated temporary scene; generated overlay under Tyler cache |
| Vision adapter | CPU ONNX YOLOv10n, checksum pinned; shared `Detection` interface; resize/pad/box decoding and fragment rejection |
| Task | New explicit search/confirm/approach/inspect/complete/failed state machine, adapted approach pattern from group control utilities |
| Safety | Shared sensor-based `SafetyLayer`, plus contact/tilt/stale-data/lost-target/blockage/time guards |
| Recorder | Exclusive per-trial RGB, raw depth, LiDAR, pose/IMU, clocks, detector/target estimates, task state, requested/applied commands and outcomes |
| Verification | New tests and read-only recorded-data/task replay, optionally repeating YOLO inference |
| Future RL | Robot-observable 34-value prototype interface and task design only; no Gymnasium environment or trained policy claimed yet |

Detailed sources and shared/adapted/new distinctions are in [CONTRIBUTIONS.md](../CONTRIBUTIONS.md). Directory ownership is not used as authorship evidence. The individually developed integration and pilot are disclosed as coding-agent-assisted work.

The shared source baseline was `adf00d308ce3633de357eed05d5e9e0e299ec38e`; the simulator submodule was `d6e54647541fb7b048db5d9817457ee11d0e2cf9`. Individual code remains uncommitted. `implementation_manifest.json` fingerprints the delivered code/config/environment snapshot; existing trial metadata's Git revision identifies the shared baseline, not an individual implementation commit.

## 2. Environment and dependency changes

The existing CPython **3.12.14** `.venv` was preserved. Repository and `.venv` write/fsync probes passed. The filesystem stayed writable; sampled available memory during the demonstration was approximately 3.4 GiB, and swap remained unused. These were sampled health checks, not continuous peak telemetry.

The only newly requested package was `onnxruntime==1.23.2`; its installation added seven distributions including the runtime. The environment grew from seventeen to twenty-four distributions. `uv pip check` passed after installation. Exact installed versions are recorded in [environment.freeze.txt](environment.freeze.txt); [requirements.txt](../requirements.txt) lists direct requirements.

CPU ONNX inference avoided installing Torch/Torchvision/Ultralytics, SciPy, a hardware SDK or DDS. The external [YOLOv10n ONNX weights](https://huggingface.co/onnx-community/yolov10n) were explicitly downloaded into ignored `Test2_Tyler/.cache/models/`, not any group model folder. Distributor/source and AGPL-3.0 identification are recorded in the README/contribution record. Model SHA-256:

```text
a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b
```

The initial PyPI/model-download attempts failed on sandbox DNS. Network-approved retries succeeded with the same writable project-local cache and existing environment; no filesystem restriction was bypassed to write into the home cache. ONNX Runtime emitted a missing GPU-device-discovery-file warning under `/sys/class/drm/`; the explicitly selected CPU provider worked. No ext4/read-only remount or storage-I/O failure was encountered in this phase.

## 3. Detection gate before autonomous motion

| Probe | Configuration and outcome |
|---|---|
| `results/probe_001` | Original chair placement, 320×240 camera, first stretched-image preprocessing: **failed**, no qualifying ≥0.30 chair box; exit 2. Original output retained |
| `results/probe_002` | Published aspect-preserving resize/pad preprocessing, 640×480 configured Go2 camera, Tyler chair placement, aligned stationary heading: **passed**, chair confidence **0.444446**; exit 0 |
| `results/probe_003_current_config` | Final documented config at mission-start XY, aligned heading: **passed**, chair confidence **0.683256**; exit 0. Confirms the README's fresh-probe command works with the delivered config |

The second probe used the same physical chair mesh, not a photo billboard or ground-truth detector. Its RGB image and drawn box were visually checked. Its actual standing pose was approximately `[4.880,1.779,1.570]`, with zero scene contacts. Model inference took approximately 0.112 s for that frame.

The final-config stationary check used `[4.88,1.1,1.57]` and had zero scene contacts, with approximately 0.139 s inference. The successful autonomous run remains the separately recorded `demo_001` trial; an additional stationary probe is not another autonomous success trial.

Several pilot conditions changed together: preprocessing, sensor resolution and object placement/view. This is not a controlled ablation or evidence of out-of-distribution detector robustness. The [published processor settings](https://huggingface.co/onnx-community/yolov10n/blob/main/preprocessor_config.json) justify correcting the image transform; the resulting recognition must still be measured on held-out views later.

`run.py` refuses navigation without a passed probe for the same model, target confidence, environment, camera setting and chair placement. The aligned probe and initial mission headings intentionally differ. The mission does not reuse the probe's box/pose as a goal; it processes fresh frames while scanning.

## 4. Actual successful object-search experiment

### Protocol

Config: `config/chair.yaml`, nominal seed 601021 (fixed deterministic scene/commands; no randomized scene sweep in this stage). Environment: furnished house 185. Chair placement: `[4.88,3.35,0]`, used only by the scene builder. Requested Go2 spawn: `[4.88,1.1,0]`, initially facing away from the chair. Camera: shared front mount and FOV with configured 640×480 RGB; depth raycast at 160×120 and upsampled.

The mission takes only sensor-derived target observations: class/confidence, box, registered-depth surface range/bearing and backprojected visible world-surface estimate. It does not receive the scene chair coordinates. Three consecutive detections confirm a target; observed-world gating maintains continuity. Motion uses bounded forward/yaw control and the shared LiDAR/depth safety filter. Stale/repeated/future frames, contacts, tilt, lost target, sustained obstruction and time budgets terminate with zero command.

Inspection requires surface range 0.95±0.10 m, bearing within 0.10 rad, horizontal speed below 0.06 m/s and three qualifying hold frames. The target range is an estimate to a visible surface, not a calibrated ground-truth object-centre measurement.

Command executed after passing unit tests:

```bash
source .env.phase1
timeout 120s .venv/bin/python -B -u -m Test2_Tyler.run \
  --gate Test2_Tyler/results/probe_002/gate.json \
  --out Test2_Tyler/data/sample/demo_001 --wall-limit 90
```

### Results from `demo_001/summary.json` and raw records

| Measure | Actual result |
|---|---:|
| Outcome / process exit | Complete, `inspection_ready` / 0 |
| Simulated task duration | **13.600 s** |
| Measured task-loop wall time | **24.012 s**, excluding model/scene initialization |
| Synchronized recording bundles | **69**, 0.2 s intervals |
| Initial sample base XY | `[4.878864,1.100000]` m |
| Final base XY/yaw | `[4.870058,1.644820,1.572406]` |
| Sampled path length | 1.009844 m; includes gait/scan oscillations, not just net displacement |
| Net base displacement | Approximately 0.545 m |
| Inspection estimated surface range | **1.049378 m**, inside the configured upper bound |
| Final chair confidence | **0.516712** |
| Inspection horizontal speed | **0.000435 m/s** |
| Maximum applied forward / yaw commands | **0.18 m/s / 0.35 rad/s** |
| Robot-scene contact steps | **0** |
| Safety-limited steps | **0**; no positive claim about guard effectiveness follows from this alone |
| Minimum upright cosine | **0.995897** |
| Mean model inference time | **0.091286 s/frame** |
| Final applied command | **[0,0,0]** |

State transitions were search at 0.0 s, confirmation at 3.6 s, approach at 4.0 s, inspection at 11.6 s, temporary return to approach at 11.8 s, inspection at 13.2 s, and completion at 13.6 s. The temporary return demonstrates rechecking the inspection conditions rather than capturing as soon as range first crossed the band.

The trial has 49 frames containing accepted chair detections; confidence among those detections ranged from approximately 0.3025 to 0.8544. This is an observation count, not labeled precision/recall. The final saved RGB and annotation show the actual chair.

Evidence: [summary](../data/sample/demo_001/summary.json), [GIF](../data/sample/demo_001/demo.gif), [trajectory](../data/sample/demo_001/trajectory.png), [inspection](../data/sample/demo_001/inspection/annotated.png), [inspection observation](../data/sample/demo_001/inspection/observation.json), and [visual evidence index](../results/Phase2A_Evidence.md).

## 5. Sensor recording and reproducibility checks

The successful sample is approximately 28 MiB and includes 69 RGB PNGs, 69 float-depth NPZs, 69 LiDAR NPZs, 69 state rows, 69 detector/task/safety events, inspection evidence, derived GIF/path plot, configuration, outcome and checksums. Raw sensor/clock fields and units are documented in [data/README.md](../data/README.md). All state rows share synchronized camera/LiDAR/IMU timestamps at each physics snapshot. Wall inference latency is recorded separately; this synchronous simulation does not model real sensor latency.

Actual read-only replay command:

```bash
timeout 60s .venv/bin/python -B -m Test2_Tyler.q2_data_simulation.replay \
  --trial Test2_Tyler/data/sample/demo_001 --repeat-yolo \
  --out Test2_Tyler/report/demo_001_replay.json
```

The [replay report](demo_001_replay.json) verifies **216 file hashes**, all 69 image/depth/LiDAR links, aligned timestamps, 0.2-second intervals, 100% finite depth in these recorded frames and 720 valid planar LiDAR returns per frame. Actual CPU YOLO reinference matched the saved boxes/confidences, and recomputed depth observations reproduced every task state and requested command. No MuJoCo simulation was started by replay.

This checks stored-data and perception/task reproducibility. It is not a second independently simulated success trial or a cross-machine determinism guarantee. Outcome rates, localization errors, detector precision/recall and robustness remain unmeasured.

## 6. Failure experiment and automated tests

### Deliberate timeout

`config/chair_timeout.yaml` changes only the task deadline to 1.0 s. The first bounded invocation used the passed gate and new `data/sample/timeout_001` output. It returned expected exit **2**, `failed/task_timeout`, with **seven bundles**, **1.2 simulated seconds**, **4.39 s task-loop wall time**, zero scene contacts and final command `[0,0,0]`. The deadline is checked at 0.2-second control ticks; floating-point timing caused one extra tick beyond the nominal one-second boundary.

The [failure replay](timeout_001_replay.json) verified 27 hashes, synchronized modalities and the reproduced timeout state/commands. It did not repeat YOLO inference. No inspection image was produced or falsely claimed as successful.

The comparison was then corrected with a bounded 1e−8-second tolerance in new configs, with a regression test at the actual problematic floating-point boundary. Legacy configs default to zero tolerance to preserve historical replay. The new `timeout_002` trial stopped at **1.000 simulated second**, **six bundles**, **4.15 s loop wall time**, expected exit **2**, no contacts and zero final command. [Replay](timeout_002_replay.json) verified **24 hashes**, synchronization and the corrected failure decision. The earlier trial was not altered.

### Unit/contract tests

Initial batch: **13 passed in 0.22 s**. After gate, low-confidence and replay/tamper checks: **15 passed in 0.27 s**. With the deadline regression: **16 passed in 0.25 s**, zero failures/skips. All batch outputs remain saved; these are overlapping suite revisions, not 44 distinct passing tests.

```bash
timeout 60s .venv/bin/python -B -m pytest -q Test2_Tyler/tests \
  --junitxml=Test2_Tyler/report/tests-reproduction.xml
```

Tests cover model-output coordinate scaling/class/fragment filtering, depth geometry and invalid returns, rejection of ground-truth detector input, consecutive confirmation, identity continuity/no blind approach, contacts/tilt/invalid state, stale/repeated timestamps, too-close/timeout/invalid limits, stopped inspection hold, sustained blockage, low-confidence rejection, passed-scene gate enforcement, sensor alignment/roundtrip, checksum tampering and output-directory preservation. They also verify that the future policy vector excludes target world coordinates.

Logs and machine-readable outputs: [initial tests](tests-initial.log), [intermediate tests](tests-final.log), [latest tests](tests-deadline.log), [latest JUnit](tests-deadline.xml), [success run](demo_001.log), [first timeout](timeout_001.log), [corrected timeout](timeout_002.log). No unchanged group suite was unnecessarily rerun, and no new broad historical pass count is claimed.

## 7. Preservation and constraints

`shared_Test2_before.sha256.json` captured all 1,468 existing Test2 files before implementation. [shared_Test2_preservation.json](shared_Test2_preservation.json) confirms the exact same file set and hashes afterward. No `Test2` file was imported by the application, edited, overwritten or deleted. Shared simulator source/assets were not modified; only generated temporary scenes were changed. Caches, weights and virtual environments remain ignored, and no credentials were placed in the individual workspace.

Experiments used separate exclusive paths. Failed probes and trials were preserved. A completed trial's raw files were not edited after checksumming. No physical commands, RL training, pushes or group branch modifications occurred. No multiple resource-intensive workloads ran concurrently.

## 8. Question coverage and next work

| Question | Phase 2A contribution | Remaining work |
|---|---|---|
| Q1 | Tyler's detector-driven simulated task, confirmation/continuity, measured approach and inspection evidence | Held-out labeled detector/task evaluation; broader placements/occlusion/distractors; physical demonstration if required |
| Q2 | New reproducible synchronized RGB-D/LiDAR/pose/IMU/command/task dataset and actual physical MuJoCo experiment | Multiple independent trials, provenance-rich splits and error analysis; physical data if required |
| Q3 | Explicit robot-observable prototype boundary and reusable architecture | New Gymnasium task, task rewards/episode accounting, Tyler PPO training/checkpoints/curves and fair classical-baseline evaluation |

Priority next steps:

1. Keep this working demo as an immutable pilot and confirm the full course rubric/physical requirements.
2. Expand measured failure cases, target placements and initial headings; add bounded scene-level regression checks in separate output directories. The observed tick-boundary deadline issue is already fixed and verified.
3. Build Tyler's held-out object-search data and a classical exploration baseline. Current search only scans in place until a visible target can be approached; it cannot search hidden rooms or route around arbitrary obstacles.
4. Quantify detection precision/recall and range/standoff error, including association failures and targets with sparse/invalid depth. Camera-to-surface depth estimates are not yet calibrated range-error results.
5. Implement the independent PPO environment and adapt attributed SB3 training scaffolding. Do not expose hidden target coordinates or copy Connor's policy/results. Use a modest approved compute budget only after the search/data pipeline's expanded validation.
6. Package Tyler's README, report, contribution table, code/configs, raw data manifest, future policy/evaluation evidence and individually identifiable GitHub submission. Nothing has been pushed in this phase.

The current delivered claim is deliberately limited: **one working, recorded, detector-conditioned simulated chair-search and inspection pilot, plus a correctly recorded failure path and passing tests**.
