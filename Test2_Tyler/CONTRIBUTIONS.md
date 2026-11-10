# Contributions and attribution — Tyler's individual Second Test

Application: **Autonomous Indoor Object Search and Inspection Using the Unitree Go2**. Phase 2A is Tyler's separately identified task integration and pilot experiment, developed with coding-agent assistance. File location alone is not an authorship claim. Contributor identities and any grading-specific attribution should be confirmed against group agreement and Git history.

Shared repository baseline: `adf00d308ce3633de357eed05d5e9e0e299ec38e`. Simulator submodule: `d6e54647541fb7b048db5d9817457ee11d0e2cf9`.

| Category | Source / current file | What is reused or contributed |
|---|---|---|
| Third-party, unchanged | syswonder `robot-unitree-go2-mujoco`, Unitree Go2 model, MuJoCo | Physical model, torque-driven scripted trot, camera/depth/LiDAR/IMU and scene construction. Not Tyler's locomotion training |
| Third-party, unchanged weights | [THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10), [ONNX Community export](https://huggingface.co/onnx-community/yolov10n), ONNX Runtime | Pretrained chair recognition; model distributor declares AGPL-3.0. Download URL/hash preserved. No Tyler YOLO training claimed |
| Shared, imported | `Phase_2_Investigator/sim_env/runtime.py`, `interfaces.py`, `environments.py` | Existing group session and sensor contracts. `adapters/shared.py` redirects generated overlay/cache paths and applies Tyler scene configuration in temporary files only |
| Shared, imported | `Phase_2_Investigator/navigation/local.py`, `velocity.py` | Existing local collision/sweep safety layer; used with measured depth and LiDAR, not hidden target coordinates |
| Shared, imported | `go2_stream/perception/detector.py` | Existing `Detection` dataclass and source convention. The Ultralytics backend itself is not used; Tyler's ONNX adapter implements preprocessing/decoding |
| Adapted algorithm pattern | `go2_stream/behavior/control.py` → `q1_vision/mission.py` | Turn-then-forward approach and bounded commands; new task state machine, consecutive confirmation, observed-world identity gate and inspection completion. No original sit/act mission or hardware commands copied |
| Adapted recording concept | `Phase_2_Investigator/sim_env/recorder.py` → `q2_data_simulation/recording.py` | Per-frame RGB/depth/LiDAR and timestamp/transform provenance. New exclusive-run format, raw depth NPZ, detector/task/command records, outcome and checksums |
| New integration | `config/`, `probe.py`, `run.py`, `q1_vision/observations.py`, `adapters/yolo.py` | Tyler's chair placement, detector gate, actual depth target estimate, detector-conditioned search/inspection and bounded runner |
| New verification/data | `tests/`, `q2_data_simulation/replay.py`, `data/sample/`, `report/`, `results/probe_*` | Tyler's task/data tests, retained failed probe, successful probe, new demonstration, replay checks and reports |
| New future interface, not trained RL | `q3_rl/observations.py` and README | Robot-observable 34-value boundary and proposed independent PPO experiment. No existing policy renamed or claimed |
| Prior work, not imported or modified | Connor/group `Test2/` | Forensic zone classifiers, data, PPO v1–v4, figures and demos remain Connor/group prior results. They are not Tyler's experimental evidence |

All 1,468 files found under `Test2/` were fingerprinted before implementation (`report/shared_Test2_before.sha256.json`). The final preservation check records whether the set and hashes stayed identical. Shared generated overlays are redirected to `Test2_Tyler/.cache/`; no shared source or simulator asset is edited. Existing group checkpoints and experiment outputs remain in place.

Future adaptations of SB3 training/checkpoint/resume scaffolding must explicitly cite their source and list task-specific changes. Tyler must train and evaluate his own object-search policy and generate his own results rather than copying forensic reward definitions, policies or learning curves.

## Phase 2B additions and validation (2026-11-10)

The following individual implementation was developed with coding-agent assistance. This is attribution of implementation work and provenance, not a claim that directory placement establishes personal authorship.

| Category | Sources / Tyler files | Contribution and limits |
|---|---|---|
| Adapted shared algorithms | `Phase_2_Investigator/navigation/{occupancy,frontier,planner}.py` → `q1_vision/sensor_map.py` | Sensor log-odds mapping, frontier grouping, inflation and Dijkstra routing adapted to NumPy/OpenCV/heapq. Shared algorithms credited; no scene geometry supplied to the controller |
| New task integration | `q1_vision/exploration.py`, `search_observations.py`, `search.py` | Sensor-map exploration, committed reachable goals, target-loss stop/recovery, upper-box depth estimate, bounded inspection evidence window and exclusive trial runner. Range/occlusion and exploration success remain limited |
| Adapted shared simulator/safety | `adapters/search_sim.py` + preserved shared adapter and `navigation.local` | Tyler scene variants and post-decision mesh-ray scoring; existing physical trot, sensors and safety layer reused. No new locomotion policy or hardware deployment claimed |
| New individual recordings | `data/phase2b/`, `report/phase2b/`, `results/phase2b/validation_figures_20261110/` | Actual YOLO/MuJoCo trials, successful and failed outcomes, partial interrupted trial retained, synchronized raw sensors, checksums, plots and detector reinference verification. Prior results not relabeled |
| Shared workflow reference, distinct task | `Test2/q3_rl/crime_scene_env.py` → `q3_rl/object_search_env.py` | Gymnasium episode organization referenced; new room geometry, observable vector, proxy sensing, rewards, dynamics, randomization and termination implemented. No imported forensic environment or privileged target direction |
| Adapted training scaffolding | `Test2/q3_rl/train_ppo.py` → `q3_rl/train_smoke.py` | SB3 Monitor/EvalCallback/checkpoint workflow reused with independent 2,048-step initialization/training, bounds, finite-gradient/weight-change/reload checks. SB3 provides PPO; Tyler did not author the PPO optimizer |
| New bounded evaluation/tests | `q3_rl/evaluate_proxy.py`, `tests/test_phase2b.py`, `q2_data_simulation/verify_search.py` | Seeded proxy random/reactive/PPO evaluation, environment contracts, observability/reward/termination and navigation regression tests; actual-YOLO data checks separate from proxy outcomes |
| New weights, limited evidence | `results/phase2b/ppo_smoke_20261110/` | Fresh seed-601022 Tyler smoke model and four intermediate checkpoints. No Connor model loaded. Pipeline functions; zero held-out successes means useful search learning is not demonstrated |

The Phase 2A source, raw evidence and original report remain preserved. `report/phase2b/preservation_before.json` fingerprints 1,770 baseline files, including all 1,468 Connor/group `Test2/` files. The recovery archive additionally preserves the earlier individual Phase 2B code, failed trials and interrupted recording. See the Phase 2B report for backup location, validation details and exact experimental limitations.

## Final submission integration (November 10, 2026)

See [Phase 2C attribution](phase2c/CONTRIBUTIONS.md) for versioned registered depth, task/control extensions, independently recorded calibration/exploration experiments, and the new 8,192-step PPO run. Final report/README, split lossless data delivery, paired evaluation and submission tooling were prepared with coding-agent assistance. The actor/value networks, rollout buffer, GAE and PPO optimization are supplied by Stable-Baselines3. No successful learned policy, physical recordings or group experiment results are claimed as new individual achievements.
