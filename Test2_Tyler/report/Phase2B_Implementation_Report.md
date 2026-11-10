# Phase 2B implementation and validation report

**Project:** Autonomous Indoor Object Search and Inspection Using the Unitree Go2  
**Individual submission:** Tyler; shared infrastructure attributed in `../CONTRIBUTIONS.md`  
**Validation date:** 2026-11-10  
**Branch / tracked baseline:** `engr6010-second-test` / `adf00d308ce3633de357eed05d5e9e0e299ec38e`

## Outcome

The Python dependency blocker is resolved. The independent Gymnasium environment passes both Gymnasium and Stable-Baselines3 checkers. A fresh **2,048-step PPO smoke run completed**, changed weights with finite gradients, saved four checkpoints and a final model, and successfully reloaded for evaluation. The final test suite reports **33 passed, 0 failed, 0 skipped**. This establishes a functioning PPO pipeline, **not useful learned search behavior**.

Four new bounded actual-YOLO/physical-MuJoCo trials produced two controller-defined inspection completions and two failures. All four recordings passed file integrity, synchronized sensor timestamp, numeric aggregation, final-stop and **every-frame YOLO reinference** checks. No recorded robot-scene contacts occurred. Successful inspection after **translational exploration** remains unverified: the two completed trials turned to discover/track the chair and then approached, with zero forward-search ticks. The harder trials executed forward exploration but failed to inspect.

Phase 2A evidence and source remain intact. Connor's `Test2/` files were not modified. No physical robot commands, extended PPO training, Git commits or pushes were performed.

## Preservation, environment and errors

Before modifications, a recovery archive was created at `.local-assets/recovery/phase2b_20261110_0830.tar.gz`, with companion `.tar.manifest.json`. All **2,396 archived files** were hash-verified against their archived contents. This includes earlier Tyler source, reports, failed trials and the interrupted recording, plus existing docs and root `.gitignore`. It excludes disposable caches. It is a local backup on the same WSL filesystem, not an independent disk/remote backup or Git commit.

Final preservation verification found:

- **1,770/1,770 baseline file hashes unchanged**, including all **1,468 Connor/group `Test2/` files** and preserved Phase 2A source/evidence.
- **2,315 prior data/results/docs/`.gitignore` files** from the recovery manifest unchanged, including unsuccessful Phase 2B results and the interrupted `shifted_001` data.
- Root `.gitignore` preserved. Git confirms `.venv`, `.cache`, `.tmp`, `.local-assets`, `.env.phase1`, individual caches and intermediate checkpoint directories are ignored.
- Individual files remain untracked: `Test2_Tyler/`, root `.gitignore`, and the existing Second Test docs/results. Tracked Git diff is empty. No existing work was discarded or staged.

Write-and-fsync probes passed for the repository, `.venv`, local uv cache and temporary directory. The WSL filesystem was mounted read/write, with about 942 GiB free. During MuJoCo checks, approximately 3 GiB RAM remained available; final available RAM was approximately 5.36 GiB. Swap remained effectively unused (less than 1 MiB occupied in observed snapshots). These are snapshots, not a peak-memory profile. No EROFS, storage EIO, OOM or WSL startup failure occurred during this validation.

The first new dependency attempt failed immediately with **sandbox DNS resolution failure** for `pypi.org`. A network-authorized retry succeeded using the same writable project-local cache and virtual environment. No elevated filesystem access, remount, environment recreation or sandbox disabling was used to bypass a storage failure. This run did not reproduce the older home-cache temporary-file failure; it cannot retrospectively prove that older failure's exact cause. ONNX Runtime emitted a nonfatal missing `/sys/class/drm/card0/device/vendor` discovery warning; inference explicitly used CPU and succeeded.

### Installed environment

| Component | Verified version |
|---|---|
| Existing Python | 3.12.14 |
| Existing CPU PyTorch | 2.15.0+cpu |
| Existing Gymnasium | 1.4.0 |
| Existing NumPy | 2.5.4 |
| Newly installed Stable-Baselines3 | 2.9.2 |
| Newly installed Matplotlib | 3.10.7 |

Only nine packages were added: SB3, Matplotlib, contourpy 1.4.0, cycler 0.12.1, fonttools 4.68.0, kiwisolver 1.5.1, pyparsing 3.3.3, python-dateutil 4.0.0 and tzdata 2026.6. Existing Torch/Gymnasium/NumPy were retained. `uv pip check` verified **42 compatible installed distributions**. SciPy, scikit-image, hardware SDKs and DDS were not added. Dependency imports passed. Full versions and installation output are indexed below.

The bounded installation command, after sourcing `.env.phase1`, was:

```bash
UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=0 timeout 90s uv pip install \
  --python .venv/bin/python --only-binary :all: \
  numpy==2.5.4 gymnasium==1.4.0 stable-baselines3==2.9.2 matplotlib==3.10.7
```

`UV_CACHE_DIR` remains `$PWD/.cache/uv`; temporary files and Matplotlib config use project-local paths. Optional direct RL dependencies are documented in `../requirements-rl.txt`; the complete environment snapshot is retained separately.

## Implementation and automated validation

The recovered Phase 2B code already contained the new sensor-map explorer, upper-box depth adapter, actual-YOLO runner, 2-D environment, training/evaluation scripts and foundation tests. This validation installed the missing libraries and exercised those implementations rather than relabeling historical tests as current success.

Changes made during this continuation:

1. **Committed exploration goals:** reachable vantages are retained while routes are refreshed. Previously `_plan()` could replace a goal every two seconds before arrival, causing route/heading churn. Unreachable, blocked or reached goals can still be replaced. Tests cover route stability and obstacle/unknown-space avoidance.
2. **Bounded inspection evidence:** the Phase 2B wrapper accepts three valid inspection observations within at most one second, allowing brief detector gaps. Current-frame validity, identity continuity, range, bearing, low speed and the original guard checks remain necessary. A missing detection stops immediately and cannot complete inspection. Out-of-band/moving observations clear evidence; old votes expire. Phase 2A's original consecutive-frame implementation is unchanged. New successful trials actually met the original consecutive-frame criterion (`inspection_ready`), so they do **not** demonstrate a benefit from the temporal window.
3. **Stronger smoke success criteria:** require exactly the requested steps and nonzero PPO update count in addition to finite gradients/losses, changed weights and checkpoint/reload checks. Close the inference test environment afterward.
4. **Recording verifier:** optional every-frame actual YOLO reinference, matching model hash, box counts, confidences and boxes; this is separate from proxy evaluation. The scoring adapter's future reference-description string now correctly identifies the depth sample pixel. Existing recordings retain their original metadata and source snapshots.
5. Added randomized complete-rollout and inspection/goal regression tests, optional RL dependency pins, updated README/Q3 documentation, attribution, diagnostic summaries and figures.

The latest run executed `pytest -q Test2_Tyler/tests`: **33 passed in 3.48 seconds**. This includes 16 preserved Phase 2A tests and 17 Phase 2B tests. Earlier current-session stages were 29/29, then 32/32; these are successive suite versions, not additional independent validation trials. Historical failed planner tests and dependency-skipped checker logs remain preserved and are superseded by the new passing runs.

Checks cover Gymnasium/SB3 contracts; seeded reproducibility and variation across seeds; finite, bounded observations and actions through complete episodes; reward-term summation; confirmation/inspection rewards; collision termination; time-limit truncation; reset-after-terminal behavior; invalid actions; no unseen target signal; occluded proxy targets; exploration routing; immediate missing-target stop/recovery; and inspection vote expiry/current-frame requirements.

## Q3 environment and PPO smoke evidence

`q3_rl/object_search_env.py` is Tyler's distinct object-search task, with shared episode organization used only as an attributed reference. It does not load Connor's environment or weights. Its **38-value float32 observation** comprises 24 ray ranges, five current target features, three measured velocities, two previous commands and four local visited-cell indicators. Target features are zero when unseen. Tests change unseen target ground truth without changing the observation. No target location or oracle bearing is included in observations or `info`.

Two normalized continuous actions request forward ≤0.18 m/s and yaw ≤0.35 rad/s; no reverse/strafe. Episodes randomize free start/target locations, heading and divider position. The explicitly labeled **proxy detector** uses field of view, range, occlusion and dropout, not rendered RGB/YOLO. Rewards cover newly visited base cells, first discovery, first confirmation, progress only while consecutively observed, inspection, collision, time and command smoothness. Collision/success terminate; 180 steps truncate. Human diagnostic renderings show geometry but are not fed to PPO.

Fresh PPO configuration: seed 601022, CPU one thread, MLP 64×64, rollout 128, batch 64, two epochs, learning rate 0.0003, gamma 0.99, GAE 0.95, entropy coefficient 0.01. The runner had a 90-second internal budget and a 120-second outer timeout. Separate seeded validation ran every 512 steps (three episodes), alongside checkpoints. Actual smoke results:

| Measurement | Actual result |
|---|---|
| Requested / actual training steps | 2,048 / 2,048 |
| SB3 `_n_updates` | 32 training epochs across 16 rollout updates; not 32 optimizer minibatches |
| Weight-vector L2 change | 1.0966352224 |
| Final retained gradient norm | 0.4999992982; finite and nonzero |
| Final weights / logged losses | Finite |
| Checkpoints / reload | Four interval checkpoints; final checkpoint reloaded and evaluated |
| Completed training episodes / successes | 14 / 0; one episode confirmed a target |
| Mean training episode return | −7.60234; range −11.42703 to −2.12864 |
| Logged total loss | 15 finite samples; min −0.023908, max 0.964856, last logged −0.015237 |
| Logged value loss | Last logged 0.023944; all samples finite |
| Timed learn/save/reload/log section | 2.8133 s; excludes initial imports/model construction |

SB3's rollout CSV logs training statistics from the preceding update; the final logged loss is not asserted to be the last optimizer step. Full loss components, validation results, episode monitor data and plots are retained. Weight/gradient checks establish optimization occurred; they are not performance claims.

Final model: `../results/phase2b/ppo_smoke_20261110/final_model.zip`  
SHA-256: `68708256c32d4a1722f8ea9f480004fa27101db4493ab1d243612d82bd16496d`

### Independent proxy evaluation

The reloaded final checkpoint, reactive classical controller and random actions were each evaluated on eight seeds starting at 801022, with identical initial poses checked. These are **24 proxy episodes**, separate from actual-YOLO/MuJoCo trials. The reactive proxy baseline is simpler than the physical sensor-map explorer.

| Method | Successes / episodes | Collision episodes | Mean return | Mean visited cells |
|---|---:|---:|---:|---:|
| Random | 0/8 | 4 | −6.71878 | 8.000 |
| Reactive classical | 0/8 | 4 | −1.45046 | 17.250 |
| PPO smoke | 0/8 | 1 | −2.95412 | 5.875 |

**Useful policy learning is not demonstrated.** Fewer PPO collisions accompany lower coverage than the reactive baseline and could reflect inactivity. No superiority, convergence, transfer, or statistically reliable safety improvement is claimed. The pipeline is ready for bounded further experiments, but not for claiming Q3 performance is complete. Extended training has not been launched or authorized by this validation.

## Q1/Q2 actual-YOLO experiments

All four new trials used `search.py`, shared physical Go2 trot dynamics, CPU YOLOv10n, measured RGB/depth/LiDAR, pose/IMU, shared local safety and Tyler's sensor-map controller. Chair coordinates configure the scene only; the deployed controller receives sensor-derived targets and map observations, never scene target coordinates. Hidden visual geometry is queried only after a completion decision for scoring. Detection confidence threshold stayed 0.30.

Each new trial has an exclusive output directory under `../data/phase2b/`, raw RGB/depth/LiDAR, pose/IMU/timestamps/requested and applied commands, detections/task events, summary, observed map, source snapshots, GIF and SHA manifest. Successful trials also contain inspection RGB/annotation/observation. These are manually specified deterministic scene variants, not a random held-out benchmark.

| New trial | Outcome | Sim s / wall s | Frames | Path m | Forward-search ticks | Mean accepted confidence | Contacts |
|---|---|---:|---:|---:|---:|---:|---:|
| `visible_20261110` | Inspection complete | 9.8 / 20.10 | 50 | 0.838 | 0 | 0.600 | 0 |
| `near_20261110` | Inspection complete; initially unseen | 13.6 / 26.12 | 69 | 1.055 | 0 | 0.577 | 0 |
| `occluded_20261110` | Safe failure: `target_too_close` | 31.0 / 56.13 | 156 | 2.515 | 69 | 0.510 | 0 |
| `shifted_20261110` | Failure: `task_timeout` | 45.0 / 84.87 | 226 | 4.204 | 98 | 0.401 | 0 |

Every trial ended with zero applied command. The visible trial detected at time zero and confirmed at 0.4 s; the initially unseen near trial detected at 5.0 s and confirmed at 6.0 s. Path includes trot/turn-induced displacement, so path length alone does not prove commanded exploration. Both hard trials had one target-loss recovery; their safety filter intervened on 45 and 38 frames respectively. These outcomes are selected development trials, **not a general 50% success-rate estimate**.

### Failure diagnosis and remaining perception problems

- Preserved `occluded_001`: five accepted YOLO frames among 19; failed `target_lost` at 3.6 s after ten missing-target-stop frames. On first detection, central-box depth estimated 2.081 m while the upper-box heuristic estimated 2.904 m. The image contains foreground furniture between robot and chair. These differing depth associations explain why correct class recognition alone does not ensure reliable approach geometry.
- Preserved `occluded_002`: 7.215 m travel and 179 forward-search ticks, but only 32/301 frames had an accepted chair detection. Three loss recoveries; 60-second timeout. Exploration existed; robust discovery-to-inspection did not.
- Preserved `near_001`: only 18/176 frames had accepted chair detections. Three separate inspection-hold frames were broken by missing detections; 35-second timeout. There were 32 missing-target-stop frames. This supports a detection-continuity/inspection limitation, rather than claiming navigation never moved.
- Preserved interrupted `shifted_001`: **162 RGB frames and event rows**, no accepted YOLO detections in those rows, metadata still `running`, no summary/checksum. It is an interrupted partial recording, not a completed success/failure or verified complete dataset. It was neither resumed into nor rewritten.
- New occluded trial: movement and rediscovery occurred, but the final detected chair's upper-box depth estimate was **0.17196 m**, triggering the existing too-close fail-stop. The chair remains visible in a furniture-occluded view; this range is not proof that its visible surface was physically 0.17 m away. RGB/depth association, foreground returns and simplified collision geometry need investigation. No blind retreat or hidden-coordinate substitute was added to force success.
- New shifted trial: actual YOLO drove confirmation/approach, followed by gaps/recovery and eventual timeout. It records genuine exploration without successful inspection. The reachable-goal fix does not solve scene coverage or intermittent target visibility by itself.

Depth is 160×120 raycast data upsampled to 640×480 RGB. The shared sensor ray mask includes collision groups and excludes visual group 1; the independent score casts against visual meshes. Consequently, a box ROI, a single sample-pixel ray and the rendered visible surface are different measurements. Foreground/depth-geometry disagreement remains a material limitation, not a calibrated accuracy result.

For the two successful controller decisions:

| Trial | Estimated inspection range | Visual-mesh horizontal reference | Estimate minus reference | Reference minus 0.95 m |
|---|---:|---:|---:|---:|
| Visible | 1.03911 m | 1.12746 m | −0.08835 m | +0.17746 m |
| Initially unseen near | 1.04418 m | 1.13150 m | −0.08732 m | +0.18150 m |

Both reference rays hit the chair. **Neither independently measured visual-surface standoff is within 0.95±0.10 m**, despite the controller's estimated-band completion. The old saved reference-description string says “box centre”; the recorded `sample_pixel` and executed scorer actually use the upper-box depth pixel. Raw results are preserved; the source description is corrected for future recordings. These are single-ray checks, not full object-distance calibration.

Every-frame actual YOLO reinference matched all **501 new frames**. Integrity verification checked **1,557 hashed files**, synchronized camera/depth/LiDAR/IMU/state timestamps, 0.2-second sample intervals, independently recomputed paths/confidence means and final zero commands. This does not constitute full exploration-controller replay, detector accuracy annotation or proof that every depth return belongs to the chair.

## Evidence index

Paths below are relative to this report unless stated otherwise.

| Evidence | Location |
|---|---|
| Preservation / writable environment | [preservation-final-20261110.json](phase2b/preservation-final-20261110.json); initial `preservation_before.json`; ignored recovery archive at repository `.local-assets/recovery/` |
| Dependency failures/success/imports | `phase2b/install-resume-20261110.log`, `install-resume-network-20261110.log`, `dependencies-20261110.txt`, [full environment snapshot](phase2b/environment-20261110.freeze.txt) |
| Latest automated tests | [JUnit XML](phase2b/tests-validation-20261110.xml), [terminal log](phase2b/tests-validation-20261110.log) |
| PPO run / actual checks | [smoke summary](../results/phase2b/ppo_smoke_20261110/smoke_summary.json), `progress.csv`, `training.monitor.csv`, `validation/evaluations.npz`, `checkpoints/`, `final_model.zip`, [learning curves](../results/phase2b/ppo_smoke_20261110/smoke_learning_curves.png), [loss statistics](phase2b/training-statistics-20261110.json) |
| Independent proxy evaluation | [metrics](../results/phase2b/proxy_eval_20261110/metrics.json), `episodes.csv`, three proxy GIFs and `proxy_comparison.png` |
| Actual successful recordings | [visible summary](../data/phase2b/visible_20261110/summary.json), [initially unseen summary](../data/phase2b/near_20261110/summary.json); both contain `inspection/`, `demo.gif`, raw sensors and source snapshots |
| Actual failed recordings | [occluded summary](../data/phase2b/occluded_20261110/summary.json), [shifted summary](../data/phase2b/shifted_20261110/summary.json); full raw sensors/GIF/snapshots retained |
| Data/reinference verification | [new recordings](phase2b/verified-new-trials-20261110.json), [prior completed recordings](phase2b/verified-prior-trials-20261110.json), `verification-new-20261110.log` |
| Diagnostics / all complete trial summaries | `phase2b/search-diagnostics-20261110.json`, [depth samples](phase2b/depth-diagnostics-20261110.json), `trial-summaries-20261110.json` |
| New actual experiment figures | [paths](../results/phase2b/validation_figures_20261110/actual_yolo_paths.png), [detection continuity](../results/phase2b/validation_figures_20261110/actual_yolo_detection_continuity.png) |
| Preserved Phase 2A | [original implementation report](Phase2A_Implementation_Report.md), `../data/sample/demo_001/`, prior replay reports |
| Reproduction and attribution | `../README.md`, `../q3_rl/README.md`, `../CONTRIBUTIONS.md`, `../requirements-rl.txt` |

## Reproduction and exact remaining work

From the repository root, source `.env.phase1`; use existing `.venv`; run one workload at a time. The READMEs contain the smoke/evaluation commands. For actual-YOLO reproduction, select a config and a **new** output name:

```bash
timeout 120s .venv/bin/python -B -m Test2_Tyler.search \
  --config Test2_Tyler/config/phase2b/near_heading.yaml \
  --out Test2_Tyler/data/phase2b/near_reproduction_001 --wall-limit 90
timeout 90s .venv/bin/python -B -m Test2_Tyler.q2_data_simulation.verify_search \
  --trial Test2_Tyler/data/phase2b/near_reproduction_001 --repeat-yolo \
  --out Test2_Tyler/report/phase2b/near_reproduction_001_verify.json
```

The exact next development task is **validate and repair the RGB-chair/depth-surface association using the preserved occluded failure and calibrated visible controls**. Compare sampled collision-depth ROIs against rendered chair pixels and independent visual/sensor rays; reject foreground-contaminated estimates using observable consistency, without feeding hidden chair locations into control. Calibrate a clearly defined standoff measurement. Then demonstrate one successful inspection that requires forward exploration, while preserving all failures and testing target-loss recovery.

After that:

1. Improve viewpoint/coverage selection and target reacquisition; add a bounded absent-target control and repeated/held-out placements. Keep contact, standoff and visibility failures in results rather than adjusting success labels.
2. Align proxy and actual observation/action/range/inspection definitions, add useful memory/coverage information if necessary, and establish an effective proxy baseline and reachable difficulty stages. Current 36-second episodes and slow actions may make some randomized searches difficult; baseline zero success needs investigation before spending a larger training budget.
3. Review/approve a bounded extended PPO experiment with independent validation and final evaluation seeds. Train only Tyler's new policies, evaluate against baselines and report success/coverage/collisions rather than reward alone. No such run has started here.
4. Evaluate a learned policy using actual YOLO/MuJoCo only after adapter/contract validation. Current proxy evaluation cannot substitute for that experiment.
5. Finish the individual technical report and submission evidence package, clarify any professor-required hardware deliverables, and obtain approval before a GitHub push. Phase 2B establishes the foundation; it does not establish complete Q1–Q3 fulfillment or robust autonomy.
