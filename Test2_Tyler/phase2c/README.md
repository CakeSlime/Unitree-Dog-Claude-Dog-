# Phase 2C: calibration, observable search and RL feasibility

This version preserves all Phase 2A/2B source, models and evidence. See the measured results and current limitations in [the implementation report](../report/Phase2C_Implementation_Report.md). Code was developed with coding-agent assistance; [attribution](CONTRIBUTIONS.md) distinguishes shared algorithms from new work.

Use the existing Python 3.12 `.venv`, from the repository root. No additional packages are required. Run one experiment at a time. Every output path must be new.

```bash
source .env.phase1
uv pip check --python .venv/bin/python
timeout 60s .venv/bin/python -B -m pytest -q Test2_Tyler/tests

# Actual YOLO and registered depth gate, before navigation:
timeout 45s .venv/bin/python -B -m Test2_Tyler.phase2c.probe \
  --config Test2_Tyler/config/phase2c/reachable_search.yaml --spawn 2.8 3.35 0 \
  --out Test2_Tyler/results/phase2c/reachable_gate_reproduction_001

# Evaluation-only frozen-base visibility test: not a robot motion demonstration.
timeout 65s .venv/bin/python -B -m Test2_Tyler.phase2c.panorama \
  --config Test2_Tyler/config/phase2c/reachable_search.yaml \
  --out Test2_Tyler/results/phase2c/reachable_panorama_reproduction_001

# Actual physical MuJoCo / YOLO search and raw sensor recording:
timeout 150s .venv/bin/python -B -m Test2_Tyler.phase2c.run \
  --config Test2_Tyler/config/phase2c/reachable_search.yaml \
  --out Test2_Tyler/data/phase2c/reachable_reproduction_001 --wall-limit 125

timeout 120s .venv/bin/python -B -m Test2_Tyler.phase2c.verify \
  --trial Test2_Tyler/data/phase2c/reachable_reproduction_001 --repeat-yolo \
  --out Test2_Tyler/report/phase2c/reachable_reproduction_verify_001.json

# Fast 2-D PROXY-only feasibility/budget comparison, not MuJoCo or training:
timeout 110s .venv/bin/python -B -m Test2_Tyler.q3_rl.feasibility \
  --out Test2_Tyler/results/phase2c/proxy_reproduction_001 --seed 903001 --episodes 8
```

`reachable_absent.yaml` hides the chair and limits the task to 10 simulation seconds. Exit code 2 denotes a failed task; preserve its recording. Older configs and failures remain separate. Scene variations are development experiments, not a held-out autonomy benchmark. Check actual outcomes in each summary; a named scenario does not establish that it succeeded.

## Sensor and inspection contract

The legacy camera combines visual RGB with 160×120 collision-geometry depth, so a chair box can receive a foreground collision-proxy return. It cannot be fixed by subtracting a constant offset. `simulation.py` additionally produces a full-frame registered simulated RGB-D image using the SAME camera and render visibility mask as RGB. `legacy_depth/` preserves the collision depth separately. This is an ideal simulated depth-sensor model, not a claim about physical Go2 RGB-D hardware or noisy real-sensor accuracy.

Navigation receives the RGB/depth/LiDAR/pose/IMU observations only. Neither semantic geom IDs nor hidden target coordinates enter `perception.py`, `control.py` or the policy vector. `score_surface` semantic mesh rays and panorama segmentation are evaluation-only; their outputs do not alter navigation. The simulator necessarily renders scene geometry into sensor images, as it does for RGB, without providing object identities in the depth image.

A YOLO box is not a mask. `perception.py` finds a near-depth component in the upper box, requiring vertical/interior support, background separation, valid local depth and temporal world consistency bounded by observable object extent. Foreground strips filling the ROI, invalid near depth and discontinuous estimates are rejected. The selected interior pixel is recorded. It may change across views. This method is conservative and heuristic: it can reject a valid chair against a nearby background and cannot guarantee semantic association in all clutter.

**Standoff:** horizontal world-XY distance from the current camera optical centre to the selected measured visible surface patch. It is not object-centre range, base range or nearest distance to the whole chair. Inspection requires `abs(range−0.95)+margin <= 0.10 m`, with a margin of at least 0.025 m derived from local spread, bearing ≤0.10 rad, horizontal speed <0.06 m/s and three consecutive fresh stable observations after confirmation. The margin is an engineering consistency allowance, not a statistically validated confidence interval. Contacts, tilt, stale data, timeout and local safety guards remain active. A current missing/invalid estimate cannot complete inspection.

The actual detector keeps confident horizontally clipped discoveries for rotation-only centering. They are rejected as range/inspection evidence. Vantages have bounded camera sweeps; a briefly lost confirmation candidate holds stationary for reacquisition. Target-loss recovery never supplies hidden coordinates. Observed maps, path, quality reasons, post-decision references, source snapshots, RGB/depth/legacy-depth/LiDAR/IMU/state/commands and outcomes are retained separately for each run. `demo.gif` is drawn from actual camera frames, not generated imagery. A post-decision completion counts as evaluated success only if the independent ray hits the chair, reference standoff is in band and no contacts occurred.

`approach.py` routes through the observed occupancy map to reachable standoff-ring cells around a current depth observation. It does not command motion without a current valid target. The final recovery guard also removes the inherited eight-second rejection of a lost calibrated target; reacquisition must still earn fresh confirmations. These conservative rules do not yet establish successful inspection after translational exploration: turning to follow a route can lose the camera target and stop progress. See all recorded outcomes before claiming autonomous task completion.

## Proxy task and training boundary

`q3_rl/feasible_env.py` versions the old environment; it does not alter old checkpoints or environment code. The 38-dimensional vector still contains current LiDAR/detection/velocity/previous-action/local-visit features, with no unseen target coordinates or oracle bearing. The camera offset is 0.32 m, measured approximately from the simulated camera/base transform. Standoff is camera-to-point-surface distance. The proxy reserves 0.025 m within the nominal ±0.10 m band and requires consecutive confirmation/hold frames. Its detector is explicitly a range/FOV/occlusion/dropout proxy with fixed confidence, not YOLO.

The time budget is 1,200×0.2 s = 240 s, at the same maximum forward/yaw speeds (0.18 m/s, 0.35 rad/s). The old 180-step task is retained for paired budget ablations. A 36 s episode allows at most 6.48 m straight travel with no turning; a full scan alone takes about 17.95 s. Unknown-target exploration, detours and inspection often cannot fit that old budget. No hidden route information is added to solve this.

`search_baseline.py` receives only the vector, integrates sensed odometry, scans and selects LiDAR-verified, less-visited local vantages. Target approach is proportional and stops for fresh inspection. It has memory; the old PPO smoke policy is a feedforward MLP. This difference and the idealized proxy sensing limit comparisons. Successful baseline episodes establish task feasibility, not a trained PPO result or safety guarantee.

`train_phase2c.py` is a versioned, bounded PPO smoke entry point importing the new task; the old Phase 2B trainer remains unchanged. It was subsequently executed for the bounded final submission check: 8,192 transitions, followed by 24 paired evaluation episodes. See the [final report](../report/ENGR6010_Second_Test_Final_Report.md) for measured results; useful learning is not demonstrated. A short reproduction can use a new output path:

```bash
timeout 120s .venv/bin/python -B -m Test2_Tyler.q3_rl.train_phase2c \
  --steps 2048 --out Test2_Tyler/results/phase2c/ppo_check_NEW
```

Do not relabel the old 2,048-step policy as trained on the new task. Review the report's readiness assessment and a compute/evaluation budget before any extended training. Actual YOLO/MuJoCo policy transfer, robust exploration over new layouts and physical operation are separate, unverified stages.
