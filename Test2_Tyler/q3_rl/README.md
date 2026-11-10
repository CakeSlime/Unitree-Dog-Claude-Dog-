# Current submission status

The final submission adds a bounded **8,192-step Phase 2C proxy PPO run** with saved checkpoints and independent paired evaluation. PPO: 0/8 successes; random: 0/8; scan-vantage: 4/8. This verifies the full training pipeline but does not demonstrate useful learned performance. See the [final report and 12-component RL matrix](../report/ENGR6010_Second_Test_Final_Report.md) and [current reproduction commands](../README.md#bounded-reproduction). The Phase 2B discussion below is preserved historical context.

# Tyler's object-search PPO foundation — Phase 2B

The new `ObjectSearchEnv` and 2,048-step CPU PPO smoke test are implemented and validated. These are **detector-proxy** results, not YOLO/MuJoCo policy results. No Connor checkpoints are loaded; useful learned navigation is not demonstrated. See the [Phase 2B report](../report/Phase2B_Implementation_Report.md) for measured results and evidence.

The numeric observation is 38 finite float32 values in `[-1,1]`: the preserved 34-value encoder described below plus four local visited-cell indicators from odometry history. Unseen target features are zero. Internal target coordinates support sensing/scoring but are absent from observations and `info`. Diagnostic rendering is for human evaluation, not policy input.

The two bounded actions are forward `0.18*max(0,a[0])` m/s and yaw `0.35*a[1]` rad/s. Motion has velocity lag and four collision substeps, without Go2 joint dynamics. Seeded episodes randomize free spawn/target positions, heading and divider position. The 24-ray LiDAR proxy ranges to 3 m. The explicitly simulated detector has range 2.7 m, half-FOV 0.61 rad, occlusion checks and 8% dropout.

Rewards: +0.25 per newly visited 0.25 m base cell, +1 first discovery, +3 first confirmation, consecutive-visible range progress clipped to ±0.2 m, +15 inspection, −5 collision, −0.02 per step and −0.015 squared action change. Unseen target geometry supplies no progress reward. Success/collision terminate; 180 steps (36 seconds) truncate. Inspection uses three consecutive confirmed detections at 0.95±0.15 m, bearing <0.1 rad and speed <0.06 m/s.

The proxy uses base-to-point distance; actual MuJoCo uses camera-to-surface depth, ±0.10 m tolerance and a one-second inspection evidence window. These differences must be aligned before policy transfer or a fair actual-YOLO comparison. The proxy's visited-cell reward is not camera coverage. The four memory indicators currently use map-axis offsets, not a full egocentric occupancy map.

Validated environment: Python 3.12.14, Torch 2.15.0+cpu, Gymnasium 1.4.0, NumPy 2.5.4, SB3 2.9.2, Matplotlib 3.10.7. Optional direct dependencies are pinned in `../requirements-rl.txt`; retain CPU Torch and the existing virtual environment. Complete versions are recorded in `../report/phase2b/environment-20261110.freeze.txt`.

Run from the repository root, with **new** output names and one workload at a time:

```bash
source .env.phase1
timeout 60s .venv/bin/python -B -m pytest -q Test2_Tyler/tests
timeout 120s .venv/bin/python -B -m Test2_Tyler.q3_rl.train_smoke \
  --steps 2048 --out Test2_Tyler/results/phase2b/ppo_smoke_reproduction_001
timeout 90s .venv/bin/python -B -m Test2_Tyler.q3_rl.evaluate_proxy \
  --model Test2_Tyler/results/phase2b/ppo_smoke_reproduction_001/final_model.zip \
  --out Test2_Tyler/results/phase2b/proxy_eval_reproduction_001 --episodes 8
```

The smoke runner checks exact step count, finite gradients/losses, changed weights, checkpoints and reloaded inference. Fresh PPO MLP: 64×64, rollout 128, batch 64, two epochs, learning rate 0.0003, γ=0.99, GAE=0.95, entropy 0.01. Seed 601022; CPU one thread; internal wall limit 90 seconds. Checkpoint/evaluation interval 512; three validation episodes on a separate seeded stream. Final evaluation uses eight different seeds per policy, identical initial poses, random and reactive baselines; the final checkpoint is not selected using those evaluation outcomes.

Attribution: `train_smoke.py` adapts the SB3 Monitor/EvalCallback/checkpoint workflow from `Test2/q3_rl/train_ppo.py`. `object_search_env.py` references episode organization in `Test2/q3_rl/crime_scene_env.py` but implements distinct observations, geometry and rewards. Neither imports Connor's task or weights. Extended training remains deferred pending a separately approved budget and difficulty/baseline review.

## Preserved Phase 2A design notes (historical)

No policy has been trained or loaded here. Connor's forensic PPO checkpoints and data remain separate. Phase 2A validates the prerequisite actual-YOLO observation/task/recording path.

`observations.py` defines a proposed 34-value boundary:

| Indices | Robot-observable input |
|---|---|
| 0–23 | 24 samples of measured LiDAR ranges, normalized to 3 m |
| 24–28 | Current target visible flag, confidence, bearing/π, measured range/5 m, reserved current-observation age (zero for current frames) |
| 29–31 | Measured body vx/vy/yaw-rate normalized by Go2 limits |
| 32–33 | Previous applied forward/yaw command normalized by pilot limits |

Hidden chair positions, scene IDs and scoring labels are not arguments. Observed-world estimates can support tracking but are excluded from this initial vector. Tests verify that changing an observation's world target coordinate does not alter the vector. This boundary is not yet a Gymnasium environment or a validated sufficient search observation; exploration memory/coverage and last-seen age handling remain to be designed.

Next, adapt appropriately attributed SB3 vectorized training, callbacks, checkpoint/resume and evaluation patterns to a new object-search environment. Use bounded forward/yaw actions, inspection completion reward, useful coverage/progress to an actually observed target, time/command costs and collision/fall penalties. Ground truth may score outcomes and shape training reward, but must not supply unseen target direction to the policy.

Use identical success/timeout/contact contracts for PPO and the classical baseline; record explicit requested and settled starts rather than assuming equal random seeds imply equal poses. Freeze validation-based selection before final held-out evaluation. If a cheap perception proxy is used during training, distinguish its results from actual YOLO-driven MuJoCo evaluation. Review a modest compute budget before training; full PPO training is deliberately deferred until the basic pipeline is validated.
