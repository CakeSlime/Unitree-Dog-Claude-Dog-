# Phase 2C contribution and attribution record

Tyler's individual application, developed with coding-agent assistance. Directory location does not prove authorship. Connor's `Test2/` and all prior Phase 2A/2B source, weights and recordings are preserved; none are presented as newly authored or newly trained results.

| Category | Source and contribution |
|---|---|
| Shared / unchanged | Unitree Go2 model, MuJoCo, native sensor suite and physical scripted trot through `Phase_2_Investigator/sim_env` and the pinned `robot-unitree-go2-mujoco` submodule. No locomotion training or hardware operation claimed |
| Shared / unchanged | `navigation.local.SafetyLayer`; imported by the individual runner for measured sensor collision filtering |
| Shared algorithms, previously adapted | `q1_vision/sensor_map.py` and `exploration.py`, adapted in Phase 2B from shared occupancy/frontier/Dijkstra code; reused by new `ProgressExplorer` |
| Third-party pretrained detector | Original THU-MIG YOLOv10n / ONNX Community export and ONNX Runtime, existing SHA pinned; no new YOLO training. `phase2c/detector.py` adapts Tyler's original CPU preprocessing and adds explicit clipped-discovery metadata, preserving the confidence threshold |
| Adapted own simulation adapter | `phase2c/simulation.py` extends `adapters/search_sim.py`: whole-image RGB-registered rendered depth, separate legacy sensor data and generated individual scene variants. Semantic mesh rays remain post-decision evaluation only |
| New perception / task code | `phase2c/perception.py`: foreground depth components, validity/support/background/temporal checks, measured sample pixel and uncertainty-aware standoff predicate. `control.py`: versioned guards, candidate hold, camera sweeps and progress checks over the existing mission/explorer |
| Adapted planning, new integration | `phase2c/approach.py` uses the previously attributed sensor-map/Dijkstra adaptation to route toward standoff-ring cells around a current measured surface. No hidden target pose is used. Missing-target stop and the final calibrated-reacquisition guard are preserved and regression-tested; full forward-search inspection remains unverified |
| Adapted own recording runner | `phase2c/run.py` derives from Phase 2B `search.py`; retains the preserved recorder format while adding depth-quality diagnostics, separate legacy depth and evaluation references. New trials use exclusive paths |
| New diagnostics / verification | `phase2c/diagnose.py`, `panorama.py`, `probe.py`, `verify.py`, comparison figures, regression tests and raw Phase 2C recordings; failed development attempts retained |
| Adapted own proxy environment | `q3_rl/feasible_env.py` versions Phase 2B `ObjectSearchEnv` with camera-surface sensing, tighter inspection margin and a realistic exploration budget. Prior task originally referenced Connor's episode organization; geometry/rewards/observations are Tyler's separately identified implementation |
| New baseline / experiments | `q3_rl/search_baseline.py`, `feasibility.py`, `diagnose_frozen.py`: observation-only stateful baseline, paired budget comparisons and inference-only diagnostics of the old policy. Proxy results remain distinct from YOLO/MuJoCo |
| Adapted training scaffold, executed at submission | `q3_rl/train_phase2c.py` derives from Tyler's `train_smoke.py`, which attributed SB3 Monitor/EvalCallback/checkpoint structure to `Test2/q3_rl/train_ppo.py`. SB3 supplies PPO. A new 8,192-step submission run and independent paired baseline evaluation are included; useful learned performance is not demonstrated |

Original attribution remains in `../CONTRIBUTIONS.md`; this addendum does not overwrite it. Exact source snapshots and configuration are stored with new experiments. The Phase 2C preservation manifest and final report document unchanged prior files and actual outcomes.

## Final submission additions (November 10, 2026)

`q3_rl/evaluate_phase2c.py` implements paired new-seed evaluation of the final calibrated proxy PPO checkpoint against random and the observation-only scan-vantage baseline. `package_recordings.py` delivers lossless split archives without removing raw recordings. `submission_tools/` builds the course report/figures, exports DOCX/PDF and checks evidence/preservation/links/staging paths. These individual additions were developed with coding-agent assistance. Stable-Baselines3 supplies PPO networks, collection, GAE, optimization and persistence; no custom PPO internals are claimed.
