# ENGR 6010 Second Test — Final Submission Audit

**Student:** Tyler · **Date:** November 10, 2026 · **Branch:** `tyler-second-test-standalone`
**Individual project:** Autonomous Indoor Object Search and Inspection Using the Unitree Go2  
**Status:** Individual code/report submission published at [tyler-second-test-standalone](https://github.com/CakeSlime/Unitree-Dog-Claude-Dog-/tree/tyler-second-test-standalone/Test2_Tyler), baseline commit `2a15dd7`. Local branch `main` tracks `personal/tyler-second-test-standalone`. These publication corrections are not committed/pushed by the agent. Dataset Release upload and Canvas submission remain pending/unconfirmed.

## Requirement-by-requirement completion

| Professor requirement | Status | Actual evidence and qualification |
|---|---|---|
| Background/motivation | Complete | [Final report](ENGR6010_Second_Test_Final_Report.md), sections 1–2 |
| Q1 vision/ML in dog task | Complete in simulation | [YOLO adapter](../adapters/yolo.py), [calibrated controller](../phase2c/control.py), [successful actual-simulation trial](../data/phase2c/calibration_visible_002/summary.json), [inspection image](../data/phase2c/calibration_visible_002/inspection/annotated.png), [verification](phase2c/verified_trials_001/calibration_visible_002.json) |
| Q1 useful unknown-target translational inspection | Partial | Six exploration trials timed out. Preserve their actual results. Not needed to replace existing successful vision/task demonstrations |
| Q2 actual dog simulation | Complete | Physical Go2 MuJoCo dynamics, bounded commands, camera/LiDAR/IMU and recorded outcomes; [simulator adapter](../phase2c/simulation.py) |
| Q2 data collection | Complete for simulation | [Data format](../data/README.md), [recording code](../q2_data_simulation/recording.py), [lossless archive inventory](../recording_archives/manifest.json), [sensor/path figure](final_figures/simulation_data.png) |
| Physical dog hardware data | Not available | No physical recordings are claimed. If the professor separately requires hardware recordings, that requirement is unmet |
| Q3 all RL components in code | Complete | [Final report section 5](ENGR6010_Second_Test_Final_Report.md#5-q3--rl-formulation-and-complete-ppo-components): all 12 requested components mapped to application/framework code |
| RL training operation | Verified | [2,048-step preserved smoke](../results/phase2b/ppo_smoke_20261110/smoke_summary.json); [8,192-step new run](../results/phase2c/ppo_submission_20261110/smoke_summary.json): finite gradients/losses, changed weights, 16 checkpoints and successful reload |
| Useful trained RL policy/convergence | Not demonstrated | Final proxy PPO 0/8 successes, random 0/8, scan-vantage 4/8. [All 24 new episodes](../results/phase2c/ppo_submission_eval_20261110/episodes.csv). No convergence/transfer claim |
| Q1/Q2/Q3 separate methods/results | Complete | [Final report Markdown](ENGR6010_Second_Test_Final_Report.md), [DOCX](ENGR6010_Second_Test_Final_Report.docx), [PDF](ENGR6010_Second_Test_Final_Report.pdf) |
| Detailed README and reproduction | Complete | [Current README](../README.md), [Phase 2C commands](../phase2c/README.md), [dependencies](../requirements.txt), [RL dependencies](../requirements-rl.txt), [environment](phase2c/environment.freeze.txt) |
| Code/config/models/evidence | Code/report published; small evidence corrections prepared | Published code/models plus the restored small evidence are enumerated by the publication correction allowlist; raw streams and shared infrastructure remain separate. YOLO weights are checksum-pinned external downloads; individual PPO models are delivered |
| Accurate attribution | Complete | [Individual contributions](../CONTRIBUTIONS.md), [Phase 2C sources](../phase2c/CONTRIBUTIONS.md), report references; SB3 internals identified as framework code |
| Shared original `Test2/` preservation | Verified in final integrity record | 1,468-file baseline compared by path/SHA-256; no original group results relabeled |
| All writing/code/data pushed to GitHub | Code/report published; full raw dataset pending | [Archive manifest](../recording_archives/manifest.json) included with this correction; 78 raw archive assets require separate Release upload |
| GitHub address submitted on Canvas | Pending student action | Use the final published individual branch/directory address below |

## Final verification and integrity

- [50 passing tests](submission_tests.xml), zero failures/errors/skips.
- [Historical development integrity record](submission_integrity.json): original full-workspace checks, not a fresh standalone audit. [Current publication link audit](publication_link_audit.json) checks the standalone deliverables. The legacy `verify_submission.py` requires the original full workspace and must not be used to regenerate standalone staging lists.
- [Archive verification](archive_verification.json): all 11,555 archived raw stream files checked against local originals, plus ZIP CRC and part hashes.
- [Nine completed Phase 2C verification records](phase2c/verified_trials_001/): every-frame actual YOLO inference matched; timestamps, path totals, hashes and terminal stops checked. Two calibrated initially-visible successes, six target-present exploration timeouts, one absent-target timeout.
- The 8,192-step experiment's [source snapshot](../results/phase2c/ppo_submission_20261110/source_snapshot/) records application source and dependency hashes. Its actor/critic/PPO implementation is supplied by the pinned installed SB3 version.
- Existing reports and historical README are retained. Current final report/README supersede historical statements about missing RL checkpoints or uncalibrated depth. Source snapshots preserve the implementations used by older experiments.

DOCX and PDF were generated locally with existing tools; Markdown remains authoritative for clickable evidence links. DOCX package XML and PDF signature were checked. No external Word/LibreOffice rendering was available for visual layout validation.

## Standalone publication findings

The standalone branch excludes the shared source trees and Phase 2B/2C recording directories. The original README/report therefore linked to missing summaries, calibrated inspection images and GIFs. This correction copies only missing, byte-identical non-stream evidence from the original development workspace; no results are recomputed, edited or overwritten. Raw RGB/depth/legacy-depth/LiDAR remain excluded.

The locally present archive manifest was ignored along with the 78 large archive parts. The ignore rule now permits the manifest while continuing to exclude `.zip.part*` assets. The manifest describes 20 trials, 11,555 raw stream files and 2,759,741,405 compressed bytes. Its contents are unchanged. [Release instructions](../RECORDINGS_RELEASE.md) provide a separate safe upload route; until those assets are published, the full dataset is not remotely accessible.

The shared `sim_env`, navigation and `go2_stream` imports are absent from the standalone branch. [PUBLICATION_SETUP.md](../PUBLICATION_SETUP.md) documents a sparse checkout at the original shared baseline, its pinned simulator submodule and non-overwriting local symlinks. Framework attribution is preserved. Recorded preservation/test results remain historical evidence; the standalone copy cannot independently compare the omitted group tree until dependencies are obtained.

## Exact reproduction and verification commands

Run from `/home/tyler/projects/engr6010/ENGR6010-Tyler-Final` after [dependency setup](../PUBLICATION_SETUP.md) and [Release download](../RECORDINGS_RELEASE.md):

```bash
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/tyler-matplotlib
# On a fresh clone, restore raw sensors before replay:
.venv/bin/python -B Test2_Tyler/package_recordings.py restore
.venv/bin/python -B -m Test2_Tyler.assets
timeout 60s .venv/bin/python -B -m pytest -q Test2_Tyler/tests
# New names are required for every reproduction experiment:
timeout 110s .venv/bin/python -B -m Test2_Tyler.q3_rl.train_phase2c \
  --steps 8192 --seed 601110 --out Test2_Tyler/results/phase2c/ppo_reproduction_NEW
timeout 110s .venv/bin/python -B -m Test2_Tyler.q3_rl.evaluate_phase2c \
  --model Test2_Tyler/results/phase2c/ppo_reproduction_NEW/final_model.zip \
  --out Test2_Tyler/results/phase2c/eval_reproduction_NEW --episodes 8 --seed 811110
# Fast report rebuild/export; no training or simulator launched:
.venv/bin/python -B Test2_Tyler/submission_tools/build_report.py
.venv/bin/python -B Test2_Tyler/submission_tools/export_report.py
# Recheck originals and delivery (bounded read-only checks):
timeout 120s .venv/bin/python -B Test2_Tyler/package_recordings.py verify
# The historical verify_submission.py requires the original full workspace.
# Current standalone link findings are in publication_link_audit.json.
```

For new actual MuJoCo runs, use the [Phase 2C README](../phase2c/README.md). The [final report](ENGR6010_Second_Test_Final_Report.md#6-verification-and-reproduction) includes the recorded-trial actual-YOLO verification command.

## Publish only these small corrections

The local branch is `main`; the published branch is `tyler-second-test-standalone` on remote `personal`. Do not push `main` to the remote's `main` branch. The explicit correction allowlist is [publication_corrections_paths.txt](publication_corrections_paths.txt). Legacy `staging_paths.txt` and `staging_batches/` describe the original full-workspace publishing plan and must not be used here.

```bash
cd /home/tyler/projects/engr6010/ENGR6010-Tyler-Final
test "$(git branch --show-current)" = main || exit 1
test -z "$(git diff --cached --name-only)" || exit 1
git add --pathspec-from-file=Test2_Tyler/report/publication_corrections_paths.txt || exit 1
git diff --cached --stat
git diff --cached --check || exit 1
# After reviewing staged corrections:
git commit -m "Fix standalone publication links, dependencies and dataset delivery" || exit 1
git push personal HEAD:tyler-second-test-standalone
```

No history rewrite, force push or recording deletion is needed. A rejected non-fast-forward push requires reconciliation with remote changes, never force. Publish the raw data separately using [the Release plan](../RECORDINGS_RELEASE.md); the draft Release is deliberately not created by this audit.

**Canvas address:** https://github.com/CakeSlime/Unitree-Dog-Claude-Dog-/tree/tyler-second-test-standalone/Test2_Tyler

Experimental results remain unchanged: final proxy PPO 0/8 successes, random 0/8, scan-vantage 4/8; no physical recordings or robust translational-search inspection is claimed. Remaining actions are review/commit/push of these corrections, separate raw dataset publication and confirmation of Canvas submission.
