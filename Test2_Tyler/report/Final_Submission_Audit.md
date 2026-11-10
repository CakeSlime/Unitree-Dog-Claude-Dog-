# ENGR 6010 Second Test — Final Submission Audit

**Student:** Tyler · **Date:** November 10, 2026 · **Branch:** `engr6010-second-test`  
**Individual project:** Autonomous Indoor Object Search and Inspection Using the Unitree Go2  
**Status:** Local submission package prepared for review. Nothing staged, committed or pushed by the agent. Canvas submission remains the student's action.

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
| Code/config/models/evidence | Prepared locally | Entire individual workspace included via explicit staging allowlist. YOLO weights are checksum-pinned external downloads; individual PPO models are delivered |
| Accurate attribution | Complete | [Individual contributions](../CONTRIBUTIONS.md), [Phase 2C sources](../phase2c/CONTRIBUTIONS.md), report references; SB3 internals identified as framework code |
| Shared original `Test2/` preservation | Verified in final integrity record | 1,468-file baseline compared by path/SHA-256; no original group results relabeled |
| All writing/code/data pushed to GitHub | Pending review/publication | Safe commands below. Not completed until every publication batch reaches GitHub |
| GitHub address submitted on Canvas | Pending student action | Use the final published individual branch/directory address below |

## Final verification and integrity

- [50 passing tests](submission_tests.xml), zero failures/errors/skips.
- [Final integrity record](submission_integrity.json): original group preservation, per-trial recorded hashes, archive coverage, document link existence, DOCX XML/PDF signature and candidate staging paths.
- [Archive verification](archive_verification.json): all 11,555 archived raw stream files checked against local originals, plus ZIP CRC and part hashes.
- [Nine completed Phase 2C verification records](phase2c/verified_trials_001/): every-frame actual YOLO inference matched; timestamps, path totals, hashes and terminal stops checked. Two calibrated initially-visible successes, six target-present exploration timeouts, one absent-target timeout.
- The 8,192-step experiment's [source snapshot](../results/phase2c/ppo_submission_20261110/source_snapshot/) records application source and dependency hashes. Its actor/critic/PPO implementation is supplied by the pinned installed SB3 version.
- Existing reports and historical README are retained. Current final report/README supersede historical statements about missing RL checkpoints or uncalibrated depth. Source snapshots preserve the implementations used by older experiments.

DOCX and PDF were generated locally with existing tools; Markdown remains authoritative for clickable evidence links. DOCX package XML and PDF signature were checked. No external Word/LibreOffice rendering was available for visual layout validation.

## Non-destructive data and Git organization

The warning about active changes reflected more than 12,000 candidate files. Only regenerable environments/caches and raw stream directories with complete lossless archived copies are ignored. No recordings, models, reports or experiments were deleted. No tracked group source was changed.

All raw streams from **20 finalized or interrupted trials** are in **78 archive parts**, each at most 40 MiB. The complete compressed raw archive payload is **2,759,741,405 bytes (about 2.57 GiB)**. Metadata, summaries, inspection images, state/event tables, source snapshots, plots and GIFs remain directly browsable. Data restoration preserves original filenames and checksums. PPO checkpoint directories are explicitly included rather than hidden by the old generic checkpoint ignore rule.

GitHub enforces a 2 GiB limit for one push; archive delivery is therefore split across smaller commits/pushes. This is an actual constraint for this package, not a reason to omit data. [GitHub's push-limit documentation](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit) recommends smaller pushes. Each generated data batch is bounded to 600 MiB. The overall upload remains several gigabytes and depends on network throughput. Avoid squashing all batches and then attempting the first push as one operation.

The publication scope is only `.gitignore` plus audited `Test2_Tyler/` paths. Other existing untracked `docs/` material remains outside the staging plan. No blanket `git add .`, force push, reset, clean, deletion or merge into `main` is needed.

## Exact reproduction and verification commands

Run from `/home/tyler/projects/engr6010/Unitree-Dog-Claude-Dog-`:

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
timeout 120s .venv/bin/python -B Test2_Tyler/submission_tools/verify_submission.py
```

For new actual MuJoCo runs, use the [Phase 2C README](../phase2c/README.md). The [final report](ENGR6010_Second_Test_Final_Report.md#6-verification-and-reproduction) includes the recorded-trial actual-YOLO verification command.

## Exact staging/publication commands — run only after reviewing the package

The generated [full allowlist](staging_paths.txt) and [batch directory](staging_batches/) are reviewable text files. `core.txt` contains code/docs/models and small evidence, followed by separate `data_*.txt` files listing archive parts. The integrity script regenerates these lists without staging anything. Review any new experiment outputs before rerunning it because they enter the individual-workspace candidate list.

```bash
cd /home/tyler/projects/engr6010/Unitree-Dog-Claude-Dog-
# Verify correct branch and no unrelated pre-staged work; stop on any failure.
test "$(git branch --show-current)" = engr6010-second-test || exit 1
test -z "$(git diff --cached --name-only)" || exit 1
.venv/bin/python -B Test2_Tyler/submission_tools/verify_submission.py || exit 1
# Read the allowlist and final audit before the first commit/push.
git status --short
cat Test2_Tyler/report/staging_batches/core.txt
# Stage only the reviewed explicit core paths, then inspect the staged result.
git add --pathspec-from-file=Test2_Tyler/report/staging_batches/core.txt || exit 1
git diff --cached --stat
git diff --cached --check || exit 1
# After reviewing the staged core:
git commit -m "Complete Tyler ENGR6010 Second Test code, report and evidence" || exit 1
git push -u origin engr6010-second-test || exit 1
# After reviewing data_*.txt, publish every complete archive batch separately.
for batch in Test2_Tyler/report/staging_batches/data_*.txt; do
  git add --pathspec-from-file="$batch" || exit 1
  git diff --cached --stat
  git commit -m "Add Tyler lossless sensor recordings: $(basename "$batch" .txt)" || exit 1
  git push origin engr6010-second-test || exit 1
done
# Confirm no listed delivery files remain untracked or modified:
git status --short -- Test2_Tyler .gitignore
git log -6 --oneline
```

If a push fails after its commit succeeds, retry that same `git push origin engr6010-second-test` before proceeding to the next data batch. Do not reset commits or force push. If the branch has remote changes, stop and reconcile rather than overwrite them. Verify all archive parts from the manifest are visible on GitHub after the final batch, and check the README/report/model links there.

**Canvas submission address after publication:**  
https://github.com/CDJ7/Unitree-Dog-Claude-Dog-/tree/engr6010-second-test/Test2_Tyler

Remaining actions are human review, all GitHub publication batches and Canvas submission. Remaining technical limitations are disclosed above; no additional open-ended exploration work or long training is needed to present the implemented requirements honestly.
