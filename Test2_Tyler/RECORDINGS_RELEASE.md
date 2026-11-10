# Complete recordings: separate GitHub Release delivery plan

**Status:** Code/report branch published. Dataset archive assets are prepared locally; this audit does not create or publish a Release. The [unchanged tracked manifest](recording_archives/manifest.json) describes 20 trials, 11,555 stream files, 78 parts and 2,759,741,405 bytes (2.57 GiB). Large parts stay ignored; small metadata, summaries, inspections, GIFs and state/event tables are committed separately.

Use a dedicated Release, **`tyler-second-test-data-20261110`**, on **CakeSlime/Unitree-Dog-Claude-Dog-**, targeted at the published **tyler-second-test-standalone** branch. The tag is a proposal, not a claim that a Release exists. GitHub supports Release assets under 2 GiB each; our parts are ≤40 MiB and there are 79 assets including the manifest. See [official Release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) and [CLI upload behavior](https://cli.github.com/manual/gh_release_upload). Avoid uploading one combined 2.6 GB file or committing all parts in a single initial Git push.

## Upload after reviewing and pushing the small publication corrections

The GitHub CLI `gh` was not available in the publishing checkout's command environment during this audit. Install/authenticate it first, or use GitHub's Releases web page to create a draft and attach the manifest plus the 78 unchanged part files. Never replace existing Release assets by name without investigating their hashes.

```bash
cd ~/projects/engr6010/ENGR6010-Tyler-Final
gh auth login
# Refuse an existing tag/release rather than silently replacing it.
# If it already exists, inspect it and use the helper to resume matching uploads.
gh release create tyler-second-test-data-20261110 \
  --repo CakeSlime/Unitree-Dog-Claude-Dog- \
  --target tyler-second-test-standalone --draft \
  --title "Tyler ENGR6010 Second Test — recorded simulation dataset" \
  --notes "Lossless simulated Go2 sensor streams. See manifest.json for part sizes and SHA-256. Includes successful, failed and interrupted trials; no hardware data or successful learned PPO policy is claimed."
# Verifies every local part before any upload; uploads one missing asset at a time.
python3 Test2_Tyler/submission_tools/release_recordings.py upload
# A separate pass verifies all remote asset names, sizes and SHA-256 digests.
python3 Test2_Tyler/submission_tools/release_recordings.py check
# Only after the helper reports every remote asset verified:
gh release edit tyler-second-test-data-20261110 \
  --repo CakeSlime/Unitree-Dog-Claude-Dog- --draft=false
```

The helper never uses `--clobber`, deletes assets, rewrites tags, or overwrites local recordings. Matching remote assets are skipped only after digest verification; differing assets stop the run. If GitHub omits an asset digest, the helper downloads that asset to a temporary file and hashes it before skipping. After interruption, rerun `upload`; it verifies and skips completed assets. The helper cannot be exercised against a real Release until the student authorizes/creates it. No remote mutation is performed by this audit.

After publishing, the dataset URL will be:

`https://github.com/CakeSlime/Unitree-Dog-Claude-Dog-/releases/tag/tyler-second-test-data-20261110`

Update the README/audit status to confirm data publication only after the Release is public and all assets verify. Canvas can link the individual branch, which links to this delivery plan. Until then, the full dataset delivery requirement remains partial.

## Reader download and restore

Once the Release exists and is public:

```bash
cd ENGR6010-Tyler-Final
# The helper downloads only absent parts to temporary files, checks hashes,
# then renames without replacing existing files. Requires authenticated gh.
python3 Test2_Tyler/submission_tools/release_recordings.py download
python3 Test2_Tyler/package_recordings.py verify
python3 Test2_Tyler/package_recordings.py restore
```

Without gh, download all manifest-listed parts through the Release page into `Test2_Tyler/recording_archives/`; keep the tracked manifest and exact filenames. Run the same verify/restore commands. Each trial ZIP is split across ordered parts; downloading only `part000` is insufficient for multi-part trials.

The archives contain raw `rgb/`, `depth/`, `legacy_depth/` and `lidar/` streams only. Small metadata/state/event/summary/inspection evidence is supplied in the branch correction. Preserve both the Git branch and Release assets to reproduce the complete dataset. Original development recordings remain untouched.
