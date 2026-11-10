# Standalone dependency setup

The individual branch contains `Test2_Tyler/`, not the original shared `Phase_2_Investigator/`, `go2_stream/`, simulator assets or `.gitmodules`. `git submodule update` in the standalone root alone cannot obtain them. Pip packages alone do not satisfy these imports.

| Required shared dependency | Use |
|---|---|
| `Phase_2_Investigator/sim_env` | Sensor dataclasses, simulation runtime, scene overlays and MuJoCo session |
| `Phase_2_Investigator/navigation` | Existing local safety filter imported after adapter setup |
| `go2_stream/perception/detector.py` | Existing Detection dataclass, also used by tests/replay; importing the dataclass does not require constructing the Ultralytics backend |
| `Phase_2_Investigator/third_party/robot-unitree-go2-mujoco` | Native physical Go2 controller, sensor suite, model and house assets |

Shared source baseline: `adf00d308ce3633de357eed05d5e9e0e299ec38e` from [CDJ7's group repository](https://github.com/CDJ7/Unitree-Dog-Claude-Dog-/tree/adf00d308ce3633de357eed05d5e9e0e299ec38e). Its simulator gitlink is `d6e54647541fb7b048db5d9817457ee11d0e2cf9`, from [syswonder](https://github.com/syswonder/robot-unitree-go2-mujoco). These are reused shared/third-party implementations, not newly authored individual code or locomotion training. See [contributions](CONTRIBUTIONS.md).

## Fresh clone: obtain pinned shared source without overwriting work

Run in the standalone repository root. Commands deliberately refuse existing destination paths. If these paths already exist, inspect them and verify revisions instead of replacing them.

```bash
cd ~/projects/engr6010/ENGR6010-Tyler-Final
test ! -e .shared-go2 && test ! -L .shared-go2 || exit 1
test ! -e Phase_2_Investigator && test ! -L Phase_2_Investigator || exit 1
test ! -e go2_stream && test ! -L go2_stream || exit 1
git clone --filter=blob:none --no-checkout https://github.com/CDJ7/Unitree-Dog-Claude-Dog-.git .shared-go2 || exit 1
git -C .shared-go2 sparse-checkout init --cone || exit 1
git -C .shared-go2 sparse-checkout set Phase_2_Investigator go2_stream || exit 1
git -C .shared-go2 checkout --detach adf00d308ce3633de357eed05d5e9e0e299ec38e || exit 1
git -C .shared-go2 submodule update --init --depth 1 -- Phase_2_Investigator/third_party/robot-unitree-go2-mujoco || exit 1
test "$(git -C .shared-go2 rev-parse HEAD)" = adf00d308ce3633de357eed05d5e9e0e299ec38e || exit 1
test "$(git -C .shared-go2/Phase_2_Investigator/third_party/robot-unitree-go2-mujoco rev-parse HEAD)" = d6e54647541fb7b048db5d9817457ee11d0e2cf9 || exit 1
ln -s .shared-go2/Phase_2_Investigator Phase_2_Investigator || exit 1
ln -s .shared-go2/go2_stream go2_stream || exit 1
```

The shared checkout and links are local ignored dependencies. Do not copy or relabel its `Test2/` results. Network access to the original group repository and simulator is required. The audit verified the imports and pinned revisions against the existing development checkout; it did not perform another network clone or simulation run.

## Python and environment

Use an existing working Python 3.12 environment when available. This publishing checkout has no bundled `.venv`. For a fresh environment, after installing Python 3.12 and uv:

```bash
# Only if .venv does not already exist:
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --only-binary :all: -r Test2_Tyler/requirements.txt
# Pin CPU Torch before SB3 to avoid pulling a CUDA stack:
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
uv pip install --python .venv/bin/python -r Test2_Tyler/requirements-rl.txt
uv pip check --python .venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/tyler-matplotlib
.venv/bin/python -B -m Test2_Tyler.assets
timeout 60s .venv/bin/python -B -m pytest -q Test2_Tyler/tests
```

The [recorded environment](report/phase2c/environment.freeze.txt) documents the versions actually used. EGL also needs a working host OpenGL/EGL stack. No ROS, physical dog SDK, newly trained detector or Colab subscription is required. Restore archived raw streams using [dataset instructions](RECORDINGS_RELEASE.md) before replay. PPO proxy scripts need the Python RL packages; actual simulation/tests also import the shared modules above.

The historical `submission_tools/verify_submission.py` and `build_report.py` assume full development recordings/group files. Do not run them blindly in this reduced standalone checkout. The final Markdown report and its export are already supplied; `export_report.py` can export that Markdown once its small linked inspection image and Matplotlib/Pillow are available.
