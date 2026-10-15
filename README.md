# pi-embodied

Robots as [pi](https://github.com/earendil-works/pi) extensions. A robot is a pi extension that
registers its tools, flags and observations; pi provides the agent loop, the models, sessions and
the interactive, print, JSON and RPC modes. The planner is any model pi can use, so a robot run is a
pi session.

- **Simulators:** LIBERO, LIBERO-PRO, LIBERO-Plus, RoboCasa, RoboTwin, ManiSkill, RoboLab, RoboDojo,
  Robosuite, Metaworld, Genesis, BEHAVIOR-1K, HumanCLAW.
- **Real robots:** Franka, dual Franka, Piper (single and dual), UR5e, with an operator gate on every
  motion.
- **Policy backends:** Pi0.5 (openpi), OpenVLA, OpenVLA-OFT, GR00T, RLDX-1, LingBot-VLA, XPolicyLab
  policies, and fine-tuned Qwen adapters.
- **Modes:** tools (the default: perception, motion and grasp primitives), units (Show-Harness action
  units), code (CaP-X `run_code` and tiers), fine-tuned, Flash replay, MotorMind.
- **Other hosts:** Codex and Claude Code reach the robots' tools mode through an MCP server.

The package lives in [`packages/embodied`](packages/embodied/README.md); the Python env and model
servers live in [`services`](services/README.md).

## Quickstart: one LIBERO episode

This path runs one LIBERO episode in units mode, which needs no VLA checkpoint. It needs Linux with
an NVIDIA GPU and EGL driver, and about 25 GB of free disk (the venv is about 6.5 GB; uv's download
cache can reach 16 GB, set `UV_CACHE_DIR` to put it elsewhere).

Prerequisites:

- **Node.js 22.19 or newer on your `PATH`.** The preflight runs `node` directly. The pi.dev installer
  can install a private Node for pi, but it does not add it to your shell's `PATH`.
- **[uv](https://docs.astral.sh/uv/)** (or Python 3.11), for `services/setup.sh`.
- **GitHub and Hugging Face reachable.** The services' extras install packages from GitHub (RLinf's
  openpi fork with submodules, LIBERO, SAM3), and the LIBERO scene assets come from Hugging Face. See
  [Mainland China network](#mainland-china-network).
- **A planner model with image input**, from a provider pi supports.

```bash
# 1. pi (or the pi.dev installer: curl -fsSL https://pi.dev/install.sh | sh)
npm install -g --ignore-scripts @earendil-works/pi-coding-agent

# 2. this repository and the embodied package
git clone https://github.com/LV-Robotics-Lab/pi-embodied ~/pi-embodied
cd ~/pi-embodied && pi install ./packages/embodied

# 3. uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. an experiment directory
mkdir -p ~/robot-runs && cd ~/robot-runs && pi
```

`pi install` asks whether to trust the repository folder (it has its own `.pi/` directory); answer
yes. Inside pi:

1. `/login` to connect a provider, or configure one in `~/.pi/agent/models.json`
   ([providers](packages/coding-agent/docs/providers.md), [models](packages/coding-agent/docs/models.md)).
   Without a model `/embodied-setup` says how to add one and exits without writing anything.
2. `/embodied-setup`: pick LIBERO, mode units, services here and your model; at
   `Experiment directory (empty: …)` press Enter for the default, `~/robot-runs/embodied-libero`.
   After you confirm, the agent runs `services/setup.sh libero` (venv, packages and the LIBERO scene
   assets, about 430 MB) and the preflight, and ends with the launch command.

The launch command `/embodied-setup` prints (with absolute paths) carries the episode budget:

```bash
cd ~/robot-runs/embodied-libero && source ~/pi-embodied/services/.venv-libero/pi-embodied.env && pi --libero-type standard --suite libero_10 --task 0 --seed 0 --units=true --max-turns 40 --time-limit 900 --dashboard=true
```

With a model that has thinking levels, `--thinking low` keeps each turn's reasoning short (pi clamps
the level to what the model supports).

pi asks once whether to trust the experiment directory (its settings load the robot); answer yes.
The robot resets the scene and waits: type `Solve the task.` to start the episode. The dashboard URL
is printed at startup. The same episode without the terminal UI:

```bash
pi -a -p --libero-type standard --suite libero_10 --task 0 --seed 0 --units=true \
  --max-turns 40 --time-limit 900 --thinking low "Solve the task."
```

`-a` trusts the directory for this run, which print mode cannot ask about. Every turn sends camera
images to the planner, and without `--max-turns` / `--time-limit` an episode runs until the model
calls `finish`, which on a long LIBERO task can take an hour and millions of input tokens. Keep a
budget on every run against a paid API.

The episode ends with a `robot_result` entry in the session (`terminated` is LIBERO's success flag).
On a machine with several GPUs, pick one with `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`.

### Mainland China network

GitHub, Hugging Face, PyPI and the PyTorch wheel index can be slow or unreachable from mainland
China. The install still works through mirrors or a proxy:

- GitHub: the extras clone repositories with submodules during `uv pip install`; route git through
  a proxy (`https_proxy`) if a clone stalls.
- Hugging Face: `HF_ENDPOINT=https://hf-mirror.com`. The mirror rate-limits (HTTP 429) downloads of
  datasets with many files, such as the LIBERO scene assets (`RLinf/LIBERO-PRO-assets`, 865 files) of
  `setup.sh libero` and `libero-pro`; rerun `setup.sh` (it resumes, and links the assets only once
  every file is there), or download from huggingface.co through a proxy. `pi-embodied.env` does not
  set `HF_ENDPOINT`, so export it in the shell that runs pi as well.
- PyPI: `UV_INDEX_URL` / `PIP_INDEX_URL` pointing at a PyPI mirror.
- PyTorch CUDA wheels: a mirror of `download.pytorch.org/whl/cu128`, used with `--find-links`.

## Documentation

- [packages/embodied/README.md](packages/embodied/README.md): the robots, modes, flags, deployment
  config, evaluation scripts, and the Codex / Claude Code plugins.
- [packages/embodied/skills/embodied-quickstart/SKILL.md](packages/embodied/skills/embodied-quickstart/SKILL.md):
  install, preflight, one episode and evaluation for every robot.
- [services/README.md](services/README.md): installing each simulator and model server by hand.
- [services/PROTOCOL.md](services/PROTOCOL.md): the RPC protocol and each env server's methods.
- [Adding a robot](packages/embodied/docs/adding-a-robot.md),
  [adding a primitive](packages/embodied/docs/adding-a-primitive.md),
  [pi compatibility](packages/embodied/docs/pi-compatibility.md),
  [renamed flags](packages/embodied/docs/flags-migration.md).

## Built on pi

pi-embodied is a fork of [earendil-works/pi](https://github.com/earendil-works/pi) by Mario Zechner,
used under the MIT license ([LICENSE-MIT](LICENSE-MIT)). The packages other than `packages/embodied`
are pi's (`packages/ai`, `agent`, `coding-agent`, `tui`, ...); their documentation is in
[packages/coding-agent/docs](packages/coding-agent/docs) and at [pi.dev/docs](https://pi.dev/docs/latest).
Issues with pi itself belong upstream.

The quickstart above uses the official pi. This fork's pi adds two extension APIs the dashboard uses
for session export and withdrawing queued messages; on the official pi those two dashboard actions
report that they are unavailable
([pi compatibility](packages/embodied/docs/pi-compatibility.md)).

## Development

```bash
npm ci --ignore-scripts      # install dependencies without lifecycle scripts
npm run build                # regenerate the model catalog from the providers' live listings, then build
npm run build:offline        # rebuild from the model data already generated, without network access
npm run check                # lint, format, type check
npm test                     # the workspaces' tests
(cd packages/embodied && node --test --experimental-strip-types test/*.test.ts)
services/test.sh             # the Python services' tests and ruff (needs uv)
./pi-test.sh                 # run this checkout's pi
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the build, the checks and these tests
with Node.js 24 on every push to `main` and every pull request. Tests use faux providers and fake
servers: no model API, no GPU. `package-lock.json` is the dependency ground truth; the pre-commit hook
refuses lockfile changes unless `PI_ALLOW_LOCKFILE_CHANGE=1` is set.

## License

The code written for this repository (`packages/embodied`, `services/`, and the changes to pi's
packages) is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). pi is licensed under the MIT
license ([LICENSE-MIT](LICENSE-MIT)), which applies to pi's code in this repository. Third-party
notices are in [NOTICE](NOTICE).
