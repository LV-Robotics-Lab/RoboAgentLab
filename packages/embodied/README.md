# RoboAgentLab

Robotics extensions for the pi agent harness. Robots and algorithms connect through callable services; skills explain their workflows, and shared primitive declarations also expose capabilities to generated Python programs.

[Website](https://roboagentlab.github.io/) · [Get started](https://roboagentlab.github.io/docs/quickstart.html) · [Documentation](https://roboagentlab.github.io/docs/)

## Quickstart

Requires Node.js 22.19+ and a Linux NVIDIA GPU machine for simulation. Build pi from this fork:

```bash
git clone https://github.com/RoboAgentLab/RoboAgentLab ~/RoboAgentLab
cd ~/RoboAgentLab
npm ci --ignore-scripts
node scripts/hydrate-ci-models.mjs
npm run build:offline
npm link --ignore-scripts --workspace @earendil-works/pi-coding-agent
pi install ./packages/embodied
mkdir -p ~/robot-runs
cd ~/robot-runs
pi
```

Inside pi, configure an image-capable planner with `/login`, then run `/embodied-setup`.
The wizard selects a robot, mode, deployment, and experiment directory, installs services,
and runs preflight checks. Follow the printed launch command, then enter `Solve the task.`.

See the [full Quickstart](https://roboagentlab.github.io/docs/quickstart.html) for installation, one LIBERO episode, and the dashboard.

## Guides

- [Environments and robots](https://roboagentlab.github.io/docs/environments.html)
- [Agent interfaces](https://roboagentlab.github.io/docs/modes.html)
- [Services and deployment](https://roboagentlab.github.io/docs/services.html)
- [Evaluation and replay](https://roboagentlab.github.io/docs/evaluation.html)
- [Data collection](https://roboagentlab.github.io/docs/data.html)
- [Develop an integration](https://roboagentlab.github.io/docs/development.html)

<details>
<summary>Robot modules (checked against the extension registrations)</summary>

| Robot | Extension | Success signal | Shared modules |
| --- | --- | --- | --- |
| LIBERO / LIBERO-PRO | `src/robots/libero` | LIBERO `terminated` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, operator, Flash |
| RoboCasa | `src/robots/robocasa` | `env._check_success()` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash |
| RoboTwin | `src/robots/robotwin` | `eval_success` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, XPolicyLab |
| ManiSkill | `src/robots/maniskill` | ManiSkill `success` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash |
| RoboLab | `src/robots/robolab` | RoboLab's task predicate | memory, explore, video, units, VDM, code, `--privileged`, Flash |
| RoboDojo (two ARX X5) | `src/robots/robodojo` | RoboDojo's `is_episode_end` (`score` = partial credit) | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, XPolicyLab |
| Robosuite | `src/robots/robosuite` | robosuite `_check_success` (Restack adds CaP-X's off-table rule), latched | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| Metaworld | `src/robots/metaworld` | Metaworld `info["success"]`, latched | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| Genesis | `src/robots/genesis` | the task predicate (cube_pick: an 8 cm lift) | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| HumanCLAW (SMPL-X humanoid, HSSD homes) | `src/robots/humanclaw` | HumanClawBench's NavSR@20cm | video, units, VDM, `--privileged` |
| BEHAVIOR-1K / R1Pro | `src/robots/behavior` | the BDDL activity's `success`, latched (`q_score` = partial credit) | memory, explore, video, units, VDM, code, `--privileged` |
| Franka (real) | `src/robots/franka` | operator verdict (`--operator`) | memory, explore, video, units, VDM, code, operator |
| Dual Franka (real) | `src/robots/dual_franka` | operator verdict (required) | memory, explore, video, units, VDM, code, operator, XPolicyLab |
| Piper / dual Piper (real) | `src/robots/piper` | operator verdict (required) | memory, explore, video, units, VDM, code, operator |
| UR5e (real) | `src/robots/ur5e` | operator verdict (required) | memory, explore, video, units, VDM, code, operator |

</details>
