# RoboAgentLab releases

## v1.0 — 20 October 2026

RoboAgentLab makes robot tasks and robotics development accessible through a shared
coding-agent workspace. pi provides the agent loop and persistent sessions; robotics
extensions expose downstream simulation, hardware, perception, planning, and policy
services. Skills explain setup and use, and the same declared capabilities can be
called by agent tools and generated Python programs.

This source release includes:

- Robotics extensions for the simulators and real-arm configurations listed in the
  [quickstart](README.md).
- Service implementations and the shared [RPC protocol](services/PROTOCOL.md).
- Behavior-unit, tool, and code interfaces, with memory, exploration, replay,
  dashboard, and data-export modules.
- Codex and Claude Code integrations through MCP.
- Setup, evaluation, and test utilities.

Use the per-backend documentation for assets, model checkpoints, system dependencies,
and hardware configuration. Docker images remain planned in this release.

The release consolidates the RoboAgentLab changes from
`LV-Robotics-Lab/RoboAgentLab` at `1dca0c21b` into one platform commit on top of pi
at `5ab7fe74c400097b386662205cd93900695b8a71`. pi's upstream history and MIT license
are retained; RoboAgentLab additions use Apache-2.0 as described in [NOTICE](NOTICE).

### Validation

The source snapshot's existing CI built the Node packages and passed the Python
services job. Its Node checks failed because provider-model test fixtures still
referenced `mimo-v2.5-pro` after live model discovery exposed `mimo-v2.6-pro`;
subsequent Node tests were skipped. This release does not claim a completed
cross-benchmark or real-robot evaluation.

The release preparation updates repository links and release documentation without
changing robot runtime code.
