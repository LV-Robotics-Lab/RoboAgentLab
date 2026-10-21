# Pi boundary and default regression

The extension targets the published pi. `peerDependencies` require pi 1.3.0 or a later 1.x
(`^1.3.0`): 1.3.0 is the version the compatibility check and the extension-loading check below ran
against, and the range is installation metadata, not a claim that every later 1.x release has passed
the same regression. `scripts/check-pi-compat.mjs` verifies an installed pi before `pi install`
(`docs/official-pi-migration.md` has the install steps and what changed when the fork stopped being
required).

The agent loop, model registry, messages, sessions, tools and extension events
belong to pi. Robot task state, motion gates, camera context and results belong
to the embodied extension. Python RPC services own simulation, perception and
bounded robot execution. New robotics features should use these boundaries
rather than a core patch: the extension imports only pi's public exports, and
the check fails when one goes missing. Pi 1.0 removed the experimental harness
and session-storage exports from `pi-agent-core`; durable sessions belong to
`pi-durable`. Embodied adapters use the retained `Agent` and coding-agent
extension APIs.

`infra/service-process.ts` owns Python process startup, readiness, shutdown,
attachment and environment construction. `defineRobot` owns episode lifecycle
and decides whether a service exit invalidates its episode. Existing robot
helper exports remain available to robot extensions.

The Codex and Claude Code plugins (`packages/embodied/integrations/`) and the
MCP server behind them (`src/integrations/mcp`) consume only the primitive
manifests, the RPC client, the deployment config and the operator's risk
classes; they register nothing in pi and change no robot or mode behaviour.
They serve tools mode only: units, code mode, VDM, memory, exploration, replay
and evaluation remain pi's, so the pi regression boundary above is unchanged.

## The compatibility check

```sh
node packages/embodied/scripts/check-pi-compat.mjs [--pi <dir>] [--strict] [--json]
```

It finds the pi to check (`--pi`: the pi-coding-agent package directory or an npm prefix; else the
`pi` on PATH, `npm root -g`, or the pi-coding-agent next to the package, which is the fork workspace),
compares its version with the peer range, then scans every `import ... from "@earendil-works/*"` and
`"typebox"` under `src/`: each module must resolve from that pi, each imported value must be an export
of it and each imported type must be declared in its `.d.ts`. A missing export fails the check.

Two kinds of module appear in its output:

- host-provided: pi-coding-agent, pi-agent-core, pi-tui, pi-ai and typebox, which pi's extension loader
  maps to its own copies for every extension (coding-agent `src/core/extensions/loader.ts`). These are
  checked against pi's own installation.
- not mapped by the loader: `@earendil-works/pi-durable` and `@earendil-works/chord` (the durable
  policy, which the MetaWorld robot loads only with `--policy-store`, and `src/scripts/eval-durable.ts`)
  and `@earendil-works/pi-mcp` (the MCP server, `src/integrations/mcp`). At runtime these come from a
  `node_modules` next to the package: the fork workspace, or `npm install --no-save` of the three at
  pi's version in `packages/embodied`. The robot extensions, the dashboard and `/embodied-setup` load
  without them. pi-durable and chord are optional peers; the check reports a missing one and skips it
  unless `--strict`.

CI runs the check against a fresh `npm install -g @earendil-works/pi-coding-agent@1.3.0`
(`.github/workflows/ci.yml`, "embodied against the official pi"), not against the workspace build.

## Official pi and the fork's context

The RoboAgentLab fork of pi adds `ExtensionContext.exportSession` and `withdrawQueuedMessage`. The
extension no longer needs either:

- The dashboard writes `GET /download/session` itself from pi's public session manager
  (`src/capabilities/dashboard/session-export.ts`): the JSONL pi's `/export` writes, or a plain HTML
  transcript. Its SSE snapshot reports `capabilities.sessionExport: true` on every pi.
- The dashboard still calls `withdrawQueuedMessage` when the context has it. pi 1.3.0 has no public way
  for an extension to take a message back out of the steering queue, so on an official pi the snapshot
  reports `capabilities.queueWithdrawal: false`, `POST /message/withdraw` answers HTTP 501 with an
  explanation, and the message stays queued until the agent takes it. Sending a steering message and
  calling an available robot tool work on both.

The fork's third change, the NInfer `expired while waiting for admission` retry pattern in pi-ai, is
dropped: pi offers no extension hook into its retry classifier and NInfer is no longer the selfhost
engine (`docs/official-pi-migration.md`).

Tests cover dashboard contexts with and without `withdrawQueuedMessage` (`test/dashboard-ops.test.ts`)
and the export's shape (`test/session-export.test.ts`). The checks run against an unmodified pi 1.3.0
so far: the compat check, and loading the package, the LIBERO and MetaWorld robots and the dashboard
through pi's own loader with `/embodied-setup` and the robot commands registered. A full session
lifecycle on an official installation (an episode with a robot extension and its services) remains the
fork-based regression below.

## Default workflow

The ordinary extension workflow uses the robot's native tools. Units, Python
code mode, fallback, ensemble, exploration, replay and privileged simulator
state are explicit modes. They should not silently change ordinary sessions.
Python code mode executes the robot service's bounded primitive API; pi's
QuickJS codemode serves a different execution environment.

The fixed Qwen comparison profile deliberately uses units mode to preserve the
existing six-cell diagnostic: MetaWorld reach-v3, ManiSkill PickCube-v1 and
LIBERO libero_spatial task 0, each at seeds 0 and 1. It fixes low thinking,
40 turns and 300 seconds, with a fresh local memory per cell. All cells run
sequentially. This diagnostic is too small to estimate general task ability.

Run from a clean, built checkout under the host's GPU reservation:

```sh
bash packages/embodied/src/scripts/fixed-regression.sh /path/to/new-results /path/to/env-files <provider>/<model>
```

The environment directory provides `metaworld-env.sh`, `maniskill-env.sh` and
`libero-env.sh`, with backend Python, rendering, GPU and asset locations. The
third argument is the planner model, as `pi --model` takes it; its credentials
remain in pi's existing registry. The runner selects this checkout's
CLI and services, rejects reused output directories, and records Git version
before and after. Build from that same clean version before running. Report
successes out of all six planned cells as well as valid cells; model/transport
errors and missing results are invalid, never silently dropped or retried.

Separately, `test/gpu-e2e.sh` covers 11 simulator adapters with fixed startup,
observation, one-action and result checks. These checks use a stub planner and
measure integration, not Qwen success. HumanCLAW WALK coverage does not certify
sitting on the intended seat. BEHAVIOR dataset incompatibility must remain a
failed check until resolved. Physical robot acceptance is separate and deferred.

Fallback-to-VirtualModels migration and further separation of episode budgets,
context pruning and robot tool registration require their own behavior tests;
they are not implied by this first process-lifecycle extraction.
