/**
 * GPU end-to-end suite: real simulators and model servers, no model API. Opt-in and skipped
 * everywhere else: it runs only with PI_EMBODIED_E2E=<robot>[,<robot>...] and an nvidia-smi that
 * answers. Each robot loads in a stub pi (flags as on the command line), starts its real env server
 * from PI_EMBODIED_PYTHON / PI_EMBODIED_SERVICES, and the test drives its tools the way the planner
 * would: one Show-Harness unit (`act MV_UP`), a status check that the env stepped, `finish`, and the
 * result entry. With SAM3_CHECKPOINT_PATH set, LIBERO also starts SAM3 itself (`--serve-models sam3`,
 * under PI_EMBODIED_GPU_LOCK) and segments its first frame. `test/gpu-e2e.sh` runs it robot by robot
 * with each one's environment (docs/adding-a-robot.md, "Testing on a GPU").
 *
 * `libero-mm` is MotorMind mode's interface checks on LIBERO (pi-embodied-work/docs/motormind/DESIGN.md
 * §8, C1-C18; C19 is M2's): the tools driven directly, privileged truth read beside the robot from the
 * env server's `env.mm_debug_*` RPCs, measured values written to `$MM_STEP1_OUT/report.json`.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { parseEndpoint, RpcClient } from "../src/infra/rpc.ts";
import { parseAction, segmentsFor, toDelta, turnBetween } from "../src/modes/motormind/command.ts";
import { apply, dot, norm, sub, transpose } from "../src/modes/motormind/frames.ts";
import { MOTORMIND_EVENT, type MotorMindHandle, type Pose } from "../src/modes/motormind/index.ts";
import { RESULT_ENTRY, SERVICES, STATUS_EVENT } from "../src/robot.ts";
import behavior from "../src/robots/behavior/index.ts";
import genesis from "../src/robots/genesis/index.ts";
import humanclaw from "../src/robots/humanclaw/index.ts";
import libero from "../src/robots/libero/index.ts";
import maniskill from "../src/robots/maniskill/index.ts";
import metaworld from "../src/robots/metaworld/index.ts";
import robocasa from "../src/robots/robocasa/index.ts";
import robodojo from "../src/robots/robodojo/index.ts";
import robolab from "../src/robots/robolab/index.ts";
import robosuite from "../src/robots/robosuite/index.ts";
import robotwin from "../src/robots/robotwin/index.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

const selected = (process.env.PI_EMBODIED_E2E ?? "")
	.split(",")
	.map((s) => s.trim())
	.filter(Boolean);
const gpu = (() => {
	if (!selected.length) return false;
	try {
		execFileSync("nvidia-smi", ["-L"], { stdio: "ignore" });
		return true;
	} catch {
		return false;
	}
})();
const skip = (robot: string) =>
	!selected.includes(robot)
		? `PI_EMBODIED_E2E does not name ${robot}`
		: !gpu
			? "no GPU (nvidia-smi)"
			: !process.env.PI_EMBODIED_PYTHON
				? "PI_EMBODIED_PYTHON is unset"
				: false;

/** One robot loaded like `pi -e <robot> <flags>`, with the handlers, tools, entries and status it publishes. */
function load(robot: (pi: ExtensionAPI) => void, values: Record<string, unknown>) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	const status: any[] = [];
	let active: string[] = [];
	let mm: MotorMindHandle | undefined;
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: {
			emit: (channel: string, data: any) => {
				if (channel === STATUS_EVENT) status.push(data);
				if (channel === MOTORMIND_EVENT) mm = data;
			},
			on: () => () => {},
		},
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const dir = mkdtempSync(join(tmpdir(), "gpu-e2e-"));
	const errors: string[] = [];
	const ctx = {
		hasUI: true,
		cwd: dir,
		ui: new Proxy({ notify: (msg: string) => errors.push(msg) }, { get: (t: any, k: string) => t[k] ?? (() => {}) }),
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionFile: () => join(dir, "session.jsonl"),
			getSessionDir: () => dir,
			getSessionId: () => "e2e",
		},
	};
	robot(pi);
	/** Run the handlers in order, as pi does; the last one that returned something answers. */
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	};
	const call = async (name: string, params: Record<string, unknown>) => {
		assert.ok(active.includes(name), `${name} is not active: ${active.join(", ")}`);
		return tools.get(name).execute(`e2e-${name}`, params, undefined, undefined, ctx);
	};
	return { emit, call, entries, status, errors, active: () => active, dir, mm: () => mm };
}

/** The services checkout this suite runs against. */
const servicesRoot = process.env.PI_EMBODIED_SERVICES ?? SERVICES;

/**
 * The common flags and deployment (a temporary config file, ./helpers/deployment.ts): the services
 * checkout, $PI_EMBODIED_PYTHON and an empty local memory corpus (no download).
 */
function flags(extra: Record<string, unknown>) {
	const dir = mkdtempSync(join(tmpdir(), "gpu-e2e-mem-"));
	writeFileSync(join(dir, "MEMORY.md"), "# GPU e2e\n");
	return deployFlags({
		python: process.env.PI_EMBODIED_PYTHON,
		services: servicesRoot,
		"memory-profile": "local",
		"memory-dir": dir,
		...(process.env.PI_EMBODIED_E2E_CUDA ? { "cuda-device": process.env.PI_EMBODIED_E2E_CUDA } : {}),
		...extra,
	});
}

const text = (r: any) => (r.content ?? []).map((c: any) => c.text ?? "").join("\n");

/** Start the robot, run one MV_UP unit, check that the env stepped, finish, and read the result entry. */
async function episode(robot: (pi: ExtensionAPI) => void, name: string, task: Record<string, unknown>) {
	const r = load(robot, flags({ units: "true", ...task }));
	try {
		await r.emit("session_start", { reason: "startup" });
		assert.deepEqual(r.errors, [], `${name} did not start`);
		assert.ok(r.active().includes("act"), r.active().join(", "));
		const before = r.status.at(-1)?.step ?? 0;
		await r.emit("before_agent_start", { prompt: "Solve the task.", systemPrompt: "" });
		await r.emit("agent_start");
		const acted = await r.call(
			"act",
			name === "humanclaw"
				? { unit: "WALK", param: "slow", target_visible: false }
				: { unit: "MV_UP", ...(name === "maniskill" ? {} : { arm: "left" }) },
		);
		assert.ok(!acted.isError, text(acted));
		await r.emit("tool_execution_end", { toolName: "act" });
		const after = r.status.at(-1);
		assert.equal(after.ready, true);
		assert.ok((after.step ?? 0) > before, `the env did not step: ${before} -> ${after.step}`);
		await r.call("finish", { status: "failure", summary: "e2e: one unit" });
		await r.emit("message_end", {
			message: {
				role: "assistant",
				stopReason: "toolUse",
				content: [{ type: "toolCall", name: "finish" }],
				usage: { cost: { total: 0 } },
			},
		});
		await r.emit("agent_end", { messages: [] });
		const result = r.entries.find((e) => e.type === RESULT_ENTRY)?.data;
		assert.ok(result, "no robot_result entry");
		assert.equal(result.robot, name);
		assert.equal(result.env_error, false, JSON.stringify(result));
		assert.equal(result.claimed, "failure");
		return result;
	} finally {
		await r.emit("session_shutdown", { reason: "quit" });
	}
}

test("metaworld: reach-v3 starts, steps a unit and reports", { skip: skip("metaworld") }, async () => {
	await episode(metaworld, "metaworld", { task: "reach-v3", seed: "0" });
});

test("robosuite: Lift starts, steps a unit and reports", { skip: skip("robosuite") }, async () => {
	await episode(robosuite, "robosuite", { task: "Lift", seed: "0" });
});

test("genesis: cube_pick starts, steps a unit and reports", { skip: skip("genesis") }, async () => {
	await episode(genesis, "genesis", { task: "cube_pick", seed: "0" });
});

test("maniskill: PickCube-v1 starts, steps a unit and reports", { skip: skip("maniskill") }, async () => {
	await episode(maniskill, "maniskill", { task: "PickCube-v1", seed: "0" });
});

test("libero: libero_10 task 0 starts, steps a unit and reports", { skip: skip("libero") }, async () => {
	await episode(libero, "libero", { suite: "libero_10", task: "0", seed: "0" });
});

/** Fixed integration cases; these check execution, not model task success. */
const INTEGRATION_CASES = [
	[robocasa, "robocasa", { task: "OpenDrawer", split: "target", seed: "1" }],
	[robotwin, "robotwin", { task: "beat_block_hammer", "task-config": "demo_randomized", seed: "100000" }],
	[robolab, "robolab", { task: "BananaInBowlTask", seed: "0" }],
	[robodojo, "robodojo", { task: "stack_bowls", seed: "0" }],
	[behavior, "behavior", { task: "turning_on_radio", seed: "0" }],
	[
		humanclaw,
		"humanclaw",
		{ episode: "104348028_171512877_ep1_couch", units: "both", "humanclaw-mode": "pi", "humanclaw-metrics": true },
	],
] as const;
for (const [robot, name, task] of INTEGRATION_CASES)
	test(`${name}: starts, steps a unit and reports`, { skip: skip(name), timeout: 780000 }, async () => {
		console.log(JSON.stringify(await episode(robot, name, task)));
	});

async function freePort(): Promise<number> {
	const s = createServer();
	await new Promise<void>((r) => s.listen(0, "127.0.0.1", r));
	const port = (s.address() as { port: number }).port;
	await new Promise((r) => s.close(r));
	return port;
}

test(
	"libero --serve-models sam3: pi starts SAM3 under the GPU lock, segments with it, and stops it",
	{
		skip: skip("libero") || (!process.env.SAM3_CHECKPOINT_PATH && "SAM3_CHECKPOINT_PATH is unset"),
		timeout: 1_800_000,
	},
	async () => {
		const port = await freePort();
		const sam3 = `http://127.0.0.1:${port}`;
		const r = load(
			libero,
			flags({
				suite: "libero_10",
				task: "0",
				seed: "0",
				units: "false",
				sam3,
				"serve-models": "sam3",
				"serve-min-free": process.env.PI_EMBODIED_E2E_MIN_FREE ?? "6000",
				"serve-timeout": "1500",
			}),
		);
		try {
			await r.emit("session_start", { reason: "startup" });
			assert.deepEqual(r.errors, [], "libero did not start");
			await new RpcClient(sam3).call("healthz", {}, 5_000);
			const seg = await r.call("segment", { prompt: "the black bowl" });
			assert.ok(!seg.isError, text(seg));
		} finally {
			await r.emit("session_shutdown", { reason: "quit" });
		}
		await assert.rejects(new RpcClient(sam3).call("healthz", {}, 3_000));
	},
);

/** ManiSkill's newer robots, each on one of its own scenes; the pair moves its left arm. */
const FLYWHEEL_ROBOTS = [
	{ robot: "panda_stick", envId: "PushT-v1", move: { delta_xyz: [0, 0, 0.02] }, action: 3, state: 8 },
	{
		robot: "panda_pair",
		envId: "TwoRobotPickCube-v1",
		move: { arm: "left", delta_xyz: [0, 0, 0.02] },
		action: 8,
		state: 16,
	},
	{ robot: "widowx250s", envId: "PutCarrotOnPlateInScene-v1", move: { delta_xyz: [0, 0, 0.02] }, action: 7, state: 8 },
];

for (const c of FLYWHEEL_ROBOTS)
	test(
		`maniskill --arm ${c.robot}: a Flywheel episode records every control step and validates in its space`,
		{ skip: skip("maniskill-flywheel") },
		async () => {
			const root = mkdtempSync(join(tmpdir(), "gpu-e2e-fly-"));
			const r = load(
				maniskill,
				flags({
					arm: c.robot,
					task: c.envId,
					seed: "0",
					"collect-flywheel-data": true,
					"flywheel-root": root,
				}),
			);
			try {
				await r.emit("session_start", { reason: "startup" });
				assert.deepEqual(r.errors, [], `${c.robot} did not start`);
				for (let k = 0; k < 2; k++) {
					await r.emit("tool_execution_start", { toolName: "move_delta" });
					const moved = await r.call("move_delta", c.move);
					assert.ok(!moved.isError, text(moved));
					await r.emit("tool_execution_end", { toolName: "move_delta" });
				}
			} finally {
				await r.emit("session_shutdown", { reason: "quit" });
			}
			const written = r.entries.find((e) => e.type === "flywheel_episode")?.data;
			assert.ok(written?.path, `no flywheel_episode entry: ${JSON.stringify(r.entries.map((e) => e.type))}`);
			assert.ok(written.path.includes(`/raw/maniskill/${c.robot}/${c.envId}/`), written.path);
			assert.ok(written.step_count >= 2, JSON.stringify(written));
			// The services' own validator, in the robot's space: shapes, names, the arm pin.
			const out = execFileSync(
				String(process.env.PI_EMBODIED_PYTHON),
				[
					"-m",
					"pi_embodied_services.flywheel.cli",
					"validate",
					written.path,
					"--robot",
					"maniskill",
					"--space",
					c.robot,
				],
				{ cwd: servicesRoot, env: { ...process.env, PYTHONPATH: servicesRoot }, encoding: "utf8" },
			);
			const meta = JSON.parse(out);
			assert.equal(meta.maniskill_robot, c.robot);
			const shapes = execFileSync(
				String(process.env.PI_EMBODIED_PYTHON),
				[
					"-c",
					"import json, sys, numpy as np; t = np.load(sys.argv[1] + '/transitions.npz'); print(json.dumps({k: list(t[k].shape) for k in ('actions', 'states')} | {'actions_rows': t['actions'].tolist(), 'states_rows': t['states'].tolist()}))",
					written.path,
				],
				{ encoding: "utf8" },
			);
			const s = JSON.parse(shapes);
			assert.equal(s.actions[1], c.action, JSON.stringify(s));
			assert.equal(s.states[1], c.state, JSON.stringify(s));
			if (c.robot === "panda_pair") {
				// Left then right: the moved left arm's actions are in dims 0-3 and its TCP rose; the
				// right arm held still with its own open command (+1).
				const [first, last] = [s.states_rows[0], s.states_rows.at(-1)];
				assert.ok(last[2] - first[2] > 0.01, `left z ${first[2]} -> ${last[2]}`);
				assert.ok(Math.abs(last[10] - first[10]) < 0.005, `right z ${first[10]} -> ${last[10]}`);
				for (const a of s.actions_rows) assert.deepEqual(a.slice(4), [0, 0, 0, 1]);
			}
			console.log(
				`${c.robot}: ${JSON.stringify({ steps: written.step_count, actions: s.actions, states: s.states, first_actions: s.actions_rows.slice(0, 2) })}`,
			);
		},
	);

// ---------------------------------------------------------------------------
// libero-mm: MotorMind mode's interface checks (DESIGN §8.2). M checks assert with the design's
// tolerances; R checks only record. Everything measured goes to $MM_STEP1_OUT/report.json.

const MM = "libero-mm";
const MM_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] as const;
const mmOut =
	process.env.MM_STEP1_OUT ??
	join(servicesRoot, "..", ".local", "runs", "mm-step1", new Date().toISOString().slice(0, 10));
const mmReport: Record<string, unknown> = {};
/** Record under `key` (merged into report.json at once, so a later failure keeps what was measured). */
function mmRecord(key: string, value: unknown) {
	const file = join(mmOut, "report.json");
	if (!Object.keys(mmReport).length && existsSync(file))
		Object.assign(mmReport, JSON.parse(readFileSync(file, "utf8")));
	mmReport[key] = value;
	mkdirSync(mmOut, { recursive: true });
	writeFileSync(file, `${JSON.stringify(mmReport, null, 2)}\n`);
}

/** The flags of a MotorMind run: privileged (the env server's debug RPCs), a short segment timeout (C15). */
function mmFlags(extra: Record<string, unknown>) {
	return deployFlags({
		python: process.env.PI_EMBODIED_PYTHON,
		services: servicesRoot,
		...(process.env.PI_EMBODIED_E2E_CUDA ? { "cuda-device": process.env.PI_EMBODIED_E2E_CUDA } : {}),
		"output-dir": mkdtempSync(join(tmpdir(), "gpu-e2e-mm-")),
		privileged: true,
		"mm-step-timeout": "10",
		...extra,
	});
}

const deg = (rad: number) => (rad * 180) / Math.PI;
const mm = (m: number) => m * 1000;
type Json = Record<string, any>;

/** LIBERO in MotorMind mode on its real env server, with a truth client beside it. */
async function mmSession(extra: Record<string, unknown>) {
	const r = load(libero, mmFlags(extra));
	await r.emit("session_start", { reason: "startup" });
	assert.deepEqual(r.errors, [], "libero (MotorMind mode) did not start");
	const h = r.mm();
	assert.ok(h, "the mode published no handle");
	const { url, token } = parseEndpoint(h.endpoint() ?? "");
	const truth = new RpcClient(url);
	if (token) truth.token = token;
	await r.emit("before_agent_start", { prompt: "Solve the task.", systemPrompt: "" });
	await r.emit("agent_start");
	const latest = () => h.latest() as NonNullable<ReturnType<MotorMindHandle["latest"]>>;
	/** One mm_act; a refusal throws. The body is the tool result's JSON. */
	const act = async (params: Json): Promise<Json> => {
		const res = await r.call("mm_act", params);
		await r.emit("tool_execution_end", { toolName: "mm_act" });
		return JSON.parse(res.content[0].text);
	};
	const observe = async (): Promise<Json> => JSON.parse((await r.call("mm_observe", {})).content[0].text);
	const pose = (): Pose => latest().pose as Pose;
	/** The latest observation's reply as the server sent it (cameras, static fields). */
	const raw = (): Json => latest().raw as Json;
	const base = (): Pose => latest().base_pose as Pose;
	const truthNow = (state = false) => truth.call<Json>("env.mm_debug_truth", { state }, 120_000);
	const project = (o: Json) => truth.call<Json>("env.mm_debug_project", o, 120_000);
	/** Move the tool by `d` (m, base frame) with one axis move. */
	const moveBy = async (d: number[], extra: Json = {}) => {
		const n = norm(d);
		if (n < 1e-4) return undefined;
		return act({ type: "move", axis: d.map((v) => v / n), distance_mm: mm(n), ...extra });
	};
	const finish = async () => {
		await r.call("finish", { status: "failure", summary: "libero-mm interface checks" });
		await r.emit("message_end", {
			message: { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", name: "finish" }] },
		});
		await r.emit("agent_end", { messages: [] });
		return r.entries.find((e) => e.type === RESULT_ENTRY)?.data as Json;
	};
	return { ...r, h, truth, latest, raw, act, observe, pose, base, truthNow, project, moveBy, finish };
}
type MmSession = Awaited<ReturnType<typeof mmSession>>;

const along = (b: Json) => b.segments.reduce((s: number, x: Json) => s + x.along_axis_mm, 0);
const turned = (b: Json) => b.segments.reduce((s: number, x: Json) => s + x.turned_deg, 0);
const outcomes = (b: Json) => b.segments.map((x: Json) => x.outcome);

/** A base-frame point's pixel (u = column, v = row) in a camera with intrinsic `k` and 4x4 `cam2base` (OpenCV). */
function projectTo(k: number[][], c2b: number[][], p: number[]): [number, number] | undefined {
	const r = [0, 1, 2].map((i) => [0, 1, 2].map((j) => c2b[i][j]));
	const t = [0, 1, 2].map((i) => c2b[i][3]);
	const c = apply(transpose(r as never), sub(p, t));
	if (c[2] <= 1e-6) return undefined;
	return [(k[0][0] * c[0]) / c[2] + k[0][2], (k[1][1] * c[1]) / c[2] + k[1][2]];
}

/** The world displacement of the tool and the base-frame one, compared through R_b (C1, C2). */
function worldErr(s: MmSession, t0: Json, t1: Json, p0: Pose, p1: Pose) {
	const dw = sub(t1.eef_pos_world, t0.eef_pos_world);
	const db = sub(p1.position_m, p0.position_m);
	return norm(sub(dw, apply(s.base().rotation, db)));
}

/** C10: the side camera where the mount rule puts it (aim at the base's +0.47 m, 35 deg elevation on +Y). */
async function checkSideMount(s: MmSession, label: string) {
	const t = await s.truthNow();
	const m = t.side_mount;
	assert.ok(m?.mounted, `${label}: side camera not mounted: ${JSON.stringify(m)}`);
	assert.equal(m.parent, "world");
	const b = s.base().position_m;
	const aim = [b[0] + 0.47, b[1], m.table_top.top_z];
	const scene = (await s.project({ cameras: ["scene"] })).scene;
	const agent = [0, 1, 2].map((i) => scene.cam2world[i][3]);
	const d = norm(sub(agent, aim));
	const el = (35 * Math.PI) / 180;
	const eye = [aim[0], aim[1] + d * Math.cos(el), aim[2] + d * Math.sin(el)];
	const posErr = norm(sub(m.pos, eye));
	assert.ok(posErr <= 1e-6, `${label}: side camera at ${m.pos}, rule ${eye} (${posErr} m)`);
	const agentFovy = deg(2 * Math.atan(384 / scene.intrinsic[0][0]));
	assert.ok(Math.abs(m.fovy - agentFovy) <= 1e-6, `${label}: side fovy ${m.fovy} vs agentview ${agentFovy}`);
	await s.observe();
	const side = s.raw().cameras.side;
	const aimBase = apply(transpose(s.base().rotation), sub(aim, b));
	const uv = projectTo(side.intrinsic, side.cam2base, aimBase) as [number, number];
	assert.ok(Math.hypot(uv[0] - 384, uv[1] - 384) <= 1, `${label}: the aim point projects to ${uv}, not the center`);
	return { pos_err_m: posErr, fovy: m.fovy, aim_uv: uv, standoff_m: d, pre_mount: m.pre ?? null };
}

/** The suites' t0: frames, directions, turns, views, side mount, table (C1-C3, C9-C11) and the tool set (C17). */
for (const suite of MM_SUITES)
	test(
		`libero-mm ${suite} t0 s0: base frame, directions, turns, views, side mount, table (C1-C3, C9-C11, C17)`,
		{ skip: skip(MM), timeout: 1_800_000 },
		async () => {
			const s = await mmSession({ suite, task: "0", seed: "0", preset: "motormind" });
			const rec: Json = {};
			try {
				// C17: pure mode is the two mm tools and finish; no file tools.
				assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);

				// C1: the base frame, R_b measured; the pose is R_bᵀ(p_world - p_b).
				await s.observe();
				const t = await s.truthNow();
				const b = s.base();
				const expect = apply(transpose(b.rotation), sub(t.eef_pos_world, b.position_m));
				const c1 = norm(sub(expect, s.pose().position_m));
				assert.ok(c1 <= 1e-6, `C1: pose differs from R_bᵀ(p - p_b) by ${c1} m`);
				rec.C1 = {
					R_b: b.rotation,
					p_b: b.position_m,
					body: s.raw().base_pose?.body,
					pose_err_m: c1,
					osc_ref_frame: s.raw().osc_ref_frame,
				};

				// C2: the six words: along, lateral, the world motion through R_b, the agentview pixel sense.
				await s.act({ type: "move", direction: "up", distance_mm: 100 });
				rec.C2 = {};
				const centroid = async () =>
					(await s.project({ bodies: ["gripper0_"], cameras: ["scene"] })).scene.bodies.gripper0_.centroid_uv;
				for (const [word, want] of [
					["forward", 50],
					["backward", 50],
					["left", 50],
					["right", 50],
					["up", 30],
					["down", 30],
				] as const) {
					const vertical = word === "up" || word === "down";
					const [t0, p0] = [await s.truthNow(), s.pose()];
					const u0 = vertical ? undefined : await centroid();
					const r = await s.act({ type: "move", direction: word, distance_mm: want, speed_mm_s: 50 });
					const [t1, p1] = [await s.truthNow(), s.pose()];
					assert.deepEqual(
						outcomes(r),
						r.segments.map(() => "done"),
						`C2 ${word}: ${JSON.stringify(r.segments)}`,
					);
					const a = along(r);
					const lateral = Math.max(...r.segments.map((x: Json) => x.lateral_mm));
					const w = mm(worldErr(s, t0, t1, p0, p1));
					assert.ok(Math.abs(a - want) <= 3, `C2 ${word}: along ${a} mm of ${want}`);
					assert.ok(lateral <= (vertical ? 5 : 4), `C2 ${word}: lateral ${lateral} mm`);
					assert.ok(w <= 0.5, `C2 ${word}: world motion differs from R_b Δ by ${w} mm`);
					const u1 = vertical ? undefined : await centroid();
					if (word === "left")
						assert.ok(u1[0] > u0[0], `C2 left: the gripper's agentview u went ${u0[0]} -> ${u1[0]}`);
					if (word === "right")
						assert.ok(u1[0] < u0[0], `C2 right: the gripper's agentview u went ${u0[0]} -> ${u1[0]}`);
					rec.C2[word] = {
						along_mm: a,
						lateral_mm: lateral,
						lateral_under_2mm: lateral <= 2,
						world_err_mm: w,
						du: u0 && u1 ? u1[0] - u0[0] : null,
					};
				}

				// C3: the turn words about their signed base axes, the tool point fixed; yaw is M, pitch and roll R.
				rec.C3 = {};
				const back: Record<string, string> = {
					yaw_left: "yaw_right",
					yaw_right: "yaw_left",
					clockwise: "ccw",
					ccw: "clockwise",
					pitch_up: "pitch_down",
					pitch_down: "pitch_up",
					roll_ccw: "roll_cw",
					roll_cw: "roll_ccw",
				};
				const turns = [
					["yaw_left", 30, [0, 0, 1]],
					["yaw_right", 30, [0, 0, -1]],
					["clockwise", 30, [0, 0, -1]],
					["ccw", 30, [0, 0, 1]],
					["pitch_up", 15, [0, -1, 0]],
					["pitch_down", 15, [0, 1, 0]],
					["roll_ccw", 15, [1, 0, 0]],
					["roll_cw", 15, [-1, 0, 0]],
				] as const;
				for (const [word, angle, axis] of turns) {
					const p0 = s.pose();
					const r = await s.act({ type: "rotate", direction: word, angle_deg: angle });
					const p1 = s.pose();
					const tb = turnBetween(p0, p1);
					const axisErr = deg(Math.acos(Math.max(-1, Math.min(1, dot(tb.axis, axis)))));
					const drift = mm(norm(sub(p1.position_m, p0.position_m)));
					const got = turned(r);
					const world = apply(s.base().rotation, tb.axis);
					rec.C3[word] = {
						turned_deg: got,
						measured_deg: deg(tb.angle),
						axis_err_deg: axisErr,
						drift_mm: drift,
						world_axis: world,
						outcomes: outcomes(r),
					};
					if (word.startsWith("yaw") || word === "clockwise" || word === "ccw") {
						assert.ok(Math.abs(got - angle) <= 1.7, `C3 ${word}: turned ${got} deg of ${angle}`);
						assert.ok(axisErr <= 3, `C3 ${word}: axis ${tb.axis} is ${axisErr} deg off ${axis}`);
						assert.ok(drift <= 3, `C3 ${word}: the tool point drifted ${drift} mm`);
					}
					// Show-Harness's ROTATE_CW turns about base +Z: MotorMind's clockwise is the opposite turn.
					if (word === "clockwise") assert.ok(world[2] < 0, `C3: clockwise turns about ${world}, not -Z`);
					// pi's rotate_pitch grows about world +x: roll_ccw turns the same way (R).
					if (word === "roll_ccw") rec.C3.roll_ccw_matches_rotate_pitch = world[0] > 0;
					await s.act({ type: "rotate", direction: back[word], angle_deg: angle });
				}

				// C9: three 768 views; K by the formula; cam2base a rotation; segmentation under our projection.
				// Back to the starting height first (C2 began 100 mm up): there libero_10's side view, mounted
				// closer by the rule, has the gripper above its frame.
				await s.act({ type: "move", direction: "down", distance_mm: 100 });
				await s.observe();
				const cams = s.raw().cameras;
				const truthObj = (await s.truthNow()).objects as Record<string, Json>;
				const names = Object.keys(truthObj);
				const dbg = await s.project({
					bodies: ["gripper0_", ...names],
					points_base_m: names.map((n) => truthObj[n].pos_base),
					// Pixel centres of a 5 x 5 grid over the wrist view.
					depth_pixels: {
						wrist: [96.5, 240.5, 384.5, 528.5, 672.5].flatMap((u) =>
							[96.5, 240.5, 384.5, 528.5, 672.5].map((v) => [u, v]),
						),
					},
				});
				rec.C9 = {};
				for (const cam of ["scene", "side", "wrist"]) {
					const c = cams[cam];
					assert.deepEqual([c.width, c.height], [768, 768], `C9 ${cam}`);
					const k = c.intrinsic;
					assert.ok(
						k[0][0] > 0 && Math.abs(k[0][0] - k[1][1]) < 1e-9 && k[0][1] === 0,
						`C9 ${cam} K ${JSON.stringify(k)}`,
					);
					assert.deepEqual([k[0][2], k[1][2]], [384, 384], `C9 ${cam}: principal point`);
					for (let i = 0; i < 3; i++)
						for (let j = 0; j < 3; j++)
							assert.ok(
								Math.abs(k[i][j] - dbg[cam].intrinsic[i][j]) < 1e-9,
								`C9 ${cam}: K vs the debug render's`,
							);
					const R = [0, 1, 2].map((i) => [0, 1, 2].map((j) => c.cam2base[i][j]));
					const rtr = [0, 1, 2].map((i) =>
						[0, 1, 2].map((j) => R[0][i] * R[0][j] + R[1][i] * R[1][j] + R[2][i] * R[2][j]),
					);
					for (let i = 0; i < 3; i++)
						for (let j = 0; j < 3; j++)
							assert.ok(Math.abs(rtr[i][j] - (i === j ? 1 : 0)) < 1e-6, `C9 ${cam}: cam2base not orthonormal`);
					const det =
						R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
						R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
						R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
					assert.ok(Math.abs(det - 1) < 1e-6, `C9 ${cam}: det ${det}`);
					// Our projection of each object center lands where the server's segmentation render looks.
					const seen: Json[] = [];
					names.forEach((n, i) => {
						const ours = projectTo(k, c.cam2base, truthObj[n].pos_base);
						const theirs = dbg[cam].points[i];
						if (ours && theirs.uv)
							assert.ok(
								Math.hypot(ours[0] - theirs.uv[0], ours[1] - theirs.uv[1]) <= 1,
								`C9 ${cam} ${n}: ${ours} vs ${theirs.uv}`,
							);
						seen.push({
							object: n,
							uv: ours ?? null,
							body: theirs.body,
							own: typeof theirs.body === "string" && theirs.body.startsWith(n),
						});
					});
					// The tool point itself is a site in the segmentation (no body): the gripper is judged by its
					// bodies' pixels, and by where our projection of the tool point falls from their centroid.
					const gripper = dbg[cam].bodies.gripper0_;
					const tool = projectTo(k, c.cam2base, s.pose().position_m);
					if (cam !== "wrist") {
						assert.ok(
							seen.some((x) => x.own),
							`C9 ${cam}: no object center projects onto its own body: ${JSON.stringify(seen)}`,
						);
						assert.ok(gripper?.pixels > 0, `C9 ${cam}: no gripper pixels: ${JSON.stringify(gripper)}`);
					}
					rec.C9[cam] = {
						K: k,
						objects: seen,
						gripper: gripper ?? null,
						tool_uv: tool ?? null,
						tool_to_gripper_centroid_px:
							tool && gripper?.centroid_uv
								? Math.hypot(tool[0] - gripper.centroid_uv[0], tool[1] - gripper.centroid_uv[1])
								: null,
					};
				}
				// The server's 16-bit mm wrist depth against a ray cast through the same pixel centre, among the
				// geoms the camera draws (the visual ones: robosuite does not render collision geoms).
				const depth = dbg.wrist.depth.filter((d: Json) => d.ray_z_m !== null && d.server_z_m > 0);
				assert.ok(depth.length >= 20, `C9: only ${depth.length} wrist depth pixels hit something`);
				for (const d of depth)
					assert.ok(
						Math.abs(d.server_z_m - d.ray_z_m) <= 0.002,
						`C9 wrist depth ${d.server_z_m} vs ray ${d.ray_z_m} at ${d.uv} (${d.ray_body})`,
					);
				rec.C9.wrist_depth = {
					max_err_mm: Math.max(...depth.map((d: Json) => mm(Math.abs(d.server_z_m - d.ray_z_m)))),
					pixels: depth,
				};
				// The gripper's pixel moves the way our projection says, for +-x and +-y.
				rec.C9.moves = {};
				for (const word of ["forward", "backward", "left", "right"]) {
					const look = async () => {
						await s.observe();
						const c = s.raw().cameras.scene;
						return {
							ours: projectTo(c.intrinsic, c.cam2base, s.pose().position_m) as [number, number],
							seg: (await s.project({ bodies: ["gripper0_"], cameras: ["scene"] })).scene.bodies.gripper0_
								.centroid_uv,
						};
					};
					const a = await look();
					await s.act({ type: "move", direction: word, distance_mm: 30 });
					const z = await look();
					const d = [0, 1].map((i) => [z.ours[i] - a.ours[i], z.seg[i] - a.seg[i]]);
					for (const [ours, seg] of d)
						if (Math.abs(ours) > 1 && Math.abs(seg) > 1)
							assert.equal(
								Math.sign(ours),
								Math.sign(seg),
								`C9 ${word}: projected ${ours} px vs segmentation ${seg} px`,
							);
					rec.C9.moves[word] = d;
				}

				// C10: the side camera after the first reset (the second: the --mm-home reset test).
				rec.C10 = await checkSideMount(s, "C10 first reset");

				// C11: the table from scene depth within 5 mm of the truth, the visible top (rays among the drawn
				// geoms onto the table body); the collision top beside it is recorded (11 mm higher in the living
				// room). A forced failure degrades, never refuses.
				const tbl = await s.truth.call<Json>("env.mm_debug_table", {}, 120_000);
				assert.equal(
					typeof tbl.truth.ray_z_base_m,
					"number",
					`C11: no ray met the table: ${JSON.stringify(tbl.truth)}`,
				);
				const err = mm(Math.abs(tbl.truth.ray_z_base_m - tbl.table_z_m));
				assert.ok(err <= 5, `C11: table ${tbl.table_z_m} vs truth ${JSON.stringify(tbl.truth)}`);
				const failed = await s.truth.call<Json>("env.mm_debug_table", { fail: true }, 120_000);
				assert.equal(failed.table_z_m, null);
				assert.equal(failed.degraded, true);
				const o = await s.observe();
				assert.equal(o.degraded?.table_z, true, "C11: the observation does not say the table is degraded");
				assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);
				await s.truth.call("env.mm_debug_table", { fail: false }, 120_000);
				rec.C11 = {
					table_z_m: tbl.table_z_m,
					visible_top_m: tbl.truth.ray_z_base_m,
					err_mm: err,
					collision_top_m: tbl.truth.geom_top_z_base_m,
					collision_above_visible_mm:
						typeof tbl.truth.geom_top_z_base_m === "number"
							? mm(tbl.truth.geom_top_z_base_m - tbl.truth.ray_z_base_m)
							: null,
					n_points: tbl.n_points,
					truth: tbl.truth,
				};
			} finally {
				mmRecord(`${suite}_t0_s0`, rec);
				await s.emit("session_shutdown", { reason: "quit" });
			}
		},
	);

/**
 * A base-frame xy over bare table, nearest (0.5, 0): at least 13 cm from every object's centre, rays
 * straight down within 3 cm along x and 5 cm along y (the fingers) meet the table body at its
 * visible top, and within 5 cm along x and 12 cm along y (the hand above them, wide across its
 * fingers) nothing stands 4 cm above it.
 */
async function clearSpot(s: MmSession): Promise<[number, number]> {
	const objects = Object.values((await s.truthNow()).objects as Record<string, Json>).map((o) => o.pos_base);
	const cands: [number, number][] = [];
	for (let x = 0.35; x <= 0.6501; x += 0.025)
		for (let y = -0.3; y <= 0.3001; y += 0.025)
			if (objects.every((o: number[]) => Math.hypot(o[0] - x, o[1] - y) >= 0.13)) cands.push([x, y]);
	cands.sort((a, b) => Math.hypot(a[0] - 0.5, a[1]) - Math.hypot(b[0] - 0.5, b[1]));
	assert.ok(cands.length, "C6: no table 13 cm from every object");
	const grid = (c: [number, number], xs: number[], ys: number[]) =>
		xs.flatMap((dx) => ys.map((dy) => [c[0] + dx, c[1] + dy]));
	const fingers = (c: [number, number]) => grid(c, [-0.03, 0, 0.03], [-0.05, -0.025, 0, 0.025, 0.05]);
	const hand = (c: [number, number]) =>
		grid(
			c,
			[-0.05, -0.025, 0, 0.025, 0.05],
			Array.from({ length: 13 }, (_, i) => -0.12 + 0.02 * i),
		);
	const probe = (c: [number, number]) => [...fingers(c), ...hand(c)];
	const t = await s.truth.call<Json>("env.mm_debug_table", { xy_base_m: cands.flatMap(probe) }, 120_000);
	const top = t.truth.ray_z_base_m as number;
	const [nf, n] = [fingers(cands[0]).length, probe(cands[0]).length];
	const k = cands.findIndex((_, i) => {
		const rays = t.truth.rays.slice(i * n, (i + 1) * n) as Json[];
		return (
			rays.slice(0, nf).every((r) => r.body === t.truth.body && Math.abs(r.z_base_m - top) < 0.004) &&
			rays.slice(nf).every((r) => r.z_base_m === null || r.z_base_m < top + 0.04)
		);
	});
	assert.ok(k >= 0, "C6: no bare table within reach");
	return cands[k];
}

/** Put the tool `height` m above the measured table (base frame), straight down. */
/**
 * C7: MotorMind's skid judge (services' motormind_rig.replay_segment) re-run on a segment's recorded
 * per-tick poses (`env.mm_run_step` with `trace`), in the robot's Python.
 */
function replaySegment(trace: Json): Json {
	const script = [
		"import json, sys",
		"import numpy as np",
		"from pi_embodied_services.robots.libero import motormind_rig as rig",
		"t = json.load(sys.stdin)",
		"s = t['segment']",
		"a = None if s['axis'] is None else np.array(s['axis'])",
		"seg = rig.Segment(s['kind'], np.array(s['goal_position_m']), np.array(s['goal_rotation']), s['speed'], a, s['distance_m'], s['push'], s['holding'])",
		"poses = [(np.array(q['position_m']), np.array(q['rotation'])) for q in t['poses']]",
		"end = (np.array(t['end']['position_m']), np.array(t['end']['rotation']))",
		"out = rig.replay_segment(seg, poses, end)",
		"print(json.dumps({k: out[k] for k in ('outcome', 'abort_reason', 'along_axis_m', 'lateral_m', 'abort_tick', 'crept')}))",
	].join("\n");
	const out = execFileSync(String(process.env.PI_EMBODIED_PYTHON), ["-c", script], {
		cwd: servicesRoot,
		env: { ...process.env, PYTHONPATH: servicesRoot },
		input: JSON.stringify(trace),
		encoding: "utf8",
	});
	return JSON.parse(out);
}

async function toHeight(s: MmSession, height: number) {
	await s.observe();
	const table = s.latest().table_z_m;
	assert.ok(table !== null, "no table height measured");
	await s.moveBy([0, 0, (table as number) + height - s.pose().position_m[2]]);
}

for (const suite of MM_SUITES)
	test(
		`libero-mm ${suite} t0 s0: segments, contact, gripper, home refused, cancel, closed loop (C4, C6, C8, C12, C14, C15)`,
		{ skip: skip(MM), timeout: 1_800_000 },
		async () => {
			const s = await mmSession({ suite, task: "0", seed: "0", preset: "motormind" });
			const rec: Json = {};
			try {
				// C12 (refuse): nothing changes, not the scene, the episode, the ticks or the latch.
				const before = await s.truthNow();
				await assert.rejects(s.act({ type: "home" }), /this robot has no home/);
				const after = await s.truthNow();
				assert.deepEqual(
					[after.episode_id, after.ticks, after.success_latched, after.frame_clock_s],
					[before.episode_id, before.ticks, before.success_latched, before.frame_clock_s],
				);
				for (const [n, o] of Object.entries(before.objects as Record<string, Json>))
					assert.ok(norm(sub(after.objects[n].pos, o.pos)) < 1e-9, `C12: ${n} moved`);

				// C4: 120 mm, 60 deg and 1 rad run as 3, 3 and 2 equal segments.
				await s.act({ type: "move", direction: "up", distance_mm: 80 });
				const c4: Json = {};
				const fwd = await s.act({ type: "move", direction: "backward", distance_mm: 120 });
				c4.move_120mm = fwd.segments.map((x: Json) => x.along_axis_mm);
				assert.equal(fwd.segments.length, 3);
				await s.act({ type: "move", direction: "forward", distance_mm: 120 });
				const yaw = await s.act({ type: "rotate", direction: "yaw_left", angle_deg: 60 });
				c4.yaw_60deg = yaw.segments.map((x: Json) => x.turned_deg);
				assert.equal(yaw.segments.length, 3);
				await s.act({ type: "rotate", direction: "yaw_right", angle_deg: 60 });
				const rad = await s.act({ type: "rotate", direction: "yaw_left", angle_deg: 180 / Math.PI });
				c4.yaw_1rad = rad.segments.map((x: Json) => x.turned_deg);
				assert.equal(rad.segments.length, 2);
				await s.act({ type: "rotate", direction: "yaw_right", angle_deg: 180 / Math.PI });
				rec.C4 = c4;

				// C6: down 200 from 120 mm above the table ends blocked (limited, 2 hold ticks); a push ends in contact.
				// Over bare table: in libero_goal the hand lands on an object's edge at the default spot.
				const spot = await clearSpot(s);
				await toHeight(s, 0.12);
				await s.moveBy([spot[0] - s.pose().position_m[0], spot[1] - s.pose().position_m[1], 0]);
				const down = await s.act({ type: "move", direction: "down", distance_mm: 200 });
				const stop = down.segments.find((x: Json) => x.outcome !== "done");
				assert.ok(stop, `C6: down 200 never stopped: ${JSON.stringify(down.segments)}`);
				assert.deepEqual([stop.outcome, stop.abort_reason], ["limited", "blocked"], JSON.stringify(stop));
				assert.ok(stop.lateral_mm <= 5, `C6: lateral ${stop.lateral_mm} mm`);
				assert.equal(stop.hold_ticks, 2, JSON.stringify(stop));
				const rest = down.segments.slice(down.segments.indexOf(stop) + 1);
				assert.ok(
					rest.every((x: Json) => x.outcome === "skipped"),
					JSON.stringify(rest),
				);
				// The push starts 75 mm above where the fingers stopped, so the contact falls inside a segment: a
				// segment that starts touching never advances and ends by creeping (limited), as MotorMind's does.
				await s.observe();
				const stopHeight = s.pose().position_m[2] - (s.latest().table_z_m as number);
				// The fingertips stopped on the table (about 10 mm below the tool point), not on something else.
				assert.ok(stopHeight <= 0.03, `C6: the descent stopped ${mm(stopHeight)} mm above the table`);
				await toHeight(s, stopHeight + 0.075);
				const push = await s.act({ type: "move", direction: "down", distance_mm: 200, stop_on_contact: true });
				const contact = push.segments.find((x: Json) => x.outcome !== "done");
				assert.equal(contact?.outcome, "contact", JSON.stringify(push.segments));
				rec.C6 = { spot, blocked: stop, stop_height_mm: mm(stopHeight), contact };

				// C8: open fully (R: the width), close on air, open from there to 40 mm and hold it through a lift,
				// then close on a can where the scene has one (libero_object, libero_10).
				await toHeight(s, 0.15);
				const open = await s.act({ type: "gripper", state: "open" });
				const air = await s.act({ type: "gripper", state: "close" });
				assert.ok(
					air.segments[0].detail.opening_mm <= 3,
					`C8: an empty close reads ${air.segments[0].detail.opening_mm} mm`,
				);
				assert.equal(air.segments[0].grasp?.holding ?? air.grasp?.holding, false);
				const openTo = await s.act({ type: "gripper", state: "open", open_to_mm: 40 });
				const width = openTo.segments[0].detail.opening_mm;
				// Bounded below only: HOLD keeps the gripper's command, which leads the opening by about two
				// ticks, so the fingers end past 40 mm (DESIGN §12, C8; rpc-contract-as-built §7.9).
				assert.ok(width >= 40, `C8: open_to 40 opened only ${width} mm`);
				const lift = await s.act({ type: "move", direction: "up", distance_mm: 50 });
				const drift = Math.abs(lift.gripper.opening_mm - width);
				assert.ok(drift <= 1, `C8: the held opening drifted ${drift} mm through a lift (HOLD)`);
				await s.act({ type: "gripper", state: "open" });
				const objects = (await s.truthNow()).objects as Record<string, Json>;
				const can = Object.keys(objects).find((n) => /soup|sauce|ketchup|can/.test(n));
				let grip: Json | undefined;
				if (can) {
					const at = objects[can].pos_base as number[];
					await s.moveBy([at[0] - s.pose().position_m[0], at[1] - s.pose().position_m[1], 0]);
					// Open fingers straddle the can on the way down; the push ends on the table beside it.
					const table = s.latest().table_z_m as number;
					await s.act({
						type: "move",
						direction: "down",
						distance_mm: mm(Math.max(0, s.pose().position_m[2] - table - 0.015)),
						stop_on_contact: true,
					});
					grip = await s.act({ type: "gripper", state: "close" });
					assert.equal(
						grip.segments[0].grasp?.holding ?? grip.grasp?.holding,
						true,
						`C8: closing on ${can}: ${JSON.stringify(grip.segments[0])}`,
					);
					await s.act({ type: "gripper", state: "open" });
				}
				rec.C8 = {
					full_open_mm: open.segments[0].detail.opening_mm,
					full_open_vs_78_8_mm: open.segments[0].detail.opening_mm - 78.8,
					empty_close_mm: air.segments[0].detail.opening_mm,
					open_to_mm: width,
					open_to_over_mm: width - 40,
					lift_drift_mm: drift,
					object: can ?? null,
					held_mm: grip?.segments[0].detail.opening_mm ?? null,
				};

				// C7: a real skid. A fixture crash ends blocked in this env (DESIGN §12, C7): a fast forward move
				// leaves the arm moving, and a 10 mm lift right after it drifts forward, across its axis, past
				// the 4 mm limit. Through mm_act first, then traced beside the robot for the rig's judge and
				// the privileged truth.
				const fastStart = async () => {
					await toHeight(s, 0.2);
					await s.moveBy([0.45 - s.pose().position_m[0], -s.pose().position_m[1], 0]);
					await s.act({ type: "move", direction: "forward", distance_mm: 50, speed_mm_s: 1000 });
				};
				let skid: Json | undefined;
				const tries: Json[] = [];
				for (let k = 0; k < 3 && !skid; k++) {
					await fastStart();
					const lift = (await s.act({ type: "move", direction: "up", distance_mm: 10, speed_mm_s: 50 }))
						.segments[0];
					tries.push(lift);
					if (lift.abort_reason === "skidded") skid = lift;
				}
				assert.ok(skid, `C7: no skid through mm_act: ${JSON.stringify(tries)}`);
				assert.equal(skid.outcome, "limited", JSON.stringify(skid));
				let traced: Json | undefined;
				const replays: Json[] = [];
				for (let k = 0; k < 3 && !traced; k++) {
					await fastStart();
					const caps = s.latest().caps;
					const [seg] = segmentsFor(
						toDelta(parseAction({ type: "move", direction: "up", distance_mm: 10, speed_mm_s: 50 }), caps),
						s.pose(),
						caps,
					);
					const before = await s.truthNow();
					const r = await s.truth.call<Json>("env.mm_run_step", { seg, trace: true }, 120_000);
					const after = await s.truthNow();
					const replay = replaySegment(r.trace);
					// The server's along and lateral against the privileged end points, through R_b.
					const b = s.base();
					const toBase = (w: number[]) => apply(transpose(b.rotation), sub(w, b.position_m));
					const d = sub(toBase(after.eef_pos_world), toBase(before.eef_pos_world));
					const a = seg.axis as number[];
					const al = dot(d, a);
					const lat = norm(
						sub(
							d,
							a.map((v) => v * al),
						),
					);
					const one = {
						server: {
							abort_reason: r.abort_reason,
							along_axis_m: r.along_axis_m,
							lateral_m: r.lateral_m,
							servo_ticks: r.servo_ticks,
						},
						replay,
						truth: { along_axis_m: al, lateral_m: lat },
					};
					replays.push(one);
					assert.equal(replay.abort_reason, r.abort_reason, `C7: the rig re-judges ${JSON.stringify(one)}`);
					assert.equal(replay.outcome, r.outcome, `C7: ${JSON.stringify(one)}`);
					assert.ok(Math.abs(replay.along_axis_m - r.along_axis_m) <= 1e-9, `C7: ${JSON.stringify(one)}`);
					assert.ok(Math.abs(replay.lateral_m - r.lateral_m) <= 1e-9, `C7: ${JSON.stringify(one)}`);
					if (r.abort_reason) assert.equal(replay.abort_tick, r.servo_ticks, `C7: ${JSON.stringify(one)}`);
					assert.ok(mm(Math.abs(al - r.along_axis_m)) <= 0.5, `C7: along vs truth ${JSON.stringify(one)}`);
					assert.ok(mm(Math.abs(lat - r.lateral_m)) <= 0.5, `C7: lateral vs truth ${JSON.stringify(one)}`);
					if (r.abort_reason === "skidded") traced = one;
				}
				assert.ok(traced, `C7: no traced skid: ${JSON.stringify(replays)}`);
				await s.observe();
				rec.C7 = { tool: skid, tool_tries: tries.length, traced, traced_tries: replays.length };

				// C14: a cancel while a segment RPC is in flight: it finishes, no new segment starts.
				await toHeight(s, 0.2);
				const t0 = Date.now();
				setTimeout(() => s.h.cancel("e2e cancel"), 300);
				const cancelled = await s.act({ type: "move", direction: "backward", distance_mm: 200 });
				s.h.clearCancel();
				assert.deepEqual(
					outcomes(cancelled).slice(1),
					["skipped", "skipped", "skipped"],
					JSON.stringify(cancelled.segments),
				);
				assert.ok(cancelled.segments.slice(1).every((x: Json) => x.cancel_reason === "e2e cancel"));
				rec.C14 = { cancelled: cancelled.cancelled, wall_s: (Date.now() - t0) / 1000 };
				await s.act({ type: "move", direction: "forward", distance_mm: along(cancelled) });

				// C15 (b): outside a robot tool the callbacks refuse.
				await assert.rejects(s.h.observe(), /robot lock/);
				// (c): mm_observe beside a running mm_act waits its turn, and both finish.
				const acting = s.call("mm_act", { type: "move", direction: "left", distance_mm: 60 });
				const looking = s.call("mm_observe", {});
				const both = await Promise.race([
					Promise.all([acting, looking]),
					new Promise((_, no) => setTimeout(() => no(new Error("C15: deadlock")), 120_000)),
				]);
				const [a, o] = (both as Json[]).map((x) => JSON.parse(x.content[0].text));
				assert.ok(
					o.obs_id > a.obs_id,
					`C15: the observation (${o.obs_id}) ran before the action (${a.obs_id}) ended`,
				);
				await s.act({ type: "move", direction: "right", distance_mm: 60 });
				// (a): a segment RPC that times out leaves the outcome unknown; the gate wants mm_observe.
				await s.truth.call("env.mm_debug_stall", { seconds: 15 });
				const stalled = await s.call("mm_act", { type: "move", direction: "down", distance_mm: 20 });
				const sb = JSON.parse(stalled.content[0].text);
				assert.equal(sb.timed_out, true, JSON.stringify(sb));
				await s.emit("tool_result", {
					toolName: "mm_act",
					isError: false,
					content: stalled.content,
					details: stalled.details,
				});
				const refused = await s.emit("tool_call", { toolName: "mm_act", input: {} });
				assert.equal(refused?.block, true, "C15: the gate let a motion through after a timeout");
				const seen = await s.call("mm_observe", {});
				await s.emit("tool_result", {
					toolName: "mm_observe",
					isError: false,
					content: seen.content,
					details: seen.details,
				});
				assert.equal(await s.emit("tool_call", { toolName: "mm_act", input: {} }), undefined);
				rec.C15 = { concurrent: { act_obs: a.obs_id, observe_obs: o.obs_id }, timed_out: sb.segments[0] };

				// C4 (per tick): segments built here, run with the server's trace: no tick sends more than v * 0.05.
				const caps = s.latest().caps;
				const slow = segmentsFor(
					toDelta(parseAction({ type: "move", direction: "up", distance_mm: 60, speed_mm_s: 25 }), caps),
					s.pose(),
					caps,
				);
				const ticks: number[] = [];
				for (const seg of slow) {
					const r = await s.truth.call<Json>("env.mm_run_step", { seg, trace: true }, 120_000);
					for (const [dp] of r.trace.commanded as number[][]) ticks.push(dp);
				}
				assert.ok(
					ticks.length && Math.max(...ticks) <= 0.025 * 0.05 + 1e-9,
					`C4: a tick sent ${Math.max(...ticks)} m`,
				);
				rec.C4.max_tick_m_at_25mm_s = Math.max(...ticks);

				const result = await s.finish();
				assert.deepEqual(result.motormind.home, { mode: "refuse", asked: 1, refused: 1, reset: 0 });
				rec.result = result.motormind;
			} finally {
				mmRecord(`${suite}_t0_s0_behaviour`, rec);
				await s.emit("session_shutdown", { reason: "quit" });
			}
		},
	);

/** A MotorMind variant run (no preset): the faithful scene flags spelled out, `extra` on top. */
const mmVariant = (extra: Json) => ({
	mm: "true",
	"memory-profile": "off",
	"libero-side-camera": "true",
	"libero-settle-ticks": "32",
	...extra,
});

for (const suite of MM_SUITES)
	test(
		`libero-mm ${suite} t0 s0 --mm-home reset: home restores the scene, remounts the side camera, opens after settling (C12, C10)`,
		{ skip: skip(MM), timeout: 1_200_000 },
		async () => {
			const s = await mmSession({
				suite,
				task: "0",
				seed: "0",
				...mmVariant({ "mm-home": "reset", "libero-max-steps": "300000" }),
			});
			const rec: Json = {};
			try {
				const t0 = await s.truthNow();
				await s.act({ type: "move", direction: "up", distance_mm: 30 });
				// Displace the first object that has a free joint (fixtures such as the cabinet have none).
				let moved: string | undefined;
				for (const [name, o] of Object.entries(t0.objects as Record<string, Json>)) {
					try {
						await s.truth.call("env.mm_debug_set_body", {
							name,
							position_m: [o.pos[0] + 0.05, o.pos[1], o.pos[2] + 0.02],
						});
						moved = name;
						break;
					} catch {}
				}
				assert.ok(moved, "C12: no object could be displaced");
				await s.act({ type: "gripper", state: "close" });
				const home = await s.act({ type: "home" });
				const t1 = await s.truthNow();
				assert.deepEqual(outcomes(home), ["done"]);
				assert.notEqual(t1.episode_id, t0.episode_id, "C12: the episode id did not change");
				for (const [n, p] of Object.entries(t0.objects as Record<string, Json>)) {
					const err = norm(sub(t1.objects[n].pos, p.pos));
					assert.ok(err <= 1e-4, `C12: ${n} is ${err} m from its start pose`);
				}
				assert.equal(t1.settle_ticks, 32);
				assert.ok(t1.frame_clock_s > t0.frame_clock_s, "C12: the frame clock went back");
				// The gripper was closed when home settled: it is still closed, with the open command given after.
				assert.equal(t1.grip_channel, -1);
				assert.ok(
					home.gripper.opening_mm < 20,
					`C12: the fingers read ${home.gripper.opening_mm} mm right after home`,
				);
				rec.C10_second_reset = await checkSideMount(s, "C10 after home");
				const result = await s.finish();
				assert.equal(result.motormind.home.reset, 1);
				assert.equal(result.env_steps, 32);
				rec.C12 = {
					episode: [t0.episode_id, t1.episode_id],
					opening_after_home_mm: home.gripper.opening_mm,
					variant: result.motormind.variant,
				};
			} finally {
				mmRecord(`${suite}_t0_s0_home_reset`, rec);
				await s.emit("session_shutdown", { reason: "quit" });
			}
		},
	);

for (const suite of MM_SUITES)
	test(
		`libero-mm ${suite} t0 s0 --libero-max-steps 60: past the horizon mm_act refuses with episode_over; the steps add up (C16)`,
		{ skip: skip(MM), timeout: 900_000 },
		async () => {
			const s = await mmSession({
				suite,
				task: "0",
				seed: "0",
				...mmVariant({ "libero-max-steps": "60" }),
			});
			const rec: Json = {};
			let shut = false;
			try {
				let last: Json = {};
				for (let i = 0; i < 12 && !last.episode_over; i++)
					last = await s.act({ type: "move", direction: i % 2 ? "down" : "up", distance_mm: 30 });
				assert.equal(last.episode_over, true, "C16: the episode never ended");
				const after = await s.act({ type: "gripper", state: "close" });
				assert.deepEqual([outcomes(after), after.episode_over], [["refused"], true]);
				const result = await s.finish();
				assert.equal(
					result.env_steps,
					result.settle_ticks + result.motormind.ticks,
					JSON.stringify(result.motormind),
				);
				// The episode video is written at shutdown.
				shut = true;
				await s.emit("session_shutdown", { reason: "quit" });
				const video = s.entries.find((e) => e.type === "episode_video")?.data;
				if (video) assert.equal(video.frames, result.env_steps);
				rec.C16 = {
					env_steps: result.env_steps,
					settle_ticks: result.settle_ticks,
					ticks: result.motormind.ticks,
					video_frames: video?.frames ?? null,
					truncated: result.truncated,
				};
			} finally {
				mmRecord(`${suite}_t0_s0_horizon`, rec);
				if (!shut) await s.emit("session_shutdown", { reason: "quit" });
			}
		},
	);

/**
 * C13 (R): the settle ticks and the initial state each cell starts from (sim time and robosuite's
 * step counter across the reset, the state's digest), for the comparison with upstream LIBERO-PRO.
 * MM_STEP1_C13=all runs the 40 base tasks x seeds 0 and 1; by default libero_10 t0 x seeds 0 and 1.
 */
test("libero-mm C13 (R): settle ticks and initial states", { skip: skip(MM), timeout: 3_600_000 }, async () => {
	const cells =
		process.env.MM_STEP1_C13 === "all"
			? MM_SUITES.flatMap((suite) =>
					Array.from({ length: 10 }, (_, t) => [0, 1].map((seed) => [suite, t, seed] as const)).flat(),
				)
			: ([
					["libero_10", 0, 0],
					["libero_10", 0, 1],
				] as const);
	const rec: Json = {};
	for (const [suite, task, seed] of cells) {
		const s = await mmSession({ suite, task: String(task), seed: String(seed), preset: "motormind" });
		try {
			const t = await s.truthNow(true);
			rec[`${suite}_t${task}_s${seed}`] = {
				settle_ticks: t.settle_ticks,
				sim_time_s: t.sim_time_s,
				robosuite_timestep: t.robosuite_timestep,
				ticks: t.ticks,
				state_sha256: createHash("sha256").update(JSON.stringify(t.sim_state)).digest("hex"),
				gripper_qpos: t.gripper_qpos,
				arm_qpos: t.arm_qpos,
				objects: Object.fromEntries(Object.entries(t.objects as Record<string, Json>).map(([n, o]) => [n, o.pos])),
			};
		} finally {
			mmRecord("C13", rec);
			await s.emit("session_shutdown", { reason: "quit" });
		}
	}
});

/** C18 (R): MotorMind's capability constants re-measured here: tracking, tilt limits, radial reach, full open. */
test("libero-mm C18 (R): capability constants", { skip: skip(MM), timeout: 2_400_000 }, async () => {
	const s = await mmSession({ suite: "libero_goal", task: "0", seed: "0", preset: "motormind" });
	const rec: Json = {};
	try {
		rec.full_open_mm = (await s.act({ type: "gripper", state: "open" })).segments[0].detail.opening_mm;
		await toHeight(s, 0.2);
		rec.tracking = {};
		for (const speed of [12.5, 50]) {
			const r = await s.act({ type: "move", direction: "left", distance_mm: 50, speed_mm_s: speed });
			const seg = r.segments[0];
			rec.tracking[speed] = {
				along_mm: seg.along_axis_mm,
				ticks: seg.ticks,
				per_tick_mm: seg.along_axis_mm / Math.max(1, seg.ticks),
				commanded_per_tick_mm: speed * 0.05,
			};
			await s.act({ type: "move", direction: "right", distance_mm: 50, speed_mm_s: 50 });
		}
		rec.tilt_deg = {};
		const back: Record<string, string> = {
			roll_cw: "roll_ccw",
			roll_ccw: "roll_cw",
			pitch_down: "pitch_up",
			pitch_up: "pitch_down",
		};
		for (const word of Object.keys(back)) {
			let total = 0;
			for (let k = 0; k < 8; k++) {
				const r = await s.act({ type: "rotate", direction: word, angle_deg: 15 });
				total += turned(r);
				if (r.segments.some((x: Json) => x.outcome !== "done") || turned(r) < 7.5) break;
			}
			rec.tilt_deg[word] = total;
			// Turn back in commands of at most 90 deg.
			for (let left = total; left > 0.5; left -= 90)
				await s.act({ type: "rotate", direction: back[word], angle_deg: Math.min(90, left) });
		}
		rec.reach_m = {};
		for (const height of [0.1, 0.2]) {
			await toHeight(s, height);
			const p = s.pose().position_m;
			await s.moveBy([0.7 - p[0], -p[1], 0]);
			let reached = Math.hypot(s.pose().position_m[0], s.pose().position_m[1]);
			for (let k = 0; k < 10; k++) {
				const r = await s.act({ type: "move", direction: "forward", distance_mm: 20 });
				reached = Math.hypot(s.pose().position_m[0], s.pose().position_m[1]);
				if (r.segments.some((x: Json) => x.outcome !== "done") || along(r) < 5) break;
			}
			rec.reach_m[height] = reached;
			await s.moveBy([0.6 - reached, 0, 0]);
		}
		rec.vs_motormind = {
			full_open_mm: rec.full_open_mm - 78.8,
			tilt_deg: Object.fromEntries(
				Object.entries({ roll_cw: 88.6, roll_ccw: 83.7, pitch_down: 91.7, pitch_up: 53.6 }).map(([w, v]) => [
					w,
					rec.tilt_deg[w] - v,
				]),
			),
			reach_m: Object.fromEntries(Object.entries(rec.reach_m).map(([h, v]) => [h, (v as number) - 0.78])),
		};
	} finally {
		mmRecord("C18", rec);
		await s.emit("session_shutdown", { reason: "quit" });
	}
});

/** C17 in a real `pi -p` (the stub always has a UI): a MotorMind start that cannot run exits 1. */
test(
	"libero-mm C17: the real pi -p fails closed on a MotorMind setting it cannot run",
	{ skip: skip(MM), timeout: 600_000 },
	() => {
		const repo = join(servicesRoot, "..");
		const cli = process.env.PI_CLI ?? join(repo, "packages", "coding-agent", "dist", "cli.js");
		assert.ok(existsSync(cli), `no pi CLI at ${cli} (set PI_CLI)`);
		const model = process.env.MM_E2E_MODEL ?? "selfhost/qwen3.8-27b";
		mmFlags({});
		const run = (args: string[]) => {
			try {
				execFileSync(
					process.execPath,
					[
						cli,
						"-p",
						"-e",
						join(repo, "packages", "embodied", "src", "robots", "libero"),
						...args,
						"Solve the task.",
					],
					{
						encoding: "utf8",
						stdio: ["ignore", "pipe", "pipe"],
						timeout: 300_000,
					},
				);
				return { code: 0, stderr: "" };
			} catch (err) {
				const e = err as { status?: number; stderr?: string };
				return { code: e.status ?? -1, stderr: String(e.stderr ?? "") };
			}
		};
		const cases = {
			mm_without_motormind_model: run(["--mm=true", "--memory-profile=off", "--model", model]),
			preset_with_side_camera_off: run(["--preset", "motormind", "--libero-side-camera=false", "--model", model]),
			motormind_model_without_mm: run(["--memory-profile=off", "--model", `motormind/${model}`]),
		};
		mmRecord("C17_cli", cases);
		assert.equal(cases.mm_without_motormind_model.code, 1);
		assert.match(cases.mm_without_motormind_model.stderr, /--mm runs the MotorMind mission/);
		assert.equal(cases.preset_with_side_camera_off.code, 1);
		assert.match(
			cases.preset_with_side_camera_off.stderr,
			/--preset motormind sets --libero-side-camera=true, but --libero-side-camera=false was given/,
		);
		// Before M2 registers the motormind provider, pi itself refuses the model; after, the provider's check does.
		assert.notEqual(cases.motormind_model_without_mm.code, 0);
	},
);
