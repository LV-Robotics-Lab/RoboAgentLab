/**
 * MotorMind M2, the mission half (src/modes/motormind): the role models and faithful sampling
 * (models.ts), the frame and the cached look (contract.ts), the clause ledger (clauses.ts), the
 * evidence log, the planner's checks and mending, the verifier's gates, the mission state machine on
 * a scripted executor, and the provider's reentrancy, reconciliation and cost accounting. Frames and
 * segment reports come from M1's recorded run on the real LIBERO (test/fixtures/motormind/
 * m1-report-extract.json, bjb2 .local/runs/mm-step1/2026-10-12-fix2); role replies are scripted.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { AssistantMessage, Message, Model } from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { alignAnswers, codeEvent, emptyClauses, fold, measuredMiss } from "../src/modes/motormind/clauses.ts";
import {
	type Ask,
	type Cycle,
	describeRobot,
	type Executor,
	type Frame,
	type FrameCache,
	frameOf,
	hasImages,
	images,
	look,
	type MmCall,
	type MmReply,
	missionClock,
	type ObserveWhy,
	type Role,
	type RolePrompt,
	type RoleReply,
	type SegmentReport,
	type Subgoal,
	type SubgoalInput,
	type SubgoalRun,
} from "../src/modes/motormind/contract.ts";
import { EvidenceLog, renderLine } from "../src/modes/motormind/evidence.ts";
import { type Caps, MOTORMIND_EVENT, type MotorMindHandle, type Observation } from "../src/modes/motormind/index.ts";
import { runMission } from "../src/modes/motormind/mission.ts";
import {
	faithfulPayload,
	MAX_TOKENS,
	makeAsk,
	parseRoleJson,
	promptFile,
	roleModels,
} from "../src/modes/motormind/models.ts";
import {
	boxFault,
	complaints,
	envelope,
	gaveUp,
	gripperOnly,
	plan,
	replan,
	sameObject,
	wantsHolding,
	wantsRelease,
} from "../src/modes/motormind/planner.ts";
import {
	classify,
	END_ENTRY,
	isClosedLoopRefusal,
	motormindArgs,
	motormindProvider,
	replyOf,
	STATIC_TEXT,
	TURN_ENTRY,
} from "../src/modes/motormind/provider.ts";
import { parseVerdict, verify } from "../src/modes/motormind/verify.ts";
import { plannerOf } from "../src/planner/kind.ts";

type Json = Record<string, unknown>;
const FIX = JSON.parse(
	readFileSync(new URL("./fixtures/motormind/m1-report-extract.json", import.meta.url), "utf8"),
) as {
	scenes: Record<
		string,
		{
			C1: Json;
			C6: { blocked: SegmentReport; contact: SegmentReport };
			C7: SegmentReport;
			C8: Json;
			C15: { timed_out: SegmentReport };
		}
	>;
};
const GOAL = FIX.scenes.libero_goal;

/** LIBERO's capability table as the env server sends it (the original's values). */
const CAPS: Caps = {
	max_translation_m: 0.05,
	max_rotation_rad: 0.5,
	max_speed_m_s: 1.0,
	max_rotation_speed_rad_s: 10,
	default_speed_m_s: 0.05,
	default_rotation_speed_rad_s: Math.PI / 3,
	control_point_offset_m: 0,
	table_z_m: 0,
	supports_contact_stop: true,
	contact_stop_trusted: true,
	supports_home: false,
	gripper_gap_measured: true,
	gripper_open_span_m: 0.0788,
	gripper_pad_length_m: 0.017,
	comfortable_reach_m: 0.727,
	reach_m: 0.78,
	tilt_limits_deg: { roll_cw: 88.6, roll_ccw: 83.7, pitch_down: 91.7, pitch_up: 53.6 },
	tilted_floor_m: 0.1,
	tilt_reach_deg: 80,
	max_object_height_m: 0.16,
	subgoal_median_s: 21,
	subgoal_verdict_s: 4.6,
	arrival_tolerance_m: 0.003,
	notes: ["no home"],
};
const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0xff, 0xd9]).toString("base64");

/** An observation as the mode keeps it, from a recorded segment's `pose_after` and `grasp` (the M1 fixture). */
function observation(
	seg: SegmentReport,
	o: {
		images?: boolean;
		holding?: boolean | null;
		commanded?: "open" | "close" | "hold";
		clock?: number;
		obs_id?: number;
		episode_over?: boolean;
	} = {},
): Observation {
	const pose = {
		position_m: seg.pose_after!.position_mm.map((v) => v / 1000) as [number, number, number],
		rotation: seg.pose_after!.rotation,
	};
	const grasp = {
		holding: o.holding ?? seg.grasp?.holding ?? null,
		reason: seg.grasp?.reason ?? "",
		closed_fraction: seg.grasp?.closed_fraction,
		opening_mm: seg.grasp?.opening_mm,
	};
	const cam = (name: string) => ({
		mujoco_camera: name,
		width: 768,
		height: 768,
		intrinsic: [
			[927.06, 0, 384],
			[0, 927.06, 384],
			[0, 0, 1],
		],
		cam2base: [
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 1],
			[0, 0, 0, 1],
		],
		marked768: JPEG,
		marked512: JPEG,
		raw512: JPEG,
		aim_px: [400, 500],
	});
	const raw: Json = {
		obs_id: o.obs_id ?? 1,
		frame_clock_s: o.clock ?? seg.frame_clock_s ?? 0,
		pose: { position_m: pose.position_m, rotation: pose.rotation },
		base_pose: { position_m: GOAL.C1.p_b, rotation: GOAL.C1.R_b },
		caps: CAPS,
		table_z_m: 0,
		gripper: {
			opening_mm: seg.grasp?.opening_mm ?? 79,
			commanded: o.commanded ?? (seg.grasp as Json | undefined)?.commanded ?? "open",
			moving: false,
		},
		grasp,
		clearance: {
			seen: true,
			mm: 120,
			median_mm: 125,
			surface_z_m: 0,
			at_range_floor: false,
			near_min_range: false,
			jaws_down: true,
		},
		camera_names: ["scene", "side", "wrist"],
		...(o.images !== false
			? {
					cameras: { scene: cam("agentview"), side: cam("sideview"), wrist: cam("robot0_eye_in_hand") },
					mosaic_jpeg: JPEG,
				}
			: {}),
		episode_over: o.episode_over === true,
	};
	return {
		raw,
		obs_id: raw.obs_id as number,
		frame_clock_s: raw.frame_clock_s as number,
		pose,
		base_pose: {
			position_m: GOAL.C1.p_b as [number, number, number],
			rotation: GOAL.C1.R_b as Frame["pose"]["rotation"],
		},
		caps: CAPS,
		table_z_m: 0,
		episode_over: o.episode_over === true,
		terminated: false,
		truncated: false,
		degraded: {},
		...(o.images !== false ? { mosaic: { data: Buffer.from(JPEG, "base64"), mime: "image/jpeg" } } : {}),
	};
}
const frame = (seg: SegmentReport, o?: Parameters<typeof observation>[1]) => frameOf(observation(seg, o))!;

/** A scripted `Ask`: replies per role in order (a function sees the prompt and images); records every call. */
function scriptedAsk(script: Partial<Record<Role, (unknown | ((p: RolePrompt, n: number) => unknown))[]>>) {
	const calls: { role: Role; prompt: RolePrompt; images: number }[] = [];
	const used: Partial<Record<Role, number>> = {};
	const ask: Ask = async (role, prompt, imgs) => {
		calls.push({ role, prompt, images: imgs.length });
		const k = used[role] ?? 0;
		used[role] = k + 1;
		const entry = script[role]?.[k];
		const value =
			typeof entry === "function" ? (entry as (p: RolePrompt, n: number) => unknown)(prompt, imgs.length) : entry;
		const usage = { cost_usd: 0.01, tokens_in: 100 + k, tokens_out: 10, latency_s: 0.1 };
		if (value === undefined)
			return {
				ok: false,
				error: `no scripted ${role} reply #${k}`,
				error_kind: "transport",
				model: "faux/m",
				usage,
				attempts: 1,
			};
		if (value === "NO_JSON")
			return {
				ok: false,
				error: "the reply holds no JSON object",
				error_kind: "no_json",
				text: "prose",
				model: "faux/m",
				usage,
				attempts: 1,
			};
		return {
			ok: true,
			text: JSON.stringify(value),
			json: value,
			model: "faux/m",
			usage,
			attempts: 1,
		} satisfies RoleReply;
	};
	return { ask, calls };
}

const subgoal = (s: Partial<Subgoal> & Pick<Subgoal, "name" | "subgoal" | "criterion">): Subgoal => ({
	target: "",
	target_boxes: {},
	clause: 0,
	max_cycles: 5,
	gripper_only: false,
	...s,
});

// ---------------------------------------------------------------------------
// models

test("models: role models default to the base, --mm-role-models overrides per role and refuses a bad spec", () => {
	const m = roleModels("selfhost/q", undefined);
	assert.equal(m.planner, "selfhost/q");
	assert.equal(m.memory, "selfhost/q");
	const o = roleModels("selfhost/q", "verify=relay/v, locate=selfhost/l");
	assert.deepEqual([o.planner, o.verify, o.locate, o.executor], ["selfhost/q", "relay/v", "selfhost/l", "selfhost/q"]);
	assert.throws(() => roleModels("selfhost/q", "verify=nomodel"), /role=provider\/id/);
	assert.throws(() => roleModels("selfhost/q", "driver=relay/v"), /role=provider\/id/);
	assert.equal(MAX_TOKENS.planner, 3000);
	assert.equal(MAX_TOKENS.verify, 1500);
});

test("models: the faithful payload is temperature 0, max_tokens, thinking off, a JSON object; reasoning effort removed", () => {
	const body = { model: "q", messages: [], temperature: 0.7, max_completion_tokens: 50, reasoning_effort: "low" };
	const out = faithfulPayload(body, { api: "openai-completions" }, 3000) as Json;
	assert.equal(out.temperature, 0);
	assert.equal(out.max_tokens, 3000);
	assert.equal(out.max_completion_tokens, undefined);
	assert.equal(out.reasoning_effort, undefined);
	assert.deepEqual(out.chat_template_kwargs, { enable_thinking: false });
	assert.deepEqual(out.response_format, { type: "json_object" });
	// Another API: only the sampling fields it already has.
	const other = faithfulPayload({ max_tokens: 10, temperature: 1 }, { api: "anthropic-messages" }, 1500) as Json;
	assert.deepEqual(other, { max_tokens: 1500, temperature: 0 });
	assert.equal(faithfulPayload("not an object", { api: "openai-completions" }, 1), undefined);
});

test("models: JSON is read whole, then from a fence, then as the first balanced object, then from the thinking text", () => {
	assert.deepEqual(parseRoleJson('{"a": 1}'), { a: 1 });
	assert.deepEqual(parseRoleJson('Sure:\n```json\n{"a": [1, 2]}\n```\ndone'), { a: [1, 2] });
	assert.deepEqual(parseRoleJson('text {"a": "b}"} trailing {"c": 2}'), { a: "b}" });
	assert.deepEqual(parseRoleJson("", '{"from": "thinking"}'), { from: "thinking" });
	assert.equal(parseRoleJson("no json here"), undefined);
	assert.equal(parseRoleJson("{broken"), undefined);
});

test("models: makeAsk sends one user message with the role's system prompt and the faithful options; a failed stream is a transport error", async () => {
	const seen: { model: string; options: Json; systemPrompt?: string; images: number }[] = [];
	const registry = {
		find: (provider: string, id: string) =>
			provider === "selfhost" && id === "q"
				? ({ provider, id, api: "openai-completions" } as Model<"openai-completions">)
				: undefined,
		streamSimple: (
			m: Model<"openai-completions">,
			c: { systemPrompt?: string; messages: { content: unknown[] }[] },
			o: Json,
		) => {
			seen.push({ model: m.id, options: o, systemPrompt: c.systemPrompt, images: c.messages[0].content.length - 1 });
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [{ type: "text", text: '{"ok": true}' }],
				api: m.api,
				provider: m.provider,
				model: m.id,
				usage: {
					input: 50,
					output: 5,
					cacheRead: 2,
					cacheWrite: 0,
					totalTokens: 57,
					cost: { input: 0.1, output: 0.1, cacheRead: 0, cacheWrite: 0, total: 0.2 },
				},
				stopReason: "stop",
				timestamp: 1,
			};
			queueMicrotask(() => {
				stream.push({ type: "done", reason: "stop", message });
				stream.end();
			});
			return stream;
		},
	};
	const records: { role: Role; ok: boolean }[] = [];
	const ask = makeAsk(registry as never, roleModels("selfhost/q", "locate=nowhere/x"), (role, r) =>
		records.push({ role, ok: r.ok }),
	);
	const r = await ask("planner", { system: "S", text: "T" }, [{ type: "image", data: JPEG, mimeType: "image/jpeg" }]);
	assert.ok(r.ok);
	assert.deepEqual(r.json, { ok: true });
	assert.deepEqual(
		{ in: r.usage.tokens_in, out: r.usage.tokens_out, cost: r.usage.cost_usd },
		{ in: 52, out: 5, cost: 0.2 },
	);
	assert.equal(seen[0].systemPrompt, "S");
	assert.equal(seen[0].images, 1);
	assert.equal(seen[0].options.maxRetries, 3);
	assert.equal(seen[0].options.timeoutMs, 60_000);
	assert.equal(seen[0].options.maxTokens, 3000);
	const payload = (seen[0].options.onPayload as (p: unknown, m: unknown) => Json)(
		{ temperature: 1 },
		{ api: "openai-completions" },
	);
	assert.equal(payload.max_tokens, 3000);
	const missing = await ask("locate", { system: "S", text: "T" }, []);
	assert.ok(!missing.ok && missing.error_kind === "unknown_model");
	assert.deepEqual(records, [
		{ role: "planner", ok: true },
		{ role: "locate", ok: false },
	]);
});

// ---------------------------------------------------------------------------
// contract: frames, the cached look, the robot's description

test("contract: a frame from M1's recorded observation; images per size; the description reads the reading", () => {
	const f = frame(GOAL.C6.blocked, { holding: false, commanded: "close" });
	assert.deepEqual(
		f.pose.position_m.map((v) => Number((v * 1000).toFixed(2))),
		GOAL.C6.blocked.pose_after!.position_mm,
	);
	assert.deepEqual(f.camera_names, ["scene", "side", "wrist"]);
	assert.equal(f.cameras.wrist?.intrinsic[0][0], 927.06);
	assert.equal(images(f, "marked768").length, 3);
	assert.equal(images(f, "raw512", ["scene", "side"]).length, 2);
	assert.ok(hasImages(f));
	assert.ok(!hasImages(frame(GOAL.C6.blocked, { images: false })));
	const text = describeRobot(f);
	assert.match(text, /Tool point at \[525, 247, -2\] mm/);
	assert.match(text, /jaws point straight down/);
	assert.match(text, /Clearance under the tool: 120 mm \(measured now/);
	assert.match(text, /FULLY OPEN.*holding nothing/);
	const unknown = describeRobot(frame(GOAL.C6.blocked, { holding: null }), {
		mm: 42,
		descended_mm: 30,
		lateral_mm: 0,
	});
	assert.match(unknown, /holding: UNKNOWN/);
	// A frame without a wrist reading falls back to the tracked clearance.
	const noClear = frame(GOAL.C6.blocked);
	noClear.clearance = { seen: false, mm: null, reason: "blind" };
	assert.match(
		describeRobot(noClear, { mm: 42, descended_mm: 30, lateral_mm: 10 }),
		/about 42 mm, NOT measured now.*30 mm of descent.*10 mm sideways/,
	);
	assert.match(describeRobot(noClear), /unknown \(blind\)\. Do not assume/);
});

test("contract: look() reuses a fresh cached frame without a robot call and yields mm_observe when stale (C19 bookkeeping)", async () => {
	const fresh = frame(GOAL.C6.blocked);
	const cache: FrameCache = { latest: fresh, stale: false };
	const observes: { why: ObserveWhy; yielded: boolean }[] = [];
	const g1 = look(cache, "cycle_start", observes);
	const r1 = await g1.next();
	assert.ok(r1.done && r1.value === fresh);
	cache.stale = true;
	const g2 = look(cache, "before_action", observes, 90);
	const r2 = await g2.next();
	assert.ok(!r2.done);
	assert.deepEqual(r2.value, { tool: "mm_observe", args: { tracked_mm: 90 }, why: "before_action" });
	const after = frame(GOAL.C6.contact);
	const r3 = await g2.next({ ok: true, frame: after, episode_over: false });
	assert.ok(r3.done && r3.value === after);
	assert.deepEqual(observes, [
		{ why: "cycle_start", yielded: false },
		{ why: "before_action", yielded: true },
	]);
	// An error reply gives no frame.
	const g3 = look({ stale: true }, "opening", observes);
	await g3.next();
	const r4 = await g3.next({ ok: false, error: "refused", episode_over: false });
	assert.ok(r4.done && r4.value === undefined);
});

test("contract: the mission clock mirrors cancels to the robot, the first cancel wins, the deadline cancels by itself", () => {
	const seen: (string | undefined)[] = [];
	const clock = missionClock((c) => seen.push(c?.reason));
	assert.equal(clock.left_s(), Infinity);
	assert.equal(clock.cancelled(), undefined);
	clock.cancel({ by: "monitor", reason: "wrong object" });
	clock.cancel({ by: "grasp_watchdog", reason: "lost" });
	assert.equal(clock.cancelled()?.reason, "wrong object");
	clock.clear();
	assert.equal(clock.cancelled(), undefined);
	clock.deadline_at = Date.now() - 1;
	assert.equal(clock.cancelled()?.by, "deadline");
	assert.deepEqual(seen, ["wrong object", undefined]);
});

// ---------------------------------------------------------------------------
// clauses

test("clauses: answers align by content-word share, else by position; folding marks, withdraws and refuses re-marks without a code event", () => {
	const clauses = ["the black bowl is on the plate", "the drawer is closed"];
	assert.deepEqual(
		alignAnswers(clauses, [
			{ what: "drawer closed", met: true, evidence: "" },
			{ what: "bowl on plate", met: false, evidence: "the black bowl sits beside the plate" },
		]),
		[1, 0],
	);
	assert.deepEqual(
		alignAnswers(clauses, [
			{ what: "first", met: true, evidence: "" },
			{ what: "second", met: true, evidence: "" },
		]),
		[0, 1],
	);
	assert.deepEqual(alignAnswers(clauses, [{ what: "nothing", met: true, evidence: "" }]), [undefined]);
	const s = emptyClauses(clauses);
	fold(s, [{ what: "black bowl on the plate", met: true, evidence: "" }], false);
	assert.deepEqual([...s.met], [0]);
	// A plain verdict cannot withdraw; a task_done / task question / final check can.
	fold(s, [{ what: "black bowl on the plate", met: false, evidence: "" }], false);
	assert.deepEqual([...s.met], [0]);
	fold(s, [{ what: "black bowl on the plate", met: false, evidence: "it fell" }], true);
	assert.deepEqual([[...s.met], [...s.withdrawn]], [[], [0]]);
	fold(s, [{ what: "black bowl on the plate", met: true, evidence: "" }], false);
	assert.deepEqual([[...s.met], s.refused_mark], [[], 1]);
	codeEvent(s, 0);
	fold(s, [{ what: "black bowl on the plate", met: true, evidence: "" }], false);
	assert.deepEqual([...s.met], [0]);
	measuredMiss(s, 1);
	fold(s, [{ what: "the drawer is closed", met: true, evidence: "" }], false);
	assert.deepEqual([...s.met], [0], "a measured miss is not undone by a model's word");
});

// ---------------------------------------------------------------------------
// evidence

test("evidence: a cycle line takes its steps' measurements and the located target; lines render with the numbers", () => {
	const log = new EvidenceLog();
	log.event("take", {
		type: "proposed",
		cycle: 0,
		proposal: {
			assessment: "over it",
			done: false,
			expect: "descend",
			confidence: 0.8,
			command: {},
			deltas: [{ kind: "translate", label: "move down 50.0 mm (base)", push: false }],
		},
	});
	log.event("take", { type: "step_end", cycle: 0, step: GOAL.C6.blocked });
	log.event("take", {
		type: "reached",
		cycle: 0,
		reach: {
			target: "bowl",
			from_plan_boxes: true,
			residual_mm: 12.4,
			located: { label: "black bowl", point_mm: [500, 200, 0], error_mm: 15, doubtful: false },
		},
	});
	const line = log.lines[0];
	assert.equal(line.kind, "cycle");
	assert.match(
		renderLine(line),
		/^\[cycle\] take: cycle 0: move down 50\.0 mm \(base\) → measured: .*\[limited: blocked\] \(cycle=0 along_axis_mm=16.7 lateral_mm=0\.7 residual_mm=12\.4\)$/,
	);
	assert.match(String(line.data.located), /^bowl at \[500, 200, 0\] mm ± 15$/);
	log.event("take", {
		type: "supervision",
		cycle: 0,
		supervision: { decision: "stop", finding: "wrong_direction", because: "moved away" },
	});
	assert.deepEqual(log.recent("take", 3).length, 2);
});

// ---------------------------------------------------------------------------
// planner

test("planner: words — holding, release, gripper-only, same object, box faults", () => {
	assert.ok(wantsHolding("the bowl is held 120 mm above the table"));
	assert.ok(!wantsHolding("the bowl rests on the plate, jaws open"));
	assert.ok(wantsRelease("the bowl is resting on the plate"));
	assert.ok(gripperOnly({ subgoal: "open the gripper", criterion: "the jaws are open" }));
	assert.ok(!gripperOnly({ subgoal: "lower and close the gripper", criterion: "the bowl is held" }));
	assert.ok(sameObject("the black bowl", "bowl, black, left of the plate"));
	assert.ok(!sameObject("the plate", "the bowl"));
	assert.equal(boxFault("scene", [100, 100, 300, 300], ["scene", "side"]), undefined);
	assert.match(boxFault("wrist", [0, 0, 10, 10], ["scene", "side"])!, /not a fixed camera/);
	assert.match(boxFault("scene", [0, 0, 700, 700], ["scene"])!, /40%/);
	assert.match(boxFault("scene", [300, 300, 100, 100], ["scene"])!, /reversed/);
	assert.match(boxFault("scene", [0, 0, 1001, 10], ["scene"])!, /0\.\.1000/);
});

test("planner: the first plan — a missing clause list is asked once, then the task is clause 0; boxes and budgets are mended", async () => {
	const f = frame(GOAL.C6.blocked);
	const step = {
		name: "over_bowl",
		subgoal: "Move the gripper over the black bowl",
		criterion: "the tool is above the bowl",
		target: "the black bowl",
		target_boxes: { scene: [100, 100, 200, 200], wrist: [0, 0, 10, 10], side: [0, 0, 900, 900] },
		clause: 0,
		max_cycles: 40,
	};
	const { ask, calls } = scriptedAsk({
		planner: [
			{ subgoals: [step], rationale: "one bowl" },
			(p: RolePrompt) => {
				assert.match(p.text, /rejected: the first plan must list the task's clauses/);
				return {
					subgoals: [
						step,
						{
							...step,
							name: "over_bowl",
							subgoal: "Lower and close on the rim of the bowl",
							criterion: "the bowl is held",
							max_cycles: 0,
						},
					],
					rationale: "one bowl",
					clauses: ["the bowl is on the plate"],
				};
			},
		],
	});
	const out = await plan(ask, {
		task: "put the black bowl on the plate",
		frame: f,
		robotText: "R",
		capabilities: "C",
		legend: "L",
		budget: 10,
	});
	assert.ok(out.plan, out.error);
	assert.equal(calls.length, 2);
	assert.equal(calls[0].images, 3, "three 768 px views");
	assert.match(calls[0].prompt.text, /BUDGET: each step may use up to 10 actions; aim for about 5/);
	assert.deepEqual(out.plan.clauses, ["the bowl is on the plate"]);
	assert.deepEqual(
		out.plan.subgoals.map((s) => s.name),
		["over_bowl", "over_bowl_2"],
	);
	assert.deepEqual(
		out.plan.subgoals.map((s) => s.max_cycles),
		[10, 1],
	);
	assert.deepEqual(
		Object.keys(out.plan.subgoals[0].target_boxes),
		["scene"],
		"the wrist box and the 81% box are dropped",
	);
	assert.equal(out.plan.subgoals[1].target, "the black bowl");
	assert.equal(out.plan.boxed_at, f.frame_clock_s);
	assert.deepEqual(out.reasks, { clauses: 1 });
	assert.ok(out.clamped.some((c) => /max_cycles clamped/.test(c)));
	// A plan that never parses spends the three attempts.
	const bad = scriptedAsk({ planner: ["NO_JSON", "NO_JSON", "NO_JSON"] });
	const failed = await plan(bad.ask, { task: "t", frame: f, robotText: "R", capabilities: "C", budget: 10 });
	assert.equal(failed.plan, undefined);
	assert.equal(failed.attempts, 3);
	assert.match(failed.error!, /3 attempts/);
});

test("planner: complaints — size, boxes, side, orphan clause, look-only clause, repeated impossible step", () => {
	const fixed = ["scene", "side"] as const;
	const raw = {
		subgoals: [
			{
				name: "a",
				subgoal: "pull the drawer handle",
				criterion: "the handle is held",
				target: "drawer",
				target_boxes: {},
				clause: 0,
			},
			{ name: "b", subgoal: "look at the bowl", criterion: "seen", target: "", clause: 1 },
			{ name: "c", subgoal: "push the lid shut", criterion: "shut", target: "", clause: 2 },
		],
		rationale: "",
		clauses: ["drawer open", "bowl placed", "lid shut"],
		recoverable: true,
	};
	const keys = complaints(raw, {
		first: true,
		fixed: [...fixed],
		cannot: ["push the lid shut"],
		budget: 10,
		task: "t",
	}).map((c) => c.key);
	assert.deepEqual(keys, ["boxes", "side", "act", "cannot"]);
	const big = {
		...raw,
		subgoals: Array.from({ length: 25 }, (_, i) => ({
			name: `s${i}`,
			subgoal: "pick x",
			criterion: "held",
			clause: 0,
		})),
	};
	assert.ok(
		complaints(big, { first: false, fixed: [...fixed], cannot: [], budget: 10, task: "t" }).some(
			(c) => c.key === "size",
		),
	);
	assert.ok(envelope([{ name: "x", subgoal: "s", criterion: "c" }]).raw);
	assert.match(envelope({ name: "x", subgoal: "s", criterion: "c" }).shape!, /single subgoal/);
	assert.match(envelope({ nothing: 1 }).error!, /no subgoals/);
	assert.match(envelope("text").error!, /not a JSON object/);
});

test("planner: a replan keeps the remaining steps' boxes for the same object drawn after the last grasp, lets go only by the rule", async () => {
	const f = frame(GOAL.C6.contact, { clock: 50 });
	const old = {
		subgoals: [
			subgoal({
				name: "over",
				subgoal: "go over the bowl",
				criterion: "above the bowl",
				target: "the black bowl",
				target_boxes: { scene: [100, 100, 200, 200] },
			}),
			subgoal({
				name: "take",
				subgoal: "grasp the bowl",
				criterion: "the bowl is held",
				target: "the black bowl",
				target_boxes: { scene: [100, 100, 200, 200] },
			}),
		],
		rationale: "",
		clauses: ["the bowl is on the plate"],
		boxed_on: ["scene" as const],
		boxed_at: 10,
		boxes_kept_from: {},
		recoverable: true,
	};
	const base = {
		task: "put the bowl on the plate",
		frame: f,
		robotText: "R",
		capabilities: "C",
		budget: 10,
		plan: old,
		verdicts: old.subgoals.map((s, i) => ({
			subgoal: s,
			verdict: i === 0 ? { status: "done", because: "ok", attempt: 1 } : undefined,
		})),
		keep: 1,
		tail: [],
		trigger: "take failed",
		unmet: [{ i: 0, text: old.clauses[0] }],
		note: "",
		strict: false,
		holding: false,
		rebox: [],
		doneNames: ["over"],
		cannot: [],
		replans: 1,
	};
	const fresh = {
		name: "take_again",
		subgoal: "grasp the bowl by its rim",
		criterion: "the bowl is held",
		target: "the black bowl",
		target_boxes: { scene: [500, 500, 600, 600] },
		clause: 0,
		max_cycles: 8,
	};
	const { ask, calls } = scriptedAsk({ planner: [{ subgoals: [fresh], rationale: "rim grasp" }] });
	// Boxes drawn at clock 10, nothing grasped since: the old box replaces the new one (§4.5).
	const kept = await replan(ask, base);
	assert.match(calls[0].prompt.text, /THE PLAN SO FAR.*1\. over: go over the bowl.*done: ok.*2\. take:.*not run/s);
	assert.deepEqual(
		kept.plan!.subgoals.map((s) => s.name),
		["over", "take_again"],
	);
	assert.deepEqual(kept.plan!.subgoals[1].target_boxes, { scene: [100, 100, 200, 200] });
	assert.deepEqual(kept.plan!.boxes_kept_from, { take_again: 10 });
	// A grasp at clock 20 since: the new box stands.
	const redrawn = await replan(scriptedAsk({ planner: [{ subgoals: [fresh], rationale: "" }] }).ask, {
		...base,
		since: 20,
	});
	assert.deepEqual(redrawn.plan!.subgoals[1].target_boxes, { scene: [500, 500, 600, 600] });
	// Holding: an open-without-destination step is removed; nothing left puts the old steps back.
	const dropper = await replan(
		scriptedAsk({
			planner: [
				{
					subgoals: [{ name: "open", subgoal: "open the gripper", criterion: "jaws open", target: "", clause: 0 }],
					rationale: "",
				},
			],
		}).ask,
		{ ...base, holding: true },
	);
	assert.deepEqual(
		dropper.plan!.subgoals.map((s) => s.name),
		["over", "take"],
	);
	assert.ok(
		dropper.clamped.some((c) => /no place to put down/.test(c)) &&
			dropper.clamped.some((c) => /old remaining steps put back/.test(c)),
	);
	// Giving up: the first replan needs an irrecoverable reason; from the second any "false" is let through.
	assert.ok(!gaveUp("try harder", 1));
	assert.ok(gaveUp("the bowl fell on the floor", 1));
	assert.ok(gaveUp("no idea", 2));
	const quit = await replan(
		scriptedAsk({ planner: [{ subgoals: [fresh], rationale: "the bowl is out of reach", recoverable: false }] }).ask,
		base,
	);
	assert.equal(quit.gave_up, "the bowl is out of reach");
});

// ---------------------------------------------------------------------------
// verifier

test("verifier: the gripper gate decides before the model; a grasp with a width asks identity only", async () => {
	const now = frame(GOAL.C6.blocked, { holding: true, commanded: "close" });
	const base = {
		task: "put the bowl on the plate",
		now,
		before: now,
		note: "",
		planText: "",
		robotText: "R",
		clauses: ["bowl on plate"],
		claim: {
			done: true,
			assessment: "",
			cycles: 1,
			stopped_by: "",
			ending: "",
			turn_net_deg: 0,
			turn_total_deg: 0,
			move_net_mm: 0,
			move_total_mm: 0,
		},
	};
	const still = scriptedAsk({
		verify: [{ status: "not_done", because: "", evidence: "", identity: "", replan: false, clauses: [] }],
	});
	const v1 = await verify(still.ask, {
		...base,
		subgoal: subgoal({ name: "put", subgoal: "put it down", criterion: "the bowl rests on the plate, jaws open" }),
		holding: true,
	});
	assert.deepEqual([v1.status, v1.gate], ["not_done", "still holding"]);
	assert.equal(still.calls.length, 1, "only the whole-task question was asked");
	assert.equal(v1.asked?.status, "not_done");
	const empty = scriptedAsk({});
	const v2 = await verify(empty.ask, {
		...base,
		subgoal: subgoal({ name: "take", subgoal: "grasp", criterion: "the bowl is held" }),
		holding: false,
	});
	assert.deepEqual([v2.status, v2.gate], ["not_done", "nothing held"]);
	// Held with a width: no valid identity answer defaults to done; a not_done at the mark with no printing read becomes done with the doubt kept aside.
	const held = {
		...base,
		claim: { ...base.claim, held: { gap_mm: 63, target_mm: 60 }, from_mark_mm: 12, located_label: "black bowl" },
	};
	const v3 = await verify(scriptedAsk({ verify: ["NO_JSON"] }).ask, {
		...held,
		subgoal: subgoal({ name: "take", subgoal: "grasp", criterion: "the bowl is held", target: "the black bowl" }),
		holding: true,
	});
	assert.deepEqual([v3.status, v3.gate], ["done", "held_width"]);
	const doubt = scriptedAsk({
		verify: [
			(p: RolePrompt) => {
				assert.match(p.text, /THE GRASP IS SETTLED/);
				return {
					status: "not_done",
					because: "it may be the other bowl",
					evidence: "",
					identity: "two bowls look alike",
					replan: false,
					clauses: [],
				};
			},
		],
	});
	const v4 = await verify(doubt.ask, {
		...held,
		subgoal: subgoal({ name: "take", subgoal: "grasp", criterion: "the bowl is held", target: "the black bowl" }),
		holding: true,
	});
	assert.deepEqual(
		[v4.status, v4.gate, v4.identity, v4.overrode?.status],
		["done", "held_width", "two bowls look alike", "not_done"],
	);
	// The located-identity gate: the camera's label shares no word with the target.
	const other = { ...held, claim: { ...held.claim, located_label: "red mug" } };
	const v5 = await verify(
		scriptedAsk({
			verify: [{ status: "done", because: "held", evidence: "", identity: "", replan: false, clauses: [] }],
		}).ask,
		{
			...other,
			subgoal: subgoal({ name: "take", subgoal: "grasp", criterion: "the bowl is held", target: "the black bowl" }),
			holding: true,
		},
	);
	assert.deepEqual([v5.status, v5.gate, v5.replan], ["not_done", "located_identity", true]);
});

test("verifier: the contradiction re-ask, the control gate and the clause gate", async () => {
	const now = frame(GOAL.C6.blocked, { holding: true, commanded: "close" });
	const base = {
		task: "turn the knob",
		now,
		note: "",
		planText: "",
		robotText: "R",
		clauses: ["the knob is turned"],
		claim: {
			done: true,
			assessment: "",
			cycles: 2,
			stopped_by: "",
			ending: "",
			turn_net_deg: 0,
			turn_total_deg: 0,
			move_net_mm: 0,
			move_total_mm: 0,
		},
	};
	const lift = subgoal({
		name: "lift",
		subgoal: "lift the bowl",
		criterion: "the bowl is held 120 mm up",
		target: "bowl",
	});
	// not_done "nothing in the jaws" against a holding reading: asked again with the reading; still contradictory → done by the gate.
	const stubborn = scriptedAsk({
		verify: [
			{ status: "not_done", because: "nothing in the jaws", evidence: "", identity: "", replan: false, clauses: [] },
			(p: RolePrompt) => {
				assert.match(p.text, /THE READING SETTLES HOLDING/);
				return {
					status: "not_done",
					because: "the jaws are empty",
					evidence: "",
					identity: "",
					replan: false,
					clauses: [],
				};
			},
		],
	});
	const v1 = await verify(stubborn.ask, {
		...base,
		subgoal: lift,
		holding: true,
		claim: { ...base.claim, located_label: "bowl" },
	});
	assert.deepEqual([v1.status, v1.gate], ["done", "holding_true"]);
	// A turn step: 40 deg total, 4 deg net → turned and turned back.
	const turn = subgoal({
		name: "turn",
		subgoal: "turn the knob about 90 degrees",
		criterion: "the knob is turned 90 degrees",
	});
	const v2 = await verify(
		scriptedAsk({
			verify: [{ status: "done", because: "turned", evidence: "", identity: "", replan: false, clauses: [] }],
		}).ask,
		{ ...base, subgoal: turn, holding: true, claim: { ...base.claim, turn_net_deg: 4, turn_total_deg: 40 } },
	);
	assert.deepEqual([v2.status, v2.gate], ["not_done", "net_turn"]);
	// 30 deg one way of 90 asked, a stop met: not done, turn the other way.
	const v3 = await verify(
		scriptedAsk({ verify: [{ status: "done", because: "", evidence: "", identity: "", replan: false, clauses: [] }] })
			.ask,
		{
			...base,
			subgoal: turn,
			holding: true,
			claim: { ...base.claim, turn_net_deg: 30, turn_total_deg: 30, turn_stop: "yaw_left" },
		},
	);
	assert.deepEqual([v3.status, v3.gate], ["not_done", "half_turn"]);
	assert.match(v3.because, /turn the other way/);
	// task_done without every clause met: a step is done, the whole task is not_done.
	const partial = {
		status: "task_done",
		because: "all good",
		evidence: "",
		identity: "",
		replan: false,
		clauses: [{ what: "knob turned", met: false, evidence: "" }],
	};
	const v4 = await verify(scriptedAsk({ verify: [partial] }).ask, { ...base, subgoal: lift, holding: null });
	assert.deepEqual([v4.status, v4.gate], ["done", "clauses"]);
	const v5 = await verify(scriptedAsk({ verify: [partial] }).ask, {
		...base,
		subgoal: lift,
		holding: null,
		wholeTask: true,
	});
	assert.deepEqual([v5.status, v5.gate], ["not_done", "clauses"]);
	assert.equal(parseVerdict({ status: "maybe" }), undefined);
	assert.equal(parseVerdict({ status: "DONE", clauses: [{ what: "a", met: "true" }] })?.clauses[0].met, true);
});

// ---------------------------------------------------------------------------
// the mission on a scripted executor

type Scripted = { calls: MmCall[]; run: Partial<SubgoalRun> & Pick<SubgoalRun, "ending">; frames?: Frame[] };
/** An executor whose attempts are scripted: a cycle_start look (the cached frame when fresh), then its robot calls, then its ending. */
function scriptedExecutor(script: (input: SubgoalInput, n: number) => Scripted) {
	const inputs: SubgoalInput[] = [];
	const factory = (input: SubgoalInput): Executor => {
		const n = inputs.length;
		inputs.push(input);
		return (async function* (): Executor {
			const s = script(input, n);
			const observes: { why: ObserveWhy; yielded: boolean }[] = [];
			const first = yield* look(input.cache, "cycle_start", observes);
			const steps: SegmentReport[] = [];
			let episodeOver = first?.episode_over === true;
			for (const call of s.calls) {
				if (input.clock.cancelled()) return { ...finish("CANCELLED"), ending_text: "cancelled" };
				const reply: MmReply = yield call;
				if (reply.segments) {
					steps.push(...reply.segments);
					for (const st of reply.segments) input.onEvent({ type: "step_end", cycle: 0, step: st });
				}
				episodeOver ||= reply.episode_over;
			}
			return finish(s.run.ending);
			function finish(ending: SubgoalRun["ending"]): SubgoalRun {
				const cycle: Cycle = {
					index: 0,
					steps,
					proposal: {
						assessment: "a",
						done: ending === "MODEL_DONE",
						expect: "e",
						confidence: 1,
						command: {},
						deltas: s.calls
							.filter((c) => c.tool === "mm_act")
							.map((c) => ({
								kind: "translate" as const,
								label: c.tool === "mm_act" ? c.label : "",
								push: false,
							})),
					},
					timing: { wall_s: 1, motion_s: 0.5, calls: [] },
				};
				input.onEvent({ type: "proposed", cycle: 0, proposal: cycle.proposal! });
				return {
					cycles: s.calls.length ? [cycle] : [],
					ending_text: ending.toLowerCase(),
					episode_over: episodeOver,
					carry: { ...(input.carry ?? {}), n },
					observes,
					timing: cycle.timing,
					...s.run,
					ending,
				};
			}
		})();
	};
	return { factory, inputs };
}

/** Drive a mission as the provider does: answer each yielded call with `reply(call)`, keeping the frame cache, until it ends. */
async function drive(
	mission: ReturnType<typeof runMission>,
	cache: FrameCache,
	reply: (call: MmCall, n: number) => MmReply,
) {
	const calls: MmCall[] = [];
	let next = await mission.next();
	while (!next.done) {
		calls.push(next.value);
		const r = reply(next.value, calls.length - 1);
		if (r.frame) {
			cache.latest = r.frame;
			cache.stale = !hasImages(r.frame) || !r.ok;
		} else cache.stale = true;
		next = await mission.next(r);
	}
	return { end: next.value, calls };
}
const act = (label: string): MmCall => ({
	tool: "mm_act",
	args: { type: "move", direction: "down", distance_mm: 50, label },
	label,
});
const PLAN = {
	clauses: ["the black bowl is on the plate"],
	rationale: "one bowl, one plate",
	subgoals: [
		{
			name: "take",
			subgoal: "Grasp the black bowl",
			criterion: "the bowl is held",
			target: "the black bowl",
			target_boxes: { scene: [100, 100, 200, 200] },
			clause: 0,
			max_cycles: 8,
		},
		{
			name: "put",
			subgoal: "Put the bowl on the plate",
			criterion: "the bowl rests on the plate, jaws open",
			target: "the black bowl",
			target_boxes: { scene: [100, 100, 200, 200] },
			clause: 0,
			max_cycles: 6,
		},
	],
};
const DONE = {
	status: "done",
	because: "it is",
	evidence: "the bowl",
	identity: "",
	replan: false,
	clauses: [{ what: "the black bowl is on the plate", met: false, evidence: "" }],
};
const TASK_DONE = {
	status: "task_done",
	because: "the bowl sits on the plate",
	evidence: "plate",
	identity: "",
	replan: false,
	clauses: [{ what: "the black bowl is on the plate", met: true, evidence: "seen" }],
};
const describers = { legend: () => "L", capabilities: () => "C" };

test("mission: plan, two subgoals, the whole-task check; one robot call per yield; looks reuse the cached frame (C19)", async () => {
	const held = frame(GOAL.C6.blocked, { holding: true, commanded: "close", clock: 1 });
	const open = frame(GOAL.C6.contact, { holding: false, commanded: "open", clock: 2 });
	const ex = scriptedExecutor((_input, n) => ({
		calls: [act(n === 0 ? "move down 50.0 mm (base)" : "move up 50.0 mm (base)")],
		run: { ending: n === 0 ? "HELD" : "MODEL_DONE", held: n === 0 ? { gap_mm: 63 } : undefined },
	}));
	// take: the held-with-width identity question; put: the release watchdog settles it in code and the whole task is asked once; then the task check.
	const { ask, calls: roleCalls } = scriptedAsk({ planner: [PLAN], verify: [DONE, DONE, TASK_DONE] });
	const entries: { entry: string; data: Json }[] = [];
	const clock = missionClock();
	const cache: FrameCache = { stale: true };
	const mission = runMission({
		task: "put the black bowl on the plate",
		budget: 10,
		budgetS: 360,
		maxReplans: 10,
		ask,
		clock,
		cache,
		describers,
		runSubgoal: ex.factory,
		log: (entry, data) => entries.push({ entry, data }),
	});
	let n = 0;
	const { end, calls } = await drive(mission, cache, (call) => {
		n++;
		// the opening look and the take's action see the bowl held; the put's action sees it released
		const f = n <= 2 ? held : open;
		if (call.tool === "mm_observe") return { ok: true, frame: f, episode_over: false, body: {} };
		return {
			ok: true,
			frame: f,
			segments: [{ ...GOAL.C6.blocked, label: call.label, outcome: "done" as const, abort_reason: undefined }],
			episode_over: false,
			body: {},
		};
	});
	assert.equal(
		end.status,
		"task_done",
		JSON.stringify({
			end,
			entries: entries.map((e) => [
				e.entry,
				e.data.subgoal,
				e.data.status,
				e.data.gate,
				e.data.because,
				e.data.trigger,
			]),
		}),
	);
	// The opening look, then one action per subgoal: every later look (the executor's cycle start, the
	// look before each verdict, the task check) reused the frame the last action returned, since nothing
	// moved in between (DESIGN §1.4 step 3) — C19's bookkeeping says which.
	assert.deepEqual(
		calls.map((c) => c.tool),
		["mm_observe", "mm_act", "mm_act"],
	);
	assert.deepEqual((calls[0] as { why: string }).why, "opening");
	const s = mission.state();
	assert.deepEqual(s.observes, [
		{ why: "opening", yielded: true },
		{ why: "cycle_start", yielded: false },
		{ why: "before_verdict", yielded: false },
		{ why: "cycle_start", yielded: false },
		{ why: "before_verdict", yielded: false },
		{ why: "task_check", yielded: false },
	]);
	assert.deepEqual(
		s.subgoals.map((a) => [a.name, a.attempt, a.verdict?.status]),
		[
			["take", 1, "done"],
			["put", 1, "done"],
		],
	);
	assert.deepEqual(s.clauses.met, [0]);
	assert.equal(s.actions, 2);
	assert.equal(s.actions_cap, 28);
	assert.deepEqual(
		entries.map((e) => e.entry),
		[
			"motormind_plan",
			"motormind_cycle",
			"motormind_verdict",
			"motormind_cycle",
			"motormind_verdict",
			"motormind_verdict",
		],
	);
	// The carry flows from one attempt to the next; the second attempt got no preface.
	assert.deepEqual(
		ex.inputs.map((i) => i.carry),
		[undefined, { n: 0 }],
	);
	assert.deepEqual(
		ex.inputs.map((i) => i.preface),
		["", ""],
	);
	// The planner saw three 768 px views; the verifier six (now and before); the "put" verdict asked the whole task too (a release step).
	assert.equal(roleCalls[0].images, 3);
	assert.equal(roleCalls.filter((c) => c.role === "verify")[0].images, 6);
	assert.equal(roleCalls.filter((c) => c.role === "verify").length, 3);
	assert.deepEqual(
		s.subgoals.map((a) => a.verdict?.gate),
		["held_width", "released"],
	);
	assert.ok(clock.deadline_at !== undefined && clock.deadline_at > Date.now());
});

test("mission: a not_done without replan is retried once with a preface; a never-moved attempt is not retried but replanned; the price gate refuses a late replan", async () => {
	// The reading is unknown throughout, so no gripper gate decides; the verifier's not_done stands.
	const f = frame(GOAL.C6.blocked, { holding: null, commanded: "open" });
	// Attempt 1 moves (not_done), attempt 2 does not move at all (never moved) → straight to a replan.
	const ex = scriptedExecutor((_input, n) => ({
		calls: n === 0 ? [act("move down 50.0 mm (base)")] : [],
		run: { ending: n === 0 ? "CYCLES_EXHAUSTED" : "FROZEN" },
	}));
	const NOT = { status: "not_done", because: "still above", evidence: "", identity: "", replan: false, clauses: [] };
	const { ask, calls: roleCalls } = scriptedAsk({
		planner: [
			PLAN,
			{
				subgoals: [{ ...PLAN.subgoals[0], name: "take2", subgoal: "Grasp the black bowl by the rim" }],
				rationale: "rim",
			},
		],
		verify: [NOT, NOT],
	});
	const clock = missionClock();
	const cache: FrameCache = { stale: true };
	const mission = runMission({
		task: "t",
		budget: 10,
		budgetS: 360,
		maxReplans: 10,
		ask,
		clock,
		cache,
		describers,
		runSubgoal: ex.factory,
		log: () => {},
	});
	let turns = 0;
	const { end } = await drive(mission, cache, (call) => {
		turns++;
		// After the action the clock has 30 s left: a replan would need 40 s at least.
		if (turns === 2) clock.deadline_at = Date.now() + 30_000;
		return call.tool === "mm_observe"
			? { ok: true, frame: f, episode_over: false }
			: {
					ok: true,
					frame: f,
					segments: [{ ...GOAL.C6.blocked, label: call.label, outcome: "done" as const, abort_reason: undefined }],
					episode_over: false,
				};
	});
	assert.deepEqual([end.status, end.ended_on], ["timeout", "replan_refused_for_price"]);
	assert.equal(ex.inputs.length, 2);
	assert.match(ex.inputs[1].preface, /judged not_done: still above.*Do it differently/);
	const s = mission.state();
	assert.deepEqual(
		s.subgoals.map((a) => [a.attempt, a.never_moved]),
		[
			[1, false],
			[2, true],
		],
	);
	assert.equal(roleCalls.filter((c) => c.role === "planner").length, 1, "the replan was refused before any call");
	assert.ok(s.replan_refused_for_price);
});

test("mission: a lost grasp (holding true → false on a carry step) cancels the attempt and backs up to the grasp step without a replan", async () => {
	const held = frame(GOAL.C6.blocked, { holding: true, commanded: "close", clock: 1 });
	const dropped = frame(GOAL.C6.contact, { holding: false, commanded: "close", clock: 2 });
	const open = frame(GOAL.C6.contact, { holding: false, commanded: "open", clock: 3 });
	const plan3 = {
		...PLAN,
		subgoals: [
			PLAN.subgoals[0],
			{ ...PLAN.subgoals[0], name: "lift", subgoal: "Lift the bowl", criterion: "the bowl is held 120 mm higher" },
			PLAN.subgoals[1],
		],
	};
	const ex = scriptedExecutor((_input, n) => ({
		calls: [act("move up 50.0 mm (base)")],
		run: {
			ending: n === 0 ? "HELD" : n === 1 ? "CANCELLED" : "MODEL_DONE",
			held: n === 0 ? { gap_mm: 63 } : undefined,
		},
	}));
	// take (identity), lift (lost grasp: the whole task asked), take again, lift, put (released: the whole task asked), the task check
	const { ask } = scriptedAsk({ planner: [plan3], verify: [DONE, DONE, DONE, DONE, DONE, TASK_DONE, TASK_DONE] });
	const clock = missionClock();
	const cache: FrameCache = { stale: true };
	const entries: { entry: string; data: Json }[] = [];
	const mission = runMission({
		task: "t",
		budget: 10,
		budgetS: 360,
		maxReplans: 10,
		ask,
		clock,
		cache,
		describers,
		runSubgoal: ex.factory,
		log: (entry, data) => entries.push({ entry, data }),
	});
	let n = 0;
	const { end } = await drive(mission, cache, (call) => {
		n++;
		// Reply 3 is the lift's action: the bowl slips (the reading goes from holding to not holding); reply 6 is the put's action.
		const f = n === 3 ? dropped : n >= 6 ? open : held;
		return call.tool === "mm_observe"
			? { ok: true, frame: f, episode_over: false }
			: {
					ok: true,
					frame: f,
					segments: [{ ...GOAL.C6.blocked, label: call.label, outcome: "done" as const, abort_reason: undefined }],
					episode_over: false,
				};
	});
	const s = mission.state();
	assert.ok(
		s.subgoals.some((a) => a.name === "lift"),
		JSON.stringify({ end, subgoals: s.subgoals.map((a) => [a.name, a.verdict]) }),
	);
	const lift = s.subgoals.find((a) => a.name === "lift")!;
	assert.deepEqual([lift.verdict?.gate, lift.verdict?.status], ["lost_grasp", "not_done"]);
	assert.deepEqual(s.backtracked, ["the black bowl"]);
	// take, lift (lost), take again, lift, put
	assert.deepEqual(
		s.subgoals.map((a) => a.name),
		["take", "lift", "take", "lift", "put"],
		JSON.stringify(
			entries.map((e) => [e.entry, e.data.subgoal, e.data.status, e.data.gate, e.data.because, e.data.trigger]),
		),
	);
	assert.equal(s.replans, 0);
	assert.equal(end.status, "task_done", JSON.stringify(end));
});

test("mission: the episode ending mid-plan judges the scene as it is and ends exhausted; an executor error ends in error", async () => {
	const f = frame(GOAL.C6.blocked, { holding: false });
	const over = frame(GOAL.C6.blocked, { holding: false, episode_over: true });
	const ex = scriptedExecutor(() => ({ calls: [act("move down 50.0 mm (base)")], run: { ending: "EPISODE_OVER" } }));
	const NOT = { status: "not_done", because: "no", evidence: "", identity: "", replan: false, clauses: [] };
	const c1: FrameCache = { stale: true };
	const m1 = runMission({
		task: "t",
		budget: 10,
		budgetS: 360,
		maxReplans: 10,
		ask: scriptedAsk({ planner: [PLAN], verify: [NOT, NOT] }).ask,
		clock: missionClock(),
		cache: c1,
		describers,
		runSubgoal: ex.factory,
		log: () => {},
	});
	const r1 = await drive(m1, c1, (call) =>
		call.tool === "mm_observe"
			? { ok: true, frame: f, episode_over: false }
			: { ok: true, frame: over, segments: [], episode_over: true },
	);
	assert.deepEqual([r1.end.status, r1.end.ended_on], ["exhausted", "episode_over"]);
	const broken = scriptedExecutor(() => ({ calls: [], run: { ending: "ERROR", error: "step error: boom" } }));
	const c2: FrameCache = { stale: true };
	const m2 = runMission({
		task: "t",
		budget: 10,
		budgetS: 360,
		maxReplans: 10,
		ask: scriptedAsk({ planner: [PLAN] }).ask,
		clock: missionClock(),
		cache: c2,
		describers,
		runSubgoal: broken.factory,
		log: () => {},
	});
	const r2 = await drive(m2, c2, () => ({ ok: true, frame: f, episode_over: false }));
	assert.deepEqual([r2.end.status, r2.end.summary], ["error", "step error: boom"]);
	// A first plan that fails ends the mission in error without any robot call beyond the opening look.
	const c3: FrameCache = { stale: true };
	const m3 = runMission({
		task: "t",
		budget: 10,
		budgetS: 360,
		maxReplans: 10,
		ask: scriptedAsk({ planner: ["NO_JSON", "NO_JSON", "NO_JSON"] }).ask,
		clock: missionClock(),
		cache: c3,
		describers,
		runSubgoal: broken.factory,
		log: () => {},
	});
	const r3 = await drive(m3, c3, () => ({ ok: true, frame: f, episode_over: false }));
	assert.deepEqual([r3.end.status, r3.end.ended_on, r3.calls.length], ["error", "plan_failed", 1]);
});

// ---------------------------------------------------------------------------
// the provider

type Handler = (event: Json, ctx: Json) => unknown;
/** A stub pi, registry and robot handle around the provider; role replies are scripted by the prompt's role. */
function fakeProvider(o: {
	flags?: Record<string, unknown>;
	argv?: string[];
	replies: (role: "plan" | "replan" | "verify" | "other", n: number) => unknown;
	executor?: (input: SubgoalInput) => Executor;
	model?: Json;
}) {
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = { mm: "true", ...o.flags };
	const providers: {
		id: string;
		getModels: () => Model<"openai-completions">[];
		streamSimple: (m: unknown, c: unknown, opt?: unknown) => ReturnType<typeof createAssistantMessageEventStream>;
	}[] = [];
	const entries: { type: string; data: Json }[] = [];
	const listeners = new Map<string, ((d: unknown) => void)[]>();
	const notices: string[] = [];
	let active: string[] | undefined;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, f: { default?: unknown }) => {
			if (!(name in flags)) flags[name] = f.default;
		},
		getFlag: (name: string) => flags[name],
		registerProvider: (p: (typeof providers)[number]) => providers.push(p),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active ?? [],
		appendEntry: (type: string, data: Json) => entries.push({ type, data }),
		events: {
			emit: (channel: string, data: unknown) => {
				for (const fn of listeners.get(channel) ?? []) fn(data);
			},
			on: (channel: string, fn: (d: unknown) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
	} as unknown as ExtensionAPI;
	const BASE = "selfhost/q";
	const base: Model<"openai-completions"> = {
		id: "q",
		provider: "selfhost",
		api: "openai-completions",
		name: "q",
		baseUrl: "",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 32_000,
		maxTokens: 4096,
	};
	const counts = { plan: 0, replan: 0, verify: 0, other: 0 };
	const roleCalls: string[] = [];
	const registry = {
		find: (provider: string, id: string) => (`${provider}/${id}` === BASE ? base : undefined),
		streamSimple: (m: Model<"openai-completions">, c: { systemPrompt?: string }, opt?: { signal?: AbortSignal }) => {
			const role = /^You plan/.test(c.systemPrompt ?? "")
				? "plan"
				: /^You replan/.test(c.systemPrompt ?? "")
					? "replan"
					: /^You judge/.test(c.systemPrompt ?? "")
						? "verify"
						: "other";
			roleCalls.push(role);
			const value = o.replies(role, counts[role]++);
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [{ type: "text", text: JSON.stringify(value) }],
				api: m.api,
				provider: m.provider,
				model: m.id,
				usage: {
					input: 100,
					output: 10,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 110,
					cost: { input: 0.01, output: 0.01, cacheRead: 0, cacheWrite: 0, total: 0.02 },
				},
				stopReason: "stop",
				timestamp: 1,
			};
			queueMicrotask(() => {
				if (opt?.signal?.aborted) {
					message.stopReason = "aborted";
					stream.push({ type: "error", reason: "aborted", error: message });
				} else stream.push({ type: "done", reason: "stop", message });
				stream.end();
			});
			return stream;
		},
	};
	const argv = o.argv ?? ["node", "pi", "--model", `motormind/${BASE}`];
	const executor =
		o.executor ??
		scriptedExecutor((_i, n) => ({
			calls: [act("move down 50.0 mm (base)")],
			run: { ending: n === 0 ? "HELD" : "MODEL_DONE", held: n === 0 ? { gap_mm: 63 } : undefined },
		})).factory;
	const prov = motormindProvider(pi, argv, { runSubgoal: executor, describers });
	let latest: Observation = observation(GOAL.C6.blocked, { holding: true, commanded: "close" });
	const handleCalls: string[] = [];
	const handle: MotorMindHandle = {
		robot: "libero",
		observe: async () => ({}),
		runStep: async () => ({}),
		endpoint: () => undefined,
		cancel: (reason) => handleCalls.push(`cancel:${reason}`),
		clearCancel: () => handleCalls.push("clear"),
		setDeadline: (at) => handleCalls.push(`deadline:${at === undefined ? "none" : "set"}`),
		latest: () => latest,
		task: () => "put the black bowl on the plate",
	};
	const ctx = {
		hasUI: true,
		cwd: "/",
		ui: { notify: (m: string) => notices.push(m) },
		shutdown: () => {},
		model: o.model ?? { provider: "motormind", id: BASE, contextWindow: 128_000, maxTokens: 16_384 },
		modelRegistry: registry,
		sessionManager: { getBranch: () => [] },
	};
	const emit = async (name: string, event: Json = {}) => {
		let result: unknown;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	};
	pi.events.emit(MOTORMIND_EVENT, handle);
	const turn = async (messages: Message[], signal?: AbortSignal) => {
		const provider = providers.find((p) => p.id === "motormind")!;
		const stream = provider.streamSimple(provider.getModels()[0], { messages }, { signal });
		for await (const _ of stream);
		return (await stream.result()) as AssistantMessage;
	};
	/** The tool result pi would append for the message's tool call, from the fake robot. */
	const resultFor = (m: AssistantMessage, body: Json, isError = false): Message => {
		const call = m.content.find((c) => c.type === "toolCall")!;
		return {
			role: "toolResult",
			toolCallId: call.id,
			toolName: call.name,
			content: [{ type: "text", text: JSON.stringify(body) }],
			isError,
			timestamp: 2,
		};
	};
	const setLatest = (obs: Observation) => {
		latest = obs;
	};
	return {
		pi,
		prov,
		emit,
		turn,
		resultFor,
		entries,
		notices,
		roleCalls,
		handleCalls,
		active: () => active,
		setLatest,
		ctx,
	};
}
const USER: Message = { role: "user", content: [{ type: "text", text: "Solve the task." }], timestamp: 1 };
const callOf = (m: AssistantMessage) => m.content.find((c) => c.type === "toolCall")!;

test("provider: off without --model motormind/...; motormindArgs; plannerOf counts a motormind episode", () => {
	const f = fakeProvider({ argv: ["node", "pi", "--model", "selfhost/q"], replies: () => ({}) });
	assert.equal(f.prov, undefined);
	assert.deepEqual(motormindArgs(["--model=motormind/selfhost/q"]), { base: "selfhost/q" });
	assert.deepEqual(motormindArgs(["--model", "selfhost/q"]), { base: undefined });
	assert.equal(plannerOf(["motormind", "motormind"]), "motormind");
	assert.ok(
		isClosedLoopRefusal(
			"mm_act refused: the last motion (mm_act) has an unknown outcome (timed out). Re-observe with mm_observe before issuing another motion.",
		),
	);
	assert.ok(!isClosedLoopRefusal("mm_act refused, nothing moved: distance_mm must be positive"));
});

test("provider: one tool call per turn; the same result twice advances the mission once; a foreign context gets the static text with zero usage", async () => {
	let verdicts = 0;
	const f = fakeProvider({
		replies: (role) => (role === "plan" ? PLAN : role === "verify" ? (++verdicts >= 3 ? TASK_DONE : DONE) : {}),
	});
	assert.ok(f.prov);
	await f.emit("session_start");
	assert.deepEqual(f.notices, []);
	// turn 1: the opening message → the mission's first look
	const m1 = await f.turn([USER]);
	assert.equal(m1.stopReason, "toolUse", m1.errorMessage);
	assert.equal(callOf(m1).name, "mm_observe");
	assert.equal(m1.usage.cost.total, 0, "no role ran before the opening look");
	const seen = f.resultFor(m1, { obs_id: 1 });
	// turn 2: the look's result → the planner runs (one call) and the executor's first action goes out
	const m2 = await f.turn([USER, m1, seen]);
	assert.equal(m2.stopReason, "toolUse", m2.errorMessage);
	assert.equal(callOf(m2).name, "mm_act");
	assert.equal(m2.usage.cost.total, 0.02, "the planner's one call");
	assert.deepEqual([m2.usage.input, m2.usage.output], [100, 10], "the last call's tokens");
	assert.equal(f.roleCalls.length, 1);
	// pi retries with the same context: the cached message, no new role call, the mission did not advance
	const again = await f.turn([USER, m1, seen]);
	assert.equal(callOf(again).id, callOf(m2).id);
	assert.deepEqual(callOf(again).arguments, callOf(m2).arguments);
	assert.equal(f.roleCalls.length, 1);
	const first = await f.turn([USER]);
	assert.equal(callOf(first).id, callOf(m1).id, "the first turn retried");
	// a compaction-shaped context (the last message is not the pending result): static text, zero usage
	const odd = await f.turn([USER, m1, seen, m2]);
	assert.equal(odd.stopReason, "stop");
	assert.deepEqual(odd.content, [{ type: "text", text: STATIC_TEXT }]);
	assert.equal(odd.usage.totalTokens, 0);
	assert.equal(f.roleCalls.length, 1);
	// compaction is cancelled while this provider plays
	assert.deepEqual(await f.emit("session_before_compact"), { cancel: true });
	// turn 3: the action's result → the verdict reads the frame the action returned (nothing moved since),
	// the take is done, and the put's first action goes out: one turn, several role calls, one tool call
	const acted = f.resultFor(m2, {
		action: "move down 50.0 mm (base)",
		segments: [{ ...GOAL.C6.blocked, label: "move down 50.0 mm (base)", outcome: "done" }],
		episode_over: false,
	});
	const m3 = await f.turn([USER, m1, seen, m2, acted]);
	assert.equal(m3.stopReason, "toolUse", m3.errorMessage);
	assert.equal(callOf(m3).name, "mm_act");
	assert.ok(m3.usage.cost.total >= 0.02 && f.roleCalls.length >= 2, "the verifier ran inside the turn");
	assert.equal(m3.usage.cost.total, 0.02 * (f.roleCalls.length - 1), "the turn's cost is every call's");
	const turns = f.entries.filter((e) => e.type === TURN_ENTRY);
	assert.equal(turns.length, 3);
	assert.deepEqual(
		turns.map((e) => (e.data.tool_call as Json).name),
		["mm_observe", "mm_act", "mm_act"],
	);
	assert.equal(
		f.handleCalls.filter((c) => c === "deadline:set").length > 0,
		true,
		"the mission deadline reached the robot",
	);
	const res = f.prov.result();
	assert.equal(res.turns, 3);
	assert.equal(res.plans, 1);
	assert.ok((res.by_role as Json).planner);
	assert.ok((res.prompts as Json).plan, "the plan prompt's hash");
	await f.emit("agent_end", { messages: [] });
	assert.equal(f.entries.filter((e) => e.type === END_ENTRY).length, 1);
});

test("provider: a closed-loop refusal is reconciled with mm_observe and the refused call reissued; the mission ends in finish", async () => {
	let verdicts = 0;
	const f = fakeProvider({
		replies: (role) => (role === "plan" ? PLAN : role === "verify" ? (++verdicts >= 3 ? TASK_DONE : DONE) : {}),
	});
	await f.emit("session_start");
	const m1 = await f.turn([USER]);
	const seen = f.resultFor(m1, {});
	const m2 = await f.turn([USER, m1, seen]);
	assert.equal(callOf(m2).name, "mm_act");
	const refused = f.resultFor(m2, { error: "x" }, true);
	(refused as Extract<Message, { role: "toolResult" }>).content = [
		{
			type: "text",
			text: "mm_act refused: the last motion (mm_act) has an unknown outcome (timed out). Re-observe with mm_observe before issuing another motion.",
		},
	];
	const m3 = await f.turn([USER, m1, seen, m2, refused]);
	assert.equal(callOf(m3).name, "mm_observe");
	assert.equal((callOf(m3).arguments as Json).tracked_mm, undefined);
	const observed = f.resultFor(m3, {});
	const m4 = await f.turn([USER, m1, seen, m2, refused, m3, observed]);
	assert.equal(callOf(m4).name, "mm_act");
	assert.deepEqual(callOf(m4).arguments, callOf(m2).arguments, "the same call again");
	assert.equal(f.roleCalls.length, 1, "no role ran during the reconciliation");
	// Play it out: the action, the look, the second subgoal, its look, the task check's look → finish.
	const history: Message[] = [USER, m1, seen, m2, refused, m3, observed, m4];
	let m = m4;
	for (let i = 0; i < 6 && callOf(m).name !== "finish"; i++) {
		const body =
			callOf(m).name === "mm_act"
				? { segments: [{ ...GOAL.C6.blocked, outcome: "done" }], episode_over: false }
				: {};
		if (i === 2) f.setLatest(observation(GOAL.C6.contact, { holding: false, commanded: "open" }));
		const r = f.resultFor(m, body);
		history.push(r);
		m = await f.turn(history);
		history.push(m);
	}
	assert.equal(callOf(m).name, "finish", JSON.stringify(m.content));
	assert.deepEqual((callOf(m).arguments as Json).status, "success");
	assert.match(String((callOf(m).arguments as Json).summary), /^task_done/);
	// After the finish, pi's result for it gets the static text.
	const after = await f.turn([...history, f.resultFor(m, {})]);
	assert.deepEqual(after.content, [{ type: "text", text: STATIC_TEXT }]);
	assert.equal(f.prov!.result().status, "task_done");
});

test("provider: fails closed at session start without --mm or with a base model the registry lacks", async () => {
	const f = fakeProvider({ flags: { mm: "false" }, replies: () => ({}) });
	await f.emit("session_start");
	assert.deepEqual(f.active(), []);
	assert.match(f.notices[0], /needs --mm=true/);
	const m = await f.turn([USER]);
	assert.equal(m.stopReason, "error");
	const g = fakeProvider({
		argv: ["node", "pi", "--model", "motormind/nowhere/x"],
		replies: () => ({}),
		model: { provider: "motormind", id: "nowhere/x" },
	});
	await g.emit("session_start");
	assert.match(g.notices[0], /unknown model nowhere\/x/);
	// Another session model: the provider does not check anything.
	const h = fakeProvider({ flags: { mm: "false" }, replies: () => ({}), model: { provider: "selfhost", id: "q" } });
	await h.emit("session_start");
	assert.deepEqual(h.notices, []);
	assert.equal(h.active(), undefined);
});

test("provider: replyOf reads the tool result and the robot's latest observation; classify tells the opening, the pending result and the rest apart", () => {
	const handle = { latest: () => observation(GOAL.C7, { images: false }) } as unknown as MotorMindHandle;
	const result: Message = {
		role: "toolResult",
		toolCallId: "motormind-2",
		toolName: "mm_act",
		content: [
			{
				type: "text",
				text: JSON.stringify({
					segments: [GOAL.C7],
					cancelled: { reason: "r", at_segment: 1 },
					timed_out: true,
					episode_over: false,
				}),
			},
		],
		isError: false,
		timestamp: 1,
	};
	const r = replyOf(result as Extract<Message, { role: "toolResult" }>, handle);
	assert.ok(r.ok && r.frame && !hasImages(r.frame));
	assert.equal(r.segments?.[0].abort_reason, "skidded");
	assert.deepEqual(r.cancelled, { reason: "r", at_segment: 1 });
	assert.equal(r.timed_out, true);
	const none = new Set<string>();
	assert.equal(classify([USER], undefined, none, false), "opening");
	assert.equal(classify([USER], undefined, none, true), "answered", "pi retrying the first turn");
	assert.equal(classify([USER, result, USER], undefined, none, true), "other", "an interactive message");
	assert.equal(classify([USER, result], { id: "motormind-2" }, none, true), "pending_result");
	assert.equal(classify([USER, result], { id: "motormind-1" }, none, true), "other");
	assert.equal(classify([USER, result], { id: "motormind-1" }, new Set(["motormind-2"]), true), "answered");
	assert.equal(classify([], undefined, none, false), "other");
	// The prompt files carry the separator the planner and verifier split on.
	for (const name of ["plan", "replan", "verify"]) assert.match(promptFile(name).text, /\n=+ user =+\n/);
});
