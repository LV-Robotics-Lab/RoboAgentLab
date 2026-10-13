/**
 * The contract between the two halves of the MotorMind mission (pi-embodied-work/docs/motormind/
 * m2-contract.md): the mission (./mission.ts, ./planner.ts, ./verify.ts, played by ./provider.ts) and
 * the executor (./executor.ts and its gates, locate, observe). Types, and the few pure helpers both
 * halves need: a frame from the robot's observation, the images of a frame, the cached look, and the
 * robot's description.
 *
 * The protocol (DESIGN §1.4): the mission is an async generator that yields ONE robot call (`MmCall`)
 * and is resumed with that call's result (`MmReply`); the executor is a generator the mission delegates
 * to with `yield*`. Role calls go through one `Ask`. Neither generator calls RPCs, tools or pi.
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import type { Observation } from "./index.ts";
import type { Caps, Delta, DeltaKind, Grasp, Mat3, Pose, Vec3 } from "./types.ts";

export type Json = Record<string, unknown>;

// ---------------------------------------------------------------------------
// robot calls

export type ObserveWhy =
	| "opening"
	| "cycle_start"
	| "before_action"
	| "after_look"
	| "after_stop"
	| "before_verdict"
	| "task_check"
	| "reconcile";
export type ObserveArgs = { tracked_mm?: number };
/** The `mm_act` tool's parameters (../motormind/index.ts ACT_PARAMS). */
export type ActArgs = {
	type: "move" | "rotate" | "gripper" | "home" | "wait";
	direction?: string;
	axis?: Vec3;
	distance_mm?: number;
	angle_deg?: number;
	speed_mm_s?: number;
	rot_speed_deg_s?: number;
	stop_on_contact?: boolean;
	state?: "open" | "close";
	width_mm?: number;
	seconds?: number;
	frame?: string;
	note?: string;
	label?: string;
	open_to_mm?: number;
	tracked_mm?: number;
};
export type MmCall =
	| { tool: "mm_observe"; args: ObserveArgs; why: ObserveWhy }
	| { tool: "mm_act"; args: ActArgs; delta?: Delta; label: string };

/** One segment as M1's `report(Measured)` writes it into the tool result (mm, deg). */
export type SegmentReport = {
	label: string;
	kind: DeltaKind;
	outcome: "done" | "contact" | "limited" | "refused" | "skipped" | "failed";
	abort_reason?: "blocked" | "skidded";
	moved_mm: number;
	along_axis_mm: number;
	lateral_mm: number;
	turned_deg: number;
	axis?: Vec3;
	ticks: number;
	elapsed_s: number;
	frame_clock_s?: number;
	detail?: { opening_mm?: number; asked_mm?: number };
	grasp?: Grasp & { opening_mm?: number };
	pose_after?: { position_mm: Vec3; rotation: Mat3 };
	message?: string;
	reason?: string;
	cancel_reason?: string;
	error?: string;
	timed_out?: boolean;
	/** Added by the executor (spec-executor §5.6): a shortened or inserted action's note. */
	note?: string;
};

export type MmReply = {
	ok: boolean;
	error?: string;
	body?: Json;
	frame?: Frame;
	segments?: SegmentReport[];
	cancelled?: { reason: string; at_segment: number; latency_s?: number };
	timed_out?: boolean;
	episode_over: boolean;
};

/** The prefix of an `mm_act` error that means the command was refused and nothing moved (a bad proposal). */
export const ACT_REFUSED = "mm_act refused, nothing moved:";

// ---------------------------------------------------------------------------
// frames

export type CameraName = "scene" | "side" | "wrist";
export const CAMERA_NAMES: readonly CameraName[] = ["scene", "side", "wrist"];
export type Camera = {
	name: CameraName;
	width: number;
	height: number;
	intrinsic: Mat3;
	cam2base: number[][];
	/** base64 JPEG */
	marked768?: string;
	marked512?: string;
	raw512?: string;
	aim_px?: [number, number] | null;
	depth_min_range_m?: number;
	timestamp?: number;
};
export type Clearance = {
	seen: boolean;
	mm: number | null;
	median_mm?: number | null;
	surface_z_m?: number | null;
	at_range_floor?: boolean;
	near_min_range?: boolean;
	jaws_down?: boolean;
	reason?: string;
};
export type Frame = {
	obs_id?: number;
	frame_clock_s: number;
	episode_id?: number;
	pose: Pose;
	base_pose?: Pose;
	caps: Caps;
	table_z_m: number | null;
	gripper: { opening_mm: number; commanded: "open" | "close" | "hold"; moving: boolean };
	grasp: Grasp & { opening_mm?: number };
	clearance?: Clearance;
	cameras: Partial<Record<CameraName, Camera>>;
	camera_names: CameraName[];
	episode_over: boolean;
	terminated: boolean;
	truncated: boolean;
	degraded: Json;
	mosaic?: ImageContent;
};

const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown, d = 0) => (typeof v === "number" && Number.isFinite(v) ? v : d);
const isMat3 = (v: unknown): v is Mat3 =>
	Array.isArray(v) && v.length === 3 && v.every((r) => Array.isArray(r) && r.length === 3);

function cameraOf(name: CameraName, v: unknown): Camera | undefined {
	if (!isObject(v) || !isMat3(v.intrinsic) || !Array.isArray(v.cam2base)) return undefined;
	const str = (k: string) => (typeof v[k] === "string" && v[k] ? (v[k] as string) : undefined);
	return {
		name,
		width: num(v.width, 768),
		height: num(v.height, 768),
		intrinsic: v.intrinsic,
		cam2base: v.cam2base as number[][],
		...(str("marked768") ? { marked768: str("marked768") } : {}),
		...(str("marked512") ? { marked512: str("marked512") } : {}),
		...(str("raw512") ? { raw512: str("raw512") } : {}),
		...(Array.isArray(v.aim_px) && v.aim_px.length === 2 ? { aim_px: [num(v.aim_px[0]), num(v.aim_px[1])] } : {}),
		...(typeof v.depth_min_range_m === "number" ? { depth_min_range_m: v.depth_min_range_m } : {}),
		...(typeof v.timestamp === "number" ? { timestamp: v.timestamp } : {}),
	};
}

/** The mode's observation as a `Frame`, or undefined without a tool pose. */
export function frameOf(o: Observation): Frame | undefined {
	if (!o.pose) return undefined;
	const r = o.raw;
	const g = isObject(r.gripper) ? r.gripper : {};
	const grasp = isObject(r.grasp) ? r.grasp : {};
	const cameras: Partial<Record<CameraName, Camera>> = {};
	if (isObject(r.cameras))
		for (const name of CAMERA_NAMES) {
			const c = cameraOf(name, r.cameras[name]);
			if (c) cameras[name] = c;
		}
	const names = Array.isArray(r.camera_names)
		? (r.camera_names.filter((n) => (CAMERA_NAMES as readonly string[]).includes(String(n))) as CameraName[])
		: (Object.keys(cameras) as CameraName[]);
	const commanded = g.commanded === "close" || g.commanded === "hold" ? g.commanded : "open";
	return {
		...(o.obs_id !== undefined ? { obs_id: o.obs_id } : {}),
		frame_clock_s: o.frame_clock_s ?? 0,
		...(typeof r.episode_id === "number" ? { episode_id: r.episode_id } : {}),
		pose: o.pose,
		...(o.base_pose ? { base_pose: o.base_pose } : {}),
		caps: o.caps,
		table_z_m: o.table_z_m,
		gripper: { opening_mm: num(g.opening_mm), commanded, moving: g.moving === true },
		grasp: {
			holding: grasp.holding === true ? true : grasp.holding === false ? false : null,
			reason: typeof grasp.reason === "string" ? grasp.reason : "",
			...(typeof grasp.closed_fraction === "number" ? { closed_fraction: grasp.closed_fraction } : {}),
			...(typeof grasp.opening_mm === "number" ? { opening_mm: grasp.opening_mm } : {}),
		},
		...(isObject(r.clearance)
			? {
					clearance: {
						seen: r.clearance.seen === true,
						mm: typeof r.clearance.mm === "number" ? r.clearance.mm : null,
						...(r.clearance.median_mm !== undefined ? { median_mm: r.clearance.median_mm as number | null } : {}),
						...(r.clearance.surface_z_m !== undefined
							? { surface_z_m: r.clearance.surface_z_m as number | null }
							: {}),
						...(typeof r.clearance.at_range_floor === "boolean" ? { at_range_floor: r.clearance.at_range_floor } : {}),
						...(typeof r.clearance.near_min_range === "boolean" ? { near_min_range: r.clearance.near_min_range } : {}),
						...(typeof r.clearance.jaws_down === "boolean" ? { jaws_down: r.clearance.jaws_down } : {}),
						...(typeof r.clearance.reason === "string" ? { reason: r.clearance.reason } : {}),
					},
				}
			: {}),
		cameras,
		camera_names: names,
		episode_over: o.episode_over,
		terminated: o.terminated,
		truncated: o.truncated,
		degraded: o.degraded,
		...(o.mosaic ? { mosaic: { type: "image", data: o.mosaic.data.toString("base64"), mimeType: o.mosaic.mime } } : {}),
	};
}

/** Whether the frame carries camera images (a numbers-only observation does not). */
export const hasImages = (f: Frame | undefined): f is Frame =>
	f !== undefined && Object.values(f.cameras).some((c) => c?.marked768 || c?.marked512 || c?.raw512);

/** The frame's images at one size, in camera order (`names`, default every camera the frame has). */
export function images(
	f: Frame,
	size: "marked768" | "marked512" | "raw512",
	names: readonly CameraName[] = f.camera_names,
): ImageContent[] {
	return names.flatMap((n) => {
		const data = f.cameras[n]?.[size];
		return data ? [{ type: "image" as const, data, mimeType: "image/jpeg" }] : [];
	});
}

// ---------------------------------------------------------------------------
// the cached look

/** The latest frame the robot reported, and whether a new render is needed before it is reused. */
export type FrameCache = { latest?: Frame; stale: boolean };

/**
 * A look (DESIGN §1.4 step 3): the cached frame when nothing moved since it was rendered, else one
 * `mm_observe`. `observes` records both kinds (C19). Returns undefined when the robot answered with an
 * error (the caller treats it as a step error).
 */
export async function* look(
	cache: FrameCache,
	why: ObserveWhy,
	observes: { why: ObserveWhy; yielded: boolean }[],
	tracked_mm?: number,
): AsyncGenerator<MmCall, Frame | undefined, MmReply> {
	if (!cache.stale && hasImages(cache.latest)) {
		observes.push({ why, yielded: false });
		return cache.latest;
	}
	observes.push({ why, yielded: true });
	const reply = yield { tool: "mm_observe", args: tracked_mm !== undefined ? { tracked_mm } : {}, why };
	return reply.ok ? reply.frame : undefined;
}

// ---------------------------------------------------------------------------
// role calls

export type Role = "planner" | "verify" | "executor" | "locate" | "supervise" | "monitor" | "memory";
export const ROLES: readonly Role[] = ["planner", "verify", "executor", "locate", "supervise", "monitor", "memory"];
export type RolePrompt = { system: string; text: string };
export type RoleUsage = { cost_usd: number; tokens_in: number; tokens_out: number; latency_s: number };
export type RoleErrorKind = "transport" | "timeout" | "aborted" | "no_json" | "empty" | "unknown_model";
export type RoleReply =
	| { ok: true; text: string; json: unknown; thinking?: string; model: string; usage: RoleUsage; attempts: number }
	| {
			ok: false;
			error: string;
			error_kind: RoleErrorKind;
			text?: string;
			model: string;
			usage: RoleUsage;
			attempts: number;
	  };
export type Ask = (
	role: Role,
	prompt: RolePrompt,
	images: ImageContent[],
	o?: { signal?: AbortSignal; maxTokens?: number },
) => Promise<RoleReply>;
/** One role call as the entries record it (DESIGN §7.2 motormind_turn). */
export type RoleCallRecord = {
	role: Role;
	model: string;
	ok: boolean;
	error_kind?: RoleErrorKind;
	attempts: number;
	cost_usd: number;
	tokens_in: number;
	tokens_out: number;
	latency_s: number;
	gate?: string;
};
export const recordOf = (role: Role, r: RoleReply, gate?: string): RoleCallRecord => ({
	role,
	model: r.model,
	ok: r.ok,
	...(r.ok ? {} : { error_kind: r.error_kind }),
	attempts: r.attempts,
	cost_usd: r.usage.cost_usd,
	tokens_in: r.usage.tokens_in,
	tokens_out: r.usage.tokens_out,
	latency_s: r.usage.latency_s,
	...(gate ? { gate } : {}),
});

// ---------------------------------------------------------------------------
// cancel and deadline

export type Cancel = {
	by: "monitor" | "deadline" | "grasp_watchdog" | "release_watchdog" | "mission";
	reason: string;
};
export type MissionClock = {
	deadline_at?: number;
	left_s(): number;
	cancelled(): Cancel | undefined;
	cancel(c: Cancel): void;
	clear(): void;
};

/** A clock: `deadline_at` is set by the mission when the first planner call starts; `onCancel` mirrors to the robot. */
export function missionClock(onCancel?: (c: Cancel | undefined) => void): MissionClock {
	let cancel: Cancel | undefined;
	const clock: MissionClock = {
		deadline_at: undefined,
		left_s: () => (clock.deadline_at === undefined ? Infinity : (clock.deadline_at - Date.now()) / 1000),
		cancelled: () => cancel ?? (clock.left_s() <= 0 ? { by: "deadline", reason: "the mission clock ran out" } : undefined),
		cancel: (c) => {
			if (cancel) return;
			cancel = c;
			onCancel?.(c);
		},
		clear: () => {
			cancel = undefined;
			onCancel?.(undefined);
		},
	};
	return clock;
}

// ---------------------------------------------------------------------------
// executor input and result

/** A box in 0..1000 normalized coordinates, top-left origin: [x0, y0, x1, y1]. */
export type Box = [number, number, number, number];
export type Subgoal = {
	name: string;
	subgoal: string;
	criterion: string;
	target: string;
	target_boxes: Partial<Record<CameraName, Box>>;
	clause: number;
	max_cycles: number;
	gripper_only: boolean;
};
/** What the executor carries from one subgoal to the next (./carry.ts owns the keys; the mission passes it through). */
export type Carry = Record<string, unknown>;

export type Ending =
	| "HELD"
	| "RELEASED_IN_CODE"
	| "ARRIVED"
	| "WALL"
	| "FROZEN"
	| "MODEL_DONE"
	| "DONE_AFTER_LAST"
	| "CANNOT_FROM_THE_SIDE"
	| "ERROR"
	| "CANCELLED"
	| "CYCLES_EXHAUSTED"
	| "EPISODE_OVER";

export type Timing = { wall_s: number; motion_s: number; calls: RoleCallRecord[] };

export type Proposal = {
	assessment: string;
	done: boolean;
	expect: string;
	confidence: number;
	command: unknown;
	deltas: { kind: DeltaKind; label: string; axis?: Vec3; push: boolean }[];
};
export type Located = {
	label?: string;
	point_mm?: Vec3;
	error_mm?: number;
	doubtful: boolean;
	error?: string;
};
export type Reach = { target: string; from_plan_boxes: boolean; residual_mm?: number; located?: Located };
export type Supervision = { decision: "continue" | "stop"; finding: string; because: string };
export type Release = {
	payload_width_mm?: number;
	allowed_mm?: number;
	offset_mm?: number;
	placing_in?: boolean;
	receiver?: string;
};
export type Cycle = {
	index: number;
	proposal?: Proposal;
	steps: SegmentReport[];
	reach?: Reach;
	batch?: {
		actions: number;
		ran: number;
		stopped_at?: number;
		reason?: string;
		residual_mm?: number;
		motion_s: number;
	};
	supervision?: Supervision;
	stopped_by?: string;
	finding?: string;
	release?: Release;
	error?: string;
	timing: Timing;
};

export type ExecutorEvent =
	| { type: "carried"; text: string }
	| { type: "observed"; why: ObserveWhy; frame_clock_s: number }
	| { type: "episode_over" }
	| { type: "held"; gap_mm: number; target_mm?: number }
	| { type: "proposed"; cycle: number; proposal: Proposal }
	| { type: "reached"; cycle: number; reach: Reach }
	| { type: "mended"; what: string }
	| { type: "batch_stopped"; cycle: number; at: number; reason: string }
	| { type: "step_start"; label: string }
	| { type: "step_end"; cycle: number; step: SegmentReport }
	| { type: "step_error"; error: string }
	| { type: "supervision"; cycle: number; supervision: Supervision }
	| { type: "limited"; why: string }
	| { type: "refused"; why: string }
	| { type: "frozen"; text: string }
	| { type: "letting_go"; text: string }
	| { type: "timing"; cycle: number; timing: Timing };

export type SubgoalInput = {
	task: string;
	subgoal: Subgoal;
	preface: string;
	carry?: Carry;
	maxCycles: number;
	ask: Ask;
	clock: MissionClock;
	cache: FrameCache;
	onEvent: (e: ExecutorEvent) => void;
};
export type SubgoalRun = {
	cycles: Cycle[];
	ending: Ending;
	ending_text: string;
	error?: string;
	episode_over: boolean;
	carry: Carry;
	held?: { gap_mm: number; target_mm?: number };
	observes: { why: ObserveWhy; yielded: boolean }[];
	timing: Timing;
};
export type Executor = AsyncGenerator<MmCall, SubgoalRun, MmReply>;

/** Text the executor half renders and the mission half injects into its prompts (./observe.ts). */
export type Describers = {
	legend: (f: Frame) => string;
	capabilities: (caps: Caps) => string;
};

// ---------------------------------------------------------------------------
// mission result

export type MissionStatus = "task_done" | "aborted" | "exhausted" | "timeout" | "gave_up" | "error" | "cancelled";
export type MissionEnd = { status: MissionStatus; ended_on: string; summary: string };
export type Mission = AsyncGenerator<MmCall, MissionEnd, MmReply>;

// ---------------------------------------------------------------------------
// the robot's description (spec-executor §8.2 block 1), shared by every prompt that shows the robot

const DOWN_WITHIN_DEG = 45;
const r0 = (v: number) => v.toFixed(0);
const r1 = (v: number) => v.toFixed(1);

/** The tool's approach axis (its +Z column) and how far it is from pointing straight down, deg. */
export function approach(pose: Pose): { axis: Vec3; off_down_deg: number } {
	const R = pose.rotation;
	const axis: Vec3 = [R[0][2], R[1][2], R[2][2]];
	const cos = Math.max(-1, Math.min(1, -axis[2]));
	return { axis, off_down_deg: (Math.acos(cos) * 180) / Math.PI };
}
export const jawsDown = (pose: Pose) => approach(pose).off_down_deg <= DOWN_WITHIN_DEG;

/** The horizontal direction word the approach axis leans toward most (MotorMind's own table: left = +Y). */
function leansToward(axis: Vec3): string {
	const words: [string, Vec3][] = [
		["forward", [1, 0, 0]],
		["backward", [-1, 0, 0]],
		["left", [0, 1, 0]],
		["right", [0, -1, 0]],
	];
	let best = words[0];
	for (const w of words) if (w[1][0] * axis[0] + w[1][1] * axis[1] > best[1][0] * axis[0] + best[1][1] * axis[1]) best = w;
	return best[0];
}

/** What the executor's clearance tracker adds when the wrist sees nothing (./observe.ts). */
export type TrackedClearance = { mm: number; descended_mm: number; lateral_mm: number };

/**
 * The robot in words: where the tool is (base frame, mm), whether the jaws point down, the clearance
 * under them (measured, tracked, unknown), and the gripper (opening, state, holding).
 */
export function describeRobot(f: Frame, tracked?: TrackedClearance): string {
	const p = f.pose.position_m.map((v) => r0(v * 1000));
	const a = approach(f.pose);
	const down = a.off_down_deg <= DOWN_WITHIN_DEG;
	const lines = [
		`Tool point at [${p.join(", ")}] mm in the base frame (forward +X, left +Y, up +Z). The jaws point ${
			down
				? "straight down"
				: `${r0(a.off_down_deg)} deg off vertical, leaning ${leansToward(a.axis)}`
		}.`,
	];
	const c = f.clearance;
	if (!f.caps.gripper_gap_measured && !c) lines.push("Clearance under the tool: not supported on this robot.");
	else if (c?.seen && c.mm !== null)
		lines.push(
			`${down ? "Clearance under the tool" : "Clearance ahead of the jaws"}: ${r0(c.mm)} mm (measured now${
				c.surface_z_m !== null && c.surface_z_m !== undefined ? `, surface at z ${r0(c.surface_z_m * 1000)} mm` : ""
			}).`,
		);
	else if (tracked)
		lines.push(
			`Clearance under the tool: about ${r0(tracked.mm)} mm, NOT measured now (last reading minus ${r0(tracked.descended_mm)} mm of descent since${
				tracked.lateral_mm > 0 ? `; the tool also moved ${r0(tracked.lateral_mm)} mm sideways, so this may be off` : ""
			}).`,
		);
	else
		lines.push(
			`Clearance under the tool: unknown${c?.reason ? ` (${c.reason})` : ""}. Do not assume there is room below.`,
		);
	const span = f.caps.gripper_open_span_m * 1000;
	const opening = f.grasp.opening_mm ?? f.gripper.opening_mm;
	const fraction = f.grasp.closed_fraction ?? (span > 0 ? opening / span : undefined);
	const state =
		fraction !== undefined
			? fraction < 0.1
				? "CLOSED"
				: fraction > 0.85
					? "FULLY OPEN"
					: "PARTLY OPEN"
			: opening < 5
				? "CLOSED"
				: "OPEN";
	const pct = fraction !== undefined ? ` (${r0(Math.max(0, Math.min(1, fraction)) * 100)}% of the ${r1(span)} mm span)` : "";
	const holding =
		f.grasp.holding === true
			? "holding something"
			: f.grasp.holding === false
				? "holding nothing (anything thinner than the closed finger gap cannot be sensed)"
				: `holding: UNKNOWN${f.grasp.reason ? ` (${f.grasp.reason})` : ""}`;
	lines.push(`Gripper: ${state}, opening ${r1(opening)} mm${pct}; ${holding}.`);
	return lines.join("\n");
}
