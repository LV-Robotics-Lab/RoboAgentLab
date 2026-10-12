/**
 * MotorMind mode (`--mm=true`, or `--preset motormind`): the robot side of the MotorMind port
 * (pi-embodied-work/docs/motormind/DESIGN.md). A robot that declares `motormind` in its spec (its
 * env server's `env.mm_observe` / `env.mm_run_step`) gets two tools, and in this mode they are the
 * only ones with `finish` (pure mode):
 *
 *   mm_observe  the observation of DESIGN §3.1: the tool pose in the base frame, gripper and grasp,
 *               clearance, table height, the frame clock; one three-view mosaic image.
 *   mm_act      ONE MotorMind action (the command language's move / rotate / gripper / home / wait,
 *               in mm and deg, with the executor's label): converted with the robot's capabilities,
 *               cut into equal segments whose goals are all computed from where it starts, and run
 *               one `env.mm_run_step` per segment. The result is every segment's measurement
 *               (outcome, abort, along, lateral, turned, opening, grasp) and the observation after it.
 *
 * Who decides what to do is not here: the `motormind/<base>` provider (M2) plays the mission, one
 * tool call per pi turn. Each tool call holds the robot lock (../../robot.ts `tool`) for one action;
 * `spec.motormind`'s callbacks throw outside it. Between segments `mm_act` checks the mode's cancel
 * flag and the mission deadline: the segment in flight finishes, the rest are `skipped` with the
 * reason. A timed-out segment RPC makes the result `timed_out`, which the closed-loop gate
 * (../../planner/closed-loop.ts) answers by refusing the next motion until `mm_observe` ran (the
 * robot hands the steps the server ran meanwhile back to pi's episode then).
 *
 * The tools are registered only when the mode is on, at the robot's start. The capabilities are the
 * env server's (its `caps`, the only copy); a server the robot attached to must run with the
 * settings its flags ask for (`spec.expect`).
 *
 * `home` is refused by default (`--mm-home refuse`: the capabilities say there is none); `--mm-home
 * reset` runs it as the robot's own episode reset, a labelled variant. The result's `motormind`
 * block lists every flag that differs from the faithful setting (`variant`, empty = faithful).
 */

import { readFileSync } from "node:fs";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, Type } from "typebox";
import { trackedFlags } from "../../infra/params.ts";
import { RpcUnavailable } from "../../infra/rpc.ts";
import { norm as flagValue, PRESETS } from "../../infra/tiers.ts";
import type { ToolRegistrar } from "../units/types.ts";
import { CommandError, measured, parseAction, poseOf, posePlain, report, segmentsFor, toDelta } from "./command.ts";
import {
	type Aim,
	CAPS_FIELDS,
	type Caps,
	type Delta,
	type Measured,
	type MotorMindSpec,
	type Pose,
	type Segment,
} from "./types.ts";

export * from "./types.ts";

/** `pi.events` channel on which the mode publishes its `MotorMindHandle` once the robot is up. */
export const MOTORMIND_EVENT = "pi-embodied:motormind";
/** The provider (`--model motormind/<base>`) that plays the mission in this mode. */
export const MM_PROVIDER = "motormind";
export const MM_TOOLS = ["mm_observe", "mm_act"] as const;
export const MM_HOME = ["refuse", "reset"] as const;

type Json = Record<string, unknown>;
type Result = {
	content: ({ type: "text"; text: string } | { type: "image"; data: string; mimeType: string })[];
	details: Json;
};
type Image = { data: Buffer; mime: string };

/** An observation as the mode keeps it (the reply normalized; `raw` is the reply, images included). */
export type Observation = {
	raw: Json;
	obs_id?: number;
	frame_clock_s?: number;
	pose?: Pose;
	base_pose?: Pose;
	caps: Caps;
	table_z_m: number | null;
	episode_over: boolean;
	terminated: boolean;
	truncated: boolean;
	degraded: Json;
	mosaic?: Image;
};

/** What the mode publishes on `MOTORMIND_EVENT` (the provider in M2, the GPU end-to-end checks). */
export type MotorMindHandle = {
	robot: string;
	/** The robot's callbacks as mounted: they throw unless the caller holds the robot lock. */
	observe: MotorMindSpec["observe"];
	runStep: MotorMindSpec["runStep"];
	/** The env server's endpoint, for a test's own privileged reads. */
	endpoint: () => string | undefined;
	/** Stop `mm_act` at the next segment boundary (the segment in flight finishes). */
	cancel: (reason: string) => void;
	clearCancel: () => void;
	/** The mission's wall-clock deadline (ms since the epoch; undefined = none), checked like a cancel. */
	setDeadline: (at: number | undefined) => void;
	/** The latest observation the tools saw (mm_observe, or after an action). */
	latest: () => Observation | undefined;
};

/** The robot base's side of the mount (../../robot.ts). */
export type MotorMindBase = {
	robot: string;
	/** Units or code mode is on (`--units` / `--code`), which cannot run beside this mode. */
	otherMode: () => string | undefined;
};

const round = (v: number, d: number) => Number(v.toFixed(d));
const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** An image as the env server sends it: base64 JPEG (or PNG) bytes; undefined for anything else. */
export function imageOf(v: unknown): Image | undefined {
	if (typeof v !== "string" || !v) return undefined;
	const data = Buffer.from(v, "base64");
	const mime =
		data[0] === 0xff && data[1] === 0xd8
			? "image/jpeg"
			: data[0] === 0x89 && data[1] === 0x50
				? "image/png"
				: undefined;
	return mime ? { data, mime } : undefined;
}

const TRACKED = Type.Optional(
	Type.Number({
		description:
			"The clearance below the tool you track (mm): where the aim marker sits when neither the table height nor the wrist camera gives it",
	}),
);
const OBSERVE_PARAMS = Type.Object({ tracked_mm: TRACKED });
const ACT_PARAMS = Type.Object({
	type: StringEnum(["move", "rotate", "gripper", "home", "wait"] as const),
	direction: Type.Optional(
		Type.String({
			description:
				"move: forward, backward, left (+Y), right, up, down; rotate: yaw_left, yaw_right (also clockwise / cw, anticlockwise / ccw), pitch_up, pitch_down, roll_ccw, roll_cw",
		}),
	),
	axis: Type.Optional(
		Type.Array(Type.Number(), {
			minItems: 3,
			maxItems: 3,
			description: "Instead of direction: [x, y, z] in the base frame",
		}),
	),
	distance_mm: Type.Optional(Type.Number({ description: "move: how far (a positive magnitude)" })),
	angle_deg: Type.Optional(Type.Number({ description: "rotate: how far (a positive magnitude)" })),
	speed_mm_s: Type.Optional(Type.Number()),
	rot_speed_deg_s: Type.Optional(Type.Number()),
	stop_on_contact: Type.Optional(
		Type.Boolean({ description: "move: a push; ending against something is its result" }),
	),
	state: Type.Optional(StringEnum(["open", "close"] as const)),
	width_mm: Type.Optional(Type.Number({ description: "gripper: parsed and ignored, as MotorMind does" })),
	seconds: Type.Optional(Type.Number({ description: "wait: how long" })),
	frame: Type.Optional(Type.String({ description: 'Always "base"' })),
	note: Type.Optional(Type.String()),
	label: Type.Optional(
		Type.String({ description: "The executor's label for this action (default: the action's own)" }),
	),
	open_to_mm: Type.Optional(
		Type.Number({ description: "gripper open: open only this wide (mm), then hold the fingers there" }),
	),
	tracked_mm: TRACKED,
});

/** The mode's two tools as the model sees them (registered at the robot's start when the mode is on). */
export const MM_TOOL_SCHEMAS = [
	{
		name: "mm_observe",
		description:
			"MotorMind: observe the scene without moving. Returns the tool pose in the robot's base frame (mm; forward +X, left +Y, up +Z), the gripper opening and grasp state, the clearance under the tool, the table height, the frame clock, and one image with the scene, side and wrist views side by side.",
		parameters: OBSERVE_PARAMS,
	},
	{
		name: "mm_act",
		description:
			"MotorMind: run ONE action in the robot's base frame (mm, deg): move (direction or axis, distance_mm; stop_on_contact = a push), rotate (direction or axis, angle_deg, about the tool point), gripper (state), wait (seconds) or home. It is cut into equal segments (at most 50 mm or 0.5 rad each) whose measured outcome, along-axis and lateral motion, turn, opening and grasp come back with the observation after it; a segment that does not end done stops the rest of the action.",
		parameters: ACT_PARAMS,
	},
] as const;

/** `tracked_mm` (the tool parameter) as the env server takes it. */
const aimOf = (tracked: unknown): Aim =>
	typeof tracked === "number" && Number.isFinite(tracked) && tracked > 0 ? { aim_tracked_m: tracked / 1000 } : {};

/**
 * Mount the mode on a robot: its flags, `mm_observe` and `mm_act` (registered with the robot's
 * `tool`, so each holds the robot lock), and what ../../robot.ts reads (`on`, `configError`,
 * `started`, `tools`, `prompt`, `result`, `reset`). `spec`'s callbacks are the robot's, as mounted.
 */
export function motormind(pi: ExtensionAPI, spec: MotorMindSpec, tool: ToolRegistrar, base: MotorMindBase) {
	pi.registerFlag("mm", {
		type: "string",
		default: "false",
		description:
			"MotorMind mode (true / false; pure only): only mm_observe, mm_act and finish, driven by a motormind/<base> model",
	});
	pi.registerFlag("mm-home", {
		type: "string",
		default: "refuse",
		description: `MotorMind's home action: refuse (default; the robot has none) | reset (the episode reset, a labelled variant)`,
	});
	pi.registerFlag("mm-step-timeout", {
		type: "string",
		default: "120",
		description: "MotorMind mode: how long one segment's env RPC may take before mm_act reports it timed out, s",
	});
	/** `--mm` as given: true or false (booleans as their words); anything else is refused at start. */
	const mmValue = () => {
		const v = pi.getFlag("mm");
		return v === true ? "true" : v === false ? "false" : String(v ?? "false").trim();
	};
	const on = () => mmValue() === "true";
	const homeMode = () => String(pi.getFlag("mm-home") ?? "refuse").trim();

	/** The latest observation, the tool's pose as last measured (undefined: unknown, observe first). */
	let last: Observation | undefined;
	let pose: Pose | undefined;
	let episodeOver = false;
	let cancel: { reason: string; at: number; seen?: number } | undefined;
	let deadline: number | undefined;
	const counts = {
		actions: 0,
		segments: 0,
		ticks: 0,
		motion_s: 0,
		outcomes: {} as Record<string, number>,
		timed_out: 0,
		cancelled: 0,
		errors: 0,
		episodes: 0,
		home: { asked: 0, refused: 0, reset: 0 },
	};
	pi.on("session_start", () => {
		last = pose = undefined;
		episodeOver = false;
		cancel = deadline = undefined;
		Object.assign(counts, {
			actions: 0,
			segments: 0,
			ticks: 0,
			motion_s: 0,
			outcomes: {},
			timed_out: 0,
			cancelled: 0,
			errors: 0,
			episodes: 0,
			home: { asked: 0, refused: 0, reset: 0 },
		});
	});

	/** Keep an observation reply: the latest observation, the pose, whether the episode is over. */
	function absorb(raw: Json): Observation {
		const image = imageOf(raw.mosaic_jpeg);
		const o: Observation = {
			raw,
			...(typeof raw.obs_id === "number" ? { obs_id: raw.obs_id } : {}),
			...(typeof raw.frame_clock_s === "number" ? { frame_clock_s: raw.frame_clock_s } : {}),
			...(poseOf(raw.pose) ? { pose: poseOf(raw.pose) } : {}),
			...(poseOf(raw.base_pose) ? { base_pose: poseOf(raw.base_pose) } : {}),
			// The server's table (every observation carries it; started() checked it is whole).
			caps: raw.caps as Caps,
			table_z_m: typeof raw.table_z_m === "number" ? raw.table_z_m : null,
			episode_over: raw.episode_over === true,
			terminated: raw.terminated === true,
			truncated: raw.truncated === true,
			degraded: isObject(raw.degraded) ? raw.degraded : {},
			...(image ? { mosaic: image } : {}),
		};
		last = o;
		if (o.pose) pose = o.pose;
		episodeOver = o.episode_over;
		return o;
	}

	async function observeNow(aim: Aim = {}): Promise<Observation> {
		return absorb(await spec.observe(aim));
	}

	/** An observation as the planner reads it (numbers only; images go as the one mosaic). */
	function observationText(o: Observation): Json {
		const r = o.raw;
		return {
			...(o.obs_id !== undefined ? { obs_id: o.obs_id } : {}),
			...(o.frame_clock_s !== undefined ? { frame_clock_s: round(o.frame_clock_s, 3) } : {}),
			...(o.pose ? { pose: { ...posePlain(o.pose), position_m: o.pose.position_m } } : {}),
			...(isObject(r.gripper) ? { gripper: r.gripper } : {}),
			...(isObject(r.grasp) ? { grasp: r.grasp } : {}),
			...(r.clearance !== undefined ? { clearance: r.clearance } : {}),
			table_z_mm: o.table_z_m === null ? null : round(o.table_z_m * 1000, 2),
			episode_over: o.episode_over,
			terminated: o.terminated,
			truncated: o.truncated,
			...(Object.keys(o.degraded).length ? { degraded: o.degraded } : {}),
		};
	}

	const shown = (body: Json, image: Image | undefined): Result => ({
		content: [
			{ type: "text", text: JSON.stringify(body) },
			...(image ? [{ type: "image" as const, data: image.data.toString("base64"), mimeType: image.mime }] : []),
		],
		details: body,
	});

	async function observeTool(params: Static<typeof OBSERVE_PARAMS>): Promise<Result> {
		const o = await observeNow(aimOf(params.tracked_mm));
		const body = {
			...observationText(o),
			...(o.base_pose ? { base_pose: { ...posePlain(o.base_pose), position_m: o.base_pose.position_m } } : {}),
			caps: o.caps,
		};
		return shown(body, o.mosaic);
	}

	/** A segment that did not run (`outcome` skipped or refused), with why. */
	const notRun = (seg: Segment, outcome: "skipped" | "refused", why: Partial<Measured>): Measured => ({
		kind: seg.kind,
		label: seg.label,
		outcome,
		abort_reason: "",
		moved_m: 0,
		along_axis_m: 0,
		lateral_m: 0,
		turned_rad: 0,
		ticks: 0,
		elapsed_s: 0,
		episode_over: episodeOver,
		message: "",
		...why,
	});

	async function actTool(params: Static<typeof ACT_PARAMS>): Promise<Result> {
		const { label, open_to_mm, tracked_mm, ...rest } = params as Json;
		const aim = aimOf(tracked_mm);
		const clean = Object.fromEntries(Object.entries(rest).filter(([, v]) => v !== undefined));
		let delta: Delta;
		if (clean.type === "home") counts.home.asked++;
		try {
			const action = parseAction(clean, 0);
			const caps = (last ?? (await observeNow(aim))).caps;
			delta = toDelta(action, caps);
		} catch (err) {
			if (clean.type === "home" && err instanceof CommandError) counts.home.refused++;
			throw err instanceof CommandError ? new Error(`mm_act refused, nothing moved: ${err.message}`) : err;
		}
		if (typeof label === "string" && label.trim()) delta.label = label.trim();
		if (typeof open_to_mm === "number" && delta.kind === "gripper" && delta.gripper_state === "open")
			delta.open_to_mm = open_to_mm;
		counts.actions++;
		const start = pose ?? (await observeNow(aim)).pose;
		if (!start) throw new Error("mm_act: the robot reports no tool pose; nothing moved");
		const segments = segmentsFor(delta, start, (last as Observation).caps);
		const timeoutMs = Number(pi.getFlag("mm-step-timeout")) * 1000;
		const results: Measured[] = [];
		let stopped: string | undefined;
		let failed: { error: string; timed_out: boolean } | undefined;
		let after: Observation | undefined;
		let cancelledAt: number | undefined;
		for (const [k, seg] of segments.entries()) {
			const late = deadline !== undefined && Date.now() > deadline;
			if (cancel || late) {
				const reason = cancel?.reason ?? "the mission deadline passed";
				if (cancel && cancel.seen === undefined) cancel.seen = Date.now();
				cancelledAt ??= results.length;
				results.push(notRun(seg, "skipped", { cancel_reason: reason }));
				continue;
			}
			if (stopped) {
				results.push(notRun(seg, "skipped", { reason: stopped }));
				continue;
			}
			if (episodeOver) {
				results.push(notRun(seg, "refused", { reason: "the episode is over", episode_over: true }));
				continue;
			}
			let raw: Json;
			try {
				// The last segment's observation carries the images (one render per action).
				raw = await spec.runStep(seg, {
					timeoutMs,
					observe: k === segments.length - 1 ? "full" : "state",
					...aim,
				});
			} catch (err) {
				// The robot is gone: the base ends the episode as an env_error.
				if (err instanceof RpcUnavailable) throw err;
				const error = message(err);
				failed = { error, timed_out: /timed out/i.test(error) };
				results.push({
					...notRun(seg, "skipped", {}),
					outcome: "failed",
					error,
					...(failed.timed_out ? { timed_out: true } : {}),
				});
				// Where the tool ended up is not known: the next action observes first.
				pose = undefined;
				stopped = `${seg.label} failed`;
				continue;
			}
			const m = measured(raw, seg);
			results.push(m);
			counts.segments++;
			counts.ticks += m.ticks;
			counts.motion_s += m.elapsed_s;
			if (m.pose_after) pose = m.pose_after;
			if (isObject(raw.observation)) after = absorb(raw.observation);
			if (m.episode_over) episodeOver = true;
			if (seg.kind === "home" && m.outcome === "done") {
				counts.home.reset++;
				counts.episodes++;
				episodeOver = after?.episode_over ?? false;
				if (!after) pose = undefined;
			}
			if (m.outcome !== "done")
				stopped = `${m.label} ended ${m.outcome}${m.abort_reason ? ` (${m.abort_reason})` : ""}`;
		}
		for (const m of results) counts.outcomes[m.outcome] = (counts.outcomes[m.outcome] ?? 0) + 1;
		if (failed) {
			counts.errors++;
			if (failed.timed_out) counts.timed_out++;
		}
		if (cancelledAt !== undefined) counts.cancelled++;
		// The state after the action with its images (an action that stopped early rendered none),
		// unless a segment failed: its outcome is unknown until mm_observe.
		if (!after?.mosaic && !failed && results.some((m) => m.outcome !== "skipped" && m.outcome !== "refused"))
			after = await observeNow(aim);
		const body: Json = {
			action: delta.label,
			kind: delta.kind,
			...(delta.note ? { note: delta.note } : {}),
			segments: results.map(report),
			...(cancelledAt !== undefined
				? {
						cancelled: {
							reason: cancel?.reason ?? "the mission deadline passed",
							at_segment: cancelledAt,
							...(cancel?.seen !== undefined ? { latency_s: round((cancel.seen - cancel.at) / 1000, 3) } : {}),
						},
					}
				: {}),
			...(after ? observationText(after) : {}),
			episode_over: episodeOver,
			...(failed ? { error: failed.error, ...(failed.timed_out ? { timed_out: true } : {}) } : {}),
		};
		return shown(body, after?.mosaic);
	}

	let registered = false;
	/** The two tools, through the robot's `tool` (each call holds the robot lock): once, when the mode starts on. */
	function registerTools() {
		if (registered) return;
		registered = true;
		const [observe, act] = MM_TOOL_SCHEMAS;
		tool(observe.name, observe.description, observe.parameters, observeTool);
		tool(act.name, act.description, act.parameters, actTool);
	}

	/** The flags that differ from the faithful setting: --preset motormind's values, the mode's own defaults. */
	function variant(): { flag: string; faithful: string; actual: string }[] {
		const t = trackedFlags(pi);
		const faithful = new Map<string, string>();
		for (const [flag, want] of Object.entries(PRESETS.motormind?.flags ?? {}))
			faithful.set(flag, flagValue(want === null ? t?.names.get(flag)?.default : want));
		for (const [flag, s] of t?.names ?? []) if (flag.startsWith("mm-")) faithful.set(flag, flagValue(s.default));
		return [...faithful]
			.filter(([flag]) => t?.names.has(flag) ?? true)
			.map(([flag, want]) => ({ flag, faithful: want, actual: flagValue(pi.getFlag(flag)) }))
			.filter((v) => v.actual !== v.faithful);
	}

	/**
	 * Whether the task held at some segment's end (the original's success check), from the server's
	 * truth log: its `segment` lines' instant `success_now` (not the latch). Null without the log.
	 */
	function successAtSegmentEnd(): boolean | null {
		const path = spec.truth?.();
		if (!path) return null;
		try {
			return readFileSync(path, "utf8")
				.split("\n")
				.filter(Boolean)
				.map((l) => JSON.parse(l) as Json)
				.some((l) => l.kind === "segment" && l.success_now === true);
		} catch {
			return null;
		}
	}

	const handle: MotorMindHandle = {
		robot: base.robot,
		observe: spec.observe,
		runStep: spec.runStep,
		endpoint: () => spec.endpoint?.(),
		cancel: (reason) => {
			cancel ??= { reason, at: Date.now() };
		},
		clearCancel: () => {
			cancel = undefined;
		},
		setDeadline: (at) => {
			deadline = at;
		},
		latest: () => last,
	};

	/** Fail closed before the agent starts: the mm tools are active but no motormind/<base> model plays the mission. */
	pi.on("before_agent_start", (_event, ctx) => {
		if (!on() || !pi.getActiveTools().includes("mm_act") || !ctx.model || ctx.model.provider === MM_PROVIDER)
			return undefined;
		const why = `--mm runs the MotorMind mission, which a ${MM_PROVIDER}/<base> model plays; the model is ${ctx.model.provider}/${ctx.model.id}`;
		pi.setActiveTools([]);
		if (ctx.hasUI) ctx.ui.notify(`motormind: ${why}`, "error");
		else {
			console.error(`[motormind] ${why}`);
			process.exitCode = 1;
			ctx.shutdown();
		}
		ctx.abort();
		return undefined;
	});

	return {
		on,
		/** Pure mode: the mm tools and finish only. */
		mode: (): "pure" | undefined => (on() ? "pure" : undefined),
		tools: () => [...MM_TOOLS],
		/**
		 * Why the robot must not start with these flags (DESIGN §1.7, robot side), else undefined: an
		 * --mm that is neither true nor false (the mode is pure only), another mode beside it,
		 * exploration (its memory and its tools would join the mission), an --xpolicy client that would
		 * not be mounted, a token budget (the provider's turn usage covers one call), an unknown
		 * --mm-home, or a model that is not the mission's. The OpenETA extras are
		 * ../../primitives/optional.ts `extrasInPureMode`'s.
		 */
		configError: (ctx?: Pick<ExtensionContext, "model">): string | undefined => {
			if (mmValue() !== "true" && mmValue() !== "false")
				return `--mm must be true or false (MotorMind mode is pure only), not ${mmValue()}`;
			if (!on()) return undefined;
			const other = base.otherMode();
			if (other) return `--mm=true and ${other} are modes of their own: run one of them`;
			if (pi.getFlag("explore") === true)
				return "--explore cannot run in MotorMind mode: its memory, tools and prompt would join the mission";
			if (String(pi.getFlag("xpolicy") ?? "").trim())
				return "--xpolicy cannot run in MotorMind mode (only mm_observe, mm_act and finish are mounted)";
			if (Number(pi.getFlag("max-tokens")) > 0)
				return "--max-tokens is not supported in MotorMind mode: a turn's usage counts one of its several model calls";
			if (!(MM_HOME as readonly string[]).includes(homeMode()))
				return `--mm-home must be one of ${MM_HOME.join(", ")}, not ${homeMode()}`;
			const model = ctx?.model;
			if (model && model.provider !== MM_PROVIDER)
				return `--mm runs the MotorMind mission, which a ${MM_PROVIDER}/<base> model plays; the model is ${model.provider}/${model.id}`;
			return undefined;
		},
		/**
		 * The robot is up (called under the robot lock): one observation must answer with the tool pose,
		 * the base pose and the whole capability table (else the start fails: the server has no mm RPCs,
		 * no base frame or another table), and with the configuration the robot's flags ask for (a
		 * server it attached to may run otherwise); then the tools are registered and the handle
		 * published.
		 */
		async started(): Promise<void> {
			const o = await observeNow();
			if (!o.pose) throw new Error("env.mm_observe returned no tool pose in the base frame");
			if (!o.base_pose) throw new Error("env.mm_observe returned no base pose (robot0_base)");
			const caps = isObject(o.raw.caps) ? o.raw.caps : {};
			const missing = CAPS_FIELDS.filter((f) => !(f in caps));
			if (missing.length)
				throw new Error(
					`env.mm_observe's capability table lacks ${missing.join(", ")} (the server's is the only one)`,
				);
			const config = isObject(o.raw.config) ? o.raw.config : {};
			const off = Object.entries(spec.expect?.() ?? {}).filter(([k, v]) => config[k] !== v);
			if (off.length)
				throw new Error(
					`the env server runs with ${off.map(([k]) => `${k}=${JSON.stringify(config[k] ?? null)}`).join(", ")}, but the flags ask for ${off.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")}: start it with the same settings, or let pi start it`,
				);
			registerTools();
			pi.events.emit(MOTORMIND_EVENT, handle);
		},
		/** A scene reset outside the mode (the operator's, exploration's): the pose is unknown until observed. */
		reset: () => {
			last = pose = undefined;
			episodeOver = false;
		},
		/** The system prompt in this mode: a description only (the provider does not read it). */
		prompt: () =>
			"MotorMind mode. The motormind provider plays the MotorMind mission (planner, executor, verifier) and acts only through mm_observe and mm_act, one action per turn, in the robot's base frame (robot0_base: mm, deg; forward +X, left +Y, up +Z). It calls finish when the mission ends.",
		handle,
		/**
		 * The result's `motormind` block (computed now; M2 adds the mission's own state). `status` here
		 * is the env's success, else why the run ended: error (the robot broke, or the model failed),
		 * timeout (--time-limit), exhausted (another budget), cancelled (aborted before finish); null
		 * when finish ended it.
		 */
		result(end: { budget?: string; finished: boolean; solved?: boolean; error?: string }): Json {
			if (!on()) return {};
			const status = end.solved
				? "env_success"
				: end.error !== undefined
					? "error"
					: end.budget === "time"
						? "timeout"
						: end.budget
							? "exhausted"
							: !end.finished
								? "cancelled"
								: null;
			return {
				motormind: {
					status,
					variant: variant(),
					home: { mode: homeMode(), ...counts.home },
					episodes: counts.episodes,
					actions: counts.actions,
					segments: counts.segments,
					outcomes: counts.outcomes,
					ticks: counts.ticks,
					motion_s: round(counts.motion_s, 3),
					timed_out: counts.timed_out,
					cancelled: counts.cancelled,
					errors: counts.errors,
					degraded: { table_z: last?.degraded.table_z === true, side_mount: last?.degraded.side_mount === true },
					...(last?.base_pose
						? { base_pose: { ...posePlain(last.base_pose), position_m: last.base_pose.position_m } }
						: {}),
					...(typeof last?.raw.settle_ticks === "number" ? { settle_ticks: last.raw.settle_ticks } : {}),
					success_at_segment_end: successAtSegmentEnd(),
				},
			};
		},
	};
}

export type MotorMindMode = ReturnType<typeof motormind>;
