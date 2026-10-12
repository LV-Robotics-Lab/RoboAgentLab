/**
 * MotorMind's command language (DESIGN §2.5; spec-executor §3-5): parse what the model wrote,
 * convert it to deltas with the robot's capabilities, apply the command-level limits, and cut each
 * delta into the segments the env server runs (`env.mm_run_step`, one per call). This is the only
 * place units change: the command, labels and every text the planner reads are mm / deg; deltas,
 * segments and the RPC are m / rad / s; a segment's measurement comes back in SI units and is
 * written out again in mm (2 decimals) and deg (1 decimal).
 *
 * Labels are an interface, not decoration: the mission's faithful net motion (`--mm-net-motion
 * label`, ./frames.ts `labelDirection`) reads a step's direction from its label's first direction
 * word, so their exact forms are kept here (C5): "move left 50.0 mm (base)", "move axis [0.0, 1.0,
 * 0.0] 20.0 mm (base)", "rotate yaw_left 30.0 deg (base)", "gripper close", "gripper open to 40.0
 * mm", "go home", "wait 0.50 s", the shortened "... (shortened to 300 mm: ...)", the executor's
 * "lift over the table" and "back down from over the table", the carry legs "... (1 of 3)", the
 * first-move test "... (the first 17 mm)" / "... (the rest)", and a segment's "... [2/3]".
 */

import {
	add,
	axisAngle,
	dot,
	matmul,
	norm,
	rotationAbout,
	rotationAxis,
	scale,
	translationAxis,
	transpose,
	unit,
} from "./frames.ts";
import {
	type Action,
	type ActionType,
	type Caps,
	type Common,
	type Delta,
	DT,
	FALLBACK_ROTATION_SPEED_RAD_S,
	FALLBACK_SPEED_M_S,
	GRIPPER_SETTLE_TICKS,
	type Grasp,
	MAX_COMPOSITE_MM,
	MAX_PUSH_MM,
	MAX_TURN_DEG,
	type Mat3,
	type Measured,
	MIN_FIT_MM,
	type Outcome,
	type Pose,
	REACH_SLACK_M,
	type Segment,
	type Vec3,
} from "./types.ts";

/** A command the robot will not run as written; the message is what the model reads to retry. */
export class CommandError extends Error {}

// ---------------------------------------------------------------------------
// Python's number formatting (labels are compared as strings with the reference's)

/** `format(x, f".{d}f")`: correctly rounded, exact ties to even (JS's toFixed rounds them up). */
export function pyFixed(x: number, d: number): string {
	if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
	const exact = Math.abs(x).toFixed(100);
	const [whole, frac] = exact.split(".");
	if (/^50*$/.test(frac.slice(d))) {
		const kept = whole + frac.slice(0, d);
		if (Number(kept.at(-1)) % 2 === 0) return `${x < 0 ? "-" : ""}${whole}${d ? `.${frac.slice(0, d)}` : ""}`;
	}
	return x.toFixed(d);
}

/** `repr(float(x))`: the shortest round-trip digits, `1.0` for integers, `1e-05` below 1e-4. */
export function pyFloat(x: number): string {
	if (Object.is(x, -0)) return "-0.0";
	if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
	const [mantissa, e] = x.toExponential().split("e");
	const exp = Number(e);
	if (exp < -4 || exp >= 16) return `${mantissa}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
	const s = String(x);
	return /[.e]/.test(s) ? s : `${s}.0`;
}

const pyList = (v: readonly number[]) => `[${v.map(pyFloat).join(", ")}]`;

// ---------------------------------------------------------------------------
// labels

export const TEST_REST = "(the rest)";
/** The executor's inserted moves before and after a sideways move over the table (spec-executor §11.1). */
export const LIFT_LABEL = "lift over the table";
export const DROP_LABEL = "back down from over the table";

/** The label an action is known by (its step labels, the history, the net motion). */
export function actionLabel(
	a: Pick<
		Action,
		"type" | "direction" | "axis" | "distance_mm" | "angle_deg" | "state" | "width_mm" | "seconds" | "frame"
	>,
): string {
	const what = () => a.direction ?? `axis ${pyList(a.axis ?? [])}`;
	switch (a.type) {
		case "move":
			return `move ${what()} ${pyFixed(a.distance_mm ?? 0, 1)} mm (${a.frame})`;
		case "rotate":
			return `rotate ${what()} ${pyFixed(a.angle_deg ?? 0, 1)} deg (${a.frame})`;
		case "gripper":
			return a.width_mm !== undefined ? `gripper ${a.state} to ${pyFixed(a.width_mm, 1)} mm` : `gripper ${a.state}`;
		case "home":
			return "go home";
		case "wait":
			return `wait ${pyFixed(a.seconds ?? 0, 2)} s`;
	}
}
/** A segment of a delta cut into `of` pieces. */
export const segmentLabel = (label: string, chunk: number, of: number) => `${label} [${chunk}/${of}]`;
/** A move cut short by a limit; `said` is why. */
export const shortenedLabel = (label: string, mm: number, said: string) =>
	`${label} (shortened to ${pyFixed(mm, 0)} mm: ${said})`;
/** One leg of a carry split into legs (spec-executor §11.2). */
export const carryLegLabel = (label: string, leg: number, legs: number) => `${label || "move"} (${leg} of ${legs})`;
/** The first-move test after a wall grasp (spec-executor §11.3): a short lead, then the rest. */
export const testLeadLabel = (label: string, mm: number) => `${label} (the first ${pyFixed(mm, 0)} mm)`;
export const testRestLabel = (label: string) => `${label} ${TEST_REST}`;
/** The executor's follow correction after a wrist look: a horizontal axis move by `shift` (m). */
export const followLabel = (shift: readonly [number, number], mm: number) =>
	actionLabel({ type: "move", axis: [shift[0], shift[1], 0], distance_mm: mm, frame: "base" });

// ---------------------------------------------------------------------------
// parsing

const TYPES: readonly ActionType[] = ["move", "rotate", "gripper", "home", "wait"];
const COMMON = ["frame", "speed_mm_s", "rot_speed_deg_s", "stop_on_contact", "note"] as const;
const KEYS: Record<ActionType, readonly string[]> = {
	move: ["type", "direction", "axis", "distance_mm", ...COMMON],
	rotate: ["type", "direction", "axis", "angle_deg", ...COMMON],
	gripper: ["type", "state", "width_mm", ...COMMON],
	home: ["type", ...COMMON],
	wait: ["type", "seconds", ...COMMON],
};
type Json = Record<string, unknown>;
const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);

/** A finite number (a numeric string counts); `what` names it in the error. */
function finite(v: unknown, what: string): number {
	const n = typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : Number.NaN;
	if (!Number.isFinite(n)) throw new CommandError(`${what} must be a finite number, got ${JSON.stringify(v)}`);
	return n;
}

function common(entry: Json, where: string, inherited: Partial<Common> = {}): Common {
	const v = { ...inherited, ...Object.fromEntries(COMMON.filter((k) => k in entry).map((k) => [k, entry[k]])) };
	const frame = v.frame === undefined ? "base" : String(v.frame).trim().toLowerCase();
	if (frame !== "base")
		throw new CommandError(`${where}: frame must be "base" (the robot's own frame), got ${JSON.stringify(v.frame)}`);
	if (v.stop_on_contact !== undefined && typeof v.stop_on_contact !== "boolean")
		throw new CommandError(
			`${where}: stop_on_contact must be true or false, got ${JSON.stringify(v.stop_on_contact)}`,
		);
	if (v.note !== undefined && typeof v.note !== "string") throw new CommandError(`${where}: note must be text`);
	return {
		frame,
		...(v.speed_mm_s !== undefined ? { speed_mm_s: finite(v.speed_mm_s, `${where}: speed_mm_s`) } : {}),
		...(v.rot_speed_deg_s !== undefined
			? { rot_speed_deg_s: finite(v.rot_speed_deg_s, `${where}: rot_speed_deg_s`) }
			: {}),
		...(v.stop_on_contact !== undefined ? { stop_on_contact: v.stop_on_contact } : {}),
		note: (v.note as string | undefined) ?? "",
	};
}

/** A magnitude: positive and finite; a negative one says how to reverse instead. */
function magnitude(v: unknown, what: string, where: string, reverse: string): number {
	if (v === undefined) throw new CommandError(`${where}: ${what} is required`);
	const n = finite(v, `${where}: ${what}`);
	if (n < 0) throw new CommandError(`${where}: ${what} is a magnitude and cannot be negative (${n}); ${reverse}`);
	if (n === 0) throw new CommandError(`${where}: ${what} is 0, which does nothing; leave the action out`);
	return n;
}

/** One action of a command (`index` names it in errors); `inherited` are the command's top-level common keys. */
export function parseAction(entry: unknown, index = 0, inherited: Partial<Common> = {}): Action {
	const where = `action ${index}`;
	if (!isObject(entry)) throw new CommandError(`${where} must be an object, got ${JSON.stringify(entry)}`);
	const type = typeof entry.type === "string" ? entry.type.trim().toLowerCase() : undefined;
	if (!type || !(TYPES as readonly string[]).includes(type))
		throw new CommandError(
			`${where}: ${type ? `unknown type "${entry.type}"` : "no type"}; the types are ${TYPES.join(", ")} (it has the keys ${Object.keys(entry).join(", ") || "none"})`,
		);
	const t = type as ActionType;
	const extra = Object.keys(entry).filter((k) => !KEYS[t].includes(k));
	if (extra.length)
		throw new CommandError(`${where} (${t}): unknown key ${extra.join(", ")}; a ${t} takes ${KEYS[t].join(", ")}`);
	const a: Action = { ...common(entry, where, inherited), index, type: t };
	if (t === "move" || t === "rotate") {
		const hasDirection = entry.direction !== undefined;
		const hasAxis = entry.axis !== undefined;
		if (hasDirection === hasAxis)
			throw new CommandError(
				`${where} (${t}): give exactly one of "direction" (a word) or "axis" ([x, y, z] in the base frame)`,
			);
		if (hasDirection) {
			if (typeof entry.direction !== "string")
				throw new CommandError(
					`${where}: direction must be a word, got ${JSON.stringify(entry.direction)}; for a raw vector use "axis"`,
				);
			a.direction = entry.direction.trim().toLowerCase();
		} else {
			if (!Array.isArray(entry.axis) || entry.axis.length !== 3)
				throw new CommandError(`${where}: axis must be [x, y, z], got ${JSON.stringify(entry.axis)}`);
			a.axis = entry.axis.map((v, i) => finite(v, `${where}: axis[${i}]`)) as Vec3;
		}
		const reverse = hasDirection ? "use the opposite direction word" : "negate the axis";
		if (t === "move") a.distance_mm = magnitude(entry.distance_mm, "distance_mm", where, reverse);
		else a.angle_deg = magnitude(entry.angle_deg, "angle_deg", where, reverse);
	} else if (t === "gripper") {
		const state = typeof entry.state === "string" ? entry.state.trim().toLowerCase() : entry.state;
		if (state !== "open" && state !== "close")
			throw new CommandError(
				`${where} (gripper): state must be "open" or "close", got ${JSON.stringify(entry.state)}`,
			);
		a.state = state;
		if (entry.width_mm !== undefined) a.width_mm = finite(entry.width_mm, `${where}: width_mm`);
	} else if (t === "wait") {
		if (entry.seconds === undefined) throw new CommandError(`${where} (wait): seconds is required`);
		const s = finite(entry.seconds, `${where}: seconds`);
		if (!(s > 0)) throw new CommandError(`${where} (wait): seconds must be more than 0, got ${s}`);
		a.seconds = s;
	}
	return a;
}

/**
 * A command as the model writes it: a JSON string, one action, a list of actions, or
 * `{actions: [...], <common keys>}` whose common keys (frame, speed_mm_s, rot_speed_deg_s,
 * stop_on_contact, note) every action inherits unless it sets its own.
 */
export function parseCommand(payload: unknown): Action[] {
	let p = payload;
	if (typeof p === "string") {
		try {
			p = JSON.parse(p);
		} catch (err) {
			throw new CommandError(`the command is not JSON: ${err instanceof Error ? err.message : String(err)}`);
		}
	}
	let entries: unknown[];
	let inherited: Partial<Common> = {};
	if (Array.isArray(p)) entries = p;
	else if (isObject(p) && "actions" in p) {
		const extra = Object.keys(p).filter((k) => k !== "actions" && !(COMMON as readonly string[]).includes(k));
		if (extra.length)
			throw new CommandError(
				`unknown top-level key ${extra.join(", ")}; next to "actions" a command takes only ${COMMON.join(", ")}`,
			);
		if (!Array.isArray(p.actions))
			throw new CommandError(`"actions" must be a list, got ${JSON.stringify(p.actions)}`);
		entries = p.actions;
		inherited = Object.fromEntries(COMMON.filter((k) => k in (p as Json)).map((k) => [k, (p as Json)[k]]));
	} else if (isObject(p)) entries = [p];
	else
		throw new CommandError(
			`a command is an action, a list of actions or {"actions": [...]}, got ${JSON.stringify(p)}`,
		);
	if (!entries.length) throw new CommandError("the command has no actions");
	return entries.map((e, i) => parseAction(e, i, inherited));
}

// ---------------------------------------------------------------------------
// conversion

/** One action as a delta in SI units, with the robot's speeds; a home the robot does not support is refused. */
export function toDelta(a: Action, caps: Caps): Delta {
	const base = { index: a.index, label: actionLabel(a), note: a.note, stop_on_contact: false, push: false };
	const axisOf = (named: (w: string) => Vec3) => {
		try {
			return a.axis ? unit(a.axis) : named(a.direction ?? "");
		} catch (err) {
			throw new CommandError(`action ${a.index}: ${err instanceof Error ? err.message : String(err)}`);
		}
	};
	switch (a.type) {
		case "move": {
			const contact = a.stop_on_contact === true;
			const mm = a.distance_mm ?? 0;
			if (contact && mm > MAX_PUSH_MM)
				throw new CommandError(
					`action ${a.index}: a contact push (stop_on_contact) moves at most ${MAX_PUSH_MM} mm; push ${pyFixed(mm, 0)} mm as several shorter pushes`,
				);
			const speed = a.speed_mm_s !== undefined ? a.speed_mm_s / 1000 : caps.default_speed_m_s || FALLBACK_SPEED_M_S;
			if (!(speed > 0))
				throw new CommandError(`action ${a.index}: speed_mm_s must be positive, got ${a.speed_mm_s}`);
			return {
				...base,
				kind: "translate",
				axis: axisOf(translationAxis),
				magnitude: mm / 1000,
				speed: Math.min(speed, caps.max_speed_m_s),
				stop_on_contact: contact,
				push: contact,
			};
		}
		case "rotate": {
			const speed =
				a.rot_speed_deg_s !== undefined
					? (a.rot_speed_deg_s * Math.PI) / 180
					: caps.default_rotation_speed_rad_s || FALLBACK_ROTATION_SPEED_RAD_S;
			if (!(speed > 0))
				throw new CommandError(`action ${a.index}: rot_speed_deg_s must be positive, got ${a.rot_speed_deg_s}`);
			return {
				...base,
				kind: "rotate",
				axis: axisOf(rotationAxis),
				magnitude: ((a.angle_deg ?? 0) * Math.PI) / 180,
				speed: Math.min(speed, caps.max_rotation_speed_rad_s),
			};
		}
		case "gripper":
			// The model's width_mm is parsed and dropped: only the executor opens to a width (open_to_mm).
			return { ...base, kind: "gripper", magnitude: 0, speed: 0, gripper_state: a.state };
		case "home":
			if (!caps.supports_home)
				throw new CommandError(
					`action ${a.index}: this robot has no home; move back with move and rotate actions instead`,
				);
			return { ...base, kind: "home", magnitude: 0, speed: 0 };
		case "wait":
			return { ...base, kind: "wait", magnitude: 0, speed: 0, seconds: a.seconds };
	}
}

/**
 * Cut the translation at `index` to `fits` m and drop every delta after it; the label and note say
 * why (`said`) and what to do (`advice`).
 */
export function shorten(deltas: Delta[], index: number, fits: number, said: string, advice: string): Delta[] {
	const d = deltas[index];
	const dropped = deltas.length - index - 1;
	const was = d.label;
	d.magnitude = fits;
	d.label = shortenedLabel(was, fits * 1000, said);
	d.note = `${said}: ${was} ran only ${pyFixed(fits * 1000, 0)} mm${dropped ? `, and the ${dropped} action(s) after it did not run` : ""}; ${advice}.`;
	deltas.length = index + 1;
	return deltas;
}

const ANOTHER_COMMAND = "send what is left as another command";

/**
 * The limits on one command as a whole: its rotations add up to at most 90 deg; nothing moves after
 * a home; its translations end within 400 mm of where it started, else the first one past that is
 * cut to end on the circle and the rest of the command is dropped (under 1 mm left: refused).
 */
export function commandLimits(deltas: Delta[]): Delta[] {
	const turned = deltas
		.filter((d) => d.kind === "rotate")
		.reduce((s, d) => s + (Math.abs(d.magnitude) * 180) / Math.PI, 0);
	if (turned > MAX_TURN_DEG + 1e-9)
		throw new CommandError(
			`the command turns ${pyFixed(turned, 1)} deg in all; one command turns at most ${MAX_TURN_DEG} deg, so ${ANOTHER_COMMAND}`,
		);
	const home = deltas.findIndex((d) => d.kind === "home");
	if (home >= 0 && deltas.slice(home + 1).some((d) => d.kind === "translate" || d.kind === "rotate"))
		throw new CommandError(
			"nothing may move after home in the same command: where home leaves the robot is not known in advance",
		);
	const limit = MAX_COMPOSITE_MM / 1000;
	let at: Vec3 = [0, 0, 0];
	for (let i = 0; i < deltas.length; i++) {
		const d = deltas[i];
		if (d.kind !== "translate" || !d.axis) continue;
		const end = add(at, scale(d.axis, d.magnitude));
		if (norm(end) <= limit + 1e-12) {
			at = end;
			continue;
		}
		// |at + t axis| = limit, t >= 0 (axis is a unit vector).
		const b = dot(at, d.axis);
		const fits = Math.max(0, -b + Math.sqrt(Math.max(0, b * b - dot(at, at) + limit * limit)));
		if (fits * 1000 < MIN_FIT_MM)
			throw new CommandError(
				`action ${d.index}: the command's translations already end ${pyFixed(norm(at) * 1000, 0)} mm from where it started; one command ends at most ${MAX_COMPOSITE_MM} mm away, so ${ANOTHER_COMMAND}`,
			);
		return shorten(
			deltas,
			i,
			fits,
			`the command's moves end at most ${MAX_COMPOSITE_MM} mm from where it started`,
			ANOTHER_COMMAND,
		);
	}
	return deltas;
}

/** A command (any form `parseCommand` takes) as the deltas it runs, the command-level limits applied. */
export function toDeltas(payload: unknown, caps: Caps): Delta[] {
	return commandLimits(parseCommand(payload).map((a) => toDelta(a, caps)));
}

/**
 * The reach limit (spec-executor §4.3), on the deltas in place from the tool at `start`: only the
 * horizontal distance to the base counts, and a push ends the check (contact decides where it
 * stops). A translation may end within the reach, or at most 5 mm further out than the tool is now; one
 * that would not is cut to what fits (whole mm, rounded down, so a re-sent number is not refused
 * again) and the rest dropped. Under 1 mm fitting it runs when it ends within the reach plus the
 * arrival tolerance, and is refused otherwise.
 */
export function fitToReach(
	deltas: Delta[],
	start: Pose,
	caps: Caps,
): { deltas: Delta[]; shortened: boolean; refused?: string } {
	let p: Vec3 = [...start.position_m];
	// The slack is from where the tool is now: a command that started inside the reach ends within it.
	const limit = Math.max(caps.reach_m, Math.hypot(p[0], p[1]) + REACH_SLACK_M);
	for (let i = 0; i < deltas.length; i++) {
		const d = deltas[i];
		if (d.kind !== "translate" || !d.axis) continue;
		if (d.push) break;
		const end = add(p, scale(d.axis, d.magnitude));
		const horizontal = Math.hypot(d.axis[0], d.axis[1]);
		const now = Math.hypot(p[0], p[1]);
		const out = Math.hypot(end[0], end[1]);
		if (horizontal < 1e-12 || out <= limit + 1e-12) {
			p = end;
			continue;
		}
		const u = [d.axis[0] / horizontal, d.axis[1] / horizontal];
		const b = p[0] * u[0] + p[1] * u[1];
		const disc = b * b - (now * now - limit * limit);
		const reach = disc >= 0 ? Math.max(0, -b + Math.sqrt(disc)) : 0;
		const fitsMm = Math.floor(reach * 1000 + 1e-9);
		if (fitsMm < MIN_FIT_MM) {
			if (out <= caps.reach_m + caps.arrival_tolerance_m) {
				p = end;
				continue;
			}
			return {
				deltas,
				shortened: false,
				refused: `action ${d.index} (${d.label}) would end ${pyFixed(out * 1000, 0)} mm from the base, past the ${pyFixed(caps.reach_m * 1000, 0)} mm reach, and no part of it fits: at the edge of the reach a move falls short and drifts, which looks like being blocked. Work from the near side of the target.`,
			};
		}
		shorten(
			deltas,
			i,
			fitsMm / 1000 / horizontal,
			`it would end ${pyFixed(out * 1000, 0)} mm from the base, past the ${pyFixed(caps.reach_m * 1000, 0)} mm reach`,
			"work from the near side of the target",
		);
		return { deltas, shortened: true };
	}
	return { deltas, shortened: false };
}

// ---------------------------------------------------------------------------
// segments

/** How many equal segments `total` takes when one may be at most `limit`. */
export const pieces = (total: number, limit: number) => Math.max(1, Math.ceil(Math.abs(total) / limit - 1e-9));
const round = (v: number, d: number) => Number(v.toFixed(d));

/**
 * A delta as the segments the env server runs one at a time. A translation or rotation is cut into
 * n = ceil(total / per-segment limit) equal segments whose goals are all computed now, from where
 * the delta starts (i/n of the way; a rotation turns about the fixed base axis through the tool
 * point, left-multiplied); the gripper, a wait and home are one segment each.
 */
export function segmentsFor(d: Delta, start: Pose, caps: Caps): Segment[] {
	const one = { chunk: 1, of: 1, label: d.label };
	switch (d.kind) {
		case "gripper":
			return [
				{
					...one,
					kind: "gripper",
					state: d.gripper_state,
					...(d.open_to_mm !== undefined ? { open_to_mm: d.open_to_mm } : {}),
					duration_s: GRIPPER_SETTLE_TICKS * DT,
				},
			];
		case "wait":
			return [
				{
					...one,
					kind: "wait",
					seconds: d.seconds,
					ticks: Math.round((d.seconds ?? 0) / DT),
					duration_s: d.seconds ?? 0,
				},
			];
		case "home":
			return [{ ...one, kind: "home", duration_s: 0.5 }];
		case "translate":
		case "rotate": {
			const axis = d.axis as Vec3;
			const turn = d.kind === "rotate";
			const n = pieces(d.magnitude, turn ? caps.max_rotation_rad : caps.max_translation_m);
			const part = d.magnitude / n;
			return Array.from({ length: n }, (_, k): Segment => {
				const i = k + 1;
				const goal: Pose = turn
					? {
							position_m: [...start.position_m],
							rotation: matmul(rotationAbout(axis, (d.magnitude * i) / n), start.rotation),
						}
					: { position_m: add(start.position_m, scale(axis, (d.magnitude * i) / n)), rotation: start.rotation };
				return {
					kind: d.kind,
					chunk: i,
					of: n,
					label: segmentLabel(d.label, i, n),
					goal_position_m: goal.position_m,
					goal_rotation: goal.rotation,
					axis: axis.map((v) => round(v, 4)) as Vec3,
					...(turn ? { angle_rad: part } : { distance_m: part }),
					speed: d.speed,
					max_step: tickLimit(d, caps),
					push: d.push,
					stop_on_contact: d.stop_on_contact || d.push,
					duration_s: Math.max(DT, part / d.speed),
				};
			});
		}
	}
}

/**
 * The most one tick moves (m) or turns (rad) toward a segment's goal: the controller's per-tick
 * limit or the speed times the tick, whichever is less.
 */
export const tickLimit = (d: Pick<Delta, "kind" | "speed">, caps: Caps) =>
	Math.min(d.kind === "rotate" ? caps.max_rotation_rad : caps.max_translation_m, d.speed * DT);

// ---------------------------------------------------------------------------
// measurements

const num = (...vs: unknown[]) => {
	for (const v of vs) if (typeof v === "number" && Number.isFinite(v)) return v;
	return undefined;
};

/** A pose from the forms a reply may carry it in: `{position_m, rotation}`, `{pos, R}`, `{pos, quat_xyzw}`. */
export function poseOf(v: unknown): Pose | undefined {
	if (!isObject(v)) return undefined;
	const p = (v.position_m ?? v.position ?? v.pos ?? v.p) as unknown;
	const r = (v.rotation ?? v.R ?? v.rot) as unknown;
	if (!Array.isArray(p) || p.length !== 3) return undefined;
	const position_m = p.map(Number) as Vec3;
	if (Array.isArray(r) && r.length === 3 && r.every((row) => Array.isArray(row) && row.length === 3))
		return { position_m, rotation: r.map((row: unknown[]) => row.map(Number)) as Mat3 };
	const q = (v.quat_xyzw ?? v.quat) as unknown;
	if (Array.isArray(q) && q.length === 4) {
		const [x, y, z, w] = q.map(Number);
		const n = Math.hypot(x, y, z, w) || 1;
		const [a, b, c, d] = [x / n, y / n, z / n, w / n];
		return {
			position_m,
			rotation: [
				[1 - 2 * (b * b + c * c), 2 * (a * b - c * d), 2 * (a * c + b * d)],
				[2 * (a * b + c * d), 1 - 2 * (a * a + c * c), 2 * (b * c - a * d)],
				[2 * (a * c - b * d), 2 * (b * c + a * d), 1 - 2 * (a * a + b * b)],
			],
		};
	}
	return undefined;
}

const OUTCOMES: readonly Outcome[] = ["done", "contact", "limited", "refused", "skipped", "failed"];

/** A segment's RPC reply (SI units) as `Measured`; `seg` gives the kind, label and axis it ran with. */
export function measured(raw: Record<string, unknown>, seg: Segment): Measured {
	const detail = isObject(raw.detail) ? raw.detail : {};
	const outcome = String(raw.outcome ?? "failed") as Outcome;
	const abort = String(raw.abort_reason ?? "");
	const opening = num(
		detail.opening_m,
		num(detail.opening_mm) !== undefined ? (detail.opening_mm as number) / 1000 : undefined,
	);
	const asked = num(
		detail.asked_m,
		num(detail.asked_mm) !== undefined ? (detail.asked_mm as number) / 1000 : undefined,
	);
	return {
		kind: seg.kind,
		label: seg.label,
		outcome: OUTCOMES.includes(outcome) ? outcome : "failed",
		abort_reason: abort === "blocked" || abort === "skidded" ? abort : "",
		moved_m: num(raw.moved_m) ?? 0,
		along_axis_m: num(raw.along_axis_m, raw.moved_m) ?? 0,
		lateral_m: num(raw.lateral_m) ?? 0,
		turned_rad: num(raw.turned_rad) ?? 0,
		...(poseOf(raw.pose_after) ? { pose_after: poseOf(raw.pose_after) } : {}),
		// Only a translation that is not a push reports its axis (the executor's `axis`).
		...(seg.kind === "translate" && !seg.push && seg.axis ? { axis: seg.axis } : {}),
		ticks: num(raw.ticks) ?? 0,
		...(num(raw.servo_ticks) !== undefined ? { servo_ticks: num(raw.servo_ticks) } : {}),
		...(num(raw.hold_ticks) !== undefined ? { hold_ticks: num(raw.hold_ticks) } : {}),
		elapsed_s: num(raw.elapsed_s) ?? 0,
		...(num(raw.frame_clock_s) !== undefined ? { frame_clock_s: num(raw.frame_clock_s) } : {}),
		...(opening !== undefined ? { opening_m: opening } : {}),
		...(asked !== undefined ? { asked_m: asked } : {}),
		...(isObject(raw.grasp) ? { grasp: raw.grasp as Grasp } : {}),
		episode_over: raw.episode_over === true,
		message: String(raw.message ?? ""),
		...(OUTCOMES.includes(outcome) ? {} : { error: `the server reported outcome ${JSON.stringify(raw.outcome)}` }),
	};
}

const mm2 = (m: number) => round(m * 1000, 2);

/** A pose as the planner reads it: position in mm (2 decimals), the rotation matrix to 6 decimals. */
export const posePlain = (p: Pose) => ({
	position_mm: p.position_m.map(mm2),
	rotation: p.rotation.map((row) => row.map((v) => round(v, 6))),
});

/** A measured segment as the planner reads it: mm (2 decimals) and deg (1 decimal). */
export function report(m: Measured): Record<string, unknown> {
	return {
		label: m.label,
		kind: m.kind,
		outcome: m.outcome,
		...(m.abort_reason ? { abort_reason: m.abort_reason } : {}),
		moved_mm: mm2(m.moved_m),
		along_axis_mm: mm2(m.along_axis_m),
		lateral_mm: mm2(m.lateral_m),
		turned_deg: round((m.turned_rad * 180) / Math.PI, 1),
		...(m.axis ? { axis: m.axis } : {}),
		ticks: m.ticks,
		...(m.servo_ticks !== undefined ? { servo_ticks: m.servo_ticks } : {}),
		...(m.hold_ticks !== undefined ? { hold_ticks: m.hold_ticks } : {}),
		elapsed_s: round(m.elapsed_s, 3),
		...(m.frame_clock_s !== undefined ? { frame_clock_s: round(m.frame_clock_s, 3) } : {}),
		...(m.opening_m !== undefined || m.asked_m !== undefined
			? {
					detail: {
						...(m.opening_m !== undefined ? { opening_mm: mm2(m.opening_m) } : {}),
						...(m.asked_m !== undefined ? { asked_mm: mm2(m.asked_m) } : {}),
					},
				}
			: {}),
		...(m.grasp ? { grasp: m.grasp } : {}),
		...(m.pose_after ? { pose_after: posePlain(m.pose_after) } : {}),
		...(m.message ? { message: m.message } : {}),
		...(m.reason ? { reason: m.reason } : {}),
		...(m.cancel_reason ? { cancel_reason: m.cancel_reason } : {}),
		...(m.error ? { error: m.error } : {}),
		...(m.timed_out ? { timed_out: true } : {}),
	};
}

/** The rotation that took `before` to `after` about base axes (left-multiplied): its unit axis and angle. */
export const turnBetween = (before: Pose, after: Pose) => axisAngle(matmul(after.rotation, transpose(before.rotation)));
