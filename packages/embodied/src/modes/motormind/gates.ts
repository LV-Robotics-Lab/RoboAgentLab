/**
 * The executor's state, words and gates (spec-executor §2.4, §10, §13, §18): what one subgoal
 * remembers (the located point, footprint, top face, empty closes, the payload, the receiver, the
 * frozen counters), the word classes of a criterion, the gate chain a proposal passes at proposal
 * time and again before every later action of a batch (first refusal wins), and the recovery rules
 * read off the measured segments. Every rule reads structured fields (`outcome`, `abort_reason`,
 * `grasp.holding`, `detail.opening_mm`), never a message.
 *
 * Clean room: written from the spec; constants are the spec's numbers.
 */

import { actionLabel, fitToReach, pyFixed } from "./command.ts";
import type { Box, CameraName, Frame, SegmentReport } from "./contract.ts";
import { approach, jawsDown } from "./contract.ts";
import { add, apply, dot, matmul, norm, ROTATION, rotationAbout, sayOffset, scale, sub, unit } from "./frames.ts";
import { apartXy, longestAcross } from "./geometry.ts";
import type { ClearanceRead } from "./observe.ts";
import type { Caps, Delta, Pose, Vec3 } from "./types.ts";

// constants (spec-executor §18)
export const DOWN_WITHIN_DEG = 45;
export const REACH_TANGENT_M = 0.005;
export const EMPTY_CLOSE_RADIUS_M = 0.008;
export const RECOVERY_UP_M = 0.1;
export const RECOVERY_JUDGED_M = 0.095;
export const DEFAULT_JAW_HALF_GAP_MM = 40;
export const PAD_MARGIN_MM = 5;
export const MIN_CLOSE_TOLERANCE_MM = 20;
export const MIN_WIDTH_MM = 8;
export const IMPLAUSIBLE_WIDTH_SPANS = 3;
export const WIDTH_JUMP_SHARE = 0.5;
export const GRASP_BELOW_SHARE = 0.25;
export const LEVEL_SLACK_MM = 1.5;
export const MIN_USEFUL_DESCENT_MM = 3;
export const SKID_MM = 3;
export const NO_MOTION_MM = 1;
export const NO_TURN_DEG = 0.5;
export const BLOCKED_SHARE = 0.25;
export const THINNEST_SHARE = 0.04;
export const THINNEST_CAP_MM = 5;
export const CENTROID_NEEDS_CORNERS = 3;
export const ON_OFFSET_SHARE = 0.25;
export const IN_OFFSET_SHARE = 0.5;
export const RELEASE_CLEARANCE_MM = 30;
export const RELEASE_MARGIN_MM = 8;
export const JAW_PAD_MM = 12;
export const FINE_PHASE_MM = 80;
export const FROZEN_CYCLES = 2;
export const MAX_CYCLES = 30;
export const SLIDE_STANDOFF_MM = 90;
export const TILT_REACHES_DEG = 80;
export const ROTATION_STOPPED_SHARE = 0.5;
export const TRAVERSE_MM = 60;
export const CLEAR_MM = 60;
export const TRAVEL_MM = 100;
export const EMPTY_BAND_MM = 3;
export const EMPTY_BAND_FRACTION = 0.04;
/** The tool is "far above" the object when more than this over its top (mm): a close there is a fist, not a grasp. */
export const FAR_ABOVE_MM = 100;

export const r0 = (v: number) => v.toFixed(0);
export const r1 = (v: number) => v.toFixed(1);
const tokens = (s: string) => s.toLowerCase().match(/[a-z][a-z'_-]*/g) ?? [];
const hasAny = (s: string, words: readonly string[]) => {
	const t = ` ${tokens(s).join(" ")} `;
	return words.some((w) => t.includes(` ${w} `));
};

// ---------------------------------------------------------------------------
// words (spec-executor §13.1, §18)

export const HOLDING_WORDS = [
	"holding",
	"hold",
	"holds",
	"held",
	"grasp",
	"grasped",
	"grasping",
	"carry",
	"carrying",
	"carried",
	"gripping",
	"grip",
	"lifted",
	"lift",
];
export const RELEASE_WORDS = [
	"release",
	"released",
	"releasing",
	"let go",
	"lets go",
	"jaws open",
	"gripper open",
	"resting on",
	"resting in",
	"sitting on",
	"sitting in",
	"placed",
	"put down",
	"set down",
	"rests",
];
export const POSITIONAL_WORDS = [
	"over",
	"above",
	"centred",
	"centered",
	"aligned",
	"within",
	"directly",
	"position",
	"positioned",
	"hovering",
	"at the",
];
export const DROPPING_WORDS = ["drop", "dropped", "release", "let go"];
export const PLACING_WORDS = ["place", "placed", "put", "set", "rest", "resting", "stack", "stacked"];
export const SLIDER_WORDS = ["drawer", "door", "lid", "slider", "slide", "knob", "dial", "button", "switch", "handle"];
export const LEANING_WORDS = [
	"push",
	"pushing",
	"press",
	"pressing",
	"slide",
	"sliding",
	"lean",
	"pull",
	"pulling",
	"nudge",
];
export const EDGE_PART_WORDS = ["rim", "edge", "lip", "handle", "wall", "side", "brim"];
export const TURN_WORDS = ["turn", "twist", "rotate", "press"];

const phrase = (s: string, words: readonly string[]) => {
	const t = ` ${tokens(s).join(" ")} `;
	return words.some((w) => t.includes(` ${w} `));
};
/** The criterion wants the thing held (holding words, no release words). */
export const wantsHolding = (criterion: string) =>
	phrase(criterion, HOLDING_WORDS) && !phrase(criterion, RELEASE_WORDS);
/** The criterion wants the thing let go. */
export const wantsRelease = (criterion: string) => phrase(criterion, RELEASE_WORDS);
/** Placing INTO (true) or ONTO (false) by the earliest in/on phrase, ignoring "in the wrist view". */
export function placingIn(text: string): boolean | undefined {
	const t = tokens(text.replace(/in the (wrist|scene|side) (view|camera|picture)/g, " "));
	for (let i = 0; i < t.length; i++) {
		if (["in", "into", "inside", "within"].includes(t[i])) return true;
		if (["on", "onto", "atop", "upon"].includes(t[i])) return false;
	}
	return undefined;
}
export const isPositional = (criterion: string) => phrase(criterion, POSITIONAL_WORDS) && !wantsHolding(criterion);
export const namesEdgePart = (s: string) => phrase(s, EDGE_PART_WORDS);
export const namesSlider = (s: string) => phrase(s, SLIDER_WORDS);
export const leans = (s: string) => phrase(s, LEANING_WORDS);
export const isSideStep = (s: string) => /from the side/i.test(s);
/** The step is about letting go: release, placing or dropping words, or "open" with a jaw word. */
export function aboutLettingGo(s: string): boolean {
	if (phrase(s, RELEASE_WORDS) || phrase(s, PLACING_WORDS) || phrase(s, DROPPING_WORDS)) return true;
	return hasAny(s, ["open", "opens", "opened"]) && hasAny(s, ["jaw", "jaws", "gripper", "finger", "fingers", "grip"]);
}
/** The step acts on what it names (a part is sought): it wants to hold, or has turn / twist / rotate / press. */
export const actsOnWhatItNames = (subgoal: string, criterion: string) =>
	wantsHolding(criterion) || phrase(subgoal, TURN_WORDS) || phrase(criterion, TURN_WORDS);
/** The part named as "X of <target>" in the subgoal sentence. */
export function partOf(subgoal: string, target: string): string | undefined {
	const m = /\b(the\s+)?([a-z]+(?:\s+[a-z]+)?)\s+of\s+(?:the\s+)?([a-z][a-z ]*)/i.exec(subgoal);
	if (!m) return undefined;
	const head = m[3].toLowerCase();
	if (!target || head.includes(target.toLowerCase().split(/\s+/)[0] ?? "")) return m[2].toLowerCase();
	return undefined;
}
/** The receiver's words while placing: what follows on / in / into / onto, else the target. */
export function receiverWords(criterion: string, subgoal: string, target: string): string {
	for (const s of [criterion, subgoal]) {
		const m =
			/\b(?:on|onto|in|into|inside|within|atop|upon)\s+(?:the\s+)?([a-z][a-z0-9 _-]*?)(?:\s+(?:with|while|and|so|until|then|,)|[.,;]|$)/i.exec(
				s.replace(/in the (wrist|scene|side) (view|camera|picture)/gi, " "),
			);
		if (m?.[1]) return m[1].trim();
	}
	return target;
}
/** The part phrase before "from the side". */
export function sidePartWords(subgoal: string, target: string): string {
	const m = /((?:the\s+)?[a-z][a-z ]*?)\s+from the side/i.exec(subgoal);
	return m?.[1]?.replace(/^(grasp|grab|take|hold|approach)\s+/i, "").trim() || target;
}

// ---------------------------------------------------------------------------
// posture

/** The jaw closing line: the tool +Y column less its component along the approach, unit. */
export function jawLine(pose: Pose): Vec3 {
	const R = pose.rotation;
	const y: Vec3 = [R[0][1], R[1][1], R[2][1]];
	const a = approach(pose).axis;
	const flat = sub(y, scale(a, dot(y, a)));
	return norm(flat) < 1e-9 ? [0, 1, 0] : unit(flat);
}
/** Horizontal (a side-take posture): -approach_z <= cos(tilt_reach). */
export const isHorizontal = (pose: Pose, caps: Caps) =>
	-approach(pose).axis[2] <= Math.cos(((caps.tilt_reach_deg || TILT_REACHES_DEG) * Math.PI) / 180);
export const jawHalfGapMm = (caps: Caps) =>
	caps.gripper_open_span_m > 0 ? (caps.gripper_open_span_m * 1000) / 2 : DEFAULT_JAW_HALF_GAP_MM;
export const padMm = (caps: Caps) => (caps.gripper_pad_length_m > 0 ? caps.gripper_pad_length_m * 1000 : JAW_PAD_MM);
/** The gripper's resolution (mm): min(3 mm, 4% of the span). */
export const gripperResolutionMm = (caps: Caps) =>
	Math.min(EMPTY_BAND_MM, EMPTY_BAND_FRACTION * caps.gripper_open_span_m * 1000);
/** The close tolerance (mm): min(half gap, max(half gap - w/2 - 5, 20)). */
export function closeToleranceMm(caps: Caps, objectWidthMm: number | undefined): number {
	const half = jawHalfGapMm(caps);
	if (objectWidthMm === undefined) return half;
	return Math.min(half, Math.max(half - objectWidthMm / 2 - PAD_MARGIN_MM, MIN_CLOSE_TOLERANCE_MM));
}
export const arrivedMm = (caps: Caps) => 2 * caps.arrival_tolerance_m * 1000;

// ---------------------------------------------------------------------------
// state

export type LocatedPoint = {
	words: string;
	point: Vec3;
	method: string;
	error_mm: number;
	error_class: string;
	/** A new answer waiting for a second look to confirm it (the sticky rule). */
	waiting?: Vec3;
	on_face: boolean;
	body?: Vec3;
	top?: Vec3;
	/** The look that gave it, for the record. */
	cameras: CameraName[];
};
export type Payload = {
	width_mm?: number;
	span_mm?: number;
	/** The object's centre along the jaw line, from the tool point (m); null = unknown (shoved before the close). */
	offset?: Vec3 | null;
	edge: boolean;
	bumped: boolean;
	voided_mm?: number;
	tracking: boolean;
};
export type Receiver = { center: Vec3; radius_mm: number; error_mm: number; words: string };

export type SubgoalState = {
	subgoal: string;
	criterion: string;
	target: string;
	wantsHolding: boolean;
	wantsRelease: boolean;
	sideStep: boolean;
	planBoxes: Partial<Record<CameraName, Box>>;
	startPose?: Pose;
	spent: number;
	// locating
	located?: LocatedPoint;
	disagreement?: string;
	ambiguous?: string;
	footprint?: Vec3[];
	footprintMethod?: string;
	footprintError?: number;
	lastWidthMm?: number;
	residualMm?: number;
	errorMm?: number;
	errorClass?: string;
	unshoved: boolean;
	lookedClose: boolean;
	lastLookPose?: Pose;
	lastLookedWords?: string;
	fromPlanBoxes: boolean;
	// grasping
	emptyCloses: Vec3[];
	unrecoveredZ?: number;
	relookedAt: Vec3[];
	relookArranged: boolean;
	relookTrigger?: Vec3;
	closeDepthRefusedAt: Vec3[];
	payload: Payload;
	receiver?: Receiver;
	heldAtStart: boolean | null;
	// motion
	blockedAxes: Vec3[];
	turnStops: Vec3[];
	atTheWall: boolean;
	frozen: number;
	wall: number;
	inheritedFrozen: number;
	askedPosition: boolean;
	lastFinding?: string;
	lastFirstMove?: Delta;
	lastReleaseWidthMm?: number;
	releasedInCode?: { width_mm: number };
};

export function newState(o: {
	subgoal: string;
	criterion: string;
	target: string;
	planBoxes: Partial<Record<CameraName, Box>>;
}): SubgoalState {
	const boxes = { ...o.planBoxes };
	// The wrist moved since the plan: its box is void (spec-executor §7.2 step 1).
	delete boxes.wrist;
	return {
		subgoal: o.subgoal,
		criterion: o.criterion,
		target: o.target,
		wantsHolding: wantsHolding(o.criterion),
		wantsRelease: wantsRelease(o.criterion) || wantsRelease(o.subgoal),
		sideStep: isSideStep(o.subgoal),
		planBoxes: boxes,
		spent: 0,
		unshoved: true,
		lookedClose: false,
		fromPlanBoxes: false,
		emptyCloses: [],
		relookedAt: [],
		relookArranged: false,
		closeDepthRefusedAt: [],
		payload: { edge: false, bumped: false, tracking: false },
		heldAtStart: null,
		blockedAxes: [],
		turnStops: [],
		atTheWall: false,
		frozen: 0,
		wall: 0,
		inheritedFrozen: 0,
		askedPosition: false,
	};
}

/** The object's width (mm) from what was measured: the footprint's longest span, else the last width. */
export const objectWidthMm = (s: SubgoalState) =>
	s.footprint && s.footprint.length >= 2 ? longestAcross(s.footprint) * 1000 : s.lastWidthMm;
/** The object is known to be wider than the jaws. */
export const widerThanJaws = (s: SubgoalState, caps: Caps) => {
	const w = objectWidthMm(s);
	return w !== undefined && w > 2 * jawHalfGapMm(caps);
};
/** The horizontal offset of the located point from the tool (m), or undefined. */
export const miss = (s: SubgoalState, pose: Pose): Vec3 | undefined =>
	s.located ? [s.located.point[0] - pose.position_m[0], s.located.point[1] - pose.position_m[1], 0] : undefined;
export const residualOf = (s: SubgoalState, pose: Pose) => {
	const m = miss(s, pose);
	return m ? Math.hypot(m[0], m[1]) * 1000 : undefined;
};
/** The step is a grasp step: wants holding, names an edge part, or the tool is not far above the object. */
export function isGraspStep(s: SubgoalState, f: Frame, topZ: number | undefined): boolean {
	if (s.wantsHolding || namesEdgePart(s.subgoal)) return true;
	if (topZ === undefined) return true;
	return (f.pose.position_m[2] - topZ) * 1000 <= FAR_ABOVE_MM;
}

// ---------------------------------------------------------------------------
// what the gates see

export type GateContext = {
	state: SubgoalState;
	frame: Frame;
	caps: Caps;
	read: ClearanceRead;
	/** Execution time (the 2nd and later actions of a batch) rather than proposal time. */
	running: boolean;
	/** The object's height over the table (mm), when measured. */
	heightMm?: number;
	/** The top face z (m), when known. */
	topZ?: number;
	/** The descent was blocked in this batch (a blocked segment going down). */
	descentBlocked: boolean;
};
export type Gate = (deltas: Delta[], ctx: GateContext) => string | undefined;

const isDescent = (d: Delta) => d.kind === "translate" && !!d.axis && d.axis[2] < -0.5;
const isTranslate = (d: Delta) => d.kind === "translate" && !!d.axis;
const isClose = (d: Delta) => d.kind === "gripper" && d.gripper_state === "close";
const isOpen = (d: Delta) => d.kind === "gripper" && d.gripper_state === "open";
const hasClose = (deltas: Delta[]) => deltas.some(isClose);
/** The tool after the (non-push, unless `pushes`) translations before index `until`. */
function poseAfterMoves(deltas: Delta[], pose: Pose, until: number, pushes = false): Vec3 {
	let p: Vec3 = [...pose.position_m];
	for (const d of deltas.slice(0, until))
		if (isTranslate(d) && (pushes || !d.push)) p = add(p, scale(d.axis as Vec3, d.magnitude));
	return p;
}

/** A floor over the table the tool may not go below (LIBERO declares none, so this never fires there). */
export const floorGate: Gate = (deltas, { frame, caps }) => {
	const floor = (caps as unknown as { floor_above_table_m?: number | null }).floor_above_table_m;
	if (typeof floor !== "number" || frame.table_z_m === null) return undefined;
	const limit = frame.table_z_m + floor - 0.001;
	let z = frame.pose.position_m[2];
	let lowest = z;
	for (const d of deltas) {
		if (!isTranslate(d)) continue;
		z += (d.axis as Vec3)[2] * d.magnitude;
		lowest = Math.min(lowest, z);
	}
	if (lowest >= limit) return undefined;
	const room = (frame.pose.position_m[2] - limit) * 1000;
	return `the moves would take the tool ${r0((limit - lowest) * 1000)} mm below the floor over the table; it can descend at most ${r0(Math.max(0, room))} mm more`;
};

/** The reach limit in place (./command.ts fitToReach); a cut or a refusal marks the wall. */
export const reachGate: Gate = (deltas, ctx) => {
	const r = fitToReach(deltas, ctx.frame.pose, ctx.caps);
	if (r.shortened || r.refused) ctx.state.atTheWall = true;
	return r.refused;
};

/** The rotation word (with its angle) that turns the approach back down most directly. */
export function turnBackDown(pose: Pose): string {
	const a = approach(pose).axis;
	const down: Vec3 = [0, 0, -1];
	let best = "";
	let bestGain = -Infinity;
	for (const [word, axis] of Object.entries(ROTATION)) {
		if (Math.abs(axis[2]) > 0.5) continue;
		const angle = Math.acos(Math.max(-1, Math.min(1, dot(a, down))));
		const after = apply(rotationAbout(axis, angle), a);
		const gain = -after[2];
		if (gain > bestGain) [best, bestGain] = [`rotate ${word} ${r0((angle * 180) / Math.PI)} deg`, gain];
	}
	return best;
}

/** Pitch and roll only for a side approach: elsewhere a turn must bring the jaws more downward. */
export const tiltGate: Gate = (deltas, { state, frame, caps }) => {
	const tilts = deltas.filter((d) => d.kind === "rotate" && d.axis && Math.abs(d.axis[2]) < 0.9);
	if (!tilts.length) return undefined;
	if (!state.sideStep) {
		let R = frame.pose.rotation;
		for (const d of tilts) R = matmul(rotationAbout(d.axis as Vec3, d.magnitude), R);
		const before = approach(frame.pose).axis[2];
		const after = R[2][2];
		if (after < before - 0.01) return undefined;
		return `pitch and roll are only for taking something from the side; this turn would not point the jaws more downward. To face down again: ${turnBackDown(frame.pose)}`;
	}
	if (state.located) {
		const d = norm(sub(frame.pose.position_m, state.located.point)) * 1000;
		const standoff = SLIDE_STANDOFF_MM - 2 * caps.arrival_tolerance_m * 1000;
		if (d < standoff)
			return `the tool is ${r0(d)} mm from the target; move to open space or to the standoff point (${r0(SLIDE_STANDOFF_MM)} mm off its face) before tilting`;
	}
	return undefined;
};

/** After an empty close: rise 100 mm (95 judged) before any close or descent; near an old empty close, re-look first. */
export const emptyPickupGate: Gate = (deltas, ctx) => {
	const { state, frame } = ctx;
	const z = frame.pose.position_m[2];
	if (state.unrecoveredZ !== undefined) {
		if (z >= state.unrecoveredZ + RECOVERY_JUDGED_M) state.unrecoveredZ = undefined;
		else if (deltas.some((d) => isClose(d) || isDescent(d))) {
			const left = (state.unrecoveredZ + RECOVERY_UP_M - z) * 1000;
			return `the jaws closed on nothing here; recover first: open, move up ${r0(Math.max(1, left))} mm more (to ${r0(RECOVERY_UP_M * 1000)} mm above the empty close), move sideways by the numbers in WHERE THE TARGET IS, then descend and close`;
		}
	}
	if (!hasClose(deltas)) return undefined;
	const near = state.emptyCloses.find((p) => norm(sub(p, frame.pose.position_m)) < EMPTY_CLOSE_RADIUS_M);
	if (!near) return undefined;
	const relooked = state.relookedAt.some((p) => norm(sub(p, frame.pose.position_m)) < EMPTY_CLOSE_RADIUS_M);
	if (!relooked) {
		state.relookArranged = true;
		state.relookTrigger = [...frame.pose.position_m];
		return undefined;
	}
	return `the jaws already closed on nothing within ${r0(EMPTY_CLOSE_RADIUS_M * 1000)} mm of here and a second look did not move the target; change the position or the height before closing again, and say what you changed`;
};

/** The depth the tool point must be below the top face for a grasp (mm): a quarter of the height for a body, the pad on an edge. */
export function graspDepthMm(s: SubgoalState, caps: Caps, heightMm: number): { depth: number; edge: boolean } {
	const edge = namesEdgePart(s.subgoal) || namesEdgePart(s.criterion) || widerThanJaws(s, caps);
	if (edge) return { depth: Math.min(padMm(caps), Math.max(heightMm - padMm(caps) / 2, 0)), edge };
	return { depth: GRASP_BELOW_SHARE * heightMm, edge };
}

/** A close must find the tool deep enough below the top face (spec-executor §10.5). */
export const closeDepthGate: Gate = (deltas, ctx) => {
	const { state, frame, caps, running } = ctx;
	const at = deltas.findIndex(isClose);
	if (at < 0) return undefined;
	const p = poseAfterMoves(deltas, frame.pose, at);
	const grasp = isGraspStep(state, frame, ctx.topZ);
	if (ctx.topZ !== undefined && (p[2] - ctx.topZ) * 1000 > FAR_ABOVE_MM && !grasp) return undefined;
	if (ctx.heightMm === undefined || ctx.topZ === undefined) {
		if (ctx.descentBlocked || !grasp) return undefined;
		const down = jawsDown(frame.pose) || frame.table_z_m === null;
		if (down && (!running || state.relookArranged)) {
			state.relookArranged = true;
			state.relookTrigger = [...frame.pose.position_m];
			return undefined;
		}
		if (!down) {
			const contactDescent = deltas
				.slice(0, at)
				.some((d) => isTranslate(d) && d.push && dot(d.axis as Vec3, approach(frame.pose).axis) > 0.9);
			if (!running && contactDescent) return undefined;
			return "with the jaws tilted, advance along the approach axis with stop_on_contact until something is met, then close in the same reply";
		}
		if (frame.table_z_m === null)
			return "the table height is unknown, so no depth can be judged: descend onto the object until blocked, then close";
		return "the object's top face has never been measured: descend over it until the wrist sees it, then close";
	}
	const { depth, edge } = graspDepthMm(state, caps, ctx.heightMm);
	const belowTop = (ctx.topZ - p[2]) * 1000;
	const slack = LEVEL_SLACK_MM + (running ? caps.arrival_tolerance_m * 1000 : 0);
	const w = objectWidthMm(state);
	const fits =
		w !== undefined &&
		(() => {
			const m = miss(state, frame.pose);
			const off = m ? Math.abs(dot(m, jawLine(frame.pose))) * 1000 : 0;
			return w / 2 + off < jawHalfGapMm(caps);
		})();
	if (belowTop + slack >= depth || (ctx.descentBlocked && fits)) return undefined;
	if (ctx.descentBlocked) {
		if (depth - belowTop <= padMm(caps) && widerThanJaws(state, caps)) return undefined;
		if (state.closeDepthRefusedAt.some((q) => norm(sub(q, frame.pose.position_m)) < EMPTY_CLOSE_RADIUS_M))
			return undefined;
		state.closeDepthRefusedAt.push([...frame.pose.position_m]);
		return `the descent was blocked ${r0(depth - belowTop)} mm short of the grasp depth: lift, look again at the target and descend over it before closing`;
	}
	const floor = edge ? Math.min(2 * depth, Math.max(depth, ctx.heightMm - padMm(caps) / 2)) : ctx.heightMm / 2;
	const already = deltas
		.slice(0, at)
		.filter(isDescent)
		.reduce((s, d) => s + d.magnitude * 1000, 0);
	const total = already + Math.max(floor - belowTop, MIN_USEFUL_DESCENT_MM);
	return `the tool point is ${belowTop >= 0 ? `only ${r0(belowTop)} mm below` : `${r0(-belowTop)} mm above`} the top face (${r0(ctx.heightMm)} mm over the table); a close needs it ${r0(depth)} mm below. Descend ${r0(total)} mm in all from where the tool is now (not an increment on what you already wrote), then close`;
};

/** A turn toward a recorded stop is refused (spec-executor §10.6). */
export const turnGate: Gate = (deltas, { state }) => {
	for (const d of deltas) {
		if (d.kind !== "rotate" || !d.axis) continue;
		const hit = state.turnStops.find((s) => dot(s, d.axis as Vec3) > 0.95);
		if (!hit) continue;
		const other = state.turnStops.some((s) => dot(s, d.axis as Vec3) < -0.95);
		return other
			? "it turned as far as it goes both ways: claim the subgoal done, or grasp another part"
			: "the last turn that way reached a stop; turn the other way";
	}
	return undefined;
};

/** How far off the receiver's centre a release may be (mm). */
export function releaseAllowanceMm(s: SubgoalState, caps: Caps): number | undefined {
	if (!s.receiver) return undefined;
	const inside = placingIn(s.criterion) ?? placingIn(s.subgoal) ?? false;
	let allowed = inside
		? s.receiver.radius_mm * IN_OFFSET_SHARE - (s.payload.width_mm ?? 0) / 2
		: s.receiver.radius_mm * ON_OFFSET_SHARE;
	allowed = Math.max(MIN_CLOSE_TOLERANCE_MM, allowed);
	const receiverOut = Math.hypot(s.receiver.center[0], s.receiver.center[1]) > caps.reach_m;
	if (s.atTheWall && !inside && receiverOut) allowed = Math.max(allowed, s.receiver.radius_mm);
	return allowed;
}
/** The payload's centre offset from the receiver's centre (m, horizontal), from the tool at `p`. */
export function payloadOffset(s: SubgoalState, p: Vec3): Vec3 | undefined {
	if (!s.receiver) return undefined;
	const centre = s.payload.offset ? add(p, s.payload.offset) : p;
	return [s.receiver.center[0] - centre[0], s.receiver.center[1] - centre[1], 0];
}

/** An open over the receiver only when the payload is within the allowance (spec-executor §10.7). */
export const releaseGate: Gate = (deltas, { state, frame, caps }) => {
	const at = deltas.findIndex(isOpen);
	if (at < 0) return undefined;
	if (frame.grasp.holding !== true || isHorizontal(frame.pose, caps) || !state.receiver) return undefined;
	const p = poseAfterMoves(deltas, frame.pose, at, true);
	const off = payloadOffset(state, p);
	const allowed = releaseAllowanceMm(state, caps);
	if (!off || allowed === undefined) return undefined;
	const d = Math.hypot(off[0], off[1]) * 1000;
	if (d <= allowed) return undefined;
	return `letting go cannot be undone: what the jaws hold would land ${r0(d)} mm from the centre of "${state.receiver.words}" (allowed ${r0(allowed)} mm). Move ${sayOffset(off)} over its centre first`;
};

/** The rest of a split first move after a wall grasp runs only while the jaws still hold (spec-executor §10.8). */
export const theTestHeld: Gate = (deltas, { state, frame, caps }) => {
	if (!deltas.some((d) => d.label.includes("(the rest)"))) return undefined;
	if (!state.payload.tracking) return undefined;
	const opening = frame.grasp.opening_mm ?? frame.gripper.opening_mm;
	if (frame.grasp.holding === false || opening <= gripperResolutionMm(caps))
		return "the test move shows the jaws hold nothing now: treat them as empty, open, find the target again and grasp it again";
	return undefined;
};

/** The chain (spec-executor §7.2 step 3), first refusal wins. */
export const GATES: readonly Gate[] = [
	floorGate,
	reachGate,
	tiltGate,
	emptyPickupGate,
	closeDepthGate,
	turnGate,
	releaseGate,
	theTestHeld,
];
export function chainGates(deltas: Delta[], ctx: GateContext, gates: readonly Gate[] = GATES): string | undefined {
	for (const g of gates) {
		const why = g(deltas, ctx);
		if (why) return why;
	}
	return undefined;
}

// ---------------------------------------------------------------------------
// what the measured segments say (spec-executor §13)

/** A close's holding read, with the too-thin rule: holding with a gap under min(5 mm, 4% of the widest) is empty. */
export function closeOutcome(step: SegmentReport, s: SubgoalState): boolean | null {
	const holding = step.grasp?.holding ?? null;
	if (holding !== true) return holding;
	const gap = step.detail?.opening_mm ?? step.grasp?.opening_mm;
	const widest = objectWidthMm(s);
	if (gap !== undefined && widest !== undefined && gap < Math.min(THINNEST_CAP_MM, THINNEST_SHARE * widest))
		return false;
	return true;
}

/** An empty close at a grasp step: remember where, and that the tool must rise 100 mm before closing again. */
export function noteEmptyClose(s: SubgoalState, step: SegmentReport, graspStep: boolean) {
	if (!graspStep || !step.pose_after) return;
	const p = scale(step.pose_after.position_mm, 0.001);
	s.emptyCloses.push(p);
	s.unrecoveredZ = p[2];
}

/** After a grasp (spec-executor §13.3): the payload, its width and offset; the located point is void. */
export function noteGrasped(s: SubgoalState, step: SegmentReport, caps: Caps, pose: Pose, aimedAtEdge: boolean) {
	s.receiver = undefined;
	const gap = step.detail?.opening_mm ?? step.grasp?.opening_mm;
	const width = gap !== undefined ? gap : undefined;
	const measured = objectWidthMm(s);
	const span = width !== undefined ? Math.max(width, measured ?? 0) : measured;
	const edge = aimedAtEdge || (width !== undefined && width < padMm(caps)) || widerThanJaws(s, caps);
	let offset: Vec3 | null | undefined;
	if (edge) {
		const m = miss(s, pose);
		const line = jawLine(pose);
		const along = m ? dot(m, line) : 0;
		if (width !== undefined && Math.abs(along) * 1000 > width / 2 + MIN_CLOSE_TOLERANCE_MM) offset = null;
		else if (widerThanJaws(s, caps) && measured !== undefined) {
			const inset = (measured / 2 - padMm(caps)) / 1000;
			offset = scale(line, Math.sign(along || 1) * inset);
		} else offset = scale(line, along);
	}
	const fresh =
		width !== undefined &&
		width >= MIN_WIDTH_MM &&
		(s.payload.voided_mm === undefined ||
			Math.abs(width - s.payload.voided_mm) > WIDTH_JUMP_SHARE * s.payload.voided_mm);
	s.payload = {
		...(width !== undefined ? { width_mm: width } : {}),
		...(span !== undefined ? { span_mm: span } : {}),
		...(offset !== undefined ? { offset } : {}),
		edge,
		bumped: false,
		tracking: fresh,
	};
	s.located = undefined;
	s.residualMm = undefined;
}

/** The jaws opened: the payload's width is void. */
export function noteOpened(s: SubgoalState) {
	if (s.payload.width_mm !== undefined) s.payload.voided_mm = s.payload.width_mm;
	s.payload.tracking = false;
}

/** A blocked segment going down: the close-in look is due again; every blocked axis is kept this cycle. */
export function noteBlocked(s: SubgoalState, step: SegmentReport) {
	if (step.abort_reason !== "blocked") return;
	if (step.axis) {
		s.blockedAxes.push(step.axis);
		if (step.axis[2] < -0.5) s.lookedClose = false;
	}
}
/** At the cycle's end: only the axes that advanced under 1 mm stay, or every one while holding. */
export function keepBlockedAxes(s: SubgoalState, steps: SegmentReport[], holding: boolean) {
	if (holding) return;
	const kept: Vec3[] = [];
	for (const st of steps)
		if (st.abort_reason === "blocked" && st.axis && Math.abs(st.along_axis_mm) < 1) kept.push(st.axis);
	s.blockedAxes = kept;
}
export const stoppedAlongTheJaws = (s: SubgoalState, pose: Pose) =>
	s.blockedAxes.some((a) => dot(a, approach(pose).axis) > 0.9);

/** This cycle may have moved the object: a push, or a descent that drifted sideways by max(3 mm, the error). */
export function shoved(steps: SegmentReport[], deltas: Delta[], errorMm: number | undefined): boolean {
	if (deltas.some((d) => d.push)) return true;
	const limit = Math.max(SKID_MM, errorMm ?? 0);
	return steps.some((st) => st.kind === "translate" && st.axis && st.axis[2] < -0.5 && st.lateral_mm >= limit);
}

/** Dragging: holding, jaws not horizontal, and a sideways move was blocked. */
export const dragging = (step: SegmentReport, frame: Frame, caps: Caps) =>
	frame.grasp.holding === true &&
	!isHorizontal(frame.pose, caps) &&
	step.abort_reason === "blocked" &&
	!!step.axis &&
	Math.abs(step.axis[2]) < 0.5;

/**
 * Whether the jaws stand where a close left them (spec-executor §13.7): holding says shut; else the
 * closed fraction under a half, else an opening under 20 mm; null when the robot cannot say.
 */
export function jawsShut(f: Frame): boolean | null {
	if (f.grasp.holding === true) return true;
	const fraction = f.grasp.closed_fraction;
	if (typeof fraction === "number" && Number.isFinite(fraction)) return fraction < 0.5;
	const opening = f.grasp.opening_mm ?? f.gripper.opening_mm;
	if (typeof opening === "number" && Number.isFinite(opening)) return opening < 20;
	return null;
}
/**
 * The cycle's command changes what the jaws are doing (spec-executor §13.7): judged by the gripper
 * actions it COMMANDED against the jaws at the batch start, whether or not the batch reached them (a
 * close the depth gate stopped still counts); an unreadable state counts as changed.
 */
export function jawsChanged(deltas: Delta[], before: Frame): boolean {
	const states = deltas.filter((d) => d.kind === "gripper" && d.gripper_state).map((d) => d.gripper_state);
	if (!states.length) return false;
	const shut = jawsShut(before);
	if (shut === null) return true;
	return !states.every((st) => (st === "close") === shut);
}

/**
 * The cycle did not move (spec-executor §13.7): under 1 mm in all, or it stopped early (a blocked
 * segment, or a translation over 3 mm that made under 25% of its way), with under 0.5 deg of turn and
 * the gripper state unchanged.
 */
export function noMotion(steps: SegmentReport[], gripperChanged: boolean): { frozen: boolean; why: string } {
	if (gripperChanged) return { frozen: false, why: "" };
	const moved = steps.reduce((s, st) => s + (st.kind === "translate" ? st.moved_mm : 0), 0);
	const turned = steps.reduce((s, st) => s + st.turned_deg, 0);
	const early = steps.some(
		(st) =>
			st.kind === "translate" &&
			(st.abort_reason === "blocked" ||
				((st.axis ? requestedMm(st) : 0) > 3 && Math.abs(st.along_axis_mm) < BLOCKED_SHARE * requestedMm(st))),
	);
	if (turned >= NO_TURN_DEG) return { frozen: false, why: "" };
	if (moved < NO_MOTION_MM) return { frozen: true, why: `the tool moved ${r1(moved)} mm in all` };
	if (early) return { frozen: true, why: "a move stopped early" };
	return { frozen: false, why: "" };
}
/** The distance a segment asked for (mm), from its label's "... N mm" or its chunking; 0 when unknown. */
export function requestedMm(st: SegmentReport): number {
	const m = /(\d+(?:\.\d+)?) mm/.exec(st.label);
	if (!m) return 0;
	const total = Number(m[1]);
	const seg = /\[(\d+)\/(\d+)\]/.exec(st.label);
	return seg ? total / Number(seg[2]) : total;
}

/** The positional criterion is met: the residual within the threshold and nothing shoved since the look. */
export function arrived(
	s: SubgoalState,
	caps: Caps,
	pose: Pose,
): { arrived: boolean; threshold_mm: number; residual_mm?: number } {
	const placing = s.payload.tracking && s.receiver;
	const threshold = placing ? (releaseAllowanceMm(s, caps) ?? arrivedMm(caps)) : arrivedMm(caps);
	const residual = placing
		? (() => {
				const off = payloadOffset(s, pose.position_m);
				return off ? Math.hypot(off[0], off[1]) * 1000 : undefined;
			})()
		: residualOf(s, pose);
	return {
		arrived: residual !== undefined && residual <= threshold && s.unshoved,
		threshold_mm: threshold,
		residual_mm: residual,
	};
}

/** The turn stop record (spec-executor §10.6): one turn in the reply, holding, no watcher finding, under half the turn, ended limited or blocked. */
export function noteTurnStop(
	s: SubgoalState,
	deltas: Delta[],
	steps: SegmentReport[],
	holding: boolean,
	finding: string | undefined,
) {
	const turns = deltas.filter((d) => d.kind === "rotate");
	if (turns.length !== 1 || !holding || finding) return;
	const asked = (turns[0].magnitude * 180) / Math.PI;
	const got = steps.filter((st) => st.kind === "rotate").reduce((a, st) => a + st.turned_deg, 0);
	const last = steps.filter((st) => st.kind === "rotate").at(-1);
	if (got < ROTATION_STOPPED_SHARE * asked && last && (last.outcome === "limited" || last.abort_reason === "blocked"))
		s.turnStops.push(turns[0].axis as Vec3);
}

// ---------------------------------------------------------------------------
// text blocks the proposer reads (spec-executor §8.2 blocks 2-7)

export function sinceStart(s: SubgoalState, pose: Pose): string {
	if (!s.startPose) return "";
	const d = sub(pose.position_m, s.startPose.position_m);
	return `Since this subgoal started the tool moved [${d.map((v) => r0(v * 1000)).join(", ")}] mm (${r0(norm(d) * 1000)} mm in all).`;
}
export function sinceLastFrame(
	prev: Frame | undefined,
	prevRead: ClearanceRead | undefined,
	f: Frame,
	read: ClearanceRead,
): string {
	if (!prev) return "This is the first frame of the subgoal.";
	const parts: string[] = [];
	const d = sub(f.pose.position_m, prev.pose.position_m);
	if (norm(d) * 1000 >= 1) parts.push(`moved [${d.map((v) => r0(v * 1000)).join(", ")}] mm`);
	if (
		prevRead?.clearance_mm !== undefined &&
		read.clearance_mm !== undefined &&
		Math.abs(read.clearance_mm - prevRead.clearance_mm) >= 1
	)
		parts.push(
			`clearance ${read.clearance_mm > prevRead.clearance_mm ? "grew" : "shrank"} by ${r0(Math.abs(read.clearance_mm - prevRead.clearance_mm))} mm`,
		);
	if (prev.grasp.holding !== f.grasp.holding)
		parts.push(`holding went from ${String(prev.grasp.holding)} to ${String(f.grasp.holding)}`);
	return parts.length
		? `Since the last frame: ${parts.join("; ")}.`
		: "Since the last frame: nothing measurable changed.";
}

/** Block 4 (spec-executor §8.2): where the tool is against the target's top face, and the depth a close needs. */
export function whereTheToolIs(
	s: SubgoalState,
	f: Frame,
	caps: Caps,
	heightMm: number | undefined,
	topZ: number | undefined,
	heightFrom: string,
	read: ClearanceRead,
): string {
	if (f.table_z_m === null)
		return "WHERE THE TOOL IS: no table height is known, so no depth can be judged; every close will be refused until the object is descended onto and blocked.";
	if (heightMm === undefined || topZ === undefined) {
		const c = read.clearance_mm;
		return `WHERE THE TOOL IS: the target's height over the table is not measured yet${c !== undefined ? `; the ${r0(c)} mm under the tool is a clearance, not a depth` : ""}; no top face to refer to.`;
	}
	const below = (topZ - f.pose.position_m[2]) * 1000;
	const { depth, edge } = graspDepthMm(s, caps, heightMm);
	return `WHERE THE TOOL IS: the target stands ${r0(heightMm)} mm over the table (${heightFrom}). The tool point is ${below >= 0 ? `${r0(below)} mm BELOW` : `${r0(-below)} mm ABOVE`} its top face. To close on it the tool point must be at least ${r0(depth)} mm below the top face${edge ? " (an edge grasp: the whole pad must go down over it)" : ""}.`;
}

/** Block 5: WHERE THE TARGET IS. */
export function whereTheTargetIs(s: SubgoalState, f: Frame, caps: Caps, topZ: number | undefined): string {
	if (!s.located) return "";
	const placing = s.payload.tracking && s.receiver;
	const from = placing && s.payload.offset ? add(f.pose.position_m, s.payload.offset) : f.pose.position_m;
	const off: Vec3 = [s.located.point[0] - from[0], s.located.point[1] - from[1], 0];
	const lines = [
		`WHERE THE TARGET IS: "${s.located.words}" is ${sayOffset(off)} of the ${placing ? "centre of what the jaws hold" : "tool point"}.`,
	];
	if (!placing && topZ !== undefined) {
		const dz = (topZ - f.pose.position_m[2]) * 1000;
		lines.push(`Its top face is ${r0(Math.abs(dz))} mm ${dz > 0 ? "above" : "below"} the tool point.`);
	}
	lines.push("These are measurements: copy the distances and direction words as written.");
	const here = Math.hypot(f.pose.position_m[0], f.pose.position_m[1]) * 1000;
	lines.push(
		`Reach: ${r0(caps.reach_m * 1000)} mm from the base; the tool is ${r0(here)} mm out; a move past the reach is cut short.`,
	);
	lines.push(
		`Within ${r0(arrivedMm(caps))} mm the tool is already there: a correction that small cannot be measured, so descend and use the jaws instead.`,
	);
	if (s.payload.edge || namesEdgePart(s.subgoal) || widerThanJaws(s, caps)) {
		lines.push(
			widerThanJaws(s, caps)
				? "The object is wider than the jaws: the offset above already aims at its edge; write it as given, do not add to it. If neither edge is reachable, yaw 90 deg first."
				: "The offset above already includes the edge offset; write it as given.",
		);
	}
	lines.push(`(from ${s.located.method}, within about ${r0(s.located.error_mm)} mm)`);
	if (s.disagreement) lines.push(`BUT TWO VIEWS DISAGREE: ${s.disagreement}`);
	if (s.ambiguous) lines.push(`BUT MORE THAN ONE THING FITS THOSE WORDS: ${s.ambiguous}`);
	return lines.join("\n");
}

/** Block 6: the side-take notes, when they apply. */
export function sideTakeBlock(s: SubgoalState, f: Frame, caps: Caps): { text: string; cannot?: string } {
	const horizontal = isHorizontal(f.pose, caps);
	const down = jawsDown(f.pose);
	if (f.grasp.holding === true && horizontal)
		return {
			text: "SIDE TAKE: the jaws hold something sideways. Pull straight back along the approach axis (the opposite of the direction the jaws face), neither up nor turning, until it stops moving; then open.",
		};
	if (!down && !s.sideStep)
		return {
			text: `SIDE TAKE: the jaws are tilted but this step is not a side take. ${horizontal ? "Back off along the approach first, then" : "Then"} ${turnBackDown(f.pose)} to face down again.`,
		};
	if (s.sideStep) {
		if (!s.located) return { text: "SIDE TAKE: the face to approach is not located yet." };
		const a = approach(f.pose).axis;
		const face = s.located.point;
		const toFace: Vec3 = [face[0] - f.pose.position_m[0], face[1] - f.pose.position_m[1], 0];
		const dist = norm(toFace);
		if (dist < 1e-6)
			return { text: "", cannot: "the tool is right over the face: a side take has no direction to come from" };
		const dir = unit(toFace);
		const standoff = scale(dir, -SLIDE_STANDOFF_MM / 1000);
		const stop: Vec3 = add(face, standoff);
		const height = f.table_z_m !== null ? (stop[2] - f.table_z_m) * 1000 : undefined;
		if (height !== undefined && height < caps.tilted_floor_m * 1000)
			return {
				text: "",
				cannot: `the face is ${r0(height)} mm over the table, below the ${r0(caps.tilted_floor_m * 1000)} mm the tilted tool can reach`,
			};
		const word = ["forward", "backward", "left", "right"].reduce((b, w) =>
			dot({ forward: [1, 0, 0], backward: [-1, 0, 0], left: [0, 1, 0], right: [0, -1, 0] }[w] as Vec3, dir) >
			dot({ forward: [1, 0, 0], backward: [-1, 0, 0], left: [0, 1, 0], right: [0, -1, 0] }[b] as Vec3, dir)
				? w
				: b,
		);
		const turn = Math.abs(a[2]) > 0.5 ? "pitch 90 deg" : "no tilt";
		return {
			text: `SIDE TAKE: face it toward ${word.toUpperCase()}. Standoff point [${stop.map((v) => r0(v * 1000)).join(", ")}] mm (${r0(SLIDE_STANDOFF_MM)} mm off the face). Needed: ${turn} so the jaws point ${word}; then move ${word} with stop_on_contact for ${r0(SLIDE_STANDOFF_MM + 2 * padMm(caps))} mm (the standoff plus two pads) and close in the same reply.`,
		};
	}
	return { text: "" };
}

/** Block 7: the payload in the jaws. */
export function payloadBlock(s: SubgoalState, caps: Caps, pose: Pose): string {
	if (!s.payload.tracking) return "";
	const lines = [
		"IN THE JAWS: nobody measured how deep the object hangs below the tool point, so the clearance under IT is unknown.",
	];
	if (s.receiver) {
		const off = payloadOffset(s, pose.position_m);
		const allowed = releaseAllowanceMm(s, caps);
		if (off && allowed !== undefined)
			lines.push(
				`It is ${r0(Math.hypot(off[0], off[1]) * 1000)} mm from the centre of "${s.receiver.words}" (${sayOffset(off)}); a release may be off by ${r0(allowed)} mm (${(placingIn(s.criterion) ?? placingIn(s.subgoal)) ? "half the receiver's radius less half the object" : "a quarter of the receiver's radius"}, at least ${MIN_CLOSE_TOLERANCE_MM} mm).`,
			);
	}
	return lines.join("\n");
}

/** The limits text (spec-executor §8.5). */
export function forbidText(s: SubgoalState, gripperOnly: boolean, finePhase: boolean): string {
	const lines: string[] = [];
	if (gripperOnly) lines.push("Only a gripper action, or claiming the subgoal done. Do not move the arm.");
	else if (s.lastFinding === "wrong_direction" && s.lastFirstMove)
		lines.push(`The last move (${s.lastFirstMove.label}) went the wrong way: do not propose it again.`);
	else if (s.lastFinding === "overshoot" && s.lastFirstMove)
		lines.push(`The last move (${s.lastFirstMove.label}) overshot: go the opposite way, by less than last time.`);
	if (finePhase && !gripperOnly)
		lines.push(
			"Fine phase: one correction per reply. One correction may be several actions: at most one move per axis (sideways first), then a descent, then a gripper action; a second correction on the same axis waits for the next look.",
		);
	return lines.join("\n");
}

/** The fine phase: the clearance within 80 mm and the residual unknown or within 80 mm. */
export const finePhase = (read: ClearanceRead, residualMm: number | undefined) =>
	read.clearance_mm !== undefined &&
	read.clearance_mm <= FINE_PHASE_MM &&
	(residualMm === undefined || residualMm <= FINE_PHASE_MM);

/** A label for an inserted move (the executor's own forms, ./command.ts). */
export const moveLabel = (direction: string, mm: number) =>
	actionLabel({ type: "move", direction, distance_mm: mm, frame: "base" });
export const fmt = pyFixed;
export const posOf = (p: Vec3) => apartXy(p, [0, 0, 0]);
