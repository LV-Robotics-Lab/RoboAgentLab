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
import { aPartWord } from "./locate.ts";
import type { ClearanceRead } from "./observe.ts";
import type { Caps, Delta, Pose, Vec3 } from "./types.ts";
import { objectHead, stripPlace } from "./words.ts";

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
export const TURN_WORDS = [
	"turn",
	"turns",
	"turned",
	"turning",
	"twist",
	"twists",
	"twisted",
	"twisting",
	"rotate",
	"rotates",
	"rotated",
	"rotating",
	"press",
];

const phrase = (s: string, words: readonly string[]) => {
	const t = ` ${tokens(s).join(" ")} `;
	return words.some((w) => t.includes(` ${w} `));
};
/**
 * The criterion wants the thing held (holding words, no release words). The place phrases that
 * describe the target's own position ("the bowl sitting on the grey ramekin") are not release words
 * (round 2 defect 3): classified on the text with them taken out.
 */
export const wantsHolding = (criterion: string, target = "") => {
	const t = stripPlace(criterion, target);
	return phrase(t, HOLDING_WORDS) && !phrase(t, RELEASE_WORDS);
};
/** The criterion wants the thing let go. */
export const wantsRelease = (criterion: string, target = "") => phrase(stripPlace(criterion, target), RELEASE_WORDS);
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
/** The step turns what the jaws hold (a knob, a dial): turn / twist / rotate words in its sentence or criterion. */
export const isTurnStep = (subgoal: string, criterion: string) =>
	phrase(subgoal, TURN_WORDS) || phrase(criterion, TURN_WORDS);
/** The step acts on what it names (a part is sought): it wants to hold, or has turn / twist / rotate / press. */
export const actsOnWhatItNames = (subgoal: string, criterion: string, target = "") =>
	wantsHolding(criterion, target) || isTurnStep(subgoal, criterion);
/** Modifiers that say which part, not what the thing is ("the near side", "the top drawer"). */
const PART_MODIFIERS = new Set([
	"near",
	"far",
	"front",
	"back",
	"left",
	"right",
	"top",
	"bottom",
	"upper",
	"lower",
	"outer",
	"inner",
]);
const PART_ARTICLES = new Set(["the", "a", "an", "its", "this", "that"]);
const stem = (w: string) => w.replace(/s$/, "");
/**
 * The part the step says the jaws close on, as "<part> of <the target>" in its sentence: a word of the
 * part vocabulary (locate's side and top parts, the edge parts) followed by "of", whose owner (the next
 * eight words) names the target's head noun. Round 3 defect 1: the vocabulary's "descend the REST of
 * the way … the PART of the box" matched the old "X of Y" regex as the part "rest", and every take step
 * then located a part that does not exist (102 of 134 take looks doubtful). "rest", "part", "way" and
 * "middle" are not parts; a part OF something else is where the thing stands, not its part.
 */
export function partOf(subgoal: string, target: string): string | undefined {
	const said = tokens(subgoal);
	const head = new Set(
		tokens(objectHead(target))
			.filter((w) => w.length > 2 && !PART_ARTICLES.has(w) && !PART_MODIFIERS.has(w))
			.map(stem),
	);
	for (let i = 0; i + 1 < said.length; i++) {
		if (said[i + 1] !== "of") continue;
		const part = aPartWord(said[i]) ?? (EDGE_PART_WORDS.includes(stem(said[i])) ? stem(said[i]) : undefined);
		if (!part) continue;
		const whose = new Set(said.slice(i + 2, i + 10).map(stem));
		if (head.size && ![...head].some((w) => whose.has(w))) continue;
		return part;
	}
	return undefined;
}
/**
 * The receiver's words while placing (spec-executor §9): what follows on / in / into / onto / over /
 * above / toward (the vocabulary's carry criterion says "over") in the criterion, then the subgoal,
 * else "" — nothing to seek. The held thing's own place phrases are taken out first, and a phrase the
 * sentence goes on from with "is / are" is part of the subject's name, not the receiver ("the black bowl
 * on the ramekin is placed on the plate" → plate; round 2 defect 3 sent bowls back to the ramekin).
 * "in the picture / view / image" is a camera, not a place. Round 3 defect 4: the fallback was the
 * target itself, so a knob-turn step located the knob in the jaws from a fixed view, 190–669 mm off,
 * and the model drove there; the original returns "" and no look is taken.
 */
export function receiverWords(criterion: string, subgoal: string, target: string): string {
	const re =
		/\b(?:on|onto|in|into|inside|within|atop|upon|over|above|toward|towards)\s+(?:the\s+)?([a-z][a-z0-9 _-]*?)(\s+(?:is|are|was|were|with|while|and|so|until|then|,)|[.,;]|$)/gi;
	for (const text of [criterion, subgoal]) {
		const t = stripPlace(text, target).replace(
			/in the (?:(?:wrist|scene|side) )?(?:view|camera|picture|image|frame)\b/gi,
			" ",
		);
		for (const m of t.matchAll(re)) {
			if (/^\s+(is|are|was|were)$/i.test(m[2])) continue;
			if (m[1]?.trim()) return m[1].trim();
		}
	}
	return "";
}
/** The part phrase before "from the side". */
/**
 * The part phrase a side-take step names, the words a camera can box it by: "the <part> of <what it is
 * part of>" up to the clause's end (a comma, "and", "then", "from the side"), with the modifiers before
 * the part word back to its article ("the middle drawer handle of the dark cabinet"); else the words
 * before "from the side" when they are not a pronoun; else the target. Round 3 defect 3: "approach it
 * from the side" gave "it", and the first look read the planner's box of the whole cabinet onto the
 * table instead of asking for the handle.
 */
export function sidePartWords(subgoal: string, target: string): string {
	const text = subgoal.replace(/\s+/g, " ").trim();
	const phraseRe = /\b([a-z]+) of (.+?)(?=[,;.]| and\b| then\b| from the side\b|$)/gi;
	for (const m of text.matchAll(phraseRe)) {
		const part =
			aPartWord(m[1]) ?? (EDGE_PART_WORDS.includes(stem(m[1].toLowerCase())) ? stem(m[1].toLowerCase()) : undefined);
		if (!part) continue;
		// Back from the part word over up to three modifiers to its article.
		const before = text.slice(0, m.index).split(/\s+/).filter(Boolean);
		const mods: string[] = [];
		for (let i = before.length - 1; i >= 0 && mods.length <= 3; i--) {
			const w = before[i].toLowerCase();
			if (PART_ARTICLES.has(w)) {
				mods.unshift(before[i]);
				break;
			}
			if (
				!/^[a-z-]+$/i.test(before[i]) ||
				/^(to|the|over|at|on|with|gripper|jaws|tool|move|take|grasp|hold)$/i.test(w)
			)
				break;
			mods.unshift(before[i]);
		}
		const lead = mods.length && PART_ARTICLES.has(mods[0].toLowerCase()) ? mods : ["the", ...mods];
		return `${lead.join(" ")} ${m[1]} of ${m[2].trim()}`.replace(/\s+/g, " ");
	}
	// The noun phrase from its last article before "from the side"; a pronoun names nothing to look for.
	const m =
		/.*\b((?:the|a|an|its)\s+[a-z][a-z ]*?)\s+from the side/i.exec(text) ??
		/((?:the\s+)?[a-z][a-z ]*?)\s+from the side/i.exec(text);
	const words = m?.[1]?.replace(/^(grasp|grab|take|hold|approach)\s+/i, "").trim() ?? "";
	return words && !/^(it|them|this|that|one)$/i.test(words) ? words : target;
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
	/** One view with a wide bar, or two views that disagreed (spec-locate §5 item 8): not a place to drive to blindly. */
	doubtful?: boolean;
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
	/** The footprint is a named part's own box (its edge is the part's), not the whole thing's outline. */
	footprintIsPart: boolean;
	footprintError?: number;
	lastWidthMm?: number;
	residualMm?: number;
	errorMm?: number;
	errorClass?: string;
	unshoved: boolean;
	lookedClose: boolean;
	lastLookPose?: Pose;
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
	/** A side take's face, read once a subgoal: the level direction word it looks out along, and the turn that points the jaws into it. */
	sideFace?: { out?: Vec3; turn?: string };
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
		wantsHolding: wantsHolding(o.criterion, o.target),
		wantsRelease: wantsRelease(o.criterion, o.target) || wantsRelease(o.subgoal, o.target),
		sideStep: isSideStep(o.subgoal),
		planBoxes: boxes,
		spent: 0,
		unshoved: true,
		lookedClose: false,
		footprintIsPart: false,
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

/**
 * The sticky rule (spec-locate §8.3) on a new answer for the words a point is kept under: the answer
 * replaces the kept point when its bar is no wider (a doubtful answer never displaces a confident one
 * by its bar alone), when it confirms the answer left waiting by the last look (within max(bar, 1 mm)),
 * or, after an empty close aimed from the kept point, when it lands within the kept bar (`emptyNear`).
 * Otherwise the kept point stands, the new one waits for a second look, and the method says so. Round
 * 3 defect 3: a wrist-only look replaced the point unconditionally, so a doubtful part read 320 mm off
 * (object t8_s1 c1) sent the tool 320 mm left; a wrist look is an answer like any other now (the
 * follow move after it then has nothing to follow when the answer was not taken).
 */
export function takesOver(
	old: LocatedPoint | undefined,
	answer: { point: Vec3; error_mm?: number; doubtful: boolean; method: string },
	emptyNear: boolean,
): { point: Vec3; method: string; jumped?: number } {
	if (!old) return { point: answer.point, method: answer.method };
	const bar = answer.error_mm ?? Number.POSITIVE_INFINITY;
	const apart = apartXy(answer.point, old.point) * 1000;
	if (emptyNear && apart <= old.error_mm) return { point: answer.point, method: answer.method };
	const noWider = bar <= old.error_mm && !(answer.doubtful && !old.doubtful);
	const confirms = old.waiting !== undefined && apartXy(answer.point, old.waiting) * 1000 <= Math.max(bar, 1);
	if (noWider || confirms) return { point: answer.point, method: answer.method };
	old.waiting = answer.point;
	return {
		point: old.point,
		jumped: apart,
		method: `${old.method}; a new answer ${Math.round(apart)} mm away with a ${answer.doubtful && !old.doubtful ? "doubtful" : "wider"} bar (${Math.round(Number.isFinite(bar) ? bar : 0)} vs ${Math.round(old.error_mm)} mm) was not taken`,
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
/** The horizontal offset of the located point (the object's middle) from the tool (m), or undefined. */
export const centreMiss = (s: SubgoalState, pose: Pose): Vec3 | undefined =>
	s.located ? [s.located.point[0] - pose.position_m[0], s.located.point[1] - pose.position_m[1], 0] : undefined;

/** A pad further in than the outline's edge: an outline is the whole thing's, not a named part's own box. */
export const OUTLINE_EDGE_PAD_MM = 12;
/** The step takes hold of the object or approaches it to be taken (not a step that leans on it: a push, a press). */
export const aimsToTake = (s: SubgoalState) =>
	s.wantsHolding || namesEdgePart(s.subgoal) || namesEdgePart(s.target) || !leans(s.subgoal);
/**
 * How far off the object's middle the jaws must aim along the line they close along (mm), and why
 * (spec-executor §8.2 block 5 "the offset is already taken off"): a thing wider than the jaws open can
 * only be held by its wall, half a pad inside its edge (a pad further in when the width is the whole
 * outline's). Not while placing (the middle is where the payload goes), nor for a step that leans on it.
 */
export function edgeOffsetMm(s: SubgoalState, caps: Caps): { mm: number; why: string } | undefined {
	if (s.payload.tracking && s.receiver) return undefined;
	if (!aimsToTake(s)) return undefined;
	const width = objectWidthMm(s);
	if (width === undefined || !widerThanJaws(s, caps)) return undefined;
	const inward = padMm(caps) / 2 + (s.footprintIsPart ? 0 : OUTLINE_EDGE_PAD_MM);
	return {
		mm: Math.max(0, width / 2 - inward),
		why: `it is about ${r0(width)} mm across and the jaws open ${r0(2 * jawHalfGapMm(caps))} mm, so no close centred on it can get round it: the jaws have to straddle its wall${s.footprintIsPart ? "" : ", aimed a pad further in because that width is the whole outline, not a named part's own box"}`,
	};
}
/**
 * The point on the object's edge to put the tool over (m): the located middle shifted along the jaw
 * line toward the side the tool is already on (nothing crosses the thing), unless that rim is past
 * the reach and the other is not. Undefined when the middle is the aim.
 */
export function edgePoint(s: SubgoalState, pose: Pose, caps: Caps): Vec3 | undefined {
	if (!s.located) return undefined;
	const e = edgeOffsetMm(s, caps);
	if (!e || e.mm <= 0) return undefined;
	const axis = jawLine(pose);
	const middle = s.located.point;
	let side = dot(sub(pose.position_m, middle), axis) >= 0 ? 1 : -1;
	const radius = (sign: number) => {
		const p = add(middle, scale(axis, (sign * e.mm) / 1000));
		return Math.hypot(p[0], p[1]);
	};
	if (radius(side) > caps.reach_m && radius(-side) <= caps.reach_m) side = -side;
	return add(middle, scale(axis, (side * e.mm) / 1000));
}
/** Where the tool has to get to: the edge point of a thing wider than the jaws, else the located point (a side take's stand-off is the side block's). */
export const aimPoint = (s: SubgoalState, pose: Pose, caps: Caps): Vec3 | undefined =>
	s.located ? (s.sideStep ? s.located.point : (edgePoint(s, pose, caps) ?? s.located.point)) : undefined;
/**
 * The horizontal offset of the aim point from the tool (m), or undefined. While placing, from the
 * centre of what the jaws hold (the tool plus the payload offset), as the target block says it: the
 * residual is then "the tool — or what it carries — from the receiver" (round 2 defect 2b: the
 * claim read 32 mm from the tool with the bowl's centre 3 mm from the plate's).
 */
export const miss = (s: SubgoalState, pose: Pose, caps: Caps): Vec3 | undefined => {
	const aim = aimPoint(s, pose, caps);
	if (!aim) return undefined;
	const from =
		s.payload.tracking && s.receiver && s.payload.offset ? add(pose.position_m, s.payload.offset) : pose.position_m;
	return [aim[0] - from[0], aim[1] - from[1], 0];
};
export const residualOf = (s: SubgoalState, pose: Pose, caps: Caps) => {
	const m = miss(s, pose, caps);
	return m ? Math.hypot(m[0], m[1]) * 1000 : undefined;
};
/**
 * The quarter turn that brings a rim within reach: when neither rim along the line the jaws close
 * along is inside the arm's reach and, turned a quarter about the vertical, the nearer one is.
 */
export function quarterTurnForReach(s: SubgoalState, pose: Pose, caps: Caps): string | undefined {
	const e = edgeOffsetMm(s, caps);
	if (!s.located || !e || e.mm <= 0) return undefined;
	const axis = jawLine(pose);
	const along: [number, number] = [axis[0], axis[1]];
	const n = Math.hypot(along[0], along[1]);
	if (n < 0.5) return undefined;
	const line: [number, number] = [along[0] / n, along[1] / n];
	const middle = s.located.point;
	const nearer = (l: [number, number]) =>
		Math.min(
			...[1, -1].map((sign) =>
				Math.hypot(middle[0] + (sign * e.mm * l[0]) / 1000, middle[1] + (sign * e.mm * l[1]) / 1000),
			),
		);
	if (nearer(line) <= caps.reach_m || nearer([-line[1], line[0]]) > caps.reach_m) return undefined;
	return 'rotate "yaw_left" 90';
}
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
	// A descent stopped dead has the pads resting on the thing: in this batch, or in the last cycle (the
	// blocked axis kept across cycles, spec-executor §13.4). Round 3 defect 2 read only this batch: the
	// cycle after a contact descent had its close refused for an unmeasured top face, a wrist re-look
	// arranged, and the handle chased from the re-look's doubtful point (B3: 165 such refusals).
	const stopped = ctx.descentBlocked || stoppedAlongTheJaws(state, frame.pose);
	if (ctx.heightMm === undefined || ctx.topZ === undefined) {
		if (stopped || !grasp) return undefined;
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
			const m = centreMiss(state, frame.pose);
			const off = m ? Math.abs(dot(m, jawLine(frame.pose))) * 1000 : 0;
			return w / 2 + off < jawHalfGapMm(caps);
		})();
	if (belowTop + slack >= depth || (stopped && fits)) return undefined;
	if (stopped) {
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
		const m = centreMiss(s, pose);
		const line = jawLine(pose);
		const along = m ? dot(m, line) : 0;
		// A middle further off than the THING's half width (its outline, not the gap a wall grip settles at) plus the
		// error a location carries was shoved before the close: where it hangs is unknown. Of a thing wider than
		// the jaws the middle is half its width in from where the jaws aim, less half a pad (and the outline pad).
		const half = (span ?? width ?? 0) / 2;
		if (Math.abs(along) * 1000 > half + MIN_CLOSE_TOLERANCE_MM) offset = null;
		else if (widerThanJaws(s, caps) && measured !== undefined) {
			const inset = (measured / 2 - (padMm(caps) / 2 + (s.footprintIsPart ? 0 : OUTLINE_EDGE_PAD_MM))) / 1000;
			offset = scale(line, Math.sign(along || 1) * Math.max(0, inset));
		} else offset = scale(line, along);
	}
	// The payload is tracked from the close that holds, whatever the gap (the original's "placing" is the
	// holding reading alone): a 7.6 mm rim in the jaws is a payload, and the next step then seeks the
	// RECEIVER. MIN_WIDTH_MM / WIDTH_JUMP_SHARE judge a measured outline's width, not whether something is held.
	s.payload = {
		...(width !== undefined ? { width_mm: width } : {}),
		...(span !== undefined ? { span_mm: span } : {}),
		...(offset !== undefined ? { offset } : {}),
		edge,
		bumped: false,
		tracking: true,
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
/**
 * At the cycle's end: the blocked axes this cycle leaves for the next (spec-executor §13.4): those
 * that advanced under 1 mm, every one while holding, and the batch's last move (the tool stands where
 * the block left it: a 50 mm contact descent that met the handle after 32 mm is the pads resting on
 * it). They stand until the next cycle's end recomputes them; a cycle that ran no move leaves none.
 */
export function keepBlockedAxes(s: SubgoalState, steps: SegmentReport[], holding: boolean) {
	const lastMove = steps.filter((st) => st.kind === "translate").at(-1);
	s.blockedAxes = steps
		.filter(
			(st) =>
				st.abort_reason === "blocked" &&
				st.axis &&
				(holding || Math.abs(st.along_axis_mm) < NO_MOTION_MM || st === lastMove),
		)
		.map((st) => st.axis as Vec3);
}
export const stoppedAlongTheJaws = (s: SubgoalState, pose: Pose) =>
	s.blockedAxes.some((a) => dot(a, approach(pose).axis) > 0.9);

/**
 * This cycle may have moved the object: a sideways push, or a descent that drifted sideways by max(3 mm,
 * the error). A descent that presses through contact is not a push of the object: counting it made a
 * take step look again every cycle while the tool stood on the bottle's cap, and each look, with the
 * tool over the thing, moved the anchor toward the tool (round 2 defect 4: 668 → 714 mm against a
 * truth of about 654). The point stays anchored to the frame that located it; a lateral skid still counts.
 */
export function shoved(steps: SegmentReport[], deltas: Delta[], errorMm: number | undefined): boolean {
	if (deltas.some((d) => d.push && !(d.kind === "translate" && d.axis && Math.abs(d.axis[2]) >= 0.5))) return true;
	const limit = Math.max(SKID_MM, errorMm ?? 0);
	return steps.some((st) => st.kind === "translate" && st.axis && st.axis[2] < -0.5 && st.lateral_mm >= limit);
}

/**
 * What the payload met (spec-executor §11.1, the original's `_payload_met`): holding, a move of this cycle
 * that did not run to its end met something with what the jaws hold; it stays met across cycles (and
 * subgoals, through the carried payload) until a cycle rises more than 1 mm. Not holding, nothing changes.
 */
export function payloadMet(steps: SegmentReport[], before: boolean, holding: boolean): boolean {
	if (!holding) return before;
	const rose = steps.some((st) => st.axis && st.axis[2] > 0.9 && st.along_axis_mm > NO_MOTION_MM);
	const met = steps.some((st) => st.axis && st.outcome !== "done");
	return met || (before && !rose);
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
 * The cycle did not move (spec-executor §13.7): under 1 mm in all, or it stopped short (a blocked
 * segment, or the WHOLE batch made under 25% of the translation its deltas asked for, when they asked
 * for over 3 mm), with under 0.5 deg of turn and the jaws unchanged. Judged over the batch, as the
 * original does: a 100 mm lift split into a 17 mm test and the rest is one asked distance, not a
 * segment that "made 14 of 100 mm".
 */
export function noMotion(
	steps: SegmentReport[],
	deltas: Delta[],
	gripperChanged: boolean,
): { frozen: boolean; why: string } {
	if (gripperChanged) return { frozen: false, why: "" };
	const moved = steps.reduce((s, st) => s + (st.kind === "translate" ? Math.abs(st.moved_mm) : 0), 0);
	const turned = steps.reduce((s, st) => s + Math.abs(st.turned_deg), 0);
	if (turned >= NO_TURN_DEG) return { frozen: false, why: "" };
	if (moved < NO_MOTION_MM) return { frozen: true, why: `the tool moved ${r1(moved)} mm in all` };
	if (steps.some((st) => st.abort_reason === "blocked"))
		return { frozen: true, why: "a move stopped early: a segment was blocked" };
	const asked = deltas.reduce((s, d) => s + (d.kind === "translate" ? d.magnitude * 1000 : 0), 0);
	const got = steps.reduce((s, st) => s + Math.abs(st.along_axis_mm), 0);
	if (asked > MIN_USEFUL_DESCENT_MM && got < BLOCKED_SHARE * asked)
		return {
			frozen: true,
			why: `a move stopped early: the batch made ${r1(got)} of the ${r1(asked)} mm it asked for`,
		};
	return { frozen: false, why: "" };
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
		: residualOf(s, pose, caps);
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
	// The aim: the edge point of a thing wider than the jaws (the offset "already taken off"), else the middle.
	const aim = aimPoint(s, f.pose, caps) ?? s.located.point;
	const off: Vec3 = [aim[0] - from[0], aim[1] - from[1], 0];
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
	const edge = placing ? undefined : edgeOffsetMm(s, caps);
	if (edge) {
		lines.push(
			`The ${r0(edge.mm)} mm the jaws need to sit off its middle, along the line they close along (${edge.why}), is already taken off those numbers: they are what is LEFT to move. Write them as given and add nothing to them.`,
		);
		const turn = quarterTurnForReach(s, f.pose, caps);
		if (turn)
			lines.push(
				`NEITHER RIM along that line is inside this arm's reach. ${turn} turns the jaws to close along the other line, whose near rim is: turn first, and these offsets are then measured for it.`,
			);
	} else if (s.payload.edge || namesEdgePart(s.subgoal)) {
		lines.push("The offset above already includes the edge offset; write it as given.");
	}
	lines.push(`(from ${s.located.method}, within about ${r0(s.located.error_mm)} mm)`);
	if (s.disagreement) lines.push(`BUT TWO VIEWS DISAGREE: ${s.disagreement}`);
	if (s.ambiguous) lines.push(`BUT MORE THAN ONE THING FITS THOSE WORDS: ${s.ambiguous}`);
	return lines.join("\n");
}

const LEVEL_WORDS: [string, Vec3][] = [
	["forward", [1, 0, 0]],
	["backward", [-1, 0, 0]],
	["left", [0, 1, 0]],
	["right", [0, -1, 0]],
];
/** The level direction word a horizontal vector points along most. */
const levelWord = (v: Vec3) => LEVEL_WORDS.reduce((b, w) => (dot(w[1], v) > dot(b[1], v) ? w : b))[0];
/**
 * Which way the upright face a part stands out of faces: of the four level direction words, the one
 * pointing most from the middle of the thing it belongs to toward the part; undefined with no body, or
 * a part at the body's middle.
 */
export function faceOut(part: Vec3, body: Vec3 | undefined): Vec3 | undefined {
	if (!body) return undefined;
	const away: Vec3 = [part[0] - body[0], part[1] - body[1], 0];
	if (norm(away) < 1e-6) return undefined;
	const word = levelWord(unit(away));
	return (LEVEL_WORDS.find((w) => w[0] === word) as [string, Vec3])[1];
}
/**
 * The tilt this arm makes in full (its limit at least the tilt reach) that points the jaws most squarely
 * into a face looking `out`, from approach axis `a`; undefined when none points them within 45 degrees.
 */
export function sideTurn(out: Vec3, a: Vec3, caps: Caps): string | undefined {
	const reach = caps.tilt_reach_deg || TILT_REACHES_DEG;
	let best: [number, string] | undefined;
	for (const [word, limit] of Object.entries(caps.tilt_limits_deg)) {
		if (!(limit >= reach) || !ROTATION[word]) continue;
		const d = dot(out, apply(rotationAbout(ROTATION[word], Math.PI / 2), a));
		if (!best || d < best[0]) best = [d, word];
	}
	return best && best[0] < -0.7 ? best[1] : undefined;
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
		// A point that is not on the part's face (one view's table read, a planner box) says where the thing
		// stands, not how high its face is: nothing is judged from it, and the next cycle asks the cameras
		// again (round 3 defect 3: a 0 mm table point ended the step CANNOT before a move was made).
		if (!s.located || !s.located.on_face) return { text: "SIDE TAKE: the face to approach is not located yet." };
		const a = approach(f.pose).axis;
		const face = s.located.point;
		// The face, read once a subgoal (spec-executor §10.6 side take): the level direction word pointing most from
		// the middle of what the part stands out of toward the part, and the turn of this arm that points the jaws
		// into it. Smoke 6 (goal t0 s1): read every cycle and from the tool when no body was outlined, the face
		// went FORWARD, LEFT, BACKWARD in one attempt, and "pitch 90 deg" turned the jaws backward.
		if (!s.sideFace) {
			const out = faceOut(face, s.located.body);
			s.sideFace = out ? { out, ...(sideTurn(out, a, caps) ? { turn: sideTurn(out, a, caps) } : {}) } : {};
		}
		const out = s.sideFace.out;
		const floor = caps.tilted_floor_m * 1000;
		const height = f.table_z_m !== null ? (face[2] - f.table_z_m) * 1000 : undefined;
		// The jaws close up-down: a part up to a pad inside the half gap below the tilted floor is still between them.
		const lowest = floor - jawHalfGapMm(caps) + padMm(caps);
		const turned = out !== undefined && dot(out, a) < -0.7;
		const cannot = !out
			? "no two views outlined the thing it stands out of, so which way its face faces is not known"
			: height !== undefined && height < lowest
				? `the face is ${r0(height)} mm over the table, and tilted this arm's tool point gets no lower than ${r0(floor)} mm (a part down to ${r0(lowest)} mm is still between the jaws)`
				: !s.sideFace.turn && !turned
					? "no turn this arm makes in full points the jaws into it"
					: undefined;
		if (cannot || !out) return { text: "", cannot: cannot ?? "" };
		const stop: Vec3 = add(face, scale(out, SLIDE_STANDOFF_MM / 1000));
		if (f.table_z_m !== null) stop[2] = Math.max(stop[2], f.table_z_m + floor / 1000);
		const into = levelWord(scale(out, -1));
		const turn = turned
			? `the jaws already point ${into}`
			: `rotate "${s.sideFace.turn}" 90 so the jaws point ${into}`;
		return {
			text: `SIDE TAKE: the face looks ${levelWord(out).toUpperCase()}; the jaws go in ${into.toUpperCase()}. Standoff point [${stop.map((v) => r0(v * 1000)).join(", ")}] mm (${r0(SLIDE_STANDOFF_MM)} mm off the face). Needed: ${turn}; then move ${into} with stop_on_contact for ${r0(SLIDE_STANDOFF_MM + 2 * padMm(caps))} mm (the standoff plus two pads) and close in the same reply.`,
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
