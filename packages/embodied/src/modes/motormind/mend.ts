/**
 * The menders (spec-executor §11): code that inserts or splits actions in a batch before it runs. A
 * lift before a sideways move over the table (and the way back down when the jaws act afterwards),
 * a carry split into legs the gates can check between, the first move after a wall grasp split into
 * a short test and "(the rest)", and the test for whether code may let go by itself.
 */

import {
	actionLabel,
	carryLegLabel,
	DROP_LABEL,
	LIFT_LABEL,
	pieces,
	testLeadLabel,
	testRestLabel,
	toDelta,
} from "./command.ts";
import type { Frame } from "./contract.ts";
import { jawsDown } from "./contract.ts";
import {
	CLEAR_MM,
	isHorizontal,
	leans,
	namesEdgePart,
	namesSlider,
	padMm,
	payloadOffset,
	RELEASE_CLEARANCE_MM,
	RELEASE_MARGIN_MM,
	releaseAllowanceMm,
	type SubgoalState,
	TRAVEL_MM,
	TRAVERSE_MM,
	widerThanJaws,
} from "./gates.ts";
import type { ClearanceRead } from "./observe.ts";
import type { Caps, Delta, Vec3 } from "./types.ts";

const isTranslate = (d: Delta) => d.kind === "translate" && !!d.axis;
const horizontalM = (d: Delta) =>
	isTranslate(d) && !d.push ? d.magnitude * Math.hypot((d.axis as Vec3)[0], (d.axis as Vec3)[1]) : 0;

/** A move the code inserts, converted on its own (so the 400 mm composite limit is not hit). */
function insertedMove(direction: "up" | "down", mm: number, caps: Caps, label: string): Delta {
	const d = toDelta({ index: -1, type: "move", direction, distance_mm: mm, frame: "base", note: "" }, caps);
	d.label = label;
	d.note = `inserted by the executor: ${label}`;
	return d;
}

/**
 * Lift before sideways (spec-executor §11.1): a sideways move over the table of more than 60 mm,
 * with less than 60 mm of clearance over what it would hit, gets "up (100 - clearance - lift so far)
 * mm" inserted before it, and the same distance back down after the last sideways move when a
 * gripper action follows. Returns what was inserted, or undefined.
 */
export function liftBeforeSideways(
	deltas: Delta[],
	s: SubgoalState,
	f: Frame,
	caps: Caps,
	o: { gripperOnly: boolean; heightMm?: number },
): string | undefined {
	if (f.table_z_m === null || o.gripperOnly || isHorizontal(f.pose, caps)) return undefined;
	const holding = s.payload.tracking;
	if (!holding && namesSlider(s.target)) return undefined;
	if (leans(s.subgoal)) return undefined;
	const sideways = deltas.reduce((a, d) => a + horizontalM(d) * 1000, 0);
	const first = deltas.findIndex((d) => horizontalM(d) > 0);
	if (first < 0) return undefined;
	const threshold = s.payload.bumped ? 0 : TRAVERSE_MM;
	if (sideways <= threshold) return undefined;
	let last = first;
	for (let i = first; i < deltas.length; i++) if (horizontalM(deltas[i]) > 0) last = i;
	if (deltas.slice(0, last).some((d) => d.kind === "gripper")) return undefined;
	const before = deltas
		.slice(0, first)
		.reduce((a, d) => a + (isTranslate(d) ? (d.axis as Vec3)[2] * d.magnitude * 1000 : 0), 0);
	let top: number;
	if (holding) top = f.table_z_m + (s.receiver ? 0 : 0);
	else {
		const height =
			o.heightMm !== undefined
				? o.heightMm / 1000
				: (s.startPose ? s.startPose.position_m[2] - 0.1 : f.pose.position_m[2] - 0.1) - f.table_z_m;
		top = f.table_z_m + Math.max(height, 0);
	}
	if (s.located) top = Math.max(top, s.located.point[2]);
	const clearance = (f.pose.position_m[2] - top) * 1000;
	if (clearance + before >= CLEAR_MM) return undefined;
	const lift = TRAVEL_MM - clearance - before;
	if (lift <= 0) return undefined;
	deltas.splice(first, 0, insertedMove("up", lift, caps, LIFT_LABEL));
	let what = `${LIFT_LABEL} ${Math.round(lift)} mm`;
	if (deltas.some((d) => d.kind === "gripper")) {
		let afterLast = last + 1;
		for (let i = first + 1; i < deltas.length; i++) if (horizontalM(deltas[i]) > 0) afterLast = i + 1;
		deltas.splice(afterLast, 0, insertedMove("down", lift, caps, DROP_LABEL));
		what += `, ${DROP_LABEL} ${Math.round(lift)} mm`;
	}
	return what;
}

/** A carry split into legs (spec-executor §11.2): while tracking a payload, every non-push, non-descent move over 50 mm. */
export function splitACarry(deltas: Delta[], s: SubgoalState, caps: Caps): boolean {
	if (!s.payload.tracking) return false;
	let split = false;
	for (let i = 0; i < deltas.length; i++) {
		const d = deltas[i];
		if (!isTranslate(d) || d.push || (d.axis as Vec3)[2] < -0.5 || d.magnitude <= caps.max_translation_m + 1e-9)
			continue;
		const n = pieces(d.magnitude, caps.max_translation_m);
		const legs: Delta[] = Array.from({ length: n }, (_, k) => ({
			...d,
			magnitude: d.magnitude / n,
			label: carryLegLabel(d.label, k + 1, n),
		}));
		deltas.splice(i, 1, ...legs);
		i += n - 1;
		split = true;
	}
	return split;
}

/**
 * The first move after a wall grasp is a test (spec-executor §11.3), at proposal time only: with a
 * payload grasped on a wall (aimed at an edge, held by its edge, or a gap under the pad), the first
 * non-push move longer than two pads is split into a lead of one pad at no more than the default
 * speed and "(the rest)", which §10.8 lets run only while the jaws still hold.
 */
export function firstMoveIsATest(
	deltas: Delta[],
	s: SubgoalState,
	caps: Caps,
	o: { running: boolean; gripperOnly: boolean },
): boolean {
	if (o.running || o.gripperOnly || !s.payload.tracking) return false;
	const wall =
		s.payload.edge ||
		namesEdgePart(s.subgoal) ||
		(s.payload.width_mm !== undefined && s.payload.width_mm < padMm(caps)) ||
		widerThanJaws(s, caps);
	if (!wall) return false;
	const pad = padMm(caps) / 1000;
	const i = deltas.findIndex((d) => isTranslate(d) && !d.push && d.magnitude > 2 * pad);
	if (i < 0) return false;
	const d = deltas[i];
	const lead: Delta = {
		...d,
		magnitude: pad,
		speed: Math.min(d.speed, caps.default_speed_m_s),
		label: testLeadLabel(d.label, pad * 1000),
	};
	const rest: Delta = { ...d, magnitude: d.magnitude - pad, label: testRestLabel(d.label) };
	deltas.splice(i, 1, lead, rest);
	return true;
}

/**
 * Whether code may let go by itself (spec-executor §11.5): holding, the step is about letting go, the
 * height allows it (clearance within 30 mm or the descent blocked) and the payload is within the
 * release allowance. Returns the opening width (mm) to use, else undefined.
 */
export function letGoWidth(
	s: SubgoalState,
	f: Frame,
	caps: Caps,
	read: ClearanceRead,
	o: { aboutLettingGo: boolean; descentBlocked: boolean },
): number | undefined {
	if (f.grasp.holding !== true || !o.aboutLettingGo) return undefined;
	const low = (read.clearance_mm !== undefined && read.clearance_mm <= RELEASE_CLEARANCE_MM) || o.descentBlocked;
	if (!low) return undefined;
	const off = payloadOffset(s, f.pose.position_m);
	const allowed = releaseAllowanceMm(s, caps);
	if (off && allowed !== undefined && Math.hypot(off[0], off[1]) * 1000 > allowed) return undefined;
	const width = (s.payload.width_mm ?? f.grasp.opening_mm ?? f.gripper.opening_mm) + RELEASE_MARGIN_MM;
	return Math.min(caps.gripper_open_span_m * 1000, width);
}

/** The width to open to when placing (spec-executor P8): the gap at the close plus 8 mm, at most the span. */
export const releaseWidthMm = (s: SubgoalState, caps: Caps) =>
	s.payload.width_mm !== undefined
		? Math.min(caps.gripper_open_span_m * 1000, s.payload.width_mm + RELEASE_MARGIN_MM)
		: undefined;

/** The follow correction after a wrist look: a horizontal axis move by `shift` (m), run alone. */
export function followDelta(shift: [number, number], caps: Caps): Delta {
	const mm = Math.hypot(shift[0], shift[1]) * 1000;
	const d = toDelta(
		{ index: -1, type: "move", axis: [shift[0], shift[1], 0], distance_mm: mm, frame: "base", note: "" },
		caps,
	);
	d.label = actionLabel({ type: "move", axis: [shift[0], shift[1], 0], distance_mm: mm, frame: "base" });
	d.note = "the executor's follow correction after a wrist look";
	return d;
}

export const toolIsDown = (f: Frame) => jawsDown(f.pose);
