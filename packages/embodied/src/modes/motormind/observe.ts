/**
 * The executor's observe wrapper (DESIGN §1.4 step 3, spec-executor §6): every look at the original's
 * call points (the cycle start, before the 2nd and later actions of a batch, after a re-look or a
 * follow correction, after a stopped batch) goes through here. A look reuses the cached frame when
 * nothing moved (contract `look`), feeds the ClearanceTracker (dead reckoning while the wrist is
 * blind) and the top-face estimate (the median of the last 12 faces measured under the jaws), and
 * publishes `latest` for the monitor. Also the legend and the capability text the prompts show.
 */

import type { Frame, FrameCache, MmCall, MmReply, ObserveWhy, TrackedClearance } from "./contract.ts";
import { approach, jawsDown, look } from "./contract.ts";
import { angleBetween, apply, sub, transpose } from "./frames.ts";
import { apartXy, overTheOutline, project, usable } from "./geometry.ts";
import type { Caps, Pose, Vec3 } from "./types.ts";

/** The tracker forgets its surface after this much sideways motion (mm). */
export const TRACKER_LATERAL_RESET_MM = 80;
/** A new reading this much lower than the latched surface is the table behind it, not the surface (mm). */
export const TRACKER_HIGHER_SURFACE_MM = 15;
/** Faces kept for the top-face median. */
export const FACES_KEPT = 12;
/** A plausible object height (mm). */
export const MIN_OBJECT_MM = 10;
export const MAX_OBJECT_MM = 300;
/** The aim marker's depth when neither the table nor the wrist gives one (m). */
export const AIM_FALLBACK_M = 0.2;
export const AIM_MIN_TABLE_M = 0.02;

const r0 = (v: number) => v.toFixed(0);

// ---------------------------------------------------------------------------
// clearance tracking

export type ClearanceRead = {
	/** The clearance the executor uses: the measured one, else the tracked one (mm). */
	clearance_mm?: number;
	/** The frame itself saw a surface (the tracked value never counts). */
	seen: boolean;
	tracked?: TrackedClearance;
	/** The frame's reading was discarded: the latched higher surface left the range and the camera read behind it. */
	discarded?: boolean;
};

/**
 * Dead reckoning for the wrist clearance (spec-executor §6.1): when the surface is out of sight, the
 * last reading minus the descent since. A turn resets it; more than 80 mm sideways unlocks it; a
 * reading more than 15 mm lower than the latched estimate is the table behind a higher surface that
 * left the range, and is discarded.
 */
export class ClearanceTracker {
	private lock?: { mm: number; z: number; xy: [number, number]; pose: Pose };

	reset() {
		this.lock = undefined;
	}

	/** The tracked clearance now, from the pose alone (for the aim marker), or undefined. */
	tracked(pose: Pose): TrackedClearance | undefined {
		if (!this.lock) return undefined;
		const descended = (this.lock.z - pose.position_m[2]) * 1000;
		return {
			mm: this.lock.mm - descended,
			descended_mm: descended,
			lateral_mm: Math.hypot(pose.position_m[0] - this.lock.xy[0], pose.position_m[1] - this.lock.xy[1]) * 1000,
		};
	}

	update(f: Frame): ClearanceRead {
		const p = f.pose;
		if (this.lock && angleBetween(this.lock.pose.rotation, p.rotation) > (1 * Math.PI) / 180) this.lock = undefined;
		if (
			this.lock &&
			Math.hypot(p.position_m[0] - this.lock.xy[0], p.position_m[1] - this.lock.xy[1]) * 1000 >
				TRACKER_LATERAL_RESET_MM
		)
			this.lock = undefined;
		const c = f.clearance;
		const seen = c?.seen === true && typeof c.mm === "number";
		if (seen) {
			const mm = c?.mm as number;
			const tracked = this.tracked(p);
			if (tracked && mm > tracked.mm + TRACKER_HIGHER_SURFACE_MM)
				return { clearance_mm: tracked.mm, seen: false, tracked, discarded: true };
			this.lock = { mm, z: p.position_m[2], xy: [p.position_m[0], p.position_m[1]], pose: p };
			return { clearance_mm: mm, seen: true };
		}
		const tracked = this.tracked(p);
		return tracked ? { clearance_mm: tracked.mm, seen: false, tracked } : { seen: false };
	}
}

// ---------------------------------------------------------------------------
// the top face

/**
 * The target's top-face height over the table (spec-locate §7): the median of the last 12 faces
 * measured while the jaws pointed down over the stored outline and the wrist saw a surface; before
 * any, the two fixed views' top point when plausible.
 */
export class TopFace {
	faces: number[] = [];
	fromTopPoint?: number;

	constructor(carried?: { faces?: number[]; top_point_mm?: number }) {
		if (carried?.faces) this.faces = [...carried.faces];
		if (carried?.top_point_mm !== undefined) this.fromTopPoint = carried.top_point_mm;
	}

	/** The height (mm) over the table, or undefined. */
	heightMm(): number | undefined {
		if (this.faces.length) {
			const s = [...this.faces].sort((a, b) => a - b);
			const m = s.length >> 1;
			return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
		}
		return this.fromTopPoint;
	}
	/** The top-face z in the base frame (m), given the table. */
	topZ(tableZ: number | null): number | undefined {
		const h = this.heightMm();
		return tableZ === null || h === undefined ? undefined : tableZ + h / 1000;
	}
	/** An early height from where the two views' top edges met, before any face was measured. */
	offerTopPoint(topPoint: Vec3, tableZ: number | null) {
		if (this.faces.length || tableZ === null) return;
		const h = (topPoint[2] - tableZ) * 1000;
		if (h >= MIN_OBJECT_MM && h <= MAX_OBJECT_MM) this.fromTopPoint = h;
	}
	/**
	 * One frame: with the jaws down over `outline` (within `halfGapM`) and a surface in sight, the face
	 * is tool z - control offset - table z - clearance, kept when plausible.
	 */
	update(f: Frame, read: ClearanceRead, outline: readonly Vec3[] | undefined, halfGapM: number) {
		if (!read.seen || read.clearance_mm === undefined || f.table_z_m === null) return;
		if (!jawsDown(f.pose) || !outline?.length || !overTheOutline(f.pose.position_m, outline, halfGapM)) return;
		const h = (f.pose.position_m[2] - f.caps.control_point_offset_m - f.table_z_m) * 1000 - read.clearance_mm;
		if (h < MIN_OBJECT_MM || h > MAX_OBJECT_MM) return;
		this.faces.push(h);
		if (this.faces.length > FACES_KEPT) this.faces.splice(0, this.faces.length - FACES_KEPT);
	}
	/** Before a face is frozen: whatever is under the tool, tool height minus the clearance (mm), when plausible. */
	static fallbackHeight(f: Frame, read: ClearanceRead): number | undefined {
		if (read.clearance_mm === undefined || f.table_z_m === null) return undefined;
		const h = (f.pose.position_m[2] - f.caps.control_point_offset_m - f.table_z_m) * 1000 - read.clearance_mm;
		return h >= MIN_OBJECT_MM && h <= MAX_OBJECT_MM ? h : undefined;
	}
	carry() {
		return {
			faces: [...this.faces],
			...(this.fromTopPoint !== undefined ? { top_point_mm: this.fromTopPoint } : {}),
		};
	}
}

// ---------------------------------------------------------------------------
// the aim marker and the legend

/** The approach axis snapped to straight down when within 45 deg of vertical (either way). */
export function snappedApproach(pose: Pose): Vec3 {
	const a = approach(pose);
	if (Math.abs(a.axis[2]) >= Math.cos((45 * Math.PI) / 180)) return [0, 0, -1];
	return a.axis;
}

/**
 * The aim marker's point (the env server's rule, M1): the tool point along the snapped approach axis
 * by the height over the table (jaws down, table known, at least 20 mm), else the wrist clearance,
 * else the tracked clearance, else 0.2 m.
 */
export function aimPoint(f: Frame, tracked?: TrackedClearance): { point: Vec3; source: string } {
	const axis = snappedApproach(f.pose);
	const p = f.pose.position_m;
	let depth: number | undefined;
	let source = "fallback";
	if (jawsDown(f.pose) && f.table_z_m !== null) {
		depth = Math.max(AIM_MIN_TABLE_M, p[2] - f.table_z_m);
		source = "table";
	} else if (f.clearance?.seen && typeof f.clearance.mm === "number") {
		depth = f.clearance.mm / 1000;
		source = "clearance";
	} else if (tracked && tracked.mm > 0) {
		depth = tracked.mm / 1000;
		source = "tracked";
	}
	if (depth === undefined) depth = AIM_FALLBACK_M;
	return { point: [p[0] + axis[0] * depth, p[1] + axis[1] * depth, p[2] + axis[2] * depth], source };
}

/**
 * The legend (spec-executor §6): for each camera, where the aim marker sits in the picture as
 * percentages, and how many mm 10% of the frame is at that depth.
 */
export function legend(f: Frame, tracked?: TrackedClearance): string {
	const { point } = aimPoint(f, tracked);
	const lines: string[] = [];
	for (const name of f.camera_names) {
		const c = f.cameras[name];
		if (!usable(c)) continue;
		const uv = project(c, point);
		if (!uv) {
			lines.push(`${name}: the magenta marker (the point under the tool) is out of this picture.`);
			continue;
		}
		const r = [0, 1, 2].map((i) => [0, 1, 2].map((j) => c.cam2base[i][j]));
		const q = apply(transpose(r as never), sub(point, [c.cam2base[0][3], c.cam2base[1][3], c.cam2base[2][3]]));
		const mmPer10 = ((0.1 * c.width * q[2]) / c.intrinsic[0][0]) * 1000;
		const x = (uv[0] / c.width) * 100;
		const y = (uv[1] / c.height) * 100;
		lines.push(
			`${name}: the magenta marker (the point under the tool) is ${r0(x)}% from the left and ${r0(y)}% from the top; at its depth 10% of the frame is about ${r0(mmPer10)} mm.`,
		);
	}
	return lines.join("\n");
}

/** The capability text (spec-executor §8.3), from the server's table. */
export function capabilityText(caps: Caps): string {
	const lines = [
		`One move runs at most ${r0(caps.max_translation_m * 1000)} mm and one turn at most ${r0((caps.max_rotation_rad * 180) / Math.PI)} deg per segment; longer ones are cut into equal segments automatically.`,
		`Speed is at most ${r0(caps.max_speed_m_s * 1000)} mm/s (default ${r0(caps.default_speed_m_s * 1000)} mm/s) and ${r0((caps.max_rotation_speed_rad_s * 180) / Math.PI)} deg/s for turns.`,
		caps.supports_contact_stop
			? caps.contact_stop_trusted
				? "A move with stop_on_contact stops when it meets resistance and reports contact; that stop is reliable."
				: "A move with stop_on_contact stops when it meets resistance, but the stop is not reliable: every acceleration can trigger it."
			: "This robot cannot stop on contact.",
		caps.supports_home ? "home resets the scene." : "There is no home: move back with move and rotate.",
		`When the jaws are tilted for a side approach, the tool point can go no lower than ${r0(caps.tilted_floor_m * 1000)} mm over the table.`,
		...caps.notes,
	];
	return lines.join("\n");
}

/** The tool's horizontal direction from the base, and distance (m), for the reach note. */
export const radius = (pose: Pose) => Math.hypot(pose.position_m[0], pose.position_m[1]);

// ---------------------------------------------------------------------------
// the wrapper

export type ObserveState = {
	tracker: ClearanceTracker;
	faces: TopFace;
	latest?: Frame;
	read?: ClearanceRead;
	/** The monitor's `latest` publisher (M3); the only place it is called from. */
	publish?: (f: Frame) => void;
};

export type Looked = { frame: Frame; read: ClearanceRead; yielded: boolean };

/**
 * One look at an original call point (`why`): the cached frame or one `mm_observe`, then the
 * clearance tracker and the top face, and `latest` published. Returns undefined when the robot
 * answered with an error.
 */
export async function* observeStep(
	cache: FrameCache,
	observes: { why: ObserveWhy; yielded: boolean }[],
	state: ObserveState,
	why: ObserveWhy,
	o: { outline?: readonly Vec3[]; halfGapM: number } = { halfGapM: 0.0394 },
): AsyncGenerator<MmCall, Looked | undefined, MmReply> {
	const before = observes.length;
	const tracked = state.latest ? state.tracker.tracked(state.latest.pose) : undefined;
	const frame = yield* look(cache, why, observes, tracked && tracked.mm > 0 ? tracked.mm : undefined);
	if (!frame) return undefined;
	const yielded = observes[before]?.yielded ?? false;
	const read = state.tracker.update(frame);
	state.faces.update(frame, read, o.outline, o.halfGapM);
	state.latest = frame;
	state.read = read;
	state.publish?.(frame);
	return { frame, read, yielded };
}

/** The clearance tracker's estimate for the aim marker of the next tool call (mm), or undefined. */
export function trackedMm(state: ObserveState): number | undefined {
	const t = state.latest ? state.tracker.tracked(state.latest.pose) : undefined;
	return t && t.mm > 0 ? t.mm : undefined;
}

/** Whether two poses differ (position over 1 mm or rotation over 0.5 deg): a frame rendered at one is stale at the other. */
export const moved = (a: Pose, b: Pose) =>
	apartXy(a.position_m, b.position_m) > 1e-3 ||
	Math.abs(a.position_m[2] - b.position_m[2]) > 1e-3 ||
	angleBetween(a.rotation, b.rotation) > (0.5 * Math.PI) / 180;
