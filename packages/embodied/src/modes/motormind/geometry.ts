/**
 * Camera geometry for MotorMind's locate (spec-locate §2.1, §4): a model's box in 0..1000 normalized
 * coordinates on a flipped (row 0 at the top) render, a pinhole K with the principal point at the
 * image centre, and a 4x4 camera-to-base extrinsic in the OpenCV convention (x right, y down, z
 * forward). Everything metric is in the robot's base frame, in metres; error bars are in mm.
 *
 * Clean room: these are the geometric rules of spec-locate, written from the spec.
 */

import type { Box, Camera } from "./contract.ts";
import { add, apply, dot, norm, scale, sub, transpose, unit } from "./frames.ts";
import type { Mat3, Vec3 } from "./types.ts";

/** A box in the render's pixels: [x0, y0, x1, y1]. */
export type PxBox = [number, number, number, number];
/** The calibration a view needs (a contract `Camera` has it). */
export type Cam = Pick<Camera, "intrinsic" | "cam2base" | "width" | "height">;

export const NORMALISED_SPAN = 1000;
/** The near-edge bias correction: the footprint is pushed away from the camera by this share of the contact edge's width. */
export const FOOTPRINT_PUSH = 0.25;
/** A view looks down when its optical axis is within this many degrees of straight down. */
export const NADIR_WITHIN_DEG = 20;
/** The error bar of one oblique fixed view (mm): at least this, else a quarter of the contact edge's width. */
export const PLANE_ERROR_FLOOR_MM = 20;
/** Two rays that met, or a nadir view with a known top: the error bar is at least this (mm). */
export const CENTRED_ERROR_MM = 10;
/** A wrist alone and far from the top: the error bar is at least this (mm). */
export const UNCHECKED_FROM_HEIGHT_MM = 76;
/** A nadir view with no top measured: the error bar of the tallest object there can be (mm). */
export const MAX_UNMEASURED_HEIGHT_MM = 160;
/** Another box hides a box's base when its bottom sits this share of the picture lower (normalised 0..1). */
export const OCCLUSION_MARGIN = 0.01;

/** The view has the calibration locate needs. */
export const usable = (c: Partial<Cam> | undefined): c is Cam =>
	!!c &&
	Array.isArray(c.intrinsic) &&
	Array.isArray(c.cam2base) &&
	c.cam2base.length === 4 &&
	typeof c.width === "number" &&
	typeof c.height === "number";

/** The camera's position in the base frame. */
export const eye = (c: Cam): Vec3 => [c.cam2base[0][3], c.cam2base[1][3], c.cam2base[2][3]];
/** The camera's rotation (columns: x right, y down, z forward) in the base frame. */
export const camRotation = (c: Cam): Mat3 => [0, 1, 2].map((i) => [0, 1, 2].map((j) => c.cam2base[i][j])) as Mat3;

/** A normalised 0..1000 box on the render's pixels (the downscale before sending changes nothing). */
export function toPixels(b: Box, c: Cam): PxBox {
	const sx = c.width / NORMALISED_SPAN;
	const sy = c.height / NORMALISED_SPAN;
	return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
}
/** A render-pixel box back in 0..1000. */
export function toNormalised(b: PxBox, c: Cam): Box {
	const sx = NORMALISED_SPAN / c.width;
	const sy = NORMALISED_SPAN / c.height;
	return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
}

/** The pixel of a base-frame point, or undefined when it is not in front of the camera. */
export function project(c: Cam, p: Vec3): [number, number] | undefined {
	const q = apply(transpose(camRotation(c)), sub(p, eye(c)));
	if (q[2] <= 1e-6) return undefined;
	const k = c.intrinsic;
	return [(k[0][0] * q[0]) / q[2] + k[0][2], (k[1][1] * q[1]) / q[2] + k[1][2]];
}

/** The ray through pixel (u, v): from the camera along R · K⁻¹ · [u, v, 1] (unit length). */
export function ray(c: Cam, u: number, v: number): { origin: Vec3; dir: Vec3 } {
	const k = c.intrinsic;
	const d: Vec3 = [(u - k[0][2]) / k[0][0], (v - k[1][2]) / k[1][1], 1];
	return { origin: eye(c), dir: unit(apply(camRotation(c), d)) };
}

/** Where the pixel ray meets the horizontal plane at `z`, or undefined when it runs level or hits behind. */
export function onPlane(c: Cam, u: number, v: number, z: number): Vec3 | undefined {
	const r = ray(c, u, v);
	if (Math.abs(r.dir[2]) < 1e-3) return undefined;
	const t = (z - r.origin[2]) / r.dir[2];
	if (t <= 0) return undefined;
	return add(r.origin, scale(r.dir, t));
}

/**
 * The closest points of two pixel rays: their midpoint and the gap between them (the skew, m);
 * undefined when the rays are parallel or a closest point lies behind its camera.
 */
export function triangulate(
	a: Cam,
	pa: [number, number],
	b: Cam,
	pb: [number, number],
): { point: Vec3; skew_m: number } | undefined {
	const ra = ray(a, pa[0], pa[1]);
	const rb = ray(b, pb[0], pb[1]);
	const w = sub(ra.origin, rb.origin);
	const aa = dot(ra.dir, ra.dir);
	const ab = dot(ra.dir, rb.dir);
	const bb = dot(rb.dir, rb.dir);
	const det = aa * bb - ab * ab;
	if (det < 1e-12) return undefined;
	const ad = dot(ra.dir, w);
	const bd = dot(rb.dir, w);
	const ta = (ab * bd - bb * ad) / det;
	const tb = (aa * bd - ab * ad) / det;
	if (ta <= 0 || tb <= 0) return undefined;
	const qa = add(ra.origin, scale(ra.dir, ta));
	const qb = add(rb.origin, scale(rb.dir, tb));
	return { point: scale(add(qa, qb), 0.5), skew_m: norm(sub(qa, qb)) };
}

/** The view looks (nearly) straight down. */
export function lookingDown(c: Cam): boolean {
	const r = camRotation(c);
	const axis: Vec3 = [r[0][2], r[1][2], r[2][2]];
	return -axis[2] >= Math.cos((NADIR_WITHIN_DEG * Math.PI) / 180);
}

/**
 * Which row of an oblique box meets the table: the bottom, unless lowering a point moves it up the
 * picture at both rows (the camera looks from below the object's base), then the top.
 */
export function standsAtTop(c: Cam, b: PxBox): boolean {
	if (lookingDown(c)) return false;
	const d = apply(transpose(camRotation(c)), [0, 0, -1]);
	const k = c.intrinsic;
	const kd_y = k[1][1] * d[1] + k[1][2] * d[2];
	const rate = (v: number) => kd_y - v * d[2];
	return rate(b[1]) < 0 && rate(b[3]) < 0;
}
/** The row (v) and ends (u) of the contact edge. */
export function contactEdge(c: Cam, b: PxBox): { v: number; u0: number; u1: number } {
	return { v: standsAtTop(c, b) ? b[1] : b[3], u0: b[0], u1: b[2] };
}

/** The pixel treated as the object: the box middle when looking down, else the contact edge's centre. */
export function boxAim(c: Cam, b: PxBox): [number, number] {
	if (lookingDown(c)) return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
	const e = contactEdge(c, b);
	return [(e.u0 + e.u1) / 2, e.v];
}

/** The two ends of the contact edge on the plane at `z`. */
function edgeOnPlane(c: Cam, b: PxBox, z: number): [Vec3, Vec3] | undefined {
	const e = contactEdge(c, b);
	const p0 = onPlane(c, e.u0, e.v, z);
	const p1 = onPlane(c, e.u1, e.v, z);
	return p0 && p1 ? [p0, p1] : undefined;
}

/** The contact edge's width on the plane at `z` (m), or undefined. */
export function edgeWidth(c: Cam, b: PxBox, z: number): number | undefined {
	const e = edgeOnPlane(c, b, z);
	return e ? Math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]) : undefined;
}

/** The contact edge's centre on the plane, unpushed. */
export function contactPoint(c: Cam, b: PxBox, z: number): Vec3 | undefined {
	const aim = boxAim(c, b);
	if (lookingDown(c)) return onPlane(c, aim[0], aim[1], z);
	const e = contactEdge(c, b);
	return onPlane(c, (e.u0 + e.u1) / 2, e.v, z);
}

/**
 * The object's footprint point on the plane at `z`: the contact edge's centre pushed away from the
 * camera (in xy) by a quarter of the edge's width on the table, undoing part of the near-edge bias.
 */
export function footprintOnPlane(c: Cam, b: PxBox, z: number): Vec3 | undefined {
	const p = contactPoint(c, b, z);
	const w = edgeWidth(c, b, z);
	if (!p || w === undefined) return undefined;
	const o = eye(c);
	const away: Vec3 = [p[0] - o[0], p[1] - o[1], 0];
	const n = Math.hypot(away[0], away[1]);
	if (n < 1e-9) return p;
	return add(p, scale(away, (FOOTPRINT_PUSH * w) / n));
}

/** xy distance (m). */
export const apartXy = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/**
 * The single-view point of a box: a nadir view reads the mean of the outline's corners; an oblique
 * view with a known top reads the box middle on the plane halfway up, accepted when it is no nearer
 * the camera (xy) than the unpushed contact edge; otherwise the pushed footprint.
 */
export function boxOnPlane(c: Cam, b: PxBox, tableZ: number, topZ?: number): Vec3 | undefined {
	if (lookingDown(c)) {
		const corners = boxOnPlaneCorners(c, b, tableZ, topZ);
		if (!corners?.length) return undefined;
		const m = corners.reduce((s, p) => add(s, p), [0, 0, 0] as Vec3);
		return scale(m, 1 / corners.length);
	}
	if (topZ !== undefined) {
		const mid = onPlane(c, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (tableZ + topZ) / 2);
		const edge = contactPoint(c, b, tableZ);
		if (mid && edge) {
			const o = eye(c);
			if (apartXy(mid, o) >= apartXy(edge, o) - 1e-9) return mid;
		}
	}
	return footprintOnPlane(c, b, tableZ);
}

/** The picture's two axes laid onto the table (unit xy directions of the camera's x and y). */
function pictureAxes(c: Cam): [Vec3, Vec3] | undefined {
	const r = camRotation(c);
	const flat = (col: number): Vec3 | undefined => {
		const v: Vec3 = [r[0][col], r[1][col], 0];
		const n = Math.hypot(v[0], v[1]);
		return n < 1e-6 ? undefined : scale(v, 1 / n);
	};
	const ax = flat(0);
	let ay = flat(1);
	if (!ax) return undefined;
	// A camera looking straight down has its y axis flat; the rare perfectly vertical one takes the perpendicular.
	if (!ay || Math.abs(dot(ax, ay)) > 0.99) ay = [-ax[1], ax[0], 0];
	return [ax, ay];
}

/**
 * The outline of a box on the plane: in an oblique view the two ends of the contact edge at `tableZ`;
 * in a nadir view the box's four corners read on the table plane and, when `topZ` is known, on the top
 * plane, both expressed in the picture's axes laid onto the table and intersected (an axis where they
 * do not overlap keeps the top plane's extent), returned at the top plane's height (the table's when
 * no top is known). This removes the side walls a nadir box includes.
 */
export function boxOnPlaneCorners(c: Cam, b: PxBox, tableZ: number, topZ?: number): Vec3[] | undefined {
	if (!lookingDown(c)) {
		const e = edgeOnPlane(c, b, tableZ);
		return e ? [e[0], e[1]] : undefined;
	}
	const px: [number, number][] = [
		[b[0], b[1]],
		[b[2], b[1]],
		[b[2], b[3]],
		[b[0], b[3]],
	];
	const read = (z: number) => {
		const pts = px.map(([u, v]) => onPlane(c, u, v, z));
		return pts.every((p) => p !== undefined) ? (pts as Vec3[]) : undefined;
	};
	const onTable = read(tableZ);
	if (!onTable) return undefined;
	const axes = pictureAxes(c);
	if (!axes) return onTable;
	const [ax, ay] = axes;
	const extent = (pts: Vec3[]) => {
		const a = pts.map((p) => dot(p, ax));
		const bb = pts.map((p) => dot(p, ay));
		return { a: [Math.min(...a), Math.max(...a)], b: [Math.min(...bb), Math.max(...bb)] };
	};
	const t = extent(onTable);
	let z = tableZ;
	let a = t.a;
	let bb = t.b;
	if (topZ !== undefined) {
		const onTop = read(topZ);
		if (onTop) {
			const u = extent(onTop);
			const meet = (p: number[], q: number[]) => {
				const lo = Math.max(p[0], q[0]);
				const hi = Math.min(p[1], q[1]);
				return hi > lo ? [lo, hi] : q;
			};
			a = meet(t.a, u.a);
			bb = meet(t.b, u.b);
			z = topZ;
		}
	}
	const at = (ca: number, cb: number): Vec3 => [ca * ax[0] + cb * ay[0], ca * ax[1] + cb * ay[1], z];
	return [at(a[0], bb[0]), at(a[1], bb[0]), at(a[1], bb[1]), at(a[0], bb[1])];
}

/** How far a single-view point may be from the middle of the thing (mm), and where that bar comes from. */
export function planeErrorMm(
	c: Cam,
	b: PxBox,
	tableZ: number,
	topZ: number | undefined,
	tallestMm = MAX_UNMEASURED_HEIGHT_MM,
): { error_mm: number; error_class: string } {
	if (lookingDown(c)) {
		if (topZ !== undefined) return { error_mm: CENTRED_ERROR_MM, error_class: "looking straight down at the middle" };
		const bar = Math.max(CENTRED_ERROR_MM, tallestMm * Math.tan((NADIR_WITHIN_DEG * Math.PI) / 180));
		return { error_mm: bar, error_class: "looking down with no height measured" };
	}
	const w = edgeWidth(c, b, tableZ);
	if (w === undefined) return { error_mm: PLANE_ERROR_FLOOR_MM, error_class: "one oblique view's footprint" };
	return {
		error_mm: Math.max(PLANE_ERROR_FLOOR_MM, FOOTPRINT_PUSH * w * 1000),
		error_class: `one oblique view's footprint of something about ${Math.round(w * 1000)} mm across`,
	};
}

/**
 * The first box (normalised) that hides the chosen box's base: the chosen bottom-centre x lies within
 * it, its bottom is lower in the picture by more than the margin, and its top is above the chosen bottom.
 */
export function hidingTheBase(chosen: Box, others: readonly Box[]): Box | undefined {
	const x = (chosen[0] + chosen[2]) / 2;
	const bottom = chosen[3];
	for (const o of others) {
		if (o === chosen) continue;
		if (x >= o[0] && x <= o[2] && o[3] > bottom + OCCLUSION_MARGIN * NORMALISED_SPAN && o[1] < bottom) return o;
	}
	return undefined;
}

/**
 * Where the pixel ray meets the vertical plane through `through` whose normal is the horizontal
 * direction from the eye to it; undefined when the camera is right above the point or the ray runs
 * parallel to the plane.
 */
export function onUprightPlane(c: Cam, u: number, v: number, through: Vec3): Vec3 | undefined {
	const o = eye(c);
	const n: Vec3 = [through[0] - o[0], through[1] - o[1], 0];
	const len = Math.hypot(n[0], n[1]);
	if (len < 1e-6) return undefined;
	const nn = scale(n, 1 / len);
	const r = ray(c, u, v);
	const denom = dot(r.dir, nn);
	if (Math.abs(denom) < 1e-9) return undefined;
	const t = dot(sub(through, r.origin), nn) / denom;
	if (t <= 0) return undefined;
	return add(r.origin, scale(r.dir, t));
}

// ---------------------------------------------------------------------------
// outlines

/** The longest distance between two of the outline's corners (m). */
export function longestAcross(outline: readonly Vec3[]): number {
	let best = 0;
	for (let i = 0; i < outline.length; i++)
		for (let j = i + 1; j < outline.length; j++) best = Math.max(best, apartXy(outline[i], outline[j]));
	return best;
}
/** The outline's centroid, needing at least 3 corners. */
export function outlineCentroid(outline: readonly Vec3[]): Vec3 | undefined {
	if (outline.length < 3) return undefined;
	return scale(
		outline.reduce((s, p) => add(s, p), [0, 0, 0] as Vec3),
		1 / outline.length,
	);
}
/** The outline is about `point`: its centroid is no further from it than the outline is wide. */
export function aboutItsPoint(outline: readonly Vec3[], point: Vec3): boolean {
	const c = outlineCentroid(outline);
	if (!c) return true;
	return apartXy(c, point) <= longestAcross(outline);
}
/**
 * Whether a new outline measures more than the stored one: four corners always replace; a contact
 * edge replaces one with fewer than four corners, unless it is from the wrist and the jaws hold something.
 */
export function measuresMore(
	candidate: readonly Vec3[],
	stored: readonly Vec3[] | undefined,
	fromWrist: boolean,
	holding: boolean,
): boolean {
	if (!stored?.length) return candidate.length > 0;
	if (candidate.length >= 4) return true;
	if (stored.length >= 4) return false;
	return !(fromWrist && holding);
}
/** The xy distance from a point to the outline (0 inside a 4-corner one; to the segment for 2 corners). */
export function distanceToOutline(point: Vec3, outline: readonly Vec3[]): number {
	if (!outline.length) return Infinity;
	if (outline.length === 1) return apartXy(point, outline[0]);
	if (outline.length >= 3 && insidePolygon(point, outline)) return 0;
	let best = Infinity;
	for (let i = 0; i < outline.length; i++) {
		const a = outline[i];
		const b = outline[(i + 1) % outline.length];
		if (outline.length === 2 && i === 1) break;
		best = Math.min(best, segmentDistance(point, a, b));
	}
	return best;
}
/** The point is over the outline within `margin` (m). */
export const overTheOutline = (point: Vec3, outline: readonly Vec3[], margin: number) =>
	distanceToOutline(point, outline) <= margin;

function segmentDistance(p: Vec3, a: Vec3, b: Vec3): number {
	const ab: [number, number] = [b[0] - a[0], b[1] - a[1]];
	const ap: [number, number] = [p[0] - a[0], p[1] - a[1]];
	const l2 = ab[0] * ab[0] + ab[1] * ab[1];
	const t = l2 < 1e-12 ? 0 : Math.max(0, Math.min(1, (ap[0] * ab[0] + ap[1] * ab[1]) / l2));
	return Math.hypot(ap[0] - t * ab[0], ap[1] - t * ab[1]);
}
function insidePolygon(p: Vec3, poly: readonly Vec3[]): boolean {
	let inside = false;
	for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
		const a = poly[i];
		const b = poly[j];
		if (a[1] > p[1] !== b[1] > p[1] && p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0])
			inside = !inside;
	}
	return inside;
}

/** The area (normalised units²) of a box, and whether one box is "one thing" with another (spec-locate §5.7). */
export const boxArea = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export function overlapArea(a: Box, b: Box): number {
	const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
	const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
	return w > 0 && h > 0 ? w * h : 0;
}
