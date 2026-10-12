/**
 * MotorMind's one frame and its direction words (DESIGN §2). Everything here is in the robot's base
 * frame (`robot0_base`), robot-centric: looking from the base along +X, +Y is on the left. The env
 * server converts world poses to this frame (`env.mm_*`); this file only names base vectors. pi's
 * own LIBERO units are observer-centric (`MV_LEFT` is world -y), so MotorMind's `left` is the
 * vector pi calls `MV_RIGHT`: the two vocabularies meet only as vectors, never by name.
 *
 * The same table serves the parser (a word to an axis), the text the model reads (an offset back
 * to words, `sayOffset` / `way`) and the label rule of the mission's net motion (`labelDirection`).
 */

import type { Mat3, Pose, Vec3 } from "./types.ts";

/** Translation words (`move`): unit base axes. */
export const TRANSLATION: Readonly<Record<string, Vec3>> = {
	forward: [1, 0, 0],
	backward: [-1, 0, 0],
	left: [0, 1, 0],
	right: [0, -1, 0],
	up: [0, 0, 1],
	down: [0, 0, -1],
};

/** Rotation words (`rotate`): signed base axes, right-handed, about the tool point. */
export const ROTATION: Readonly<Record<string, Vec3>> = {
	yaw_left: [0, 0, 1],
	yaw_right: [0, 0, -1],
	pitch_up: [0, -1, 0],
	pitch_down: [0, 1, 0],
	roll_ccw: [1, 0, 0],
	roll_cw: [-1, 0, 0],
};

/** Seen from above with z up, clockwise is about -Z: aliases resolve before the table is read. */
export const ROTATION_ALIASES: Readonly<Record<string, string>> = {
	clockwise: "yaw_right",
	cw: "yaw_right",
	anticlockwise: "yaw_left",
	counterclockwise: "yaw_left",
	counter_clockwise: "yaw_left",
	ccw: "yaw_left",
};

/**
 * pi's LIBERO units naming the same base vector, for comparison only (C2): MotorMind's `left` (+Y)
 * is pi's `MV_RIGHT`. Execution never goes through these names.
 */
export const PI_UNIT: Readonly<Record<string, string>> = {
	forward: "MV_FWD",
	backward: "MV_BACK",
	left: "MV_RIGHT",
	right: "MV_LEFT",
	up: "MV_UP",
	down: "MV_DOWN",
};

const words = (table: Readonly<Record<string, unknown>>) => Object.keys(table).join(", ");

/** The base axis of a translation word; an unknown word names every word there is. */
export function translationAxis(word: string): Vec3 {
	const axis = TRANSLATION[word.trim().toLowerCase()];
	if (!axis) throw new Error(`unknown move direction "${word}"; the directions are ${words(TRANSLATION)}`);
	return [...axis];
}

/** The signed base axis of a rotation word or alias; an unknown word names every word and alias. */
export function rotationAxis(word: string): Vec3 {
	const w = word.trim().toLowerCase();
	const axis = ROTATION[ROTATION_ALIASES[w] ?? w];
	if (!axis)
		throw new Error(
			`unknown rotate direction "${word}"; the directions are ${words(ROTATION)} (aliases: ${words(ROTATION_ALIASES)})`,
		);
	return [...axis];
}

/**
 * Words the net-motion label rule reads (DESIGN §2.2, the faithful `--mm-net-motion label`): the
 * first of them in a step's label decides the step's direction. `back` counts as `backward` (so the
 * executor's inserted "back down ..." reads as a move toward -X); only yaw words count for turns.
 */
const LABEL_TRANSLATION: Readonly<Record<string, Vec3>> = { ...TRANSLATION, back: [-1, 0, 0] };
const LABEL_YAW: Readonly<Record<string, Vec3>> = {
	yaw_left: [0, 0, 1],
	yaw_right: [0, 0, -1],
	...Object.fromEntries(
		Object.entries(ROTATION_ALIASES).map(([alias, word]) => [alias, word === "yaw_left" ? [0, 0, 1] : [0, 0, -1]]),
	),
};

/**
 * The direction a step's label names under the label rule: its first direction word, as a base
 * axis. Undefined when it names none (an `axis [...]` move, the inserted "lift over the table", a
 * pitch or roll): such a step is left out of the net motion.
 */
export function labelDirection(label: string): { word: string; axis: Vec3; kind: "translate" | "rotate" } | undefined {
	for (const token of label.toLowerCase().match(/[a-z_]+/g) ?? []) {
		if (token in LABEL_TRANSLATION) return { word: token, axis: [...LABEL_TRANSLATION[token]], kind: "translate" };
		if (token in LABEL_YAW) return { word: token, axis: [...LABEL_YAW[token]], kind: "rotate" };
	}
	return undefined;
}

/**
 * An offset (m, base frame) in the model's words: each axis component of at least 1 mm, x then y
 * then z, as "N mm WORD", joined with "and" (e.g. "35 mm FORWARD and 12 mm LEFT").
 */
export function sayOffset(offset: Vec3): string {
	const parts: string[] = [];
	for (let i = 0; i < 3; i++) {
		const mm = offset[i] * 1000;
		if (Math.abs(mm) < 1) continue;
		const want: Vec3 = [0, 0, 0];
		want[i] = Math.sign(mm);
		const word = Object.entries(TRANSLATION).find(([, a]) => a.every((v, k) => v === want[k]))?.[0];
		parts.push(`${Math.round(Math.abs(mm))} mm ${word ? word.toUpperCase() : `along axis ${i}`}`);
	}
	return parts.length ? parts.join(" and ") : "less than 1 mm";
}

/** The horizontal direction word closest to `v` (largest dot product). */
export function way(v: Vec3): string {
	let best = "forward";
	let score = -Infinity;
	for (const [word, a] of Object.entries(TRANSLATION)) {
		if (a[2] !== 0) continue;
		const s = dot(a, v);
		if (s > score) [best, score] = [word, s];
	}
	return best;
}

// ---------------------------------------------------------------------------
// vectors and rotations

export const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a: readonly number[]) => Math.hypot(a[0], a[1], a[2]);
export const add = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: readonly number[], s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];

/** `a` normalized; a zero (or non-finite) vector throws. */
export function unit(a: readonly number[]): Vec3 {
	const n = norm(a);
	if (!(n > 1e-12) || !Number.isFinite(n)) throw new Error(`axis [${a.join(", ")}] has no direction`);
	return scale(a, 1 / n);
}

export const transpose = (m: Mat3): Mat3 => [
	[m[0][0], m[1][0], m[2][0]],
	[m[0][1], m[1][1], m[2][1]],
	[m[0][2], m[1][2], m[2][2]],
];
export const matmul = (a: Mat3, b: Mat3): Mat3 =>
	[0, 1, 2].map((i) => [0, 1, 2].map((j) => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])) as Mat3;
export const apply = (m: Mat3, v: readonly number[]): Vec3 => [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
export const IDENTITY: Mat3 = [
	[1, 0, 0],
	[0, 1, 0],
	[0, 0, 1],
];

/** The rotation by `angle` (rad) about unit `axis`, right-handed (Rodrigues). */
export function rotationAbout(axis: readonly number[], angle: number): Mat3 {
	const [x, y, z] = unit(axis);
	const [c, s] = [Math.cos(angle), Math.sin(angle)];
	const C = 1 - c;
	return [
		[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
		[y * x * C + z * s, c + y * y * C, y * z * C - x * s],
		[z * x * C - y * s, z * y * C + x * s, c + z * z * C],
	];
}

/** The angle (rad, in [0, pi]) of the rotation taking `a` to `b`: angle(aᵀ b). */
export function angleBetween(a: Mat3, b: Mat3): number {
	const r = matmul(transpose(a), b);
	return Math.acos(Math.min(1, Math.max(-1, (r[0][0] + r[1][1] + r[2][2] - 1) / 2)));
}

/** The axis (unit) and angle of rotation `r`; the axis is [0, 0, 0] for the identity. */
export function axisAngle(r: Mat3): { axis: Vec3; angle: number } {
	const angle = Math.acos(Math.min(1, Math.max(-1, (r[0][0] + r[1][1] + r[2][2] - 1) / 2)));
	const w: Vec3 = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
	if (angle < 1e-9) return { axis: [0, 0, 0], angle: 0 };
	if (Math.PI - angle < 1e-6) {
		// Half turn: the axis is the column of R + I with the largest norm.
		const cols = [0, 1, 2].map((j) => [0, 1, 2].map((i) => r[i][j] + (i === j ? 1 : 0)));
		const best = cols.reduce((p, c) => (norm(c) > norm(p) ? c : p));
		return { axis: unit(best), angle };
	}
	return { axis: unit(w), angle };
}

/** Rotation matrix of an xyzw quaternion (normalized first), as LIBERO reports `robot0_eef_quat`. */
export function quatToMatrix(q: readonly number[]): Mat3 {
	const n = Math.hypot(q[0], q[1], q[2], q[3]);
	if (!(n > 0)) throw new Error("zero-norm quaternion");
	const [x, y, z, w] = q.map((v) => v / n);
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	];
}

/** A world pose in the base frame: `R_bᵀ(p - p_b)`, `R_bᵀ R` (what the env server reports as `pose`). */
export function toBase(base: Pose, world: Pose): Pose {
	const rt = transpose(base.rotation);
	return { position_m: apply(rt, sub(world.position_m, base.position_m)), rotation: matmul(rt, world.rotation) };
}

/** A base-frame vector in the world frame: `R_b v` (a displacement; no translation). */
export const baseToWorld = (base: Pose, v: readonly number[]): Vec3 => apply(base.rotation, v);
