/**
 * MotorMind's locate (spec-locate §5-§9): one model call per eligible view asks for every object's
 * box and which is the target; everything after is geometry (./geometry.ts). Two fixed views
 * (`scene`, `side`) are triangulated from their boxes' contact-edge centres; one view, or two that
 * disagree, reads the box onto the table plane; the wrist is a third, single-view read that can
 * replace the fixed answer from close in. The answer is a base-frame point with a heuristic xy error
 * bar, how it was built, and every signal against it.
 *
 * Clean room: the rules, constants and failure handling are spec-locate's; the prompt is
 * ./prompts/locate.md, written to the intent spec-locate §2.4 describes. Unlike the original, a
 * failed model call is recorded apart from "not visible" (`failures`), with the same effect.
 */

import type { Ask, Box, Camera, CameraName, RoleCallRecord, RoleReply } from "./contract.ts";
import { recordOf } from "./contract.ts";
import { add, scale, sub } from "./frames.ts";
import {
	apartXy,
	boxAim,
	boxArea,
	boxOnPlane,
	boxOnPlaneCorners,
	type Cam,
	CENTRED_ERROR_MM,
	footprintOnPlane,
	hidingTheBase,
	lookingDown,
	onPlane,
	onUprightPlane,
	overlapArea,
	PLANE_ERROR_FLOOR_MM,
	type PxBox,
	planeErrorMm,
	project,
	toNormalised,
	toPixels,
	triangulate,
	UNCHECKED_FROM_HEIGHT_MM,
	usable,
} from "./geometry.ts";
import { fill, promptFile } from "./prompts.ts";
import type { Vec3 } from "./types.ts";

/** Two fixed rays are on one object when they pass within this (mm); also the "apart" threshold. */
export const MAX_SKEW_MM = 50;
/** The two views' top edges meet within this (mm). */
export const TOP_SKEW_MM = 20;
/** A box is tall when its height is at least this many times its width. */
export const TALL = 1.5;
/** The wrist is useful (and the pre-descent look happens) below this height over the top (mm). */
export const WRIST_USEFUL_BELOW_MM = 150;
/** A wrist box no bigger than this times the held payload's box is the payload. */
export const HELD_WIDER = 1.3;
/** A body box lies within this share of the part's longest side. */
export const TOUCHES_SHARE = 0.5;
/** A body box's area is at most this many times the part's. */
export const BODY_AT_MOST_TIMES = 25;
/** A part, or a "held up" meet, sits at most this high over the table (mm). */
export const PART_HEIGHT_LIMIT_MM = 400;
/** One thing: the smaller box is at most this share of the larger's area ... */
export const PART_AT_MOST = 0.6;
/** ... and at least this share of it lies inside the larger. */
export const INSIDE_SHARE = 0.5;
/** A part sits on the top face when its bottom is within this share of the body's height below the body's top. */
export const ON_TOP_SHARE = 0.25;
/** The size the locate model is sent (the longest side, px). */
export const IMAGE_SIDE = 512;

const SIDE_PARTS = ["handle", "rim", "edge", "face", "front", "side", "lip", "spout", "grip"];
const TOP_PARTS = ["knob", "lid", "cap", "button", "top"];
const TOP_EDGE_PARTS = ["rim", "lip"];
const ARTICLES = ["the", "a", "an", "its", "this", "that"];
/** Words that do not make two labels "the same words" (articles, generic nouns, side words). */
const IGNORED = new Set([
	...ARTICLES,
	"of",
	"on",
	"in",
	"and",
	"with",
	"to",
	"at",
	"object",
	"thing",
	"item",
	"left",
	"right",
	"front",
	"back",
	"near",
	"far",
	"side",
	"middle",
	"centre",
	"center",
	"top",
	"bottom",
	"small",
	"large",
	"big",
	"little",
]);
const DIRECTION_WORDS = new Set([
	"left",
	"right",
	"front",
	"side",
	"middle",
	"centre",
	"center",
	"foreground",
	"background",
]);
const NOT_A_HOLDER = new Set(["table", "tabletop", "floor", "it", "them"]);

const tokens = (s: string) => s.toLowerCase().match(/[a-z][a-z_'-]*/g) ?? [];

/**
 * The part the words name, if any: a side part (handle, rim, edge, face, front, side, lip, spout,
 * grip) or a top part (knob, lid, cap, button, top), with a leading article and one modifier allowed
 * ("the near side of the bowl"); a compound head ("the mug handle") counts too.
 */
export function namesAPart(words: string): string | undefined {
	const t = tokens(words);
	let i = 0;
	while (i < t.length && ARTICLES.includes(t[i])) i++;
	const head = t.slice(i, i + 3);
	// "the top surface of the cabinet" / "the top face of the box" name the top as "the top of" does
	// (smoke 6, goal t4 s0: the put step's "top surface of the drawer unit" was read on the table, 437 mm down).
	if (TOP_PARTS.includes(head[0]) && (head[1] === "surface" || head[1] === "face") && head[2] === "of") return head[0];
	const parts = [...SIDE_PARTS, ...TOP_PARTS];
	// "handle of the mug" / "near side of the bowl" / "mug handle" / "lid".
	for (let k = 0; k < Math.min(head.length, 2); k++) {
		const w = head[k];
		if (!parts.includes(w)) continue;
		const next = head[k + 1];
		if (next === undefined || next === "of") return w;
		if (k === 1 && next === undefined) return w;
	}
	if (head.length >= 2 && parts.includes(head[1]) && (head[2] === undefined || head[2] === "of")) return head[1];
	return undefined;
}
export const isTopPart = (part: string) => TOP_PARTS.includes(part);
/**
 * A receiver's height is its top (round 3 defect 5): the plate's rim, the cabinet's top face ("top of the
 * drawer unit" names a top part), never the table read under it, or the put step presses the payload into
 * the body and the lift mender measures clearance to the table. The point's x and y are kept.
 */
export function receiverTop(point: Vec3, l: Pick<Located, "top_point" | "part_point">, words: string): Vec3 {
	const topPart = isTopPart(namesAPart(words) ?? "");
	const z = Math.max(
		point[2],
		l.top_point?.[2] ?? Number.NEGATIVE_INFINITY,
		topPart ? (l.part_point?.[2] ?? Number.NEGATIVE_INFINITY) : Number.NEGATIVE_INFINITY,
	);
	return z > point[2] ? [point[0], point[1], z] : point;
}
/**
 * The word as one of the parts, singular ("handles" → "handle"), or undefined when it names no part
 * (round 3 defect 1: "rest", "part", "way" and "middle" are not parts, whatever precedes "of").
 */
export function aPartWord(word: string): string | undefined {
	const stem = word.toLowerCase().replace(/[,.']/g, "").replace(/s$/, "");
	return [...SIDE_PARTS, ...TOP_PARTS].find((p) => p.replace(/s$/, "") === stem);
}

/**
 * The words say the thing is held up by something: on, in, inside, atop, upon or within, followed
 * within three words (articles skipped) by something that is not a direction word and not the table.
 */
export function heldUp(words: string): boolean {
	const t = tokens(words);
	for (let i = 0; i < t.length; i++) {
		if (!["on", "in", "inside", "atop", "upon", "within"].includes(t[i])) continue;
		let seen = 0;
		for (let j = i + 1; j < t.length && seen < 3; j++) {
			if (ARTICLES.includes(t[j])) continue;
			seen++;
			if (!DIRECTION_WORDS.has(t[j]) && !NOT_A_HOLDER.has(t[j])) return true;
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// the model's reply

export type Inventory = { label: string; box: Box }[];

/** A locate reply parsed leniently: `objects` (or a bare list) with `label`/`name` and a 4-number box; `target` must be a label. */
export function parseLocateReply(json: unknown): { inventory: Inventory; target: string } {
	const list = Array.isArray(json)
		? json
		: json && typeof json === "object" && Array.isArray((json as Record<string, unknown>).objects)
			? ((json as Record<string, unknown>).objects as unknown[])
			: [];
	const inventory: Inventory = [];
	for (const e of list) {
		if (!e || typeof e !== "object") continue;
		const o = e as Record<string, unknown>;
		const label = String(o.label ?? o.name ?? "").trim();
		const raw = o.bbox_2d ?? o.bbox ?? o.box_2d;
		let nums: number[] | undefined;
		if (Array.isArray(raw)) nums = raw.map(Number);
		else if (raw && typeof raw === "object") {
			const r = raw as Record<string, unknown>;
			nums = [r.x1, r.y1, r.x2, r.y2].map(Number);
		}
		if (!label || !nums || nums.length < 4 || nums.slice(0, 4).some((v) => !Number.isFinite(v))) continue;
		inventory.push({ label, box: nums.slice(0, 4) as Box });
	}
	const want = Array.isArray(json) ? "" : String((json as Record<string, unknown> | null)?.target ?? "").trim();
	const target = inventory.find((o) => o.label.toLowerCase().trim() === want.toLowerCase())?.label ?? "";
	return { inventory, target };
}

// ---------------------------------------------------------------------------
// inputs and output

export type LocateInput = {
	/** The views offered (the loop drops the wrist on a side step or when the jaws are tilted). */
	views: Partial<Record<CameraName, Camera>>;
	words: string;
	table_z_m: number | null;
	tool_z_m: number;
	top_z_m?: number;
	/** undefined: detect from the words; false: no part; true: `part_word` or detection. */
	part?: boolean;
	part_word?: string;
	/** Planner boxes: these views are read from the box, with no model call. */
	boxes?: Partial<Record<CameraName, Box>>;
	last_seen?: Vec3;
	hint_boxes?: Partial<Record<CameraName, Box>>;
	wrist_only?: boolean;
	wrist_within_mm?: number;
	/** The payload's box in wrist pixels while placing: a wrist box no bigger than it is the payload. */
	held_px?: PxBox;
	tallest_mm: number;
	/** Something is held: two agreeing fixed views beat the wrist. */
	holding: boolean;
	ask: Ask;
	signal?: AbortSignal;
};

export type Located = {
	point?: Vec3;
	ok: boolean;
	method: string;
	cameras: CameraName[];
	label?: string;
	candidates: string[];
	skew_mm?: number;
	disagreement?: string;
	doubtful: boolean;
	error_mm?: number;
	error_class?: string;
	confirmed_mm?: number;
	occluded: boolean;
	footprint?: Vec3[];
	footprint_method?: string;
	/** The chosen box in the chosen view's render pixels (single-view path). */
	box_px?: PxBox;
	view_name?: CameraName;
	part?: string;
	part_z_mm?: number;
	part_point?: Vec3;
	part_outline?: Vec3[];
	top_point?: Vec3;
	part_body?: Vec3;
	look_alikes: [string, number][];
	ambiguous?: string;
	part_unresolved: boolean;
	hinted: CameraName[];
	hint_kept: number;
	jumped_mm?: number;
	calls: number;
	/** Model calls that failed, apart from "not visible" (D8). */
	failures: { view: CameraName; error: string }[];
	records: RoleCallRecord[];
	/** The views the fixed round read from the planner's boxes (no model call). */
	from_boxes: CameraName[];
	error?: string;
};

type Reading = {
	view: CameraName;
	cam: Camera;
	aim: [number, number];
	box: PxBox;
	hiding?: PxBox;
	label: string;
	inventory: Inventory;
	boxNorm: Box;
};

const mm = (m: number) => Math.round(m * 1000 * 10) / 10;
const asMm = (p: Vec3): Vec3 => [mm(p[0]), mm(p[1]), mm(p[2])];

/** `Located` with its points in mm, for records (spec-locate §2.5 `as_dict`). */
export function locatedReport(l: Located): Record<string, unknown> {
	return {
		ok: l.ok,
		...(l.point ? { point_mm: asMm(l.point) } : {}),
		method: l.method,
		cameras: l.cameras,
		...(l.label ? { label: l.label } : {}),
		candidates: l.candidates,
		...(l.skew_mm !== undefined ? { skew_mm: Math.round(l.skew_mm * 10) / 10 } : {}),
		...(l.disagreement ? { disagreement: l.disagreement } : {}),
		doubtful: l.doubtful,
		...(l.error_mm !== undefined ? { error_mm: Math.round(l.error_mm * 10) / 10, error_class: l.error_class } : {}),
		...(l.confirmed_mm !== undefined ? { confirmed_mm: Math.round(l.confirmed_mm * 10) / 10 } : {}),
		occluded: l.occluded,
		...(l.footprint
			? { footprint_mm: l.footprint.map((p) => [mm(p[0]), mm(p[1])]), footprint_method: l.footprint_method }
			: {}),
		...(l.box_px ? { box_px: l.box_px.map(Math.round), view_name: l.view_name } : {}),
		...(l.part ? { part: l.part } : {}),
		...(l.part_z_mm !== undefined ? { part_z_mm: Math.round(l.part_z_mm * 10) / 10 } : {}),
		...(l.part_point ? { part_point_mm: asMm(l.part_point) } : {}),
		...(l.part_outline ? { part_outline_mm: l.part_outline.map((p) => [mm(p[0]), mm(p[1])]) } : {}),
		look_alikes: l.look_alikes,
		...(l.ambiguous ? { ambiguous: l.ambiguous } : {}),
		part_unresolved: l.part_unresolved,
		hinted: l.hinted,
		hint_kept: l.hint_kept,
		...(l.jumped_mm !== undefined ? { jumped_mm: Math.round(l.jumped_mm * 10) / 10 } : {}),
		calls: l.calls,
		failures: l.failures,
		from_boxes: l.from_boxes,
		...(l.error ? { error: l.error } : {}),
	};
}

// ---------------------------------------------------------------------------
// asking

/** The hint for a re-ask: where the thing was found before, projected into this view. */
function hintFor(view: Camera, lastSeen: Vec3 | undefined, hintBox: Box | undefined, hinted: CameraName[]): string {
	if (!lastSeen) return "";
	const uv = project(view, lastSeen);
	const spot = uv ? toNormalised([uv[0], uv[1], uv[0], uv[1]], view) : undefined;
	const inView = spot && spot[0] >= 0 && spot[0] <= 1000 && spot[1] >= 0 && spot[1] <= 1000;
	let where: string | undefined;
	if (
		hintBox &&
		(!inView || (spot[0] >= hintBox[0] && spot[0] <= hintBox[2] && spot[1] >= hintBox[1] && spot[1] <= hintBox[3]))
	)
		where = `inside the box [${hintBox.map(Math.round).join(", ")}] (its middle is about (${Math.round((hintBox[0] + hintBox[2]) / 2)}, ${Math.round((hintBox[1] + hintBox[3]) / 2)}))`;
	else if (inView) where = `at about (${Math.round(spot[0])}, ${Math.round(spot[1])})`;
	if (!where) return "";
	hinted.push(view.name);
	return `\nIt was found before ${where} in normalised coordinates. It is asked again because the scene may have changed, not because that answer was wrong: choose the same object nearest that spot unless it plainly moved. Readable printing that names a different product overrides the spot.`;
}

async function askView(
	o: LocateInput,
	view: Camera,
	out: {
		calls: number;
		failures: Located["failures"];
		records: RoleCallRecord[];
		hinted: CameraName[];
		hintKept: number;
	},
): Promise<{ inventory: Inventory; target: string; hinted: boolean } | undefined> {
	const image = view.raw512 ?? view.marked512;
	if (!image) return undefined;
	const scaleTo = Math.min(1, IMAGE_SIDE / Math.max(view.width, view.height));
	const p = promptFile("locate");
	const hintedBefore = out.hinted.length;
	const hint = hintFor(view, o.last_seen, o.hint_boxes?.[view.name], out.hinted);
	const text = fill(p.text, {
		width: Math.round(view.width * scaleTo),
		height: Math.round(view.height * scaleTo),
		target: o.words,
		hint,
	});
	out.calls++;
	let reply: RoleReply;
	try {
		reply = await o.ask(
			"locate",
			{ system: p.system, text },
			[{ type: "image", data: image, mimeType: "image/jpeg" }],
			{
				signal: o.signal,
				maxTokens: 1500,
			},
		);
	} catch (err) {
		out.failures.push({ view: view.name, error: err instanceof Error ? err.message : String(err) });
		return undefined;
	}
	out.records.push(recordOf("locate", reply));
	if (!reply.ok) {
		out.failures.push({ view: view.name, error: `${reply.error_kind}: ${reply.error}` });
		return undefined;
	}
	const parsed = parseLocateReply(reply.json);
	const hinted = out.hinted.length > hintedBefore;
	if (hinted && parsed.target) {
		const box = parsed.inventory.find((e) => e.label === parsed.target)?.box;
		const uv = o.last_seen ? project(view, o.last_seen) : undefined;
		const spot = uv ? toNormalised([uv[0], uv[1], uv[0], uv[1]], view) : undefined;
		if (box && spot && spot[0] >= box[0] && spot[0] <= box[2] && spot[1] >= box[1] && spot[1] <= box[3])
			out.hintKept++;
	}
	return { ...parsed, hinted };
}

function readingOf(view: Camera, boxNorm: Box, label: string, inventory: Inventory, checkHidden: boolean): Reading {
	const box = toPixels(boxNorm, view);
	const hiding = checkHidden
		? hidingTheBase(
				boxNorm,
				inventory.map((e) => e.box),
			)
		: undefined;
	return {
		view: view.name,
		cam: view,
		aim: boxAim(view, box),
		box,
		...(hiding ? { hiding: toPixels(hiding, view) } : {}),
		label,
		inventory,
		boxNorm,
	};
}

// ---------------------------------------------------------------------------
// the procedure

/** Locate `words` from the views; never moves the robot (spec-locate §5). */
export async function locate(o: LocateInput): Promise<Located> {
	const out: Located = {
		ok: false,
		method: "",
		cameras: [],
		candidates: [],
		doubtful: false,
		occluded: false,
		look_alikes: [],
		part_unresolved: false,
		hinted: [],
		hint_kept: 0,
		calls: 0,
		failures: [],
		records: [],
		from_boxes: [],
	};
	const asked = { calls: 0, failures: out.failures, records: out.records, hinted: out.hinted, hintKept: 0 };
	const kept = (Object.values(o.views).filter(Boolean) as Camera[]).filter(
		(v) => usable(v) && (v.raw512 || v.marked512),
	);
	const fixed = kept.filter((v) => v.name !== "wrist").sort((a, b) => a.name.localeCompare(b.name));
	const wrist = kept.find((v) => v.name === "wrist");
	if (!fixed.length) {
		out.error = "no fixed view has an image and its calibration: a pixel cannot be turned into a position";
		return out;
	}
	// A given part word that is not in the part vocabulary is no part: the words are read instead.
	if (o.part !== false) out.part = (o.part_word && aPartWord(o.part_word)) || namesAPart(o.words);
	const tableZ = o.table_z_m;
	const closeIn = tableZ !== null && (o.tool_z_m - (o.top_z_m ?? tableZ)) * 1000 < WRIST_USEFUL_BELOW_MM;

	// Ask the fixed views (sorted) and, when close in, the wrist, all at once; a given box replaces the ask.
	const toAsk: Camera[] = o.wrist_only ? [] : fixed.filter((v) => !o.boxes?.[v.name]);
	if (wrist && closeIn && !o.boxes?.wrist) toAsk.push(wrist);
	const replies = new Map<CameraName, Awaited<ReturnType<typeof askView>>>();
	await Promise.all(
		toAsk.map(async (v) => {
			replies.set(v.name, await askView(o, v, asked));
		}),
	);
	out.calls = asked.calls;
	out.hint_kept = asked.hintKept;

	const readings: Reading[] = [];
	const labels = new Set<string>();
	if (!o.wrist_only)
		for (const v of fixed) {
			const given = o.boxes?.[v.name];
			if (given) {
				out.from_boxes.push(v.name);
				readings.push(readingOf(v, given, o.words, [], false));
				continue;
			}
			const r = replies.get(v.name);
			if (!r?.target) continue;
			const box = r.inventory.find((e) => e.label === r.target)?.box;
			if (!box) continue;
			for (const e of r.inventory) labels.add(e.label);
			readings.push(readingOf(v, box, r.target, r.inventory, true));
		}

	let wristReading: Reading | undefined;
	if (wrist && (closeIn || !readings.length)) {
		let r = replies.get("wrist");
		if (r === undefined && !replies.has("wrist")) {
			r = await askView(o, wrist, asked);
			out.calls = asked.calls;
			out.hint_kept = asked.hintKept;
		}
		if (r?.target) {
			const box = r.inventory.find((e) => e.label === r.target)?.box;
			if (box) {
				const wr = readingOf(wrist, box, r.target, r.inventory, false);
				const longest = Math.max(wr.box[2] - wr.box[0], wr.box[3] - wr.box[1]);
				const held = o.held_px ? Math.max(o.held_px[2] - o.held_px[0], o.held_px[3] - o.held_px[1]) : undefined;
				if (held !== undefined && longest <= HELD_WIDER * held)
					out.method += "the wrist boxed only the payload in the jaws, not the target; ";
				else {
					wristReading = wr;
					for (const e of r.inventory) labels.add(e.label);
				}
			}
		}
	}
	out.candidates = [...labels];
	if (!readings.length && !wristReading) {
		out.error = o.wrist_only
			? `the wrist camera did not find "${o.words}" below the jaws`
			: `no view found "${o.words}": raise the tool, move the arm out of the fixed cameras' line of sight, or name it by a distinguishing property`;
		return out;
	}

	// Two fixed views: triangulate.
	if (readings.length >= 2) {
		const [a, b] = readings;
		const middle = (r: Reading): [number, number] => [(r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2];
		const middles = triangulate(a.cam, middle(a), b.cam, middle(b));
		const middlesAgree = middles !== undefined && middles.skew_m * 1000 <= MAX_SKEW_MM;
		if (out.part && middlesAgree && middles) {
			out.part_point = middles.point;
			// What the part stands out of: the biggest box round the part's middle (a cabinet, not the drawer
			// front round the handle), a margin of half the part's box allowed (a handle stands proud of its
			// body's outline), never the robot, and one sharing a word with the words first (smoke 6, goal t0 s1:
			// the side view's handle sat 6 px past the cabinet's box, inside the robot arm's).
			const want = contentWords(o.words);
			const holder = (r: Reading): Box | undefined => {
				const m = middle(r);
				const mn = toNormalised([m[0], m[1], m[0], m[1]], r.cam);
				const margin = TOUCHES_SHARE * Math.max(r.boxNorm[2] - r.boxNorm[0], r.boxNorm[3] - r.boxNorm[1]);
				let best: { box: Box; shares: boolean } | undefined;
				for (const e of r.inventory) {
					if (e.box === r.boxNorm || ROBOT_LABEL.test(e.label)) continue;
					const [x1, y1, x2, y2] = e.box;
					if (mn[0] < x1 - margin || mn[0] > x2 + margin || mn[1] < y1 - margin || mn[1] > y2 + margin) continue;
					const shares = [...contentWords(e.label)].some((w) => want.has(w));
					if (!best || (shares && !best.shares) || (shares === best.shares && boxArea(e.box) > boxArea(best.box)))
						best = { box: e.box, shares };
				}
				return best?.box;
			};
			const ha = holder(a);
			const hb = holder(b);
			if (ha && hb) {
				const pa = toPixels(ha, a.cam);
				const pb = toPixels(hb, b.cam);
				const meet = triangulate(a.cam, [(pa[0] + pa[2]) / 2, (pa[1] + pa[3]) / 2], b.cam, [
					(pb[0] + pb[2]) / 2,
					(pb[1] + pb[3]) / 2,
				]);
				if (meet) out.part_body = meet.point;
			}
		}
		let aim = triangulate(a.cam, a.aim, b.cam, b.aim);
		let crossed: string | undefined;
		if (aim && aim.skew_m * 1000 <= MAX_SKEW_MM && tableZ !== null) {
			const fa = footprintOnPlane(a.cam, a.box, tableZ);
			const fb = footprintOnPlane(b.cam, b.box, tableZ);
			if (fa && fb) {
				const apart = apartXy(fa, fb) * 1000;
				const up = (aim.point[2] - tableZ) * 1000;
				const allowed =
					out.part || heldUp(o.words)
						? PART_HEIGHT_LIMIT_MM
						: Math.max(50, o.top_z_m !== undefined ? (o.top_z_m - tableZ) * 1000 : 0);
				if (apart > MAX_SKEW_MM && (up < 0 || up > allowed))
					crossed = `the two views' rays crossed ${Math.round(up)} mm over the table where their boxes stand ${Math.round(apart)} mm apart: two different things`;
			}
		}
		const tall = (r: Reading) => r.box[3] - r.box[1] >= TALL * (r.box[2] - r.box[0]);
		if (tall(a) && tall(b)) {
			const top = triangulate(a.cam, [(a.box[0] + a.box[2]) / 2, a.box[1]], b.cam, [
				(b.box[0] + b.box[2]) / 2,
				b.box[1],
			]);
			if (top && top.skew_m * 1000 <= TOP_SKEW_MM) out.top_point = top.point;
		}
		const aimFailed = !aim || aim.skew_m * 1000 > MAX_SKEW_MM || crossed !== undefined;
		if (middlesAgree && middles && aimFailed && (a.hiding || b.hiding)) {
			aim = middles;
			crossed = undefined;
			out.method += "the bottom edge was supplied by something in front, so the box middles were used; ";
		}
		if (aim && aim.skew_m * 1000 <= MAX_SKEW_MM && !crossed) {
			out.point = aim.point;
			out.skew_mm = aim.skew_m * 1000;
			out.cameras = [a.view, b.view];
			out.error_mm = Math.max(CENTRED_ERROR_MM, out.skew_mm);
			out.error_class = "how far apart the two views' rays passed";
			out.method += `${a.view} and ${b.view} met ${Math.round(out.skew_mm)} mm apart at the contact edge`;
			if (tableZ !== null) {
				const z = Math.max(tableZ, aim.point[2]);
				const fp = boxOnPlaneCorners(a.cam, a.box, z, o.top_z_m);
				if (fp) {
					out.footprint = fp;
					out.footprint_method = `${a.view}'s outline on the plane`;
				}
			}
			out.label = a.label;
			out.ok = true;
			if (wristReading && !o.holding) wristPosition(o, out, wristReading, tableZ);
			return out;
		}
		// Two views that did not meet (or whose rays never crossed) are doubtful: the one view read below
		// is a fallback, not a place to drive to (spec-locate §5 item 8; step 4 defect 6).
		out.doubtful = true;
		const nearest = nearestTo(readings, o.last_seen, tableZ);
		if (aim) {
			out.skew_mm = aim.skew_m * 1000;
			out.disagreement = `${crossed ?? `the two views' rays passed ${Math.round(aim.skew_m * 1000)} mm apart`}; ${nearest.view}'s answer is used`;
		} else out.disagreement = `the two views' rays did not meet; ${nearest.view}'s answer is used`;
		readings.splice(0, readings.length, nearest);
	}

	// Single view.
	const chosen = readings[0] ?? wristReading;
	if (!chosen) {
		out.error = "nothing to read";
		return out;
	}
	if (tableZ === null) {
		out.error = "no commissioned table height: a single view cannot be read onto the table";
		return out;
	}
	const hidden = chosen.hiding !== undefined;
	const topZ = hidden ? undefined : o.top_z_m;
	const point = boxOnPlane(chosen.cam, chosen.box, tableZ, topZ);
	if (!point) {
		out.error = `${chosen.view}'s ray through the box misses the table`;
		return out;
	}
	if (chosen.view === "wrist" && o.last_seen && o.wrist_within_mm !== undefined) {
		const away = apartXy(point, o.last_seen) * 1000;
		if (away > o.wrist_within_mm) {
			out.error = `the wrist's answer is ${Math.round(away)} mm from where the target was, further than the jaws span (${Math.round(o.wrist_within_mm)} mm)`;
			return out;
		}
	}
	out.point = point;
	out.cameras = [chosen.view];
	out.view_name = chosen.view;
	out.box_px = chosen.box;
	if (!(chosen.view === "wrist" && lookingDown(chosen.cam) && topZ === undefined)) {
		const fp = boxOnPlaneCorners(chosen.cam, chosen.box, tableZ, topZ);
		if (fp) {
			out.footprint = fp;
			out.footprint_method = `${chosen.view}'s box on the plane`;
		}
	}
	const bar = planeErrorMm(chosen.cam, chosen.box, tableZ, topZ, o.tallest_mm);
	out.error_mm = bar.error_mm;
	out.error_class = bar.error_class;
	if (chosen.view === "wrist" && !closeIn && out.error_mm < UNCHECKED_FROM_HEIGHT_MM) {
		out.error_mm = UNCHECKED_FROM_HEIGHT_MM;
		out.error_class = "the only view, from up high, unchecked";
	}
	out.label = chosen.label;
	out.method += `${chosen.view} alone read the box onto the ${topZ !== undefined ? "plane halfway up" : "table"}${out.disagreement ? ` (${out.disagreement})` : ""}`;
	if (chosen.view !== "wrist") lookAlikes(o, out, chosen, tableZ);
	if (chosen.hiding && !lookingDown(chosen.cam)) baseOutOfSight(out, chosen, tableZ);
	out.ok = true;
	if (chosen.view !== "wrist" && wristReading) {
		if (wristPosition(o, out, wristReading, tableZ)) return wideBarIsDoubtful(out);
	}
	if (out.part) asAPart(o, out, chosen, tableZ);
	return wideBarIsDoubtful(out);
}

/**
 * A single view read with a bar wider than two views of one object stay within (`MAX_SKEW_MM`) is a
 * doubtful answer (step 4 defect 6: spatial t0 s0 drove 160 mm to the other bowl on a 203 mm bar
 * twice, flagged nothing). The point stands; the flag reaches the proposer, the claim and the
 * executor's re-look rule.
 */
function wideBarIsDoubtful(out: Located): Located {
	if ((out.error_mm ?? 0) > MAX_SKEW_MM && !out.doubtful) {
		out.doubtful = true;
		out.method += `; a ${Math.round(out.error_mm ?? 0)} mm bar from one view: doubtful`;
	}
	return out;
}

/** The reading whose table read is nearest `last_seen`, else the first. */
function nearestTo(readings: Reading[], lastSeen: Vec3 | undefined, tableZ: number | null): Reading {
	if (!lastSeen || tableZ === null) return readings[0];
	let best = readings[0];
	let bestD = Infinity;
	for (const r of readings) {
		const p = footprintOnPlane(r.cam, r.box, tableZ);
		if (!p) continue;
		const d = apartXy(p, lastSeen);
		if (d < bestD) [best, bestD] = [r, d];
	}
	return best;
}

/**
 * The wrist's position read from close in (spec-locate §5.5): its box on the plane at the known top
 * overwrites the error bar with the gap to the current point, and replaces the point when it lies
 * within `wrist_within_mm` of the anchor. Returns true when the wrist was taken.
 */
function wristPosition(o: LocateInput, out: Located, w: Reading, tableZ: number | null): boolean {
	if (o.top_z_m === undefined && lookingDown(w.cam)) return false;
	if (tableZ === null || !out.point) return false;
	const at = boxOnPlane(w.cam, w.box, tableZ, o.top_z_m);
	if (!at) return false;
	let within = o.wrist_within_mm;
	if (!o.last_seen && within !== undefined) within = Math.max(within, out.error_mm ?? 0);
	const gap = apartXy(at, out.point) * 1000;
	out.error_mm = Math.max(CENTRED_ERROR_MM, gap);
	out.error_class = "how far apart the two views that answered put it";
	const anchor = o.last_seen ?? out.point;
	const fromAnchor = apartXy(at, anchor) * 1000;
	if (within === undefined || fromAnchor > within) {
		out.confirmed_mm = gap;
		out.method += `; the wrist put it ${Math.round(gap)} mm away and was not used`;
		return false;
	}
	out.point = at;
	out.cameras = ["wrist"];
	out.confirmed_mm = fromAnchor;
	out.method += `; the wrist, ${Math.round(fromAnchor)} mm from the anchor, gives the position`;
	return true;
}

/** The chosen box's base is hidden by a nearer box (spec-locate §5.6): doubtful, with a wider bar. */
function baseOutOfSight(out: Located, chosen: Reading, tableZ: number) {
	out.occluded = true;
	out.doubtful = true;
	const front = chosen.hiding ? footprintOnPlane(chosen.cam, chosen.hiding, tableZ) : undefined;
	const o = eye3(chosen.cam);
	if (!front || !out.point || apartXy(out.point, o) <= apartXy(front, o)) {
		out.error_mm = Math.max(out.error_mm ?? 0, 2 * PLANE_ERROR_FLOOR_MM);
		out.method += "; its base is hidden behind something in front, so the position is a guess";
		return;
	}
	const span = apartXy(out.point, front) * 1000;
	out.error_mm = Math.max(out.error_mm ?? 0, span / 2);
	out.error_class = "the width of the strip its base could be hiding in";
	out.method += `; its base is hidden: it stands somewhere in a ${Math.round(span)} mm strip behind what is in front`;
}
const eye3 = (c: Cam): Vec3 => [c.cam2base[0][3], c.cam2base[1][3], c.cam2base[2][3]];

const contentWords = (s: string) => new Set(tokens(s).filter((w) => !IGNORED.has(w)));
/** A label naming the robot itself, never what a part stands out of. */
const ROBOT_LABEL = /\b(?:robot|arm|gripper|jaws?|franka|manipulator|end effector)\b/i;

/** Other labels sharing a word with the target, not one thing with it, read within the bar (spec-locate §5.7). */
function lookAlikes(o: LocateInput, out: Located, chosen: Reading, tableZ: number) {
	const bar = out.error_mm ?? 20;
	const want = contentWords(o.words);
	if (!out.point) return;
	for (const e of chosen.inventory) {
		if (e.label === chosen.label) continue;
		const theirs = contentWords(e.label);
		if (![...theirs].some((w) => want.has(w))) continue;
		if (oneThing(e.box, chosen.boxNorm)) continue;
		const p = footprintOnPlane(chosen.cam, toPixels(e.box, chosen.cam), tableZ);
		if (!p) continue;
		const d = apartXy(p, out.point) * 1000;
		if (d <= bar) out.look_alikes.push([e.label, Math.round(d)]);
	}
	if (out.look_alikes.length)
		out.ambiguous = `AMBIGUOUS: TWO THINGS HERE MATCH THOSE WORDS. ${out.look_alikes
			.map(([l, d]) => `"${l}" is ${d} mm away`)
			.join(
				", ",
			)}, within the ${Math.round(bar)} mm bar. Name a property that tells them apart: printing, colour, shape, or a neighbour.`;
}
/** The smaller box is at most 0.6 of the larger's area and at least half of it lies inside the larger. */
export function oneThing(a: Box, b: Box): boolean {
	const [small, large] = boxArea(a) <= boxArea(b) ? [a, b] : [b, a];
	const sa = boxArea(small);
	if (sa <= 0) return false;
	return sa <= PART_AT_MOST * boxArea(large) && overlapArea(small, large) >= INSIDE_SHARE * sa;
}

/** The part on its body (spec-locate §5.8), on the single-view path. */
function asAPart(o: LocateInput, out: Located, chosen: Reading, tableZ: number) {
	const part = out.part as string;
	const partBox = chosen.boxNorm;
	const pa = boxArea(partBox);
	const longest = Math.max(partBox[2] - partBox[0], partBox[3] - partBox[1]);
	const margin = TOUCHES_SHARE * longest;
	const want = contentWords(o.words);
	let body: { label: string; box: Box } | undefined;
	let bodyScore = -Infinity;
	for (const e of chosen.inventory) {
		if (e.box === partBox) continue;
		const ea = boxArea(e.box);
		if (!(ea > pa) || ea > BODY_AT_MOST_TIMES * pa) continue;
		const touches =
			e.box[0] <= partBox[2] + margin &&
			e.box[2] >= partBox[0] - margin &&
			e.box[1] <= partBox[3] + margin &&
			e.box[3] >= partBox[1] - margin;
		if (!touches) continue;
		const shares = [...contentWords(e.label)].some((w) => want.has(w)) ? 1 : 0;
		const score = shares * 1e9 - ea;
		if (score > bodyScore) [body, bodyScore] = [e, score];
	}
	const unresolved = (why: string) => {
		out.part_unresolved = true;
		out.doubtful = true;
		out.method += `; the ${part} could not be told from its body (${why})`;
		if (body) out.point = boxOnPlane(chosen.cam, toPixels(body.box, chosen.cam), tableZ, o.top_z_m) ?? out.point;
	};
	if (!body) return unresolved("no body box near it");
	const bodyPx = toPixels(body.box, chosen.cam);
	const anchor = boxOnPlane(chosen.cam, bodyPx, tableZ, o.top_z_m);
	if (!anchor || !out.point) return unresolved("the body's point could not be read");
	out.part_body = anchor;
	const bodyHeight = body.box[3] - body.box[1];
	const onTop = TOP_EDGE_PARTS.includes(part) || partBox[3] <= body.box[1] + ON_TOP_SHARE * bodyHeight;
	const middle: [number, number] = [(chosen.box[0] + chosen.box[2]) / 2, (chosen.box[1] + chosen.box[3]) / 2];
	let point: Vec3 | undefined;
	let planeZ: number | undefined;
	if (lookingDown(chosen.cam)) {
		planeZ =
			onTop && o.top_z_m !== undefined ? o.top_z_m : o.top_z_m !== undefined ? (tableZ + o.top_z_m) / 2 : tableZ;
		point = onPlane(chosen.cam, middle[0], middle[1], planeZ);
	} else point = onUprightPlane(chosen.cam, middle[0], middle[1], anchor);
	if (!point) return unresolved("its ray missed the plane");
	const across = apartXy(point, anchor) * 1000;
	const bar = out.error_mm ?? 20;
	if (!isTopPart(part) && !TOP_EDGE_PARTS.includes(part) && across <= bar)
		return unresolved(`a side part cannot sit in the middle (${Math.round(across)} mm from the body's point)`);
	const height = (point[2] - tableZ) * 1000;
	if (height < 0 || height > PART_HEIGHT_LIMIT_MM)
		return unresolved(`${Math.round(height)} mm over the table is not a part's height`);
	out.point = point;
	out.part_z_mm = height;
	const outline = lookingDown(chosen.cam)
		? boxOnPlaneCorners(chosen.cam, chosen.box, planeZ ?? tableZ, o.top_z_m ?? planeZ)
		: [point];
	if (outline) {
		out.footprint = outline;
		out.footprint_method = `the ${part}'s own box`;
		out.part_outline = outline;
	}
	out.method += `; the ${part} read on ${lookingDown(chosen.cam) ? "its plane" : `the upright plane through "${body.label}"`}`;
}

/** The distance (m) a candidate point is from the stored one in xy, for the sticky rule. */
export const jumped = (stored: Vec3, candidate: Vec3) => apartXy(stored, candidate);

/** The planner boxes' middles projected: a wrist box in pixels around what the jaws hold (spec-locate §2.3 `held_px`). */
export function heldPixels(wrist: Camera, toolPoint: Vec3, payloadWidthM: number, approach: Vec3): PxBox | undefined {
	const across: Vec3 = Math.abs(approach[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
	const corners = [
		add(toolPoint, scale(across, payloadWidthM / 2)),
		sub(toolPoint, scale(across, payloadWidthM / 2)),
		add(toolPoint, scale([0, 1, 0], payloadWidthM / 2)),
		sub(toolPoint, scale([0, 1, 0], payloadWidthM / 2)),
	].map((p) => project(wrist, p));
	if (corners.some((c) => !c)) return undefined;
	const us = corners.map((c) => (c as [number, number])[0]);
	const vs = corners.map((c) => (c as [number, number])[1]);
	return [Math.min(...us), Math.min(...vs), Math.max(...us), Math.max(...vs)];
}
