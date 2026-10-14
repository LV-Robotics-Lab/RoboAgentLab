/**
 * MotorMind M2, the executor half (src/modes/motormind/{executor,proposal,gates,mend,supervision,
 * carry,locate,geometry,observe}.ts): pure logic on M1's recordings from the real LIBERO env server
 * (test/fixtures/motormind/exec-*.json, bjb2 .local/runs/mm-step1/2026-10-12-fix2), the camera rig
 * rebuilt from the recorded mount and checked against the recorded projections, a perfect detector
 * through locate (spec-locate §13.4), the clearance tracker on the recorded blind-onset rows, and the
 * generator protocol with a scripted model and a fake robot (C19: the observe call points).
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import type { ImageContent } from "@earendil-works/pi-ai";
import { carryLegLabel, DROP_LABEL, LIFT_LABEL, TEST_REST, toDelta } from "../src/modes/motormind/command.ts";
import {
	type Ask,
	type Box,
	type Camera,
	type CameraName,
	type Executor,
	type Frame,
	type FrameCache,
	hasImages,
	type MmCall,
	type MmReply,
	missionClock,
	type Role,
	type RoleReply,
	type SegmentReport,
	type Subgoal,
	type SubgoalInput,
} from "../src/modes/motormind/contract.ts";
import { actArgs, runSubgoal } from "../src/modes/motormind/executor.ts";
import { apply, dot, IDENTITY, labelDirection, matmul, sub, transpose, unit } from "../src/modes/motormind/frames.ts";
import {
	actsOnWhatItNames,
	centreMiss,
	closeDepthGate,
	closeOutcome,
	dragging,
	edgeOffsetMm,
	edgePoint,
	isTurnStep,
	jawLine,
	jawsChanged,
	jawsShut,
	keepBlockedAxes,
	newState,
	noMotion,
	noteEmptyClose,
	noteGrasped,
	partOf,
	quarterTurnForReach,
	receiverWords,
	residualOf,
	shoved,
	sidePartWords,
	sideTakeBlock,
	stoppedAlongTheJaws,
	takesOver,
	wantsHolding,
	wantsRelease,
	whereTheTargetIs,
	widerThanJaws,
} from "../src/modes/motormind/gates.ts";
import { boxOnPlane, planeErrorMm, project, toPixels, triangulate } from "../src/modes/motormind/geometry.ts";
import { aPartWord, locate, namesAPart, parseLocateReply } from "../src/modes/motormind/locate.ts";
import { firstMoveIsATest, liftBeforeSideways, splitACarry } from "../src/modes/motormind/mend.ts";
import { ClearanceTracker, legend, TopFace } from "../src/modes/motormind/observe.ts";
import {
	commandSchemaText,
	emptyCommand,
	historyText,
	propose,
	SCHEMA_CHARS,
	validateProposal,
} from "../src/modes/motormind/proposal.ts";
import { parseSupervision, supervise } from "../src/modes/motormind/supervision.ts";
import type { Caps, Delta, Mat3, Pose, Vec3 } from "../src/modes/motormind/types.ts";

type Json = Record<string, any>;
const fixtures = join(import.meta.dirname, "fixtures", "motormind");
const read = (name: string): Json => JSON.parse(readFileSync(join(fixtures, name), "utf8"));
const RIG = read("exec-rig-spatial.json");
const SEG = read("exec-segments.json");
const CLR = read("exec-clearance-rows.json");

/** LIBERO's capability table (the server's values; test/motormind.test.ts has the same). */
const CAPS: Caps = {
	max_translation_m: 0.05,
	max_rotation_rad: 0.5,
	max_speed_m_s: 1.0,
	max_rotation_speed_rad_s: 10,
	default_speed_m_s: 0.05,
	default_rotation_speed_rad_s: Math.PI / 3,
	control_point_offset_m: 0,
	table_z_m: RIG.table_z_m,
	supports_contact_stop: true,
	contact_stop_trusted: true,
	supports_home: false,
	gripper_gap_measured: true,
	gripper_open_span_m: 0.0788,
	gripper_pad_length_m: 0.017,
	comfortable_reach_m: 0.727,
	reach_m: 0.78,
	tilt_limits_deg: { roll_cw: 88.6, roll_ccw: 83.7, pitch_down: 91.7, pitch_up: 53.6 },
	tilted_floor_m: 0.1,
	tilt_reach_deg: 80,
	max_object_height_m: 0.16,
	subgoal_median_s: 21.0,
	subgoal_verdict_s: 4.6,
	arrival_tolerance_m: 0.003,
	notes: [],
};
const close = (a: number, b: number, tol: number, what = "") =>
	assert.ok(Math.abs(a - b) <= tol, `${what} ${a} != ${b} (tol ${tol})`);

// ---------------------------------------------------------------------------
// the camera rig, rebuilt from the recorded mount and checked against the recorded projections

/** A MuJoCo quaternion (wxyz) as a rotation matrix. */
function quatToMatrix(q: number[]): Mat3 {
	const [w, x, y, z] = q;
	return [
		[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
		[2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
		[2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
	];
}
/** A MuJoCo camera frame (x right, y up, looks along -z) in the OpenCV convention (x right, y down, z forward). */
const mujocoToOpenCv = (r: Mat3): Mat3 =>
	matmul(r, [
		[1, 0, 0],
		[0, -1, 0],
		[0, 0, -1],
	]);
const basePose: Pose = { position_m: RIG.base_pose.position_m, rotation: RIG.base_pose.rotation };
const toBase = (p: Vec3): Vec3 => apply(transpose(basePose.rotation), sub(p, basePose.position_m));
const K: Mat3 = RIG.K;

function cameraOf(name: CameraName, rotCv: Mat3, posWorld: Vec3): Camera {
	const r = matmul(transpose(basePose.rotation), rotCv);
	const t = toBase(posWorld);
	return {
		name,
		width: RIG.render,
		height: RIG.render,
		intrinsic: K,
		cam2base: [
			[r[0][0], r[0][1], r[0][2], t[0]],
			[r[1][0], r[1][1], r[1][2], t[1]],
			[r[2][0], r[2][1], r[2][2], t[2]],
			[0, 0, 0, 1],
		],
		marked512: "/9j/stand-in",
		raw512: "/9j/stand-in",
	};
}
/** The scene camera: LIBERO's agentview, converted. */
const scene = cameraOf("scene", mujocoToOpenCv(quatToMatrix(RIG.agentview.quat_wxyz)), RIG.agentview.pos);
/** The side camera: the mount rule (DESIGN §3.2) applied to the recorded standoff. */
const side = (() => {
	const aim: Vec3 = [
		basePose.position_m[0] + RIG.side_mount.aim_xy_offset_m[0],
		basePose.position_m[1] + RIG.side_mount.aim_xy_offset_m[1],
		RIG.table_top_world_m,
	];
	const d = Math.hypot(...sub(RIG.agentview.pos, aim));
	close(d, RIG.side_mount.standoff_m, 1e-3, "standoff");
	const el = (RIG.side_mount.elevation_deg * Math.PI) / 180;
	const eye: Vec3 = [aim[0], aim[1] + RIG.side_mount.sign_y * d * Math.cos(el), aim[2] + d * Math.sin(el)];
	const forward = unit(sub(aim, eye));
	const right = unit([
		forward[1] * 1 - forward[2] * 0,
		forward[2] * 0 - forward[0] * 1,
		forward[0] * 0 - forward[1] * 0,
	]);
	const down: Vec3 = [
		forward[1] * right[2] - forward[2] * right[1],
		forward[2] * right[0] - forward[0] * right[2],
		forward[0] * right[1] - forward[1] * right[0],
	];
	const rotCv: Mat3 = [
		[right[0], down[0], forward[0]],
		[right[1], down[1], forward[1]],
		[right[2], down[2], forward[2]],
	];
	return cameraOf("side", rotCv, eye);
})();
const objectsBase: Record<string, Vec3> = Object.fromEntries(
	Object.entries(RIG.objects_world as Record<string, Vec3>).map(([k, p]) => [k, toBase(p)]),
);
const TABLE_Z: number = RIG.table_z_m;

test("executor: the rebuilt camera rig reproduces M1's recorded projections of every object (scene and side)", () => {
	for (const cam of [scene, side]) {
		const recorded = RIG.recorded_uv[cam.name] as Record<string, [number, number]>;
		for (const [name, uv] of Object.entries(recorded)) {
			const p = project(cam, objectsBase[name]);
			assert.ok(p, `${cam.name} ${name} projects`);
			close(p[0], uv[0], 1.5, `${cam.name} ${name} u`);
			close(p[1], uv[1], 1.5, `${cam.name} ${name} v`);
		}
	}
	// The side camera aims at the base's +0.47 m on the table: the image centre (C10).
	const aim = toBase([basePose.position_m[0] + 0.47, basePose.position_m[1], RIG.table_top_world_m]);
	const uv = project(side, aim) as [number, number];
	close(uv[0], 384, 0.5, "aim u");
	close(uv[1], 384, 0.5, "aim v");
	// The tool point as recorded, in both fixed views.
	const tool = toBase(RIG.eef_pos_world);
	for (const cam of [scene, side]) {
		const p = project(cam, tool) as [number, number];
		close(p[0], RIG.recorded_tool_uv[cam.name][0], 1.5, `${cam.name} tool u`);
		close(p[1], RIG.recorded_tool_uv[cam.name][1], 1.5, `${cam.name} tool v`);
	}
});

/** A perfect detector: the object's extents (m) as an upright box on the table, projected to a normalized box. */
const EXTENTS: Record<string, { rx: number; ry: number; h: number }> = {
	akita_black_bowl_1: { rx: 0.08, ry: 0.08, h: 0.045 },
	akita_black_bowl_2: { rx: 0.08, ry: 0.08, h: 0.045 },
	plate_1: { rx: 0.1, ry: 0.1, h: 0.015 },
	glazed_rim_porcelain_ramekin_1: { rx: 0.045, ry: 0.045, h: 0.045 },
	cookies_1: { rx: 0.045, ry: 0.035, h: 0.08 },
};
function perfectBox(cam: Camera, name: string): Box {
	const c = objectsBase[name];
	const e = EXTENTS[name];
	const us: number[] = [];
	const vs: number[] = [];
	for (const sx of [-1, 1])
		for (const sy of [-1, 1])
			for (const z of [TABLE_Z, TABLE_Z + e.h]) {
				const p = project(cam, [c[0] + sx * e.rx, c[1] + sy * e.ry, z]) as [number, number];
				us.push(p[0]);
				vs.push(p[1]);
			}
	const s = 1000 / RIG.render;
	return [Math.min(...us) * s, Math.min(...vs) * s, Math.max(...us) * s, Math.max(...vs) * s];
}
const noAsk: Ask = async () => {
	throw new Error("locate must not call the model when every fixed view has a planner box");
};

test("executor: a perfect detector through locate meets within 24 mm and lands near the true middle (spec-locate §13.4)", async () => {
	for (const name of Object.keys(EXTENTS)) {
		const truth = objectsBase[name];
		const l = await locate({
			views: { scene, side },
			words: name.replace(/_\d$/, "").replace(/_/g, " "),
			table_z_m: TABLE_Z,
			tool_z_m: TABLE_Z + 0.3,
			boxes: { scene: perfectBox(scene, name), side: perfectBox(side, name) },
			tallest_mm: 160,
			holding: false,
			ask: noAsk,
		});
		assert.ok(l.ok, `${name}: ${l.error}`);
		assert.equal(l.calls, 0);
		assert.ok((l.skew_mm as number) <= 24, `${name} skew ${l.skew_mm}`);
		const err = Math.hypot((l.point as Vec3)[0] - truth[0], (l.point as Vec3)[1] - truth[1]) * 1000;
		if (name === "plate_1") {
			// A 200 mm wide, 15 mm tall plate: its two contact edges stand over 50 mm apart and the rays meet
			// under the table, so the pair is refused as two different things and the scene view is read alone.
			assert.deepEqual(l.cameras, ["scene"]);
			assert.equal(l.doubtful, true);
			assert.match(l.disagreement ?? "", /two different things/);
			assert.ok(err <= 60, `${name} off by ${err.toFixed(1)} mm`);
		} else {
			assert.deepEqual(l.cameras, ["scene", "side"]);
			assert.ok(!l.disagreement, `${name} ${l.disagreement}`);
			// The contact edges of two oblique views meet short of the middle by part of the object's size.
			assert.ok(
				err <= Math.max(30, EXTENTS[name].rx * 1000 * 0.6),
				`${name} off by ${err.toFixed(1)} mm (${l.method})`,
			);
		}
		// The rays may meet under the table; the footprint never stands below it.
		assert.ok(l.footprint && l.footprint.length >= 2);
		for (const c of l.footprint) assert.ok(c[2] >= TABLE_Z - 1e-9, `${name} footprint z ${c[2]}`);
		// A single oblique view's bar covers its own error.
		for (const cam of [scene, side]) {
			const box = toPixels(perfectBox(cam, name), cam);
			const p = boxOnPlane(cam, box, TABLE_Z) as Vec3;
			const bar = planeErrorMm(cam, box, TABLE_Z, undefined);
			const single = Math.hypot(p[0] - truth[0], p[1] - truth[1]) * 1000;
			assert.ok(
				single <= bar.error_mm + EXTENTS[name].rx * 1000 * 0.5,
				`${cam.name} ${name}: ${single.toFixed(1)} mm vs bar ${bar.error_mm.toFixed(1)}`,
			);
			assert.ok(bar.error_mm >= 20);
		}
	}
	// Two different objects' boxes cross far apart: the pair is refused, one view is read, doubtful.
	const l = await locate({
		views: { scene, side },
		words: "akita black bowl",
		table_z_m: TABLE_Z,
		tool_z_m: TABLE_Z + 0.3,
		boxes: { scene: perfectBox(scene, "akita_black_bowl_1"), side: perfectBox(side, "cookies_1") },
		tallest_mm: 160,
		holding: false,
		ask: noAsk,
	});
	assert.ok(l.ok);
	assert.equal(l.doubtful, true);
	assert.ok(l.disagreement, "two different things");
	assert.deepEqual(l.cameras, ["scene"]);
});

test("executor: locate reads the model's boxes, flags a hidden base, needs a fixed view and a table height", async () => {
	const bowl = "akita_black_bowl_1";
	const inventory = (cam: Camera) =>
		Object.keys(EXTENTS).map((n) => ({
			label: n.replace(/_\d$/, "").replace(/_/g, " ") + (n.endsWith("2") ? " (second)" : ""),
			bbox_2d: perfectBox(cam, n).map(Math.round),
		}));
	const asked: Role[] = [];
	const ask: Ask = async (role, prompt, images) => {
		asked.push(role);
		assert.equal(images.length, 1);
		assert.match(prompt.text, /512 x 512/);
		const which = images[0].data === "scene" ? scene : side;
		return ok({ objects: inventory(which), target: "akita black bowl", confusable: [], note: "" });
	};
	const tag = (c: Camera): Camera => ({ ...c, raw512: c.name, marked512: c.name });
	const l = await locate({
		views: { scene: tag(scene), side: tag(side) },
		words: "akita black bowl",
		table_z_m: TABLE_Z,
		tool_z_m: TABLE_Z + 0.3,
		tallest_mm: 160,
		holding: false,
		ask,
	});
	assert.ok(l.ok, l.error);
	assert.equal(l.calls, 2);
	assert.deepEqual(asked, ["locate", "locate"]);
	assert.equal(l.label, "akita black bowl");
	assert.ok(l.candidates.includes("plate"));
	const err =
		Math.hypot((l.point as Vec3)[0] - objectsBase[bowl][0], (l.point as Vec3)[1] - objectsBase[bowl][1]) * 1000;
	assert.ok(err <= 50, `${err}`);
	// A reply naming no label: not visible in that view; the other reads alone, with a wider bar.
	const one: Ask = async (_r, _p, images) =>
		ok(
			images[0].data === "scene"
				? { objects: inventory(scene), target: "akita black bowl" }
				: { objects: [], target: "" },
		);
	const l1 = await locate({
		views: { scene: tag(scene), side: tag(side) },
		words: "akita black bowl",
		table_z_m: TABLE_Z,
		tool_z_m: TABLE_Z + 0.3,
		tallest_mm: 160,
		holding: false,
		ask: one,
	});
	assert.ok(l1.ok);
	assert.deepEqual(l1.cameras, ["scene"]);
	assert.ok((l1.error_mm as number) >= 20);
	// A failed call is recorded apart from "not visible" (D8), with the same effect.
	const failing: Ask = async (_r, _p, images) =>
		images[0].data === "scene"
			? ok({ objects: inventory(scene), target: "akita black bowl" })
			: ({ ok: false, error: "boom", error_kind: "transport", model: "m", usage: USAGE, attempts: 3 } as RoleReply);
	const l2 = await locate({
		views: { scene: tag(scene), side: tag(side) },
		words: "akita black bowl",
		table_z_m: TABLE_Z,
		tool_z_m: TABLE_Z + 0.3,
		tallest_mm: 160,
		holding: false,
		ask: failing,
	});
	assert.ok(l2.ok);
	assert.deepEqual(
		l2.failures.map((f) => f.view),
		["side"],
	);
	// No table height: a single view cannot be read.
	const l3 = await locate({
		views: { scene: tag(scene), side: tag(side) },
		words: "akita black bowl",
		table_z_m: null,
		tool_z_m: 0.3,
		tallest_mm: 160,
		holding: false,
		ask: one,
	});
	assert.equal(l3.ok, false);
	assert.match(l3.error ?? "", /table height/);
	// No fixed view: an error even for a wrist-only look.
	const l4 = await locate({
		views: { wrist: { ...tag(scene), name: "wrist" } },
		words: "bowl",
		table_z_m: TABLE_Z,
		tool_z_m: 0.3,
		tallest_mm: 160,
		holding: false,
		ask: one,
		wrist_only: true,
	});
	assert.equal(l4.ok, false);
	assert.match(l4.error ?? "", /fixed view/);
	// A hidden base: a nearer box under the chosen one's bottom-centre marks it occluded and doubtful.
	const hidden: Ask = async () =>
		ok({
			objects: [
				{ label: "bowl", bbox_2d: [400, 300, 520, 420] },
				{ label: "box in front", bbox_2d: [380, 380, 560, 600] },
			],
			target: "bowl",
		});
	const l5 = await locate({
		views: { scene: tag(scene) },
		words: "bowl",
		table_z_m: TABLE_Z,
		tool_z_m: 0.3,
		tallest_mm: 160,
		holding: false,
		ask: hidden,
	});
	assert.ok(l5.ok);
	assert.equal(l5.occluded, true);
	assert.equal(l5.doubtful, true);
	assert.ok((l5.error_mm as number) >= 40);
	// The reply parser: bare lists, bbox / box_2d / {x1..y2}, a target that is not a label.
	const parsed = parseLocateReply([
		{ name: "a", bbox: [1, 2, 3, 4] },
		{ label: "b", box_2d: { x1: 5, y1: 6, x2: 7, y2: 8 } },
		{ label: "", bbox_2d: [1, 2, 3, 4] },
	]);
	assert.deepEqual(
		parsed.inventory.map((e) => e.label),
		["a", "b"],
	);
	assert.equal(parsed.target, "");
	assert.equal(
		parseLocateReply({ objects: [{ label: "Mug", bbox_2d: [1, 2, 3, 4] }], target: " mug " }).target,
		"Mug",
	);
	assert.equal(namesAPart("the handle of the mug"), "handle");
	assert.equal(namesAPart("the mug handle"), "handle");
	assert.equal(namesAPart("the near side of the bowl"), "side");
	assert.equal(namesAPart("akita black bowl"), undefined);
	const t = triangulate(
		scene,
		project(scene, objectsBase[bowl]) as [number, number],
		side,
		project(side, objectsBase[bowl]) as [number, number],
	);
	assert.ok(t && t.skew_m < 1e-6, "rays through the same point meet");
});

// ---------------------------------------------------------------------------
// clearance tracking and the top face on the recorded blind-onset rows

const SPATIAL_ROWS = CLR.suites.libero_spatial;
const USAGE = { cost_usd: 0, tokens_in: 0, tokens_out: 0, latency_s: 0 };
const ok = (json: unknown): RoleReply => ({
	ok: true,
	text: JSON.stringify(json),
	json,
	model: "fake/m",
	usage: USAGE,
	attempts: 1,
});

/** A frame of the recorded descent: the tool at `h_mm` over the table, the wrist's reading as recorded. */
function rowFrame(row: Json, cameras = false): Frame {
	const z = TABLE_Z + row.h_mm / 1000;
	return frameAt(
		{
			position_m: [0.575, 0, z],
			rotation: [
				[1, 0, 0],
				[0, -1, 0],
				[0, 0, -1],
			],
		},
		{
			clearance: {
				seen: row.seen,
				mm: row.mm,
				near_min_range: row.near,
				jaws_down: true,
				...(row.seen ? {} : { reason: "blind" }),
			},
			cameras,
			clock: row.h_mm,
		},
	);
}
function frameAt(
	pose: Pose,
	o: {
		clearance?: Frame["clearance"];
		cameras?: boolean;
		clock?: number;
		holding?: boolean | null;
		opening?: number;
		commanded?: "open" | "close" | "hold";
		episode_over?: boolean;
	} = {},
): Frame {
	const wrist: Camera = {
		...scene,
		name: "wrist",
		intrinsic: [
			[500.44, 0, 384],
			[0, 500.44, 384],
			[0, 0, 1],
		],
		depth_min_range_m: 0.21,
	};
	const opening = o.opening ?? 79.99;
	return {
		obs_id: 1,
		frame_clock_s: o.clock ?? 1.65,
		pose,
		base_pose: basePose,
		caps: CAPS,
		table_z_m: TABLE_Z,
		gripper: { opening_mm: opening, commanded: o.commanded ?? "open", moving: false },
		grasp: {
			holding: o.holding ?? null,
			reason: o.holding === null || o.holding === undefined ? "the last gripper command was not close" : "",
			opening_mm: opening,
			closed_fraction: opening / 78.8,
		},
		...(o.clearance ? { clearance: o.clearance } : {}),
		cameras: o.cameras === false ? {} : { scene, side, wrist },
		camera_names: ["scene", "side", "wrist"],
		episode_over: o.episode_over ?? false,
		terminated: false,
		truncated: false,
		degraded: {},
	};
}

test("executor: the clearance tracker dead-reckons through the recorded blind zone and discards a reading 15 mm below the latched surface", () => {
	const tracker = new ClearanceTracker();
	let lastSeen: Json | undefined;
	for (const row of SPATIAL_ROWS.rows) {
		const r = tracker.update(rowFrame(row));
		if (row.seen) {
			assert.equal(r.seen, true);
			assert.equal(r.clearance_mm, row.mm);
			assert.equal(r.tracked, undefined);
			lastSeen = row;
		} else {
			assert.equal(r.seen, false);
			assert.ok(r.tracked, `tracked at ${row.h_mm} mm`);
			// The last reading minus the descent since: within a few mm of the true height (the reading itself read ~2 mm short).
			const expected = lastSeen!.mm - (lastSeen!.h_mm - row.h_mm);
			close(r.tracked!.mm, expected, 1e-6, `tracked at ${row.h_mm}`);
			close(r.tracked!.mm, row.h_mm, 4, `tracked vs height at ${row.h_mm}`);
			assert.equal(r.clearance_mm, r.tracked!.mm);
		}
	}
	// The blind onset the probe recorded: the last reading was 116-118 mm over the table.
	assert.ok(lastSeen!.h_mm >= 116 && lastSeen!.h_mm <= 135, `${lastSeen!.h_mm}`);
	// A higher surface latched at 60 mm; the camera then reads the table 110 mm behind it: discarded, the track continues.
	const t2 = new ClearanceTracker();
	const over = (h: number, mm: number | null, seen: boolean) =>
		t2.update(rowFrame({ h_mm: h, mm, seen, near: false, axis_z: -1 }));
	assert.equal(over(200, 60, true).clearance_mm, 60);
	const far = over(190, 100, true);
	assert.equal(far.discarded, true);
	assert.equal(far.seen, false);
	close(far.clearance_mm!, 50, 1e-9);
	// A reading within 15 mm is taken as the surface again.
	assert.equal(over(180, 45, true).seen, true);
	// A turn resets the track; 80 mm sideways unlocks it.
	assert.equal(
		t2.update(
			frameAt(
				{ position_m: [0.575, 0, TABLE_Z + 0.17], rotation: IDENTITY },
				{ clearance: { seen: false, mm: null } },
			),
		).tracked,
		undefined,
	);
	over(200, 60, true);
	assert.equal(
		t2.update(
			frameAt(
				{
					position_m: [0.575, 0.09, TABLE_Z + 0.19],
					rotation: [
						[1, 0, 0],
						[0, -1, 0],
						[0, 0, -1],
					],
				},
				{ clearance: { seen: false, mm: null } },
			),
		).tracked,
		undefined,
	);
});

test("executor: the top face is the median of the last 12 faces seen under the jaws over the outline, within [10, 300] mm", () => {
	const faces = new TopFace();
	const outline: Vec3[] = [
		[0.55, -0.02, TABLE_Z],
		[0.6, -0.02, TABLE_Z],
		[0.6, 0.02, TABLE_Z],
		[0.55, 0.02, TABLE_Z],
	];
	// Fifteen frames 200 mm up with the wrist seeing a face 45 mm high, two outliers among them.
	const heights = [45, 44, 46, 45, 45, 120, 45, 44, 46, 45, 2, 45, 45, 46, 44];
	for (const h of heights) {
		const f = rowFrame({ h_mm: 200, seen: true, mm: 200 - h, near: false, axis_z: -1 });
		faces.update(f, { clearance_mm: 200 - h, seen: true }, outline, 0.0394);
	}
	assert.equal(faces.faces.length, 12);
	close(faces.heightMm()!, 45, 0.5);
	close(faces.topZ(TABLE_Z)!, TABLE_Z + 0.045, 1e-6);
	// Not over the outline, or no surface seen: nothing is added.
	const before = faces.faces.length;
	faces.update(
		frameAt(
			{
				position_m: [0.9, 0.3, TABLE_Z + 0.2],
				rotation: [
					[1, 0, 0],
					[0, -1, 0],
					[0, 0, -1],
				],
			},
			{ clearance: { seen: true, mm: 150 } },
		),
		{ clearance_mm: 150, seen: true },
		outline,
		0.0394,
	);
	faces.update(
		rowFrame({ h_mm: 200, seen: false, mm: null, near: true, axis_z: -1 }),
		{ clearance_mm: 150, seen: false },
		outline,
		0.0394,
	);
	assert.equal(faces.faces.length, before);
	// Before any face, the two views' top point gives an early height when plausible.
	const early = new TopFace();
	early.offerTopPoint([0.5, 0, TABLE_Z + 0.5], TABLE_Z);
	assert.equal(early.heightMm(), undefined);
	early.offerTopPoint([0.5, 0, TABLE_Z + 0.08], TABLE_Z);
	close(early.heightMm()!, 80, 1e-9);
	// The fallback: whatever is under the tool.
	close(
		TopFace.fallbackHeight(rowFrame({ h_mm: 150, seen: true, mm: 100, near: false, axis_z: -1 }), {
			clearance_mm: 100,
			seen: true,
		})!,
		50,
		1e-6,
	);
	// The legend names every camera with a percentage position and a mm scale.
	const text = legend(rowFrame({ h_mm: 150, seen: true, mm: 148, near: false, axis_z: -1 }, true));
	assert.match(text, /scene: .*% from the left and .*% from the top; at its depth 10% of the frame is about \d+ mm/);
	assert.match(text, /side: /);
	assert.match(text, /wrist: /);
});

// ---------------------------------------------------------------------------
// the measured segments M1 recorded, read by the rules

test("executor: the recovery rules read M1's recorded segments by their fields, never the message", () => {
	const blocked: SegmentReport = SEG.blocked;
	const contact: SegmentReport = SEG.contact;
	const skidded: SegmentReport = SEG.skidded;
	const timedOut: SegmentReport = SEG.timed_out;
	assert.equal(blocked.outcome, "limited");
	assert.equal(blocked.abort_reason, "blocked");
	// A blocked descent that made 15 of 50 mm: the cycle "did not move" (stopped early), unless the jaws changed.
	const down50 = toDelta(
		{ index: 0, type: "move", direction: "down", distance_mm: 50, frame: "base", note: "" },
		CAPS,
	);
	assert.equal(noMotion([blocked], [down50], false).frozen, true);
	assert.equal(noMotion([blocked], [down50], true).frozen, false);
	// The skid: 5.3 mm sideways on a 10 mm rise; a descent with that drift would count as shoved.
	assert.equal(skidded.abort_reason, "skidded");
	assert.equal(shoved([{ ...skidded, axis: [0, 0, -1] }], [], undefined), true);
	assert.equal(shoved([skidded], [], undefined), false);
	assert.equal(shoved([], [{ kind: "translate", push: true } as never], undefined), true);
	// Contact: a push that met something continues the batch (done / limited / contact do).
	assert.equal(contact.outcome, "contact");
	assert.ok(["done", "limited", "contact"].includes(contact.outcome));
	// A timed-out segment is failed: the outcome is unknown.
	assert.equal(timedOut.outcome, "failed");
	assert.equal(timedOut.timed_out, true);
	// Dragging: holding, jaws down, a sideways move blocked.
	const f = frameAt(
		{
			position_m: [0.57, 0, 0.1],
			rotation: [
				[1, 0, 0],
				[0, -1, 0],
				[0, 0, -1],
			],
		},
		{ holding: true, opening: 63, commanded: "close" },
	);
	const s = newState({
		subgoal: "carry the bowl to the plate",
		criterion: "the bowl rests on the plate",
		target: "the bowl",
		planBoxes: {},
	});
	assert.equal(dragging({ ...blocked, axis: [0, 1, 0] }, f, CAPS), true);
	assert.equal(dragging(blocked, f, CAPS), false);
	// An empty close (M1 C8: 1.14 mm, holding false) at a grasp step records the spot and the recovery height.
	const emptyClose: SegmentReport = {
		label: "gripper close",
		kind: "gripper",
		outcome: "done",
		moved_mm: 0,
		along_axis_mm: 0,
		lateral_mm: 0,
		turned_deg: 0,
		ticks: 32,
		elapsed_s: 1,
		detail: { opening_mm: SEG.gripper.empty_close_mm },
		grasp: {
			holding: false,
			reason: "opening 1.1 mm is within the 3 mm empty band",
			opening_mm: SEG.gripper.empty_close_mm,
		},
		pose_after: { position_mm: [572, 0, 30], rotation: IDENTITY },
	};
	assert.equal(closeOutcome(emptyClose, s), false);
	noteEmptyClose(s, emptyClose, true);
	assert.equal(s.emptyCloses.length, 1);
	close(s.unrecoveredZ!, 0.03, 1e-9);
	// A close on the can (M1 C8 on libero_object: 63 mm, holding true) holds; one thinner than 4% of a 63 mm object does not.
	const can: SegmentReport = {
		...emptyClose,
		detail: { opening_mm: 63 },
		grasp: { holding: true, reason: "", opening_mm: 63 },
	};
	assert.equal(closeOutcome(can, s), true);
	s.lastWidthMm = 63;
	assert.equal(
		closeOutcome({ ...can, detail: { opening_mm: 2 }, grasp: { holding: true, reason: "", opening_mm: 2 } }, s),
		false,
	);
	assert.equal(wantsHolding("the bowl is held in the jaws"), true);
	assert.equal(wantsHolding("the bowl is released on the plate"), false);
	assert.equal(wantsRelease("the bowl rests on the plate, jaws open"), true);
});

test("executor: the menders insert and split with the reference labels the net-motion rule reads (DESIGN §12 B1)", () => {
	const s = newState({
		subgoal: "move the bowl over the plate",
		criterion: "the bowl is held over the plate",
		target: "the plate",
		planBoxes: {},
	});
	s.startPose = { position_m: [0.5, 0, TABLE_Z + 0.3], rotation: IDENTITY };
	const f = frameAt(
		{
			position_m: [0.5, 0, TABLE_Z + 0.03],
			rotation: [
				[1, 0, 0],
				[0, -1, 0],
				[0, 0, -1],
			],
		},
		{ holding: true, opening: 63, commanded: "close" },
	);
	const mk = (type: "move" | "gripper", o: Json) => toDelta({ index: 0, type, frame: "base", note: "", ...o }, CAPS);
	s.payload = { width_mm: 63, span_mm: 63, edge: false, bumped: false, tracking: true };
	const deltas = [mk("move", { direction: "left", distance_mm: 120 }), mk("gripper", { state: "open" })];
	const what = liftBeforeSideways(deltas, s, f, CAPS, { gripperOnly: false, heightMm: 45 });
	assert.ok(what, "a lift was inserted");
	assert.equal(deltas[0].label, LIFT_LABEL);
	assert.equal(deltas[2].label, DROP_LABEL);
	// The lift brings the tool 100 mm over the receiver (height 0 while holding): 100 - 30 = 70 mm up, and 70 mm back down.
	close(deltas[0].magnitude, 0.07, 1e-9);
	close(deltas[2].magnitude, 0.07, 1e-9);
	// Under the label rule the lift names no direction and the way down reads as a backward move; the carry keeps its word.
	assert.equal(labelDirection(deltas[0].label), undefined);
	assert.deepEqual(labelDirection(deltas[2].label)?.axis, [-1, 0, 0]);
	assert.equal(labelDirection(deltas[1].label)?.word, "left");
	// A tracked payload splits the 120 mm carry into legs of at most 50 mm.
	assert.equal(splitACarry(deltas, s, CAPS), true);
	// The inserted 70 mm lift is a carry too (two legs); the sideways move is three.
	assert.equal(deltas.filter((d) => d.label.startsWith(LIFT_LABEL)).length, 2);
	const legs = deltas.filter((d) => d.label.startsWith("move left"));
	assert.equal(legs.length, 3);
	assert.equal(legs[0].label, carryLegLabel("move left 120.0 mm (base)", 1, 3));
	close(legs[0].magnitude, 0.04, 1e-9);
	for (const l of legs) assert.equal(labelDirection(l.label)?.word, "left");
	// After a wall grasp the first move over two pads is a 17 mm test and "(the rest)", at proposal time only.
	const s2 = newState({
		subgoal: "lift the bowl by its rim",
		criterion: "the bowl is held 120 mm up",
		target: "the bowl",
		planBoxes: {},
	});
	s2.payload = { width_mm: 10, edge: true, bumped: false, tracking: true };
	const up = [mk("move", { direction: "up", distance_mm: 100 })];
	assert.equal(firstMoveIsATest(up, s2, CAPS, { running: true, gripperOnly: false }), false);
	assert.equal(firstMoveIsATest(up, s2, CAPS, { running: false, gripperOnly: false }), true);
	assert.equal(up.length, 2);
	close(up[0].magnitude, 0.017, 1e-9);
	assert.ok(up[0].speed <= CAPS.default_speed_m_s);
	assert.ok(up[1].label.endsWith(TEST_REST));
	close(up[1].magnitude, 0.083, 1e-9);
	// The mm_act arguments carry the exact axis, the speed in mm/s and the executor's label.
	const args = actArgs(legs[0], 42);
	assert.equal(args.type, "move");
	assert.deepEqual(args.axis, [0, 1, 0]);
	close(args.distance_mm as number, 40, 1e-9);
	assert.equal(args.speed_mm_s, 50);
	assert.equal(args.label, legs[0].label);
	assert.equal(args.tracked_mm, 42);
	assert.equal(args.stop_on_contact, undefined);
});

// ---------------------------------------------------------------------------
// the proposer and the watcher with a scripted model

function scripted(replies: Partial<Record<Role, (unknown | Error)[]>>): {
	ask: Ask;
	calls: { role: Role; text: string; images: number }[];
} {
	const calls: { role: Role; text: string; images: number }[] = [];
	const ask: Ask = async (role, prompt, images) => {
		calls.push({ role, text: prompt.text, images: images.length });
		const queue = replies[role] ?? [];
		const next = queue.length ? queue.shift() : undefined;
		if (next instanceof Error)
			return { ok: false, error: next.message, error_kind: "transport", model: "fake/m", usage: USAGE, attempts: 3 };
		if (next === undefined)
			return {
				ok: false,
				error: "no scripted reply",
				error_kind: "no_json",
				model: "fake/m",
				usage: USAGE,
				attempts: 1,
			};
		return ok(next);
	};
	return { ask, calls };
}

test("executor: the proposer tries three times on the same pictures and accepts an empty command as no actions", async () => {
	assert.ok(commandSchemaText().length <= SCHEMA_CHARS);
	assert.deepEqual(
		validateProposal({ assessment: "a", done: false, command: null, expect: "e", confidence: 2 }).problems,
		[],
	);
	assert.equal(
		validateProposal({ assessment: "a", done: false, command: null, expect: "e", confidence: 2 }).proposal.confidence,
		1,
	);
	assert.match(validateProposal({ done: "yes", actions: [] }).problems.join(";"), /unknown key "actions"/);
	assert.equal(emptyCommand({ actions: [] }), true);
	assert.equal(emptyCommand(null), true);
	assert.equal(emptyCommand({ type: "move" }), false);
	const bad = {
		assessment: "",
		done: false,
		command: { actions: [{ type: "move", direction: "sideways", distance_mm: 50 }] },
		expect: "",
		confidence: 0.5,
	};
	const good = {
		assessment: "",
		done: false,
		command: { actions: [{ type: "move", direction: "left", distance_mm: 50 }] },
		expect: "the tool moves 50 mm left",
		confidence: 0.9,
	};
	const m = scripted({ executor: [bad, { extra: 1, done: true }, good] });
	const images: ImageContent[] = [{ type: "image", data: "x", mimeType: "image/jpeg" }];
	const gated: string[] = [];
	const p = await propose({
		ask: m.ask,
		subgoal: "s",
		target: "t",
		criterion: "c",
		history: "",
		robot: "r",
		capabilities: "cap",
		images,
		imageNames: ["scene"],
		legend: "",
		forbid: "",
		caps: CAPS,
		gate: (d) => {
			gated.push(d[0].label);
			return undefined;
		},
	});
	assert.ok(p.proposal);
	assert.equal(p.deltas.length, 1);
	assert.equal(p.tries.length, 3);
	assert.match(p.tries[0].refused ?? "", /unknown move direction/);
	assert.match(p.tries[1].refused ?? "", /unknown key "extra"/);
	assert.equal(p.tries[2].refused, undefined);
	// Every try saw the same pictures, and the later prompts carry the earlier refusals.
	assert.deepEqual(
		m.calls.map((c) => c.images),
		[1, 1, 1],
	);
	assert.match(m.calls[2].text, /The previous answer was refused: .*unknown move direction/s);
	assert.match(m.calls[2].text, /unknown key "extra"/);
	assert.deepEqual(gated, ["move left 50.0 mm (base)"]);
	// A gate refusal three times with a claim of done falls back to the claim alone.
	const claim = {
		assessment: "",
		done: true,
		command: { actions: [{ type: "gripper", state: "close" }] },
		expect: "",
		confidence: 0.5,
	};
	const m2 = scripted({ executor: [claim, claim, claim] });
	const p2 = await propose({
		ask: m2.ask,
		subgoal: "s",
		target: "t",
		criterion: "c",
		history: "",
		robot: "r",
		capabilities: "cap",
		images,
		imageNames: ["scene"],
		legend: "",
		forbid: "",
		caps: CAPS,
		gate: () => "not here",
	});
	assert.equal(p2.claimOnly, true);
	assert.equal(p2.proposal?.done, true);
	assert.deepEqual(p2.deltas, []);
	// Three failures and no claim: an error.
	const m3 = scripted({ executor: [new Error("down"), new Error("down"), new Error("down")] });
	const p3 = await propose({
		ask: m3.ask,
		subgoal: "s",
		target: "t",
		criterion: "c",
		history: "",
		robot: "r",
		capabilities: "cap",
		images,
		imageNames: ["scene"],
		legend: "",
		forbid: "",
		caps: CAPS,
		gate: () => undefined,
	});
	assert.equal(p3.proposal, undefined);
	assert.match(p3.error ?? "", /no acceptable proposal/);
	// An empty command is accepted at once.
	const m4 = scripted({ executor: [{ assessment: "done", done: true, command: null, expect: "", confidence: 1 }] });
	const p4 = await propose({
		ask: m4.ask,
		subgoal: "s",
		target: "t",
		criterion: "c",
		history: "",
		robot: "r",
		capabilities: "cap",
		images,
		imageNames: ["scene"],
		legend: "",
		forbid: "",
		caps: CAPS,
		gate: () => undefined,
	});
	assert.equal(p4.proposal?.done, true);
	assert.deepEqual(p4.deltas, []);
	// The history keeps three cycles and reads the measured results.
	const h = historyText("Last time: not done.", [
		{
			index: 0,
			steps: [
				{
					label: "move left 50.0 mm (base) [1/1]",
					kind: "translate",
					outcome: "done",
					moved_mm: 47.5,
					along_axis_mm: 47.5,
					lateral_mm: 0.7,
					turned_deg: 0,
					ticks: 80,
					elapsed_s: 2,
				},
			],
			proposal: { assessment: "", done: false, expect: "", confidence: 1, command: { actions: [] }, deltas: [] },
			timing: { wall_s: 1, motion_s: 2, calls: [] },
		},
		{ index: 1, steps: [], timing: { wall_s: 1, motion_s: 0, calls: [] }, stopped_by: "refused: past the reach" },
	]);
	assert.match(h, /Last time: not done\./);
	assert.match(h, /Cycle 0:[\s\S]*47\.5 mm along/);
	assert.match(h, /out of reach: do not propose that direction again/);
});

test("executor: the watcher judges only a newer frame; no reply means continue, an invalid reply means stop", async () => {
	assert.deepEqual(parseSupervision({ decision: "stop", finding: "overshoot", because: "b", evidence: "e" }), {
		decision: "stop",
		finding: "overshoot",
		because: "b",
	});
	assert.equal(parseSupervision({ decision: "stop", finding: "overshoot", because: "b", actions: [] }), undefined);
	assert.equal(parseSupervision({ decision: "maybe", finding: "unsure" }), undefined);
	const before = frameAt({ position_m: [0.5, 0, 0.3], rotation: IDENTITY }, { clock: 10 });
	const after = frameAt({ position_m: [0.5, 0.05, 0.3], rotation: IDENTITY }, { clock: 14 });
	const base = {
		subgoal: "s",
		deltas: [toDelta({ index: 0, type: "move", direction: "left", distance_mm: 50, frame: "base", note: "" }, CAPS)],
		expect: "left 50",
		steps: [] as SegmentReport[],
		before,
		after,
		legend: "",
	};
	const m = scripted({
		supervise: [{ decision: "stop", finding: "wrong_direction", because: "it went right", evidence: "scene" }],
	});
	const r = await supervise({ ...base, ask: m.ask, judgedUpTo: 9 });
	assert.equal(r?.supervision.finding, "wrong_direction");
	assert.equal(r?.judged, 14);
	// Both sets of pictures: before and after, three each.
	assert.equal(m.calls[0].images, 6);
	assert.match(m.calls[0].text, /move 50 mm/);
	assert.equal(await supervise({ ...base, ask: m.ask, judgedUpTo: 14 }), undefined);
	const none = scripted({ supervise: [new Error("down")] });
	assert.equal((await supervise({ ...base, ask: none.ask, judgedUpTo: 0 }))?.supervision.decision, "continue");
	const invalid = scripted({ supervise: [{ decision: "go", finding: "fine" }] });
	assert.equal((await supervise({ ...base, ask: invalid.ask, judgedUpTo: 0 }))?.supervision.decision, "stop");
});

// ---------------------------------------------------------------------------
// the generator protocol with a fake robot (C19: the observe call points)

/** A robot that answers `mm_observe` and `mm_act` from a pose it keeps; every action lands exactly. */
function fakeRobot(
	o: {
		holdingOnClose?: boolean;
		onAct?: (call: MmCall) => void;
		imagesAfterAct?: boolean;
		blockDescent?: boolean;
	} = {},
) {
	const tool = toBase(RIG.eef_pos_world);
	const R: Mat3 = RIG.grip_xmat_world;
	const state = {
		pose: { position_m: tool, rotation: R } as Pose,
		clock: 1.65,
		holding: null as boolean | null,
		opening: 79.99,
		commanded: "open" as "open" | "close" | "hold",
	};
	const frame = (cameras = true) =>
		frameAt(state.pose, {
			clearance: { seen: true, mm: (state.pose.position_m[2] - TABLE_Z) * 1000 - 2, jaws_down: true },
			cameras,
			clock: state.clock,
			holding: state.holding,
			opening: state.opening,
			commanded: state.commanded,
		});
	const calls: MmCall[] = [];
	const reply = (call: MmCall): MmReply => {
		calls.push(call);
		if (call.tool === "mm_observe") return { ok: true, body: {}, frame: frame(), episode_over: false };
		o.onAct?.(call);
		const a = call.args;
		const segs: SegmentReport[] = [];
		if (a.type === "move") {
			const axis = a.axis as Vec3;
			const d = (a.distance_mm as number) / 1000;
			const n = Math.max(1, Math.ceil(d / 0.05 - 1e-9));
			// A descent is blocked halfway when asked (M1 C6: limited / blocked, 2 hold ticks).
			const blocked = o.blockDescent === true && axis[2] < -0.5;
			for (let i = 1; i <= n; i++) {
				const part = blocked ? d / n / 2 : d / n;
				state.pose = {
					...state.pose,
					position_m: [
						state.pose.position_m[0] + axis[0] * part,
						state.pose.position_m[1] + axis[1] * part,
						state.pose.position_m[2] + axis[2] * part,
					],
				};
				state.clock += 4;
				segs.push({
					label: `${call.label} [${i}/${n}]`,
					kind: "translate",
					outcome: blocked ? "limited" : "done",
					...(blocked ? { abort_reason: "blocked" } : {}),
					moved_mm: part * 1000,
					along_axis_mm: part * 1000,
					lateral_mm: 0.5,
					turned_deg: 0,
					axis,
					ticks: blocked ? 34 : 80,
					elapsed_s: 2.1,
					frame_clock_s: state.clock,
					pose_after: {
						position_mm: state.pose.position_m.map((v) => v * 1000) as Vec3,
						rotation: state.pose.rotation,
					},
					grasp: { holding: state.holding, reason: "", opening_mm: state.opening },
				});
				if (blocked) break;
			}
		} else if (a.type === "gripper") {
			state.clock += 1.6;
			state.commanded = a.state === "close" ? "close" : a.open_to_mm !== undefined ? "hold" : "open";
			if (a.state === "close") {
				state.holding = o.holdingOnClose ?? false;
				state.opening = state.holding ? 63 : 1.14;
			} else {
				state.holding = null;
				state.opening = a.open_to_mm ?? 79.99;
			}
			segs.push({
				label: call.label,
				kind: "gripper",
				outcome: "done",
				moved_mm: 0,
				along_axis_mm: 0,
				lateral_mm: 0,
				turned_deg: 0,
				ticks: 32,
				elapsed_s: 1,
				frame_clock_s: state.clock,
				detail: { opening_mm: state.opening },
				grasp: { holding: state.holding, reason: "", opening_mm: state.opening },
				pose_after: {
					position_mm: state.pose.position_m.map((v) => v * 1000) as Vec3,
					rotation: state.pose.rotation,
				},
			});
		} else if (a.type === "rotate") {
			state.clock += 2;
			segs.push({
				label: call.label,
				kind: "rotate",
				outcome: "done",
				moved_mm: 0,
				along_axis_mm: 0,
				lateral_mm: 0,
				turned_deg: a.angle_deg as number,
				ticks: 40,
				elapsed_s: 1,
				frame_clock_s: state.clock,
			});
		}
		return { ok: true, body: {}, frame: frame(o.imagesAfterAct ?? true), segments: segs, episode_over: false };
	};
	return { state, reply, calls, frame };
}

/** Drive the generator as the provider would: one call per turn, the cache kept from each reply. */
async function drive(
	gen: Executor,
	robot: ReturnType<typeof fakeRobot>,
	cache: FrameCache,
	between?: (call: MmCall, i: number) => void,
) {
	let next = await gen.next();
	let i = 0;
	while (!next.done) {
		between?.(next.value, i++);
		const reply = robot.reply(next.value);
		if (reply.frame) {
			cache.latest = reply.frame;
			cache.stale = !hasImages(reply.frame);
		} else cache.stale = true;
		next = await gen.next(reply);
	}
	return next.value;
}
const bowlBoxes = () => ({
	scene: perfectBox(scene, "akita_black_bowl_1"),
	side: perfectBox(side, "akita_black_bowl_1"),
});
function input(o: {
	subgoal?: Partial<Subgoal>;
	ask: Ask;
	cache: FrameCache;
	clock?: SubgoalInput["clock"];
	events?: SubgoalInput["onEvent"];
	preface?: string;
	carry?: SubgoalInput["carry"];
}): SubgoalInput {
	return {
		task: "put the bowl on the plate",
		subgoal: {
			name: "over_bowl",
			subgoal: "move the jaws over the akita black bowl",
			criterion: "the tool is centred over the bowl",
			target: "akita black bowl",
			target_boxes: bowlBoxes(),
			clause: 0,
			max_cycles: 10,
			gripper_only: false,
			...o.subgoal,
		},
		preface: o.preface ?? "",
		...(o.carry ? { carry: o.carry } : {}),
		maxCycles: 10,
		ask: o.ask,
		clock: o.clock ?? missionClock(),
		cache: o.cache,
		onEvent: o.events ?? (() => {}),
	};
}

test("executor: C19, one robot call per yield, the observe call points of spec-executor §6.2, spent after a run batch", async () => {
	const moves = {
		assessment: "the bowl is left and ahead",
		done: false,
		command: {
			actions: [
				{ type: "move", direction: "left", distance_mm: 30 },
				{ type: "move", direction: "down", distance_mm: 30 },
			],
		},
		expect: "30 mm left then 30 mm down",
		confidence: 0.8,
	};
	const done = { assessment: "over it", done: true, command: null, expect: "", confidence: 0.9 };
	const m = scripted({ executor: [moves, done] });
	const robot = fakeRobot();
	const cache: FrameCache = { stale: true };
	const events: string[] = [];
	const run = await drive(runSubgoal(input({ ask: m.ask, cache, events: (e) => events.push(e.type) })), robot, cache);
	assert.equal(run.ending, "MODEL_DONE");
	// The calls: one look at the first cycle's start (nothing cached), then the two actions; the look
	// before the second action and the second cycle's start reuse the frame the action returned.
	assert.deepEqual(
		robot.calls.map((c) => c.tool),
		["mm_observe", "mm_act", "mm_act"],
	);
	assert.deepEqual(run.observes, [
		{ why: "cycle_start", yielded: true },
		{ why: "before_action", yielded: false },
		{ why: "cycle_start", yielded: false },
	]);
	assert.equal(run.cycles.length, 2);
	assert.equal(run.cycles[0].steps.length, 2);
	assert.equal(run.cycles[0].batch?.ran, 2);
	// The first look came from the planner's boxes without a model call; the executor was asked twice.
	assert.equal(run.cycles[0].reach?.from_plan_boxes, true);
	assert.ok(run.cycles[0].reach?.located?.point_mm);
	assert.deepEqual(
		m.calls.map((c) => c.role),
		["executor", "executor"],
	);
	// The labels the mission's net-motion rule reads, and the axis form the robot ran.
	assert.equal((robot.calls[1] as { label: string }).label, "move left 30.0 mm (base)");
	assert.deepEqual((robot.calls[1] as { args: Json }).args.axis, [0, 1, 0]);
	assert.equal(labelDirection((robot.calls[2] as { label: string }).label)?.word, "down");
	// The prompt carried the target block with the measured offset and the robot report.
	assert.match(m.calls[0].text, /WHERE THE TARGET IS: "akita black bowl" is \d+ mm (FORWARD|BACKWARD|LEFT|RIGHT)/);
	assert.match(m.calls[0].text, /Tool point at \[/);
	assert.match(m.calls[0].text, /PICTURES: scene \(now\), side \(now\), wrist \(now\)/);
	assert.equal(m.calls[0].images, 3);
	assert.ok(
		events.includes("proposed") &&
			events.includes("step_end") &&
			events.includes("reached") &&
			events.includes("timing"),
	);
	assert.equal(run.timing.calls.length, 2);
	assert.ok(run.timing.motion_s > 4);
	// Carry: the located target and its measurements go to the next step of the same target.
	assert.equal((run.carry as Json).target, "akita black bowl");
	assert.ok((run.carry as Json).same.located);
});

test("executor: a cancel is read before every mm_act; a cancelled batch is not yielded and does not count", async () => {
	const moves = {
		assessment: "",
		done: false,
		command: {
			actions: [
				{ type: "move", direction: "left", distance_mm: 30 },
				{ type: "move", direction: "forward", distance_mm: 30 },
			],
		},
		expect: "",
		confidence: 0.8,
	};
	// The monitor stops the attempt while the first action runs: the second is never sent.
	const m = scripted({ executor: [moves] });
	const clock = missionClock();
	const robot = fakeRobot({ onAct: () => clock.cancel({ by: "monitor", reason: "wrong object" }) });
	const cache: FrameCache = { stale: true };
	const run = await drive(runSubgoal(input({ ask: m.ask, cache, clock })), robot, cache);
	assert.equal(run.ending, "CANCELLED");
	assert.deepEqual(
		robot.calls.map((c) => c.tool),
		["mm_observe", "mm_act"],
	);
	assert.equal(run.cycles[0].batch?.ran, 1);
	assert.equal(run.cycles[0].steps.length, 1);
	// Cancelled before the proposal: no model call, no action, and the batch does not count toward spent.
	const m2 = scripted({ executor: [moves] });
	const clock2 = missionClock();
	clock2.cancel({ by: "deadline", reason: "the mission clock ran out" });
	const robot2 = fakeRobot();
	const cache2: FrameCache = { stale: true };
	const run2 = await drive(runSubgoal(input({ ask: m2.ask, cache: cache2, clock: clock2 })), robot2, cache2);
	assert.equal(run2.ending, "CANCELLED");
	assert.equal(run2.ending_text, "the mission clock ran out");
	assert.deepEqual(robot2.calls, []);
	assert.equal(m2.calls.length, 0);
	assert.equal(run2.cycles.length, 0);
});

test("executor: a close after a blocked descent that reports holding ends the subgoal HELD in code; an empty close is refused until the tool recovers", async () => {
	// Holding wanted and the top unmeasured: at proposal time the close-depth gate arranges a re-look and
	// passes; the descent is blocked (M1 C6) so the close is allowed at execution; the re-look asks the wrist.
	const descendAndClose = {
		assessment: "residual 2 mm, clearance 20 mm",
		done: false,
		command: {
			actions: [
				{ type: "move", direction: "down", distance_mm: 10 },
				{ type: "gripper", state: "close" },
			],
		},
		expect: "the jaws hold the bowl",
		confidence: 0.9,
	};
	const m = scripted({
		executor: [descendAndClose],
		locate: [
			{ objects: [], target: "" },
			{ objects: [], target: "" },
			{ objects: [], target: "" },
		],
	});
	const robot = fakeRobot({ holdingOnClose: true, blockDescent: true });
	// The tool low over the bowl's recorded position.
	robot.state.pose = {
		position_m: [objectsBase.akita_black_bowl_1[0], objectsBase.akita_black_bowl_1[1], TABLE_Z + 0.12],
		rotation: RIG.grip_xmat_world,
	};
	const cache: FrameCache = { stale: true };
	const events: Json[] = [];
	const run = await drive(
		runSubgoal(
			input({
				ask: m.ask,
				cache,
				events: (e) => events.push(e),
				subgoal: { subgoal: "grasp the akita black bowl", criterion: "the bowl is held in the jaws" },
			}),
		),
		robot,
		cache,
	);
	assert.equal(run.ending, "HELD", run.cycles[0].stopped_by ?? run.error);
	assert.equal(run.held?.gap_mm, 63);
	assert.equal(run.cycles[0].proposal?.done, true);
	assert.deepEqual(
		robot.calls.map((c) => c.tool),
		["mm_observe", "mm_act", "mm_act"],
	);
	// The first look read the planner boxes and asked only the wrist (close in, no model call for the fixed
	// views); the wrist was asked again before the descent and at the re-look before the close (within 8 mm
	// of where it was arranged).
	assert.deepEqual(
		m.calls.map((c) => c.role),
		["locate", "executor", "locate", "locate"],
	);
	assert.equal(run.cycles[0].reach?.from_plan_boxes, true);
	assert.equal(run.cycles[0].steps[0].abort_reason, "blocked");
	// The looks after each wrist look and before the close reused cached frames: nothing moved in between.
	assert.deepEqual(
		run.observes.map((o) => o.why),
		["cycle_start", "after_look", "before_action", "after_look"],
	);
	assert.deepEqual(
		run.observes.map((o) => o.yielded),
		[true, false, false, false],
	);
	assert.ok(events.some((e) => e.type === "held" && e.gap_mm === 63));
	assert.equal((run.carry as Json).payload.tracking, true);
	// An empty close: the next close in place is refused by the empty-pickup gate three times, and the error says how to recover.
	const closeIt = { ...descendAndClose, command: { actions: [{ type: "gripper", state: "close" }] } };
	const m2 = scripted({
		executor: [descendAndClose, closeIt, closeIt, closeIt],
		locate: [
			{ objects: [], target: "" },
			{ objects: [], target: "" },
		],
	});
	const robot2 = fakeRobot({ holdingOnClose: false, blockDescent: true });
	robot2.state.pose = {
		position_m: [objectsBase.akita_black_bowl_1[0], objectsBase.akita_black_bowl_1[1], TABLE_Z + 0.12],
		rotation: RIG.grip_xmat_world,
	};
	const cache2: FrameCache = { stale: true };
	const run2 = await drive(
		runSubgoal(
			input({
				ask: m2.ask,
				cache: cache2,
				subgoal: { subgoal: "grasp the akita black bowl", criterion: "the bowl is held in the jaws" },
			}),
		),
		robot2,
		cache2,
	);
	assert.equal(run2.ending, "ERROR");
	assert.equal(run2.cycles.length, 2);
	assert.equal(run2.cycles[0].steps[1].grasp?.holding, false);
	assert.match(run2.cycles[1].error ?? "", /closed on nothing/);
	assert.equal((run2.carry as Json).same.emptyCloses.length, 1);
	assert.match(m2.calls.at(-1)?.text ?? "", /The previous answer was refused: .*closed on nothing/s);
});

test("executor: an action whose reply has no pictures makes the next look a real mm_observe; episode over ends the run", async () => {
	const moves = {
		assessment: "",
		done: false,
		command: {
			actions: [
				{ type: "move", direction: "left", distance_mm: 30 },
				{ type: "move", direction: "forward", distance_mm: 30 },
			],
		},
		expect: "",
		confidence: 0.8,
	};
	const done = { assessment: "", done: true, command: null, expect: "", confidence: 1 };
	const m = scripted({ executor: [moves, done] });
	const robot = fakeRobot({ imagesAfterAct: false });
	const cache: FrameCache = { stale: true };
	const run = await drive(runSubgoal(input({ ask: m.ask, cache })), robot, cache);
	assert.equal(run.ending, "MODEL_DONE");
	assert.deepEqual(
		robot.calls.map((c) => c.tool),
		["mm_observe", "mm_act", "mm_observe", "mm_act", "mm_observe"],
	);
	assert.deepEqual(run.observes, [
		{ why: "cycle_start", yielded: true },
		{ why: "before_action", yielded: true },
		{ why: "cycle_start", yielded: true },
	]);
	// The episode ends under the action: EPISODE_OVER, with the steps kept.
	const m2 = scripted({ executor: [moves] });
	const robot2 = fakeRobot();
	const over = robot2.reply;
	robot2.reply = (call) => {
		const r = over(call);
		return call.tool === "mm_act" ? { ...r, episode_over: true } : r;
	};
	const cache2: FrameCache = { stale: true };
	const run2 = await drive(runSubgoal(input({ ask: m2.ask, cache: cache2 })), robot2, cache2);
	assert.equal(run2.ending, "EPISODE_OVER");
	assert.equal(run2.episode_over, true);
	assert.equal(run2.cycles[0].steps.length, 1);
	// A refused action (a CommandError at mm_act) stops the batch as a rejected proposal; the next cycle's history says so.
	const m3 = scripted({ executor: [moves, done] });
	const robot3 = fakeRobot();
	const base3 = robot3.reply;
	let refusedOnce = false;
	robot3.reply = (call) => {
		if (call.tool === "mm_act" && !refusedOnce) {
			refusedOnce = true;
			return {
				ok: false,
				error: "mm_act refused, nothing moved: action 0: this robot has no home",
				episode_over: false,
			};
		}
		return base3(call);
	};
	const cache3: FrameCache = { stale: true };
	const run3 = await drive(runSubgoal(input({ ask: m3.ask, cache: cache3 })), robot3, cache3);
	assert.equal(run3.ending, "MODEL_DONE");
	assert.equal(run3.cycles[0].batch?.ran, 0);
	assert.match(run3.cycles[0].stopped_by ?? "", /refused/);
	assert.match(m3.calls[1].text, /out of reach: do not propose that direction again/);
});

// ---------------------------------------------------------------------------
// the step 4 port defects (docs/motormind/step4-diagnosis.md), on arm B's recorded traces
// (fixtures/motormind/step4-traces.json: bjb2 .local/runs/mm-step4/base-B, M2 @92ce66a58)

const S4 = read("step4-traces.json");
const S4_CARRY = (frozen: number) => ({
	payload: { edge: false, bumped: false, tracking: false },
	target: "akita black bowl",
	same: { unshoved: true, faces: { faces: [] }, emptyCloses: [], frozen },
});

test("executor: step 4 defect 1 — an inherited frozen count freezes only a bare claim, never a cycle that moved; the carry passes this subgoal's count alone", async () => {
	// spatial t0 s0, pick_akita_bowl attempt 2 cycle 0: the cycle moved 7.2 mm (forward 10 → 7.2 along), yet B
	// ended it "frozen: " (no reason) on the two frozen cycles inherited from attempt 1; three later attempts on
	// the same words died the same way after moving 112, 143 and 190 mm.
	const c = S4.spatial_t0_s0.a2_c0_forward_10;
	assert.equal(c.stopped_by, "frozen: ");
	assert.ok(c.segments.reduce((s: number, st: SegmentReport) => s + st.moved_mm, 0) >= 7);
	const far = S4.spatial_t0_s0.top_a1_c0_moved_190;
	assert.equal(far.stopped_by, "frozen: ");
	assert.ok(far.segments.reduce((s: number, st: SegmentReport) => s + st.moved_mm, 0) >= 190);
	const forward = {
		assessment: "",
		done: false,
		command: { actions: [{ type: "move", direction: "forward", distance_mm: 10 }] },
		expect: "",
		confidence: 0.8,
	};
	const lastAndDone = {
		assessment: "",
		done: true,
		command: { actions: [{ type: "move", direction: "left", distance_mm: 20 }] },
		expect: "",
		confidence: 0.8,
	};
	const m = scripted({ executor: [forward, lastAndDone] });
	const robot = fakeRobot();
	const cache: FrameCache = { stale: true };
	const run = await drive(runSubgoal(input({ ask: m.ask, cache, carry: S4_CARRY(2) as never })), robot, cache);
	assert.equal(run.ending, "DONE_AFTER_LAST", run.ending_text);
	assert.equal(run.cycles.length, 2);
	assert.equal(run.cycles[0].stopped_by, undefined);
	// The carry passes this subgoal's own count (0 after two cycles that moved), not the sum.
	assert.equal((run.carry as Json).same.frozen, 0);
	// A bare claim after no motion still reads the inherited count (spec-executor §7.2 step 11).
	const claim = { assessment: "", done: true, command: null, expect: "", confidence: 0.9 };
	const m2 = scripted({ executor: [claim] });
	const robot2 = fakeRobot();
	const cache2: FrameCache = { stale: true };
	const run2 = await drive(
		runSubgoal(input({ ask: m2.ask, cache: cache2, carry: S4_CARRY(1) as never })),
		robot2,
		cache2,
	);
	assert.equal(run2.ending, "FROZEN");
	assert.match(run2.cycles[0].stopped_by ?? "", /before this claim/);
});

/** A recorded delta of the fixture as the executor's `Delta` (its magnitude read from the label when recorded). */
const s4Delta = (d: Json, i: number): Delta => ({
	index: i,
	kind: d.kind,
	...(d.axis ? { axis: d.axis as Vec3 } : {}),
	magnitude: (d.magnitude_mm ?? 0) / 1000,
	speed: 0.05,
	stop_on_contact: false,
	push: d.push === true,
	...(d.gripper_state ? { gripper_state: d.gripper_state } : {}),
	label: d.label,
	note: "",
});
/** The fixture cycle's deltas as the executor held them: the model's and the menders' inserts, not a re-look's follow move. */
const s4Deltas = (c: Json): Delta[] => (c.deltas as Json[]).filter((d) => !d.follow).map(s4Delta);
/** The frame the fixture recorded before the batch, on the test rig's pose. */
const s4Frame = (c: Json): Frame =>
	frameAt(
		{ position_m: c.frame_before.pose.position_m, rotation: c.frame_before.pose.rotation },
		{
			holding: c.frame_before.grasp.holding,
			opening: c.frame_before.grasp.opening_mm,
			commanded: c.frame_before.gripper.commanded,
		},
	);

test("executor: step 4 defect 2 — the jaws count as changed by the close this cycle COMMANDED, not by whether the close ran", () => {
	// spatial t0 s0, pick_akita_bowl attempt 1 cycle 2: left 4, down 41 (blocked 44 mm short), close — the
	// close-depth gate stopped the batch before the close. B: no gripper segment → unchanged → frozen=2 →
	// FROZEN. The original: a close was commanded while the jaws stood open → changed → frozen=0. Arm A's
	// take step on this very cell survived its cycle 1 by that rule.
	const c = S4.spatial_t0_s0.a1_c2_close_gated;
	assert.equal(c.ending, "FROZEN");
	const deltas = s4Deltas(c);
	assert.deepEqual(
		deltas.map((d) => d.kind),
		["translate", "translate", "gripper"],
	);
	const steps: SegmentReport[] = c.segments;
	assert.ok(steps.some((st) => st.kind === "translate" && st.abort_reason === "blocked"));
	assert.ok(!steps.some((st) => st.kind === "gripper"), "the close never ran");
	const before = s4Frame(c);
	assert.equal(before.grasp.holding, null);
	assert.ok(before.grasp.opening_mm! > 70);
	assert.equal(jawsShut(before), false);
	assert.equal(jawsChanged(deltas, before), true);
	assert.equal(noMotion(steps, deltas, true).frozen, false);
	// The rule B had: by the segments alone the cycle is "a move stopped early".
	assert.equal(noMotion(steps, deltas, false).frozen, true);
	// A close commanded on jaws already shut (holding) changes nothing; an open does; no gripper action, nothing.
	const shut = frameAt(before.pose, { holding: true, opening: 42, commanded: "close" });
	assert.equal(jawsShut(shut), true);
	assert.equal(jawsChanged(deltas, shut), false);
	assert.equal(
		jawsChanged(
			deltas.map((d) => (d.kind === "gripper" ? { ...d, gripper_state: "open" as const } : d)),
			shut,
		),
		true,
	);
	assert.equal(jawsChanged(deltas.slice(0, 2), before), false);
	// No reading at all: nobody can say the jaws did not change.
	const unreadable: Frame = {
		...before,
		gripper: { ...before.gripper, opening_mm: Number.NaN },
		grasp: { holding: null, reason: "" },
	};
	assert.equal(jawsShut(unreadable), null);
	assert.equal(jawsChanged(deltas, unreadable), true);
	// Shut by the closed fraction alone (no holding reading): a quarter open.
	assert.equal(jawsShut({ ...before, grasp: { holding: null, reason: "", closed_fraction: 0.25 } }), true);
});

test("executor: step 4 defect 4 — stopped short is judged over the whole batch (asked vs along), not per segment from the label", () => {
	// object t1 s0, lift_cream_cheese cycles 1 and 2: `move up 100` split by the wall-grasp mender into "the
	// first 17 mm" and "the rest" (two legs). B read 100 mm off the first leg's label, saw 14 of 100 and froze
	// twice although each cycle rose 92 mm; the verifier's done saved the cell. The batch made 92 of 100.
	for (const key of ["lift_c1", "lift_c2"]) {
		const c = S4.object_t1_s0[key];
		const deltas = s4Deltas(c);
		assert.deepEqual(
			deltas.map((d) => Math.round(d.magnitude * 1000)),
			[17, 83],
		);
		const steps: SegmentReport[] = c.segments;
		assert.ok(steps.every((st) => st.outcome === "done" && !st.abort_reason));
		const got = steps.reduce((s, st) => s + Math.abs(st.along_axis_mm), 0);
		assert.ok(got > 90 && got < 100, `${got}`);
		assert.deepEqual(noMotion(steps, deltas, false), { frozen: false, why: "" });
	}
	assert.equal(S4.object_t1_s0.lift_c2.stopped_by, "frozen: a move stopped early");
	// A genuinely blocked batch is still frozen: spatial t0 s0 attempt 1 cycle 1, down 236 blocked on the 5th
	// segment (arm A froze its matching cycle too); so is one that made under a quarter of what it asked for.
	const blocked = S4.spatial_t0_s0.a1_c1_down_blocked;
	const r = noMotion(blocked.segments, s4Deltas(blocked), false);
	assert.equal(r.frozen, true);
	assert.match(r.why, /^a move stopped early: a segment was blocked/);
	const asked200 = [
		s4Delta({ kind: "translate", label: "move left 200.0 mm (base)", axis: [0, 1, 0], magnitude_mm: 200 }, 0),
	];
	const short: SegmentReport[] = [
		{ ...blocked.segments[0], outcome: "done", abort_reason: undefined, moved_mm: 30, along_axis_mm: 30 },
	];
	const r2 = noMotion(short, asked200, false);
	assert.equal(r2.frozen, true);
	assert.match(r2.why, /^a move stopped early: the batch made 30\.0 of the 200\.0 mm/);
	assert.equal(noMotion([{ ...short[0], moved_mm: 60, along_axis_mm: 60 }], asked200, false).frozen, false);
	// Moved under 1 mm in all: frozen whatever was asked; a turn of half a degree is motion.
	assert.match(
		noMotion([{ ...short[0], moved_mm: 0.4, along_axis_mm: 0.4 }], asked200, false).why,
		/moved 0\.4 mm in all/,
	);
	assert.equal(
		noMotion([{ ...short[0], moved_mm: 0.4, along_axis_mm: 0.4, turned_deg: 0.6 }], asked200, false).frozen,
		false,
	);
	// The 190 mm cycle of defect 1 asked 72 (inserted lift) + 162 + 105 and made it all: not frozen by any rule.
	const far = S4.spatial_t0_s0.top_a1_c0_moved_190;
	assert.equal(noMotion(far.segments, s4Deltas(far), false).frozen, false);
});

test("executor: step 4 defect 6 — a one-view answer with a wide bar is doubtful: the kept point beats it, the proposer is told, the executor looks again after moving", async () => {
	// spatial t0 s0, pick_akita_bowl_retry attempt 1 cycle 0: the point carried from the attempt before
	// ([640, 202, 23] ± 40) gave way to a one-view answer on the OTHER black bowl ([472, 306, 6] ± 204,
	// doubtful=false) and the tool drove 115 mm toward it; grasp_akita_bowl_top attempt 1 did it again.
	const w = S4.spatial_t0_s0.retry_a1_c0_wrong_bowl;
	assert.ok(w.located.error_mm > 190 && w.located.doubtful === false);
	assert.ok(Math.hypot(w.located.point_mm[0] - 472, w.located.point_mm[1] - 306) < 2);
	// 1. locate: one fixed view alone, the other seeing nothing, gives a bar as wide as the object is deep. On this
	// rig the 200 mm plate from the scene camera is 55 mm (the bowls 41-49; the recorded case, from B's cameras
	// and heights, was 204): over MAX_SKEW_MM the answer is doubtful.
	const otherBowl = {
		objects: [{ label: "akita black bowl", bbox_2d: perfectBox(scene, "akita_black_bowl_2").map(Math.round) }],
		target: "akita black bowl",
	};
	const plate = {
		objects: [{ label: "plate", bbox_2d: perfectBox(scene, "plate_1").map(Math.round) }],
		target: "plate",
	};
	const nothing = { objects: [], target: "" };
	const tag = (c: Camera): Camera => ({ ...c, raw512: c.name, marked512: c.name });
	const l = await locate({
		views: { scene: tag(scene), side: tag(side) },
		words: "plate",
		table_z_m: TABLE_Z,
		tool_z_m: TABLE_Z + 0.3,
		tallest_mm: 160,
		holding: false,
		ask: async (_r, _p, images) => ok(images[0].data === "scene" ? plate : nothing),
	});
	assert.ok(l.ok, l.error);
	assert.deepEqual(l.cameras, ["scene"]);
	assert.ok((l.error_mm as number) > 50, `${l.error_mm}`);
	assert.equal(l.doubtful, true);
	assert.match(l.method, /mm bar from one view: doubtful/);
	// 2. The executor keeps the point it carried for the same words when a wider-barred answer comes (the
	// sticky rule keyed on the words, as the original); first, with nothing shoved, it does not even look.
	const truth = objectsBase.akita_black_bowl_1;
	const kept: Vec3 = [truth[0], truth[1], TABLE_Z + 0.02];
	const located = (unshoved: boolean) => ({
		...S4_CARRY(0),
		same: {
			...S4_CARRY(0).same,
			unshoved,
			located: {
				words: "akita black bowl",
				point: kept,
				method: "scene and side met 12 mm apart at the contact edge",
				error_mm: 39.6,
				error_class: "how far apart the two views' rays passed",
				on_face: false,
				cameras: ["scene", "side"],
			},
		},
	});
	const claim = { assessment: "", done: true, command: null, expect: "", confidence: 0.9 };
	const m0 = scripted({ executor: [claim] });
	const robot0 = fakeRobot();
	const cache0: FrameCache = { stale: true };
	const run0 = await drive(
		runSubgoal(input({ ask: m0.ask, cache: cache0, subgoal: { target_boxes: {} }, carry: located(true) as never })),
		robot0,
		cache0,
	);
	assert.equal(run0.ending, "MODEL_DONE");
	assert.deepEqual(
		m0.calls.map((c) => c.role),
		["executor"],
		"the carried point is still good: no look",
	);
	assert.match(m0.calls[0].text, /WHERE THE TARGET IS: "akita black bowl"/);
	assert.equal(run0.cycles[0].reach, undefined);
	// Shoved since the last look: the executor looks, gets the other bowl on one view, and keeps its point.
	const m = scripted({ executor: [claim], locate: [otherBowl, nothing] });
	const robot = fakeRobot();
	const cache: FrameCache = { stale: true };
	const run = await drive(
		runSubgoal(input({ ask: m.ask, cache, subgoal: { target_boxes: {} }, carry: located(false) as never })),
		robot,
		cache,
	);
	assert.deepEqual(
		m.calls.map((c) => c.role),
		["locate", "locate", "executor"],
	);
	const reach = run.cycles[0].reach!;
	const got = reach.located!.point_mm!;
	assert.ok(Math.hypot(got[0] / 1000 - kept[0], got[1] / 1000 - kept[1]) * 1000 < 1, JSON.stringify(reach));
	assert.equal(reach.located!.doubtful, false);
	assert.match(m.calls[2].text, /with a wider bar \(\d+ vs 40 mm\) was not taken/);
	const carriedOut = (run.carry as Json).same.located;
	assert.ok(Math.hypot(carriedOut.point[0] - kept[0], carriedOut.point[1] - kept[1]) * 1000 < 1);
	// 3. With no kept point the doubtful answer stands, the proposer is told, and after the tool moves further
	// than a close survives the executor looks again instead of driving on the same answer.
	const forward = {
		assessment: "",
		done: false,
		command: { actions: [{ type: "move", direction: "forward", distance_mm: 60 }] },
		expect: "",
		confidence: 0.8,
	};
	const m1 = scripted({ executor: [forward, claim], locate: [plate, nothing, plate, nothing] });
	const robot1 = fakeRobot();
	const cache1: FrameCache = { stale: true };
	const run1 = await drive(
		runSubgoal(
			input({
				ask: m1.ask,
				cache: cache1,
				subgoal: {
					subgoal: "move the jaws over the plate",
					criterion: "the tool is centred over the plate",
					target: "plate",
					target_boxes: {},
				},
			}),
		),
		robot1,
		cache1,
	);
	assert.equal(run1.ending, "MODEL_DONE");
	assert.deepEqual(
		m1.calls.map((c) => c.role),
		["locate", "locate", "executor", "locate", "locate", "executor"],
	);
	assert.equal(run1.cycles[0].reach?.located?.doubtful, true);
	assert.equal(run1.cycles[1].reach?.located?.doubtful, true);
	assert.equal((run1.carry as Json).same.located.doubtful, true);
});

test("executor: step 4 defect 7 — a thing wider than the jaws is aimed at its wall: the edge point along the jaw line on the tool's side, inside the reach; the target block says the offset is taken off", () => {
	// Both spatial smoke cells after fixes 1-6 died on the take: the target block aimed the tool at the bowl's
	// middle while saying "the offset already aims at its edge", a pad landed on the rim and every descent was
	// blocked 25 mm above the top face. The original shifts the aim by width/2 - (pad/2 + 12 mm) along the jaw line.
	const truth = objectsBase.akita_black_bowl_1;
	const s = newState({
		subgoal:
			"descend the rest of the way until the jaws are level with the rim of the akita black bowl, with it between them, and close",
		criterion: "The robot reports that it is HOLDING the akita black bowl.",
		target: "akita black bowl",
		planBoxes: {},
	});
	const middle: Vec3 = [truth[0], truth[1], TABLE_Z + 0.02];
	s.located = {
		words: "akita black bowl",
		point: middle,
		method: "two views met",
		error_mm: 10,
		error_class: "",
		on_face: false,
		cameras: ["scene", "side"],
	};
	s.footprint = [
		[truth[0] - 0.059, truth[1], TABLE_Z],
		[truth[0] + 0.059, truth[1], TABLE_Z],
	];
	assert.ok(widerThanJaws(s, CAPS));
	const pose: Pose = { position_m: [truth[0] + 0.01, truth[1] + 0.03, TABLE_Z + 0.1], rotation: RIG.grip_xmat_world };
	const line = jawLine(pose);
	const e = edgeOffsetMm(s, CAPS)!;
	close(e.mm, 59 - 17 / 2 - 12, 1e-9, "the offset off the middle");
	assert.match(e.why, /118 mm across and the jaws open 79 mm/);
	const p = edgePoint(s, pose, CAPS)!;
	const side = Math.sign(dot(sub(pose.position_m, middle), line));
	close(dot(sub(p, middle), line) * 1000, side * e.mm, 1e-9, "on the tool's side along the jaw line");
	close(Math.abs(dot(sub(p, middle), [line[1], -line[0], 0])), 0, 1e-9, "nothing across the line");
	// The residual is to the edge point; the middle's own offset is kept for what the pads enclose.
	const r = residualOf(s, pose, CAPS)!;
	const c = centreMiss(s, pose)!;
	close(r, Math.hypot(p[0] - pose.position_m[0], p[1] - pose.position_m[1]) * 1000, 1e-9);
	assert.ok(Math.abs(r - Math.hypot(c[0], c[1]) * 1000) > 10);
	const text = whereTheTargetIs(s, frameAt(pose), CAPS, TABLE_Z + 0.034);
	assert.match(text, /The 3[89] mm the jaws need to sit off its middle, along the line they close along/);
	assert.match(text, /is already taken off those numbers/);
	assert.doesNotMatch(text, /NEITHER RIM/);
	// A named part's own box is the part's edge: no extra pad. A step that leans on the thing aims at the middle.
	s.footprintIsPart = true;
	close(edgeOffsetMm(s, CAPS)!.mm, 59 - 17 / 2, 1e-9);
	s.footprintIsPart = false;
	const push = newState({
		subgoal: "push the akita black bowl along the table with the jaws closed",
		criterion: "the bowl has moved",
		target: "akita black bowl",
		planBoxes: {},
	});
	push.located = s.located;
	push.footprint = s.footprint;
	assert.equal(edgeOffsetMm(push, CAPS), undefined);
	assert.equal(edgePoint(push, pose, CAPS), undefined);
	// The hover step before the take aims at the edge too; a narrower object is aimed at its middle.
	const hover = newState({
		subgoal:
			"open the jaws, move the gripper over the akita black bowl and lower it until it hovers about a palm's height above the bowl, jaws still open",
		criterion: "the bowl sits directly under the gripper, the jaws are open",
		target: "akita black bowl",
		planBoxes: {},
	});
	hover.located = s.located;
	hover.footprint = s.footprint;
	assert.ok(edgePoint(hover, pose, CAPS));
	hover.footprint = [
		[truth[0] - 0.03, truth[1], TABLE_Z],
		[truth[0] + 0.03, truth[1], TABLE_Z],
	];
	assert.equal(edgePoint(hover, pose, CAPS), undefined);
	// Reach: with the jaws closing along x and the tool past the thing, the far rim lies beyond 780 mm; the near one is taken.
	const alongX: Pose = {
		position_m: [0.8, 0, TABLE_Z + 0.1],
		rotation: [
			[0, 1, 0],
			[1, 0, 0],
			[0, 0, -1],
		],
	};
	s.located = { ...s.located!, point: [0.76, 0, TABLE_Z + 0.02] };
	s.footprint = [
		[0.76 - 0.059, 0, TABLE_Z],
		[0.76 + 0.059, 0, TABLE_Z],
	];
	close(edgePoint(s, alongX, CAPS)![0], 0.76 - e.mm / 1000, 1e-9, "the rim nearer the base");
	assert.equal(quarterTurnForReach(s, alongX, CAPS), undefined);
	// Neither rim along the jaw line (here y) inside the reach, the rims along x are: a quarter turn is advised.
	s.located = { ...s.located!, point: [0.79, 0, TABLE_Z + 0.02] };
	s.footprint = [
		[0.79, -0.059, TABLE_Z],
		[0.79, 0.059, TABLE_Z],
	];
	const alongY: Pose = {
		position_m: [0.75, 0.05, TABLE_Z + 0.1],
		rotation: [
			[-1, 0, 0],
			[0, 1, 0],
			[0, 0, -1],
		],
	};
	assert.equal(quarterTurnForReach(s, alongY, CAPS), 'rotate "yaw_left" 90');
	assert.match(
		whereTheTargetIs(s, frameAt(alongY), CAPS, TABLE_Z + 0.034),
		/NEITHER RIM along that line is inside this arm's reach\. rotate "yaw_left" 90/,
	);
});

test("executor: step 4 defect 8 — a close that holds tracks the payload whatever the gap, so the carry seeks the receiver", () => {
	// smoke-fix2 spatial t0 s0: the rim closed at 7.6 mm (under MIN_WIDTH_MM) so the payload was not tracked,
	// the carry step kept aiming at the carried bowl point ("2 mm forward") and never looked for the plate.
	const s = newState({
		subgoal: "descend until the jaws are level with the rim of the akita black bowl and close",
		criterion: "The robot reports that it is HOLDING the akita black bowl.",
		target: "akita black bowl",
		planBoxes: {},
	});
	s.located = {
		words: "akita black bowl",
		point: [0.6, 0.2, 0.0],
		method: "m",
		error_mm: 10,
		error_class: "",
		on_face: false,
		cameras: ["scene"],
	};
	const rim: SegmentReport = {
		label: "gripper close",
		kind: "gripper",
		outcome: "done",
		moved_mm: 0,
		along_axis_mm: 0,
		lateral_mm: 0,
		turned_deg: 0,
		ticks: 32,
		elapsed_s: 1,
		detail: { opening_mm: 7.6 },
		grasp: { holding: true, reason: "", opening_mm: 7.6 },
	};
	// The bowl (138 mm across, measured) is held by its -x rim: the jaws close along x on this rig, the tool stands
	// 52 mm from the middle. Its middle hangs 138/2 - (17/2 + 12) = 48.5 mm from the tool toward +x, not "unknown"
	// (the shoved test reads the thing's half width, 69 mm, not the 7.6 mm gap).
	s.footprint = [
		[0.6 - 0.069, 0.2, 0],
		[0.6 + 0.069, 0.2, 0],
	];
	const pose: Pose = { position_m: [0.6 - 0.052, 0.2, 0.02], rotation: RIG.grip_xmat_world };
	const line = jawLine(pose);
	noteGrasped(s, rim, CAPS, pose, true);
	assert.equal(s.payload.tracking, true);
	assert.equal(s.payload.width_mm, 7.6);
	assert.equal(s.payload.edge, true);
	assert.ok(s.payload.offset, "the middle's offset is known");
	close(
		dot(s.payload.offset as Vec3, line) * 1000 * Math.sign(dot([0.052, 0, 0], line)),
		48.5,
		1e-9,
		"48.5 mm toward the middle",
	);
	close(s.payload.span_mm as number, 138, 1e-9);
	assert.equal(s.located, undefined, "the thing left where it was");
	// The receiver's words come from the criterion of the carry step, now that something is held.
	assert.equal(
		receiverWords(
			"The akita black bowl is over the plate and the robot still reports holding it.",
			"move the gripper sideways until the bowl is over the plate",
			"akita black bowl",
		),
		"plate",
	);
});

// ---------------------------------------------------------------------------
// the step 4 round 2 defects (docs/motormind/step4-diagnosis-2.md), on arm B2's recorded traces
// (fixtures/motormind/step4-diag2.json: bjb2 .local/runs/mm-step4/base-B2, main @ce9928415)

test("executor: round 2 defect 4 — a descent pressing through contact is not a shove: the take step keeps its anchor instead of looking again every cycle and drifting with the tool", () => {
	const D2 = read("step4-diag2.json");
	const t = D2.object_t3_s0_take_drift;
	const AXIS: Record<string, [number, number, number]> = {
		forward: [1, 0, 0],
		backward: [-1, 0, 0],
		left: [0, 1, 0],
		right: [0, -1, 0],
		down: [0, 0, -1],
		up: [0, 0, 1],
	};
	const dir = (label: string) => AXIS[/move (\w+)/.exec(label)?.[1] ?? ""];
	const drift: number[] = [];
	for (const c of t.cycles as Json[]) {
		const deltas = (c.deltas as Json[]).map((d, index) => ({
			index,
			kind: d.kind,
			axis: d.axis,
			magnitude: 0.03,
			speed: 0.05,
			stop_on_contact: true,
			push: d.push,
		})) as Delta[];
		const steps = (c.steps as Json[]).map((st) => ({
			label: st.label,
			kind: "translate",
			outcome: st.outcome,
			...(st.abort_reason ? { abort_reason: st.abort_reason } : {}),
			moved_mm: Math.hypot(st.along_axis_mm as number, st.lateral_mm as number),
			along_axis_mm: st.along_axis_mm,
			lateral_mm: st.lateral_mm,
			turned_deg: st.turned_deg,
			axis: dir(st.label as string),
			ticks: 1,
			elapsed_s: 0.2,
		})) as SegmentReport[];
		// Each cycle: a push=true descent blocked by contact with a lateral drift under max(3 mm, the 10 mm bar).
		assert.ok(
			deltas.some((d) => d.push && d.axis?.[2] === -1),
			`cycle ${c.index}: the descent was a push`,
		);
		assert.ok(steps.some((st) => st.abort_reason === "blocked" && st.axis?.[2] === -1));
		assert.ok(steps.every((st) => st.lateral_mm < 3));
		assert.equal(shoved(steps, deltas, 10), false, `cycle ${c.index}`);
		if (c.located) drift.push(Math.round((c.located as Json).point_mm[0]));
	}
	// What B2 did instead: a look every cycle, the anchor walking away from the truth with the tool.
	assert.deepEqual(drift, [687, 700, 707, 710, 712]);
	assert.ok(drift.every((x, i) => i === 0 || x > drift[i - 1]));
	assert.ok(Math.abs(t.carried_point_mm[0] - t.truth_x_mm) < Math.abs(drift.at(-1)! - t.truth_x_mm));
	// A sideways push, or a descent that skids, still counts.
	const push = {
		index: 0,
		kind: "translate",
		axis: [1, 0, 0],
		magnitude: 0.02,
		speed: 0.05,
		stop_on_contact: true,
		push: true,
	} as Delta;
	assert.equal(shoved([], [push], 10), true);
	const skid = {
		label: "move down 30.0 mm (base) [1/1]",
		kind: "translate",
		outcome: "contact",
		abort_reason: "blocked",
		moved_mm: 12,
		along_axis_mm: 0.9,
		lateral_mm: 12,
		turned_deg: 0.2,
		axis: [0, 0, -1],
		ticks: 1,
		elapsed_s: 0.2,
	} as SegmentReport;
	assert.equal(shoved([skid], [], 10), true);
});

// ---------------------------------------------------------------------------
// the step 4 round 3 defects (docs/motormind/step4-diagnosis-3.md), on arm B3's recorded traces
// (fixtures/motormind/step4-diag3.json: bjb2 .local/runs/mm-step4/base-B3, main @fe7fe9bf0)

const D3 = read("step4-diag3.json");

test("executor: round 3 defect 1 — the part a take step names is a word of the part vocabulary owned by the target; 'the rest of the way' and 'the part of the box' name no part, and locate ignores a part word outside the vocabulary", () => {
	for (const c of D3.part_words.cases as Json[]) {
		assert.equal(partOf(c.subgoal, c.target), c.want ?? undefined, `${c.cell}: "${c.subgoal}"`);
	}
	// The three B3 take sentences all gave "rest" before: a part that does not exist, every look down the part branch.
	assert.equal((D3.part_words.cases as Json[]).filter((c) => c.old === "rest").length, 4);
	// The vocabulary: a side or top part, singular, else nothing.
	assert.equal(aPartWord("handles"), "handle");
	assert.equal(aPartWord("Lid"), "lid");
	assert.equal(aPartWord("rest"), undefined);
	assert.equal(aPartWord("way"), undefined);
	assert.equal(aPartWord("middle"), undefined);
	// A compound head still reads as a part ("the mug handle"); a bare object does not.
	assert.equal(namesAPart("the chocolate pudding box"), undefined);
	assert.equal(namesAPart("top of the dark brown drawer unit"), "top");
});

test("executor: round 3 defect 2 — a descent that met the thing last cycle lets this cycle's close through: the blocked axis is kept across cycles, and the close-depth gate reads it", () => {
	const t = D3.close_after_contact;
	const tableZ = t.table_z_mm / 1000;
	const down: Mat3 = [
		[1, 0, 0],
		[0, -1, 0],
		[0, 0, -1],
	];
	const closeDelta = toDelta({ index: 0, type: "gripper", state: "close", frame: "base", note: "" }, CAPS);
	const ctx = (s: ReturnType<typeof newState>, f: Frame, running: boolean) => ({
		state: s,
		frame: f,
		caps: CAPS,
		read: { clearance_mm: undefined, seen: false } as never,
		running,
		heightMm: undefined,
		topZ: undefined,
		descentBlocked: false,
	});
	for (const pair of t.pairs as Json[]) {
		const s = newState({ subgoal: t.subgoal, criterion: t.criterion, target: t.target, planBoxes: {} });
		assert.equal(s.wantsHolding, true);
		s.located = {
			words: t.target,
			point: (t.located_mm as number[]).map((v) => v / 1000) as Vec3,
			method: "scene and side met 19 mm apart",
			error_mm: t.located_error_mm,
			error_class: "",
			on_face: false,
			cameras: ["scene", "side"],
		};
		// The frame after the contact descent: the tool over the handle, jaws down, no top face measured (a handle has none).
		const f = { ...frameAt({ position_m: [0.711, -0.141, tableZ + 0.195], rotation: down }), table_z_m: tableZ };
		const descent: SegmentReport = {
			...pair.descent,
			kind: "translate",
			moved_mm: pair.descent.along_axis_mm,
			axis: [0, 0, -1],
			ticks: 34,
			elapsed_s: 2,
		};
		// What the old code did: the gate saw no block (the batch was new), arranged a re-look, and refused the close after it.
		const old = newState({ subgoal: t.subgoal, criterion: t.criterion, target: t.target, planBoxes: {} });
		assert.equal(closeDepthGate([closeDelta], ctx(old, f, false)), undefined);
		assert.equal(old.relookArranged, true);
		old.relookArranged = false;
		assert.match(closeDepthGate([closeDelta], ctx(old, f, true)) ?? "", /top face has never been measured/);
		assert.equal(closeDepthGate([closeDelta], ctx(old, f, true)), t.refused);
		// Now: the cycle's end keeps the blocked descent (32 mm then contact: the batch's last move; 0 mm: under 1 mm).
		keepBlockedAxes(s, [descent], false);
		assert.deepEqual(s.blockedAxes, [[0, 0, -1]], pair.cycles);
		assert.equal(stoppedAlongTheJaws(s, f.pose), true);
		assert.equal(closeDepthGate([closeDelta], ctx(s, f, false)), undefined, pair.cycles);
		assert.equal(
			s.relookArranged,
			false,
			`${pair.cycles}: no re-look arranged, so no doubtful wrist point replaces the handle`,
		);
		assert.equal(closeDepthGate([closeDelta], ctx(s, f, true)), undefined, pair.cycles);
		// The kept axis stands one cycle: a cycle whose batch ran no move leaves none for the next.
		keepBlockedAxes(s, [], false);
		assert.deepEqual(s.blockedAxes, []);
		assert.equal(stoppedAlongTheJaws(s, f.pose), false);
	}
	// A sideways block that is not the batch's last move and advanced over 1 mm is not kept; holding keeps every block.
	const s2 = newState({ subgoal: t.subgoal, criterion: t.criterion, target: t.target, planBoxes: {} });
	const right: SegmentReport = {
		label: "move right 151.0 mm (base) [1/4]",
		kind: "translate",
		outcome: "limited",
		abort_reason: "blocked",
		moved_mm: 35,
		along_axis_mm: 35,
		lateral_mm: 1,
		turned_deg: 0,
		axis: [0, -1, 0],
		ticks: 30,
		elapsed_s: 1,
	};
	const downDone: SegmentReport = {
		label: "move down 50.0 mm (base) [1/1]",
		kind: "translate",
		outcome: "done",
		moved_mm: 50,
		along_axis_mm: 50,
		lateral_mm: 0,
		turned_deg: 0,
		axis: [0, 0, -1],
		ticks: 80,
		elapsed_s: 2,
	};
	keepBlockedAxes(s2, [right, downDone], false);
	assert.deepEqual(s2.blockedAxes, []);
	keepBlockedAxes(s2, [right, downDone], true);
	assert.deepEqual(s2.blockedAxes, [[0, -1, 0]]);
	assert.deepEqual(t.refusals_per_suite.B3, [77, 60, 28]);
});

test("executor: round 3 defect 3 — a wrist-only look is an answer under the sticky rule, not an override: the doubtful part read 320 mm off does not replace the carried point", () => {
	const t = D3.wrist_relook_override;
	const mm = (p: number[]): Vec3 => [p[0] / 1000, p[1] / 1000, p[2] / 1000];
	const kept = () => ({
		words: t.target,
		point: mm(t.kept.point_mm),
		method: t.kept.method,
		error_mm: t.kept.error_mm,
		error_class: "",
		on_face: false,
		doubtful: t.kept.doubtful,
		cameras: ["scene", "side"] as CameraName[],
	});
	const old = kept();
	for (const a of t.wrist_answers as Json[]) {
		const r = takesOver(
			old,
			{ point: mm(a.point_mm), error_mm: a.error_mm, doubtful: a.doubtful, method: "wrist alone" },
			false,
		);
		assert.deepEqual(r.point, old.point, a.cycle);
		assert.ok(r.jumped !== undefined && r.jumped > 1, a.cycle);
		assert.match(r.method, /was not taken/);
	}
	// The second answer was 320 mm from the kept point: what the model was told to move.
	const far = t.wrist_answers[1];
	assert.ok(Math.hypot(far.point_mm[0] - t.kept.point_mm[0], far.point_mm[1] - t.kept.point_mm[1]) > 300);
	assert.equal(t.model_move_after, "move left 320.0 mm (base)");
	// A confident wrist answer with a bar no wider is taken (the pre-descent follow still works); a doubtful one with a
	// narrower bar is not, until a second look confirms it; an empty close aimed from the kept point lets a near answer in.
	const near: Vec3 = [0.476, -0.246, -0.021];
	assert.deepEqual(takesOver(kept(), { point: near, error_mm: 8, doubtful: false, method: "w" }, false).point, near);
	const o2 = kept();
	const doubtfulNear = takesOver(o2, { point: near, error_mm: 8, doubtful: true, method: "w" }, false);
	assert.deepEqual(doubtfulNear.point, o2.point);
	assert.match(doubtfulNear.method, /doubtful bar/);
	assert.deepEqual(takesOver(o2, { point: near, error_mm: 8, doubtful: true, method: "w" }, false).point, near);
	const o3 = kept();
	const wideButNear = takesOver(o3, { point: near, error_mm: 40, doubtful: false, method: "w" }, true);
	assert.deepEqual(wideButNear.point, near);
	assert.deepEqual(
		takesOver(undefined, { point: near, error_mm: 40, doubtful: true, method: "w" }, false).point,
		near,
	);
});

test("executor: round 3 defect 4 — a turn step holding the knob seeks no receiver, and a step that names no destination seeks nothing; a carry step keeps its receiver", () => {
	const t = D3.turn_steps;
	for (const c of t.cases as Json[]) {
		assert.equal(isTurnStep(c.subgoal, c.criterion), true, c.cell);
		assert.equal(receiverWords(c.criterion, c.subgoal, c.target), "", c.cell);
		assert.equal(actsOnWhatItNames(c.subgoal, c.criterion, c.target), true, c.cell);
	}
	// What B3 did instead: located "the knob" 669 and 499 mm from the jaws that held it, and drove.
	assert.deepEqual(
		(t.cases as Json[]).filter((c) => c.located_mm).map((c) => c.located_error_mm),
		[28, 459],
	);
	const lift = t.lift_while_holding;
	assert.equal(isTurnStep(lift.subgoal, lift.criterion), false);
	assert.equal(receiverWords(lift.criterion, lift.subgoal, lift.target), "");
	const carry = t.carry_keeps_its_receiver;
	assert.equal(receiverWords(carry.criterion, carry.subgoal, carry.target), carry.want);
	// "in the picture" is a camera, not a destination.
	assert.equal(receiverWords("the knob's pointer aims somewhere new in the picture", "", "the knob"), "");
	assert.equal(receiverWords("the bowl is in the wrist view over the plate", "", "the bowl"), "plate");
});

test("executor: round 3 defect 3 (side take) — the part phrase is asked for, a table point is not a face, the face's height is judged against floor - half gap + pad, and the feed goes into the face", () => {
	const t = D3.side_take;
	const b = t.B3_t0_s0;
	// The words: the part phrase with its modifiers, not "it" (what "approach it from the side" gave).
	assert.equal(sidePartWords(b.subgoal, b.target), "the middle drawer handle of the dark cabinet");
	const a = t.A_t0_s1;
	assert.equal(sidePartWords(a.subgoal, a.target), "the handle of the middle drawer of the cabinet");
	assert.equal(sidePartWords("come at the top drawer from the side and close", "the top drawer"), "the top drawer");
	assert.equal(sidePartWords("approach it from the side", "the cabinet"), "the cabinet");
	const down: Mat3 = [
		[1, 0, 0],
		[0, -1, 0],
		[0, 0, -1],
	];
	const mm = (p: number[]): Vec3 => [p[0] / 1000, p[1] / 1000, p[2] / 1000];
	const frame = (tool: number[], tableZ: number) => ({
		...frameAt({ position_m: mm(tool), rotation: down }),
		table_z_m: tableZ,
	});
	// B3's first cycle: the planner's box read onto the table, not on the face. No CANNOT: the face is not located yet.
	const s = newState({ subgoal: b.subgoal, criterion: b.criterion, target: b.target, planBoxes: b.plan_boxes });
	assert.equal(s.sideStep, true);
	s.located = {
		words: "it",
		point: mm(b.located_mm),
		method: "scene's box on the table",
		error_mm: 20,
		error_class: "",
		on_face: false,
		doubtful: true,
		cameras: ["scene", "side"],
	};
	const f = frame(b.tool_mm, b.table_z_mm / 1000);
	const block = sideTakeBlock(s, f, CAPS);
	assert.equal(block.cannot, undefined, b.old_cannot);
	assert.match(block.text, /not located yet/);
	// Arm A's part point on the face, with the cabinet's middle as the body: the face points LEFT (+y), the jaws move RIGHT.
	const s2 = newState({ subgoal: a.subgoal, criterion: a.criterion, target: a.target, planBoxes: {} });
	s2.located = {
		words: sidePartWords(a.subgoal, a.target),
		point: mm(a.part_point_mm),
		method: "scene and side met on the part",
		error_mm: 15,
		error_class: "",
		on_face: true,
		body: mm(a.body_mm),
		cameras: ["scene", "side"],
	};
	const f2 = frame(a.tool_mm, a.table_z_mm / 1000);
	const side = sideTakeBlock(s2, f2, CAPS);
	assert.equal(side.cannot, undefined);
	assert.match(side.text, /face it toward RIGHT/);
	assert.match(side.text, /Standoff point \[69[34], -42, 123\]/);
	assert.match(side.text, /pitch 90 deg/);
	assert.match(side.text, /move right with stop_on_contact for 124 mm/);
	// Without a body the feed comes from where the tool is (forward of it here).
	s2.located.body = undefined;
	assert.match(sideTakeBlock(s2, f2, CAPS).text, /face it toward FORWARD/);
	// The height threshold: floor - half gap + pad = 77.6 mm, not the 100 mm floor; the standoff never goes below the floor.
	const th = t.threshold_mm;
	assert.equal(Math.round((th.tilted_floor - th.jaw_half_gap + th.pad) * 10) / 10, th.lowest_face);
	s2.located.body = mm(a.body_mm);
	s2.located.point = [a.part_point_mm[0] / 1000, a.part_point_mm[1] / 1000, a.table_z_mm / 1000 + 0.08];
	const low = sideTakeBlock(s2, f2, CAPS);
	assert.equal(low.cannot, undefined, "80 mm over the table is taken");
	assert.match(low.text, /Standoff point \[69[34], -42, 89\]/);
	s2.located.point = [a.part_point_mm[0] / 1000, a.part_point_mm[1] / 1000, a.table_z_mm / 1000 + 0.07];
	assert.match(sideTakeBlock(s2, f2, CAPS).cannot ?? "", /70 mm over the table/);
});
