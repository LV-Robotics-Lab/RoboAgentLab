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
import { apply, IDENTITY, labelDirection, matmul, sub, transpose, unit } from "../src/modes/motormind/frames.ts";
import {
	closeOutcome,
	dragging,
	newState,
	noMotion,
	noteEmptyClose,
	requestedMm,
	shoved,
	wantsHolding,
	wantsRelease,
} from "../src/modes/motormind/gates.ts";
import { boxOnPlane, planeErrorMm, project, toPixels, triangulate } from "../src/modes/motormind/geometry.ts";
import { locate, namesAPart, parseLocateReply } from "../src/modes/motormind/locate.ts";
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
import type { Caps, Mat3, Pose, Vec3 } from "../src/modes/motormind/types.ts";

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
	// A blocked descent that made 15 of 50 mm: the cycle "did not move" (stopped early), unless the gripper changed.
	assert.equal(requestedMm(blocked), 50);
	assert.equal(noMotion([blocked], false).frozen, true);
	assert.equal(noMotion([blocked], true).frozen, false);
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
