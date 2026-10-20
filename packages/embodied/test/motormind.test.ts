/**
 * MotorMind mode (src/modes/motormind): the pure functions (the direction table and frames, the
 * command language's parsing, limits and labels, segmentation and units: DESIGN §8 C4, C5) and the
 * robot side on a fake LIBERO env server (pure mode's tools, the lock, cancellation, the closed-loop
 * gate after a timeout, home, the preset and the robot-side configuration checks). The same checks
 * on the real env server are the GPU end-to-end suite's (test/gpu-e2e.test.ts, libero-mm).
 */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { choose, conflicts, PRESETS, unsupported } from "../src/infra/tiers.ts";
import {
	actionLabel,
	CommandError,
	carryLegLabel,
	commandLimits,
	DROP_LABEL,
	fitToReach,
	followLabel,
	LIFT_LABEL,
	measured,
	parseAction,
	parseCommand,
	pyFixed,
	pyFloat,
	report,
	segmentLabel,
	segmentsFor,
	shortenedLabel,
	TEST_REST,
	testLeadLabel,
	testRestLabel,
	tickLimit,
	toDelta,
	toDeltas,
	turnBetween,
} from "../src/modes/motormind/command.ts";
import {
	angleBetween,
	IDENTITY,
	labelDirection,
	matmul,
	PI_UNIT,
	ROTATION,
	ROTATION_ALIASES,
	rotationAbout,
	rotationAxis,
	sayOffset,
	TRANSLATION,
	toBase,
	translationAxis,
	way,
} from "../src/modes/motormind/frames.ts";
import { type Caps, MOTORMIND_EVENT, type MotorMindHandle, type Pose } from "../src/modes/motormind/index.ts";
import { CLOSED_LOOP_ENTRY } from "../src/planner/closed-loop.ts";
import { RESULT_ENTRY } from "../src/robot.ts";
import libero from "../src/robots/libero/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { deployFlags } from "./helpers/deployment.ts";
import { type Call, f32, fakeEnv, nd } from "./sim-stub.ts";

/** LIBERO's capability table as env_server.py sends it (motormind_rig.libero_caps, the original's values). */
const CAPS: Caps = {
	max_translation_m: 0.05,
	max_rotation_rad: 0.5,
	max_speed_m_s: 1.0,
	max_rotation_speed_rad_s: 10,
	default_speed_m_s: 0.05,
	default_rotation_speed_rad_s: Math.PI / 3,
	control_point_offset_m: 0,
	table_z_m: 0,
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
const AT: Pose = { position_m: [0.45, 0, 0.3], rotation: IDENTITY };
const close = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} != ${b} (tol ${tol})`);
const closeVec = (a: readonly number[], b: readonly number[], tol = 1e-9) => {
	for (const [i, v] of a.entries()) close(v, b[i], tol);
};

// ---------------------------------------------------------------------------
// frames (DESIGN §2.1-2.3)

test("the direction table is robot-centric: left is base +Y, which pi's LIBERO units call MV_RIGHT", () => {
	assert.deepEqual(TRANSLATION, {
		forward: [1, 0, 0],
		backward: [-1, 0, 0],
		left: [0, 1, 0],
		right: [0, -1, 0],
		up: [0, 0, 1],
		down: [0, 0, -1],
	});
	assert.equal(PI_UNIT.left, "MV_RIGHT");
	assert.equal(PI_UNIT.right, "MV_LEFT");
	assert.deepEqual(translationAxis("  LEFT "), [0, 1, 0]);
	assert.throws(() => translationAxis("sideways"), /directions are forward, backward, left, right, up, down/);
	assert.deepEqual(ROTATION.yaw_left, [0, 0, 1]);
	assert.deepEqual(ROTATION.pitch_up, [0, -1, 0]);
	assert.deepEqual(ROTATION.roll_ccw, [1, 0, 0]);
	// clockwise is yaw_right (-Z seen from above): the opposite of Show-Harness's ROTATE_CW (+Z).
	for (const w of ["clockwise", "cw"]) assert.deepEqual(rotationAxis(w), [0, 0, -1]);
	for (const w of ["anticlockwise", "counterclockwise", "counter_clockwise", "ccw", "Yaw_Left"])
		assert.deepEqual(rotationAxis(w), [0, 0, 1]);
	assert.equal(Object.keys(ROTATION_ALIASES).length, 6);
	assert.throws(() => rotationAxis("twist"), /aliases: clockwise, cw, anticlockwise/);
});

test("the label rule reads the first direction word, with MotorMind's quirks kept", () => {
	assert.deepEqual(labelDirection("move left 50.0 mm (base)"), { word: "left", axis: [0, 1, 0], kind: "translate" });
	assert.deepEqual(labelDirection("move left 50.0 mm (base) [2/3]")?.axis, [0, 1, 0]);
	// An axis move (the executor's follow correction too) names no direction: skipped.
	assert.equal(labelDirection(followLabel([0.012, -0.004], 12.6)), undefined);
	assert.equal(labelDirection("move axis [0.0, 1.0, 0.0] 20.0 mm (base)"), undefined);
	// The inserted lift names none; the inserted drop reads as a move back (-X), not down.
	assert.equal(labelDirection(LIFT_LABEL), undefined);
	assert.deepEqual(labelDirection(DROP_LABEL), { word: "back", axis: [-1, 0, 0], kind: "translate" });
	// Only yaw counts for turns; counterclockwise is one word, not "clockwise".
	assert.deepEqual(labelDirection("rotate counterclockwise 30.0 deg (base)")?.axis, [0, 0, 1]);
	assert.deepEqual(labelDirection("rotate cw 30.0 deg (base)")?.axis, [0, 0, -1]);
	assert.equal(labelDirection("rotate pitch_up 15.0 deg (base)"), undefined);
	assert.equal(labelDirection("gripper close"), undefined);
});

test("offsets go back to the same words; frames convert world poses to the base frame", () => {
	assert.equal(sayOffset([0.035, 0.012, 0.0004]), "35 mm FORWARD and 12 mm LEFT");
	assert.equal(sayOffset([-0.02, -0.0306, -0.05]), "20 mm BACKWARD and 31 mm RIGHT and 50 mm DOWN");
	assert.equal(sayOffset([0, 0, 0.0005]), "less than 1 mm");
	assert.equal(way([0.1, -0.5, 0.9]), "right");
	assert.equal(way([0.4, 0.1, -1]), "forward");
	// A base turned 90 deg about z at (1, 2, 0): world +y is base -x.
	const base: Pose = { position_m: [1, 2, 0], rotation: rotationAbout([0, 0, 1], Math.PI / 2) };
	const p = toBase(base, { position_m: [1, 3, 0.5], rotation: base.rotation });
	closeVec(p.position_m, [1, 0, 0.5]);
	closeVec(p.rotation.flat(), IDENTITY.flat());
});

// ---------------------------------------------------------------------------
// C4: segments and units

test("C4: 120 mm, 60 deg and 1 rad are 3, 3 and 2 equal segments, every goal from where the delta starts", () => {
	const move = toDelta(parseAction({ type: "move", direction: "forward", distance_mm: 120 }), CAPS);
	const segs = segmentsFor(move, AT, CAPS);
	assert.equal(segs.length, 3);
	segs.forEach((s, k) => {
		close(s.distance_m as number, 0.04);
		closeVec(s.goal_position_m as number[], [0.45 + 0.04 * (k + 1), 0, 0.3]);
		assert.deepEqual(s.goal_rotation, IDENTITY);
		assert.equal(s.label, `move forward 120.0 mm (base) [${k + 1}/3]`);
		assert.equal(s.stop_on_contact, false);
	});
	const turn = toDelta(parseAction({ type: "rotate", direction: "yaw_left", angle_deg: 60 }), CAPS);
	const tsegs = segmentsFor(turn, AT, CAPS);
	assert.equal(tsegs.length, 3);
	tsegs.forEach((s, k) => {
		close(s.angle_rad as number, Math.PI / 9);
		// Left-multiplied: about the fixed base axis through the tool point, which stays put.
		const want = matmul(rotationAbout([0, 0, 1], ((k + 1) * Math.PI) / 9), AT.rotation);
		close(angleBetween(s.goal_rotation as never, want), 0, 1e-12);
		assert.deepEqual(s.goal_position_m, AT.position_m);
	});
	const rad = toDelta(parseAction({ type: "rotate", direction: "roll_ccw", angle_deg: (180 / Math.PI) * 1 }), CAPS);
	assert.equal(segmentsFor(rad, AT, CAPS).length, 2);
	// Exactly two limits long is two segments, not three.
	assert.equal(
		segmentsFor(toDelta(parseAction({ type: "move", direction: "up", distance_mm: 100 }), CAPS), AT, CAPS).length,
		2,
	);
	// A start that is not the identity: the goals turn it about the base axis.
	const tilted: Pose = { position_m: [0.4, 0.1, 0.2], rotation: rotationAbout([1, 0, 0], 0.3) };
	const last = segmentsFor(turn, tilted, CAPS).at(-1);
	const g = { position_m: last?.goal_position_m, rotation: last?.goal_rotation } as Pose;
	const t = turnBetween(tilted, g);
	closeVec(t.axis, [0, 0, 1], 1e-9);
	close(t.angle, Math.PI / 3, 1e-9);
});

test("C4: speeds are SI and bounded per tick: 25 mm/s sends at most 1.25 mm a tick", () => {
	const slow = toDelta(parseAction({ type: "move", direction: "left", distance_mm: 60, speed_mm_s: 25 }), CAPS);
	close(slow.speed, 0.025);
	for (const s of segmentsFor(slow, AT, CAPS)) {
		assert.ok((s.max_step as number) <= 0.025 * 0.05 + 1e-9);
		close(s.max_step as number, 0.00125);
		close(s.duration_s, 0.03 / 0.025);
	}
	// Defaults: 50 mm/s and 60 deg/s; above the cap, the cap.
	close(toDelta(parseAction({ type: "move", direction: "up", distance_mm: 10 }), CAPS).speed, 0.05);
	close(toDelta(parseAction({ type: "rotate", direction: "cw", angle_deg: 10 }), CAPS).speed, Math.PI / 3);
	close(toDelta(parseAction({ type: "move", direction: "up", distance_mm: 10, speed_mm_s: 5000 }), CAPS).speed, 1.0);
	close(tickLimit({ kind: "translate", speed: 1.0 }, CAPS), 0.05);
	close(tickLimit({ kind: "rotate", speed: 10 }, CAPS), 0.5);
	const wait = segmentsFor(toDelta(parseAction({ type: "wait", seconds: 0.52 }), CAPS), AT, CAPS);
	assert.deepEqual([wait.length, wait[0].ticks], [1, 10]);
	// The original's count: at least one tick, the seconds over the period truncated.
	const ticks = (seconds: number) =>
		segmentsFor(toDelta(parseAction({ type: "wait", seconds }), CAPS), AT, CAPS)[0].ticks;
	assert.deepEqual([ticks(0.3), ticks(0.02)], [5, 1]);
	const grip = segmentsFor(toDelta(parseAction({ type: "gripper", state: "close", width_mm: 30 }), CAPS), AT, CAPS);
	assert.equal(grip.length, 1);
	assert.equal(grip[0].state, "close");
	close(grip[0].duration_s, 1.6);
});

test("C4: a measurement comes back in SI units and reads in mm (2 decimals) and deg (1 decimal)", () => {
	const move = toDelta(parseAction({ type: "move", direction: "forward", distance_mm: 50 }), CAPS);
	const seg = segmentsFor(move, AT, CAPS)[0];
	const m = measured(
		{
			outcome: "done",
			abort_reason: "",
			moved_m: 0.0491234,
			along_axis_m: 0.0490004,
			lateral_m: 0.0012345,
			turned_rad: 0.0174533,
			ticks: 21,
			elapsed_s: 0.4567,
			frame_clock_s: 12.35,
			pose_after: { position_m: [0.499, 0, 0.3], rotation: IDENTITY },
		},
		seg,
	);
	const r = report(m);
	assert.deepEqual([r.moved_mm, r.along_axis_mm, r.lateral_mm, r.turned_deg, r.axis], [49.12, 49, 1.23, 1, [1, 0, 0]]);
	assert.deepEqual((r.pose_after as { position_mm: number[] }).position_mm, [499, 0, 300]);
	// A push reports no axis; a gripper segment its opening in mm.
	const push = segmentsFor(
		toDelta(parseAction({ type: "move", direction: "down", distance_mm: 30, stop_on_contact: true }), CAPS),
		AT,
		CAPS,
	)[0];
	assert.equal(report(measured({ outcome: "contact", abort_reason: "blocked" }, push)).axis, undefined);
	const grip = segmentsFor(toDelta(parseAction({ type: "gripper", state: "close" }), CAPS), AT, CAPS)[0];
	assert.deepEqual(report(measured({ outcome: "done", detail: { opening_mm: 21.2 } }, grip)).detail, {
		opening_mm: 21.2,
	});
	assert.equal(measured({ outcome: "exploded" }, grip).outcome, "failed");
});

// ---------------------------------------------------------------------------
// C5: the command language, its limits and labels

test("C5: a command is a JSON string, an action, a list or {actions} whose common keys every action inherits", () => {
	const one = parseCommand('{"type": "move", "direction": "Left", "distance_mm": 20}');
	assert.equal(one[0].direction, "left");
	const listed = parseCommand([{ type: "home" }, { type: "wait", seconds: "0.5" }]);
	assert.deepEqual(
		listed.map((a) => a.type),
		["home", "wait"],
	);
	assert.equal(listed[1].seconds, 0.5);
	const wrapped = parseCommand({
		frame: "BASE",
		speed_mm_s: 20,
		note: "top",
		actions: [
			{ type: "move", axis: [0, 0, -2], distance_mm: 30 },
			{ type: "move", direction: "up", distance_mm: 10, speed_mm_s: 40 },
		],
	});
	assert.deepEqual(
		wrapped.map((a) => [a.frame, a.speed_mm_s, a.note]),
		[
			["base", 20, "top"],
			["base", 40, "top"],
		],
	);
	assert.deepEqual(wrapped[0].axis, [0, 0, -2]);
});

test("C5: every malformed action is refused with a message the model can act on", () => {
	const bad: [unknown, RegExp][] = [
		["{nope", /not JSON/],
		[[], /no actions/],
		[{ actions: [] }, /no actions/],
		[{ actions: [{ type: "home" }], speed: 3 }, /unknown top-level key speed/],
		[{ actions: "home" }, /"actions" must be a list/],
		[{ direction: "up" }, /no type; the types are move, rotate, gripper, home, wait \(it has the keys direction\)/],
		[{ type: "jump" }, /unknown type "jump"/],
		[
			{ type: "move", direction: "up", distance_mm: 5, distance: 5 },
			/unknown key distance; a move takes type, direction, axis, distance_mm/,
		],
		[{ type: "move", distance_mm: 5 }, /exactly one of "direction"/],
		[{ type: "move", direction: "up", axis: [0, 0, 1], distance_mm: 5 }, /exactly one of "direction"/],
		[{ type: "move", direction: 3, distance_mm: 5 }, /direction must be a word/],
		[{ type: "move", axis: [0, 1], distance_mm: 5 }, /axis must be \[x, y, z\]/],
		[
			{ type: "move", direction: "up", distance_mm: -5 },
			/cannot be negative \(-5\); use the opposite direction word/,
		],
		[{ type: "move", axis: [0, 0, 1], distance_mm: -5 }, /negate the axis/],
		[{ type: "move", direction: "up", distance_mm: 0 }, /is 0, which does nothing/],
		[{ type: "move", direction: "up" }, /distance_mm is required/],
		[{ type: "move", direction: "up", distance_mm: "far" }, /must be a finite number/],
		[{ type: "move", direction: "up", distance_mm: Number.POSITIVE_INFINITY }, /must be a finite number/],
		[{ type: "rotate", direction: "cw", angle_deg: Number.NaN }, /must be a finite number/],
		[
			{ type: "move", direction: "up", distance_mm: 5, stop_on_contact: "yes" },
			/stop_on_contact must be true or false/,
		],
		[{ type: "move", direction: "up", distance_mm: 5, frame: "world" }, /frame must be "base"/],
		[{ type: "gripper", state: "half" }, /state must be "open" or "close"/],
		[{ type: "wait", seconds: 0 }, /seconds must be more than 0/],
		[{ type: "wait" }, /seconds is required/],
	];
	for (const [payload, why] of bad) assert.throws(() => parseCommand(payload), why, JSON.stringify(payload));
	assert.ok(new CommandError("x") instanceof Error);
});

test("C5: conversion refuses what the robot cannot do and drops what MotorMind drops", () => {
	const deltas = (p: unknown, caps = CAPS) => toDeltas(p, caps);
	assert.throws(
		() => deltas({ type: "move", direction: "down", distance_mm: 251, stop_on_contact: true }),
		/at most 250 mm/,
	);
	assert.equal(deltas({ type: "move", direction: "down", distance_mm: 250, stop_on_contact: true })[0].push, true);
	assert.throws(() => deltas({ type: "move", direction: "up", distance_mm: 5, speed_mm_s: 0 }), /must be positive/);
	assert.throws(
		() => deltas({ type: "rotate", direction: "cw", angle_deg: 5, rot_speed_deg_s: -1 }),
		/must be positive/,
	);
	assert.throws(() => deltas({ type: "move", direction: "sideways", distance_mm: 5 }), /directions are/);
	assert.throws(() => deltas({ type: "rotate", direction: "spin", angle_deg: 5 }), /aliases:/);
	assert.throws(() => deltas({ type: "move", axis: [0, 0, 0], distance_mm: 5 }), /has no direction/);
	// The model's width is parsed, then dropped.
	assert.equal(deltas({ type: "gripper", state: "open", width_mm: 40 })[0].open_to_mm, undefined);
	// No home unless the robot has one (--mm-home reset).
	assert.throws(() => deltas({ type: "home" }), /this robot has no home/);
	assert.equal(deltas({ type: "home" }, { ...CAPS, supports_home: true })[0].kind, "home");
	const axis = deltas({ type: "move", axis: [0, 3, 4], distance_mm: 10 })[0].axis as number[];
	closeVec(axis, [0, 0.6, 0.8]);
});

test("C5: one command turns at most 90 deg, moves nothing after home, and ends within 400 mm", () => {
	assert.throws(
		() =>
			toDeltas(
				[
					{ type: "rotate", direction: "cw", angle_deg: 60 },
					{ type: "rotate", direction: "ccw", angle_deg: 45 },
				],
				CAPS,
			),
		/turns 105.0 deg in all; one command turns at most 90 deg/,
	);
	assert.equal(
		toDeltas(
			[
				{ type: "rotate", direction: "cw", angle_deg: 45 },
				{ type: "rotate", direction: "ccw", angle_deg: 45 },
			],
			CAPS,
		).length,
		2,
	);
	const home = { ...CAPS, supports_home: true };
	assert.throws(
		() => toDeltas([{ type: "home" }, { type: "move", direction: "up", distance_mm: 5 }], home),
		/nothing may move after home/,
	);
	assert.equal(toDeltas([{ type: "home" }, { type: "gripper", state: "open" }], home).length, 2);
	// 300 forward then 300 left: the second ends on the 400 mm circle (sqrt(400^2 - 300^2) = 264.6 mm),
	// and everything after it is dropped.
	const cut = toDeltas(
		[
			{ type: "move", direction: "forward", distance_mm: 300 },
			{ type: "move", direction: "left", distance_mm: 300 },
			{ type: "gripper", state: "close" },
		],
		CAPS,
	);
	assert.equal(cut.length, 2);
	close(cut[1].magnitude, Math.sqrt(0.4 ** 2 - 0.3 ** 2), 1e-12);
	assert.equal(
		cut[1].label,
		"move left 300.0 mm (base) (shortened to 265 mm: the command's moves end at most 400 mm from where it started)",
	);
	assert.match(
		cut[1].note,
		/ran only 265 mm, and the 1 action\(s\) after it did not run; send what is left as another command/,
	);
	// Straight-line distance, not the sum: back and forth is fine.
	assert.equal(
		toDeltas(
			[
				{ type: "move", direction: "forward", distance_mm: 350 },
				{ type: "move", direction: "backward", distance_mm: 350 },
				{ type: "move", direction: "forward", distance_mm: 350 },
			],
			CAPS,
		).length,
		3,
	);
	assert.throws(
		() =>
			commandLimits(
				toDeltas([{ type: "move", direction: "up", distance_mm: 400 }], CAPS).concat(
					toDelta(parseAction({ type: "move", direction: "up", distance_mm: 5 }, 1), CAPS),
				),
			),
		/already end 400 mm from where it started/,
	);
});

test("C5: the reach limit cuts a move to whole mm that fit and drops the rest; a push is not checked", () => {
	const from: Pose = { position_m: [0.7, 0, 0.3], rotation: IDENTITY };
	const fresh = () =>
		toDeltas(
			[
				{ type: "move", direction: "forward", distance_mm: 100 },
				{ type: "move", direction: "down", distance_mm: 30 },
			],
			CAPS,
		);
	const r = fitToReach(fresh(), from, CAPS);
	assert.equal(r.shortened, true);
	assert.equal(r.deltas.length, 1);
	close(r.deltas[0].magnitude, 0.08, 1e-12);
	assert.match(
		r.deltas[0].label,
		/^move forward 100\.0 mm \(base\) \(shortened to 80 mm: it would end 800 mm from the base, past the 780 mm reach\)$/,
	);
	// Inward or along the edge from outside the reach: up to 5 mm further out is allowed.
	const outside: Pose = { position_m: [0.79, 0, 0.3], rotation: IDENTITY };
	assert.equal(
		fitToReach(toDeltas({ type: "move", direction: "backward", distance_mm: 50 }, CAPS), outside, CAPS).shortened,
		false,
	);
	assert.equal(
		fitToReach(toDeltas({ type: "move", direction: "forward", distance_mm: 4 }, CAPS), outside, CAPS).shortened,
		false,
	);
	// Nothing of a later move fits once an earlier one reached the edge: refused, unless it ends within the
	// arrival tolerance of the reach.
	const edge = (mm: number) =>
		fitToReach(
			toDeltas(
				[
					{ type: "move", direction: "forward", distance_mm: 80 },
					{ type: "move", direction: "forward", distance_mm: mm },
				],
				CAPS,
			),
			from,
			CAPS,
		);
	assert.match(
		String(edge(40).refused),
		/would end 820 mm from the base, past the 780 mm reach, and no part of it fits/,
	);
	assert.deepEqual([edge(2).refused, edge(2).shortened], [undefined, false]);
	// A push decides where it stops by contact: it and what follows are not checked.
	assert.equal(
		fitToReach(
			toDeltas({ type: "move", direction: "forward", distance_mm: 200, stop_on_contact: true }, CAPS),
			from,
			CAPS,
		).shortened,
		false,
	);
	// Each move is judged from where it starts, and what fits is measured against the reach: from 790 mm
	// out, 10 mm further is refused (the 5 mm slack does not make room to cut it to), and after 30 mm back
	// a 34 mm move out again is cut to the 20 mm that end inside the reach.
	assert.match(
		String(
			fitToReach(toDeltas({ type: "move", direction: "forward", distance_mm: 10 }, CAPS), outside, CAPS).refused,
		),
		/would end 800 mm from the base, past the 780 mm reach, and no part of it fits/,
	);
	const back = fitToReach(
		toDeltas(
			[
				{ type: "move", direction: "backward", distance_mm: 30 },
				{ type: "move", direction: "forward", distance_mm: 34 },
			],
			CAPS,
		),
		outside,
		CAPS,
	);
	assert.equal(back.shortened, true);
	close(back.deltas[1].magnitude, 0.02, 1e-12);
	// Vertical moves never change the horizontal reach.
	assert.equal(
		fitToReach(toDeltas({ type: "move", direction: "up", distance_mm: 300 }, CAPS), outside, CAPS).shortened,
		false,
	);
});

test("C5: labels are the reference's forms, the executor's inserted moves included", () => {
	const label = (a: Record<string, unknown>) => actionLabel(parseAction(a));
	assert.equal(label({ type: "move", direction: "left", distance_mm: 50 }), "move left 50.0 mm (base)");
	assert.equal(label({ type: "move", axis: [0, 1, 0], distance_mm: 20 }), "move axis [0.0, 1.0, 0.0] 20.0 mm (base)");
	assert.equal(label({ type: "rotate", direction: "yaw_left", angle_deg: 30 }), "rotate yaw_left 30.0 deg (base)");
	assert.equal(label({ type: "rotate", direction: "CW", angle_deg: 12.25 }), "rotate cw 12.2 deg (base)");
	assert.equal(
		label({ type: "rotate", axis: [0, 0, -1], angle_deg: 15 }),
		"rotate axis [0.0, 0.0, -1.0] 15.0 deg (base)",
	);
	assert.equal(label({ type: "gripper", state: "close" }), "gripper close");
	assert.equal(label({ type: "gripper", state: "open", width_mm: 40 }), "gripper open to 40.0 mm");
	assert.equal(label({ type: "home" }), "go home");
	assert.equal(label({ type: "wait", seconds: 0.5 }), "wait 0.50 s");
	assert.equal(segmentLabel("move left 50.0 mm (base)", 2, 3), "move left 50.0 mm (base) [2/3]");
	assert.equal(
		shortenedLabel("move up 90.0 mm (base)", 64.6, "why"),
		"move up 90.0 mm (base) (shortened to 65 mm: why)",
	);
	assert.equal(carryLegLabel("move forward 120.0 mm (base)", 1, 3), "move forward 120.0 mm (base) (1 of 3)");
	assert.equal(carryLegLabel("", 2, 2), "move (2 of 2)");
	assert.equal(testLeadLabel("move up 80.0 mm (base)", 17), "move up 80.0 mm (base) (the first 17 mm)");
	assert.equal(testRestLabel("move up 80.0 mm (base)"), `move up 80.0 mm (base) ${TEST_REST}`);
	assert.equal(TEST_REST, "(the rest)");
	assert.equal(LIFT_LABEL, "lift over the table");
	assert.equal(DROP_LABEL, "back down from over the table");
	assert.equal(followLabel([0.0123, -0.00004], 12.31), "move axis [0.0123, -4e-05, 0.0] 12.3 mm (base)");
	// Python's formatting: exact ties to even, repr's float forms.
	assert.deepEqual(
		[
			pyFixed(12.25, 1),
			pyFixed(12.75, 1),
			pyFixed(0.5, 0),
			pyFixed(1.5, 0),
			pyFixed(2.5, 0),
			pyFixed(0.15, 1),
			pyFixed(-0.25, 1),
		],
		["12.2", "12.8", "0", "2", "2", "0.1", "-0.2"],
	);
	assert.deepEqual(
		[
			pyFloat(1),
			pyFloat(0.1),
			pyFloat(1e-5),
			pyFloat(1.5e-7),
			pyFloat(-0),
			pyFloat(1e16),
			pyFloat(123456.789),
			pyFloat(-2),
		],
		["1.0", "0.1", "1e-05", "1.5e-07", "-0.0", "1e+16", "123456.789", "-2.0"],
	);
});

test("--preset motormind expands to the faithful flags and needs a robot with MotorMind mode", () => {
	const p = PRESETS.motormind;
	assert.deepEqual(p.flags, {
		mm: "true",
		"libero-side-camera": "true",
		"libero-settle-ticks": "32",
		"libero-max-steps": "300000",
		"memory-profile": "off",
		units: null,
		code: null,
		vdm: false,
		"vdm-video": false,
	});
	assert.deepEqual(p.axes, { turns: "multi", feedback: "image", api: "raw" });
	const c = choose("", "motormind");
	assert.ok(c && !("error" in c));
	// A variant does not take the preset: a contradicting flag is a start error.
	assert.deepEqual(conflicts(c, new Map([["libero-side-camera", "false"]])), [
		"--preset motormind sets --libero-side-camera=true, but --libero-side-camera=false was given",
	]);
	assert.deepEqual(conflicts(c, new Map([["memory-profile", "local"]])), [
		"--preset motormind sets --memory-profile=off, but --memory-profile=local was given",
	]);
	const toy = {
		robot: "toy",
		code: false,
		units: false,
		vdm: false,
		groundTruth: false,
		memory: false,
		xpolicy: false,
		vla: false,
		codeTiers: () => [],
	};
	assert.equal(unsupported(c, toy), "--preset motormind runs MotorMind mode (--mm), which toy does not mount");
	assert.equal(unsupported(c, { ...toy, motormind: true }), undefined);
});

// ---------------------------------------------------------------------------
// the robot side, on a fake LIBERO env server

type Handler = (event: any, ctx: any) => unknown;
type Json = Record<string, any>;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
/** An image as the env server sends one: base64 JPEG bytes (a marker and an end; nothing decodes it). */
const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xd9]).toString("base64");

/** The configuration a server pi starts for `values` reports (`config`), as env_server.py's static fields carry it. */
function startedWith(values: Record<string, unknown>): Json {
	const preset = values.preset === "motormind";
	return {
		side_camera: String(values["libero-side-camera"] ?? (preset ? "true" : "false")) === "true",
		settle_ticks: Number(values["libero-settle-ticks"] ?? (preset ? 32 : 0)),
		mm_home: String(values["mm-home"] ?? "refuse"),
		max_episode_steps: Number(values["libero-max-steps"] ?? (preset ? 300000 : 10000)),
		truth: null,
	};
}

/**
 * A fake LIBERO env server with MotorMind's two RPCs, answering in env_server.py's wire format: a
 * segment reaches its goal in ticks of max_step, every tick one transition. Like the real one it keeps
 * the steps of a tool call until the next call acknowledges the reply (`delivered`), and hands them
 * back otherwise. `successAt`: LIBERO's success latches at that env step.
 */
async function fakeMmEnv(o: { mm?: boolean; maxSteps?: number; successAt?: number; config: Json }) {
	const s = {
		steps: 0,
		clock: 0,
		obsId: 0,
		pos: [0.45, 0, 0.3],
		rot: IDENTITY as number[][],
		segments: [] as any[],
		resets: [] as Record<string, unknown>[],
		homes: 0,
		delayMs: 0,
		stallMs: 0,
		over: false,
		success: false,
		call: 0,
		unacked: undefined as { id: number; transitions: Json[] } | undefined,
	};
	const obs = () => ({
		main_images: nd("uint8", [4, 4, 3], Buffer.alloc(48)),
		wrist_images: nd("uint8", [4, 4, 3], Buffer.alloc(48)),
		states: f32([...s.pos, 0, 0, 0, 0.04, -0.04]),
	});
	const transition = () => {
		s.steps++;
		s.clock += 0.05;
		if (o.successAt && s.steps >= o.successAt) s.success = true;
		const truncated = s.steps >= (o.maxSteps ?? 1e9);
		return { action: f32([0, 0, 0, 0, 0, 0, -1]), obs: obs(), reward: 0, terminated: s.success, truncated };
	};
	const view = () => ({ marked768: JPEG, marked512: JPEG, raw512: JPEG });
	const observation = (images = true) => ({
		obs_id: ++s.obsId,
		frame_clock_s: s.clock,
		episode_id: 1,
		ticks: s.steps,
		base_pose: { position_m: [-0.66, 0, 0.912], rotation: IDENTITY, quat_xyzw: [0, 0, 0, 1] },
		caps: { ...CAPS, supports_home: o.config.mm_home === "reset" },
		output_max: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
		pose: { position_m: [...s.pos], rotation: s.rot, quat_xyzw: [0, 0, 0, 1] },
		gripper: { opening_mm: 80, commanded: "open", channel: -1, moving: false },
		grasp: { holding: null, reason: "not closed", opening_mm: 80, closed_fraction: 1.015 },
		table_z_m: 0,
		terminated: s.success,
		truncated: s.over,
		episode_over: s.over || s.success,
		degraded: { table_z: false, side_mount: false, side_mount_reason: null },
		config: o.config,
		...(images ? { cameras: { scene: view(), side: view(), wrist: view() }, mosaic_jpeg: JPEG } : {}),
	});
	/** The env steps of a call whose reply pi did not receive (`delivered` is not its id). */
	const handBack = (delivered: unknown): Json[] => {
		if (typeof delivered !== "number") return [];
		const held = s.unacked;
		s.unacked = undefined;
		return held && held.id !== delivered ? held.transitions : [];
	};
	const keep = (transitions: Json[]) => {
		if (!transitions.length) return null;
		s.unacked = { id: ++s.call, transitions };
		return s.call;
	};
	const answer = async ({ method, kwargs }: Call): Promise<unknown> => {
		if (method === "code.api") return codeApiReply("libero", kwargs.tier);
		if (method === "env.reset") {
			s.resets.push(kwargs);
			Object.assign(s, { steps: 0, over: false, success: false, pos: [0.45, 0, 0.3], rot: IDENTITY });
			s.unacked = undefined;
			if (!kwargs.tool_call) return [obs(), {}];
			const transitions = [transition(), transition(), transition()];
			return [obs(), { transitions, settle_ticks: 3 }];
		}
		if (method === "env.get_task_language") return "put the bowl on the plate";
		if (method === "env.get_env_meta") return {};
		if (method === "env.render_camera") {
			const n = kwargs.height as number;
			const rgb = nd("uint8", [n, n, 3], Buffer.alloc(n * n * 3, 40));
			const depth = nd("float32", [n, n], Buffer.from(new Float32Array(n * n).fill(1).buffer));
			return kwargs.depth ? [rgb, depth] : rgb;
		}
		if (method === "env.get_camera_meta") {
			const n = kwargs.height as number;
			return {
				intrinsic_K: [
					[n / 2, 0, n / 2],
					[0, n / 2, n / 2],
					[0, 0, 1],
				],
				extrinsic_cam2world: [
					[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			};
		}
		if (method === "env.mm_observe") {
			if (o.mm === false) throw new Error("no such method env.mm_observe");
			const late = handBack(kwargs.delivered);
			const out = observation(kwargs.images !== false);
			return kwargs.delivered === undefined ? out : { ...out, transitions: late, call_id: keep(late) };
		}
		if (method === "env.mm_run_step") {
			const seg = kwargs.seg;
			s.segments.push(seg);
			// A home starts a new episode: the last one's undelivered steps are dropped, not handed back.
			const held = handBack(kwargs.delivered);
			const late = seg.kind === "home" ? [] : held;
			if (s.stallMs) {
				const ms = s.stallMs;
				s.stallMs = 0;
				await sleep(ms);
			}
			if (s.delayMs) await sleep(s.delayMs);
			const after = () => {
				const full = observation(kwargs.observe === "full");
				return full;
			};
			const reply = (out: Json, fresh: Json[]) => {
				const transitions = [...late, ...fresh];
				return {
					...out,
					observation: after(),
					...(kwargs.tool_call
						? { transitions, late_steps: late.length, call_id: seg.kind === "home" ? null : keep(transitions) }
						: {}),
				};
			};
			if (seg.kind === "home") {
				s.homes++;
				Object.assign(s, { steps: 0, over: false, success: false, pos: [0.45, 0, 0.3], rot: IDENTITY });
				const settle = [transition(), transition(), transition()];
				return reply(
					{ outcome: "done", message: "env.reset()", episode_reset: true, settle_ticks: 3, ticks: 3, obs: obs() },
					settle,
				);
			}
			if (s.over || s.success)
				return reply({ outcome: "refused", message: "the episode is over", episode_over: true }, []);
			let ticks = 1;
			let moved = 0;
			let along = 0;
			let turned = 0;
			if (seg.kind === "translate") {
				const d = seg.goal_position_m.map((v: number, i: number) => v - s.pos[i]);
				moved = Math.hypot(...d);
				along = d.reduce((a: number, v: number, i: number) => a + v * seg.axis[i], 0);
				ticks = Math.max(1, Math.ceil(moved / seg.max_step - 1e-9));
				s.pos = [...seg.goal_position_m];
			} else if (seg.kind === "rotate") {
				turned = angleBetween(s.rot as never, seg.goal_rotation);
				ticks = Math.max(1, Math.ceil(turned / seg.max_step - 1e-9));
				s.rot = seg.goal_rotation;
			} else if (seg.kind === "gripper") ticks = 32;
			else if (seg.kind === "wait") ticks = seg.ticks;
			const fresh = Array.from({ length: ticks }, transition);
			s.over = s.steps >= (o.maxSteps ?? 1e9);
			return reply(
				{
					outcome: "done",
					abort_reason: "",
					moved_m: moved,
					along_axis_m: seg.kind === "translate" ? along : moved,
					lateral_m: 0,
					turned_rad: turned,
					pose_after: { position_m: [...s.pos], rotation: s.rot, quat_xyzw: [0, 0, 0, 1] },
					ticks,
					elapsed_s: 0.01,
					frame_clock_s: s.clock,
					...(seg.kind === "gripper"
						? { detail: { opening_mm: 1.2 }, grasp: { holding: false, reason: "closed on nothing" } }
						: {}),
					episode_over: s.over || s.success,
					terminated: s.success,
				},
				fresh,
			);
		}
		return undefined;
	};
	const server = await fakeEnv(answer);
	return { ...server, state: s };
}

/** LIBERO in a stub pi on the fake env (flags as on the command line); `model` is the session's model. */
async function session(
	t: { after: (fn: () => void) => void },
	values: Record<string, unknown>,
	o: {
		mm?: boolean;
		maxSteps?: number;
		successAt?: number;
		model?: { provider: string; id: string };
		config?: Json;
	} = {},
) {
	const env = await fakeMmEnv({ ...o, config: { ...startedWith(values), ...o.config } });
	t.after(env.close);
	const cwd = mkdtempSync(join(tmpdir(), "motormind-"));
	const flags: Record<string, unknown> = deployFlags({
		"env-url": env.url,
		suite: "libero_10",
		task: "0",
		seed: "0",
		...values,
	});
	const handlers = new Map<string, Handler[]>();
	const registered: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	let handle: MotorMindHandle | undefined;
	const errors: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, f: { default?: unknown }) => {
			registered[name] = name in flags ? flags[name] : f.default;
		},
		getFlag: (name: string) => registered[name],
		registerTool: (tool: any) => tools.set(tool.name, tool),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: {
			emit: (channel: string, data: any) => {
				if (channel === MOTORMIND_EVENT) handle = data;
			},
			on: () => () => {},
		},
	};
	const pi = new Proxy(api, { get: (target, k: string) => target[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		cwd,
		model: o.model,
		ui: new Proxy(
			{ notify: (m: string) => errors.push(m) },
			{ get: (target: any, k: string) => target[k] ?? (() => {}) },
		),
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionDir: () => cwd,
			getSessionFile: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	const call = async (name: string, params: Record<string, unknown>) => {
		assert.ok(active.includes(name), `${name} is not active: ${active.join(", ")}`);
		try {
			const r = await tools.get(name).execute(`id-${name}`, params, undefined, undefined, ctx);
			return { ...r, isError: false };
		} catch (err) {
			return {
				content: [{ type: "text", text: String(err instanceof Error ? err.message : err) }],
				details: {},
				isError: true,
			};
		}
	};
	libero(pi);
	const loaded = [...tools.keys()];
	await emit("session_start", { reason: "startup" });
	const finish = async () => {
		await emit("before_agent_start", { prompt: "Solve the task.", systemPrompt: "" });
		await emit("agent_start");
		await call("finish", { status: "failure", summary: "test" });
		await emit("agent_end", { messages: [] });
		return entries.find((e) => e.type === RESULT_ENTRY)?.data;
	};
	/** The start failed closed: no tools, and the UI was told why. */
	const unavailable = () => errors.some((e) => /^libero unavailable: /.test(e));
	return {
		env,
		emit,
		call,
		entries,
		errors,
		unavailable,
		finish,
		flags: registered,
		active: () => active,
		handle: () => handle,
		/** The tools registered at load, and now. */
		loaded,
		tools: () => [...tools.keys()],
	};
}

const body = (r: any) => JSON.parse(r.content[0].text);

test("--preset motormind: pure mode is mm_observe, mm_act and finish; mm_act cuts one action into segments", async (t) => {
	const s = await session(t, { preset: "motormind" });
	assert.equal(s.unavailable(), false, s.errors.join("\n"));
	assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);
	// The tools are registered at the start that turns the mode on, not at load.
	assert.ok(!s.loaded.some((n) => n.startsWith("mm_")), s.loaded.join(", "));
	assert.equal(s.flags["memory-profile"], undefined, "the preset expands the flag, the raw value stays unset");
	// The settle ticks of the reset count as env steps (and went to the recorder).
	assert.deepEqual(s.env.state.resets, [{ tool_call: true }]);
	const seen = await s.call("mm_observe", {});
	assert.ok(!seen.isError, seen.content[0].text);
	const o = body(seen);
	assert.deepEqual(o.pose.position_mm, [450, 0, 300]);
	assert.equal(o.caps.supports_home, false);
	assert.equal(o.caps.reach_m, 0.78, "the server's capability table");
	assert.deepEqual(
		seen.content.filter((c: any) => c.type === "image").map((c: any) => c.mimeType),
		["image/jpeg"],
		"one mosaic image",
	);
	// tracked_mm goes to the server as the aim marker's tracked depth (m).
	await s.call("mm_observe", { tracked_mm: 90 });
	assert.equal(s.env.calls.filter((c) => c.method === "env.mm_observe").at(-1)?.kwargs.aim_tracked_m, 0.09);
	const acted = await s.call("mm_act", { type: "move", direction: "left", distance_mm: 120, speed_mm_s: 25 });
	assert.ok(!acted.isError, acted.content[0].text);
	const r = body(acted);
	assert.equal(r.action, "move left 120.0 mm (base)");
	assert.deepEqual(
		r.segments.map((x: any) => [x.label, x.outcome, x.along_axis_mm, x.axis]),
		[1, 2, 3].map((i) => [`move left 120.0 mm (base) [${i}/3]`, "done", 40, [0, 1, 0]]),
	);
	const sent = s.env.state.segments;
	assert.deepEqual(
		sent.map((x: any) => x.goal_position_m.map((v: number) => Number(v.toFixed(6)))),
		[
			[0.45, 0.04, 0.3],
			[0.45, 0.08, 0.3],
			[0.45, 0.12, 0.3],
		],
	);
	for (const x of sent) close(x.max_step, 0.00125, 1e-12);
	assert.deepEqual(r.pose.position_mm, [450, 120, 300]);
	// Every mm call says which reply it got last: each one acknowledges the one before.
	const runs = s.env.calls.filter((c) => c.method === "env.mm_run_step").map((c) => c.kwargs.delivered);
	assert.deepEqual(runs, [0, 1, 2]);
	// Outside a robot tool the callbacks refuse to run: the lock is held only by a tool call.
	const handle = s.handle();
	assert.ok(handle, "the mode published its handle");
	await assert.rejects(
		handle.runStep(sent[0], { timeoutMs: 1000, observe: "state" }),
		/runs only under the robot lock/,
	);
	await assert.rejects(handle.observe(), /runs only under the robot lock/);
	assert.equal(s.env.state.segments.length, 3);
	const result = await s.finish();
	assert.equal(result.preset, "motormind");
	assert.equal(result.params.mm, "true");
	assert.equal(result.params["libero-side-camera"], "true");
	assert.equal(result.params["memory-profile"], "off");
	assert.equal(result.settle_ticks, 3);
	assert.equal(result.env_steps, 3 + 3 * 32);
	assert.deepEqual(result.motormind.variant, []);
	assert.equal(result.motormind.segments, 3);
	assert.equal(result.motormind.ticks, 96);
	assert.equal(result.motormind.status, null);
	assert.deepEqual(result.motormind.home, { mode: "refuse", asked: 0, refused: 0, reset: 0 });
});

test("a success mid-segment: the segment runs to its end, then the episode is over", async (t) => {
	// Success latches at env step 10 (3 settle steps, then the 7th of the first 40 mm segment's 16 ticks).
	const s = await session(t, { preset: "motormind" }, { successAt: 10 });
	const r = body(await s.call("mm_act", { type: "move", direction: "up", distance_mm: 120 }));
	assert.deepEqual(
		r.segments.map((x: any) => [x.outcome, x.ticks]),
		[
			["done", 16],
			["refused", 0],
			["refused", 0],
		],
	);
	assert.equal(r.episode_over, true);
	const result = await s.finish();
	assert.deepEqual([result.terminated, result.success_step, result.env_steps], [true, 10, 3 + 16]);
	assert.equal(result.motormind.status, "env_success");
});

test("a cancel stops mm_act at the next segment boundary: the segment in flight finishes, the rest are skipped", async (t) => {
	const s = await session(t, { preset: "motormind" });
	s.env.state.delayMs = 150;
	const handle = s.handle() as MotorMindHandle;
	setTimeout(() => handle.cancel("the monitor said stop"), 50);
	const r = body(await s.call("mm_act", { type: "move", direction: "forward", distance_mm: 200 }));
	assert.deepEqual(
		r.segments.map((x: any) => [x.outcome, x.cancel_reason ?? null]),
		[["done", null], ...Array(3).fill(["skipped", "the monitor said stop"])],
	);
	assert.equal(r.cancelled.at_segment, 1);
	assert.ok(r.cancelled.latency_s >= 0.05 && r.cancelled.latency_s < 2, JSON.stringify(r.cancelled));
	assert.equal(s.env.state.segments.length, 1);
	// Cleared, the next action runs; a passed deadline stops it like a cancel.
	handle.clearCancel();
	s.env.state.delayMs = 0;
	assert.equal(
		body(await s.call("mm_act", { type: "move", direction: "up", distance_mm: 10 })).segments[0].outcome,
		"done",
	);
	handle.setDeadline(Date.now() - 1);
	assert.equal(
		body(await s.call("mm_act", { type: "move", direction: "up", distance_mm: 10 })).segments[0].outcome,
		"skipped",
	);
	assert.equal((await s.finish()).motormind.cancelled, 2);
});

test("a segment RPC that times out makes mm_act timed_out; the closed loop refuses mm_act until mm_observe", async (t) => {
	// The stalled segment (env steps 4 to 19) latches success on the server after the RPC timed out.
	const s = await session(t, { preset: "motormind", "mm-step-timeout": "1" }, { successAt: 12 });
	s.env.state.stallMs = 1500;
	const acted = await s.call("mm_act", { type: "move", direction: "down", distance_mm: 80 });
	const r = body(acted);
	assert.equal(r.timed_out, true);
	assert.deepEqual(
		r.segments.map((x: any) => x.outcome),
		["failed", "skipped"],
	);
	assert.match(r.segments[0].error, /timed out/);
	await s.emit("tool_result", { toolName: "mm_act", isError: false, content: acted.content, details: acted.details });
	const refused = await s.emit("tool_call", { toolName: "mm_act", input: {} });
	assert.equal(refused?.block, true);
	assert.match(refused.reason, /unknown outcome \(timed out\)\. Re-observe with mm_observe/);
	// The stalled call still holds the server; the observation waits for it, then releases the gate.
	const seen = await s.call("mm_observe", {});
	assert.ok(!seen.isError, seen.content[0].text);
	await s.emit("tool_result", {
		toolName: "mm_observe",
		isError: false,
		content: seen.content,
		details: seen.details,
	});
	assert.equal(await s.emit("tool_call", { toolName: "mm_act", input: {} }), undefined);
	assert.deepEqual(
		s.entries.filter((e) => e.type === CLOSED_LOOP_ENTRY).map((e) => e.data.kind),
		["unknown_outcome", "refused", "reobserved"],
	);
	// The steps the server ran while pi had given up came back with the observation: the episode
	// counts them, and their success.
	assert.equal(body(seen).terminated, true);
	const result = await s.finish();
	assert.deepEqual([result.env_steps, result.terminated, result.success_step], [3 + 16, true, 12]);
	assert.equal(result.motormind.timed_out, 1);
	assert.deepEqual(result.unknown_outcomes, 1);
	// mm-step-timeout differs from the faithful default: a labelled variant.
	assert.deepEqual(result.motormind.variant, [{ flag: "mm-step-timeout", faithful: "120", actual: "1" }]);
});

test("home is refused by default; --mm-home reset is the episode reset, recorded as a variant", async (t) => {
	const s = await session(t, { preset: "motormind" });
	const refused = await s.call("mm_act", { type: "home" });
	assert.equal(refused.isError, true);
	assert.match(refused.content[0].text, /nothing moved: action 0: this robot has no home/);
	assert.equal(s.env.state.resets.length, 1);
	assert.deepEqual((await s.finish()).motormind.home, { mode: "refuse", asked: 1, refused: 1, reset: 0 });

	const v = await session(t, { mm: "true", "mm-home": "reset", "memory-profile": "off", "libero-settle-ticks": "32" });
	assert.deepEqual(v.active(), ["mm_observe", "mm_act", "finish"]);
	await v.call("mm_act", { type: "move", direction: "up", distance_mm: 20 });
	const home = body(await v.call("mm_act", { type: "home" }));
	assert.deepEqual([home.segments[0].label, home.segments[0].outcome, home.segments[0].ticks], ["go home", "done", 3]);
	assert.deepEqual(home.pose.position_mm, [450, 0, 300]);
	// The server reset the scene (its home); pi's episode started over with its settle ticks.
	assert.deepEqual([v.env.state.resets.length, v.env.state.homes], [1, 1]);
	const result = await v.finish();
	assert.deepEqual(result.motormind.home, { mode: "reset", asked: 1, refused: 0, reset: 1 });
	assert.equal(result.env_steps, 3);
	assert.deepEqual(result.motormind.variant.map((x: any) => x.flag).sort(), [
		"libero-max-steps",
		"libero-side-camera",
		"mm-home",
	]);
});

test("past the episode's end every segment is refused with episode_over", async (t) => {
	const s = await session(t, { preset: "motormind" }, { maxSteps: 10 });
	const r = body(await s.call("mm_act", { type: "move", direction: "forward", distance_mm: 100 }));
	assert.deepEqual(
		r.segments.map((x: any) => x.outcome),
		["done", "refused"],
	);
	assert.equal(r.episode_over, true);
	const again = body(await s.call("mm_act", { type: "gripper", state: "close" }));
	assert.deepEqual([again.segments[0].outcome, again.episode_over], ["refused", true]);
});

test("the robot-side checks fail the start closed: another mode, exploration, extras, a token budget, no mm RPCs, another model, another server", async (t) => {
	for (const [values, o] of [
		[{ preset: "motormind", units: "true" }, {}],
		[{ mm: "true", code: "true", "memory-profile": "off" }, {}],
		[{ mm: "both", "memory-profile": "off" }, {}],
		[{ mm: "1", "memory-profile": "off" }, {}],
		[{ mm: "true", explore: true }, {}],
		[{ preset: "motormind", "max-tokens": "1000" }, {}],
		[{ preset: "motormind", waypoints: true }, {}],
		[{ preset: "motormind", "mm-home": "arm" }, {}],
		[{ preset: "motormind" }, { mm: false }],
		[{ preset: "motormind" }, { model: { provider: "strata", id: "qwen3.8-flash-next" } }],
		// --env-url attached to a server started otherwise than the flags say.
		[{ preset: "motormind" }, { config: { side_camera: false } }],
		[{ preset: "motormind" }, { config: { mm_home: "reset" } }],
		[{ preset: "motormind" }, { config: { settle_ticks: 0, max_episode_steps: 10000 } }],
	] as const) {
		const s = await session(t, values, o);
		assert.deepEqual(s.active(), [], JSON.stringify(values));
		assert.ok(s.unavailable(), `${JSON.stringify([values, o])}: ${s.errors.join("; ")}`);
	}
	const why = async (values: Record<string, unknown>, o = {}) => (await session(t, values, o)).errors.join("; ");
	assert.match(await why({ mm: "both", "memory-profile": "off" }), /--mm must be true or false/);
	assert.match(await why({ mm: "true", explore: true }), /--explore cannot run in MotorMind mode/);
	assert.match(await why({ preset: "motormind", waypoints: true }), /--waypoints cannot run in MotorMind mode/);
	assert.match(
		await why({ preset: "motormind" }, { config: { side_camera: false } }),
		/runs with side_camera=false, but the flags ask for side_camera=true/,
	);
	const ok = await session(
		t,
		{ preset: "motormind" },
		{ model: { provider: "motormind", id: "strata/qwen3.8-flash-next" } },
	);
	assert.equal(ok.unavailable(), false, ok.errors.join("\n"));
	// Off, the tools are neither registered nor served: tool mode never sees them.
	const off = await session(t, { "memory-profile": "off" });
	assert.ok(!off.tools().some((n) => n.startsWith("mm_")), off.tools().join(", "));
	assert.ok(off.active().includes("move_to"));
});

test("the result's status: the robot or the model failing is an error", async (t) => {
	const s = await session(t, { preset: "motormind" });
	await s.emit("before_agent_start", { prompt: "Solve the task.", systemPrompt: "" });
	await s.emit("agent_start");
	await s.emit("message_end", {
		message: { role: "assistant", stopReason: "error", errorMessage: "provider down", content: [] },
	});
	await s.emit("agent_end", { messages: [] });
	await s.emit("session_shutdown", { reason: "quit" });
	const result = s.entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.equal(result.planner_error, "provider down");
	assert.equal(result.motormind.status, "error");
});
