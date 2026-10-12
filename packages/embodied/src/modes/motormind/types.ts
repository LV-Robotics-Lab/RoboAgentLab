/**
 * MotorMind mode types (DESIGN §2-4, docs in pi-embodied-work/docs/motormind): the base frame
 * (`robot0_base`), the command language's parsed actions, the TcpDelta-like deltas they convert to
 * (m, rad), the segments the env server runs one at a time (`env.mm_run_step`), and what it
 * measures. Units follow MotorMind's three layers: the command and every text the model or the
 * planner reads are mm / deg; deltas, segments and RPC payloads are m / rad / s.
 */

export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];
/** A pose in the robot's base frame: position in m, rotation matrix (columns are the tool axes). */
export type Pose = { position_m: Vec3; rotation: Mat3 };

/** The env's control period: every motion tick is 0.05 s of simulation (20 Hz). */
export const DT = 0.05;
/** Gripper close and full open: the channel is set, then this many zero-delta ticks let it settle. */
export const GRIPPER_SETTLE_TICKS = 32;

/**
 * What the robot accepts (MotorMind's `Capabilities`, LIBERO values from spec-adapter §4.9). The env
 * server's `env.mm_observe` returns them as static fields, the only copy; `supports_home` follows
 * the server's `--mm-home`.
 */
export type Caps = {
	max_translation_m: number;
	max_rotation_rad: number;
	max_speed_m_s: number;
	max_rotation_speed_rad_s: number;
	default_speed_m_s: number;
	default_rotation_speed_rad_s: number;
	control_point_offset_m: number;
	table_z_m: number | null;
	supports_contact_stop: boolean;
	contact_stop_trusted: boolean;
	supports_home: boolean;
	gripper_gap_measured: boolean;
	gripper_open_span_m: number;
	gripper_pad_length_m: number;
	comfortable_reach_m: number;
	reach_m: number;
	tilt_limits_deg: { roll_cw: number; roll_ccw: number; pitch_down: number; pitch_up: number };
	tilted_floor_m: number;
	tilt_reach_deg: number;
	max_object_height_m: number;
	subgoal_median_s: number;
	subgoal_verdict_s: number;
	arrival_tolerance_m: number;
	notes: string[];
};

/** Every field `Caps` has: the env server's table must carry them all (its `caps` is the only source). */
export const CAPS_FIELDS: readonly (keyof Caps)[] = [
	"max_translation_m",
	"max_rotation_rad",
	"max_speed_m_s",
	"max_rotation_speed_rad_s",
	"default_speed_m_s",
	"default_rotation_speed_rad_s",
	"control_point_offset_m",
	"table_z_m",
	"supports_contact_stop",
	"contact_stop_trusted",
	"supports_home",
	"gripper_gap_measured",
	"gripper_open_span_m",
	"gripper_pad_length_m",
	"comfortable_reach_m",
	"reach_m",
	"tilt_limits_deg",
	"tilted_floor_m",
	"tilt_reach_deg",
	"max_object_height_m",
	"subgoal_median_s",
	"subgoal_verdict_s",
	"arrival_tolerance_m",
	"notes",
];

/** Speeds when neither the action nor the capabilities give one (MotorMind's own fallbacks). */
export const FALLBACK_SPEED_M_S = 0.0125;
export const FALLBACK_ROTATION_SPEED_RAD_S = (8 * Math.PI) / 180;
/** A contact push moves at most this far in one action (mm). */
export const MAX_PUSH_MM = 250;
/** One command's translations end at most this far from where it started (straight line, mm). */
export const MAX_COMPOSITE_MM = 400;
/** One command's rotations add up to at most this many degrees. */
export const MAX_TURN_DEG = 90;
/** A translation shortened to under this many mm is refused instead of run. */
export const MIN_FIT_MM = 1;
/** When already at the edge of the reach, a move may end this much further out (m). */
export const REACH_SLACK_M = 0.005;

export type ActionType = "move" | "rotate" | "gripper" | "home" | "wait";

/** Keys an action may inherit from the command's top level, and override. */
export type Common = {
	frame: string;
	speed_mm_s?: number;
	rot_speed_deg_s?: number;
	stop_on_contact?: boolean;
	note: string;
};

/** One parsed action of the command language: magnitudes are positive, units mm / deg / s. */
export type Action = Common & {
	index: number;
	type: ActionType;
	/** move / rotate: the direction word as written (trimmed, lower case), or an explicit base-frame axis. */
	direction?: string;
	axis?: Vec3;
	distance_mm?: number;
	angle_deg?: number;
	state?: "open" | "close";
	width_mm?: number;
	seconds?: number;
};

export type DeltaKind = "translate" | "rotate" | "gripper" | "wait" | "home";

/** One semantic motion in SI units (MotorMind's TcpDelta): what one `mm_act` call runs. */
export type Delta = {
	index: number;
	kind: DeltaKind;
	/** Unit axis in the base frame (translate, rotate). */
	axis?: Vec3;
	/** m (translate) or rad (rotate). */
	magnitude: number;
	/** m/s (translate) or rad/s (rotate). */
	speed: number;
	stop_on_contact: boolean;
	push: boolean;
	gripper_state?: "open" | "close";
	/** Executor only: open just this wide (mm), then hold the fingers there. */
	open_to_mm?: number;
	seconds?: number;
	label: string;
	note: string;
};

/** One segment the env server runs (`env.mm_run_step`); goals are computed before any of them runs. */
export type Segment = {
	kind: DeltaKind;
	/** 1-based within its delta, and the delta's segment count. */
	chunk: number;
	of: number;
	label: string;
	/** The pose at the segment's end (translate, rotate), base frame. */
	goal_position_m?: Vec3;
	goal_rotation?: Mat3;
	axis?: Vec3;
	distance_m?: number;
	angle_rad?: number;
	/** m/s or rad/s. */
	speed?: number;
	/** The largest delta one tick sends: min(per-tick limit, speed * DT), m or rad. */
	max_step?: number;
	push?: boolean;
	stop_on_contact?: boolean;
	state?: "open" | "close";
	open_to_mm?: number;
	seconds?: number;
	ticks?: number;
	duration_s: number;
};

/** done / contact / limited / refused / skipped, as the env server reports a segment; failed = the call raised. */
export type Outcome = "done" | "contact" | "limited" | "refused" | "skipped" | "failed";
/** `closed_fraction`: 0 at the closed end, 1 wide open (opening over the open span). */
export type Grasp = { holding: boolean | null; reason: string; closed_fraction?: number | null };

/** A segment as measured, normalized from the RPC reply (SI units). */
export type Measured = {
	kind: DeltaKind;
	label: string;
	outcome: Outcome;
	abort_reason: "" | "blocked" | "skidded";
	moved_m: number;
	along_axis_m: number;
	lateral_m: number;
	turned_rad: number;
	pose_after?: Pose;
	axis?: Vec3;
	ticks: number;
	/** Of `ticks`: the servo's own, and the zero-delta hold after an abort (the server's diagnostics). */
	servo_ticks?: number;
	hold_ticks?: number;
	elapsed_s: number;
	frame_clock_s?: number;
	opening_m?: number;
	asked_m?: number;
	grasp?: Grasp;
	episode_over: boolean;
	message: string;
	/** skipped: why (an earlier segment's outcome, a cancel, the deadline); failed: the error. */
	reason?: string;
	cancel_reason?: string;
	error?: string;
	timed_out?: boolean;
};

/** The aim marker's depth when neither the table nor the wrist clearance gives one: the caller's tracked clearance (m). */
export type Aim = { aim_tracked_m?: number };

/** The robot's side of the mode (`RobotSpec.motormind`): both run only while the robot lock is held. */
export type MotorMindSpec = {
	/** `env.mm_observe`: the observation of DESIGN §3.1 (static fields included). */
	observe: (o?: Aim) => Promise<Record<string, unknown>>;
	/**
	 * `env.mm_run_step` for one segment; `observe` is the observation after it: `state` (numbers) or
	 * `full` (with the camera images). A `home` that resets the scene starts the robot's episode over.
	 */
	runStep: (
		segment: Segment,
		o: { timeoutMs: number; observe: "state" | "full" } & Aim,
	) => Promise<Record<string, unknown>>;
	/**
	 * The env server configuration the robot's flags ask for (the observation's `config`, key by key):
	 * a server it attached to that runs otherwise fails the start.
	 */
	expect?: () => Record<string, unknown>;
	/** The env server's endpoint (`URL[#token=HEX]`), for tests that read privileged truth beside the robot. */
	endpoint?: () => string | undefined;
	/** The env server's privileged truth log (`mm_truth.jsonl`, one line per observation), read at the end. */
	truth?: () => string | undefined;
};
