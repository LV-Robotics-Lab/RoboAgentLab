/**
 * The executor: one subgoal as an async generator (spec-executor §7, §9, §12-§14; DESIGN §1.4 step
 * 4). Each cycle: the gates' state is reset, a look is taken at the cycle start, the target is
 * located when the rules say so, the robot report and the target blocks are written, the proposer is
 * asked (up to 3 tries on the same pictures), the menders fix the batch, and the batch runs one
 * `mm_act` per action with a fresh look and the gate chain before every action after the first. The
 * cancel flag is read before every `mm_act`; `spent` grows only after a batch actually ran. The
 * endings are the enum of the contract, never a string anyone parses.
 *
 * The generator yields robot calls (`MmCall`) and is resumed with their results (`MmReply`); it never
 * calls RPCs, tools or pi. Role calls go through the mission's `Ask`.
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import { adopt, carryOver } from "./carry.ts";
import { actionLabel } from "./command.ts";
import {
	ACT_REFUSED,
	type ActArgs,
	type Cycle,
	describeRobot,
	type Ending,
	type Executor,
	type Frame,
	images,
	jawsDown,
	type MmCall,
	type MmReply,
	type Proposal,
	type Reach,
	type RoleCallRecord,
	type SegmentReport,
	type SubgoalInput,
	type SubgoalRun,
} from "./contract.ts";
import { norm, sub } from "./frames.ts";
import {
	aboutLettingGo,
	actsOnWhatItNames,
	arrived,
	chainGates,
	closeOutcome,
	closeToleranceMm,
	dragging,
	FROZEN_CYCLES,
	finePhase,
	forbidText,
	type GateContext,
	isPositional,
	jawHalfGapMm,
	keepBlockedAxes,
	MAX_CYCLES,
	MIN_CLOSE_TOLERANCE_MM,
	namesEdgePart,
	newState,
	noMotion,
	noteBlocked,
	noteEmptyClose,
	noteGrasped,
	noteOpened,
	noteTurnStop,
	partOf,
	payloadBlock,
	payloadOffset,
	receiverWords,
	releaseAllowanceMm,
	residualOf,
	type SubgoalState,
	shoved,
	sidePartWords,
	sideTakeBlock,
	sinceLastFrame,
	sinceStart,
	whereTheTargetIs,
	whereTheToolIs,
} from "./gates.ts";
import { apartXy, longestAcross, measuresMore } from "./geometry.ts";
import { heldPixels, type Located, locate, WRIST_USEFUL_BELOW_MM } from "./locate.ts";
import { firstMoveIsATest, followDelta, letGoWidth, liftBeforeSideways, releaseWidthMm, splitACarry } from "./mend.ts";
import {
	type ClearanceRead,
	ClearanceTracker,
	capabilityText,
	type Looked,
	legend,
	type ObserveState,
	observeStep,
	TopFace,
	trackedMm,
} from "./observe.ts";
import { historyText, propose } from "./proposal.ts";
import { supervise } from "./supervision.ts";
import type { Caps, Delta, Vec3 } from "./types.ts";

export { MAX_CYCLES };

/** The `mm_act` arguments of a delta: the axis form (exact), with the executor's label and note. */
export function actArgs(d: Delta, trackedMmNow?: number): ActArgs {
	const tracked = trackedMmNow !== undefined ? { tracked_mm: trackedMmNow } : {};
	const note = d.note ? { note: d.note } : {};
	switch (d.kind) {
		case "translate":
			return {
				type: "move",
				axis: d.axis,
				distance_mm: d.magnitude * 1000,
				speed_mm_s: d.speed * 1000,
				...(d.push ? { stop_on_contact: true } : {}),
				label: d.label,
				...note,
				...tracked,
			};
		case "rotate":
			return {
				type: "rotate",
				axis: d.axis,
				angle_deg: (d.magnitude * 180) / Math.PI,
				rot_speed_deg_s: (d.speed * 180) / Math.PI,
				label: d.label,
				...note,
				...tracked,
			};
		case "gripper":
			return {
				type: "gripper",
				state: d.gripper_state,
				...(d.open_to_mm !== undefined ? { open_to_mm: d.open_to_mm } : {}),
				label: d.label,
				...note,
				...tracked,
			};
		case "wait":
			return { type: "wait", seconds: d.seconds, label: d.label, ...note, ...tracked };
		case "home":
			return { type: "home", label: d.label, ...note, ...tracked };
	}
}

const ENDING_TEXT: Record<Ending, string> = {
	HELD: "the jaws closed and report holding something",
	RELEASED_IN_CODE: "the code let go over the receiver",
	ARRIVED: "the tool stopped moving within the arrival threshold of the target",
	WALL: "the tool is frozen at the edge of the reach with something in the jaws",
	FROZEN: "the tool stopped moving and the subgoal is not achieved",
	MODEL_DONE: "the model says the subgoal is achieved",
	DONE_AFTER_LAST: "the model gave its last actions and said the subgoal is achieved",
	CANNOT_FROM_THE_SIDE: "the side take cannot be done",
	ERROR: "the executor stopped on an error",
	CANCELLED: "the attempt was cancelled",
	CYCLES_EXHAUSTED: "the subgoal's action budget is spent",
	EPISODE_OVER: "the episode is over",
};

const isDescent = (d: Delta) => d.kind === "translate" && !!d.axis && (d.axis as Vec3)[2] < -0.5;

/** Run one subgoal (spec-executor §7.2). */
export async function* runSubgoal(input: SubgoalInput): Executor {
	const started = Date.now();
	const { subgoal: sg, ask, clock, cache, onEvent } = input;
	const s: SubgoalState = newState({
		subgoal: sg.subgoal,
		criterion: sg.criterion,
		target: sg.target,
		planBoxes: sg.target_boxes,
	});
	const faces = new TopFace();
	const obs: ObserveState = { tracker: new ClearanceTracker(), faces };
	const observes: SubgoalRun["observes"] = [];
	const cycles: Cycle[] = [];
	const calls: RoleCallRecord[] = [];
	let motion = 0;
	let held: SubgoalRun["held"];
	let episodeOver = false;
	let preDescentLooked = false;
	let judgedUpTo = -Infinity;
	let prevFrame: Frame | undefined;
	let prevRead: ClearanceRead | undefined;
	const maxCycles = Math.max(1, Math.min(input.maxCycles, MAX_CYCLES));

	const carried = adopt(s, input.carry, faces);
	if (carried) onEvent({ type: "carried", text: carried });

	const finish = (ending: Ending, o: { error?: string; text?: string } = {}): SubgoalRun => ({
		cycles,
		ending,
		ending_text: o.text ?? ENDING_TEXT[ending],
		...(o.error ? { error: o.error } : {}),
		episode_over: episodeOver,
		carry: carryOver(s, faces),
		...(held ? { held } : {}),
		observes,
		timing: { wall_s: (Date.now() - started) / 1000, motion_s: motion, calls },
	});

	async function* drain(
		gen: AsyncGenerator<MmCall, Looked | undefined, MmReply>,
		why: string,
	): AsyncGenerator<MmCall, Looked | undefined, MmReply> {
		let next = await gen.next();
		while (!next.done) {
			const reply: MmReply = yield next.value;
			next = await gen.next(reply);
		}
		const looked = next.value;
		if (looked) onEvent({ type: "observed", why: why as never, frame_clock_s: looked.frame.frame_clock_s });
		return looked;
	}

	for (;;) {
		if (s.spent >= maxCycles) return finish("CYCLES_EXHAUSTED");
		if (clock.cancelled()) return finish("CANCELLED", { text: clock.cancelled()?.reason });
		s.atTheWall = false;
		s.blockedAxes = [];
		const cycleStart = Date.now();
		const cycleCalls: RoleCallRecord[] = [];
		let cycleMotion = 0;
		const cycle: Cycle = { index: cycles.length, steps: [], timing: { wall_s: 0, motion_s: 0, calls: cycleCalls } };
		const closeCycle = () => {
			cycle.timing = { wall_s: (Date.now() - cycleStart) / 1000, motion_s: cycleMotion, calls: cycleCalls };
			calls.push(...cycleCalls);
			onEvent({ type: "timing", cycle: cycle.index, timing: cycle.timing });
		};

		// 3. The look at the cycle start.
		const looked = yield* drain(
			observeStep(cache, observes, obs, "cycle_start", {
				outline: s.footprint,
				halfGapM: cache.latest ? jawHalfGapMm(cache.latest.caps) / 1000 : 0.0394,
			}),
			"cycle_start",
		);
		if (!looked) {
			cycles.push(cycle);
			cycle.error = "the robot answered the look with an error";
			onEvent({ type: "step_error", error: cycle.error });
			closeCycle();
			return finish("ERROR", { error: cycle.error });
		}
		let { frame, read } = looked;
		const caps: Caps = frame.caps;
		if (frame.episode_over) {
			episodeOver = true;
			onEvent({ type: "episode_over" });
			cycles.push(cycle);
			closeCycle();
			return finish("EPISODE_OVER");
		}
		if (clock.cancelled()) {
			cycles.push(cycle);
			closeCycle();
			return finish("CANCELLED", { text: clock.cancelled()?.reason });
		}
		cycles.push(cycle);
		s.startPose ??= frame.pose;
		s.heldAtStart = frame.grasp.holding;
		const batchStartFrame = frame;

		// 4. The robot report.
		const robotLines = [
			describeRobot(frame, read.tracked),
			sinceStart(s, frame.pose),
			sinceLastFrame(prevFrame, prevRead, frame, read),
		];
		prevFrame = frame;
		prevRead = read;

		// 5. The residual, then the target.
		s.residualMm = residualOf(s, frame.pose);
		const fine = finePhase(read, s.residualMm);
		const reach = await lookForTheTarget(frame, fine);
		if (reach) {
			cycle.reach = reach.reach;
			cycleCalls.push(...reach.records);
			onEvent({ type: "reached", cycle: cycle.index, reach: reach.reach });
		}
		const topZ = faces.topZ(frame.table_z_m);
		const heightMm = faces.heightMm() ?? TopFace.fallbackHeight(frame, read);
		const heightFrom = faces.faces.length
			? "the wrist's median of the faces it saw"
			: faces.fromTopPoint !== undefined
				? "where the two views' top edges met"
				: "the clearance under the tool, whatever is there";

		// 6. The blocks.
		robotLines.push(whereTheToolIs(s, frame, caps, heightMm, topZ, heightFrom, read));
		const target = whereTheTargetIs(s, frame, caps, topZ);
		if (target) robotLines.push(target);
		const side = sideTakeBlock(s, frame, caps);
		if (side.cannot) {
			cycle.stopped_by = `cannot take it from the side: ${side.cannot}`;
			closeCycle();
			return finish("CANNOT_FROM_THE_SIDE", { text: cycle.stopped_by });
		}
		if (side.text) robotLines.push(side.text);
		const payload = payloadBlock(s, caps, frame.pose);
		if (payload) robotLines.push(payload);

		// 8. The limits; 9. the proposal.
		const forbid = forbidText(s, sg.gripper_only, fine);
		const names = frame.camera_names.filter((n) => frame.cameras[n]?.marked512);
		const pics: ImageContent[] = images(frame, "marked512", names);
		const descentBlockedSoFar = false;
		const gateCtx = (running: boolean, f: Frame, r: ClearanceRead, descentBlocked: boolean): GateContext => ({
			state: s,
			frame: f,
			caps,
			read: r,
			running,
			heightMm: faces.heightMm() ?? TopFace.fallbackHeight(f, r),
			topZ: faces.topZ(f.table_z_m),
			descentBlocked,
		});
		const proposed = await propose({
			ask,
			subgoal: sg.subgoal,
			target: sg.target,
			criterion: sg.criterion,
			history: historyText(input.preface, cycles.slice(0, -1)),
			robot: robotLines.filter(Boolean).join("\n\n"),
			capabilities: capabilityText(caps),
			images: pics,
			imageNames: names,
			legend: legend(frame, read.tracked),
			forbid,
			caps,
			gate: (deltas) => {
				const why = chainGates(deltas, gateCtx(false, frame, read, descentBlockedSoFar));
				if (why) return why;
				firstMoveIsATest(deltas, s, caps, { running: false, gripperOnly: sg.gripper_only });
				return undefined;
			},
		});
		cycleCalls.push(...proposed.records);
		if (proposed.proposal) {
			cycle.proposal = proposed.proposal;
			onEvent({ type: "proposed", cycle: cycle.index, proposal: proposed.proposal });
		}

		// 10. No proposal: let go in code, else an error.
		if (!proposed.proposal) {
			const width = letGoWidth(s, frame, caps, read, {
				aboutLettingGo: aboutLettingGo(`${sg.subgoal} ${sg.criterion}`),
				descentBlocked: false,
			});
			if (width !== undefined && !clock.cancelled()) {
				const r = yield* letGo(width, frame, caps, cycle);
				cycleMotion += r.motion;
				motion += r.motion;
				closeCycle();
				if (r.ending) return finish(r.ending, r.error ? { error: r.error } : {});
			}
			cycle.error = proposed.error ?? "no proposal";
			onEvent({ type: "step_error", error: cycle.error });
			closeCycle();
			return finish("ERROR", { error: cycle.error });
		}
		const deltas = proposed.deltas;
		const claim = proposed.proposal.done;

		// 11. A claim with no actions.
		if (claim && !deltas.length) {
			closeCycle();
			if (s.inheritedFrozen > 0 && s.inheritedFrozen + s.frozen + 1 >= FROZEN_CYCLES) {
				cycle.stopped_by = "frozen: the tool had stopped moving before this claim";
				onEvent({ type: "frozen", text: cycle.stopped_by });
				return finish("FROZEN", { text: cycle.stopped_by });
			}
			cycle.stopped_by = "the model says the subgoal is achieved";
			return finish("MODEL_DONE");
		}
		// 12. Nothing to do and no claim.
		if (!deltas.length) {
			cycle.error = "the proposal has no actions and does not claim the subgoal done";
			onEvent({ type: "step_error", error: cycle.error });
			closeCycle();
			return finish("ERROR", { error: cycle.error });
		}

		// 13. The batch.
		const mended: string[] = [];
		const lifted = liftBeforeSideways(deltas, s, frame, caps, { gripperOnly: sg.gripper_only, heightMm });
		if (lifted) mended.push(lifted);
		if (splitACarry(deltas, s, caps)) mended.push("the carry was split into legs");
		for (const m of mended) onEvent({ type: "mended", what: m });
		const batch = { actions: deltas.length, ran: 0, motion_s: 0 } as NonNullable<Cycle["batch"]>;
		// The watcher judges only frames newer than the one before the batch (spec-executor P6).
		judgedUpTo = Math.max(judgedUpTo, batchStartFrame.frame_clock_s);
		cycle.batch = batch;
		let stopped: { at: number; reason: string } | undefined;
		let ended: { ending: Ending; error?: string } | undefined;
		let descentBlocked = false;
		let graspedThisCycle: { gap_mm: number } | undefined;
		let gripperChanged = false;
		let cancelledMid = false;
		for (let i = 0; i < deltas.length; i++) {
			const d = deltas[i];
			if (clock.cancelled()) {
				cancelledMid = true;
				break;
			}
			if (i > 0) {
				const l = yield* drain(
					observeStep(cache, observes, obs, "before_action", {
						outline: s.footprint,
						halfGapM: jawHalfGapMm(caps) / 1000,
					}),
					"before_action",
				);
				if (!l) {
					ended = { ending: "ERROR", error: "the robot answered the look with an error" };
					break;
				}
				({ frame, read } = l);
				s.residualMm = residualOf(s, frame.pose);
				if (frame.episode_over) {
					episodeOver = true;
					ended = { ending: "EPISODE_OVER" };
					break;
				}
				let why = chainGates([d], gateCtx(true, frame, read, descentBlocked));
				if (
					!why &&
					d.kind === "gripper" &&
					d.gripper_state === "close" &&
					s.relookArranged &&
					s.relookTrigger &&
					norm(sub(frame.pose.position_m, s.relookTrigger)) <= 0.008
				) {
					s.relookArranged = false;
					s.relookedAt.push([...frame.pose.position_m]);
					const r = yield* relook(frame, caps, cycle, "re-look before the close");
					if (r.ended) {
						ended = r.ended;
						break;
					}
					if (r.looked) ({ frame, read } = r.looked);
					cycleMotion += r.motion;
					motion += r.motion;
					why = chainGates([d], gateCtx(true, frame, read, descentBlocked));
				}
				if (why) {
					stopped = { at: i, reason: why };
					onEvent({ type: "batch_stopped", cycle: cycle.index, at: i, reason: why });
					break;
				}
			} else if (
				d.kind === "gripper" &&
				d.gripper_state === "close" &&
				s.relookArranged &&
				s.relookTrigger &&
				norm(sub(frame.pose.position_m, s.relookTrigger)) <= 0.008
			) {
				s.relookArranged = false;
				s.relookedAt.push([...frame.pose.position_m]);
				const r = yield* relook(frame, caps, cycle, "re-look before the close");
				if (r.ended) {
					ended = r.ended;
					break;
				}
				if (r.looked) ({ frame, read } = r.looked);
				cycleMotion += r.motion;
				motion += r.motion;
				const why = chainGates([d], gateCtx(true, frame, read, descentBlocked));
				if (why) {
					stopped = { at: i, reason: why };
					onEvent({ type: "batch_stopped", cycle: cycle.index, at: i, reason: why });
					break;
				}
			}
			// The pre-descent wrist look (spec-executor §9), once per subgoal.
			if (
				isDescent(d) &&
				!d.push &&
				s.wantsHolding &&
				!preDescentLooked &&
				s.located &&
				jawsDown(frame.pose) &&
				frame.table_z_m !== null &&
				(frame.pose.position_m[2] - frame.table_z_m) * 1000 < WRIST_USEFUL_BELOW_MM
			) {
				preDescentLooked = true;
				const r = yield* relook(frame, caps, cycle, "the look before the descent");
				if (r.ended) {
					ended = r.ended;
					break;
				}
				if (r.looked) ({ frame, read } = r.looked);
				cycleMotion += r.motion;
				motion += r.motion;
			}
			// A model's open while something is held opens only as wide as the payload plus a margin.
			if (d.kind === "gripper" && d.gripper_state === "open" && s.payload.tracking && frame.grasp.holding === true) {
				const w = releaseWidthMm(s, caps);
				if (w !== undefined) d.open_to_mm = w;
			}
			if (clock.cancelled()) {
				cancelledMid = true;
				break;
			}
			onEvent({ type: "step_start", label: d.label });
			const reply: MmReply = yield { tool: "mm_act", args: actArgs(d, trackedMm(obs)), delta: d, label: d.label };
			if (!reply.ok) {
				if ((reply.error ?? "").startsWith(ACT_REFUSED)) {
					stopped = { at: i, reason: reply.error ?? "refused" };
					onEvent({ type: "refused", why: stopped.reason });
					break;
				}
				ended = { ending: "ERROR", error: reply.error ?? "the robot answered the action with an error" };
				onEvent({ type: "step_error", error: ended.error ?? "" });
				break;
			}
			batch.ran++;
			const steps = (reply.segments ?? []).map((st) => ({ ...st, ...(d.note ? { note: d.note } : {}) }));
			for (const st of steps) {
				cycle.steps.push(st);
				cycleMotion += st.elapsed_s;
				motion += st.elapsed_s;
				batch.motion_s += st.elapsed_s;
				onEvent({ type: "step_end", cycle: cycle.index, step: st });
				if (st.kind === "translate" && st.abort_reason === "blocked" && st.axis && st.axis[2] < -0.5)
					descentBlocked = true;
				noteBlocked(s, st);
				if (st.kind === "gripper" && st.outcome === "done") {
					gripperChanged = true;
					if (d.gripper_state === "close") {
						const holding = closeOutcome(st, s);
						const gap = st.detail?.opening_mm ?? st.grasp?.opening_mm;
						const topZNow = faces.topZ(frame.table_z_m);
						if (holding === false)
							noteEmptyClose(
								s,
								st,
								s.wantsHolding ||
									namesEdgePart(sg.subgoal) ||
									topZNow === undefined ||
									(frame.pose.position_m[2] - topZNow) * 1000 <= 100,
							);
						else if (holding === true) {
							noteGrasped(s, st, caps, frame.pose, namesEdgePart(sg.subgoal));
							graspedThisCycle = { gap_mm: gap ?? 0 };
						}
					} else {
						if (s.payload.tracking && s.receiver) {
							const off = payloadOffset(s, frame.pose.position_m);
							cycle.release = {
								...(s.payload.width_mm !== undefined ? { payload_width_mm: s.payload.width_mm } : {}),
								...(releaseAllowanceMm(s, caps) !== undefined
									? { allowed_mm: releaseAllowanceMm(s, caps) }
									: {}),
								...(off ? { offset_mm: Math.hypot(off[0], off[1]) * 1000 } : {}),
								placing_in: (s.criterion.match(/\b(in|into|inside)\b/i) ?? null) !== null,
								receiver: s.receiver.words,
							};
						}
						noteOpened(s);
					}
				}
			}
			if (reply.frame) frame = reply.frame;
			if (reply.timed_out) {
				ended = { ending: "ERROR", error: "a segment's outcome is unknown (the RPC timed out)" };
				onEvent({ type: "step_error", error: ended.error ?? "" });
				break;
			}
			if (reply.episode_over || episodeOver) {
				episodeOver = true;
				onEvent({ type: "episode_over" });
				ended = { ending: "EPISODE_OVER" };
				break;
			}
			if (reply.cancelled) {
				cancelledMid = true;
				break;
			}
			const last = steps.at(-1);
			if (last && (dragging(last, frame, caps) || !["done", "limited", "contact"].includes(last.outcome))) {
				stopped = {
					at: i,
					reason: dragging(last, frame, caps)
						? "the payload is dragging: a sideways move was blocked while holding"
						: `${last.label} ended ${last.outcome}`,
				};
				onEvent({ type: "batch_stopped", cycle: cycle.index, at: i, reason: stopped.reason });
				if (last.outcome === "limited" && last.abort_reason) onEvent({ type: "limited", why: stopped.reason });
				break;
			}
		}
		if (batch.ran > 0) s.spent++;
		if (stopped) {
			batch.stopped_at = stopped.at;
			batch.reason = stopped.reason;
		}
		if (s.located) batch.residual_mm = residualOf(s, frame.pose);

		// After a stopped batch: look once more and ask the watcher.
		if (stopped && !ended && !cancelledMid && batch.ran > 0) {
			const l = yield* drain(
				observeStep(cache, observes, obs, "after_stop", {
					outline: s.footprint,
					halfGapM: jawHalfGapMm(caps) / 1000,
				}),
				"after_stop",
			);
			if (l) {
				({ frame, read } = l);
				const residualNow = residualOf(s, frame.pose);
				if (residualNow !== undefined && batch.residual_mm !== undefined && residualNow > batch.residual_mm + 1)
					batch.reason += " (the residual grew)";
				const sv = await supervise({
					ask,
					subgoal: sg.subgoal,
					deltas,
					expect: proposed.proposal.expect,
					steps: cycle.steps,
					before: batchStartFrame,
					after: frame,
					legend: legend(frame, read.tracked),
					judgedUpTo,
				});
				if (sv) {
					judgedUpTo = sv.judged;
					if (sv.record) cycleCalls.push(sv.record);
					cycle.supervision = sv.supervision;
					cycle.finding = sv.supervision.finding;
					onEvent({ type: "supervision", cycle: cycle.index, supervision: sv.supervision });
				}
			}
		}
		s.lastFinding = cycle.finding;
		s.lastFirstMove = deltas.find((d) => d.kind === "translate");
		if (stopped) cycle.stopped_by = stopped.reason;

		// The cycle's measurements (spec-executor §7.2 step 13).
		const holdingNow = frame.grasp.holding === true;
		keepBlockedAxes(s, cycle.steps, holdingNow);
		noteTurnStop(s, deltas, cycle.steps, holdingNow, cycle.finding);
		if (shoved(cycle.steps, deltas, s.errorMm)) s.unshoved = false;
		if (holdingNow && cycle.steps.some((st) => st.kind === "translate" && st.abort_reason === "blocked"))
			s.payload.bumped = true;
		const motionCheck = noMotion(cycle.steps, gripperChanged);
		if (motionCheck.frozen) {
			if (s.atTheWall && holdingNow) s.wall++;
			else s.frozen++;
		} else {
			s.frozen = 0;
			s.wall = 0;
		}
		closeCycle();

		// 14. The exit conditions.
		if (ended) return finish(ended.ending, ended.error ? { error: ended.error } : {});
		if (cancelledMid) return finish("CANCELLED", { text: clock.cancelled()?.reason ?? "cancelled" });
		if (s.wantsHolding && graspedThisCycle && s.heldAtStart !== true) {
			held = {
				gap_mm: graspedThisCycle.gap_mm,
				...(s.payload.span_mm !== undefined ? { target_mm: s.payload.span_mm } : {}),
			};
			onEvent({ type: "held", ...held });
			cycle.stopped_by = `held in code: the jaws settled ${held.gap_mm.toFixed(1)} mm apart${held.target_mm !== undefined ? ` on something about ${held.target_mm.toFixed(0)} mm across` : ""}`;
			if (cycle.proposal) cycle.proposal.done = true;
			return finish("HELD", { text: cycle.stopped_by });
		}
		if (s.frozen + s.inheritedFrozen >= FROZEN_CYCLES || s.wall >= FROZEN_CYCLES) {
			const width = letGoWidth(s, frame, caps, read, {
				aboutLettingGo: aboutLettingGo(`${sg.subgoal} ${sg.criterion}`),
				descentBlocked,
			});
			if (width !== undefined && !clock.cancelled()) {
				const r = yield* letGo(width, frame, caps, cycle);
				motion += r.motion;
				if (r.ending) return finish(r.ending, r.error ? { error: r.error } : {});
			}
			const positional = isPositional(sg.criterion);
			const a = positional ? arrived(s, caps, frame.pose) : undefined;
			if (s.wall >= FROZEN_CYCLES && !a?.arrived) {
				cycle.stopped_by = `frozen at the wall: ${motionCheck.why}; the target is out of reach from here`;
				onEvent({ type: "frozen", text: cycle.stopped_by });
				return finish("WALL", { text: cycle.stopped_by });
			}
			if (a && !a.arrived && !s.askedPosition) {
				s.askedPosition = true;
				cycle.stopped_by = `the tool stopped moving ${a.residual_mm !== undefined ? `${Math.round(a.residual_mm)} mm` : "an unknown distance"} from the target (threshold ${Math.round(a.threshold_mm)} mm): is the subgoal achieved, or what remains?`;
				s.frozen = 0;
				continue;
			}
			if (a?.arrived) {
				cycle.stopped_by = `arrived: the tool is ${Math.round(a.residual_mm ?? 0)} mm from the target, within ${Math.round(a.threshold_mm)} mm, and nothing was shoved since the look`;
				return finish("ARRIVED", { text: cycle.stopped_by });
			}
			cycle.stopped_by = `frozen: ${motionCheck.why}`;
			onEvent({ type: "frozen", text: cycle.stopped_by });
			return finish("FROZEN", { text: cycle.stopped_by });
		}
		if (claim && !stopped) return finish("DONE_AFTER_LAST");
	}

	// ---- helpers that yield ----------------------------------------------------------------

	/** The code lets go (spec-executor §11.5): one open to `width`, then RELEASED_IN_CODE. */
	async function* letGo(
		width: number,
		frame: Frame,
		caps: Caps,
		cycle: Cycle,
	): AsyncGenerator<MmCall, { ending?: Ending; error?: string; motion: number }, MmReply> {
		const label = actionLabel({ type: "gripper", state: "open", width_mm: width, frame: "base" });
		onEvent({ type: "letting_go", text: `opening to ${width.toFixed(0)} mm over the receiver` });
		const reply: MmReply = yield {
			tool: "mm_act",
			args: {
				type: "gripper",
				state: "open",
				open_to_mm: width,
				label,
				...(trackedMm(obs) !== undefined ? { tracked_mm: trackedMm(obs) } : {}),
			},
			label,
		};
		let m = 0;
		if (!reply.ok) return { ending: "ERROR", error: reply.error ?? "the open failed", motion: 0 };
		for (const st of reply.segments ?? []) {
			cycle.steps.push(st);
			m += st.elapsed_s;
			onEvent({ type: "step_end", cycle: cycle.index, step: st });
		}
		const off = payloadOffset(s, frame.pose.position_m);
		cycle.release = {
			...(s.payload.width_mm !== undefined ? { payload_width_mm: s.payload.width_mm } : {}),
			...(releaseAllowanceMm(s, caps) !== undefined ? { allowed_mm: releaseAllowanceMm(s, caps) } : {}),
			...(off ? { offset_mm: Math.hypot(off[0], off[1]) * 1000 } : {}),
			placing_in: /\b(in|into|inside)\b/i.test(s.criterion),
			...(s.receiver ? { receiver: s.receiver.words } : {}),
		};
		s.releasedInCode = { width_mm: width };
		noteOpened(s);
		if (reply.episode_over) episodeOver = true;
		cycle.stopped_by = `released in code: opened to ${width.toFixed(0)} mm`;
		return { ending: "RELEASED_IN_CODE", motion: m };
	}

	/**
	 * A wrist-only look that may be followed by a horizontal move (spec-executor §9): the shift
	 * between the new and the old point is applied when within [max(1 mm, error), the jaw half-gap].
	 */
	async function* relook(
		frame: Frame,
		caps: Caps,
		cycle: Cycle,
		why: string,
	): AsyncGenerator<MmCall, { looked?: Looked; ended?: { ending: Ending; error?: string }; motion: number }, MmReply> {
		const before = s.located?.point;
		const r = await lookForTheTarget(frame, true, { wristOnly: true });
		if (r) {
			cycleCalls_push(cycle, r.records);
			// The cycle keeps its first look; a wrist look is reported as an event.
			cycle.reach ??= r.reach;
			onEvent({ type: "reached", cycle: cycle.index, reach: r.reach });
		}
		let motion = 0;
		const after = s.located?.point;
		if (before && after) {
			const shift: [number, number] = [after[0] - before[0], after[1] - before[1]];
			const mm = Math.hypot(shift[0], shift[1]) * 1000;
			const lo = Math.max(1, s.located?.error_mm ?? 0);
			if (mm >= lo && mm <= jawHalfGapMm(caps)) {
				if (clock.cancelled()) return { motion };
				const d = followDelta(shift, caps);
				onEvent({ type: "step_start", label: d.label });
				const reply: MmReply = yield { tool: "mm_act", args: actArgs(d, trackedMm(obs)), delta: d, label: d.label };
				if (!reply.ok)
					return { ended: { ending: "ERROR", error: reply.error ?? "the follow move failed" }, motion };
				for (const st of reply.segments ?? []) {
					cycle.steps.push({ ...st, note: d.note });
					motion += st.elapsed_s;
					onEvent({ type: "step_end", cycle: cycle.index, step: st });
				}
				if (reply.episode_over) {
					episodeOver = true;
					return { ended: { ending: "EPISODE_OVER" }, motion };
				}
				onEvent({ type: "mended", what: `followed the look by ${mm.toFixed(0)} mm (${why})` });
			}
		}
		const looked = yield* drain(
			observeStep(cache, observes, obs, "after_look", { outline: s.footprint, halfGapM: jawHalfGapMm(caps) / 1000 }),
			"after_look",
		);
		if (!looked) return { ended: { ending: "ERROR", error: "the robot answered the look with an error" }, motion };
		s.residualMm = residualOf(s, looked.frame.pose);
		return { looked, motion };
	}
	function cycleCalls_push(cycle: Cycle, records: RoleCallRecord[]) {
		cycle.timing.calls.push(...records);
	}

	// ---- locating (spec-executor §9) ---------------------------------------------------------

	/** Whether and how to look for the target this cycle; runs locate and updates the state. */
	async function lookForTheTarget(
		frame: Frame,
		fine: boolean,
		o: { wristOnly?: boolean } = {},
	): Promise<{ reach: Reach; records: RoleCallRecord[] } | undefined> {
		const caps = frame.caps;
		const words = s.payload.tracking
			? receiverWords(sg.criterion, sg.subgoal, sg.target)
			: s.sideStep
				? sidePartWords(sg.subgoal, sg.target)
				: sg.target || sg.subgoal;
		let why: string | undefined;
		if (o.wristOnly) why = "a wrist look";
		else if (!s.located || s.lastLookedWords !== words || (s.sideStep && !s.located.on_face)) why = "not located yet";
		else if (!s.unshoved) why = "something was shoved since the last look";
		else if (fine && !s.lookedClose && frame.cameras.wrist && jawsDown(frame.pose))
			why = "the fine phase: one close-in look";
		else if (
			fine &&
			s.lastLookPose &&
			apartXy(frame.pose.position_m, s.lastLookPose.position_m) * 1000 >
				Math.max(MIN_CLOSE_TOLERANCE_MM, closeToleranceMm(caps, undefined))
		)
			why = "the tool moved further than a close can take since the last look";
		if (!why) return undefined;
		const useBoxes =
			!o.wristOnly && !s.located && !fine && Object.keys(s.planBoxes).length > 0 && why === "not located yet";
		const views: typeof frame.cameras = { ...frame.cameras };
		if (s.sideStep || !jawsDown(frame.pose)) delete views.wrist;
		const wrist = frame.cameras.wrist;
		const heldPx =
			s.payload.tracking && wrist && s.payload.width_mm !== undefined
				? heldPixels(wrist, frame.pose.position_m, s.payload.width_mm / 1000, [
						frame.pose.rotation[0][2],
						frame.pose.rotation[1][2],
						frame.pose.rotation[2][2],
					])
				: undefined;
		const acts = actsOnWhatItNames(sg.subgoal, sg.criterion);
		const located: Located = await locate({
			views,
			words,
			table_z_m: frame.table_z_m,
			tool_z_m: frame.pose.position_m[2],
			top_z_m: faces.topZ(frame.table_z_m),
			part: !!acts,
			...(acts && s.wantsHolding ? { part_word: partOf(sg.subgoal, sg.target) } : {}),
			...(useBoxes ? { boxes: s.planBoxes } : {}),
			...(s.located && s.lastLookedWords === words ? { last_seen: s.located.point, hint_boxes: s.planBoxes } : {}),
			...(o.wristOnly ? { wrist_only: true } : {}),
			wrist_within_mm: Math.max(2 * jawHalfGapMm(caps), s.located?.error_mm ?? 0),
			...(heldPx ? { held_px: heldPx } : {}),
			tallest_mm: caps.max_object_height_m * 1000,
			holding: frame.grasp.holding === true,
			ask,
		});
		const reach: Reach = { target: words, from_plan_boxes: useBoxes && located.from_boxes.length > 0 };
		if (!located.ok || !located.point) {
			reach.located = { doubtful: located.doubtful, error: located.error ?? "not found" };
			return { reach, records: located.records };
		}
		s.fromPlanBoxes = reach.from_plan_boxes;
		let point = located.point;
		let disagreement = located.disagreement;
		let onFace = false;
		if (s.sideStep) {
			if (located.part_point) {
				point = located.part_point;
				disagreement = undefined;
				onFace = true;
			} else if (s.located?.on_face) {
				reach.located = { doubtful: true, error: "no two views met on its face" };
				return { reach, records: located.records };
			}
		}
		// The sticky rule (spec-locate §8.3).
		const old = s.located && s.lastLookedWords === words ? s.located : undefined;
		const emptyNear = old ? s.emptyCloses.some((p) => apartXy(p, old.point) * 1000 <= old.error_mm) : false;
		const override = o.wristOnly || (emptyNear && old && apartXy(point, old.point) * 1000 <= old.error_mm);
		let method = located.method;
		let jumped: number | undefined;
		if (old && !override) {
			const confirms = old.waiting && apartXy(point, old.waiting) * 1000 <= Math.max(located.error_mm ?? 0, 1);
			if (!((located.error_mm ?? Infinity) <= old.error_mm || confirms)) {
				jumped = apartXy(point, old.point) * 1000;
				old.waiting = point;
				method = `${old.method}; a new answer ${Math.round(jumped)} mm away with a wider bar (${Math.round(located.error_mm ?? 0)} vs ${Math.round(old.error_mm)} mm) was not taken`;
				point = old.point;
			}
		}
		if (point !== old?.point) {
			s.located = {
				words,
				point,
				method,
				error_mm: located.error_mm ?? 20,
				error_class: located.error_class ?? "",
				on_face: onFace,
				...(located.part_body ? { body: located.part_body } : {}),
				...(located.top_point ? { top: located.top_point } : {}),
				cameras: located.cameras,
			};
			s.errorMm = s.located.error_mm;
			s.errorClass = s.located.error_class;
		} else if (old) old.method = method;
		if (
			located.footprint &&
			measuresMore(located.footprint, s.footprint, located.cameras.includes("wrist"), frame.grasp.holding === true)
		) {
			s.footprint = located.footprint;
			s.footprintMethod = located.footprint_method;
			s.footprintError = located.error_mm;
			s.lastWidthMm = longestAcross(located.footprint) * 1000 || s.lastWidthMm;
		}
		s.disagreement = disagreement;
		s.ambiguous = located.ambiguous;
		if (s.payload.tracking) {
			const radius =
				s.footprint && s.footprint.length >= 2 ? (longestAcross(s.footprint) * 1000) / 2 : (located.error_mm ?? 20);
			s.receiver = { center: point, radius_mm: radius, error_mm: located.error_mm ?? 20, words };
		}
		if (located.top_point) faces.offerTopPoint(located.top_point, frame.table_z_m);
		s.residualMm = residualOf(s, frame.pose);
		s.lastLookPose = frame.pose;
		s.lastLookedWords = words;
		s.unshoved = true;
		if (fine && !o.wristOnly) s.lookedClose = true;
		reach.residual_mm = s.residualMm;
		reach.located = {
			label: located.label,
			point_mm: [point[0] * 1000, point[1] * 1000, point[2] * 1000],
			error_mm: s.located?.error_mm ?? located.error_mm ?? 20,
			doubtful: located.doubtful,
			...(jumped !== undefined ? { error: undefined } : {}),
		};
		return { reach, records: located.records };
	}
}

export type { Proposal, SegmentReport };
