/**
 * The MotorMind mission (spec-planner §3): plan once, then run the subgoals one by one through the
 * executor, judge each attempt (code gates first, the verifier otherwise), and choose between going
 * on, one retry as written, backing up to the grasp step, or a replan; a whole-task check ends it.
 * Clauses are kept in a ledger (./clauses.ts); net motion since the last grasp is summed from the
 * step labels' first direction word, as the original does, with the structured sum beside it for
 * the record (DESIGN §2.2).
 *
 * It is an async generator (./contract.ts `Mission`): it yields one robot call at a time and is
 * resumed with the result; the provider (./provider.ts) plays it one call per pi turn. It never
 * touches pi: the entries it wants written go through `log`.
 */

import {
	allMet,
	type ClauseState,
	clausesResult,
	codeEvent,
	emptyClauses,
	fold,
	isTurnClause,
	measuredHit,
	measuredMiss,
	PUSHED_ALONG_MM,
	TURNED_DEG,
	unmet,
} from "./clauses.ts";
import {
	type Ask,
	type Carry,
	type Cycle,
	type Describers,
	describeRobot,
	type Executor,
	type ExecutorEvent,
	type Frame,
	type FrameCache,
	type Json,
	jawsDown,
	look,
	type Mission,
	type MissionClock,
	type MissionEnd,
	type MissionStatus,
	type MmCall,
	type MmReply,
	type ObserveWhy,
	type RoleCallRecord,
	type SegmentReport,
	type Subgoal,
	type SubgoalInput,
	type SubgoalRun,
} from "./contract.ts";
import { EvidenceLog } from "./evidence.ts";
import { labelDirection } from "./frames.ts";
import {
	acts,
	describePlan,
	plan as firstPlan,
	type Plan,
	type PlanOutcome,
	replan,
	sameObject,
	type Verdicted,
	wantsHolding,
	wantsRelease,
} from "./planner.ts";
import type { Vec3 } from "./types.ts";
import { askTask, type Claim, type Verdict, verify } from "./verify.ts";

/** spec-planner §3.1, §9 */
export const DEFAULT_BUDGET = 10;
export const DEFAULT_MAX_REPLANS = 10;
export const MISSION_BUDGET_S = 360;
export const REPLAN_NEEDS_S = 40;
export const SUBGOAL_MEDIAN_S = 21;
export const SUBGOAL_TAX_S = 4.6;
export const NEVER_MOVED_MM = 3;
export const NO_TURN_DEG = 0.5;
export const SETTLES_WITHIN = 1.5;
/** The executor's hard cap on cycles per subgoal (spec-executor §7.1). */
export const MAX_CYCLES = 30;
/** The jaw resolution: an opening wider than the payload by this is a release (spec-executor §18 EMPTY_BAND / 4%). */
export const EMPTY_BAND_MM = 3;
export const EMPTY_BAND_SHARE = 0.04;

export type MissionInput = {
	task: string;
	budget: number;
	budgetS: number;
	maxReplans: number;
	ask: Ask;
	clock: MissionClock;
	cache: FrameCache;
	describers: Describers;
	/** The executor half's generator for one subgoal attempt. */
	runSubgoal: (input: SubgoalInput) => Executor;
	/** Entries for the session (`motormind_plan`, `motormind_verdict`, `motormind_cycle`). */
	log: (entry: string, data: Json) => void;
	/** The memory note's current text (M3); "" without memory. */
	note?: () => string;
	/** After every verdict: ask the summarizer (M3). */
	requestNote?: () => void;
};

export type AttemptRecord = {
	name: string;
	attempt: number;
	cycles: number;
	ending: string;
	verdict?: { status: string; because: string; gate: string; replan: boolean };
	never_moved: boolean;
	stopped_by_monitor: boolean;
	elapsed_s: number;
	obs_id?: number;
	net_motion: { label: { move_mm: number; turn_deg: number }; structured: { move_mm: number; turn_deg: number } };
};

export type MissionState = {
	status?: MissionStatus;
	ended_on?: string;
	plans: number;
	replans: number;
	subgoals: AttemptRecord[];
	clauses: ReturnType<typeof clausesResult>;
	net_motion_disagreements: number;
	actions: number;
	actions_cap: number;
	budget: number;
	budget_s: number;
	elapsed_s: number;
	left_s: number;
	plan_calls_s: number[];
	no_new_steps: number;
	backtracked: string[];
	observes: { why: ObserveWhy; yielded: boolean }[];
	replan_refused_for_price?: Json;
};

type Verdicts = Map<string, { status: string; because: string; attempt: number }>;
type NetMotion = { move: Vec3; turn: number; moveTotal: number; turnTotal: number; turnStop?: string };
const zeroNet = (): NetMotion => ({ move: [0, 0, 0], turn: 0, moveTotal: 0, turnTotal: 0 });
const norm = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
const r1 = (v: number) => Number(v.toFixed(1));
const r3 = (v: number) => Number(v.toFixed(3));

/** The ending's one-line description for the claim (spec-planner §2.8 `_how_it_ended`). */
function howItEnded(run: SubgoalRun, neverMoved: boolean): string {
	if (neverMoved) return "never moved";
	switch (run.ending) {
		case "MODEL_DONE":
			return `the executor said the step was achieved after ${run.cycles.length} cycle(s)`;
		case "DONE_AFTER_LAST":
			return "the executor finished its last action and said the step was achieved";
		case "HELD":
			return "the jaws closed on something and the robot reports holding";
		case "RELEASED_IN_CODE":
			return "the code opened the jaws over the place";
		case "ARRIVED":
			return "the tool stopped at the measured place";
		case "WALL":
			return "the tool froze at the edge of its reach while holding";
		case "FROZEN":
			return "the tool froze: two cycles without motion";
		case "CANNOT_FROM_THE_SIDE":
			return "the side approach cannot be made here";
		case "CYCLES_EXHAUSTED":
			return `the step's ${run.cycles.length} cycles were used up`;
		case "ERROR":
			return `the executor stopped on an error: ${run.error ?? run.ending_text}`;
		default:
			return run.ending_text || run.ending.toLowerCase();
	}
}

export function runMission(m: MissionInput): Mission & { state: () => MissionState } {
	const state: MissionState = {
		plans: 0,
		replans: 0,
		subgoals: [],
		clauses: clausesResult(emptyClauses()),
		net_motion_disagreements: 0,
		actions: 0,
		actions_cap: 0,
		budget: m.budget,
		budget_s: m.budgetS,
		elapsed_s: 0,
		left_s: m.budgetS,
		plan_calls_s: [],
		no_new_steps: 0,
		backtracked: [],
		observes: [],
	};
	const gen = play();
	return Object.assign(gen, { state: () => state }) as Mission & { state: () => MissionState };

	async function* play(): Mission {
		const evidence = new EvidenceLog();
		const verdicts: Verdicts = new Map();
		const note = () => m.note?.() ?? "";
		let startedAt: number | undefined;
		const elapsed = () => (startedAt === undefined ? 0 : (Date.now() - startedAt) / 1000);
		const end = (status: MissionStatus, ended_on: string, summary: string): MissionEnd => {
			state.status = status;
			state.ended_on = ended_on;
			state.elapsed_s = r1(elapsed());
			state.left_s = r1(Math.max(0, m.clock.left_s()));
			return { status, ended_on, summary };
		};
		const robotText = (f: Frame) => describeRobot(f);

		// 2. the opening picture and the first plan
		const opened = yield* look(m.cache, "opening", state.observes);
		if (!opened) return end("error", "no_opening_observation", "the robot gave no opening observation");
		const picture: Frame = opened;
		startedAt = Date.now();
		m.clock.deadline_at = startedAt + m.budgetS * 1000;
		let plan: Plan;
		let ledger: ClauseState;
		{
			const t0 = Date.now();
			const p = await firstPlan(m.ask, {
				task: m.task,
				frame: picture,
				robotText: robotText(picture),
				capabilities: m.describers.capabilities(picture.caps),
				legend: m.describers.legend(picture) || undefined,
				budget: m.budget,
			});
			state.plan_calls_s.push((Date.now() - t0) / 1000);
			state.plans++;
			logPlan("plan", p, "first plan");
			if (!p.plan) return end("error", "plan_failed", p.error ?? "the first plan failed");
			plan = p.plan;
			ledger = emptyClauses(plan.clauses);
			state.clauses = clausesResult(ledger);
		}
		let index = 0;
		let attempt = 1;
		let carry: Carry | undefined;
		let lastNow: Frame | undefined;
		let holdAt: number | undefined;
		let net = zeroNet();
		let netStructured = zeroNet();
		const marks = new Map<string, Vec3>();
		const replansOf = new Map<string, number>();
		const triedWays = new Map<string, string[]>();
		const failures = new Map<string, string[]>();
		const backtracked = new Set<string>();
		const cannot: string[] = [];
		const firstDuration = new Map<string, number>();
		const clauseStarted = new Map<number, number>();
		const retryPreface = new Map<string, string>();
		let askedWholeOnce = false;
		state.actions_cap = 2 * plan.subgoals.reduce((a, s) => a + s.max_cycles, 0);

		for (;;) {
			const cancel = m.clock.cancelled();
			if (cancel?.by === "mission") return end("cancelled", "cancelled", cancel.reason);
			if (cancel?.by === "deadline" || m.clock.left_s() <= 0)
				return end("timeout", "clock_ran_out", "the mission clock ran out");
			// the plan ran out: the whole-task check
			if (index >= plan.subgoals.length) {
				const check = yield* wholeTask(plan, picture, "every step of the plan was judged done", false);
				if (check.outcome) return check.outcome;
				if (check.replan) {
					const r = yield* doReplan(
						plan,
						plan.subgoals.length,
						"every step of the plan was judged done, but the task is not complete",
						false,
					);
					if (r.outcome) return r.outcome;
					plan = r.plan;
					index = r.index;
					attempt = 1;
				}
				continue;
			}
			if (state.actions >= state.actions_cap)
				return end(
					"exhausted",
					"actions_exhausted",
					`${state.actions} actions used; the cap was ${state.actions_cap}`,
				);
			const subgoal = plan.subgoals[index];
			if (!clauseStarted.has(subgoal.clause) && subgoal.clause >= 0) clauseStarted.set(subgoal.clause, Date.now());
			// 3. one attempt
			const a = yield* attemptOnce(subgoal, attempt);
			state.actions += a.run.cycles.filter((c) => c.steps.length).length;
			if (a.run.ending === "EPISODE_OVER" || a.run.episode_over) {
				const check = yield* wholeTask(
					plan,
					picture,
					"the robot cannot move any more; judge the scene as it is",
					true,
				);
				if (check.outcome) return check.outcome;
				return end("exhausted", "episode_over", "the episode ended before the task was complete");
			}
			// only a robot-layer error reaches here without a verdict (attemptOnce step 8)
			if (a.run.ending === "ERROR" && !a.verdict)
				return end("error", "executor_error", a.run.error ?? a.run.ending_text);
			if (!a.verdict) {
				const c = m.clock.cancelled();
				if (c?.by === "mission") return end("cancelled", "cancelled", c.reason);
				return end("timeout", "clock_ran_out", "the mission clock ran out during a step");
			}
			const v = a.verdict;
			// 4. fold the verdict
			noteSettled(subgoal, a.run);
			fold(ledger, v.clauses, v.status === "task_done", subgoal.criterion);
			if (v.asked) fold(ledger, v.asked.clauses, true);
			state.clauses = clausesResult(ledger);
			let status = v.status;
			if (status === "task_done") status = "done";
			verdicts.set(subgoal.name, { status, because: v.because, attempt });
			if (!firstDuration.has(subgoal.name)) firstDuration.set(subgoal.name, a.elapsed_s);
			if (status === "abort") return end("aborted", "verdict_abort", v.because);
			if (m.clock.left_s() <= 0) return end("timeout", "clock_ran_out", "the mission clock ran out");
			if (status === "done") {
				index++;
				attempt = 1;
				continue;
			}
			// not_done
			const fails = failures.get(subgoal.target || subgoal.name) ?? [];
			fails.push(`${subgoal.name} (attempt ${attempt}): ${v.because}`);
			failures.set(subgoal.target || subgoal.name, fails);
			if (!v.replan && attempt === 1 && !a.never_moved) {
				retryPreface.set(
					subgoal.name,
					`Your last attempt at this step was judged ${v.status}: ${v.because}. Evidence read: ${v.evidence || "(none)"}. It ended: ${a.claim.ending}. Do it differently this time.`,
				);
				attempt = 2;
				continue;
			}
			if (v.gate === "lost_grasp") {
				const back = graspStepFor(plan, index);
				if (back !== undefined && back !== index && !backtracked.has(subgoal.target)) {
					backtracked.add(subgoal.target);
					state.backtracked.push(subgoal.target);
					const grasp = plan.subgoals[back];
					grasp.target_boxes = {};
					const ways = triedWays.get(grasp.target) ?? [];
					ways.push("it slipped after being picked up: grasp a steadier spot or part");
					triedWays.set(grasp.target, ways);
					for (let i = back; i <= index; i++) verdicts.delete(plan.subgoals[i].name);
					index = back;
					attempt = 1;
					continue;
				}
			}
			const trigger = triggerText(subgoal, attempt, v, a, fails);
			const r = yield* doReplan(plan, index, trigger, v.gate === "cannot");
			if (r.outcome) return r.outcome;
			plan = r.plan;
			index = r.index;
			attempt = 1;
		}

		// -----------------------------------------------------------------------
		// one subgoal attempt (spec-planner §3.4)

		async function* attemptOnce(
			subgoal: Subgoal,
			attemptNo: number,
		): AsyncGenerator<
			MmCall,
			{ run: SubgoalRun; verdict?: Verdict; claim: Claim; never_moved: boolean; elapsed_s: number },
			MmReply
		> {
			const t0 = Date.now();
			const holdStep = wantsHolding(subgoal.criterion, subgoal.target);
			const releaseStep = wantsRelease(subgoal.criterion, subgoal.target);
			let before: Frame | undefined = !m.cache.stale ? m.cache.latest : undefined;
			let holdingAtStart: boolean | undefined;
			let lostGrasp = false;
			let released: { opening_mm: number } | undefined;
			let payloadWidth: number | undefined;
			let heldThisStep = false;
			const localNet = zeroNet();
			const localStructured = zeroNet();
			let gripperActed = false;
			const cycles: Cycle[] = [];
			const onEvent = (e: ExecutorEvent) => {
				evidence.event(subgoal.name, e);
				if (e.type === "step_end") {
					gripperActed ||= e.step.kind === "gripper";
					accumulate(e.step, net, false);
					accumulate(e.step, netStructured, true);
					accumulate(e.step, localNet, false);
					accumulate(e.step, localStructured, true);
				}
				if (e.type === "held") {
					heldThisStep = true;
					net = zeroNet();
					netStructured = zeroNet();
					holdAt = m.cache.latest?.frame_clock_s;
				}
				if (
					e.type === "reached" &&
					e.reach.from_plan_boxes &&
					e.reach.located?.point_mm &&
					!marks.has(subgoal.target)
				)
					marks.set(subgoal.target, e.reach.located.point_mm);
			};
			// the watchdogs read every frame the executor sees (the original's tee on observe)
			const watch = (f: Frame | undefined) => {
				if (!f) return;
				before ??= f;
				if (!(holdStep || releaseStep)) return;
				if (f.grasp.holding !== null && holdingAtStart === undefined) holdingAtStart = f.grasp.holding;
				if (
					holdingAtStart === true &&
					f.grasp.holding === false &&
					!lostGrasp &&
					!released &&
					!(releaseStep && f.gripper.commanded !== "close")
				) {
					lostGrasp = true;
					evidence.add(
						"alert",
						subgoal.name,
						"stop/dropped: the gripper reading went from holding to not holding",
						{ level: "stop", finding: "dropped" },
					);
					m.clock.cancel({ by: "grasp_watchdog", reason: "the grasp was lost" });
				}
				if (
					releaseStep &&
					holdingAtStart === true &&
					f.gripper.commanded !== "close" &&
					!released &&
					f.grasp.holding !== true
				) {
					released = { opening_mm: f.gripper.opening_mm };
					evidence.add("verdict", subgoal.name, "released: the jaws opened while holding on a release step", {
						gate: "released",
					});
					m.clock.cancel({ by: "release_watchdog", reason: "the payload was released" });
				}
			};
			m.clock.clear();
			// The watchdogs see the frame the executor starts from (the original's tee sees its first observe).
			watch(before);
			const ex = m.runSubgoal({
				task: m.task,
				subgoal,
				preface: retryPreface.get(subgoal.name) ?? "",
				carry,
				maxCycles: Math.min(m.budget, subgoal.max_cycles, MAX_CYCLES),
				ask: m.ask,
				clock: m.clock,
				cache: m.cache,
				onEvent,
			});
			let reply: MmReply | undefined;
			let run: SubgoalRun;
			for (;;) {
				const next = reply === undefined ? await ex.next() : await ex.next(reply);
				if (next.done) {
					run = next.value;
					break;
				}
				reply = yield next.value;
				watch(reply.frame);
			}
			watch(before);
			state.observes.push(...run.observes);
			cycles.push(...run.cycles);
			carry = run.carry;
			for (const c of run.cycles)
				if (c.release?.payload_width_mm !== undefined) payloadWidth = c.release.payload_width_mm;
			if (lostGrasp || released) holdAt = m.cache.latest?.frame_clock_s ?? holdAt;
			if (lostGrasp) {
				net = zeroNet();
				netStructured = zeroNet();
			}
			const neverMoved = !gripperActed && localNet.moveTotal < NEVER_MOVED_MM && localNet.turnTotal < NO_TURN_DEG;
			const claim = claimOf(run, subgoal, neverMoved);
			const elapsed_s = (Date.now() - t0) / 1000;
			const record: AttemptRecord = {
				name: subgoal.name,
				attempt: attemptNo,
				cycles: run.cycles.length,
				ending: run.ending,
				never_moved: neverMoved,
				stopped_by_monitor: m.clock.cancelled()?.by === "monitor",
				elapsed_s: r1(elapsed_s),
				net_motion: {
					label: { move_mm: r1(norm(localNet.move)), turn_deg: r1(localNet.turn) },
					structured: { move_mm: r1(norm(localStructured.move)), turn_deg: r1(localStructured.turn) },
				},
			};
			if (
				Math.abs(norm(localNet.move) - norm(localStructured.move)) > 5 ||
				Math.abs(localNet.turn - localStructured.turn) > 5
			)
				state.net_motion_disagreements++;
			state.subgoals.push(record);
			m.log("motormind_cycle", {
				subgoal: subgoal.name,
				attempt: attemptNo,
				cycles: run.cycles.map((c) => ({
					index: c.index,
					proposal: c.proposal
						? {
								assessment: c.proposal.assessment,
								done: c.proposal.done,
								expect: c.proposal.expect,
								deltas: c.proposal.deltas,
							}
						: null,
					steps: c.steps.map((s) => ({
						label: s.label,
						outcome: s.outcome,
						abort_reason: s.abort_reason ?? "",
						along_axis_mm: s.along_axis_mm,
						lateral_mm: s.lateral_mm,
						turned_deg: s.turned_deg,
					})),
					located: c.reach?.located ?? null,
					batch: c.batch ?? null,
					release: c.release ?? null,
					stopped_by: c.stopped_by ?? null,
					finding: c.finding ?? null,
					timing: c.timing,
				})),
				ending: run.ending,
				ending_text: run.ending_text,
				observes: run.observes.length,
				observes_yielded: run.observes.filter((o) => o.yielded).length,
				timing: run.timing,
			});
			// 8. no verdict on a robot-layer error, the episode's end, a cancel or the clock. A cycle's own
			// error (no acceptable proposal in three tries, no actions and no claim) is in the claim and is
			// judged like any other ending (spec-planner §3.4: only an executor that raised ends the mission).
			const cancel = m.clock.cancelled();
			if (
				(run.ending === "ERROR" && run.error_source !== "cycle") ||
				run.ending === "EPISODE_OVER" ||
				run.episode_over
			)
				return { run, claim, never_moved: neverMoved, elapsed_s };
			if (cancel && (cancel.by === "mission" || cancel.by === "deadline"))
				return { run, claim, never_moved: neverMoved, elapsed_s };
			if (run.ending === "CANCELLED" && !cancel) return { run, claim, never_moved: neverMoved, elapsed_s };
			// 9. a fresh picture
			const now = yield* look(m.cache, "before_verdict", state.observes);
			if (!now) return { run, claim, never_moved: neverMoved, elapsed_s };
			lastNow = now;
			if (heldThisStep || lostGrasp || released) holdAt = now.frame_clock_s;
			record.obs_id = now.obs_id;
			const planText = describePlan(plan.subgoals.map((s) => ({ subgoal: s, verdict: verdicts.get(s.name) })));
			const input = {
				task: m.task,
				subgoal,
				before,
				now,
				claim,
				note: note(),
				planText,
				robotText: robotText(now),
				boxHint: boxHint(subgoal),
				holding: now.grasp.holding,
				clauses: plan.clauses,
			};
			// 10. the verdict, code gates first
			let verdict: Verdict;
			if (lostGrasp) {
				verdict = coded(
					"not_done",
					"the grasp was lost: the gripper reading went from holding to not holding",
					"lost_grasp",
					true,
				);
				verdict = await withTask(verdict, input);
			} else if (run.ending === "CANNOT_FROM_THE_SIDE") {
				verdict = coded("not_done", run.ending_text || "the side approach cannot be made here", "cannot", true);
				cannot.push(subgoal.subgoal);
			} else if (released && releaseMeasured(run, released.opening_mm, payloadWidth, now)) {
				const miss = releaseMiss(run);
				verdict =
					miss === undefined || !miss.missed
						? coded(
								"done",
								`released ${miss ? `${r1(miss.offset_mm)} mm from the mark (allowed ${r1(miss.allowed_mm)})` : "over the place"}`,
								"released",
								false,
							)
						: coded(
								"not_done",
								`released ${r1(miss.offset_mm)} mm from the mark; ${r1(miss.allowed_mm)} mm were allowed`,
								"release_missed",
								true,
							);
				if (subgoal.clause >= 0) (miss?.missed ? measuredMiss : measuredHit)(ledger, subgoal.clause);
				verdict = await withTask(verdict, input);
			} else {
				verdict = await verify(m.ask, input);
				// 11. a placing step judged by the model: ask the whole task once more
				if (releaseStep && !verdict.gate && !verdict.asked && (released || verdict.status === "done"))
					verdict = await withTask(verdict, input);
			}
			record.verdict = {
				status: verdict.status,
				because: verdict.because,
				gate: verdict.gate,
				replan: verdict.replan,
			};
			evidence.add("verdict", subgoal.name, `${verdict.status}: ${verdict.because}`, {
				gate: verdict.gate,
				cycle: run.cycles.length,
			});
			m.log("motormind_verdict", {
				subgoal: subgoal.name,
				attempt: attemptNo,
				status: verdict.status,
				because: verdict.because,
				evidence: verdict.evidence,
				identity: verdict.identity,
				replan: verdict.replan,
				clauses: verdict.clauses,
				gate: verdict.gate,
				asked: verdict.asked
					? { status: verdict.asked.status, because: verdict.asked.because, clauses: verdict.asked.clauses }
					: null,
				overrode: verdict.overrode ?? null,
				obs_id: now.obs_id ?? null,
				note_version: 0,
				never_moved: neverMoved,
				net_motion: record.net_motion,
				calls: verdict.calls,
			});
			m.requestNote?.();
			return { run, verdict, claim, never_moved: neverMoved, elapsed_s };

			function coded(status: Verdict["status"], because: string, gate: string, replanIt: boolean): Verdict {
				return {
					status,
					because,
					evidence: "code measurement",
					identity: "",
					replan: replanIt,
					clauses: [],
					gate,
					calls: [],
				};
			}
			async function withTask(v: Verdict, input: Parameters<typeof askTask>[1]): Promise<Verdict> {
				const asked = await askTask(m.ask, input);
				if (!asked) return v;
				v.calls.push(...asked.calls);
				if (asked.status === "task_done" && v.status === "done") return { ...v, status: "task_done", asked };
				return { ...v, asked };
			}
		}

		// -----------------------------------------------------------------------
		// the whole-task check (spec-planner §3.6)

		async function* wholeTask(
			p: Plan,
			opening: Frame,
			claimText: string,
			cannotMove: boolean,
		): AsyncGenerator<MmCall, { outcome?: MissionEnd; replan?: boolean }, MmReply> {
			const now = yield* look(m.cache, "task_check", state.observes);
			if (!now)
				return { outcome: end("error", "no_observation", "the robot gave no observation for the task check") };
			lastNow = now;
			const planText = describePlan(p.subgoals.map((s) => ({ subgoal: s, verdict: verdicts.get(s.name) })));
			const task: Subgoal = {
				name: "task",
				subgoal: m.task,
				criterion: "every clause of the task is met; task_done only when all are",
				target: "",
				target_boxes: {},
				clause: -1,
				max_cycles: 1,
				gripper_only: false,
			};
			const input = {
				task: m.task,
				subgoal: task,
				before: opening,
				now,
				claim: {
					done: true,
					assessment: claimText,
					cycles: 0,
					stopped_by: "",
					ending: claimText,
					turn_net_deg: 0,
					turn_total_deg: 0,
					move_net_mm: 0,
					move_total_mm: 0,
				},
				note: note(),
				planText,
				robotText: robotText(now),
				holding: now.grasp.holding,
				wholeTask: true,
				clauses: p.clauses,
			};
			let v = await verify(m.ask, input);
			fold(ledger, v.clauses, true);
			// one more ask on a new picture when every clause has a code event behind it
			if (
				!askedWholeOnce &&
				v.status !== "task_done" &&
				ledger.clauses.every((_, i) => ledger.events.has(i)) &&
				ledger.clauses.length
			) {
				askedWholeOnce = true;
				const again = yield* look(m.cache, "task_check", state.observes);
				if (again) {
					const second = await askTask(m.ask, { ...input, now: again });
					if (second) {
						v.calls.push(...second.calls);
						fold(ledger, second.clauses, true);
						if (second.status === "task_done") v = { ...second, calls: v.calls };
					}
				}
			}
			state.clauses = clausesResult(ledger);
			m.log("motormind_verdict", {
				subgoal: "task",
				attempt: 1,
				status: v.status,
				because: v.because,
				evidence: v.evidence,
				identity: v.identity,
				replan: v.replan,
				clauses: v.clauses,
				gate: v.gate,
				obs_id: now.obs_id ?? null,
				note_version: 0,
				cannot_move: cannotMove,
				calls: v.calls,
			});
			if (v.status === "task_done" && allMet(ledger)) return { outcome: end("task_done", "task_done", v.because) };
			if (v.status === "abort") return { outcome: end("aborted", "verdict_abort", v.because) };
			if (allMet(ledger))
				return {
					outcome: end(
						"exhausted",
						"clauses_met_but_not_done",
						`every clause was seen met, yet the check says: ${v.because}`,
					),
				};
			if (cannotMove) return { outcome: end("exhausted", "episode_over", v.because) };
			return { replan: true };
		}

		// -----------------------------------------------------------------------
		// the replan (spec-planner §3.5)

		async function* doReplan(
			p: Plan,
			keep: number,
			trigger: string,
			fromCannot: boolean,
		): AsyncGenerator<MmCall, { outcome?: MissionEnd; plan: Plan; index: number }, MmReply> {
			const failed = p.subgoals[keep];
			const object = failed?.target ?? "";
			const key = [...replansOf.keys()].find((k) => sameObject(k, object)) ?? object;
			const strict = (replansOf.get(key) ?? 0) >= 1;
			const unmetNow =
				keep >= p.subgoals.length
					? unmet(ledger)
					: unmet(ledger).filter((u) => p.subgoals.slice(keep).some((s) => s.clause === u.i));
			// the price gate
			const estimate = unmetNow.reduce((sum, u) => {
				const started = clauseStarted.get(u.i);
				const remaining = p.subgoals.slice(keep).filter((s) => s.clause === u.i);
				if (started === undefined) return sum + SUBGOAL_MEDIAN_S + SUBGOAL_TAX_S;
				if (remaining.length)
					return (
						sum +
						remaining.reduce(
							(a, s) => a + Math.min(firstDuration.get(s.name) ?? SUBGOAL_MEDIAN_S, SUBGOAL_MEDIAN_S),
							0,
						)
					);
				return sum + (Date.now() - started) / 1000;
			}, 0);
			const wouldNeed = (state.plan_calls_s.at(-1) ?? 0) + estimate;
			const left = m.clock.left_s();
			if (left < Math.max(REPLAN_NEEDS_S, wouldNeed)) {
				state.replan_refused_for_price = {
					left_s: r1(left),
					would_need_s: r1(wouldNeed),
					estimate_s: r1(estimate),
					unmet: unmetNow,
				};
				return {
					outcome: end(
						"timeout",
						"replan_refused_for_price",
						`${r1(left)} s left, a replan would need ${r1(wouldNeed)} s`,
					),
					plan: p,
					index: keep,
				};
			}
			if (state.replans >= m.maxReplans)
				return { outcome: end("exhausted", "max_replans", `${state.replans} replans used`), plan: p, index: keep };
			state.replans++;
			if (object) replansOf.set(key, (replansOf.get(key) ?? 0) + 1);
			const frame = lastNow ?? picture;
			const last = state.subgoals.at(-1);
			const rebox: string[] = [];
			if (
				failed &&
				(fromCannot || /could not find|no coordinates|another product|printed/i.test(last?.verdict?.because ?? ""))
			)
				rebox.push(failed.target);
			const t0 = Date.now();
			const r = await replan(m.ask, {
				task: m.task,
				frame,
				robotText: robotText(frame),
				capabilities: m.describers.capabilities(frame.caps),
				legend: m.describers.legend(frame) || undefined,
				budget: m.budget,
				plan: p,
				verdicts: p.subgoals.map((s) => ({ subgoal: s, verdict: verdicts.get(s.name) })) as Verdicted[],
				keep,
				tail: p.subgoals.slice(keep + 1),
				trigger,
				unmet: unmetNow,
				note: note(),
				strict,
				triedWays:
					strict || fromCannot
						? (triedWays.get(key) ?? failures.get(key)?.map((f) => f.split(": ").slice(1).join(": ")))
						: undefined,
				holding: frame.grasp.holding === true,
				rebox,
				since: holdAt,
				doneNames: p.subgoals.slice(0, keep).map((s) => s.name),
				cannot,
				replans: state.replans,
			});
			state.plan_calls_s.push((Date.now() - t0) / 1000);
			state.plans++;
			logPlan("replan", r, trigger);
			if (!r.plan)
				return { outcome: end("error", "replan_failed", r.error ?? "the replan failed"), plan: p, index: keep };
			const fresh = freshSteps(r.plan.subgoals, keep, ledger.met);
			if (!fresh.some((s) => acts(s))) {
				state.no_new_steps++;
				if (state.no_new_steps >= 2)
					return {
						outcome: end("exhausted", "nothing_to_do", "the planner twice found nothing left to do"),
						plan: p,
						index: keep,
					};
				const truncated: Plan = { ...p, subgoals: p.subgoals.slice(0, keep) };
				return { plan: truncated, index: truncated.subgoals.length };
			}
			if (r.gave_up) return { outcome: end("gave_up", "planner_gave_up", r.gave_up), plan: p, index: keep };
			const merged: Plan = { ...r.plan, subgoals: [...p.subgoals.slice(0, keep), ...fresh] };
			state.actions_cap = state.actions + 2 * fresh.reduce((a, s) => a + s.max_cycles, 0);
			return { plan: merged, index: keep };
		}

		// -----------------------------------------------------------------------
		// helpers

		function logPlan(kind: "plan" | "replan", p: PlanOutcome, trigger: string) {
			m.log("motormind_plan", {
				kind,
				trigger,
				ok: p.plan !== undefined,
				error: p.error ?? null,
				gave_up: p.gave_up ?? null,
				attempts: p.attempts,
				reasks: p.reasks,
				clamped: p.clamped,
				plan: p.plan
					? {
							rationale: p.plan.rationale,
							clauses: p.plan.clauses,
							subgoals: p.plan.subgoals,
							boxed_on: p.plan.boxed_on,
							boxed_at: p.plan.boxed_at ?? null,
							boxes_kept_from: p.plan.boxes_kept_from,
							recoverable: p.plan.recoverable,
						}
					: null,
				calls: p.calls,
				elapsed_s: r1(state.plan_calls_s.at(-1) ?? 0),
			});
			evidence.add("plan", "mission", `${kind}: ${p.plan ? `${p.plan.subgoals.length} steps` : p.error}`, {
				trigger,
			});
		}

		function boxHint(s: Subgoal): string | undefined {
			const entries = Object.entries(s.target_boxes);
			if (!entries.length) return undefined;
			return entries
				.map(
					([cam, b]) =>
						`${cam}: around (${Math.round(((b![0] + b![2]) / 2 / 1000) * 100)}%, ${Math.round(((b![1] + b![3]) / 2 / 1000) * 100)}%) of the view`,
				)
				.join("; ");
		}

		/** The executor's claim for the verifier (spec-planner §2.8). */
		function claimOf(run: SubgoalRun, s: Subgoal, neverMoved: boolean): Claim {
			const cycles = run.cycles;
			const lastProposal = [...cycles].reverse().find((c) => c.proposal)?.proposal;
			const reach = [...cycles].reverse().find((c) => c.reach?.located?.point_mm)?.reach;
			const residual = latestResidual(cycles);
			const mark = marks.get(s.target);
			const located = reach?.located;
			const fromMark =
				located?.point_mm && mark
					? Math.hypot(located.point_mm[0] - mark[0], located.point_mm[1] - mark[1])
					: undefined;
			return {
				done:
					cycles.some((c) => c.proposal?.done) ||
					["HELD", "ARRIVED", "RELEASED_IN_CODE", "MODEL_DONE", "DONE_AFTER_LAST"].includes(run.ending),
				assessment: lastProposal?.assessment ?? "",
				cycles: cycles.length,
				stopped_by:
					run.ending === "FROZEN" || run.ending === "WALL" ? run.ending : (cycles.at(-1)?.stopped_by ?? ""),
				ending: howItEnded(run, neverMoved),
				...(reach && located && !located.doubtful && residual !== undefined
					? {
							reached: {
								label: located.label ?? "",
								residual_mm: residual,
								target: reach.target,
								...(fromMark !== undefined ? { from_mark_mm: fromMark } : {}),
							},
						}
					: {}),
				turn_net_deg: net.turn,
				turn_total_deg: net.turnTotal,
				...(net.turnStop ? { turn_stop: net.turnStop } : {}),
				move_net_mm: norm(net.move),
				move_total_mm: net.moveTotal,
				...(run.error ? { error: run.error } : {}),
				...(run.held ? { held: run.held } : {}),
				...(located?.label ? { located_label: located.label } : {}),
				...(fromMark !== undefined ? { from_mark_mm: fromMark } : {}),
			};
		}

		/** Net motion from a step (spec-planner §3.7): by the label's first direction word, or by the structured axis. */
		function accumulate(step: SegmentReport, into: NetMotion, structured: boolean) {
			if (step.kind === "translate") {
				into.moveTotal += Math.abs(step.along_axis_mm);
				const axis = structured
					? step.axis
					: labelDirection(step.label)?.kind === "translate"
						? labelDirection(step.label)?.axis
						: undefined;
				if (!axis) return;
				const mm = Math.abs(step.along_axis_mm);
				into.move = [into.move[0] + axis[0] * mm, into.move[1] + axis[1] * mm, into.move[2] + axis[2] * mm];
			} else if (step.kind === "rotate") {
				into.turnTotal += Math.abs(step.turned_deg);
				const d = labelDirection(step.label);
				if (!d || d.kind !== "rotate" || Math.abs(d.axis[2]) < 0.5) return;
				into.turn += Math.sign(d.axis[2]) * Math.abs(step.turned_deg);
				if (step.outcome === "limited") into.turnStop = d.word;
			}
		}

		/** Code events after an attempt (spec-planner §3.7 `what_the_code_settled`): a push ≥ 5 mm, a turn ≥ 1° on a turn clause. */
		function noteSettled(s: Subgoal, run: SubgoalRun) {
			if (s.clause < 0) return;
			const pushed =
				run.cycles.some((c) =>
					c.steps.some(
						(st) =>
							st.kind === "translate" &&
							/\bpush|contact\b/i.test(st.label) &&
							Math.abs(st.along_axis_mm) >= PUSHED_ALONG_MM &&
							st.axis &&
							Math.abs(st.axis[2]) < 0.5,
					),
				) ||
				run.cycles.some(
					(c) =>
						c.proposal?.deltas.some((d) => d.push) &&
						c.steps.some((st) => st.outcome === "contact" && Math.abs(st.along_axis_mm) >= PUSHED_ALONG_MM),
				);
			const turned =
				isTurnClause(plan.clauses[s.clause] ?? "") &&
				run.cycles.some((c) =>
					c.steps.some(
						(st) =>
							st.kind === "rotate" &&
							Math.abs(st.turned_deg) >= TURNED_DEG &&
							st.outcome !== "failed" &&
							!st.abort_reason,
					),
				);
			if (pushed || turned) codeEvent(ledger, s.clause);
		}

		function triggerText(
			s: Subgoal,
			attemptNo: number,
			v: Verdict,
			a: { never_moved: boolean; run: SubgoalRun },
			fails: string[],
		): string {
			const parts = [`${s.name} (attempt ${attemptNo}) was judged ${v.status}: ${v.because}`];
			if (a.never_moved) parts.push("The robot never moved during the attempt.".slice(0, 200));
			else if (a.run.ending === "FROZEN" || a.run.ending === "WALL")
				parts.push(`The executor froze: ${a.run.ending_text}`.slice(0, 200));
			const recent = fails.slice(-4, -1);
			if (recent.length) parts.push(`Earlier failures on the same object: ${recent.join(" | ")}`);
			return parts.join(" ");
		}

		/** The grasp step the failed step's object was taken in (spec-planner §3.3): walk back over the holding steps of the same object. */
		function graspStepFor(p: Plan, failedIndex: number): number | undefined {
			const object = p.subgoals[failedIndex].target;
			let i = failedIndex;
			while (
				i - 1 >= 0 &&
				wantsHolding(p.subgoals[i - 1].criterion, p.subgoals[i - 1].target) &&
				sameObject(p.subgoals[i - 1].target, object)
			)
				i--;
			return wantsHolding(p.subgoals[i].criterion, p.subgoals[i].target) ? i : undefined;
		}

		/** A release was measured (spec-planner §3.8): opening > payload width + resolution, and the reading is not holding. */
		function releaseMeasured(run: SubgoalRun, opening: number, payload: number | undefined, now: Frame): boolean {
			if (now.grasp.holding === true) return false;
			const span = now.caps.gripper_open_span_m * 1000;
			const resolution = Math.min(EMPTY_BAND_MM, EMPTY_BAND_SHARE * span);
			const width =
				payload ??
				run.cycles
					.map((c) => c.release?.payload_width_mm)
					.filter((w): w is number => w !== undefined)
					.at(-1);
			return width === undefined ? true : opening > width + resolution;
		}
		function releaseMiss(run: SubgoalRun): { missed: boolean; offset_mm: number; allowed_mm: number } | undefined {
			const rel = [...run.cycles]
				.reverse()
				.find((c) => c.release?.offset_mm !== undefined && c.release.allowed_mm !== undefined)?.release;
			if (!rel || rel.offset_mm === undefined || rel.allowed_mm === undefined) return undefined;
			const allowed = rel.allowed_mm * (rel.placing_in ? SETTLES_WITHIN : 1);
			return { missed: rel.offset_mm > allowed, offset_mm: rel.offset_mm, allowed_mm: allowed };
		}
	}
}

/** Whether the jaws of a frame point down (for callers that need it without the whole contract). */
export const framesJawsDown = (f: Frame) => jawsDown(f.pose);
/**
 * A replan's new steps (spec-planner §3.5 rule 1): the returned steps from `keep` on, less any that
 * serves a clause the ledger holds met. The ledger must therefore hold only the task's own clauses as
 * met: a step criterion echoed into `clauses` (round 2 defect 1) deleted the take and lift steps here.
 */
/**
 * The residual the claim reports (spec-planner §2.8 "at the last look the tool — or what it carries —
 * was N mm from it"): the latest measured one, a batch's (after its moves) before a look's. The look's
 * own residual is the distance before the step's moves: a carry located its plate 250 mm away, moved,
 * and the verifier was told 250 mm with the bowl over the plate (round 2 defect 2b).
 */
export function latestResidual(cycles: readonly Cycle[]): number | undefined {
	for (const c of [...cycles].reverse()) {
		if (c.batch?.residual_mm !== undefined) return c.batch.residual_mm;
		if (c.reach?.residual_mm !== undefined) return c.reach.residual_mm;
	}
	return undefined;
}
export const freshSteps = (subgoals: readonly Subgoal[], keep: number, met: ReadonlySet<number>): Subgoal[] =>
	subgoals.slice(keep).filter((s) => !(s.clause >= 0 && met.has(s.clause)));
/** The attempt records' summary for the result's `subgoals`. */
export const subgoalSummary = (s: AttemptRecord[]) =>
	s.map((a) => ({ ...a, verdict: a.verdict ?? null, obs_id: a.obs_id ?? null }));
export const roleTotals = (calls: RoleCallRecord[]) => {
	const by: Record<
		string,
		{
			calls: number;
			failures_by_kind: Record<string, number>;
			cost_usd: number;
			tokens_in: number;
			tokens_out: number;
			latencies: number[];
		}
	> = {};
	for (const c of calls) {
		let r = by[c.role];
		if (!r) {
			r = { calls: 0, failures_by_kind: {}, cost_usd: 0, tokens_in: 0, tokens_out: 0, latencies: [] };
			by[c.role] = r;
		}
		r.calls++;
		if (!c.ok)
			r.failures_by_kind[c.error_kind ?? "unknown"] = (r.failures_by_kind[c.error_kind ?? "unknown"] ?? 0) + 1;
		r.cost_usd += c.cost_usd;
		r.tokens_in += c.tokens_in;
		r.tokens_out += c.tokens_out;
		r.latencies.push(c.latency_s);
	}
	const pct = (xs: number[], p: number) =>
		xs.length ? [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(p * xs.length))] : 0;
	return Object.fromEntries(
		Object.entries(by).map(([role, r]) => [
			role,
			{
				calls: r.calls,
				failures_by_kind: r.failures_by_kind,
				cost_usd: r3(r.cost_usd),
				tokens_in: r.tokens_in,
				tokens_out: r.tokens_out,
				latency_s_p50: r1(pct(r.latencies, 0.5)),
				latency_s_p90: r1(pct(r.latencies, 0.9)),
			},
		]),
	);
};
