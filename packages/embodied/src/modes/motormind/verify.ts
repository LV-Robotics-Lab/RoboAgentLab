/**
 * The MotorMind verifier (spec-planner §5): after a subgoal attempt it judges the step from the
 * pictures now and before (three 768 px views each), the executor's claim (what it said and what the
 * code measured), the last alert and the memory note (hearsay), answering the whole task clause by
 * clause every time. Code gates run before and after the model, in the spec's order: the gripper
 * reading (a release step still holding, a holding step with nothing held), a grasp with a width (an
 * identity-only question), the contradiction re-ask (not_done against a holding reading), the
 * measurement re-ask (the camera located the target under the tool), the located-identity gate, the
 * control gate (a turn that went and came back), and the clause gate (task_done needs every clause).
 * `gate` says which rule decided; "" means the model did.
 */

import { type Ask, type Frame, images, type RoleCallRecord, recordOf, type Subgoal } from "./contract.ts";
import { type ClauseAnswer, nouns, sameObject, wantsHolding, wantsRelease } from "./planner.ts";
import { fill, promptFile } from "./prompts.ts";

export const VERIFY_ATTEMPTS = 3;
/** A closed point this near the planned mark counts as "reached" (spec-planner §5.4 `_REACHED_MM`). */
export const REACHED_MM = 30;
/** The control gate (spec-planner §5.4 rule 8). */
export const TURNED_AT_ALL_DEG = 30;
export const UNDONE_SHARE = 0.2;
export const HALF_WAY = 0.5;

export type VerdictStatus = "done" | "not_done" | "task_done" | "abort";
export type Verdict = {
	status: VerdictStatus;
	because: string;
	evidence: string;
	identity: string;
	replan: boolean;
	clauses: ClauseAnswer[];
	/** Which rule decided ("" = the model). */
	gate: string;
	/** The whole-task question asked beside a code verdict or a placing step (spec-planner §5.1). */
	asked?: Verdict;
	/** The model's own answer when a gate overrode it. */
	overrode?: { status: VerdictStatus; because: string };
	calls: RoleCallRecord[];
};

/** What the mission hands the verifier about the attempt (spec-planner §2.8 executor_claim). */
export type Claim = {
	done: boolean;
	assessment: string;
	cycles: number;
	stopped_by: string;
	ending: string;
	reached?: { label: string; residual_mm: number; target: string; from_mark_mm?: number };
	turn_net_deg: number;
	turn_total_deg: number;
	turn_stop?: string;
	move_net_mm: number;
	move_total_mm: number;
	error?: string;
	/** The grasp closed with a width (HELD_WITH_A_WIDTH): the identity-only question.  */
	held?: { gap_mm: number; target_mm?: number };
	/** The label the camera gave the object it located for this step, when any. */
	located_label?: string;
	/** Distance from the closed point to the planner's mark, when known (mm). */
	from_mark_mm?: number;
};

export type VerifyInput = {
	task: string;
	subgoal: Subgoal;
	before?: Frame;
	now: Frame;
	claim: Claim;
	alert?: { level: string; finding: string; because: string; identity?: string };
	note: string;
	planText: string;
	robotText: string;
	/** The planner's box hint: where it marked the target when planning. */
	boxHint?: string;
	holding: boolean | null;
	/** The whole-task question: the task as a step whose criterion asks every clause. */
	wholeTask?: boolean;
	clauses: string[];
};

const PRINT = /\b(printed|printing|reads|label (says|reads)|text (says|reads)|lettering|written|logo)\b/i;
const EMPTY_OR_PLACE =
	/\b(nothing (in|between) the (jaws|gripper|fingers)|(jaws|gripper|fingers) (are|is) empty|not holding|holds? nothing|empty (jaws|gripper)|not (grasped|held|gripped)|(wrong|not the right|incorrect) (position|place|spot|location)|not (over|above|at|on|in) the|missed)\b/i;
const OTHER_OBJECT =
	/\b(different|another|other|wrong) (object|item|thing)|\bnot the (right|correct)\b|\binstead of\b/i;
const LIFT = /\b(lift|lifted|raise|raised|rise|risen|up by|higher)\b/i;
const TURN_STEP = /\b(turn|twist|rotate|knob|dial|screw)\w*\b/i;

const r0 = (v: number) => v.toFixed(0);
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** The degrees a turn step asks for, from its words ("about 90 degrees"); undefined when none. */
export function askedDegrees(s: Pick<Subgoal, "subgoal" | "criterion">): number | undefined {
	const m = /(\d+(?:\.\d+)?)\s*(?:deg|degrees|°)/i.exec(`${s.subgoal} ${s.criterion}`);
	return m ? Number(m[1]) : undefined;
}

/** A reply parsed as a verdict, or undefined when it is not one. */
export function parseVerdict(json: unknown): Omit<Verdict, "gate" | "calls"> | undefined {
	if (!isObj(json)) return undefined;
	const status = String(json.status ?? "")
		.trim()
		.toLowerCase();
	if (!["done", "not_done", "task_done", "abort"].includes(status)) return undefined;
	const clauses = Array.isArray(json.clauses)
		? json.clauses.filter(isObj).map((c) => {
				const index = c.index === undefined || c.index === null || c.index === "" ? Number.NaN : Number(c.index);
				return {
					what: String(c.what ?? c.clause ?? "").trim(),
					met: c.met === true || String(c.met).toLowerCase() === "true",
					evidence: String(c.evidence ?? "").trim(),
					...(Number.isInteger(index) && index >= 0 ? { index } : {}),
				};
			})
		: [];
	return {
		status: status as VerdictStatus,
		because: String(json.because ?? "").trim(),
		evidence: String(json.evidence ?? "").trim(),
		identity: String(json.identity ?? "").trim(),
		replan: json.replan === true || String(json.replan).toLowerCase() === "true",
		clauses,
	};
}

type Prompt = { system: string; user: string };
function promptOf(): Prompt {
	const { system, text } = promptFile("verify");
	return { system, user: text };
}

function claimText(c: Claim): string {
	const lines = [
		`It said: ${c.done ? "DONE" : "not done"}; its own assessment: ${c.assessment || "(none)"}.`,
		`Measured by code/cameras: ${c.cycles} action cycle(s); ending: ${c.ending}${c.stopped_by ? `; stopped by: ${c.stopped_by}` : ""}.`,
	];
	if (c.reached)
		lines.push(
			`Camera located ${c.reached.target}${c.reached.label ? ` (labelled "${c.reached.label}")` : ""}${c.reached.from_mark_mm !== undefined ? `, ${r0(c.reached.from_mark_mm)} mm from the planner's mark` : ""}; after the last move the tool — or what it carries — was ${r0(c.reached.residual_mm)} mm from it horizontally.`,
		);
	lines.push(
		`Net turn since the last grasp or drop: ${r0(c.turn_net_deg)} deg of ${r0(c.turn_total_deg)} deg total; net move: ${r0(c.move_net_mm)} mm of ${r0(c.move_total_mm)} mm total (a net far below the total means it went and came back)${c.turn_stop ? `; the turn met a stop turning ${c.turn_stop}` : ""}.`,
	);
	if (c.held)
		lines.push(
			`The jaws closed to ${r0(c.held.gap_mm)} mm${c.held.target_mm !== undefined ? ` on an object expected about ${r0(c.held.target_mm)} mm across` : ""} and report holding.`,
		);
	if (c.error) lines.push(`Executor error: ${c.error}`);
	return lines.join("\n");
}

function userPrompt(p: VerifyInput, t: Prompt, extra: string[]): string {
	const views = (f: Frame | undefined, when: string) =>
		f
			? f.camera_names
					.filter((n) => f.cameras[n]?.marked768)
					.map((n) => `- ${n} (${when}, frame ${f.frame_clock_s.toFixed(2)} s)`)
					.join("\n")
			: `- (no ${when} pictures: for criteria about change, read the numbers instead)`;
	const step = p.wholeTask
		? `THE WHOLE TASK as one step: criterion = every clause below is met; task_done only when ALL are.`
		: `STEP ${p.subgoal.name}: ${p.subgoal.subgoal}\nCRITERION: ${p.subgoal.criterion}${p.subgoal.target ? `\nTARGET (the planner's words; not evidence of identity): ${p.subgoal.target}` : ""}`;
	return fill(t.user, {
		task: p.task,
		step,
		clauses: p.clauses.map((c, i) => `[${i}] ${c}`).join("\n"),
		plan: p.planText,
		now_views: views(p.now, "now"),
		before_views: views(p.before, "before"),
		claim: claimText(p.claim),
		alert: p.alert
			? `${p.alert.level}/${p.alert.finding}: ${p.alert.because}${p.alert.identity ? ` (identity: ${p.alert.identity})` : ""}`
			: "(none)",
		note: p.note || "No summary yet.",
		robot: p.robotText,
		hint: p.boxHint ?? "(no planner box hint)",
		extra: extra.join("\n"),
	});
}

const gated = (
	status: VerdictStatus,
	because: string,
	gate: string,
	replan: boolean,
	calls: RoleCallRecord[],
): Verdict => ({
	status,
	because,
	evidence: "code measurement",
	identity: "",
	replan,
	clauses: [],
	gate,
	calls,
});

/** The whole-task question, asked once; it can only lift done to task_done (spec-planner §5.1). */
export async function askTask(ask: Ask, p: VerifyInput, signal?: AbortSignal): Promise<Verdict | undefined> {
	const t = promptOf();
	const whole: VerifyInput = { ...p, wholeTask: true };
	const imgs = [...images(p.now, "marked768"), ...(p.before ? images(p.before, "marked768") : [])];
	const r = await ask("verify", { system: t.system, text: userPrompt(whole, t, []) }, imgs, { signal });
	const calls = [recordOf("verify", r, "ask_task")];
	const v = r.ok ? parseVerdict(r.json) : undefined;
	if (!v) return undefined;
	const clauseGate = v.status === "task_done" && !(v.clauses.length && v.clauses.every((c) => c.met));
	return clauseGate
		? { ...v, status: "not_done", gate: "clauses", overrode: { status: v.status, because: v.because }, calls }
		: { ...v, gate: "", calls };
}

/** The verdict on one attempt (spec-planner §5.4), gates in order. */
export async function verify(ask: Ask, p: VerifyInput, signal?: AbortSignal): Promise<Verdict> {
	const calls: RoleCallRecord[] = [];
	const s = p.subgoal;
	const holdStep = wantsHolding(s.criterion);
	const releaseStep = wantsRelease(s.criterion);
	const withTask = async (v: Verdict): Promise<Verdict> => {
		const asked = await askTask(ask, p, signal);
		if (asked) {
			v.calls.push(...asked.calls);
			if (asked.status === "task_done" && v.status === "done") return { ...v, status: "task_done", asked };
			v.asked = asked;
		}
		return v;
	};
	// 1. the gripper gate
	if (!p.wholeTask && p.holding !== null) {
		if (releaseStep && p.holding === true)
			return withTask(
				gated("not_done", "the gripper still reports holding after a release step", "still holding", false, calls),
			);
		if (holdStep && p.holding === false)
			return withTask(
				gated("not_done", "the gripper reports nothing held after a holding step", "nothing held", false, calls),
			);
	}
	const t = promptOf();
	const imgs = [...images(p.now, "marked768"), ...(p.before ? images(p.before, "marked768") : [])];
	const say = async (extra: string[], gate?: string) => {
		const r = await ask("verify", { system: t.system, text: userPrompt(p, t, extra) }, imgs, { signal });
		calls.push(recordOf("verify", r, gate));
		return r.ok ? parseVerdict(r.json) : undefined;
	};
	const noPrint = (v: { identity: string; because: string }) => !PRINT.test(`${v.identity} ${v.because}`);
	// 2. a grasp with a width: identity only
	if (!p.wholeTask && p.claim.held && holdStep) {
		const v = await say(
			[
				`THE GRASP IS SETTLED BY THE READING: the jaws closed to ${r0(p.claim.held.gap_mm)} mm${p.claim.held.target_mm !== undefined ? ` (the object was expected about ${r0(p.claim.held.target_mm)} mm across)` : ""} and report holding. Answer ONLY whether the thing in the jaws is this step's object; done unless readable printing says it is another product.`,
			],
			"held_width",
		);
		if (!v)
			return {
				...gated("done", "held with a width; no valid identity answer", "held_width", false, calls),
				evidence: "gripper reading",
			};
		let out: Verdict = { ...v, gate: "held_width", calls };
		if (
			v.status === "not_done" &&
			p.holding === true &&
			p.claim.from_mark_mm !== undefined &&
			p.claim.from_mark_mm <= REACHED_MM &&
			noPrint(v)
		)
			out = {
				...out,
				status: "done",
				identity: v.identity || v.because,
				because: `held at the planned mark (${r0(p.claim.from_mark_mm)} mm from it); identity doubt kept aside`,
				overrode: { status: v.status, because: v.because },
			};
		return locatedIdentity(out, p);
	}
	// 3.–4. the normal question
	const extra: string[] = [];
	if (!p.wholeTask && p.holding === true && holdStep)
		extra.push(
			"HOLDING IS SETTLED BY THE GRIPPER READING: treat the holding part of the criterion as met; not_done only for printed text naming another product, or another part of the criterion (height, place, state).",
		);
	let v: ReturnType<typeof parseVerdict>;
	for (let attempt = 0; attempt < VERIFY_ATTEMPTS && !v; attempt++) v = await say(extra);
	if (!v) return gated("not_done", "the verifier gave no valid answer", "no_answer", false, calls);
	let out: Verdict = { ...v, gate: "", calls };
	// 5. the contradiction re-ask
	if (
		!p.wholeTask &&
		out.status === "not_done" &&
		p.holding === true &&
		EMPTY_OR_PLACE.test(out.because) &&
		!(out.identity && out.replan)
	) {
		const again = await say(
			[
				...extra,
				"THE READING SETTLES HOLDING: the gripper reports holding something. Judge the rest of the criterion; do not say the jaws are empty or the object was missed.",
			],
			"holding_true",
		);
		if (again && !(again.status === "not_done" && EMPTY_OR_PLACE.test(again.because)))
			out = { ...again, gate: "", calls };
		else
			out = {
				...out,
				status: "done",
				gate: "holding_true",
				because: "the gripper reading settles holding; the model's not_done contradicted it twice",
				overrode: { status: out.status, because: out.because },
			};
	}
	// 6. the measurement re-ask
	if (
		!p.wholeTask &&
		out.status === "not_done" &&
		OTHER_OBJECT.test(out.because) &&
		p.claim.reached &&
		p.claim.reached.residual_mm <= REACHED_MM &&
		p.claim.located_label &&
		sameObject(p.claim.located_label, p.claim.reached.target)
	) {
		const again = await say(
			[
				...extra,
				`MEASURED: the camera located "${p.claim.located_label}" (matching the target) ${r0(p.claim.reached.residual_mm)} mm from the tool. Reconsider with this measurement; this answer is final.`,
			],
			"regrounded",
		);
		if (again) out = { ...again, gate: "regrounded", calls };
	}
	out = locatedIdentity(out, p);
	// 8. the control gate
	if (!p.wholeTask && TURN_STEP.test(`${s.subgoal} ${s.criterion}`) && out.status !== "abort") {
		const { turn_net_deg: net, turn_total_deg: total, turn_stop } = p.claim;
		const asked = askedDegrees(s);
		if (
			asked !== undefined &&
			total - Math.abs(net) < 1 &&
			Math.abs(net) < asked * HALF_WAY &&
			out.status !== "not_done"
		)
			out = {
				...out,
				status: "not_done",
				gate: "half_turn",
				because: `the wrist turned only ${r0(Math.abs(net))} of the ${r0(asked)} deg asked, in one direction; ${turn_stop ? "a stop was met: turn the other way" : "keep turning"}`,
				overrode: { status: out.status, because: out.because },
			};
		else if (total >= TURNED_AT_ALL_DEG && Math.abs(net) <= UNDONE_SHARE * total && out.status !== "not_done")
			out = {
				...out,
				status: "not_done",
				gate: "net_turn",
				because: `the wrist turned ${r0(total)} deg in all but only ${r0(Math.abs(net))} deg net: it turned and turned back`,
				overrode: { status: out.status, because: out.because },
			};
	}
	// 9. the clause gate
	if (out.status === "task_done" && !(out.clauses.length && out.clauses.every((c) => c.met)))
		out = {
			...out,
			status: p.wholeTask ? "not_done" : "done",
			gate: "clauses",
			overrode: { status: out.status, because: out.because },
		};
	return out;
}

/**
 * 7. the located-identity gate: a holding or lift step done, but the camera's label shares no noun
 * with the target. Only when that look was for this step's target (spec-planner §5.4 rule 7): a carry
 * or lift step while holding looks for the receiver, or senses the landmark the thing stood on, and
 * "plate" sharing no word with "akita black bowl" says nothing about what is in the jaws (round 2
 * defect 2c: 13 of 15 such verdicts had the thing over its receiver).
 */
function locatedIdentity(v: Verdict, p: VerifyInput): Verdict {
	const s = p.subgoal;
	if (p.wholeTask || v.status !== "done") return v;
	if (!(wantsHolding(s.criterion) || LIFT.test(s.criterion))) return v;
	const label = p.claim.located_label;
	if (!label || !s.target) return v;
	if (!p.claim.reached || !sameObject(p.claim.reached.target, s.target)) return v;
	const a = nouns(label);
	let shared = false;
	for (const w of nouns(s.target)) if (a.has(w)) shared = true;
	if (shared) return v;
	return {
		...v,
		status: "not_done",
		replan: true,
		gate: "located_identity",
		because: `the camera located "${label}", which shares no word with the target "${s.target}"`,
		overrode: { status: v.status, because: v.because },
	};
}
