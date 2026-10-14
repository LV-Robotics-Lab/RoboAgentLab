/**
 * The MotorMind planner (spec-planner §4): one model call turns the task and three marked 768 px
 * views into a plan — the task's clauses, then subgoals (name, sentence, criterion, target, boxes
 * on the fixed cameras, the clause served, an action budget). A replan returns only the remaining
 * steps and must change the failed one. Replies are checked: hard failures (no JSON, wrong shape)
 * spend the 3 attempts; keyed complaints (`empty`, `size`, `boxes`, `side`, `clauses`, `clause`,
 * `act`, `cannot`) are re-asked at most twice each, then accepted and mended by code (§4.4). Old
 * boxes are kept across a replan for the same object when they were drawn after the last grasp,
 * drop or release (§4.5). Giving up (`recoverable=false`) is let through by the rule of §4.6.
 *
 * Prompts are ./prompts/plan.md and ./prompts/replan.md, written from the spec's description of
 * their intent (DESIGN §10).
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import {
	type Ask,
	type Box,
	type CameraName,
	type Frame,
	images,
	type RoleCallRecord,
	recordOf,
	type Subgoal,
} from "./contract.ts";
import { fill, type PromptName, promptFile } from "./prompts.ts";

export const MAX_SUBGOALS = 24;
export const PLAN_ATTEMPTS = 3;
export const REASKS = 2;
export const BOX_SPAN = 1000;
export const MAX_BOX_SHARE = 0.4;
/** Caps the planner describes: default budget per subgoal, and the soft target it is told to aim for. */
export const DEFAULT_BUDGET = 10;

/** One verdict line about a task clause; `index` is the clause's number when the verifier gave it. */
export type ClauseAnswer = { what: string; met: boolean; evidence: string; index?: number };
export type Plan = {
	subgoals: Subgoal[];
	rationale: string;
	/** Fixed for the mission: the first plan's clauses. */
	clauses: string[];
	boxed_on: CameraName[];
	boxed_at?: number;
	/** Subgoal name → the frame clock of the observation its boxes were drawn on (kept from an older plan). */
	boxes_kept_from: Record<string, number>;
	recoverable: boolean;
};
export type PlanOutcome = {
	plan?: Plan;
	error?: string;
	/** Why the planner gave up (replans only, after §4.6), with the plan it returned. */
	gave_up?: string;
	attempts: number;
	reasks: Record<string, number>;
	clamped: string[];
	calls: RoleCallRecord[];
};

// ---------------------------------------------------------------------------
// words

const HOLD_WORDS = /\b(hold|holding|held|grasp|grasped|grasping|grip|gripping|carry|carrying|lift|lifted|lifting)\b/i;
const RELEASE_WORDS =
	/\b(release|released|let go|open the (jaws|gripper)|jaws open|gripper open|resting (on|in)|sitting (on|in)|placed (on|in)|put down|rests? (on|in)|lies (on|in))\b/i;
const MOVE_WORDS =
	/\b(move|moving|lift|lower|descend|rise|raise|carry|slide|push|pull|turn|rotate|approach|retreat|go|travel)\w*\b/i;
const GRIPPER_NOUNS = /\b(gripper|jaws?|fingers?|grip)\b/i;
const GRIPPER_VERBS = /\b(open|close|release|let go|squeeze|clamp)\w*\b/i;
const ACT_WORDS =
	/\b(pick|grasp|grab|take|lift|carry|place|put|release|push|pull|press|turn|twist|rotate|slide|open|close|drop|move|insert|pour|stack)\w*\b/i;
const SIDE_WORDS = /\bfrom the side\b/i;
const TILT_WORDS = /\b(tilt|tilted|sideways grasp|horizontal(ly)? grasp|pitch|roll)\w*\b/i;
const HANDLE_WORDS = /\b(handle|knob|lever|pull|drawer|door)\b/i;
/** Part words stripped from a target so it names the whole object (spec-planner §4.4 step 2). */
const PART_WORDS =
	/^(?:the\s+)?(?:top|lid|rim|edge|handle|side|front|back|corner|body|base|cap|spout|wall|face|middle|center|centre)\s+of\s+/i;
const IRRECOVERABLE =
	/\b(out of reach|unreachable|cannot reach|can't reach|dropped|on the floor|fell|fallen|outside the workspace|out of the workspace|broken|shattered|spilled|spilt|cannot be grasped|can't be grasped|ungraspable|permanent|irreversibl)\w*\b/i;

export const wantsHolding = (criterion: string) => HOLD_WORDS.test(criterion) && !RELEASE_WORDS.test(criterion);
export const wantsRelease = (criterion: string) => RELEASE_WORDS.test(criterion);
/** The step moves a hand (spec-planner §4.3 `act`, §3.5 `acts`): it names an action on an object, not only a look. */
export const acts = (s: Pick<Subgoal, "subgoal" | "criterion">) =>
	ACT_WORDS.test(s.subgoal) && !/^\s*(look|observe|inspect|check|view|see)\b/i.test(s.subgoal);
/** Not written by the model: a gripper-only step mentions the jaws and a jaw verb, no movement word, and does not want holding. */
export const gripperOnly = (s: Pick<Subgoal, "subgoal" | "criterion">) =>
	GRIPPER_NOUNS.test(s.subgoal) &&
	GRIPPER_VERBS.test(s.subgoal) &&
	!MOVE_WORDS.test(s.subgoal) &&
	!wantsHolding(s.criterion);

const STOP = new Set([
	"the",
	"a",
	"an",
	"of",
	"on",
	"in",
	"to",
	"and",
	"with",
	"from",
	"at",
	"it",
	"its",
	"into",
	"onto",
]);
export const nouns = (s: string) =>
	new Set(
		s
			.toLowerCase()
			.split(/[^a-z0-9]+/)
			.filter((w) => w.length >= 3 && !STOP.has(w)),
	);
/** Two target phrases name the same object: they share a content word (spec-planner §3.3, §4.5 `same_object`). */
export function sameObject(a: string, b: string): boolean {
	if (!a.trim() || !b.trim()) return false;
	const x = nouns(a);
	for (const w of nouns(b)) if (x.has(w)) return true;
	return false;
}

/** The target as the mission uses it: quoted labels without a task word dropped, part words stripped. */
export function cleanTarget(target: string, task: string): string {
	const taskWords = nouns(task);
	let t = target.replace(/["“”']([^"“”']+)["“”']/g, (m, inner: string) => {
		for (const w of nouns(inner)) if (taskWords.has(w)) return m;
		return "";
	});
	t = t.replace(/\s+/g, " ").trim();
	const stripped = t.replace(PART_WORDS, "");
	return (stripped || t).trim();
}

// ---------------------------------------------------------------------------
// boxes

/** Why a box is dropped (spec-planner §2.2), else undefined. */
export function boxFault(camera: string, box: unknown, cameras: readonly string[]): string | undefined {
	if (!cameras.includes(camera)) return `camera ${camera} is not a fixed camera`;
	if (!Array.isArray(box) || box.length !== 4 || !box.every((v) => typeof v === "number" && Number.isFinite(v)))
		return "a box is four numbers";
	const [x0, y0, x1, y1] = box as number[];
	if ([x0, y0, x1, y1].some((v) => v < 0 || v > BOX_SPAN)) return `box coordinates are 0..${BOX_SPAN}`;
	if (x1 <= x0 || y1 <= y0) return "box corners are reversed";
	if (((x1 - x0) * (y1 - y0)) / (BOX_SPAN * BOX_SPAN) >= MAX_BOX_SHARE) return "box covers 40% or more of the view";
	return undefined;
}

/** The fixed cameras: every name without "wrist" (spec-planner §2.1). */
export const fixedCameras = (names: readonly CameraName[]) => names.filter((n) => !n.includes("wrist"));

// ---------------------------------------------------------------------------
// prompts

type Prompt = { system: string; user: string };
function promptOf(name: PromptName): Prompt {
	const { system, text } = promptFile(name);
	return { system, user: text };
}

export type PlanInput = {
	task: string;
	frame: Frame;
	robotText: string;
	capabilities: string;
	/** The aim-point legend, or undefined when the frame has none (the prompt then says directions are the executor's). */
	legend?: string;
	budget: number;
};
export type Verdicted = { subgoal: Subgoal; verdict?: { status: string; because: string; attempt: number } };
export type ReplanInput = PlanInput & {
	/** The plan so far with each step's verdict; `keep` steps are done and stay; `tail` are the not-yet-run steps after the failed one. */
	plan: Plan;
	verdicts: Verdicted[];
	keep: number;
	tail: Subgoal[];
	trigger: string;
	unmet: { i: number; text: string }[];
	note: string;
	strict: boolean;
	triedWays?: string[];
	holding: boolean;
	/** Objects whose boxes may be redrawn (spec-planner §3.5 rebox). */
	rebox: string[];
	/** Boxes drawn before this frame clock are not kept (the last grasp, drop or release). */
	since?: number;
	/** Names of steps already done (not to be reused). */
	doneNames: string[];
	/** Steps the robot measured as impossible (spec-planner §4.3 `cannot`), verbatim. */
	cannot: string[];
	replans: number;
};

const cameraLines = (f: Frame, fixed: readonly CameraName[]) =>
	f.camera_names
		.map((n) => {
			const c = f.cameras[n];
			return `- ${n}: ${c ? `${c.width}x${c.height} px as sent` : "no image"}${fixed.includes(n) ? " (fixed; boxes go here)" : " (moving; no boxes)"}`;
		})
		.join("\n");

/** The plan as the replanner and verifier read it: each step with its verdict (spec-planner §4.1 describe_plan). */
export function describePlan(items: Verdicted[]): string {
	return items
		.map(({ subgoal: s, verdict: v }, i) => {
			const state = v ? `${v.status}${v.attempt > 1 ? ` (attempt ${v.attempt})` : ""}: ${v.because}` : "not run";
			return `${i + 1}. ${s.name}: ${s.subgoal} | criterion: ${s.criterion}${s.target ? ` | target: ${s.target}` : ""} | clause ${s.clause} | ${state}`;
		})
		.join("\n");
}

function planPrompt(p: PlanInput, fixed: readonly CameraName[]): Prompt {
	const t = promptOf("plan");
	return {
		system: t.system,
		user: fill(t.user, {
			task: p.task,
			robot: p.robotText,
			capabilities: p.capabilities,
			cameras: cameraLines(p.frame, fixed),
			legend: p.legend ?? "No legend: do not name directions yourself; the executor has its own legend.",
			budget: p.budget,
			soft: Math.floor(p.budget / 2),
			max_subgoals: MAX_SUBGOALS,
			fixed: fixed.join(", "),
		}),
	};
}

function replanPrompt(p: ReplanInput, fixed: readonly CameraName[]): Prompt {
	const t = promptOf("replan");
	const extras: string[] = [];
	if (p.strict)
		extras.push(
			`STRICT: ${p.tail.length ? "this" : "the failed"} object already cost one replan. Do not add preparation steps for it; either change the approach in a single step, or do the rest of the task first and put it last.`,
		);
	if (p.triedWays?.length)
		extras.push(
			`Already tried on this object (directions and ways): ${p.triedWays.join("; ")}. Change at least one of them, or change the order.`,
		);
	if (p.holding)
		extras.push(
			"The robot is holding something now. Opening the gripper without naming where to put it down means dropping it; such a step will be removed.",
		);
	return {
		system: t.system,
		user: fill(t.user, {
			task: p.task,
			robot: p.robotText,
			capabilities: p.capabilities,
			cameras: cameraLines(p.frame, fixed),
			legend: p.legend ?? "No legend: do not name directions yourself; the executor has its own legend.",
			budget: p.budget,
			soft: Math.floor(p.budget / 2),
			max_subgoals: MAX_SUBGOALS,
			fixed: fixed.join(", "),
			plan: describePlan(p.verdicts),
			trigger: p.trigger,
			unmet: p.unmet.length ? p.unmet.map((u) => `- [${u.i}] ${u.text}`).join("\n") : "- (none listed)",
			clauses: p.plan.clauses.map((c, i) => `[${i}] ${c}`).join("\n"),
			note: p.note || "No summary yet.",
			done_names: p.doneNames.join(", ") || "(none)",
			extras: extras.join("\n\n"),
		}),
	};
}

// ---------------------------------------------------------------------------
// parsing and checking

type RawSubgoal = Record<string, unknown>;
type Raw = { subgoals: RawSubgoal[]; rationale: string; clauses?: string[]; recoverable: boolean };

/** The reply's shape as a plan: an envelope is wrapped, a bare subgoal or list is a plan (spec-planner §4.3 `_envelope`, `_shape`). */
export function envelope(json: unknown): { raw?: Raw; shape?: string; error?: string } {
	if (Array.isArray(json)) return { raw: { subgoals: json.filter(isObj), rationale: "", recoverable: true } };
	if (!isObj(json)) return { error: "the reply is not a JSON object" };
	if (Array.isArray(json.subgoals)) {
		const clauses = Array.isArray(json.clauses)
			? json.clauses.map((c) => String(c ?? "").trim()).filter(Boolean)
			: undefined;
		return {
			raw: {
				subgoals: json.subgoals.filter(isObj),
				rationale: String(json.rationale ?? ""),
				...(clauses ? { clauses } : {}),
				recoverable: json.recoverable !== false,
			},
		};
	}
	if (typeof json.subgoal === "string" || typeof json.name === "string")
		return {
			raw: { subgoals: [json], rationale: String(json.rationale ?? ""), recoverable: true },
			shape: "a single subgoal was returned; return {subgoals: [...]} with every step",
		};
	return { error: "the reply has no subgoals list" };
}
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

function subgoalOf(
	r: RawSubgoal,
	fixed: readonly CameraName[],
	budget: number,
	clauseCount: number,
	task: string,
): Subgoal {
	const boxes: Partial<Record<CameraName, Box>> = {};
	if (isObj(r.target_boxes))
		for (const [cam, box] of Object.entries(r.target_boxes))
			if (!boxFault(cam, box, fixed)) boxes[cam as CameraName] = box as Box;
	const clause = Number(r.clause);
	const cycles = Number(r.max_cycles);
	const s: Subgoal = {
		name: String(r.name ?? "").trim(),
		subgoal: String(r.subgoal ?? "").trim(),
		criterion: String(r.criterion ?? "").trim(),
		target: cleanTarget(String(r.target ?? ""), task),
		target_boxes: boxes,
		clause: Number.isInteger(clause) && clause >= 0 && clause < clauseCount ? clause : -1,
		max_cycles: Number.isFinite(cycles) ? Math.min(budget, Math.max(1, Math.round(cycles))) : budget,
		gripper_only: false,
	};
	if (!s.target) s.target_boxes = {};
	s.gripper_only = gripperOnly(s);
	return s;
}

/** The keyed complaints about a parsed plan (spec-planner §4.3), in order; `boxes`, `clauses`, `cannot` are asked once. */
export function complaints(
	raw: Raw,
	o: { first: boolean; fixed: readonly CameraName[]; cannot: readonly string[]; budget: number; task: string },
): { key: string; text: string }[] {
	const out: { key: string; text: string }[] = [];
	const steps = raw.subgoals;
	if (o.first && !steps.length) out.push({ key: "empty", text: "the plan is empty" });
	const missing = steps.filter(
		(s) => !String(s.name ?? "").trim() || !String(s.subgoal ?? "").trim() || !String(s.criterion ?? "").trim(),
	);
	if (missing.length) out.push({ key: "empty", text: `${missing.length} step(s) lack name, subgoal or criterion` });
	if (steps.length > MAX_SUBGOALS)
		out.push({ key: "size", text: `${steps.length} steps exceed the ${MAX_SUBGOALS} allowed` });
	const unboxed = steps.filter((s) => {
		const target = cleanTarget(String(s.target ?? ""), o.task);
		if (!target) return false;
		const tb = isObj(s.target_boxes) ? s.target_boxes : {};
		return !Object.entries(tb).some(([cam, b]) => !boxFault(cam, b, o.fixed));
	});
	if (unboxed.length)
		out.push({
			key: "boxes",
			text: `${unboxed.map((s) => String(s.name)).join(", ")} name a target but draw no valid box on a fixed camera (${o.fixed.join(", ")})`,
		});
	const side = steps.filter(
		(s) =>
			(TILT_WORDS.test(String(s.subgoal ?? "")) ||
				(HANDLE_WORDS.test(String(s.subgoal ?? "")) && HOLD_WORDS.test(String(s.criterion ?? "")))) &&
			!SIDE_WORDS.test(String(s.subgoal ?? "")),
	);
	if (side.length)
		out.push({
			key: "side",
			text: `${side.map((s) => String(s.name)).join(", ")} tilt the gripper or grasp a sliding handle but do not say "from the side"`,
		});
	if (o.first && !raw.clauses?.length)
		out.push({ key: "clauses", text: "the first plan must list the task's clauses" });
	if (raw.clauses?.length) {
		const served = new Set(steps.map((s) => Number(s.clause)));
		const orphan = raw.clauses.map((_, i) => i).filter((i) => !served.has(i));
		if (orphan.length && o.first)
			out.push({ key: "clause", text: `clause(s) ${orphan.join(", ")} have no step serving them` });
		if (o.first) {
			const lookOnly = raw.clauses
				.map((_, i) => i)
				.filter(
					(i) =>
						served.has(i) &&
						!steps.some(
							(s) =>
								Number(s.clause) === i &&
								acts({ subgoal: String(s.subgoal ?? ""), criterion: String(s.criterion ?? "") }),
						),
				);
			if (lookOnly.length)
				out.push({ key: "act", text: `clause(s) ${lookOnly.join(", ")} are only looked at, never acted on` });
		}
	}
	const repeated = steps.filter((s) => o.cannot.some((c) => c.trim() && c.trim() === String(s.subgoal ?? "").trim()));
	if (repeated.length)
		out.push({
			key: "cannot",
			text: `${repeated.map((s) => String(s.name)).join(", ")} repeat a step the robot measured as impossible`,
		});
	return out;
}
const ONCE = new Set(["boxes", "clauses", "cannot"]);

// ---------------------------------------------------------------------------
// the calls

async function askPlan(
	ask: Ask,
	role: "planner",
	prompt: Prompt,
	imgs: ImageContent[],
	o: {
		first: boolean;
		fixed: readonly CameraName[];
		cannot: readonly string[];
		budget: number;
		task: string;
		signal?: AbortSignal;
	},
): Promise<{ raw?: Raw; error?: string; attempts: number; reasks: Record<string, number>; calls: RoleCallRecord[] }> {
	const calls: RoleCallRecord[] = [];
	const reasks: Record<string, number> = {};
	let attempts = 0;
	let hard = 0;
	let text = prompt.user;
	let last: Raw | undefined;
	while (hard < PLAN_ATTEMPTS) {
		attempts++;
		const r = await ask(role, { system: prompt.system, text }, imgs, { signal: o.signal });
		calls.push(recordOf(role, r));
		if (!r.ok) {
			hard++;
			if (r.error_kind === "aborted") return { error: r.error, attempts, reasks, calls };
			text = `${prompt.user}\n\nYour last answer was rejected: ${r.error}. Answer with the JSON object only.`;
			continue;
		}
		const env = envelope(r.json);
		if (!env.raw) {
			hard++;
			text = `${prompt.user}\n\nYour last answer was rejected: ${env.error}.`;
			continue;
		}
		last = env.raw;
		const found = complaints(env.raw, o);
		const open = found.filter((c) => {
			const n = reasks[c.key] ?? 0;
			return ONCE.has(c.key) ? n < 1 : n < REASKS;
		});
		if (!open.length && !env.shape) return { raw: env.raw, attempts, reasks, calls };
		for (const c of open) reasks[c.key] = (reasks[c.key] ?? 0) + 1;
		if (!open.length) return { raw: env.raw, attempts, reasks, calls };
		text = `${prompt.user}\n\nYour last answer was rejected: ${[...open.map((c) => c.text), ...(env.shape ? [env.shape] : [])].join("; ")}. Fix these and answer again with the whole plan.`;
	}
	return last
		? { raw: last, attempts, reasks, calls }
		: { error: "the planner gave no usable plan in 3 attempts", attempts, reasks, calls };
}

/** Unique names, avoiding the done steps' (spec-planner §2.2: `_2`, `_3` suffixes). */
function dedupe(steps: Subgoal[], taken: readonly string[]): void {
	const used = new Set(taken);
	for (const s of steps) {
		let name = s.name || "step";
		let k = 2;
		while (used.has(name)) name = `${s.name || "step"}_${k++}`;
		s.name = name;
		used.add(name);
	}
}

/** The first plan (spec-planner §4.1–4.4). */
export async function plan(ask: Ask, p: PlanInput, signal?: AbortSignal): Promise<PlanOutcome> {
	const fixed = fixedCameras(p.frame.camera_names);
	const prompt = planPrompt(p, fixed);
	const imgs = images(p.frame, "marked768");
	const r = await askPlan(ask, "planner", prompt, imgs, {
		first: true,
		fixed,
		cannot: [],
		budget: p.budget,
		task: p.task,
		signal,
	});
	if (!r.raw) return { error: r.error, attempts: r.attempts, reasks: r.reasks, clamped: [], calls: r.calls };
	const clamped: string[] = [];
	let clauses = r.raw.clauses ?? [];
	if (!clauses.length) {
		clauses = [p.task.trim()];
		clamped.push("no clauses: the task is clause 0 and every step serves it");
	}
	const steps = r.raw.subgoals.map((s) => subgoalOf(s, fixed, p.budget, clauses.length, p.task));
	if (!r.raw.clauses?.length) for (const s of steps) s.clause = 0;
	r.raw.subgoals.forEach((s, i) => {
		if (steps[i].max_cycles !== Number(s.max_cycles))
			clamped.push(`${steps[i].name}: max_cycles clamped to ${steps[i].max_cycles}`);
	});
	dedupe(steps, []);
	if (!steps.length)
		return {
			error: "the first plan is empty after mending",
			attempts: r.attempts,
			reasks: r.reasks,
			clamped,
			calls: r.calls,
		};
	const boxed = steps.some((s) => Object.keys(s.target_boxes).length);
	return {
		plan: {
			subgoals: steps,
			rationale: r.raw.rationale,
			clauses,
			boxed_on: boxed ? fixed : [],
			...(boxed ? { boxed_at: p.frame.frame_clock_s } : {}),
			boxes_kept_from: {},
			recoverable: true,
		},
		attempts: r.attempts,
		reasks: r.reasks,
		clamped,
		calls: r.calls,
	};
}

/** Whether a `recoverable=false` is let through (spec-planner §4.6). */
export const gaveUp = (rationale: string, replans: number) => replans >= 2 || IRRECOVERABLE.test(rationale);

/** A replan (spec-planner §4.1–4.6): the kept steps plus the new ones; the clauses stay the first plan's. */
export async function replan(ask: Ask, p: ReplanInput, signal?: AbortSignal): Promise<PlanOutcome> {
	const fixed = fixedCameras(p.frame.camera_names);
	const prompt = replanPrompt(p, fixed);
	const imgs = images(p.frame, "marked768");
	const r = await askPlan(ask, "planner", prompt, imgs, {
		first: false,
		fixed,
		cannot: p.cannot,
		budget: p.budget,
		task: p.task,
		signal,
	});
	if (!r.raw) return { error: r.error, attempts: r.attempts, reasks: r.reasks, clamped: [], calls: r.calls };
	const clamped: string[] = [];
	const clauses = p.plan.clauses;
	const kept = p.plan.subgoals.slice(0, p.keep);
	let steps = r.raw.subgoals.map((s) => subgoalOf(s, fixed, p.budget, clauses.length, p.task));
	// Recover missing clauses: claim a new acting step of the same object or name, else put the old steps back without boxes.
	const old = p.plan.subgoals.slice(p.keep);
	for (const [i] of clauses.entries()) {
		if (p.unmet.every((u) => u.i !== i)) continue;
		if (steps.some((s) => s.clause === i)) continue;
		const olds = old.filter((s) => s.clause === i);
		if (!olds.length) continue;
		const claim = steps.find((s) => acts(s) && olds.some((o) => o.name === s.name || sameObject(o.target, s.target)));
		if (claim) {
			clamped.push(`${claim.name}: marked as serving clause ${i}`);
			claim.clause = i;
		} else {
			clamped.push(`clause ${i}: its old steps put back without boxes`);
			steps.push(...olds.map((s) => ({ ...s, target_boxes: {} })));
		}
	}
	if (p.holding) {
		const before = steps.length;
		steps = steps.filter(
			(s) =>
				!(
					GRIPPER_VERBS.test(s.subgoal) &&
					/\bopen|release|let go\b/i.test(s.subgoal) &&
					!/\b(on|in|into|onto|over|at)\b/i.test(s.subgoal) &&
					!s.target
				),
		);
		if (steps.length !== before)
			clamped.push(`${before - steps.length} step(s) that open the gripper with no place to put down removed`);
	}
	if (!steps.length && old.length) {
		clamped.push("nothing left after mending: the old remaining steps put back");
		steps = old.map((s) => ({ ...s }));
	}
	dedupe(steps, p.doneNames.concat(kept.map((s) => s.name)));
	if (p.strict && old.length) {
		const object = old[0]?.target ?? "";
		const allowed = p.tail.filter((s) => sameObject(s.target, object)).length + 1;
		const same = steps.filter((s) => sameObject(s.target, object) && !acts(s));
		if (same.length > allowed) {
			const drop = new Set(same.slice(allowed));
			steps = steps.filter((s) => !drop.has(s));
			clamped.push(`strict: ${drop.size} extra preparation step(s) for ${object} removed`);
		}
	}
	// Keep old boxes for the same object when they were drawn after the last grasp/drop/release (§4.5), unless reboxed.
	const boxes_kept_from: Record<string, number> = {};
	const oldBoxed = p.plan.subgoals.filter((s) => Object.keys(s.target_boxes).length);
	for (const s of steps) {
		if (!s.target || p.rebox.some((r) => sameObject(r, s.target))) continue;
		const prior = oldBoxed.find((o) => sameObject(o.target, s.target));
		if (!prior) continue;
		const drawnAt = p.plan.boxes_kept_from[prior.name] ?? p.plan.boxed_at;
		if (drawnAt === undefined || (p.since !== undefined && drawnAt < p.since)) continue;
		s.target_boxes = { ...prior.target_boxes };
		boxes_kept_from[s.name] = drawnAt;
	}
	const boxedNow = steps.some((s) => Object.keys(s.target_boxes).length && !(s.name in boxes_kept_from));
	const out: Plan = {
		subgoals: [...kept, ...steps],
		rationale: r.raw.rationale,
		clauses,
		boxed_on: boxedNow ? fixed : p.plan.boxed_on,
		...(boxedNow
			? { boxed_at: p.frame.frame_clock_s }
			: p.plan.boxed_at !== undefined
				? { boxed_at: p.plan.boxed_at }
				: {}),
		boxes_kept_from: {
			...Object.fromEntries(
				kept.filter((s) => s.name in p.plan.boxes_kept_from).map((s) => [s.name, p.plan.boxes_kept_from[s.name]]),
			),
			...boxes_kept_from,
		},
		recoverable: r.raw.recoverable,
	};
	const gave_up =
		!r.raw.recoverable && gaveUp(r.raw.rationale, p.replans)
			? r.raw.rationale || "the planner declared the task unrecoverable"
			: undefined;
	return {
		plan: out,
		...(gave_up ? { gave_up } : {}),
		attempts: r.attempts,
		reasks: r.reasks,
		clamped,
		calls: r.calls,
	};
}
