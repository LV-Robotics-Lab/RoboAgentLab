/**
 * The proposer (spec-executor §8): one model call per try, up to 3 tries on the same pictures, each
 * reply validated (the Proposal's five fields, no extras), converted (./command.ts) and gated
 * (./gates.ts) before it counts; a refused try is re-asked with the reason appended. An empty
 * command is accepted as "no actions". A reply that claims done and whose actions were refused three
 * times falls back to the claim alone.
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import { CommandError, toDeltas } from "./command.ts";
import { type Ask, type Cycle, type Proposal, type RoleCallRecord, recordOf } from "./contract.ts";
import { fill, promptFile } from "./prompts.ts";
import { type Caps, type Delta, MAX_COMPOSITE_MM, MAX_PUSH_MM, MAX_TURN_DEG } from "./types.ts";

export const PROPOSAL_TRIES = 3;
/** The history keeps this many cycles, and a rejection reason this many characters. */
export const HISTORY_CYCLES = 3;
export const REASON_CHARS = 220;
/** The schema text is cut at this many characters, as the original's compact_json does. */
export const SCHEMA_CHARS = 1800;

/** The command language as the model is told it (spec-executor §3.4): shapes, fields, enums and limits; no direction words. */
export function commandSchemaText(): string {
	const text = [
		'A command is {"actions": [action, ...]} (top-level "frame", "speed_mm_s", "rot_speed_deg_s", "stop_on_contact", "note" apply to every action unless it sets its own), a single action, or a list of actions.',
		'move: {"type": "move", "direction": word | "axis": [x, y, z], "distance_mm": > 0}; optional speed_mm_s (clamped to the robot), stop_on_contact (a push).',
		'rotate: {"type": "rotate", "direction": word | "axis": [x, y, z], "angle_deg": > 0}; optional rot_speed_deg_s.',
		'gripper: {"type": "gripper", "state": "open" | "close"}; optional width_mm.',
		'home: {"type": "home"}. wait: {"type": "wait", "seconds": > 0}.',
		'Enums: type in move, rotate, gripper, home, wait; frame is always "base"; state in open, close.',
		`Limits: one command's moves end within ${MAX_COMPOSITE_MM} mm of where it started and its rotations add up to ${MAX_TURN_DEG} deg at most; a push moves at most ${MAX_PUSH_MM} mm. Distances and angles are magnitudes: to go the other way use the opposite direction word.`,
	].join("\n");
	return text.length > SCHEMA_CHARS ? text.slice(0, SCHEMA_CHARS) : text;
}

export type Validated = { proposal: Omit<Proposal, "deltas">; problems: string[] };

/** The reply's fields checked (spec-executor §8.6): assessment, done, command, expect, confidence; no extra keys; up to 6 problems named. */
export function validateProposal(json: unknown): Validated {
	const problems: string[] = [];
	if (!json || typeof json !== "object" || Array.isArray(json))
		return {
			proposal: { assessment: "", done: false, expect: "", confidence: 0, command: null },
			problems: ["the reply is not a JSON object"],
		};
	const o = json as Record<string, unknown>;
	const allowed = new Set(["assessment", "done", "command", "expect", "confidence"]);
	for (const k of Object.keys(o)) if (!allowed.has(k)) problems.push(`unknown key "${k}"`);
	if (typeof o.done !== "boolean") problems.push('"done" must be true or false');
	if (o.assessment !== undefined && typeof o.assessment !== "string") problems.push('"assessment" must be text');
	if (o.expect !== undefined && typeof o.expect !== "string") problems.push('"expect" must be text');
	let confidence = 0;
	if (o.confidence !== undefined) {
		const c = Number(o.confidence);
		if (!Number.isFinite(c)) problems.push('"confidence" must be a number');
		else confidence = Math.max(0, Math.min(1, c));
	}
	return {
		proposal: {
			assessment: typeof o.assessment === "string" ? o.assessment : "",
			done: o.done === true,
			expect: typeof o.expect === "string" ? o.expect : "",
			confidence,
			command: o.command ?? null,
		},
		problems: problems.slice(0, 6),
	};
}

/** The command has no actions: null, empty, or an object with an empty `actions`. */
export function emptyCommand(command: unknown): boolean {
	if (command === null || command === undefined || command === "") return true;
	if (Array.isArray(command)) return command.length === 0;
	if (typeof command === "object") {
		const o = command as Record<string, unknown>;
		if ("actions" in o) return !Array.isArray(o.actions) || o.actions.length === 0;
		return Object.keys(o).length === 0;
	}
	return false;
}

export type ProposalContext = {
	ask: Ask;
	subgoal: string;
	target: string;
	criterion: string;
	history: string;
	robot: string;
	capabilities: string;
	images: ImageContent[];
	imageNames: string[];
	legend: string;
	forbid: string;
	caps: Caps;
	/** The gates on a converted batch; the first refusal, or undefined. May mutate the deltas. */
	gate: (deltas: Delta[]) => string | undefined;
	signal?: AbortSignal;
};

export type Proposed = {
	proposal?: Proposal;
	deltas: Delta[];
	/** Each try's reply text and why it was refused. */
	tries: { text?: string; refused?: string }[];
	records: RoleCallRecord[];
	/** Every try failed: the error. */
	error?: string;
	/** The claim was kept and its refused actions dropped. */
	claimOnly?: boolean;
};

/** The slots of the executor prompt (spec-executor §8.1). */
export function proposalPrompt(c: ProposalContext, refusals: string[]): { system: string; text: string } {
	const p = promptFile("executor");
	let text = fill(p.text, {
		subgoal: c.subgoal,
		target: c.target || "(not given: find it from the subgoal's words)",
		criterion: c.criterion || "(not given: judge by the subgoal)",
		history: c.history || "First attempt at this subgoal.",
		robot: c.robot,
		capabilities: c.capabilities,
		schema: commandSchemaText(),
		images: c.imageNames.map((n) => `${n} (now)`).join(", "),
		legend:
			c.legend || "No legend: do not infer directions from the pictures; the executor's own geometry gives them.",
		forbid: c.forbid ? `LIMITS ON THIS MOTION:\n${c.forbid}\n` : "",
	});
	for (const r of refusals) text += `\n\nThe previous answer was refused: ${r}`;
	return { system: p.system, text };
}

/** Up to 3 tries on the same pictures (spec-executor §8.6). */
export async function propose(c: ProposalContext): Promise<Proposed> {
	const out: Proposed = { deltas: [], tries: [], records: [] };
	const refusals: string[] = [];
	let claim: Omit<Proposal, "deltas"> | undefined;
	for (let attempt = 0; attempt < PROPOSAL_TRIES; attempt++) {
		const prompt = proposalPrompt(c, refusals);
		let reply: Awaited<ReturnType<Ask>>;
		try {
			reply = await c.ask("executor", prompt, c.images, { signal: c.signal, maxTokens: 1500 });
		} catch (err) {
			const why = `the model call failed: ${err instanceof Error ? err.message : String(err)}`;
			out.tries.push({ refused: why });
			refusals.push(why);
			continue;
		}
		out.records.push(recordOf("executor", reply));
		if (!reply.ok) {
			const why =
				reply.error_kind === "no_json"
					? "the reply had no JSON object"
					: `the model call failed (${reply.error_kind}): ${reply.error}`;
			out.tries.push({ text: reply.text, refused: why });
			refusals.push(why);
			continue;
		}
		const v = validateProposal(reply.json);
		if (v.problems.length) {
			const why = `the reply's fields: ${v.problems.join("; ")}`;
			out.tries.push({ text: reply.text, refused: why });
			refusals.push(why);
			continue;
		}
		if (emptyCommand(v.proposal.command)) {
			out.tries.push({ text: reply.text });
			out.proposal = { ...v.proposal, deltas: [] };
			return out;
		}
		let deltas: Delta[];
		try {
			deltas = toDeltas(v.proposal.command, c.caps);
		} catch (err) {
			const why =
				err instanceof CommandError
					? err.message
					: `the command could not be converted: ${err instanceof Error ? err.message : String(err)}`;
			out.tries.push({ text: reply.text, refused: why });
			refusals.push(why);
			if (v.proposal.done) claim = v.proposal;
			continue;
		}
		const refused = c.gate(deltas);
		if (refused) {
			out.tries.push({ text: reply.text, refused });
			refusals.push(refused);
			if (v.proposal.done) claim = v.proposal;
			continue;
		}
		out.tries.push({ text: reply.text });
		out.deltas = deltas;
		out.proposal = { ...v.proposal, deltas: deltas.map(summary) };
		return out;
	}
	if (claim) {
		out.proposal = { ...claim, command: null, deltas: [] };
		out.claimOnly = true;
		return out;
	}
	out.error = `no acceptable proposal in ${PROPOSAL_TRIES} tries: ${refusals.at(-1) ?? "no reply"}`;
	return out;
}

/** A delta as the proposal records it (spec-planner §2.9 reads kind, label, axis, push). */
export const summary = (d: Delta): Proposal["deltas"][number] => ({
	kind: d.kind,
	label: d.label,
	...(d.axis ? { axis: d.axis } : {}),
	push: d.push,
});

/** The history (spec-executor §8.4): the mission's preface, then the last 3 cycles. */
export function historyText(preface: string, cycles: readonly Cycle[]): string {
	const lines: string[] = [];
	if (preface.trim()) lines.push(preface.trim());
	for (const c of cycles.slice(-HISTORY_CYCLES)) {
		const part: string[] = [`Cycle ${c.index}:`];
		const refused = c.error && /refused|no acceptable/.test(c.error) ? c.error : undefined;
		if (refused) part.push(`  last refusal: ${refused.slice(0, REASON_CHARS)}`);
		if (c.reach)
			part.push(
				`  located: ${c.reach.located?.error ? `could not find "${c.reach.target}": ${c.reach.located.error}` : `"${c.reach.target}"${c.reach.residual_mm !== undefined ? ` ${Math.round(c.reach.residual_mm)} mm from the tool` : ""}${c.reach.located?.doubtful ? " (doubtful)" : ""}`}`,
			);
		if (c.proposal) {
			part.push(`  proposed: ${JSON.stringify(c.proposal.command)}`);
			const measured = c.steps.map((s) => {
				if (s.kind === "translate")
					return `${s.label}: ${s.along_axis_mm.toFixed(1)} mm along${s.lateral_mm >= 1 ? `, ${s.lateral_mm.toFixed(1)} mm sideways` : ""}${s.outcome !== "done" ? ` (${s.outcome}${s.abort_reason ? `, ${s.abort_reason}` : ""})` : ""}`;
				if (s.kind === "rotate")
					return `${s.label}: turned ${s.turned_deg.toFixed(1)} deg${s.outcome !== "done" ? ` (${s.outcome})` : ""}`;
				return `${s.label}: ${s.message ?? s.outcome}`;
			});
			if (measured.length) part.push(`  measured: ${measured.join("; ")}`);
			const notes = c.steps.map((s) => s.note).filter(Boolean);
			if (notes.length) part.push(`  notes: ${[...new Set(notes)].join("; ")}`);
		}
		if (c.batch?.stopped_at !== undefined)
			part.push(`  the batch was stopped at action ${c.batch.stopped_at + 1}: ${c.batch.reason ?? ""}`);
		else if (c.supervision)
			part.push(`  the motion was stopped; the watcher found: ${c.supervision.finding} (${c.supervision.because})`);
		if (c.stopped_by) part.push(`  stopped by: ${c.stopped_by}`);
		if (
			(c.stopped_by ?? "").includes("refused") ||
			(c.batch?.reason ?? "").includes("refused") ||
			c.steps.some((s) => s.outcome === "refused")
		)
			part.push("  out of reach: do not propose that direction again.");
		lines.push(part.join("\n"));
	}
	return lines.join("\n");
}
