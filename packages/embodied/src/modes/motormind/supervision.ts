/**
 * The motion supervisor, the watcher (spec-executor §15): called once, synchronously, after a batch
 * was stopped (a gate refused mid-batch, a drag, or a segment that was neither done, limited nor
 * contact). It sees the subgoal, the batch in words, the proposer's own `expect`, the measured
 * progress, the robot before and after, and both sets of pictures. Its `decision` is recorded, not
 * acted on; its `finding` limits the next proposal (./gates.ts forbidText). No reply means continue;
 * an invalid reply means stop.
 */

import type { ImageContent } from "@earendil-works/pi-ai";
import type { Ask, Frame, RoleCallRecord, SegmentReport, Supervision } from "./contract.ts";
import { describeRobot, images, recordOf } from "./contract.ts";
import { fill, promptFile } from "./prompts.ts";
import type { Delta } from "./types.ts";

export const FINDINGS = ["on_course", "wrong_direction", "overshoot", "obstacle", "scene_changed", "unsure"] as const;
export type Finding = (typeof FINDINGS)[number];

/** The batch in words: "move N mm" / "push until contact" / "turn N deg" / "gripper open|close". */
export function describeBatch(deltas: readonly Delta[]): string {
	return deltas
		.map((d) => {
			switch (d.kind) {
				case "translate":
					return d.push
						? `push until contact (up to ${Math.round(d.magnitude * 1000)} mm)`
						: `move ${Math.round(d.magnitude * 1000)} mm`;
				case "rotate":
					return `turn ${Math.round((d.magnitude * 180) / Math.PI)} deg`;
				case "gripper":
					return `gripper ${d.gripper_state}`;
				case "wait":
					return `wait ${d.seconds ?? 0} s`;
				default:
					return "home";
			}
		})
		.join("; ");
}

/** The progress line: the last segment's motion against what the batch asked in all. */
export function describeProgress(steps: readonly SegmentReport[], deltas: readonly Delta[]): string {
	const last = steps.at(-1);
	const asked = deltas.filter((d) => d.kind === "translate").reduce((s, d) => s + d.magnitude * 1000, 0);
	if (!last) return `Nothing ran yet; the batch asked for ${Math.round(asked)} mm of motion in all.`;
	return `The last segment (${last.label}) moved ${last.moved_mm.toFixed(1)} mm (${last.along_axis_mm.toFixed(1)} mm along its axis, ${last.lateral_mm.toFixed(1)} mm sideways) and turned ${last.turned_deg.toFixed(1)} deg, ending ${last.outcome}${last.abort_reason ? ` (${last.abort_reason})` : ""}; the batch asked for ${Math.round(asked)} mm of motion in all.`;
}

/** The reply as a `Supervision`, or undefined when it is not one (extra keys, unknown values). */
export function parseSupervision(json: unknown): Supervision | undefined {
	if (!json || typeof json !== "object" || Array.isArray(json)) return undefined;
	const o = json as Record<string, unknown>;
	const allowed = new Set(["decision", "finding", "because", "evidence"]);
	if (Object.keys(o).some((k) => !allowed.has(k))) return undefined;
	const decision = String(o.decision ?? "").toLowerCase();
	const finding = String(o.finding ?? "").toLowerCase();
	if (decision !== "continue" && decision !== "stop") return undefined;
	if (!(FINDINGS as readonly string[]).includes(finding)) return undefined;
	return { decision, finding, because: String(o.because ?? "") };
}

export type SupervisionInput = {
	ask: Ask;
	subgoal: string;
	deltas: readonly Delta[];
	expect: string;
	steps: readonly SegmentReport[];
	before: Frame;
	after: Frame;
	legend: string;
	/** The frame clock of the last frame judged; a frame no newer is not judged again. */
	judgedUpTo: number;
	signal?: AbortSignal;
};

/**
 * One watcher call. Returns the supervision, with `judged` = the frame clock now judged, or
 * `undefined` when the after frame is not newer than the last judged one (then nothing is asked).
 */
export async function supervise(
	o: SupervisionInput,
): Promise<{ supervision: Supervision; record?: RoleCallRecord; judged: number; failure?: string } | undefined> {
	if (!(o.after.frame_clock_s > o.judgedUpTo)) return undefined;
	const p = promptFile("supervise");
	const text = fill(p.text, {
		subgoal: o.subgoal,
		batch: describeBatch(o.deltas),
		expect: o.expect || "(the proposer wrote no expectation)",
		progress: describeProgress(o.steps, o.deltas),
		robot: describeRobot(o.after),
		legend: o.legend,
	});
	const pics: ImageContent[] = [...images(o.before, "marked512"), ...images(o.after, "marked512")];
	let reply: Awaited<ReturnType<Ask>>;
	try {
		reply = await o.ask("supervise", { system: p.system, text }, pics, { signal: o.signal, maxTokens: 1500 });
	} catch (err) {
		const failure = err instanceof Error ? err.message : String(err);
		return {
			supervision: { decision: "continue", finding: "unsure", because: `no reply (${failure})` },
			judged: o.after.frame_clock_s,
			failure,
		};
	}
	const record = recordOf("supervise", reply);
	if (!reply.ok)
		return {
			supervision: { decision: "continue", finding: "unsure", because: `no reply (${reply.error_kind})` },
			record,
			judged: o.after.frame_clock_s,
			failure: reply.error,
		};
	const parsed = parseSupervision(reply.json);
	if (!parsed)
		return {
			supervision: { decision: "stop", finding: "unsure", because: "the reply was not a valid supervision" },
			record,
			judged: o.after.frame_clock_s,
			failure: "invalid reply",
		};
	return { supervision: parsed, record, judged: o.after.frame_clock_s };
}
