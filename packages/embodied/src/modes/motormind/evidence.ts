/**
 * The mission's evidence log (spec-planner §2.7, §8): one line per thing that happened, append-only
 * except that a cycle's line takes the measurements of its later `step_end` and `reached` events.
 * Lines render as `[kind] subgoal: text (cycle=3 along_axis_mm=48 ...)`. The verifier's claim, the
 * memory summarizer (M3) and the monitor's "last 3 lines" (M3) read it.
 */

import type { ExecutorEvent, SegmentReport } from "./contract.ts";

export type EvidenceKind = "cycle" | "supervision" | "alert" | "verdict" | "plan" | "note";
export type EvidenceLine = {
	t: number;
	subgoal: string;
	kind: EvidenceKind;
	text: string;
	data: Record<string, unknown>;
};

const NUMBERS = ["cycle", "along_axis_mm", "lateral_mm", "turned_deg", "clearance_mm", "residual_mm"] as const;
const r1 = (v: number) => Number(v.toFixed(1));

/** `[kind] subgoal: text (k=v ...)` with the numeric fields that are set. */
export function renderLine(l: EvidenceLine): string {
	const nums = NUMBERS.filter((k) => typeof l.data[k] === "number").map((k) => `${k}=${r1(l.data[k] as number)}`);
	return `[${l.kind}] ${l.subgoal}: ${l.text}${nums.length ? ` (${nums.join(" ")})` : ""}`;
}

/** One step as the cycle line says it: "move down 50 mm → 48 mm along, 1.2 mm sideways" style. */
export function stepText(s: SegmentReport): string {
	const head = s.label;
	if (s.kind === "translate") {
		const parts = [`${r1(s.along_axis_mm)} mm along`];
		if (s.lateral_mm >= 1) parts.push(`${r1(s.lateral_mm)} mm sideways`);
		return `${head} → ${parts.join(", ")}${s.outcome !== "done" ? ` [${s.outcome}${s.abort_reason ? `: ${s.abort_reason}` : ""}]` : ""}`;
	}
	if (s.kind === "rotate")
		return `${head} → turned ${r1(s.turned_deg)} deg${s.outcome !== "done" ? ` [${s.outcome}]` : ""}`;
	const opening = s.detail?.opening_mm !== undefined ? `, opening ${r1(s.detail.opening_mm)} mm` : "";
	const held = s.grasp ? `, holding ${s.grasp.holding === null ? "unknown" : s.grasp.holding}` : "";
	return `${head} → ${s.outcome}${opening}${held}`;
}

export class EvidenceLog {
	readonly lines: EvidenceLine[] = [];
	/** The cycle lines of the current attempt, by cycle index (reset per attempt by `cycle 0`). */
	private cycles = new Map<number, EvidenceLine>();

	add(kind: EvidenceKind, subgoal: string, text: string, data: Record<string, unknown> = {}): EvidenceLine {
		const line: EvidenceLine = { t: Date.now(), subgoal, kind, text, data };
		this.lines.push(line);
		return line;
	}

	/** The rendered lines from `from` (a line index) on. */
	render(from = 0): string[] {
		return this.lines.slice(from).map(renderLine);
	}

	/** The last `n` cycle/supervision lines of `subgoal` (the monitor's recent context). */
	recent(subgoal: string, n = 3): string[] {
		return this.lines
			.filter((l) => l.subgoal === subgoal && (l.kind === "cycle" || l.kind === "supervision"))
			.slice(-n)
			.map(renderLine);
	}

	/** Fold one executor event into the log (spec-planner §8). */
	event(subgoal: string, e: ExecutorEvent): void {
		switch (e.type) {
			case "proposed": {
				if (e.cycle === 0) this.cycles = new Map();
				const command = e.proposal.deltas.map((d) => d.label).join("; ") || "(no action)";
				const line = this.add("cycle", subgoal, `cycle ${e.cycle}: ${command}`, {
					cycle: e.cycle,
					assessment: e.proposal.assessment,
					claimed_done: e.proposal.done,
					steps: [] as string[],
				});
				this.cycles.set(e.cycle, line);
				return;
			}
			case "step_end": {
				const line = this.cycles.get(e.cycle);
				if (!line) return;
				const d = line.data;
				(d.steps as string[]).push(stepText(e.step));
				if (e.step.kind === "translate") {
					d.along_axis_mm = ((d.along_axis_mm as number) ?? 0) + e.step.along_axis_mm;
					d.lateral_mm = ((d.lateral_mm as number) ?? 0) + e.step.lateral_mm;
				}
				if (e.step.kind === "rotate") d.turned_deg = ((d.turned_deg as number) ?? 0) + e.step.turned_deg;
				line.text = `${line.text.split(" → measured: ")[0]} → measured: ${(d.steps as string[]).join("; ")}`;
				return;
			}
			case "reached": {
				const line = this.cycles.get(e.cycle);
				if (!line) return;
				const l = e.reach.located;
				line.data.residual_mm = e.reach.residual_mm;
				line.data.located = l?.point_mm
					? `${e.reach.target} at [${l.point_mm.map((v) => v.toFixed(0)).join(", ")}] mm${l.error_mm !== undefined ? ` ± ${l.error_mm.toFixed(0)}` : ""}${l.doubtful ? " (doubtful)" : ""}`
					: `could not find ${e.reach.target}${l?.error ? `: ${l.error}` : ""}`;
				return;
			}
			case "supervision":
				if (e.supervision.decision === "stop")
					this.add("supervision", subgoal, `stopped: ${e.supervision.finding} — ${e.supervision.because}`, {
						cycle: e.cycle,
						finding: e.supervision.finding,
					});
				return;
			case "limited":
			case "refused":
				this.add("cycle", subgoal, `${e.type}: ${e.why}`);
				return;
			case "step_error":
				this.add("cycle", subgoal, `error: ${e.error}`);
				return;
			case "batch_stopped": {
				const line = this.cycles.get(e.cycle);
				if (line) line.data.stopped = `batch stopped at action ${e.at}: ${e.reason}`;
				return;
			}
			default:
				return;
		}
	}
}
