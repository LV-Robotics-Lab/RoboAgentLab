/**
 * The clause ledger (spec-planner §0.5, §3.7): the first plan splits the task into clauses, fixed for
 * the mission; every verdict answers them one by one; the mission keeps which are met, which were
 * withdrawn, and which a code event (a measured push or turn) supports. A withdrawn clause is not
 * met again on a model's word alone until a code event happened in between (`refused_mark`); a
 * measured miss (a release off its mark) is not undone by a later model answer.
 */

import type { ClauseAnswer } from "./planner.ts";

/** How much of a clause's content words a verdict line must share to count as answering it. */
export const SAME_CLAUSE_SHARE = 0.5;
/** A horizontal push that went this far along its axis is a code event (spec-planner §3.7). */
export const PUSHED_ALONG_MM = 5;
/** A wrist turn this large on a "turn" clause is a code event. */
export const TURNED_DEG = 1;

const STOP = new Set([
	"the",
	"a",
	"an",
	"and",
	"or",
	"of",
	"to",
	"in",
	"on",
	"into",
	"onto",
	"at",
	"it",
	"its",
	"is",
	"be",
	"with",
	"from",
	"by",
	"for",
	"then",
	"that",
	"this",
	"up",
	"down",
	"put",
	"place",
	"pick",
]);
/** The content words of a clause or a verdict line (lower case, no stop words, 3+ letters). */
export function contentWords(s: string): Set<string> {
	return new Set(
		s
			.toLowerCase()
			.split(/[^a-z0-9_]+/)
			.filter((w) => w.length >= 3 && !STOP.has(w)),
	);
}

/**
 * The clause each verdict line answers (index into `clauses`): by the line's own `index` when the
 * verifier gave one; else by content-word share (threshold `SAME_CLAUSE_SHARE`), else by position
 * when the counts match. A line without an index whose words restate this step's `criterion` more
 * than the clause it would land on is not an answer about the task and is dropped (step 4 round 2
 * defect 1: "the bowl sits directly under the gripper" marked "the bowl is placed on the plate" met at
 * the hover step in 32 of 40 cells, and the replan then deleted the take and lift steps).
 */
export function alignAnswers(
	clauses: readonly string[],
	answers: readonly ClauseAnswer[],
	criterion?: string,
): (number | undefined)[] {
	const words = clauses.map(contentWords);
	const crit = criterion ? contentWords(criterion) : undefined;
	let dropped = false;
	let indexed = false;
	const matched = answers.map((a) => {
		if (a.index !== undefined && Number.isInteger(a.index) && a.index >= 0 && a.index < clauses.length) {
			indexed = true;
			return a.index;
		}
		const line = contentWords(`${a.what} ${a.evidence}`);
		let best: { i: number; share: number } | undefined;
		words.forEach((w, i) => {
			if (!w.size) return;
			let hit = 0;
			for (const x of w) if (line.has(x)) hit++;
			const share = hit / w.size;
			if (share >= SAME_CLAUSE_SHARE && (!best || share > best.share)) best = { i, share };
		});
		if (best && crit && echoesCriterion(a, words[best.i], crit)) {
			dropped = true;
			return undefined;
		}
		return best?.i;
	});
	if (!dropped && !indexed && matched.every((m) => m === undefined) && answers.length === clauses.length)
		return answers.map((_, i) => i);
	return matched;
}

/** The line's own words (its `what`, else its evidence) sit more in the step criterion than in the clause. */
function echoesCriterion(a: ClauseAnswer, clause: Set<string>, criterion: Set<string>): boolean {
	const line = contentWords(a.what || a.evidence);
	if (!line.size) return false;
	let inClause = 0;
	let inCriterion = 0;
	for (const w of line) {
		if (clause.has(w)) inClause++;
		if (criterion.has(w)) inCriterion++;
	}
	return inCriterion > inClause;
}

export type ClauseState = {
	clauses: string[];
	met: Set<number>;
	withdrawn: Set<number>;
	/** A code event supports the clause: a measured push or turn since its last withdrawal. */
	events: Set<number>;
	/** A measured miss (release off its mark): not undone by a model's answer. */
	measured_unmet: Set<number>;
	refused_mark: number;
	withdrawals: { clause: number; because: string }[];
};

export const emptyClauses = (clauses: string[] = []): ClauseState => ({
	clauses,
	met: new Set(),
	withdrawn: new Set(),
	events: new Set(),
	measured_unmet: new Set(),
	refused_mark: 0,
	withdrawals: [],
});

/**
 * Fold a verdict's clause answers into the ledger (spec-planner §3.7 `_fold`). `allowWithdraw` is
 * true for the sources that may unmark: a step judged task_done, a step's whole-task question, the
 * final check. `criterion` is the step's, for a step verdict: lines that restate it are not answers
 * about the task (see `alignAnswers`).
 */
export function fold(
	state: ClauseState,
	answers: readonly ClauseAnswer[],
	allowWithdraw: boolean,
	criterion?: string,
): void {
	const at = alignAnswers(state.clauses, answers, criterion);
	answers.forEach((a, k) => {
		const i = at[k];
		if (i === undefined) return;
		if (a.met) {
			if (state.measured_unmet.has(i)) return;
			if (state.withdrawn.has(i) && !state.events.has(i)) {
				state.refused_mark++;
				return;
			}
			state.met.add(i);
			state.withdrawn.delete(i);
		} else if (allowWithdraw && state.met.has(i)) {
			// A code event behind the clause survives the withdrawal: a later answer may mark it again.
			state.met.delete(i);
			state.withdrawn.add(i);
			state.withdrawals.push({ clause: i, because: a.evidence || a.what });
		}
	});
}

/** A code event for clause `i`: clears its withdrawal, lets a later answer mark it met. */
export function codeEvent(state: ClauseState, i: number): void {
	if (i < 0 || i >= state.clauses.length) return;
	state.events.add(i);
	state.withdrawn.delete(i);
}

/** A measured miss for clause `i` (a release off its mark): unmet, and not re-markable by a model. */
export function measuredMiss(state: ClauseState, i: number): void {
	if (i < 0 || i >= state.clauses.length) return;
	state.met.delete(i);
	state.events.delete(i);
	state.measured_unmet.add(i);
}

/** A measured hit for clause `i` (a release within its mark): the code settled it. */
export function measuredHit(state: ClauseState, i: number): void {
	if (i < 0 || i >= state.clauses.length) return;
	state.measured_unmet.delete(i);
	codeEvent(state, i);
	state.met.add(i);
}

export const allMet = (s: ClauseState) => s.clauses.length > 0 && s.clauses.every((_, i) => s.met.has(i));
export const unmet = (s: ClauseState) => s.clauses.map((c, i) => ({ i, text: c })).filter(({ i }) => !s.met.has(i));

/** Whether a clause's words ask for a turn (spec-planner §3.7: the turn event applies to these). */
export const isTurnClause = (text: string) => /\b(turn|twist|rotate|rotat|knob|dial|screw|unscrew)\w*\b/i.test(text);

export const clausesResult = (s: ClauseState) => ({
	clauses: s.clauses,
	met: [...s.met].sort((a, b) => a - b),
	withdrawn: [...s.withdrawn].sort((a, b) => a - b),
	events: [...s.events].sort((a, b) => a - b),
	measured_unmet: [...s.measured_unmet].sort((a, b) => a - b),
	refused_mark: s.refused_mark,
	withdrawals: s.withdrawals,
});
