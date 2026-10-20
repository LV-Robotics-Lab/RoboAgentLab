/**
 * Object names and the place phrases inside them (spec-executor §9 "找什么", spec-planner §5.4 rule 7
 * "核心名词"). The planner's vocabulary writes an object with where it stands — "the patterned bowl
 * sitting on the grey ramekin", "the akita black bowl next to the ramekin" — and repeats that name in
 * every criterion. The executor and the verifier key on the head: the words before the first place
 * phrase. A criterion is classified (holding / release / the receiver's words) with the place phrases
 * that describe the held thing taken out, else "sitting on the ramekin" reads as a release and the
 * ramekin as the receiver (step 4 round 2 defect 3).
 *
 * Derived from MotorMind (Motor-Mind/MotorMind-Code @3732bf3), used under the MIT License; see NOTICE.
 */

/** Place prepositions that start a phrase describing where a thing stands. */
const PLACE =
	"(?:on top of|next to|close to|in front of|on|onto|in|inside|into|at|from|beside|between|near|behind|under|underneath|beneath|atop|upon|by|over|above|against|along)";
/** Participles that introduce a place description of a thing ("sitting on …"); a finite "sits on" is a state, not a name. */
const PARTICIPLE =
	"(?:sitting|resting|standing|lying|located|situated|positioned|placed|kept|found|that is|which is|that sits|that rests)";
/** A place phrase inside a name is followed by a verb (the subject goes on) or ends the name. */
const VERB = "(?:is|are|was|were|has|have|sits|rests|lies|stands|hangs|remains|reports|still)";
const STOP = "(?:and|with|while|so|until|then|but|or)";

const bare = (s: string) => s.replace(/\s+/g, " ").trim();
const words = (s: string) => s.toLowerCase().match(/[a-z][a-z'_-]*/g) ?? [];

/** Where the first place phrase starts inside a name (character index), or -1. */
function placeStart(name: string): number {
	const re = new RegExp(`\\s+(?:${PARTICIPLE}\\s+)?${PLACE}\\s+(?:the|a|an|its|that|this)\\b`, "i");
	const m = re.exec(name);
	return m ? m.index : -1;
}

/**
 * The head of an object's name: the words before its first place phrase ("the patterned bowl sitting
 * on the grey ramekin" → "the patterned bowl"; "the akita black bowl next to the ramekin" → "the akita
 * black bowl"). A name that is only a place phrase, or has none, is returned whole. "with" is not a
 * place: "the bottle with the red cap" keeps its cap.
 */
export function objectHead(name: string): string {
	const t = bare(name);
	const at = placeStart(t);
	if (at <= 0) return t;
	const head = t.slice(0, at).trim();
	return words(head).length ? head : t;
}

/** The place tail of a name ("sitting on the grey ramekin"), or "" when it has none. */
export function placeTail(name: string): string {
	const t = bare(name);
	const at = placeStart(t);
	return at <= 0 ? "" : t.slice(at).trim();
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");

/**
 * The text with the place phrases that describe the held thing taken out: the target's own tail
 * wherever it appears, a participial place phrase ("sitting on the grey ramekin"), and a bare place
 * phrase a verb follows ("the bowl on the stove is over the plate" → "the bowl is over the plate").
 * "between the plate and the ramekin" goes as one phrase. A finite state stays: "the bowl rests on the
 * plate", "the bowl is sitting on the plate" are what a put step asks for.
 */
export function stripPlace(text: string, target = ""): string {
	let t = bare(text);
	const tail = placeTail(target);
	if (tail) t = t.replace(new RegExp(`\\s*${escapeRe(tail)}`, "gi"), "");
	const NP = `(?:the|a|an|its|that|this)\\s+[a-z0-9'’-]+(?:\\s+(?!${VERB}\\b|${STOP}\\b|${PLACE}\\b)[a-z0-9'’-]+)*`;
	const BETWEEN = `between\\s+${NP}\\s+and\\s+${NP}`;
	// 2. a participial place phrase, not a progressive one ("is sitting on" is a state)
	t = t.replace(
		new RegExp(`(?<!\\b(?:is|are|was|were|still))\\s+${PARTICIPLE}\\s+(?:${BETWEEN}|${PLACE}\\s+${NP})`, "gi"),
		"",
	);
	// 3. a bare place phrase inside a subject, which a verb follows
	t = t.replace(new RegExp(`\\s+(?:${BETWEEN}|${PLACE}\\s+${NP})(?=\\s+${VERB}\\b)`, "gi"), "");
	return bare(t);
}
