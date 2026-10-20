/**
 * What one subgoal hands the next (spec-executor §16): the payload and the receiver always; the
 * target's measurements (residual, error, width, footprint, located point, disagreement, top face,
 * empty closes, frozen count) only when the next subgoal names the very same target, whose planner
 * boxes are then dropped (the look is newer than the plan). Within one mission only: a new episode is
 * a new pi process.
 */

import type { Carry } from "./contract.ts";
import type { SubgoalState } from "./gates.ts";
import type { TopFace } from "./observe.ts";
import type { Vec3 } from "./types.ts";

export type CarryState = {
	payload: SubgoalState["payload"];
	receiver?: SubgoalState["receiver"];
	target: string;
	/** Only adopted by a subgoal with the same target words. */
	same?: {
		residualMm?: number;
		errorMm?: number;
		errorClass?: string;
		lastWidthMm?: number;
		footprintError?: number;
		unshoved: boolean;
		located?: SubgoalState["located"];
		disagreement?: string;
		footprint?: Vec3[];
		footprintMethod?: string;
		footprintIsPart: boolean;
		faces: ReturnType<TopFace["carry"]>;
		emptyCloses: Vec3[];
		frozen: number;
		residualText?: string;
	};
};

/** The state a finished subgoal carries over (`SubgoalRun.carry`). */
export function carryOver(s: SubgoalState, faces: TopFace): Carry {
	const c: CarryState = {
		payload: { ...s.payload },
		...(s.receiver ? { receiver: { ...s.receiver } } : {}),
		target: s.target,
		same: {
			residualMm: s.residualMm,
			errorMm: s.errorMm,
			errorClass: s.errorClass,
			lastWidthMm: s.lastWidthMm,
			footprintError: s.footprintError,
			unshoved: s.unshoved,
			located: s.located ? { ...s.located } : undefined,
			disagreement: s.disagreement,
			footprint: s.footprint ? [...s.footprint] : undefined,
			footprintMethod: s.footprintMethod,
			footprintIsPart: s.footprintIsPart,
			faces: faces.carry(),
			emptyCloses: [...s.emptyCloses],
			// Only this subgoal's run of frozen cycles (spec-executor §16): the count does not accumulate across steps.
			frozen: s.frozen,
			residualText:
				s.residualMm !== undefined
					? `the last step measured the tool ${Math.round(s.residualMm)} mm from the target`
					: undefined,
		},
	};
	return c as unknown as Carry;
}

/**
 * Adopt a carry into a fresh state: the payload and receiver always; the target's measurements only
 * for the same target words, whose planner boxes are then dropped. Returns the `carried` event text.
 */
export function adopt(s: SubgoalState, carry: Carry | undefined, faces: TopFace): string | undefined {
	if (!carry) return undefined;
	const c = carry as unknown as CarryState;
	if (c.payload) s.payload = { ...c.payload };
	if (c.receiver) s.receiver = { ...c.receiver };
	const same = c.same && c.target.trim().toLowerCase() === s.target.trim().toLowerCase();
	if (!same || !c.same) return `carried the payload${c.receiver ? " and the receiver" : ""} from the last step`;
	const m = c.same;
	s.residualMm = m.residualMm;
	s.errorMm = m.errorMm;
	s.errorClass = m.errorClass;
	s.lastWidthMm = m.lastWidthMm;
	s.footprintError = m.footprintError;
	s.unshoved = m.unshoved;
	s.located = m.located ? { ...m.located } : undefined;
	s.disagreement = m.disagreement;
	s.footprint = m.footprint ? [...m.footprint] : undefined;
	s.footprintMethod = m.footprintMethod;
	s.footprintIsPart = m.footprintIsPart === true;
	s.emptyCloses = [...m.emptyCloses];
	s.inheritedFrozen = m.frozen;
	if (m.faces.faces?.length) faces.faces = [...m.faces.faces];
	if (m.faces.top_point_mm !== undefined) faces.fromTopPoint = m.faces.top_point_mm;
	// The plan's boxes are older than the last look.
	s.planBoxes = {};
	return m.residualText ?? "carried the last step's measurements of the same target";
}
