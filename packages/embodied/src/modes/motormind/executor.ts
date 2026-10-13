/**
 * STUB — the executor half of M2 (worktree mm-m2-executor) writes this module; the integrator takes
 * that file over this one wholesale. It exists so the mission half compiles and its tests inject
 * their own executors (./contract.ts `SubgoalInput` → `Executor`).
 */

import type { Executor, SubgoalInput } from "./contract.ts";

// biome-ignore lint/correctness/useYield: a stub that ends the subgoal without a robot call
export async function* runSubgoal(input: SubgoalInput): Executor {
	return {
		cycles: [],
		ending: "ERROR",
		ending_text: "the executor is not built into this checkout",
		error: `executor missing for ${input.subgoal.name}`,
		episode_over: false,
		carry: input.carry ?? {},
		observes: [],
		timing: { wall_s: 0, motion_s: 0, calls: [] },
	};
}
