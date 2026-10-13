/**
 * STUB — the executor half of M2 (worktree mm-m2-executor) writes this module (the observe wrapper,
 * the ClearanceTracker, the top-surface estimate, `legend`, `capabilityText`); the integrator takes
 * that file over this one wholesale. The two exports here are the contract's `Describers` with the
 * capability text written from spec-executor §8.3 and a legend that names the aim marker's place.
 */

import type { Frame } from "./contract.ts";
import type { Caps } from "./types.ts";

const r0 = (v: number) => v.toFixed(0);

/** spec-executor §8.3: the step limits, the speed cap, contact stop, home, the tilted floor. */
export function capabilityText(caps: Caps): string {
	return [
		`One action moves at most ${r0(caps.max_translation_m * 1000)} mm or ${r0((caps.max_rotation_rad * 180) / Math.PI)} deg per segment; longer actions are cut into equal segments automatically.`,
		`Speed is capped at ${r0(caps.max_speed_m_s * 1000)} mm/s (default ${r0(caps.default_speed_m_s * 1000)} mm/s).`,
		caps.supports_contact_stop
			? `A move with stop_on_contact is a push: it stops against the first thing it meets${caps.contact_stop_trusted ? "" : " (not trusted: each acceleration can trigger it)"}.`
			: "Contact stop is not supported.",
		caps.supports_home ? "home resets the scene." : "There is no home: the robot never resets the scene.",
		`The arm reaches about ${r0(caps.reach_m * 1000)} mm from its base; with the jaws tilted the tool point stays at least ${r0(caps.tilted_floor_m * 1000)} mm above the table.`,
		...caps.notes,
	].join("\n");
}

/** Where the aim marker sits in each view, as a share of the frame. */
export function legend(f: Frame): string {
	const lines = f.camera_names.flatMap((n) => {
		const c = f.cameras[n];
		if (!c?.aim_px) return [];
		return [
			`- ${n}: the magenta marker (the point under the tool) is at ${r0((c.aim_px[0] / c.width) * 100)}% from the left, ${r0((c.aim_px[1] / c.height) * 100)}% from the top`,
		];
	});
	return lines.join("\n");
}
