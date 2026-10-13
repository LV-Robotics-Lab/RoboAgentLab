/**
 * The executor half's prompt files (./prompts/{executor,locate,supervise}.md): read once, filled by
 * `{name}` slots. Their hashes are the configuration evidence (DESIGN §7.3 `prompts`), so the files
 * are the single source of what the model reads; nothing here edits them.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const cache = new Map<string, string>();

export type PromptName = "executor" | "locate" | "supervise";

/** The prompt file's text (system and user parts split at a line `---user---`). */
export function promptFile(name: PromptName): { system: string; text: string } {
	let raw = cache.get(name);
	if (raw === undefined) {
		raw = readFileSync(join(here, "prompts", `${name}.md`), "utf8");
		cache.set(name, raw);
	}
	const at = raw.indexOf("\n---user---\n");
	if (at < 0) return { system: "", text: raw.trim() };
	return { system: raw.slice(0, at).trim(), text: raw.slice(at + "\n---user---\n".length).trim() };
}

/** `{slot}` placeholders filled; a slot without a value is left empty. */
export function fill(template: string, slots: Record<string, string | number | undefined>): string {
	return template.replace(/\{([a-z_]+)\}/g, (m, k: string) => (k in slots ? String(slots[k] ?? "") : m));
}

/** sha256 of each executor-half prompt file (the `prompts` evidence of DESIGN §7.3). */
export function promptHashes(): Record<PromptName, string> {
	const out = {} as Record<PromptName, string>;
	for (const name of ["executor", "locate", "supervise"] as const) {
		const raw = readFileSync(join(here, "prompts", `${name}.md`), "utf8");
		out[name] = createHash("sha256").update(raw).digest("hex");
	}
	return out;
}
