/**
 * The MotorMind prompt files (./prompts/<name>.md, one per role): read once, split into the system
 * and user parts at a `===== user =====` line, filled by `{name}` slots. Their sha256 hashes are the
 * configuration evidence (DESIGN §7.3 `prompts`, keyed by eval.sh against mixing), so the hashes
 * cover every file regardless of which roles ran, and the files are the single source of what the
 * model reads; nothing here edits them.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export const PROMPT_NAMES = ["plan", "replan", "verify", "executor", "locate", "supervise"] as const;
export type PromptName = (typeof PROMPT_NAMES)[number];

/** The line between the system and the user part of a prompt file. */
export const PROMPT_SEPARATOR = /\n=+\s*user\s*=+\n/i;

export const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

export type PromptFile = { system: string; text: string; sha: string };
const cache = new Map<string, PromptFile>();

/** ./prompts/<name>.md: its system part, its user part and the sha256 of the whole file (read once). */
export function promptFile(name: PromptName): PromptFile {
	const had = cache.get(name);
	if (had) return had;
	const raw = readFileSync(join(here, "prompts", `${name}.md`), "utf8");
	const [system, user] = raw.split(PROMPT_SEPARATOR);
	if (user === undefined) throw new Error(`prompts/${name}.md lacks the "===== user =====" separator`);
	const entry = { system: system.trim(), text: user.trim(), sha: sha256(raw) };
	cache.set(name, entry);
	return entry;
}

/** `{slot}` placeholders filled; a slot without a value becomes "", an unknown slot is left as is. */
export function fill(template: string, slots: Record<string, string | number | undefined>): string {
	return template.replace(/\{([a-z_]+)\}/g, (m, k: string) => (k in slots ? String(slots[k] ?? "") : m));
}

/** sha256 of every prompt file, by name (the `prompts` evidence of DESIGN §7.3). */
export function promptHashes(): Record<PromptName, string> {
	const out = {} as Record<PromptName, string>;
	for (const name of PROMPT_NAMES) out[name] = promptFile(name).sha;
	return out;
}
