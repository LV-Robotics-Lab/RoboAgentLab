/**
 * `--cache-breakpoints`: tell an NInfer endpoint which prompt prefixes to keep for the planner's next turn.
 *
 * NInfer serves Qwen3.8, a hybrid recurrent model: it reuses a prompt prefix only at a checkpoint it kept
 * (the previous request's endpoint, the frontier before its answer, or a shared prefix), and a Chat
 * Completions request carries no session key, so between two turns the scheduler may drop exactly the
 * checkpoint the next request needs. Its Chat Completions route takes `prompt_cache_breakpoint:
 * {"mode": "explicit"}` on a content part: an explicit shared-prefix candidate that may displace other
 * cached prefixes and holds a credit until the next exact match. With `true`, every openai-completions
 * request marks the end of its first user message (the task, which every later request of the episode
 * starts with) and the end of its last message (where the next request continues when the history only
 * grew: --image-trim-every and --replay-thinking none / frames / true, ../robot.ts). The marker does not
 * change the prompt. Another endpoint may reject the field, so it is a flag, off by default.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const FLAG = "cache-breakpoints";

const MARK = { mode: "explicit" } as const;

type Part = { type?: string; text?: string; [k: string]: unknown };
type ChatMessage = { role?: string; content?: string | Part[] | null; [k: string]: unknown };

/** `message` with a breakpoint on its last content part (a string becomes one text part), else undefined. */
function marked(message: ChatMessage): ChatMessage | undefined {
	if (message.role !== "user" && message.role !== "tool") return undefined;
	const c = message.content;
	if (typeof c === "string")
		return c ? { ...message, content: [{ type: "text", text: c, prompt_cache_breakpoint: MARK }] } : undefined;
	if (!Array.isArray(c) || c.length === 0) return undefined;
	return { ...message, content: [...c.slice(0, -1), { ...c[c.length - 1], prompt_cache_breakpoint: MARK }] };
}

/** A Chat Completions payload with the first user message and the last message marked, else undefined (left alone). */
export function markBreakpoints(payload: unknown): unknown {
	const messages = (payload as { messages?: unknown } | undefined)?.messages;
	if (!Array.isArray(messages) || messages.length === 0) return undefined;
	const out = [...(messages as ChatMessage[])];
	const first = out.findIndex((m) => m.role === "user");
	let changed = false;
	for (const i of new Set([first, out.length - 1])) {
		const m = i >= 0 ? marked(out[i]) : undefined;
		if (m) {
			out[i] = m;
			changed = true;
		}
	}
	return changed ? { ...(payload as object), messages: out } : undefined;
}

export const cacheBreakpoints = (pi: ExtensionAPI) => String(pi.getFlag(FLAG)) === "true";

/** Why the flag's value is refused (not true / false), else undefined. */
export function cacheBreakpointsError(pi: ExtensionAPI): string | undefined {
	const v = String(pi.getFlag(FLAG) ?? "false");
	return v === "true" || v === "false" ? undefined : `--${FLAG} must be true or false, got ${JSON.stringify(v)}`;
}
