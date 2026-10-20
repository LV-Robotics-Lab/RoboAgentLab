/**
 * `--replay-thinking`: whether the planner's earlier turns' thinking goes back out in every request.
 *
 * pi replays an assistant message as recorded, thinking included: pi-ai's openai-completions
 * provider sends each earlier turn's thinking as `reasoning_content` (packages/ai/src/api/
 * openai-completions.ts, convertMessages), and an endpoint whose chat template keeps it (NInfer /
 * vLLM with Qwen's) counts it as prompt tokens. On the fixed Qwen regression that was ~36% of the
 * planner's input (docs/reports, SoL-Pi A/B). The model does not read its own earlier reasoning as
 * instructions, so with `false` (the default) a request keeps only the latest assistant message's
 * thinking; `true` is pi's own behaviour; `none` leaves out the latest assistant message's thinking too;
 * `frames` keeps it until the camera frames before it are stubbed. `false` rewrites the history every
 * turn (the previous answer loses its thinking in the next request), so an endpoint's prefix cache never
 * reuses past it: NInfer's Qwen3.8 is a hybrid recurrent model that reuses only exact checkpoints at
 * earlier prompt frontiers. `true` and `none` are append-only, and so is `frames` between the image cuts
 * of `--image-trim-every` (../robot.ts); `true` and `frames` also replay the answer exactly as generated,
 * which NInfer's most dependable checkpoint (the previous request's endpoint) needs.
 * pi-ai's compat has no such switch (`requiresThinkingAsText`
 * rewrites thinking, nothing omits it), and a models.json field would not be recorded in `params`, so
 * it is a flag and a `context` handler (../robot.ts): only the outgoing request changes, the session
 * keeps every block, and `params` records the value for params-match.mjs.
 *
 * Fail safe: only unsigned thinking of an openai-completions assistant message is dropped. A message
 * of any other api (Anthropic's signed blocks, OpenAI Responses' encrypted reasoning items, Bedrock,
 * Google) stays whole, as does a block that is redacted or carries a signature other than the plain
 * reasoning field name pi-ai uses for a text-only replay ("reasoning", "reasoning_content",
 * "reasoning_text"), such as OpenRouter's `reasoning_details`.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ThinkingContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const FLAG = "replay-thinking";

/** Signatures pi-ai's openai-completions provider gives a thinking block that is only text: the reasoning field it came in. */
const TEXT_ONLY = new Set(["reasoning", "reasoning_content", "reasoning_text"]);

/** Whether `block` is thinking an openai-completions request could replay only as text: safe to leave out. */
export function unsigned(block: { type: string } & Partial<Omit<ThinkingContent, "type">>): block is ThinkingContent {
	if (block.type !== "thinking") return false;
	const t = block as ThinkingContent;
	return !t.redacted && (!t.thinkingSignature || TEXT_ONLY.has(t.thinkingSignature));
}

/**
 * `messages` with the unsigned thinking of every openai-completions assistant message before index
 * `keepFrom` left out (default: the last assistant message, which keeps its thinking), or undefined when
 * nothing would change (the context is left alone). New arrays, never edited in place: the session's
 * messages are untouched.
 */
export function dropReplayedThinking(
	messages: readonly AgentMessage[],
	keepFrom?: number,
): { messages: AgentMessage[] } | undefined {
	let last = keepFrom ?? -1;
	for (let i = messages.length - 1; i >= 0 && keepFrom === undefined; i--)
		if (messages[i].role === "assistant") {
			last = i;
			break;
		}
	let changed = false;
	const out = messages.map((m, i) => {
		if (i >= last || m.role !== "assistant") return m;
		const a = m as AssistantMessage;
		if (a.api !== "openai-completions" || !Array.isArray(a.content) || !a.content.some(unsigned)) return m;
		changed = true;
		return { ...a, content: a.content.filter((block) => !unsigned(block)) };
	});
	return changed ? { messages: out } : undefined;
}

/** The flag's value, true or false; pi stores a flag's command-line value verbatim, so a bare `--replay-thinking` reads true. */
export const replayThinking = (pi: ExtensionAPI) => String(pi.getFlag(FLAG)) === "true";

/**
 * Where the context handler starts keeping thinking, by `--replay-thinking`: undefined (false: the last
 * assistant message), the end (none), or right after the newest tool result whose camera frame became
 * `stub` (frames: thinking leaves together with the frames it saw, so with `--image-trim-every` both
 * change on the same request and the history only grows in between).
 */
export function keepThinkingFrom(pi: ExtensionAPI, messages: readonly AgentMessage[], stub: string) {
	const v = String(pi.getFlag(FLAG));
	if (v === "none") return messages.length;
	if (v !== "frames") return undefined;
	let from = 0;
	messages.forEach((m, i) => {
		if (m.role === "toolResult" && m.content.some((p) => p.type === "text" && p.text === stub)) from = i + 1;
	});
	return from;
}

/** Why the flag's value is refused (not true / false / none / frames), else undefined. */
export function replayThinkingError(pi: ExtensionAPI): string | undefined {
	const v = String(pi.getFlag(FLAG) ?? "false");
	return ["true", "false", "none", "frames"].includes(v)
		? undefined
		: `--${FLAG} must be true, false, none or frames, got ${JSON.stringify(v)}`;
}
