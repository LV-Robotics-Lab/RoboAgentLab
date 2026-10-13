/**
 * The MotorMind roles' model calls (DESIGN §1.2–1.3): every role runs on `<base>` (the model after
 * `motormind/` in --model) unless `--mm-role-models role=provider/id,...` names another; a request is
 * sent through pi's model registry with the sampling the original's local path used: temperature 0,
 * the role's output cap (planner 3000 tokens, the others 1500), thinking off, a JSON object asked
 * for, 3 retries and a 60 s timeout (pi-ai's `maxRetries`, `timeoutMs`, `onPayload`). The reply's
 * JSON is parsed whole, then from a fenced block, then as the first brace-balanced object, then
 * from the thinking text when the body is empty.
 *
 * Prompts live in ./prompts/<name>.md (rewritten from the specs' intent, never the original's text);
 * `promptFile` reads one and records its sha256 for the result's `prompts` (DESIGN §7.3).
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type {
	Api,
	ImageContent,
	Model,
	SimpleStreamOptions,
	TextContent,
	ThinkingContent,
} from "@earendil-works/pi-ai";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import { type Ask, ROLES, type Role, type RoleErrorKind, type RoleReply, type RoleUsage } from "./contract.ts";

/** Output caps per role (spec-planner §1.1: plan/replan 3000, every other role the client's 1500). */
export const MAX_TOKENS: Record<Role, number> = {
	planner: 3000,
	verify: 1500,
	executor: 1500,
	locate: 1500,
	supervise: 1500,
	monitor: 1500,
	memory: 1500,
};
/** The original client's local path: 3 retries, 60 s per request. */
export const MAX_RETRIES = 3;
export const TIMEOUT_MS = 60_000;

/** `--mm-role-models role=provider/id,...` parsed; a role not named uses the base. Throws on a bad spec. */
export function roleModels(base: string, spec: string | undefined): Record<Role, string> {
	const out = Object.fromEntries(ROLES.map((r) => [r, base])) as Record<Role, string>;
	for (const part of (spec ?? "")
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean)) {
		const eq = part.indexOf("=");
		const role = part.slice(0, eq).trim();
		const ref = part.slice(eq + 1).trim();
		if (eq <= 0 || !ref.includes("/") || !(ROLES as readonly string[]).includes(role))
			throw new Error(`--mm-role-models: expected role=provider/id with role in ${ROLES.join(", ")}, got ${part}`);
		out[role as Role] = ref;
	}
	return out;
}

/** "provider/id" → the registry's model, or undefined. */
export function findModel(registry: Pick<ModelRegistry, "find">, ref: string): Model<Api> | undefined {
	const i = ref.indexOf("/");
	return i > 0 ? (registry.find(ref.slice(0, i), ref.slice(i + 1)) ?? undefined) : undefined;
}

/**
 * The request body as the original's local path sends it (spec-planner §1.1): temperature 0,
 * `max_tokens`, thinking off through the chat template, a JSON object. Fields another API does not
 * know are only set on OpenAI-style completions; the sampling fields are set wherever the payload is
 * an object. Any reasoning effort pi added is removed (thinking is off).
 */
export function faithfulPayload(payload: unknown, model: Pick<Model<Api>, "api">, maxTokens: number): unknown {
	if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return undefined;
	const p = payload as Record<string, unknown>;
	p.temperature = 0;
	delete p.reasoning_effort;
	delete p.reasoning;
	if (model.api === "openai-completions") {
		p.max_tokens = maxTokens;
		delete p.max_completion_tokens;
		p.chat_template_kwargs = {
			...(p.chat_template_kwargs as Record<string, unknown> | undefined),
			enable_thinking: false,
		};
		p.response_format = { type: "json_object" };
	} else if ("max_tokens" in p || "max_completion_tokens" in p || "max_output_tokens" in p) {
		for (const k of ["max_tokens", "max_completion_tokens", "max_output_tokens"]) if (k in p) p[k] = maxTokens;
	}
	return p;
}

/** The first brace-balanced JSON object in `s` (quotes and escapes respected), parsed, or undefined. */
function firstObject(s: string): unknown {
	const start = s.indexOf("{");
	if (start < 0) return undefined;
	let depth = 0;
	let inString = false;
	for (let i = start; i < s.length; i++) {
		const c = s[i];
		if (inString) {
			if (c === "\\") i++;
			else if (c === '"') inString = false;
			continue;
		}
		if (c === '"') inString = true;
		else if (c === "{") depth++;
		else if (c === "}") {
			depth--;
			if (depth === 0) {
				try {
					return JSON.parse(s.slice(start, i + 1));
				} catch {
					return undefined;
				}
			}
		}
	}
	return undefined;
}

/** The reply's JSON (DESIGN §1.3 order): whole → fenced block → first balanced object → the thinking text when the body is empty. */
export function parseRoleJson(text: string, thinking?: string): unknown {
	const body = text.trim();
	if (!body) return thinking ? parseRoleJson(thinking) : undefined;
	try {
		return JSON.parse(body);
	} catch {}
	const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(body);
	if (fence) {
		try {
			return JSON.parse(fence[1].trim());
		} catch {}
	}
	return firstObject(body);
}

const ZERO: RoleUsage = { cost_usd: 0, tokens_in: 0, tokens_out: 0, latency_s: 0 };
const kindOf = (error: string, aborted: boolean): RoleErrorKind =>
	aborted ? "aborted" : /timed? ?out|timeout|ETIMEDOUT|idle/i.test(error) ? "timeout" : "transport";

/**
 * An `Ask` over the registry: one user message (the prompt, then the images) with the role's
 * system prompt, the faithful sampling, the retries and timeout; `ok` only with parsable JSON.
 */
export function makeAsk(
	registry: Pick<ModelRegistry, "find" | "streamSimple">,
	models: Record<Role, string>,
	onCall?: (role: Role, reply: RoleReply) => void,
): Ask {
	return async (role, prompt, images, o) => {
		const ref = models[role];
		const model = findModel(registry, ref);
		const started = Date.now();
		const done = (r: RoleReply) => {
			onCall?.(role, r);
			return r;
		};
		if (!model)
			return done({
				ok: false,
				error: `unknown model ${ref}`,
				error_kind: "unknown_model",
				model: ref,
				usage: ZERO,
				attempts: 0,
			});
		const maxTokens = o?.maxTokens ?? MAX_TOKENS[role];
		const content: (TextContent | ImageContent)[] = [{ type: "text", text: prompt.text }, ...images];
		const options: SimpleStreamOptions = {
			signal: o?.signal,
			maxRetries: MAX_RETRIES,
			timeoutMs: TIMEOUT_MS,
			maxTokens,
			onPayload: (payload, m) => faithfulPayload(payload, m, maxTokens),
		};
		let reply: Awaited<ReturnType<ReturnType<typeof registry.streamSimple>["result"]>>;
		try {
			reply = await registry
				.streamSimple(
					model,
					{ systemPrompt: prompt.system, messages: [{ role: "user", content, timestamp: Date.now() }] },
					options,
				)
				.result();
		} catch (err) {
			const error = err instanceof Error ? err.message : String(err);
			return done({
				ok: false,
				error,
				error_kind: kindOf(error, o?.signal?.aborted === true),
				model: ref,
				usage: { ...ZERO, latency_s: (Date.now() - started) / 1000 },
				attempts: 1,
			});
		}
		const usage: RoleUsage = {
			cost_usd: reply.usage?.cost?.total ?? 0,
			tokens_in: (reply.usage?.input ?? 0) + (reply.usage?.cacheRead ?? 0) + (reply.usage?.cacheWrite ?? 0),
			tokens_out: reply.usage?.output ?? 0,
			latency_s: (Date.now() - started) / 1000,
		};
		const answered = `${reply.provider}/${reply.responseModel ?? reply.model}`;
		if (reply.stopReason === "error" || reply.stopReason === "aborted") {
			const error = reply.errorMessage ?? reply.stopReason;
			return done({
				ok: false,
				error,
				error_kind: kindOf(error, reply.stopReason === "aborted" || o?.signal?.aborted === true),
				model: answered,
				usage,
				attempts: 1,
			});
		}
		const text = reply.content
			.filter((c): c is TextContent => c.type === "text")
			.map((c) => c.text)
			.join("\n");
		const thinking = reply.content
			.filter((c): c is ThinkingContent => c.type === "thinking")
			.map((c) => c.thinking)
			.join("\n");
		const json = parseRoleJson(text, thinking || undefined);
		if (json === undefined)
			return done({
				ok: false,
				error: text.trim() ? "the reply holds no JSON object" : "the reply is empty",
				error_kind: text.trim() ? "no_json" : "empty",
				text,
				model: answered,
				usage,
				attempts: 1,
			});
		return done({ ok: true, text, json, ...(thinking ? { thinking } : {}), model: answered, usage, attempts: 1 });
	};
}

// ---------------------------------------------------------------------------
// prompt files

export const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const PROMPTS = new Map<string, { text: string; sha: string }>();

/** ./prompts/<name>.md: its text and sha256 (read once). */
export function promptFile(name: string): { text: string; sha: string } {
	const had = PROMPTS.get(name);
	if (had) return had;
	const text = readFileSync(fileURLToPath(new URL(`./prompts/${name}.md`, import.meta.url)), "utf8");
	const entry = { text, sha: sha256(text) };
	PROMPTS.set(name, entry);
	return entry;
}

/** The sha256 of every prompt file read so far, by name (the result's `prompts`). */
export const promptHashes = (): Record<string, string> =>
	Object.fromEntries([...PROMPTS].sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, v.sha]));

/** Fill `{name}` slots of a prompt template; a slot with no value becomes "". */
export function fill(template: string, slots: Record<string, string | number | undefined>): string {
	return template.replace(/\{([a-z_]+)\}/g, (m, k: string) => (k in slots ? String(slots[k] ?? "") : m));
}
