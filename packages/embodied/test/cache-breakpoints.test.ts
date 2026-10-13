import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { markBreakpoints } from "../src/planner/cache-breakpoints.ts";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A toy robot on a stub pi, like context-images.test.ts; `emit` passes `ctx` (the model's api) to the handlers. */
function stubPi(values: Record<string, unknown> = {}, provider?: string) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	async function emit(name: string, event: Record<string, unknown> = {}, modelApi = "openai-completions") {
		const ctx = {
			hasUI: true,
			ui: { notify: () => {} },
			sessionManager: { getBranch: () => [] },
			model: { api: modelApi, provider },
		};
		let out: unknown;
		for (const fn of handlers.get(name) ?? []) out = (await fn({ type: name, ...event }, ctx)) ?? out;
		return out;
	}
	pi.registerFlag("seed", { type: "string", default: "0" });
	defineRobot(pi, {
		name: "toy",
		task: ["seed"],
		keepImages: 2,
		start: async () => ["look"],
		result: () => ({ success: false }),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	return { entries, emit };
}

const BP = { mode: "explicit" };
/** A Chat Completions payload as pi-ai's openai-completions provider builds it for a robot turn. */
const payload = () => ({
	model: "qwen3.8-27b",
	stream: true,
	messages: [
		{ role: "system", content: "robot" },
		{ role: "user", content: [{ type: "text", text: "Solve the task." }] },
		{
			role: "assistant",
			content: null,
			tool_calls: [{ id: "c1", type: "function", function: { name: "look", arguments: "{}" } }],
		},
		{ role: "tool", content: "step 1", tool_call_id: "c1" },
		{
			role: "user",
			content: [
				{ type: "text", text: "Attached image(s) from tool result:" },
				{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
			],
		},
	],
});

test("markBreakpoints marks the task message and the newest message, nothing else, and leaves the input alone", () => {
	const before = payload();
	const out = markBreakpoints(before) as ReturnType<typeof payload>;
	assert.deepEqual(before, payload(), "not edited in place");
	const marks = out.messages.map((m: any) =>
		Array.isArray(m.content) ? m.content.map((p: any) => p.prompt_cache_breakpoint ?? null) : null,
	);
	assert.deepEqual(marks, [null, [BP], null, null, [null, BP]]);
	assert.equal(out.stream, true);
	// A string last message (a tool result) becomes one text part with the same text.
	const tool = { ...payload(), messages: payload().messages.slice(0, 4) };
	assert.deepEqual((markBreakpoints(tool) as any).messages[3].content, [
		{ type: "text", text: "step 1", prompt_cache_breakpoint: BP },
	]);
	// Not a Chat Completions payload: left alone.
	assert.equal(markBreakpoints({ input: [] }), undefined);
	assert.equal(markBreakpoints(undefined), undefined);
});

test("the robot marks openai-completions requests only with --cache-breakpoints=true, and params records it", async () => {
	const off = stubPi();
	assert.equal(await off.emit("before_provider_request", { payload: payload() }), undefined);
	const on = stubPi({ "cache-breakpoints": "true" });
	const out = (await on.emit("before_provider_request", { payload: payload() })) as any;
	assert.deepEqual(out.messages[1].content[0].prompt_cache_breakpoint, BP);
	assert.equal(
		await on.emit("before_provider_request", { payload: payload() }, "anthropic-messages"),
		undefined,
		"another api keeps its own cache controls",
	);
	await on.emit("session_start");
	await on.emit("agent_start");
	await on.emit("session_shutdown");
	assert.equal(on.entries.find((e) => e.type === RESULT_ENTRY)?.data.params["cache-breakpoints"], "true");
	const bad = stubPi({ "cache-breakpoints": "yes" });
	await bad.emit("session_start");
	await bad.emit("session_shutdown");
	assert.match(
		bad.entries.find((e) => e.type === RESULT_ENTRY)?.data.error,
		/--cache-breakpoints must be true or false/,
	);
});

test("on the selfhost provider (NInfer) the breakpoints are the default; given, a flag keeps its value; elsewhere the default is off", async () => {
	// The provider is read at session start from the session's model (robot.ts, params.ts PROVIDER_DEFAULTS).
	const self = stubPi({}, "selfhost");
	await self.emit("session_start");
	const out = (await self.emit("before_provider_request", { payload: payload() })) as any;
	assert.deepEqual(out.messages[1].content[0].prompt_cache_breakpoint, BP);
	await self.emit("agent_start");
	await self.emit("session_shutdown");
	const r = self.entries.find((e) => e.type === RESULT_ENTRY)?.data;
	assert.deepEqual(
		[r.params["cache-breakpoints"], r.params["replay-thinking"], r.params["image-trim-every"]],
		["true", "none", "8"],
		"the cache-friendly context is the selfhost default, recorded as what ran",
	);
	assert.deepEqual(
		[r.params_default["cache-breakpoints"], r.params_default["replay-thinking"], r.params_default["image-trim-every"]],
		["false", "false", "0"],
		"params_default stays the registered default",
	);
	// A flag given at another value than its registered default is not overridden (one given at the
	// registered default is known from the command line: params.test.ts).
	const given = stubPi({ "image-trim-every": "4" }, "selfhost");
	await given.emit("session_start");
	await given.emit("agent_start");
	await given.emit("session_shutdown");
	assert.equal(given.entries.find((e) => e.type === RESULT_ENTRY)?.data.params["image-trim-every"], "4");
	// Another provider keeps the registered defaults.
	const other = stubPi({}, "relay");
	await other.emit("session_start");
	assert.equal(await other.emit("before_provider_request", { payload: payload() }), undefined);
	await other.emit("agent_start");
	await other.emit("session_shutdown");
	assert.equal(other.entries.find((e) => e.type === RESULT_ENTRY)?.data.params["cache-breakpoints"], "false");
});
