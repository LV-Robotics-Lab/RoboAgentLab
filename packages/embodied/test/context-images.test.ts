import assert from "node:assert/strict";
import { test } from "node:test";
import { isDeepStrictEqual } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { defineRobot, RESULT_ENTRY } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

function stubPi(values: Record<string, unknown> = {}) {
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
	const ctx = { hasUI: true, ui: { notify: () => {} }, sessionManager: { getBranch: () => [] } };
	// Like pi's runner, each context handler sees the messages the previous one returned.
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let out: unknown;
		let current: Record<string, unknown> = { type: name, ...event };
		for (const fn of handlers.get(name) ?? []) {
			const r = (await fn(current, ctx)) as { messages?: unknown } | undefined;
			if (name === "context" && r?.messages) current = { ...current, messages: r.messages };
			out = r ?? out;
		}
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
	return { flags, entries, emit };
}

const image = (id: string) => ({ type: "image", data: id, mimeType: "image/png" });
/** A tool result with the main view then the wrist view of observation `n`. */
const look = (n: number) => ({
	role: "toolResult",
	toolName: "look",
	content: [{ type: "text", text: `step ${n}` }, image(`main${n}`), image(`wrist${n}`)],
});
const kept = (messages: any[]) =>
	messages.flatMap((m) =>
		(m.role === "toolResult" ? m.content : []).filter((p: any) => p.type === "image").map((p: any) => p.data),
	);

const messages = [{ role: "user", content: "Solve the task." }, look(0), look(1), look(2)];

/** Turn `n` of an openai-completions planner: its thinking (Qwen's reasoning_content, as pi-ai records it), then a look. */
const turn = (n: number) => ({
	role: "assistant",
	api: "openai-completions",
	provider: "selfhost",
	model: "qwen3.8-27b",
	content: [
		{ type: "thinking", thinking: `reasoning ${n}`, thinkingSignature: "reasoning_content" },
		{ type: "toolCall", id: `c${n}`, name: "look", arguments: {} },
	],
	stopReason: "toolUse",
});

/** The robot's requests over an episode of `turns` looks: what the context handlers make of each growing history. */
async function requests(f: ReturnType<typeof stubPi>, turns: number) {
	const history: any[] = [{ role: "user", content: "Solve the task." }];
	const out: any[][] = [];
	for (let n = 0; n < turns; n++) {
		history.push(turn(n), look(n));
		const messages = structuredClone(history);
		out.push(((await f.emit("context", { messages })) as { messages: any[] } | undefined)?.messages ?? messages);
	}
	return out;
}

test("--keep-images keeps the latest frames and stubs the rest", async () => {
	const f = stubPi();
	const out = (await f.emit("context", { messages })) as { messages: any[] };
	assert.deepEqual(kept(out.messages), ["main2", "wrist2"]);
	assert.equal(out.messages[1].content[1].text, "[older camera frame omitted]");
});

test("--anchor-image also keeps the episode's first main frame, and the result records it", async () => {
	const f = stubPi({ "anchor-image": true });
	const out = (await f.emit("context", { messages })) as { messages: any[] };
	assert.deepEqual(kept(out.messages), ["main0", "main2", "wrist2"], "the anchor is extra to --keep-images");
	assert.equal(out.messages[1].content[2].text, "[older camera frame omitted]", "only the first frame of it");
	// Nothing to prune yet: the context is left alone.
	assert.equal(await f.emit("context", { messages: messages.slice(0, 2) }), undefined);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	assert.equal(f.entries.find((e) => e.type === RESULT_ENTRY)?.data.anchor_image, true);
});

test("without --anchor-image the result does not mention it", async () => {
	const f = stubPi();
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	const result = f.entries.find((e) => e.type === RESULT_ENTRY);
	assert.ok(result);
	assert.equal("anchor_image" in result.data, false);
});

test("--image-trim-every N moves the cut every N frames: between cuts the earlier context is the next one's prefix", async () => {
	const f = stubPi({ "image-trim-every": "4" });
	const contexts = [1, 2, 3, 4, 5, 6].map((n) => [
		{ role: "user", content: "Solve the task." },
		...[...Array(n).keys()].map(look),
	]);
	const out: any[][] = [];
	for (const messages of contexts)
		out.push(((await f.emit("context", { messages })) as { messages: any[] } | undefined)?.messages ?? messages);
	// keep-images 2 (the toy robot's), cut every 4: 2, 4, 6, 8, 10, 12 frames keep 2, 4, 2, 4, 2, 4.
	assert.deepEqual(
		out.map((m) => kept(m).length),
		[2, 4, 2, 4, 2, 4],
	);
	assert.deepEqual(kept(out[3]), ["main2", "wrist2", "main3", "wrist3"]);
	// No cut between 3 and 4 observations (6 -> 8 frames): the request only grows.
	assert.deepEqual(out[3].slice(0, out[2].length), out[2]);
	// The default (0) cuts every turn: the same step rewrites the oldest kept observation.
	const every = stubPi();
	const a = (await every.emit("context", { messages: contexts[2] })) as { messages: any[] };
	const b = (await every.emit("context", { messages: contexts[3] })) as { messages: any[] };
	assert.notDeepEqual(b.messages.slice(0, a.messages.length), a.messages);
	await f.emit("session_start");
	await f.emit("agent_start");
	await f.emit("session_shutdown");
	assert.equal(f.entries.find((e) => e.type === RESULT_ENTRY)?.data.params["image-trim-every"], "4");
});

test("--replay-thinking=frames drops a turn's thinking with the frames before it: with --image-trim-every both cut together", async () => {
	const out = await requests(stubPi({ "image-trim-every": "4", "replay-thinking": "frames" }), 5);
	const thinking = (messages: any[]) =>
		messages.filter((m) => m.role === "assistant").map((m) => m.content.some((c: any) => c.type === "thinking"));
	// 2, 4 frames: nothing stubbed, every turn's thinking goes out (the answer comes back as generated).
	assert.deepEqual(thinking(out[1]), [true, true]);
	assert.deepEqual(out[1].slice(0, out[0].length), out[0]);
	// 6 frames: the cut stubs observations 0 and 1, and the thinking of the turns before them goes too.
	assert.deepEqual(thinking(out[2]), [false, false, true]);
	// 8 frames: no new cut, the request only grows.
	assert.deepEqual(thinking(out[3]), [false, false, true, true]);
	assert.deepEqual(out[3].slice(0, out[2].length), out[2]);
	// 10 frames: the next cut.
	assert.deepEqual(thinking(out[4]), [false, false, false, false, true]);
});

test("append-only: each request starts with the previous one's messages, except where the cut moves, once every N frames", async () => {
	const stubs = (messages: any[]) =>
		messages
			.flatMap((m) => (m.role === "toolResult" ? m.content : []))
			.filter((p: any) => p.type === "text" && p.text === "[older camera frame omitted]").length;
	const grows = (before: any[], after: any[]) => isDeepStrictEqual(after.slice(0, before.length), before);
	const configs: [Record<string, unknown>, number][] = [
		[{ "replay-thinking": "none", "image-trim-every": "4" }, 4],
		[{ "replay-thinking": "frames", "image-trim-every": "4" }, 4],
		[{ "replay-thinking": "none", "image-trim-every": "3", "anchor-image": true }, 3],
		[{ "replay-thinking": "true", "image-trim-every": "8" }, 8],
	];
	for (const [flags, every] of configs) {
		const label = JSON.stringify(flags);
		const anchor = flags["anchor-image"] ? 1 : 0;
		const out = await requests(stubPi(flags), 24);
		let cuts = 0;
		for (let n = 1; n < out.length; n++) {
			if (stubs(out[n]) === stubs(out[n - 1])) assert.ok(grows(out[n - 1], out[n]), `${label}: request ${n}`);
			else {
				cuts++;
				assert.equal(stubs(out[n]) - stubs(out[n - 1]), every, `${label}: request ${n} cuts N frames`);
			}
		}
		for (const [n, request] of out.entries()) {
			// keep-images 2 (the toy robot's): 2 to 2+N-1 frames stay once there are 2, the anchor on top.
			const frames = 2 * (n + 1) - anchor;
			const k = kept(request).length - anchor;
			assert.ok(k >= Math.min(frames, 2) && k <= 2 + every - 1, `${label}: request ${n} keeps ${k} frames`);
		}
		assert.equal(cuts, Math.floor((48 - anchor - 2) / every), `${label}: one cut per N frames`);
	}
	// The defaults rewrite an earlier turn on every request: no request starts with the previous one.
	const out = await requests(stubPi(), 24);
	for (let n = 1; n < out.length; n++) assert.equal(grows(out[n - 1], out[n]), false, `default: request ${n}`);
});
