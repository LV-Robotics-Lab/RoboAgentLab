import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { defineRobot, withServerLog } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that runs handlers in registration order and records the notifications. */
function stubPi() {
	const values: Record<string, unknown> = deployFlags({});
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const notes: string[] = [];
	const api: Record<string, unknown> = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	};
	const pi = new Proxy(api, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = { hasUI: true, ui: { notify: (m: string) => notes.push(m) }, sessionManager: { getBranch: () => [] } };
	async function emit(name: string) {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	}
	return { pi, notes, emit };
}

/**
 * An env server that comes up and answers healthz, but whose `env.reset` fails the way a LIBERO install
 * without scene assets does: the RPC error is one line, the traceback goes to the server's output (its log).
 */
const SERVER = `
import { createServer } from "node:http";
const srv = createServer((req, res) => {
	let body = "";
	req.on("data", (c) => (body += c));
	req.on("end", () => {
		const { method } = JSON.parse(body || "{}");
		if (method === "env.reset") console.error("FileNotFoundError: assets/scenes/libero_living_room_tabletop_base_style.xml");
		const reply = method === "env.reset" ? { ok: false, error: "[Errno 104] Connection reset by peer" } : { ok: true, result: {} };
		res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(reply));
		if (method === "shutdown") setTimeout(() => process.exit(0), 10);
	});
});
srv.listen(0, "127.0.0.1", () => console.log("RPC server listening on http://127.0.0.1:" + srv.address().port));
process.stdin.on("end", () => process.exit(0));
process.stdin.resume();
`;

test("a start that fails once the env server is up names the server's log", async () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-embodied-start-"));
	const script = join(dir, "server.mjs");
	writeFileSync(script, SERVER);
	const f = stubPi();
	f.pi.registerFlag("seed", { type: "string", default: "0" });
	const robot = defineRobot(f.pi, {
		name: "toy",
		task: ["seed"],
		keepImages: 1,
		start: async () => {
			const rpc = await robot.serve({
				python: process.execPath,
				args: [script],
				cwd: dir,
				env: process.env,
				log: (port) => join(dir, `pi-embodied-env-toy-${port}.log`),
			});
			await rpc.call("env.reset", {}, 10_000);
			return ["move"];
		},
		result: () => ({ success: false }),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String() }),
			result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
		},
	});
	try {
		await f.emit("session_start");
		const note = f.notes.find((n) => n.startsWith("toy unavailable")) ?? "";
		const m = /^toy unavailable: env\.reset: \[Errno 104\] Connection reset by peer \(env server log: (\S+)\)$/.exec(
			note,
		);
		assert.ok(m, note);
		assert.match(m[1], /pi-embodied-env-toy-\d+\.log$/);
		assert.ok(existsSync(m[1]));
		// The server's own error is where the message points.
		for (let i = 0; i < 50 && !readFileSync(m[1], "utf8").includes("FileNotFoundError"); i++)
			await new Promise((r) => setTimeout(r, 20));
		assert.match(readFileSync(m[1], "utf8"), /FileNotFoundError: .*libero_living_room_tabletop_base_style\.xml/);
	} finally {
		await f.emit("session_shutdown");
	}
});

test("withServerLog names the log once, and only when there is one", () => {
	assert.equal(withServerLog("env.reset: boom", "/tmp/x.log"), "env.reset: boom (env server log: /tmp/x.log)");
	assert.equal(
		withServerLog("env server exited (1); see /tmp/x.log", "/tmp/x.log"),
		"env server exited (1); see /tmp/x.log",
	);
	assert.equal(withServerLog("attach failed", undefined), "attach failed");
});
