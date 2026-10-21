/**
 * The dashboard's session download from pi's public session manager (src/capabilities/dashboard/session-export.ts):
 * JSONL in pi's /export shape, and an HTML transcript that escapes everything it quotes.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { CURRENT_SESSION_VERSION } from "@earendil-works/pi-coding-agent";
import { sessionHtml, sessionJsonl } from "../src/capabilities/dashboard/session-export.ts";

const branch = [
	{
		type: "message",
		id: "u1",
		parentId: "x0",
		timestamp: "2026-10-21T10:00:00.000Z",
		message: { role: "user", content: "Solve the task." },
	},
	{
		type: "message",
		id: "a1",
		parentId: "u1",
		timestamp: "2026-10-21T10:00:01.000Z",
		message: {
			role: "assistant",
			content: [
				{ type: "thinking", thinking: "look first" },
				{ type: "text", text: "Observing <camera>." },
				{ type: "toolCall", id: "c1", name: "move", arguments: { dx: 0.02 } },
			],
			stopReason: "toolUse",
		},
	},
	{
		type: "message",
		id: "r1",
		parentId: "a1",
		timestamp: "2026-10-21T10:00:02.000Z",
		message: {
			role: "toolResult",
			toolCallId: "c1",
			toolName: "move",
			content: [
				{ type: "text", text: "moved" },
				{ type: "image", data: "AAAA", mimeType: "image/png" },
			],
			isError: false,
		},
	},
	{
		type: "custom",
		id: "e1",
		parentId: "r1",
		timestamp: "2026-10-21T10:00:03.000Z",
		customType: "robot_task",
		data: { task: "pick" },
	},
	{
		type: "model_change",
		id: "m1",
		parentId: "e1",
		timestamp: "2026-10-21T10:00:04.000Z",
		provider: "faux",
		modelId: "planner",
	},
];
const session = { getBranch: () => branch as any, getSessionId: () => "sess-7", getCwd: () => "/runs/one" };

test("JSONL: pi's /export shape (header, then the branch chained by parentId)", () => {
	const lines = sessionJsonl(session)
		.split("\n")
		.filter((l) => l)
		.map((l) => JSON.parse(l));
	assert.equal(lines.length, branch.length + 1);
	assert.equal(lines[0].type, "session");
	assert.equal(lines[0].version, CURRENT_SESSION_VERSION);
	assert.equal(lines[0].id, "sess-7");
	assert.equal(lines[0].cwd, "/runs/one");
	assert.match(lines[0].timestamp, /^\d{4}-\d{2}-\d{2}T/);
	// The first entry's parentId is reset: the export is a linear session of its own.
	assert.equal(lines[1].parentId, null);
	assert.equal(lines[2].parentId, "u1");
	assert.equal(lines[5].parentId, "e1");
	assert.deepEqual(lines[1].message, branch[0].message);
	assert.ok(sessionJsonl(session).endsWith("\n"));
});

test("HTML: a self-contained transcript with every part escaped", () => {
	const html = sessionHtml(session);
	assert.match(html, /^<!doctype html>/);
	assert.match(html, /<title>pi session sess-7<\/title>/);
	assert.match(html, /Solve the task\./);
	assert.match(html, /<summary>thinking<\/summary><pre>look first<\/pre>/);
	assert.match(html, /Observing &lt;camera&gt;\./);
	assert.ok(!html.includes("<camera>"));
	assert.match(html, /<b>move<\/b><pre>\{\n\s+&quot;dx&quot;: 0\.02\n\}<\/pre>/);
	assert.match(html, /tool result · move/);
	assert.match(html, /<img src="data:image\/png;base64,AAAA"/);
	assert.match(html, /<summary>robot_task<\/summary>/);
	assert.match(html, /model faux\/planner/);
});
