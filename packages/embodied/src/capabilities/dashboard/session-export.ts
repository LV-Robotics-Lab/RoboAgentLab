/**
 * The current session branch as a download, from pi's public extension API alone
 * (`ExtensionContext.sessionManager`: getBranch, getSessionId, getCwd), so the dashboard's
 * `GET /download/session` works on an unmodified pi.
 *
 * - JSONL: the same lines pi's own `/export` writes (coding-agent src/core/session-export.ts
 *   serializeSessionBranch): a fresh session header, then every entry of the current branch with its
 *   parentId rewritten to the previous line, so pi loads the file back as a linear session.
 * - HTML: a self-contained transcript (user, assistant thinking and text, tool calls and results,
 *   the camera frames the robot returned, the extension's own session entries). It is a plain
 *   rendering, not pi's themed export: no markdown, no syntax highlighting.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { CURRENT_SESSION_VERSION, type ExtensionContext, type SessionHeader } from "@earendil-works/pi-coding-agent";

/** What the export reads: the read-only session manager pi hands every extension context. */
export type SessionSource = Pick<ExtensionContext["sessionManager"], "getBranch" | "getSessionId" | "getCwd">;

type Part = { type: string; text?: string; thinking?: string; data?: string; mimeType?: string; [k: string]: unknown };

/** The current branch as pi's /export JSONL (a header line, then the entries chained by parentId). */
export function sessionJsonl(session: SessionSource): string {
	const header: SessionHeader = {
		type: "session",
		version: CURRENT_SESSION_VERSION,
		id: session.getSessionId(),
		timestamp: new Date().toISOString(),
		cwd: session.getCwd(),
	};
	const lines: object[] = [header];
	let parentId: string | null = null;
	for (const entry of session.getBranch()) {
		lines.push({ ...entry, parentId });
		parentId = entry.id;
	}
	return `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`;
}

const esc = (s: unknown) =>
	String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string);

const pre = (s: unknown) => `<pre>${esc(s)}</pre>`;
const json = (v: unknown) => pre(JSON.stringify(v, null, 2) ?? "");

/** Text, image and thinking parts of a message, in order, as HTML. */
function parts(content: string | Part[] | undefined): string {
	if (content === undefined) return "";
	if (typeof content === "string") return pre(content);
	return content
		.map((p) => {
			if (p.type === "text") return pre(p.text ?? "");
			if (p.type === "thinking") return `<details><summary>thinking</summary>${pre(p.thinking ?? "")}</details>`;
			if (p.type === "image" && p.data)
				return `<img src="data:${esc(p.mimeType ?? "image/png")};base64,${esc(p.data)}" alt="image">`;
			if (p.type === "toolCall") return `<div class="call"><b>${esc(p.name)}</b>${json(p.arguments)}</div>`;
			return json(p);
		})
		.join("");
}

function message(m: AgentMessage): string {
	const x = m as unknown as Part;
	if (m.role === "toolResult") {
		const r = m as unknown as { toolName: string; isError?: boolean; content: Part[] };
		return `<section class="result${r.isError ? " error" : ""}"><h2>tool result · ${esc(r.toolName)}${r.isError ? " (error)" : ""}</h2>${parts(r.content)}</section>`;
	}
	const text = parts(x.content as string | Part[] | undefined);
	return `<section class="${esc(m.role)}"><h2>${esc(m.role)}</h2>${text}</section>`;
}

/** The current branch as a self-contained HTML transcript. */
export function sessionHtml(session: SessionSource): string {
	const id = session.getSessionId();
	const body = session
		.getBranch()
		.map((e) => {
			const when = `<time>${esc(e.timestamp)}</time>`;
			switch (e.type) {
				case "message":
					return `${message(e.message)}${when}`;
				case "custom_message":
					return `<section class="custom"><h2>${esc(e.customType)}</h2>${parts(e.content as string | Part[])}</section>${when}`;
				case "custom":
					return `<section class="meta"><details><summary>${esc(e.customType)}</summary>${json(e.data)}</details></section>`;
				case "compaction":
					return `<section class="meta"><h2>compaction</h2>${pre(e.summary)}</section>${when}`;
				case "branch_summary":
					return `<section class="meta"><h2>branch summary</h2>${pre(e.summary)}</section>${when}`;
				case "model_change":
					return `<section class="meta">model ${esc(e.provider)}/${esc(e.modelId)}</section>`;
				case "thinking_level_change":
					return `<section class="meta">thinking ${esc(e.thinkingLevel)}</section>`;
				default:
					return `<section class="meta">${esc(e.type)}</section>`;
			}
		})
		.join("\n");
	return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>pi session ${esc(id)}</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;max-width:60em;margin:1em auto;padding:0 1em;color:#222;background:#fff}
section{border:1px solid #ddd;border-radius:6px;padding:.5em .8em;margin:.6em 0}
section.user{background:#eef4ff}section.assistant{background:#fafafa}section.result{background:#f3fbf3}section.result.error{background:#fff0f0}
section.meta{color:#666;font-size:12px;background:#f7f7f7}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#666;margin:0 0 .3em}
pre{white-space:pre-wrap;word-break:break-word;margin:.2em 0;font-size:13px}
img{max-width:100%;display:block;margin:.3em 0}
.call{border-left:3px solid #c9a;padding-left:.6em;margin:.3em 0}
time{display:block;font-size:11px;color:#999;margin:-.4em 0 .6em .2em}
</style></head>
<body><h1>pi session ${esc(id)}</h1><p class="meta">${esc(session.getCwd())}</p>
${body}
</body></html>
`;
}
