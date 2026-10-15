/**
 * A perception server stand-in: answers every RPC, healthz included, with `ok`. A robot probes its
 * perception servers at start and drops the tools of one that does not answer (src/capabilities/
 * skills.ts `probePerception`); tests point `sam3` here so they do not depend on a SAM3 server
 * happening to listen on the default port of the machine they run on.
 */

import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

export async function fakePerception(): Promise<string> {
	const server = createServer((req, res) => {
		req.resume();
		req.on("end", () => {
			const body = JSON.stringify({ ok: true, result: { ok: true } });
			res.writeHead(200, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) });
			res.end(body);
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	server.unref();
	return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}
