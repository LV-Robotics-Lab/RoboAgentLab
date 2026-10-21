/**
 * scripts/check-pi-compat.mjs against a pi that is too old and exports too little: it must name the
 * version and every missing export, and fail. (CI runs it for real against the published pi.)
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

const script = new URL("../scripts/check-pi-compat.mjs", import.meta.url).pathname;

/** A pi-coding-agent package at `version` whose only export is `parseFrontmatter`. */
function fakePi(version: string) {
	const prefix = mkdtempSync(join(tmpdir(), "fake-pi-"));
	const dir = join(prefix, "lib", "node_modules", "@earendil-works", "pi-coding-agent");
	mkdirSync(join(dir, "dist"), { recursive: true });
	writeFileSync(
		join(dir, "package.json"),
		JSON.stringify({
			name: "@earendil-works/pi-coding-agent",
			version,
			type: "module",
			exports: { ".": { types: "./dist/index.d.ts", import: "./dist/index.js" } },
		}),
	);
	writeFileSync(join(dir, "dist", "index.js"), "export const parseFrontmatter = () => ({});\n");
	writeFileSync(join(dir, "dist", "index.d.ts"), "export declare const parseFrontmatter: () => object;\n");
	return prefix;
}

function run(...args: string[]) {
	const r = spawnSync(process.execPath, [script, "--json", ...args], { encoding: "utf8" });
	return { status: r.status, report: JSON.parse(r.stdout), stderr: r.stderr };
}

test("an old pi fails the peer range; a pi without the exports fails on each missing one", () => {
	const { status, report } = run("--pi", fakePi("1.1.0"));
	assert.equal(status, 1);
	assert.equal(report.pi.version, "1.1.0");
	assert.ok(report.failures.some((f: string) => /1\.1\.0 does not satisfy peerDependencies \^1\.2\.0/.test(f)));
	const host = report.modules.find((m: any) => m.specifier === "@earendil-works/pi-coding-agent");
	assert.equal(host.status, "missing-exports");
	assert.ok(host.values.includes("parseFrontmatter"));
	assert.ok(!report.failures.some((f: string) => /no export parseFrontmatter/.test(f)));
	assert.ok(report.failures.some((f: string) => /no export parseSessionEntries \(used by /.test(f)));
	assert.ok(report.failures.some((f: string) => /no type ExtensionAPI in .*index\.d\.ts/.test(f)));
});

test("no pi at the given path is exit status 2", () => {
	const r = spawnSync(process.execPath, [script, "--pi", mkdtempSync(join(tmpdir(), "no-pi-"))], {
		encoding: "utf8",
	});
	assert.equal(r.status, 2);
	assert.match(r.stderr, /no @earendil-works\/pi-coding-agent package there/);
});
