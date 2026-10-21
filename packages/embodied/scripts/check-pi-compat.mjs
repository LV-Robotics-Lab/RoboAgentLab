#!/usr/bin/env node
/**
 * Checks an installed pi against what this package imports from it, before `pi install`.
 *
 *   node scripts/check-pi-compat.mjs [--pi <dir>] [--strict] [--json]
 *
 * The pi to check is, in order: --pi (the @earendil-works/pi-coding-agent package directory, or
 * an npm prefix such as /usr/local or ~/.npm-global), the `pi` on PATH, `npm root -g`, or the
 * pi-coding-agent this package resolves from its own node_modules (the monorepo workspace).
 *
 * What it checks:
 * - pi-coding-agent's version satisfies this package's `peerDependencies` range.
 * - Every `import ... from "@earendil-works/*"` and `"typebox"` under src/: the module resolves from
 *   that pi (host-provided packages from pi's own dependencies, the rest from next to this package),
 *   every imported value is an export of the resolved module, and every imported type is declared
 *   in its `.d.ts`. A missing export fails the check.
 * - Optional peers (`peerDependenciesMeta.optional`) that are not installed anywhere are reported and
 *   skipped, with the files that need them; --strict fails on them too.
 * - Modules pi does not map to its own copy (coding-agent src/core/extensions/loader.ts getAliases)
 *   are reported as "resolved next to the package": a plain `pi install` of a git clone leaves them
 *   to that clone's node_modules.
 *
 * Exit status: 0 when everything resolves, 1 on a failure, 2 when no pi is found.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const PKG = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const HOST = "@earendil-works/pi-coding-agent";
/** Specifiers pi 1.3 resolves to its own modules for every extension (loader.ts getAliases). */
const HOST_PROVIDED = new Set([
	"@earendil-works/pi-coding-agent",
	"@earendil-works/pi-agent-core",
	"@earendil-works/pi-tui",
	"@earendil-works/pi-ai",
	"@earendil-works/pi-ai/compat",
	"@earendil-works/pi-ai/oauth",
	"@earendil-works/pi-ai/providers/all",
	"typebox",
	"typebox/compile",
	"typebox/value",
]);
/** pi maps the pi-ai root to its compat entry (a superset of the root); check what extensions get. */
const RUNTIME_ENTRY = { "@earendil-works/pi-ai": "@earendil-works/pi-ai/compat" };

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name) => {
	const i = args.indexOf(name);
	return i >= 0 ? args[i + 1] : undefined;
};
const strict = flag("--strict");
const asJson = flag("--json");

const manifest = JSON.parse(readFileSync(join(PKG, "package.json"), "utf8"));
const optionalPeers = new Set(
	Object.entries(manifest.peerDependenciesMeta ?? {})
		.filter(([, meta]) => meta?.optional)
		.map(([name]) => name),
);

// ---- which pi ---------------------------------------------------------------------------------------

/** The pi-coding-agent package directory at or above `start`, or under it as an npm prefix. */
function hostPackageDir(start) {
	let real = start;
	try {
		real = realpathSync(start);
	} catch {
		return undefined;
	}
	for (const base of [real, join(real, "lib", "node_modules", HOST), join(real, "node_modules", HOST)]) {
		let dir = base;
		for (let i = 0; i < 8; i++) {
			const file = join(dir, "package.json");
			if (existsSync(file)) {
				try {
					if (JSON.parse(readFileSync(file, "utf8")).name === HOST) return dir;
				} catch {}
			}
			const parent = dirname(dir);
			if (parent === dir) break;
			dir = parent;
		}
	}
	return undefined;
}

function run(cmd, cmdArgs) {
	try {
		return execFileSync(cmd, cmdArgs, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
	} catch {
		return "";
	}
}

function findHost() {
	const candidates = [];
	const given = option("--pi");
	if (given) {
		const dir = hostPackageDir(resolve(given));
		if (!dir) fail(2, `--pi ${given}: no ${HOST} package there`);
		return { dir, how: "--pi" };
	}
	const bin = run(process.platform === "win32" ? "where" : "which", ["pi"]).split(/\r?\n/)[0];
	if (bin) candidates.push([bin, `pi on PATH (${bin})`]);
	const globalRoot = run("npm", ["root", "-g"]);
	if (globalRoot) candidates.push([join(globalRoot, HOST), "npm root -g"]);
	for (const nm of nodeModulesUp(PKG)) candidates.push([join(nm, HOST), `${nm} (next to this package)`]);
	for (const [start, how] of candidates) {
		const dir = hostPackageDir(start);
		if (dir) return { dir, how };
	}
	return fail(2, "no pi found: pass --pi <dir>, put pi on PATH, or install @earendil-works/pi-coding-agent");
}

function fail(code, message) {
	console.error(message);
	process.exit(code);
}

// ---- version range ----------------------------------------------------------------------------------

const parseVersion = (v) => String(v).split("-")[0].split(".").map(Number);
const compare = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

/** Whether `version` satisfies a single `^x.y.z`, `~x.y.z`, `>=x.y.z`, `x.y.z` or `*` range. */
function satisfies(version, range) {
	const r = String(range).trim();
	if (r === "" || r === "*") return true;
	const m = r.match(/^(\^|~|>=|=)?\s*(\d+)\.(\d+)\.(\d+)/);
	if (!m) return undefined;
	const want = [Number(m[2]), Number(m[3]), Number(m[4])];
	const have = parseVersion(version);
	const c = compare(have, want);
	switch (m[1]) {
		case ">=":
			return c >= 0;
		case "~":
			return c >= 0 && have[0] === want[0] && have[1] === want[1];
		case "^":
			if (c < 0) return false;
			if (want[0] > 0) return have[0] === want[0];
			if (want[1] > 0) return have[0] === 0 && have[1] === want[1];
			return c === 0;
		default:
			return c === 0;
	}
}

// ---- imports under src/ -----------------------------------------------------------------------------

function* sourceFiles(dir) {
	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		const path = join(dir, entry.name);
		if (entry.isDirectory()) {
			if (entry.name !== "node_modules") yield* sourceFiles(path);
		} else if (/\.(m?ts|m?js)$/.test(entry.name) && !entry.name.endsWith(".d.ts")) yield path;
	}
}

const STATIC_IMPORT = /^\s*import\s+(type\s+)?([^;'"]*?)\s*from\s*["']([^"']+)["']/gm;
const DYNAMIC_IMPORT = /\bimport\(\s*["']([^"']+)["']\s*\)/g;
const isChecked = (spec) => spec.startsWith("@earendil-works/") || spec === "typebox" || spec.startsWith("typebox/");

/** specifier -> { values: Map<name, Set<file>>, types: Map<name, Set<file>>, files: Set<file> } */
function collectImports() {
	const uses = new Map();
	const use = (spec, file) => {
		let u = uses.get(spec);
		if (!u) {
			u = { values: new Map(), types: new Map(), files: new Set() };
			uses.set(spec, u);
		}
		u.files.add(file);
		return u;
	};
	const add = (map, name, file) => {
		if (!map.has(name)) map.set(name, new Set());
		map.get(name).add(file);
	};
	for (const path of sourceFiles(join(PKG, "src"))) {
		const file = relative(PKG, path);
		const text = readFileSync(path, "utf8");
		for (const m of text.matchAll(STATIC_IMPORT)) {
			const [, typeOnly, clause, spec] = m;
			if (!isChecked(spec)) continue;
			const u = use(spec, file);
			const braces = clause.match(/\{([^}]*)\}/);
			const outside = clause.replace(/\{[^}]*\}/, "").trim();
			if (outside) {
				// `X` or `* as X`: the default export, or the module itself.
				if (!outside.startsWith("*")) add(typeOnly ? u.types : u.values, "default", file);
			}
			for (const item of (braces?.[1] ?? "").split(",")) {
				const t = item.trim();
				if (!t) continue;
				const asType = typeOnly || /^type\s/.test(t);
				const name = t
					.replace(/^type\s+/, "")
					.split(/\s+as\s+/)[0]
					.trim();
				add(asType ? u.types : u.values, name, file);
			}
		}
		for (const m of text.matchAll(DYNAMIC_IMPORT)) if (isChecked(m[1])) use(m[1], file);
	}
	return uses;
}

// ---- resolution -------------------------------------------------------------------------------------

const packageName = (spec) => (spec.startsWith("@") ? spec.split("/").slice(0, 2).join("/") : spec.split("/")[0]);

/** node_modules directories a module next to `dir` would search, nearest first. */
function nodeModulesUp(dir) {
	const out = [];
	let d = dir;
	for (;;) {
		out.push(join(d, "node_modules"));
		const parent = dirname(d);
		if (parent === d) return out;
		d = parent;
	}
}

/** Where `name` lives: pi's own dependencies first (host-provided), then next to this package. */
function locatePackage(name, host) {
	if (name === HOST) return { dir: host.dir, from: "host" };
	for (const nm of nodeModulesUp(host.dir)) {
		const dir = join(nm, name);
		if (existsSync(join(dir, "package.json"))) return { dir, from: "host" };
	}
	for (const nm of nodeModulesUp(PKG)) {
		const dir = join(nm, name);
		if (existsSync(join(dir, "package.json"))) return { dir, from: "package" };
	}
	return undefined;
}

/** The `import` and `types` targets of `subpath` ("." or "./x") in a package's exports map. */
function exportTargets(pkgDir, subpath) {
	const pkg = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
	const pick = (value, condition) => {
		if (typeof value === "string") return value;
		if (!value || typeof value !== "object") return undefined;
		for (const key of [condition, "default", "import", "node"]) {
			if (key in value) {
				const v = pick(value[key], condition);
				if (v) return v;
			}
		}
		return undefined;
	};
	const map = pkg.exports;
	if (map === undefined) {
		if (subpath !== ".") return undefined;
		return {
			js: join(pkgDir, pkg.module ?? pkg.main ?? "index.js"),
			types: pkg.types ? join(pkgDir, pkg.types) : undefined,
		};
	}
	const entries = typeof map === "string" || !Object.keys(map).some((k) => k.startsWith(".")) ? { ".": map } : map;
	let hit;
	let star = "";
	for (const [key, value] of Object.entries(entries)) {
		if (key === subpath) {
			hit = value;
			break;
		}
		if (key.includes("*")) {
			const [before, after] = key.split("*");
			if (subpath.startsWith(before) && subpath.endsWith(after) && subpath.length >= key.length - 1) {
				hit = value;
				star = subpath.slice(before.length, subpath.length - after.length);
				break;
			}
		}
	}
	if (hit === undefined) return undefined;
	const fill = (t) => (t ? join(pkgDir, t.replace("*", star)) : undefined);
	const types = fill(pick(hit, "types")) ?? (subpath === "." && pkg.types ? join(pkgDir, pkg.types) : undefined);
	return { js: fill(pick(hit, "import")), types };
}

function resolveSpec(spec, host) {
	const name = packageName(spec);
	const where = locatePackage(name, host);
	if (!where) return { missing: true };
	const subpath = `.${spec.slice(name.length)}`;
	const targets = exportTargets(where.dir, subpath);
	if (!targets?.js || !existsSync(targets.js)) return { missing: true, packageDir: where.dir };
	return { ...targets, from: where.from, packageDir: where.dir };
}

// ---- declarations -----------------------------------------------------------------------------------

const declCache = new Map();

/** Whether the `.d.ts` at `file`, or one it re-exports wholesale, mentions `name` as an export. */
function declares(file, name, depth = 0) {
	if (!file || depth > 4 || !existsSync(file)) return false;
	let text = declCache.get(file);
	if (text === undefined) {
		text = readFileSync(file, "utf8");
		declCache.set(file, text);
	}
	if (new RegExp(`\\b${name}\\b`).test(text)) return true;
	for (const m of text.matchAll(/export\s+\*\s+from\s+["'](\.[^"']+)["']/g)) {
		// Declarations re-export with .ts, .js or .mjs names; the declaration file is the .d.ts / .d.mts next to them.
		const base = resolve(dirname(file), m[1]).replace(/(\.d)?\.m?[jt]s$/, "");
		const candidates = [`${base}.d.ts`, `${base}.d.mts`, join(base, "index.d.ts"), join(base, "index.d.mts")];
		const decl = candidates.find((c) => existsSync(c));
		if (decl && declares(decl, name, depth + 1)) return true;
	}
	return false;
}

// ---- main -------------------------------------------------------------------------------------------

const host = findHost();
const hostVersion = JSON.parse(readFileSync(join(host.dir, "package.json"), "utf8")).version;
const range = manifest.peerDependencies?.[HOST] ?? "*";
const report = {
	pi: { dir: host.dir, version: hostVersion, how: host.how, range },
	modules: [],
	failures: [],
	notes: [],
};

const versionOk = satisfies(hostVersion, range);
if (versionOk === false) report.failures.push(`${HOST} ${hostVersion} does not satisfy peerDependencies ${range}`);
if (versionOk === undefined) report.notes.push(`peerDependencies range ${range} not understood; version not checked`);

for (const [spec, u] of [...collectImports()].sort(([a], [b]) => a.localeCompare(b))) {
	const entry = {
		specifier: spec,
		files: [...u.files].sort(),
		values: [...u.values.keys()],
		types: [...u.types.keys()],
	};
	report.modules.push(entry);
	const resolved = resolveSpec(RUNTIME_ENTRY[spec] ?? spec, host);
	const optional = optionalPeers.has(packageName(spec));
	if (resolved.missing) {
		entry.status = optional ? "skipped" : "missing";
		const where = resolved.packageDir ? ` (${resolved.packageDir} has no such entry)` : "";
		const msg = `${spec}: not installed${where}; needed by ${entry.files.join(", ")}`;
		if (optional && !strict) report.notes.push(`optional peer ${msg}`);
		else report.failures.push(msg);
		continue;
	}
	entry.resolved = resolved.js;
	entry.from = HOST_PROVIDED.has(spec) ? "host" : resolved.from;
	if (!HOST_PROVIDED.has(spec))
		report.notes.push(
			`${spec} is not mapped by pi's extension loader: at runtime it must be in a node_modules next to this package (${
				resolved.from === "host"
					? "checked here against pi's own copy; not found next to this package"
					: "found next to this package"
			}); needed by ${entry.files.join(", ")}`,
		);
	let mod;
	try {
		mod = await import(pathToFileURL(resolved.js).href);
	} catch (e) {
		entry.status = "unloadable";
		report.failures.push(`${spec}: ${resolved.js} fails to load: ${e?.message ?? e}`);
		continue;
	}
	const missingValues = entry.values.filter((n) => !(n in mod));
	const missingTypes = entry.types.filter((n) => !(n in mod) && !declares(resolved.types, n));
	entry.status = missingValues.length || missingTypes.length ? "missing-exports" : "ok";
	for (const n of missingValues)
		report.failures.push(`${spec}: no export ${n} (used by ${[...u.values.get(n)].join(", ")})`);
	for (const n of missingTypes)
		report.failures.push(
			`${spec}: no type ${n} in ${resolved.types ?? "its declarations"} (used by ${[...u.types.get(n)].join(", ")})`,
		);
}

if (asJson) {
	console.log(JSON.stringify(report, null, 2));
} else {
	const ok = versionOk === false ? "FAIL" : "ok";
	console.log(`${HOST} ${hostVersion} at ${host.dir} (${host.how}); peer range ${range}: ${ok}`);
	for (const m of report.modules) {
		const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
		const counts = `${plural(m.values.length, "value")}, ${plural(m.types.length, "type")}`;
		const where = `${m.from ?? "-"}; ${plural(m.files.length, "file")}`;
		console.log(`  ${m.status.padEnd(16)} ${m.specifier}  (${counts}; ${where})`);
	}
	for (const n of report.notes) console.log(`note: ${n}`);
	for (const f of report.failures) console.log(`FAIL: ${f}`);
	console.log(report.failures.length ? `${report.failures.length} failure(s)` : "compatible");
}
process.exit(report.failures.length ? 1 : 0);
