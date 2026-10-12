#!/usr/bin/env node
/**
 * params-match.mjs <result.json>: whether a recorded episode ran with the configuration the eval
 * script runs now (PARAMS.md 2.6). The robot records `params` (every experiment flag's effective
 * value) and `params_default` (its default) in its result (src/infra/params.ts); PI_ARGS_JSON is
 * the pi arguments the script passes (a JSON list). A flag given on the command line must have
 * run with that value, every other one with its default; `--tier` / `--preset` stand for the flags
 * they expand to (src/infra/tiers.ts), which must have run with the expanded values. A flag the
 * result does not record was added after it ran: it ran with the default the flag was introduced
 * with (params.ts INTRODUCED), compared like a recorded one; a flag listed nowhere is skipped and
 * named on stderr as unrecorded. Exit 0 = the same configuration (or a result written before
 * `params` existed: the script's own checks judge it), 2 = another one (the differing flags on
 * stderr). The cell's own flags (task, seed, ...) are the directory's, not compared.
 */
import { readFileSync } from "node:fs";
import { argvFlags, DEPLOYMENT, INTRODUCED } from "../infra/params.ts";
import { choose } from "../infra/tiers.ts";

const CELL = new Set(["task", "seed", "suite", "task-config", "split", "scene", "layout-set", "episode"]);
// pi's own flags the eval scripts pass: not experiment parameters, never recorded.
const PI = new Set(["model", "thinking", "provider", "api-key", "mode", "session-dir"]);

const r = JSON.parse(readFileSync(process.argv[2], "utf8"));
if (!r.params || !r.params_default) process.exit(0);
// Every `--name[=value]` given, by pi's rules (the robot's and tier-flags.mjs' parse).
const given = argvFlags(JSON.parse(process.env.PI_ARGS_JSON || "[]"));
const norm = (v) => (v === null || v === undefined ? "" : v === "pure" ? "true" : String(v));
// The choice's flags read as its values unless given themselves (the robot's rule: tiers.ts `expand`).
const choice = choose(given.get("tier"), given.get("preset"));
const expanded = choice && !("error" in choice) ? choice.flags : {};
const differ = [];
const unrecorded = [];
for (const name of new Set([...Object.keys(r.params), ...Object.keys(INTRODUCED), ...given.keys()])) {
	if (CELL.has(name) || DEPLOYMENT.has(name) || PI.has(name)) continue;
	const recorded = name in r.params;
	if (!recorded && !(name in INTRODUCED)) {
		unrecorded.push(`--${name}`);
		continue;
	}
	const ran = recorded ? r.params[name] : INTRODUCED[name].introduced;
	const def = recorded ? r.params_default[name] : INTRODUCED[name].default;
	const want = given.has(name) ? given.get(name) : name in expanded && expanded[name] !== null ? expanded[name] : def;
	if (norm(ran) !== norm(want))
		differ.push(`--${name} ran ${JSON.stringify(ran)}${recorded ? "" : " (unrecorded: its default when introduced)"}, now ${JSON.stringify(want)}`);
}
if (unrecorded.length) console.error(`${process.argv[2]}: unrecorded in the old result, not compared: ${unrecorded.join(", ")}`);
if (differ.length) {
	console.error(`${process.argv[2]}: another configuration: ${differ.join("; ")}`);
	process.exit(2);
}
