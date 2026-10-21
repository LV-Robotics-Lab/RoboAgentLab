# Migrating to the official pi

Until this change the extension was built against the RoboAgentLab fork of pi, and a user had to
clone the fork, build it, `npm link` its CLI and only then `pi install ./packages/embodied`. The fork
differed from the published pi in three places, all of which the extension used:

| Fork change | Where the extension used it | Now |
|---|---|---|
| `ExtensionContext.exportSession(format, path)` | dashboard `GET /download/session` | written by the extension itself from pi's public session manager (`src/capabilities/dashboard/session-export.ts`) |
| `ExtensionContext.withdrawQueuedMessage(text)` (with `Agent.removeQueuedMessage`) | dashboard `POST /message/withdraw` | still used when present; on an official pi the route answers HTTP 501 and the message stays queued |
| `expired while waiting for admission` in pi-ai's retryable-error pattern | nothing in the extension: it made pi retry an NInfer admission timeout that arrived mid-stream | dropped (see below) |

## What to do

```bash
npm install -g @earendil-works/pi-coding-agent@1.3.0
git clone https://github.com/RoboAgentLab/RoboAgentLab ~/RoboAgentLab
node ~/RoboAgentLab/packages/embodied/scripts/check-pi-compat.mjs
pi install ~/RoboAgentLab/packages/embodied
```

`check-pi-compat.mjs` finds the installed pi (or takes `--pi <dir>`), checks its version against the
package's `peerDependencies` (now `^1.3.0`: 1.3.0 is the version these checks ran against) and
verifies that every value and type the extension imports from `@earendil-works/*` and `typebox` is
exported by that pi. It fails on a missing export, so a future pi that drops something the extension
needs is caught before `pi install`.

An existing fork-based install keeps working: nothing here needs the fork, and the fork's own pi
still has the two context methods. `services/` is unchanged.

## What the dashboard loses

- **Session download.** The JSONL is byte-for-byte what pi's `/export` writes (a fresh header, then
  the branch chained by `parentId`). The HTML is a plain transcript rendered by the extension (user and
  assistant text, thinking, tool calls and results, the camera frames, the extension's own entries),
  not pi's themed export: no markdown rendering, no syntax highlighting, no theme colours.
- **Withdrawing a queued message.** pi 1.3.0 has no public way for an extension to take a message
  back out of the steering queue. The dashboard's SSE snapshot reports `capabilities.queueWithdrawal:
  false`, the Withdraw button's request answers 501 with that explanation, and the message is
  delivered when the agent takes it. On the fork the button keeps working.

## The retry pattern

The fork added `expired while waiting for admission` to pi-ai's list of retryable provider errors so a
planner request that NInfer dropped from its admission queue mid-stream (its SSE error carries only
that text; the HTTP 503 form already matches pi's `503` pattern) was retried like an overload. pi
1.3.0 offers no extension hook into that classifier, and NInfer stopped being the selfhost engine on
2026-10-13 (`src/planner/cache-breakpoints.ts`, `src/infra/params.ts`), so the pattern is dropped
rather than re-implemented. On an NInfer endpoint such a mid-stream drop now ends the turn as an
error that pi reports; the operator re-sends the prompt. An endpoint that answers with HTTP 503
before streaming is still retried by pi.

## Not covered by this migration

The durable policy (`src/capabilities/policy`, which the MetaWorld robot registers and loads only when
`--policy-store` is set) and `src/scripts/eval-durable.ts` import `@earendil-works/pi-durable` and
`@earendil-works/chord`, and the MCP server behind the Codex and Claude Code plugins
(`src/integrations/mcp`) imports `@earendil-works/pi-mcp`. pi's extension loader maps only
pi-coding-agent, pi-agent-core, pi-tui, pi-ai and typebox to its own copies, so these resolve from a
`node_modules` next to the package: the fork workspace, or `npm install --no-save
@earendil-works/pi-durable@1.3.0 @earendil-works/chord@1.3.0 @earendil-works/pi-mcp@1.3.0` in
`packages/embodied`. Every robot extension, the dashboard and `/embodied-setup` load without them (the
MetaWorld robot included). pi-durable and chord are optional peers in `package.json`, and the compat
check reports all three as "not mapped by pi's extension loader".
