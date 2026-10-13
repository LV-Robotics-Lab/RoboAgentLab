/**
 * The `motormind/<base>` provider (DESIGN §1.2, §1.4): the model that plays the MotorMind mission.
 *
 *   pi -p -e robots/libero --preset motormind --model motormind/<base> "..."
 *
 * Every pi turn is one step of the mission generator (./mission.ts): the provider reads the result
 * of its pending tool call from the transcript, resumes the generator with it, and answers with the
 * next robot call (`mm_act`, `mm_observe`) or `finish`. Within the turn the roles (planner, verifier,
 * executor, locate, supervise) are asked synchronously on `<base>` through ./models.ts; the turn's
 * usage carries the sum of their costs and the last call's tokens (ensemble's convention), and a
 * `motormind_turn` entry lists every call.
 *
 * Reentrancy (DESIGN §1.4 step 1): the pending tool call's id keys the turn. A result already
 * answered (pi retrying that turn) gets the message it was answered with, the mission does not
 * advance; a context that is
 * neither the opening user message nor the pending result (compaction, an interactive message) gets
 * a static text with zero usage. Reconciliation (step 2): a closed-loop refusal of the pending call
 * (the previous motion's outcome is unknown) is answered with `mm_observe`, after which the same
 * call is reissued; `episode_over` reaches the generator, which finishes.
 *
 * Fail-closed (DESIGN §1.7, provider side): at session start the model is `motormind/<base>` but
 * `--mm` is not true, or `<base>` (or a `--mm-role-models` entry) is not in the registry → the
 * tools are cleared and a non-UI run exits 1. Compaction is cancelled while this provider plays
 * (`session_before_compact`). agent_end aborts the in-flight role calls and writes `motormind_end`;
 * the mode's result block takes the mission's state (`result()`).
 */

import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	createProvider,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type TextContent,
	type ToolCall,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ModelRegistry } from "@earendil-works/pi-coding-agent";
import {
	type Ask,
	type Describers,
	type Executor,
	type FrameCache,
	frameOf,
	hasImages,
	type Json,
	type MissionEnd,
	type MmCall,
	type MmReply,
	missionClock,
	type Role,
	type RoleCallRecord,
	type RoleReply,
	recordOf,
	type SegmentReport,
	type SubgoalInput,
} from "./contract.ts";
import { runSubgoal as builtExecutor } from "./executor.ts";
import { MM_PROVIDER, MOTORMIND_EVENT, type MotorMindHandle } from "./index.ts";
import {
	DEFAULT_BUDGET,
	DEFAULT_MAX_REPLANS,
	MISSION_BUDGET_S,
	type MissionState,
	roleTotals,
	runMission,
	subgoalSummary,
} from "./mission.ts";
import { findModel, makeAsk, promptHashes, roleModels } from "./models.ts";
import { capabilityText, legend } from "./observe.ts";

/** The executor half's modules, injectable for tests (the defaults are ./executor.ts and ./observe.ts). */
export type ProviderDeps = { runSubgoal: (input: SubgoalInput) => Executor; describers: Describers };
const BUILT: ProviderDeps = { runSubgoal: builtExecutor, describers: { legend, capabilities: capabilityText } };

export const TURN_ENTRY = "motormind_turn";
export const END_ENTRY = "motormind_end";
/** The static answer to a context that is not the mission's (compaction, an interactive message). */
export const STATIC_TEXT =
	"MotorMind mission in progress: this provider answers only the mission's own turns (one robot call per turn).";

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const zeroUsage = (): Usage => ({ ...USAGE, cost: { ...USAGE.cost } });
const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);
const r3 = (v: number) => Number(v.toFixed(6));

/** `--model motormind/<base>` from `argv` (the last --model, which pi uses). */
export function motormindArgs(argv: readonly string[]): { base?: string } {
	const i = argv.map((a) => a === "--model" || a.startsWith("--model=")).lastIndexOf(true);
	const model = i < 0 ? undefined : argv[i] === "--model" ? argv[i + 1] : argv[i].slice("--model=".length);
	return { base: model?.startsWith(`${MM_PROVIDER}/`) ? model.slice(MM_PROVIDER.length + 1) : undefined };
}

/**
 * The last message of the context and what it is to the mission: the opening user message (the first
 * turn, or pi retrying it), the pending call's result (the next turn), a result already answered (pi
 * retrying that turn), or anything else.
 */
export function classify(
	messages: readonly Message[],
	pending: { id: string } | undefined,
	answered: { has: (id: string) => boolean },
	started: boolean,
): "opening" | "pending_result" | "answered" | "other" {
	const last = messages.at(-1);
	if (!last) return "other";
	if (last.role === "user")
		return started ? (messages.filter((m) => m.role === "user").length === 1 ? "answered" : "other") : "opening";
	if (last.role === "toolResult" && pending && last.toolCallId === pending.id) return "pending_result";
	if (last.role === "toolResult" && answered.has(last.toolCallId)) return "answered";
	return "other";
}

/** A closed-loop refusal of a motion: the previous motion's outcome is unknown (../../planner/closed-loop.ts). */
export const isClosedLoopRefusal = (text: string) => /refused: the last motion .* has an unknown outcome/i.test(text);

/** A tool result as the mission's reply; `frame` is the robot's latest observation. */
export function replyOf(
	result: Extract<Message, { role: "toolResult" }>,
	handle: MotorMindHandle | undefined,
): MmReply {
	const text = result.content
		.filter((c): c is TextContent => c.type === "text")
		.map((c) => c.text)
		.join("\n");
	let body: Json | undefined;
	try {
		const parsed = JSON.parse(text);
		if (isObject(parsed)) body = parsed;
	} catch {}
	const latest = handle?.latest();
	const frame = latest ? frameOf(latest) : undefined;
	const cancelled = body && isObject(body.cancelled) ? (body.cancelled as MmReply["cancelled"]) : undefined;
	return {
		ok: !result.isError,
		...(result.isError ? { error: text } : {}),
		...(body ? { body } : {}),
		...(frame ? { frame } : {}),
		...(body && Array.isArray(body.segments) ? { segments: body.segments as SegmentReport[] } : {}),
		...(cancelled ? { cancelled } : {}),
		...(body?.timed_out === true ? { timed_out: true } : {}),
		episode_over: body?.episode_over === true || frame?.episode_over === true,
	};
}

/** Register `motormind/<base>` when `argv` asks for it; returns the result block for the robot's row, or undefined. */
export function motormindProvider(
	pi: ExtensionAPI,
	argv: readonly string[] = process.argv,
	deps: ProviderDeps = BUILT,
) {
	const { base: found } = motormindArgs(argv);
	if (!found) return undefined;
	const base: string = found;
	pi.registerFlag("mm-budget", {
		type: "string",
		default: String(DEFAULT_BUDGET),
		description: "MotorMind: actions per subgoal (the planner's budget)",
	});
	pi.registerFlag("mm-budget-s", {
		type: "string",
		default: String(MISSION_BUDGET_S),
		description: "MotorMind: the mission's wall clock in seconds, from the first planner call",
	});
	pi.registerFlag("mm-max-replans", {
		type: "string",
		default: String(DEFAULT_MAX_REPLANS),
		description: "MotorMind: how many replans a mission may make",
	});
	pi.registerFlag("mm-role-models", {
		type: "string",
		default: "",
		description: `MotorMind: role=provider/id,... overriding <base> per role (${["planner", "verify", "executor", "locate", "supervise", "monitor", "memory"].join(", ")}); empty = every role on <base>`,
	});

	let registry: ModelRegistry | undefined;
	let ctx: ExtensionContext | undefined;
	let handle: MotorMindHandle | undefined;
	pi.events.on(MOTORMIND_EVENT, (h) => {
		handle = h as MotorMindHandle;
	});

	type Pending = { id: string; call: MmCall; message: AssistantMessage; consumed: boolean };
	type Run = {
		mission: ReturnType<typeof runMission>;
		cache: FrameCache;
		turns: number;
		calls: RoleCallRecord[];
		models: Record<Role, string>;
		answered: Set<string>;
		pending?: Pending;
		/** The message each consumed result (by its tool call id; "" = the opening message) was answered with. */
		answers: Map<string, AssistantMessage>;
		/** The call to reissue after a reconciliation observe. */
		reissue?: MmCall;
		end?: MissionEnd;
		finished: boolean;
		episode: AbortController;
	};
	let run: Run | undefined;
	let configError: string | undefined;

	const model: Model<Api> = {
		id: base,
		name: `MotorMind: ${base}`,
		api: "openai-completions" as Api,
		provider: MM_PROVIDER,
		baseUrl: "",
		reasoning: false,
		input: ["text", "image"],
		cost: ZERO_COST,
		contextWindow: 128_000,
		maxTokens: 16_384,
	};

	/** Fail closed: no tools, and a non-UI run exits 1. */
	function failClosed(why: string, c: ExtensionContext) {
		configError = why;
		pi.setActiveTools([]);
		if (c.hasUI) c.ui.notify(`motormind: ${why}`, "error");
		else {
			console.error(`[motormind] ${why}`);
			process.exitCode = 1;
			c.shutdown();
		}
	}

	pi.on("session_start", (_event, c) => {
		ctx = c;
		registry = c.modelRegistry;
		run = undefined;
		configError = undefined;
		if (c.model?.provider !== MM_PROVIDER) return;
		const mm = pi.getFlag("mm");
		if (!(mm === true || String(mm ?? "").trim() === "true"))
			return failClosed(
				`--model ${MM_PROVIDER}/${base} plays the MotorMind mission, which needs --mm=true (or --preset motormind)`,
				c,
			);
		let models: Record<Role, string>;
		try {
			models = roleModels(base, String(pi.getFlag("mm-role-models") ?? ""));
		} catch (err) {
			return failClosed(err instanceof Error ? err.message : String(err), c);
		}
		const missing = [...new Set(Object.values(models))].filter((ref) => !findModel(registry!, ref));
		if (missing.length)
			return failClosed(`unknown model ${missing.join(", ")} (the registry has no such provider/id)`, c);
		for (const name of ["mm-budget", "mm-budget-s", "mm-max-replans"]) {
			const v = Number(pi.getFlag(name));
			if (!(Number.isFinite(v) && v > 0))
				return failClosed(`--${name} must be a positive number, got ${String(pi.getFlag(name))}`, c);
		}
		const found = findModel(registry!, base);
		if (found && c.model) {
			c.model.contextWindow = found.contextWindow;
			c.model.maxTokens = found.maxTokens;
		}
	});

	/** No compaction while the mission plays: the summary would be asked of this provider (DESIGN §1.2). */
	pi.on("session_before_compact", (_event, c) => (c.model?.provider === MM_PROVIDER ? { cancel: true } : undefined));

	/** The wrap-up (DESIGN §7.3): abort the in-flight calls, record how the mission stands. Registered before the robot's report. */
	pi.on("agent_end", () => {
		if (!run) return;
		run.episode.abort();
		pi.appendEntry(END_ENTRY, {
			status: run.end?.status ?? run.mission.state().status ?? null,
			ended_on: run.end?.ended_on ?? run.mission.state().ended_on ?? null,
			turns: run.turns,
			finished: run.finished,
			calls: run.calls.length,
		});
	});

	function startRun(): Run {
		const models = roleModels(base, String(pi.getFlag("mm-role-models") ?? ""));
		const episode = new AbortController();
		const cache: FrameCache = { stale: true };
		const r = {
			turns: 0,
			calls: [],
			models,
			answered: new Set(),
			answers: new Map(),
			finished: false,
			episode,
			cache,
		} as unknown as Run;
		let turnSignal: AbortSignal | undefined;
		const baseAsk = makeAsk(registry!, models, (role, reply: RoleReply) => {
			r.answered.add(reply.model);
			r.calls.push(recordOf(role, reply));
		});
		const ask: Ask = (role, prompt, images: ImageContent[], o) => {
			const signals = [episode.signal, ...(turnSignal ? [turnSignal] : []), ...(o?.signal ? [o.signal] : [])];
			return baseAsk(role, prompt, images, { ...o, signal: AbortSignal.any(signals) });
		};
		const clock = missionClock((c) => (c ? handle?.cancel(c.reason) : handle?.clearCancel()));
		r.mission = runMission({
			task: handle?.task() ?? "",
			budget: Number(pi.getFlag("mm-budget")) || DEFAULT_BUDGET,
			budgetS: Number(pi.getFlag("mm-budget-s")) || MISSION_BUDGET_S,
			maxReplans: Number(pi.getFlag("mm-max-replans")) || DEFAULT_MAX_REPLANS,
			ask,
			clock,
			cache,
			describers: deps.describers,
			runSubgoal: deps.runSubgoal,
			log: (entry, data) => pi.appendEntry(entry, data),
		});
		const setSignal = (s: AbortSignal | undefined) => {
			turnSignal = s;
		};
		Object.assign(r, { setSignal, clock });
		return r;
	}
	type RunX = Run & { setSignal: (s: AbortSignal | undefined) => void; clock: ReturnType<typeof missionClock> };

	/** The call as pi's tool call. */
	function toolCallOf(call: MmCall | { tool: "finish"; args: Json }, n: number): ToolCall {
		return {
			type: "toolCall",
			id: `motormind-${n}`,
			name: call.tool,
			arguments: call.args as unknown as ToolCall["arguments"],
		};
	}

	function textOf(calls: RoleCallRecord[], call: ToolCall | undefined, end: MissionEnd | undefined): string {
		const roles = calls.map((c) => `${c.role}${c.ok ? "" : `✗${c.error_kind ? `(${c.error_kind})` : ""}`}`).join(" ");
		const act = call
			? call.name === "mm_act"
				? `mm_act ${String((call.arguments as Json).label ?? (call.arguments as Json).type ?? "")}`
				: call.name
			: "";
		return `[motormind] ${roles || "no role call"}${act ? ` → ${act}` : ""}${end ? ` → ${end.status}: ${end.summary}` : ""}`.trim();
	}

	async function turn(
		context: TranscriptContext,
		options: SimpleStreamOptions | undefined,
		out: AssistantMessageEventStream,
	) {
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: zeroUsage(),
			stopReason: "stop",
			timestamp: Date.now(),
		};
		const say = (m: AssistantMessage) => {
			out.push({ type: "start", partial: m });
			m.content.forEach((c, i) => {
				if (c.type === "text") {
					out.push({ type: "text_start", contentIndex: i, partial: m });
					out.push({ type: "text_delta", contentIndex: i, delta: c.text, partial: m });
					out.push({ type: "text_end", contentIndex: i, content: c.text, partial: m });
				} else if (c.type === "toolCall") {
					out.push({ type: "toolcall_start", contentIndex: i, partial: m });
					out.push({ type: "toolcall_end", contentIndex: i, toolCall: c, partial: m });
				}
			});
			out.push({ type: "done", reason: m.stopReason as "stop" | "toolUse", message: m });
			out.end();
		};
		const fail = (why: string, aborted = false) => {
			message.stopReason = aborted ? "aborted" : "error";
			message.errorMessage = why;
			out.push({ type: "error", reason: message.stopReason, error: message });
			out.end();
		};
		if (configError) return fail(configError);
		if (!registry || !ctx) return fail("motormind: no session");
		const messages = context.messages as Message[];
		const kind = classify(messages, run?.pending, run?.answers ?? new Set<string>(), run !== undefined);
		if (kind === "other" || (run?.finished && kind !== "pending_result" && kind !== "answered")) {
			message.content.push({ type: "text", text: STATIC_TEXT });
			return say(message);
		}
		// pi retrying a turn (its own retry, or after an aborted stream): the message that turn produced, the mission does not advance
		if (kind === "answered" && run) {
			const last = messages.at(-1)!;
			const was = run.answers.get(last.role === "toolResult" ? last.toolCallId : "");
			return was ? say(was) : fail("motormind: no answer recorded for this context");
		}
		run ??= startRun();
		const r = run as RunX;
		if (!handle) return fail("motormind: the robot published no MotorMind handle (is --mm on and the robot up?)");
		if (r.finished) {
			message.content.push({ type: "text", text: STATIC_TEXT });
			return say(message);
		}
		const before = r.calls.length;
		let reply: MmReply | undefined;
		// what this turn answers: the pending result's id, or "" for the opening message
		const answering =
			kind === "pending_result" ? (messages.at(-1) as Extract<Message, { role: "toolResult" }>).toolCallId : "";
		if (kind === "pending_result") {
			const result = messages.at(-1) as Extract<Message, { role: "toolResult" }>;
			r.pending!.consumed = true;
			reply = replyOf(result, handle);
			// the cache: a reply with images is fresh; one without (an action that stopped early, an error) is stale
			if (reply.frame) {
				r.cache.latest = reply.frame;
				r.cache.stale = !hasImages(reply.frame) || !reply.ok;
			} else r.cache.stale = true;
			if (r.pending!.call.tool === "mm_observe" && r.reissue && reply.ok) {
				// reconciled: the world is known again; the refused call goes out as it was
				const call = r.reissue;
				r.reissue = undefined;
				return emit(call, [], undefined);
			}
			if (!reply.ok && reply.error && isClosedLoopRefusal(reply.error)) {
				r.reissue = r.pending!.call;
				return emit({ tool: "mm_observe", args: {}, why: "reconcile" }, [], undefined);
			}
		}
		r.setSignal(options?.signal);
		let next: IteratorResult<MmCall, MissionEnd>;
		try {
			next = reply === undefined ? await r.mission.next() : await r.mission.next(reply);
		} catch (err) {
			r.setSignal(undefined);
			const why = err instanceof Error ? err.message : String(err);
			r.end = { status: "error", ended_on: "mission_threw", summary: why };
			r.finished = true;
			return fail(`motormind: ${why}`, options?.signal?.aborted === true);
		}
		r.setSignal(undefined);
		handle?.setDeadline(r.clock.deadline_at);
		const calls = r.calls.slice(before);
		if (options?.signal?.aborted) return fail("aborted", true);
		if (next.done) {
			r.end = next.value;
			r.finished = true;
			return emit(
				{
					tool: "finish",
					args: {
						status: next.value.status === "task_done" ? "success" : "failure",
						summary: `${next.value.status} (${next.value.ended_on}): ${next.value.summary}`,
					},
				},
				calls,
				next.value,
			);
		}
		return emit(next.value, calls, undefined);

		function emit(
			call: MmCall | { tool: "finish"; args: Json },
			calls: RoleCallRecord[],
			end: MissionEnd | undefined,
		) {
			r.turns++;
			const tc = toolCallOf(call, r.turns);
			const text = textOf(calls, tc, end);
			message.content.push({ type: "text", text }, tc);
			message.stopReason = "toolUse";
			// ensemble's convention: the turn's cost is every call's; its tokens are the last call's.
			const last = calls.at(-1);
			const usage = zeroUsage();
			usage.input = last?.tokens_in ?? 0;
			usage.output = last?.tokens_out ?? 0;
			usage.totalTokens = usage.input + usage.output;
			usage.cost.total = r3(calls.reduce((a, c) => a + c.cost_usd, 0));
			message.usage = usage;
			r.answers.set(answering, message);
			if (call.tool !== "finish") r.pending = { id: tc.id, call, message, consumed: false };
			else r.pending = undefined;
			pi.appendEntry(TURN_ENTRY, {
				turn: r.turns,
				calls,
				tool_call: { name: tc.name, arguments: tc.arguments },
				cost_usd: usage.cost.total,
				...(end ? { end } : {}),
				...(r.reissue ? { reconcile: r.reissue.tool } : {}),
			});
			say(message);
		}
	}

	function streamSimple(_model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const out = createAssistantMessageEventStream();
		void turn(context, options, out);
		return out;
	}
	pi.registerProvider(
		createProvider({
			id: MM_PROVIDER,
			name: "MotorMind",
			auth: { apiKey: { name: "MotorMind", resolve: async () => ({ auth: {} }) } },
			models: [model],
			api: { stream: streamSimple, streamSimple },
		}),
	);

	return {
		/** The mission's part of `robot_result.motormind` (DESIGN §7.3); merged over the mode's block by ../../robot.ts. */
		result: (): Json => {
			const s: MissionState | undefined = run?.mission.state();
			return {
				...(s?.status ? { status: s.status, ended_on: s.ended_on } : {}),
				models: {
					configured: run?.models ?? roleModels(base, String(pi.getFlag("mm-role-models") ?? "")),
					answered: [...(run?.answered ?? [])],
				},
				prompts: promptHashes(),
				turns: run?.turns ?? 0,
				plans: s?.plans ?? 0,
				replans: s?.replans ?? 0,
				subgoals: s ? subgoalSummary(s.subgoals) : [],
				clauses: s?.clauses ?? null,
				net_motion_disagreements: s?.net_motion_disagreements ?? 0,
				budget: Number(pi.getFlag("mm-budget")) || DEFAULT_BUDGET,
				budget_s: Number(pi.getFlag("mm-budget-s")) || MISSION_BUDGET_S,
				elapsed_s: s?.elapsed_s ?? 0,
				left_s: s?.left_s ?? null,
				actions_cap: s?.actions_cap ?? 0,
				mission_actions: s?.actions ?? 0,
				observes: s ? { total: s.observes.length, yielded: s.observes.filter((o) => o.yielded).length } : null,
				...(s?.replan_refused_for_price ? { replan_refused_for_price: s.replan_refused_for_price } : {}),
				by_role: roleTotals(run?.calls ?? []),
				cost_usd: r3((run?.calls ?? []).reduce((a, c) => a + c.cost_usd, 0)),
			};
		},
	};
}
