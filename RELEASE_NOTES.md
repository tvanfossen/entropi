_Last 10 releases. Older history: [OLD_NOTES.md](OLD_NOTES.md). Kept short
because `gh release create --notes-file` hits GitHub's 125,000-char release
body limit once this file accumulates full project history — see v2.9.3._

# entropic v2.12.0

Minor release — **one hosted engine can now serve several callers without them
seeing each other**, plus a tool-call crash that killed whole runs.

All three issues came from one consumer building the thing the bridge always
documented as its purpose: a host that keeps one model resident and serves
several MCP clients. Two of the three were caused by contracts our own headers
stated and nothing implemented.

## Highlights

- **Per-caller sessions.** A `session` key on the bridge's ask tool, and an
  `entropic_*_session` C API family, give each caller its own conversation,
  context, system-prompt seed and KV sequence.
- **Concurrent runs are no longer undefined behaviour.** A second run on a
  handle is refused with `ENTROPIC_ERROR_ALREADY_RUNNING`; the bridge queues
  callers FIFO instead of racing them.
- **An argument-free tool call no longer kills the run.**
- **Consumers name their own tools.** `tool_prefix`, `server_name` and
  per-tool description overrides.

## Engine bug fixes

- **gh#143** — a tool call carrying no arguments serialised to the string
  `"null"`; every built-in server then threw `type_error.306` out of dispatch
  and aborted the turn. `git.diff` with no arguments is a legitimate shape, so
  this was reachable from any model on any turn. Fixed at the origin, plus a
  dispatch-level exception barrier that turns any throwing tool into a tool
  error the model can correct — the in-process counterpart of the guarantee
  gh#133 gave plugin servers. `ContextInspectTool` was independently
  unguarded and is fixed too.
- **gh#144 (data race)** — `@threadsafety Serialized per-handle.` on the six
  run entry points has been false since gh#109 removed `api_mutex` from them.
  The bridge serves each client on its own thread, so two concurrent asks
  raced the shared conversation and decoded concurrently on one
  `llama_context`. The doc is corrected and the documented error code is now
  actually returned.
- Two missing `handle->engine` null checks in `entropic_context_get` and
  `entropic_context_count`, which their siblings already had.
- The `try_warm_reuse` comment claiming an interleaved conversation "falls
  back" was wrong — neither stated branch fires. Corrected, and the underlying
  hazard fixed by per-sequence residency.

## New features

- **gh#144 — keyed conversations.** `entropic_run_session`,
  `_run_session_as`, `_run_session_streaming`, `entropic_session_context_get`
  / `_count` / `_clear`, `entropic_session_drop`, `entropic_session_list`.
  A NULL or empty key means the default session, so every existing caller is
  unaffected. The key is opaque: two callers passing the same string share a
  conversation deliberately.
- **gh#144 — session pool.** `max_sessions` on a tier derives the whole KV
  geometry (`n_seq_max`, `kv_unified`, `n_ctx`) rather than exposing three
  knobs that can disagree. `context_length` is PER SESSION.
- **gh#144 — occupancy.** The status tool reports `busy`, `queue_depth` and
  the session table, so a queued caller is distinguishable from a hung one.
- **gh#145 — consumer identity.** `mcp.external.tool_prefix`,
  `server_name` and `tool_descriptions`. All default to today's values.

## Breaking changes

None to the C ABI. Every addition is a new named function, so
`ENTROPIC_API_VERSION` is unchanged.

One behaviour change consumers will notice: **a concurrent run on one handle
now returns `ENTROPIC_ERROR_ALREADY_RUNNING` instead of racing.** Callers that
were relying on the stale "serialized per-handle" doc were getting undefined
behaviour, not queueing. A host that wants callers to WAIT should serialize
above the C API, as the external bridge now does.

`include/entropic/types/config.h` gains members on `ExternalMCPConfig` and
`ModelConfig`. The C ABI is opaque-handle based, so this affects only C++
consumers that construct `ParsedConfig` directly and must recompile.

## Distribution

- CPU tarball: `entropic-2.12.0-linux-x86_64-cpu.tar.gz` (sha256 in companion file)
- CUDA tarball: `entropic-2.12.0-linux-x86_64-cuda.tar.gz` (sha256 in companion file)
- Python wrapper: `pip install entropic-engine==2.12.0` then `entropic install-engine`

## Known limitations

- **Prefill under `speculative.mtp` is unchanged.** `mtp_init_run` clears the
  whole context and re-prefills every turn, so warm-keep and the prompt cache
  never run in that configuration. A consumer measured 95:1 prefill-to-generate
  with ~85% of prefill being a provably invariant prefix. Prefix retention is
  tracked separately.
- The session pool is mutually exclusive with `entropic_run_batch`: gh#98's
  fan-out needs a unified KV buffer and a pool needs private streams. The
  combination is refused at configure time with a typed error naming both keys.
- A separate draft model (as opposed to a target-owned MTP head) with
  `max_sessions > 1` is likewise refused — both contexts decode the same
  sequence id, and no target-to-draft sequence mapping exists.

# entropic v2.11.1

Patch release — **a config bug that only ever hit fresh installs and CI, and the
build gate that was hiding it.**

Both defects were invisible on a developer machine by construction. That is the
theme.

## The bundled default outranked layers it should never have touched

`load_layered` runs a bundled-default fallback when no model tiers are
configured anywhere. It ran that fallback **after** the project layer, and it
re-parsed the entire `data/default_config.yaml` straight over the
already-populated config — so every setting an explicit, higher-precedence layer
had established was silently overwritten by the layer that is supposed to be the
*least* specific one.

`default_config.yaml` sets `mcp.enable_bash: true`. A project config asking for
`false` got `true` back. That is REQ-CFG-001's most-specific-layer-wins rule
being broken from below.

**Who this hit:** anyone with no `~/.entropic/config.yaml` declaring tiers —
every fresh install, and every CI runner. On a machine that *does* have one, the
fallback never fires and the bug cannot be observed.

The fallback now parses into a scratch config and transplants only the model
block. The condition that triggers it is a missing model set, so the model set
is the only thing it may supply. `REQ-CFG-007` (usable configuration with no
user config present) still holds and is pinned by its own assertion, so this
cannot regress into the opposite bug.

## doxygen-guard is pinned, and CI is green again

The hook was on `rev: main`, a mutable ref. CI installs it fresh every run; a
developer machine keeps whatever it cached. The two ran **different builds of
the same tool** and disagreed about identical code — 3 violations on one, 578 on
the other, same commit. The two builds even print different exemption
vocabularies in their own error text, so the message could not be trusted to
describe the build that produced it.

Now pinned to `v1.4.2`. Upgrades become a deliberate, reviewed act.

With the guard passing, CI reached the unit tests for the first time in over
three weeks — which is how the config bug above was found. A gate that fails
early hides everything behind it.

## `@internal` → `@dg_internal`

`@internal` is a reserved doxygen command; using it as a guard exemption
overloaded a tag doxygen already owns. v1.4.2 defines its own
(`EXEMPTION_TAGS = {"utility", "dg_internal", "callback"}`), so all 929
occurrences across 108 files move to the tag the tool owns outright.

Nothing is newly exempt — every one of these was already exempt under the old
spelling, and the exemption ratio is unchanged.

## Three documentation defects found underneath

| | |
|---|---|
| `backend.cpp` | `evaluate_logprobs` had **two** `@return` tags; only the first was ever used, so the detailed one was dead text. Kept the detailed one. |
| `filesystem.cpp` | two functions' docs **merged into one block** — an "apply string replacement" brief and its four params sat atop `count_occurrences`, documenting nothing. |
| `external_bridge.h` | stale class `@version`. |

## Distribution

- CPU tarball: `entropic-2.11.1-linux-x86_64-cpu.tar.gz` (sha256 in companion file)
- CUDA tarball: `entropic-2.11.1-linux-x86_64-cuda.tar.gz` (sha256 in companion file)
- Python wrapper: `pip install entropic-engine==2.11.1` then `entropic install-engine`

## Known limitations

- Patch release: unit tests only per the version gate. The v2.11.0 model-suite
  result (74/74, 0 skipped, 0 failed) stands; no engine inference path changed
  here.
- `docs/roadmap.md` still reports a stale "Current State" and is not maintained
  alongside GitHub issues.

---

# entropic v2.11.0

Minor release — **the requirements catalog is back and enforced, and four
places where the engine documented behaviour it did not have are corrected.**

Mostly about the engine telling the truth about itself. One feature, #141.

## Requirements traceability, restored and made durable

`docs/requirements.yaml` has been missing since v2.1.0, where a `docs/` purge
deleted all 1008 lines as collateral. Nothing failed, for fifteen months,
because all three consumers of that file fail *open* when it is absent — the
cross-reference check self-disables on an empty catalog, the coverage check only
verifies the config names a path, and impact reporting quietly degrades to "No
requirements affected."

The catalog is rebuilt **from the shipped implementation**, not written ahead of
it: eight parallel subsystem sweeps produced 112 requirements, reconciled to
105. Cross-cutting rules that each sweep had restated at its own boundary were
merged (`REQ-ABI-001/002`, `REQ-SAFE-001`), and one subsystem that no sweep
owned — `ExternalBridge`, whose header and implementation sat in different
scopes — was written by hand as `REQ-BRIDGE-001`.

Coverage went **10/105 → 105/105**. Requirement tags went from 21 against 2318
exemptions (0.9%) to 1634 against 1530 (51.6%).

**New gate: `inv check-requirements`**, wired into pre-commit. Three checks:

- orphan requirement ids, *including on bodiless header declarations* —
  doxygen-guard skips those entirely, which is how a dead `REQ-INFER-003`
  survived unnoticed in `i_inference_backend.h`
- catalog entries nothing implements
- exemption creep, which doxygen-guard's own coverage command is structurally
  blind to because it never collects an `@internal`-only function

A missing catalog now fails closed.

## Documented behaviour that did not exist

The sweeps were told the implementation is the specification. They read headers,
which describe intent — and five times the intent had never been built:

| | |
|---|---|
| `entropic_set_error_callback` | returned `ENTROPIC_OK` while discarding the callback; the type is invoked from nowhere. Now returns `ENTROPIC_ERROR_NOT_IMPLEMENTED`, and the header says so. |
| `FileAccessTracker::was_read_unchanged` | no production caller; the gate only ever checked *was read*, not *unchanged*. Removed, along with a test named `..._detects_external_change` whose own comment conceded the write succeeds. |
| Bash/git command timeout | stored, logged, exposed by an accessor, never enforced. Catalog corrected; filed as #140. |
| `REQ-MCP-021` / `REQ-MCP-023` | both asserted the two above. Rewritten to describe what the code does. |
| VRAM budget resolution | `orchestrator.h:446` documented `env -> cudaMemGetInfo -> 0`; line 594 conceded the `cudaMemGetInfo` half was "intentionally deferred". The gate it feeds has therefore never run on a default deployment. Built — see below. |

**Behaviour change:** `entropic_set_error_callback` now returns
`ENTROPIC_ERROR_NOT_IMPLEMENTED` where it previously returned `ENTROPIC_OK`.
Nothing that relied on the callback firing can break, because it never fired.

## The VRAM admission gate now actually runs (#142)

**This is the behaviour change in this release. Read it before upgrading.**

`ModelOrchestrator::residency_admits()` has refused over-large tiers with
`ENTROPIC_ERROR_TIER_MODEL_TOO_LARGE` since v2.2.4. It never fired. The gate is
guarded by `vram_budget_bytes_ > 0`, and that value came only from
`ENTROPIC_VRAM_BUDGET_BYTES` — unset on every real deployment, so the budget read
as "unknown" and the gate disabled itself. A tier that could not fit went straight
to llama.cpp and took the **host process** down on
`ggml-backend.cpp:179 GGML_ASSERT(buffer)`.

The budget now falls back to the free VRAM the device reports. **Free, not
total**: the reported failure was an operator whose GPU was busier than the
developer's, and a total-derived budget would admit the load and abort anyway.

Switching the gate on required fixing the estimate first, because the old one was
wrong in three ways that all bias toward refusing configurations that work:

| | before | now |
|---|---|---|
| weights | entire file, regardless of `gpu_layers` | priced by offload placement |
| KV cache | flat 16 KiB/token, any `cache_type` | scaled by type — q4_0 is ~0.28x f16, not 1.0x |
| vision projector | not counted | counted — it is the allocation that failed |

**A partially offloaded tier is deliberately not priced at all.** How much of the
file lands on the device depends on a layer count only GGUF metadata carries, so
the estimate reports *unknown* and the gate stays open rather than guessing.
Qwen3.6-35B-A3B IQ3_XXS (~13 GB) runs at `gpu_layers=15` on an 11 GB card, and a
gate that guessed would refuse it. Refusing a working configuration is worse than
missing a broken one, because the operator cannot tell a false refusal from a real
one.

A refusal also logs the largest context length that *would* fit, so what comes
back is a setting rather than a wall.

### What the estimate does not count

llama.cpp reserves graph/activation scratch ("compute buffers") per context at
load time, sized by ubatch and model internals rather than by context length.
Measured here: **1222 MiB** for a gemma-4 E4B MTP head's context at a 512-token
ubatch — more than twice the default `vram_reserve_mb` of 512, and a speculative
configuration pays it twice because it holds two contexts.

It is not estimated, because it cannot be derived without reading GGUF metadata.
`vram_reserve_mb` is the knob that covers it; raise it if you run near the edge.

So, stated plainly: **the estimate can admit a configuration that then fails to
load.** It exists to prevent the catastrophic case — an abort that takes the host
process down — and to hand back an actionable recommendation. It is not a
guarantee. A load that fails after admission surfaces as a typed error, which is
the outcome #142 asked for.

### What consumers should expect

- **Full-offload tiers that never fit** now fail fast with
  `ENTROPIC_ERROR_TIER_MODEL_TOO_LARGE` instead of aborting the process. If you
  were relying on the abort... you were not.
- **Partial-offload tiers are unaffected** — never refused, by design.
- **CPU-only builds and machines with no GPU are unaffected** — no device, budget
  0, gate stays disabled, which is correct: there is no VRAM to exhaust.
- **A busy GPU can now refuse a load that used to succeed** on an idle card,
  because the budget is sampled from free VRAM at `initialize()`. That is the
  intended behaviour — it is the reported scenario — but it does mean the same
  config can be admitted or refused depending on what else holds the card.
  `ENTROPIC_VRAM_BUDGET_BYTES` overrides the device query if you need
  determinism.

## app_context accepts inline content (#141)

`app_context` could only ever name a file. A consumer holding the text in memory
had no supported way to deliver it, and for the reporter writing the file was not
an option: it is a provenance boundary on their side (what the app can rewrite at
runtime it can rewrite wrongly and silently), and on their Android target there is
no stable writable path across launches.

```yaml
app_context:
  content: |
    This family follows a Scandinavian, play-first approach...
```

Resolution order is **explicit opt-out > inline content > path**, so
`app_context: false` still wins over supplied text. Content never touches the
filesystem. Every prior spelling keeps its meaning — bare string is a path, `true`
is bundled, `false` disables, absent is opt-out — each pinned by a regression
test.

Not taken: a `entropic_set_app_context()` C setter, the reporter's other option.
That is public-ABI surface and a larger decision than a bug report should settle.

## Storage write failures are reported

`save_messages` ran an INSERT per row, discarded every result, and returned
`true` unconditionally. `create_conversation` returned a generated UUID even
when its INSERT never landed — so the caller went on to reference an id that did
not exist, and the real failure resurfaced later as something unrelated. Both
now report.

## Speculative decode failures were completely silent

`spec_error()` built the error result and returned it without logging. The
message reached only `GenerationResult::error_message`; nothing appeared in the
log. A consumer whose speculative configuration failed saw `finish=error`, empty
content, and no explanation anywhere — the same shape as the gh#138 gap.

Found the hard way, diagnosing a benchmark config that failed every turn with
`LOAD_FAILED` and produced not one line saying why. With the log line in place
the cause was immediate:

```
Speculative decode failed (ENTROPIC_ERROR_LOAD_FAILED):
  MTP head setup failed: .../mtp-gemma-4-E4B-it.gguf
```

`spec_error()` now logs at ERROR with the code name and the message.

## Diagnostics for gh#137 and gh#138

Neither issue is closed. Both were missing the observability needed to diagnose
them at all.

**gh#138** — there was no signal anywhere that a tool-call grammar was or was
not in force; the orchestration line reports `params.grammar` (the *request*
grammar) and so reads `unconstrained` even when a tool grammar is fully active.
Now logged: what the render derived, whether it reached the sampler, and — at
ERROR — a tier that set `require_tool_call` whose render produced no grammar.
Measured on Gemma-4 QAT, the mechanism the flag actually controls:

```
require_tool_call: true    grammar 4782 bytes, lazy=false   enforced from token 1
require_tool_call: false   grammar 1272 bytes, lazy=true    never triggers
```

`tool_choice: AUTO` yields a *lazy* grammar armed by a trigger the model never
emits — inert by construction.

**gh#137** — a turn that produced tokens and delivered zero content was told to
"Raise max_tokens". That is only sound when the generation was truncated. The
reported case was `finish=stop`: the model ended the turn itself, and no budget
increase can help. The diagnosis is now `finish_reason`-aware and says plainly
when budget is *not* the problem.

The underlying gh#137 defect **did not reproduce** — three GPU runs across E2B
and E4B QAT, including the reporter's exact model, all emitted no reasoning
markers and delivered 100% of their content. The issue stays open with the
untested surface named: `speculative.mtp`, real tool staging, delegation.

## Also

- #139: partial CPU/GPU offload of a hybrid architecture can exceed
  `GGML_SCHED_MAX_SPLIT_INPUTS` and abort in `do_activate`. Filed; full offload
  is unaffected.
- #140: unenforced bash/git timeout.
- gh#131 closed — its three dependencies shipped in v2.10.0.
- **The benchmark gate was reporting failures for benchmarks that never ran.**
  `add_bench_test` registered the binary with no arguments, and the benchmark
  cases are Catch2 `[.]`-hidden — so a bare run collected nothing, exited 2, and
  ctest recorded all three as ~0.1s failures. They are now invoked with
  `"[benchmark]"`.
- **The agentic benchmark was loading the wrong model's vision tower.** It
  repointed its tier at a gemma-4 model but never cleared `mmproj_path`, which
  the default tier inherits from the global config as `mmproj: primary_mmproj`
  — the Qwen3.6-35B projector. 857.6 MiB of an unrelated model's weights, in a
  text-only throughput benchmark, and the exact allocation that produced the
  #142 abort. Cleared, and the matrix now sizes its context per config from the
  same estimator the admission gate uses, reporting configs that do not fit
  instead of attempting them. A run that fits nothing fails rather than
  reporting an empty table.
- Model-suite skip audit: all 78 `SKIP()` sites in `tests/model` are either GGUF
  guards whose files are present, or sit in single-case binaries where a skip
  exits 4 and ctest records a failure. The suite's 74/74 is a real green, not an
  absence of running tests.

---

# entropic v2.10.4

Patch release — **tools-staged tiers were decoding unconstrained (gh#134), and
`type_error.316` is closed at its source rather than per-call-site (gh#136).**

## gh#134 — the tool-call grammar was built and discarded

llama.cpp derives a GBNF from your staged tool schemas during prompt rendering.
entropic harvested the prompt, format, generation prompt and parser from that
render — and **threw the grammar away**. So every tier with tools staged has
been decoding completely unconstrained, and llama.cpp's own
`tool_choice: REQUIRED` mechanism would have had no effect even once exposed.

**New: `require_tool_call` per tier** (opt-in, off by default):

```yaml
models:
  researcher:
    require_tool_call: true
```

The turn then *cannot* end with prose — narrate-then-stop becomes
unrepresentable rather than corrected after the fact.

Measured on Gemma-4 E4B QAT, 10 turns, only the flag varying:

```
off: C C C C C [stop]p C C [stop]p C    2 prose-only turns
on : C C C C C C C C C C               10 tool calls, 0 prose-only
```

The off arm still stalls with a larger budget, so **raising `max_tokens` alone
does not fix this** — the grammar does.

### Budget matters, and the engine now says so

Under `REQUIRED` the grammar allows unbounded text *before* the mandated call,
so the call still has to fit inside `max_tokens`. Too tight a budget produces a
turn that ends on `length` with no call. That case is now logged explicitly,
naming both levers — raise `max_tokens`, or disable `enable_thinking` on the
tier, since the thinking channel is what consumes the preamble.

## gh#136 — `type_error.316`, closed at the source

This crash has been fixed four times (gh#112/113, gh#114, gh#118, gh#132), each
time by sanitizing one more `.dump()` site. There are ~18 in the facade alone,
so the next occurrence always landed somewhere nobody had reached.

Model bytes now get sanitized **where they enter** — the three points where
generated output first becomes a string. Every downstream serialization is safe
by construction, and a newly added `.dump()` anywhere cannot bring it back.

If you were seeing `invalid UTF-8 byte at index N` kill an `entropic.ask`
response, that is closed.

## Standing invariants

Both bugs recurred because every previous fix was per-instance. This release
adds tests that pin the *property*: a fifth grammar source cannot be declared
without being wired to the sampler, sanitized output is always serializable
(with a control proving the test can fail), and the new per-turn backend state
is per-instance — verified, not assumed, for consumers running concurrent
handles.

## Upgrade notes

Nothing is required. `require_tool_call` is opt-in and off by default; tiers
that do not set it behave exactly as before.

# entropic v2.10.3

Patch release — **Gemma-4 reasoning (`<|channel>`) leaked into content and the
live token stream on every generate path (gh#108).**

## The bug

A toolless generate on a Gemma-4 tier returned raw
`<|channel>thought…<channel|>` in `result.content` **and** streamed it live to
consumers. Plain decode, streaming, MTP, batch — all four. Anything rendering a
stream showed the model's private reasoning; conversation history kept it.

If you run a Gemma-4 tier and have seen thinking text in output, this is why.

## Root cause — not what it looked like

Every model family has an adapter that strips its own reasoning markers.
**Gemma-4 had none.** `adapter_registry` deliberately omitted it because its
tool calls are parsed by llama.cpp's `PEG_GEMMA4` grammar — so
`adapter: gemma4` silently resolved to `GenericAdapter`, which strips `<think>`
(a marker Gemma-4 never emits) and left `<|channel>` untouched.

The one `<|channel>` handler lived inside `parse_response`, reachable only when
`common_chat_parse_reliable()` is true — which requires **both** a tooled render
**and** `PEG_GEMMA4`. A toolless call fell through to the adapter branch with no
channel handling at all. Reasoning stripping had been attached to a gate that
exists to answer an unrelated question: *is this captured format
multi-parameter safe?*

MTP, streaming, and grammar were red herrings — a plain-decode non-streaming
control leaks identically. The v2.9.1 MTP-streaming guard was gating one feature
over a defect belonging to a different layer, and never protected anyone.

## What changed

- **`Gemma4Adapter`** — the missing fallback, owning the `<|channel>` pair.
  `PEG_GEMMA4` remains primary whenever a parser arena exists.
- **`ChatAdapter::thinking_markers()`** — each family declares its delimiters
  once, consumed by both the buffered strip and the live stream filter so they
  cannot drift apart again.
- **One shared parse rule** (`response_parse.h`) — template first, adapter
  second — replacing a duplicated branch in the orchestrator and the interface
  factory. Content cleanup always runs the adapter strip (idempotent); tool-call
  extraction falls back to the adapter when the template result fails validation
  against the staged tool schema, which catches `common_chat`'s *silent*
  first-parameter-only extraction.
- **`StreamThinkFilter`** takes adapter-resolved markers. This is load-bearing:
  the agent-loop streaming path builds content from its own token accumulator
  and discards the parsed result, so the filter is the only defense there.
- **Constitutional validator** resolves markers per tier, so critique calls on a
  Gemma-4 tier no longer hand raw reasoning to the critique model as claims.

## Also

Three dead methods removed from `adapter_base.h` (`extract_thinking`,
`parse_bare_json_tool_calls`, `format_system_prompt`) — zero callers in `src/`,
kept alive only by their own tests. `do_unload` now invalidates the sticky
parse snapshot, which nothing previously cleared.

## Known gaps

Model tests for qwen36 and gemma4-a4b remain skipped on the release box for
lack of disk for those GGUFs; both families retain full CPU unit coverage.

# entropic v2.10.2

Patch release — **the bridge no longer answers `"(no response)"` when the
answer is already in the conversation (gh#130).**

## The bug

A turn that ends without `entropic.complete` leaves a trailing **empty**
assistant message — e.g. anti-spiral rejects the lead's tool call and the next
generation returns `finish=stop`, 0 tool calls, 0 chars.
`extract_final_text` scanned backwards, found that empty message first, and
returned it, never looking further back. Operators got the literal string
`"(no response)"` while the real answer sat one or two messages earlier.
Reported at ~4 of 16 runs in a live consumer acceptance matrix.

The worst case involved a completed sub-tier delegation. `fold_delegation_summary`
(gh#119, v2.9.17) already folds a child's summary into the lead's empty
assistant turn precisely so this function can find it — but a *later* terminal
empty assistant turn shadowed it, so an answer the engine had correctly
produced was thrown away at the last step.

**Fix:** skip empty assistant messages and keep scanning backwards.

## Better diagnostics on a genuinely empty turn

`"(no response)"` could not distinguish an engine failure from a model that
simply stalled. It now says which:

- `(no response: the turn produced no assistant message at all)`
- `(no response: the turn ended with every assistant message empty — the tier
  most likely stopped without calling entropic.complete)`
- `(no response: the engine returned no readable conversation)`

`"(no response"` remains the leading substring, so prefix/substring matching on
the old sentinel still fires. **Exact-equality matching on `"(no response)"`
will not** — adjust if you match that string exactly.

## Async ask had it worse

`derive_async_final_state` had no fallback at all: a stalled async
`entropic.ask` returned `status: "done"` with empty text — less diagnosable
than the sync path's sentinel. All three ask paths (plain, streaming, async)
now share one selection rule.

## A note on scope

The report suggested also falling back to "the most recent delegation/pipeline
result text." That is **not** implemented, deliberately. Tool and delegation
results are injected as `role: "user"`, and the serialized conversation carries
only `{role, content}` — so at that layer a delegation summary is
indistinguishable from the operator's own prompt, and using it would echo the
user's question back as the answer. Delegation summaries reach the extractor
through the assistant-turn fold instead. A regression test pins that a user
message is never returned as the answer.

# entropic v2.10.1

Patch release — **the MCP server plugin loader `i_mcp_server.h` has documented
since v1.8.5 now actually exists (gh#133).**

## The gap

`include/entropic/interfaces/i_mcp_server.h` stated "ServerManager discovers
plugins via dlopen and calls these functions through the opaque handle." No
such loader existed. A consumer who implemented the documented nine-entry-point
contract produced a `.so` that nothing in the engine could load — reachable
only by disassembling the shipped binary, since the headers said the opposite.

Reported by the sassafras-class consumer, who had a conformant implementation
written and tested against the contract before establishing it was unloadable.

## Loading a plugin

```yaml
mcp:
  plugins:
    - /path/to/libmy_mcp_server.so
    - ~/plugins/libother_server.so
```

Each entry is dlopened at startup, version-checked against
`ENTROPIC_MCP_PLUGIN_API_VERSION`, and registered under the name its
`entropic_mcp_server_name()` reports. Its tools are then addressable as
`<name>.<tool>` exactly like a built-in server's, including argument
validation against the plugin's declared `inputSchema`.

Failures are loud, never silent: a `.so` that will not open, is missing an
entry point, reports a different API version, or collides with an existing
server name is rejected with `ENTROPIC_ERROR_PLUGIN_LOAD_FAILED` /
`ENTROPIC_ERROR_PLUGIN_VERSION_MISMATCH`. Every configured path is attempted
so one broken entry does not hide the diagnosis of the rest.

## Header corrections

- **`ENTROPIC_EXPORT` on all nine entry points.** Previously absent, so a
  plugin defining them the obvious way — plain `extern "C"`, inheriting
  visibility from the header — exported *nothing* under `-fvisibility=hidden`,
  the way most plugin projects build. Verified on GCC 11.4: 0 symbols exported
  before, all 9 after. Visibility only; no signature or ABI change, so no
  plugin-API version bump, and plugins that exported by other means still load
  unchanged.
- **`entropic_plugin_api_version()` and `entropic_create_server()` are now real
  declarations** rather than prose in a comment block.
- **Threading contract documented**: the engine serialises calls into a given
  server instance, so a plugin needs no internal locking for its own state.
  `PluginServer` takes its own mutex, making that guarantee hold by
  construction rather than by assumption about callers.
- **`inputSchema` camelCase** stated explicitly.

## Notes for plugin authors

Plugins are loaded `RTLD_LOCAL`, so two plugins exporting the same entry-point
names cannot collide. Strings returned by `list_tools`/`execute` are freed
through *that plugin's* `entropic_free`, not the engine's allocator. A plugin
returning a malformed tool list or response is contained to itself — it does
not throw through the agent loop or empty the tool list for other servers.

# entropic v2.10.0

Minor release — **MTP grammar + streaming support, tool-call robustness, and
filesystem/pipeline polish.**

## Highlights

- **MTP grammar (gh#108)**: tiers with `speculative.mtp: true` and a static
  GBNF grammar now work correctly. `to_common_sampling` propagates
  `params.grammar` to the MTP sampler chain; the loader rejection and the
  orchestrator routing gate are removed.
- **MTP streaming (gh#108)**: `speculative.mtp: true` is now compatible with
  streaming calls. `generate_streaming` wraps `on_token` with `StreamThinkFilter`
  for incremental thinking-channel stripping, and calls `apply_adapter_parse` on
  return — matching the non-streaming path.
- **MTP head guard (gh#107)**: using a Gemma-4 MTP head GGUF on the classical
  separate-draft path now fails loud with `INCOMPATIBLE_CONFIG` instead of
  crashing in `fattn.cu`. Message names `speculative.mtp: true` as the fix.
- **Lenient tool-call parse (gh#127)**: fenced JSON blocks containing only an
  arguments object (no `name` key) are now matched against registered tool
  schemas and synthesized into a `ToolCall` when exactly one schema matches.
- **Pipeline stage validation (gh#129)**: `PipelineTool` rejects unknown stage
  names at emission time with an `invalid_stage` error, instead of silently
  passing them to `DelegationManager` and failing per-stage.
- **Per-stage pipeline output (gh#125)**: pipeline context messages now include
  per-stage `{tier, task}` summaries in addition to the final result.
- **`read_file` guidance (gh#124)**: not-found errors now name `list_directory`
  as the corrective action.
- **`glob` path matching (gh#126)**: `**/*.cpp` and similar patterns now match
  root-level files and path-relative entries; `**` maps to `.*` (cross-directory)
  while bare `*` maps to `[^/]*` (single segment).
- **UTF-8 safety (gh#132)**: `CompleteTool::execute`, `serialize_batch_results`,
  and `entropic_validation_last_result` sanitize output before JSON serialization.

## Engine bug fixes

- gh#132: `type_error.316` on malformed model output in `CompleteTool::execute`
- gh#127: tool-call lost when model emits arguments-only fence (no `name` key)
- gh#129: silent per-stage failure on unknown tier names in `pipeline` tool
- gh#126: `glob("**/*.cpp")` returned nothing for root-level and path-relative files
- gh#124: `read_file` not-found error provided no recovery guidance
- gh#107: crash (`GGML_ABORT` in `fattn.cu`) when MTP head GGUF routed to classical draft path
- gh#108: MTP sampler did not enforce GBNF grammar constraints
- gh#108: MTP streaming emitted raw `<think>` tokens and skipped `apply_adapter_parse`

## New features

- gh#125: pipeline output includes per-stage tier + task summary
- gh#107: `looks_like_mtp_head(n_layer)` + `mtp_head_classical_path_error` in `mtp_envelope.h`

## Breaking changes

- Loader no longer rejects `speculative.mtp: true` + static grammar combination
  (was: validation error at parse time). Existing configs that relied on this
  gate as a safety net may now route to MTP with grammar applied.
- `mtp_unsupported_reason` always returns `""` — all three guards (temperature,
  grammar, streaming) are removed. Direct callers asserting non-empty for any
  condition should update their tests.

## Distribution

- CPU tarball: `entropic-2.10.0-linux-x86_64-cpu.tar.gz` (sha256 in companion file)
- CUDA tarball: `entropic-2.10.0-linux-x86_64-cuda.tar.gz` (sha256 in companion file)
- Python wrapper: `pip install entropic-engine==2.10.0` then `entropic install-engine`

# entropic v2.9.8

Patch — **completes the gh#111 UTF-8 fix that v2.9.7 left half-done.**
`entropic_run` still threw `nlohmann::json::type_error 316` mid-turn in a
lead→delegate turn under MTP, at the exact site named (but not patched) in the
v2.9.7 notes: `fire_delegate_complete_hook`'s `j.dump()` on a raw child summary.

## Why v2.9.7 missed it

v2.9.7 sanitized the hook-plugin *return* boundaries (`fire_post_generate_hook`,
`fire_complete_hook`, `fire_post_tool_hook`) — but only on the branch where a
plugin **revises** content (`out != nullptr`). On the headless path (no
content-revising `POST_GENERATE` hook) that branch never runs, so the raw
summary sailed straight into the dump.

Root cause: the summary reaches `fire_delegate_complete_hook` via the child's
**last-assistant-content fallback** in `extract_summary`, not the tool-arg path.
That content comes from `AgentEngine::parse_tool_calls`, whose backend callback
re-derives `*cleaned` / `*tool_calls_json` from the model's **raw** generation —
a channel entirely separate from the content sanitize at
`response_generator.cpp:470`. A split multi-byte UTF-8 codepoint (routine under
MTP speculative decode, when a character splits across the draft/target token
boundary) therefore survives into the message and, downstream, into the
delegate-complete hook's `j.dump()`.

## The fix (one boundary, not scattered sinks)

Sanitize **both** outputs of the tool-call parse channel at the single seam
where they cross into engine-owned state — `AgentEngine::parse_tool_calls`
(`src/core/engine.cpp`):

```cpp
std::string cleaned_str = mcp::sanitize_utf8(cleaned ? cleaned : raw_content);
std::string tc_str      = mcp::sanitize_utf8(tc_json ? tc_json : "[]");
```

This is the tool-call-channel sibling of the existing content sanitize. It
closes every downstream `json::dump()` at once: the assistant message /
delegation-summary fallback (`cleaned_str`) and the tool-call args
(`tc_str` → `CompleteTool` / directive JSON). Documented in the boundary-policy
table in `include/entropic/mcp/utf8_sanitize.h`.

Secondary benefit: `tc_str` sanitize also stops MTP from **silently dropping** a
tool call — a raw arg previously failed `nlohmann::json::parse` and the model's
directive (e.g. `entropic.complete`) was discarded.

## Tests (red-first)

Added to `tests/unit/core/engine_test.cpp`, each proven to FAIL on the
unmodified v2.9.7 code and PASS with the fix:
- **Delegation reproduction** — drives a real lead→child delegation whose child
  produces raw content; without the fix this throws
  `type_error.316 ... byte at index 9: 0x28` out of `fire_delegate_complete_hook`
  (the exact reported crash).
- **Content channel** — a backend parse returning raw cleaned content is
  sanitized before it becomes a message.
- **Tool-call survival** — a raw-arg tool call is preserved and dispatched, not
  silently dropped.

Also fixes `tasks.py`'s model-test runner to honor each test's CMake `TIMEOUT`
(carried from the develop branch; was a source of false model-test failures).

No `interfaces/i_*.h` touched.

---

# entropic v2.9.7

Patch — **UTF-8 sanitize gap at the hook-plugin return boundary** (gh#3
recurrence, gh#111). `entropic_run()` could throw `nlohmann::json::type_error
316` mid-agentic-turn in a lead→researcher delegation, immediately after
generation completed.

## The bug

The v2.1.1 fix for gh#3 established a boundary-of-ownership UTF-8 sanitize
policy covering four boundaries: MCP tool-result inbound, llama.cpp stream
inbound, audit-log inbound, and C-API outbound. It missed a class of
boundary: **a hook plugin's returned content crossing back into the
engine.** Three call sites accepted a hook's output verbatim, with no
sanitize call before the bytes could re-enter engine state and later reach
an unguarded `nlohmann::json::dump()`:

- `fire_post_generate_hook` (`src/core/engine.cpp`) — POST_GENERATE hook
  revision. In a delegation, unsanitized content here became the child
  loop's summary, which `fire_delegate_complete_hook` dumps directly.
- `fire_complete_hook` (`src/core/engine.cpp`) — ON_COMPLETE hook feedback,
  injected into a `Message`.
- `ToolExecutor::fire_post_tool_hook` (`src/mcp/tool_executor.cpp`) —
  POST_TOOL_CALL hook transform, applied to the tool-result `Message`.

v2.9.6/gh#110 made MTP reachable from the agent loop's *batch* dispatch path
for the first time — exactly the path (`generate_batch` →
`fire_post_generate_hook` → delegation summary) that exercises the first
gap, which is why the recurrence surfaced now rather than earlier.

## The fix

- All three call sites now sanitize a hook's returned bytes via
  `mcp::sanitize_utf8` before they re-enter engine state, matching the
  treatment already given to MCP tool results.
- Documented the hook-plugin boundary in
  `include/entropic/mcp/utf8_sanitize.h`'s policy table; corrected prior text
  that incorrectly listed hook contexts as "interior/trusted."
- Added regression coverage in `tests/unit/core/engine_test.cpp` and
  `tests/unit/mcp/tool_executor_test.cpp` exercising all three hook points
  with malformed UTF-8, asserting the sanitized content JSON-dumps without
  throwing.

## Deferred

`src/storage/backend.cpp`'s SQLite message-load path reads `content` off the
column with no sanitize before a later `.dump()` — same class of gap as the
(already-fixed) audit-replay path, for the SQLite backend. Not the confirmed
root cause of this crash; fixing it cleanly needs `entropic-storage` to gain
access to the sanitizer (currently only linked into `entropic-core`). Tracked
separately, not blocking this release.

---

# entropic v2.9.6

Patch — **MTP/speculative decoding is now reachable through the agent loop**
(gh#110). v2.9.0–v2.9.4 proved MTP correct and fast when the orchestrator is
called directly, but every agent-loop turn (`entropic_run` and friends) with
`speculative.mtp` enabled failed loud — the kernel never ran.

## The bug (two independent gates)

1. `build_loop_config()` hardcoded `LoopConfig::stream_output = true`, so the
   agent loop always streamed. The streaming path unconditionally binds a
   non-empty `on_token` callback, and `LlamaCppBackend::mtp_guard` derives its
   "is this a streaming call" check as `static_cast<bool>(on_token)` — a bound
   callback is indistinguishable from "this is streaming," so every agent-loop
   MTP call tripped `mtp_unsupported_reason`'s streaming rejection and
   returned `ENTROPIC_ERROR_SPECULATIVE_INCOMPATIBLE_CONFIG`, every time.
2. Even with streaming disabled, the batch path's cancel-aware bridge
   (`inference_.generate_cancellable`, always wired in production) calls an
   orchestrator overload that deliberately bypasses `run_generate_dispatch` —
   batch-with-cancel only ever ran plain decode, never speculative.

Existing MTP tests never caught this because they call
`orchestrator->generate()` directly — shaped like the agent loop's traffic,
but never actually routed through `AgentEngine`/`ResponseGenerator`/the
facade.

## The fix

- New `generation.stream_output` config key (default `true`, no behavior
  change for existing consumers) threads through `build_loop_config()`,
  making batch mode reachable from config.
- `dispatch_batch_generate` now prefers the dispatching (non-cancellable)
  `generate` entry point over the cancel-aware one whenever speculative
  decoding is enabled, so the batch path actually reaches
  `run_generate_dispatch` → MTP. v1 tradeoff, documented not hidden: a
  speculative batch turn is not cancellable mid-decode.

To use MTP from the agent loop: set `generation.stream_output: false` +
`inference.speculative.{enabled,mtp}: true`.

## Tests

- `test_gh110_mtp_agent_loop.cpp` — drives the real `entropic_create` →
  `entropic_configure_dir` → `entropic_run` path (not a direct orchestrator
  call) and asserts on the backend's own `"Speculative: generated=..."` log
  line, the only MTP-engagement signal that crosses the C-ABI boundary.
  Verified on real hardware (RTX PRO 4000 Blackwell, gemma-4-E2B-it-Q8_0 +
  MTP head): the kernel engaged across multiple turns of the same
  conversation (`accept_rate` 0.08–0.14).

No `interfaces/i_*.h` touched.

# entropic v2.9.5

Patch — **turn/run entry points now log to `session.log` with the console
sink disabled** (gh#109). `entropic_run`, `entropic_run_as`,
`entropic_run_batch`, `entropic_run_streaming`, `entropic_run_messages`, and
`entropic_run_messages_streaming` never entered a `HandleLogScope`, so the
thread-local handle id stayed unset for the whole turn and
`HandleAwareSink` silently dropped every log line emitted during
generation. Consumers running with `console_logging: false` (e.g. a TUI
that keeps stderr clean for its own paint) got zero turn diagnostics —
`session.log` stopped at "configure complete" and never logged another
line, even on failure.

These six entry points intentionally skip the full `HandleApiLock` so a
long-running turn doesn't block `entropic_interrupt()` called from another
thread — but dropping the lock also dropped the log scope bundled inside
it. Fix enters a bare `HandleLogScope` (no `api_mutex`) at the top of each
instead; `run_turn`/`run_streaming` execute synchronously on the calling
thread with no internal logging worker threads, so a single scope per
entry point is sufficient — no change to the interrupt/cancel contract.

Adds a regression test (`facade_integration_test.cpp`) that configures a
handle via `entropic_configure_dir` with `console_logging: false`, runs a
turn, and asserts `session.log` grows with `[core.*]`-style content —
locking in that every run entry point holds a log scope.
