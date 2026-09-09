## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-ABI-001 | Pure C at every .so boundary — opaque handles and explicit ownership | entropic_context_count |
| REQ-ABI-002 | C++ exceptions never cross any .so boundary | entropic_run, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming |
| REQ-API-002 | Handle lifecycle — create, configure, destroy, NULL-safe teardown | entropic_run |
| REQ-API-005 | Uniform precondition guard on every exported entry point | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming, entropic_context_get, entropic_context_count |
| REQ-API-008 | Single cross-boundary allocator pair and explicit ownership transfer | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_messages, entropic_context_get |
| REQ-API-009 | Run entry-point family, result contract, and cross-thread interruptibility | try_begin_turn, end_turn, claim, entropic_run, entropic_run_as, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming, entropic_run_session, entropic_run_session_as, entropic_run_session_streaming |
| REQ-API-010 | Observer and callback slots survive configure and fire uniformly | entropic_run, entropic_run_as, entropic_run_streaming |
| REQ-API-012 | Final-answer selection from a serialized conversation | final_answer_from_context, handle_ask_plain, handle_ask |
| REQ-BRIDGE-001 | External bridge exposes the engine over a peer-authenticated unix socket | final_answer_from_context, handle_ask_plain, handle_ask, handle_status, handle_clear, handle_count, dispatch_ask, dispatch_tool, begin_turn_wait, end_turn_wait, dispatch, run_async_ask, qualify_tool_name, strip_tool_prefix |
| REQ-CFG-006 | Fail-loud validation — reject bad or inert configuration at load time | validate |
| REQ-COMPACT-002 | Fill-gated tool-result pruning and persistent context anchors | run |
| REQ-IDEN-001 | Tier resolution contract and per-tier loop overrides | run_turn_as, seed_system_prompt_for_tier, build_params_json, entropic_run_session_as |
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | do_unload |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | generate_mtp |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | generate_mtp |
| REQ-INFER-018 | Same-prefix batch generation is gated, per-request-constrained and seq-safe | entropic_run_batch |
| REQ-INFER-019 | Model pool dedup and VRAM residency policy govern which tier is resident | footprint_inputs_for, partial_gpu_layers_for, host_can_hold_warm_load, derive_pool_geometry, session_pool_conflict_reason, slot_for, acquire, resident, set_resident, invalidate, invalidate_all, estimate_vram_footprint, recommend_context_length |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | entropic_run_messages, entropic_run_messages_streaming, generate_multimodal |
| REQ-LOOP-001 | Agent state machine with observable, dual-channel transitions | set_active_session, messages_for, clear_conversation_for, drop_session, run, run_turn, run_turn, entropic_run_session, entropic_session_context_get, entropic_session_context_count, entropic_session_context_clear, entropic_session_drop, entropic_session_list |
| REQ-LOOP-002 | Bounded loop termination with synthetic completion on cap | run |
| REQ-LOOP-007 | Engine-authored transcript shaping keeps the system prompt bit-stable | prepare_prompts |
| REQ-MCP-001 | MCPServerBase holds the shared logic; concrete servers override only deltas | execute |
| REQ-MCP-002 | Tool results always cross boundaries as a ServerResponse JSON envelope | execute, dispatch |
| REQ-MCP-003 | ToolRegistry dispatch is defensive about unknown and null tools | dispatch |
| REQ-MCP-013 | Tool arguments are validated against the declared JSON schema before dispatch | normalize_args, serialize_args |
| REQ-MCP-015 | Duplicates, errors, and denials return corrective guidance, not bare failure | tool_call_key |
| REQ-MCP-024 | Engine-level entropic.* tools validate their arguments and emit typed directives | execute |
| REQ-SAFE-001 | Untrusted bytes are sanitized at ingress, never at egress | entropic_context_get |

**Total: 28 requirement(s) affected, 102 function(s) changed**
