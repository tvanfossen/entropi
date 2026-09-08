## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-ABI-002 | C++ exceptions never cross any .so boundary | entropic_run, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming |
| REQ-API-002 | Handle lifecycle — create, configure, destroy, NULL-safe teardown | entropic_run |
| REQ-API-005 | Uniform precondition guard on every exported entry point | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming, entropic_context_get |
| REQ-API-008 | Single cross-boundary allocator pair and explicit ownership transfer | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_messages, entropic_context_get |
| REQ-API-009 | Run entry-point family, result contract, and cross-thread interruptibility | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming, entropic_run_session, entropic_run_session_as, entropic_run_session_streaming |
| REQ-API-010 | Observer and callback slots survive configure and fire uniformly | entropic_run, entropic_run_as, entropic_run_streaming |
| REQ-API-012 | Final-answer selection from a serialized conversation | final_answer_from_context, handle_ask_plain, handle_ask |
| REQ-BRIDGE-001 | External bridge exposes the engine over a peer-authenticated unix socket | final_answer_from_context, handle_ask_plain, handle_ask, handle_status, handle_clear, handle_count, handle_ask_status, dispatch_ask, dispatch_tool, accept_loop, serve_client, dispatch, attach_phase_observer, detach_phase_observer, run_async_ask |
| REQ-IDEN-001 | Tier resolution contract and per-tier loop overrides | entropic_run_session_as |
| REQ-INFER-018 | Same-prefix batch generation is gated, per-request-constrained and seq-safe | entropic_run_batch |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | entropic_run_messages, entropic_run_messages_streaming |
| REQ-LOOP-001 | Agent state machine with observable, dual-channel transitions | entropic_run_session, entropic_session_context_get, entropic_session_context_count, entropic_session_context_clear, entropic_session_drop, entropic_session_list |
| REQ-SAFE-001 | Untrusted bytes are sanitized at ingress, never at egress | entropic_context_get |

**Total: 13 requirement(s) affected, 59 function(s) changed**
