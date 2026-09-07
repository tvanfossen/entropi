## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-ABI-001 | Pure C at every .so boundary — opaque handles and explicit ownership | entropic_context_clear |
| REQ-ABI-002 | C++ exceptions never cross any .so boundary | entropic_run, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming |
| REQ-API-002 | Handle lifecycle — create, configure, destroy, NULL-safe teardown | entropic_run |
| REQ-API-005 | Uniform precondition guard on every exported entry point | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming, entropic_set_queue_observer, entropic_context_clear, entropic_context_get |
| REQ-API-008 | Single cross-boundary allocator pair and explicit ownership transfer | entropic_run, entropic_run_as, entropic_run_batch, entropic_run_messages, entropic_context_get |
| REQ-API-009 | Run entry-point family, result contract, and cross-thread interruptibility | try_begin_turn, end_turn, claim, entropic_run, entropic_run_as, entropic_run_batch, entropic_run_streaming, entropic_run_messages, entropic_run_messages_streaming |
| REQ-API-010 | Observer and callback slots survive configure and fire uniformly | entropic_run, entropic_run_as, entropic_run_streaming, entropic_set_queue_observer |
| REQ-IDEN-001 | Tier resolution contract and per-tier loop overrides | run_turn_as |
| REQ-INFER-018 | Same-prefix batch generation is gated, per-request-constrained and seq-safe | entropic_run_batch |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | entropic_run_messages, entropic_run_messages_streaming |
| REQ-LOOP-001 | Agent state machine with observable, dual-channel transitions | run_turn, run_turn |
| REQ-SAFE-001 | Untrusted bytes are sanitized at ingress, never at egress | entropic_context_get |

**Total: 12 requirement(s) affected, 41 function(s) changed**
