## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | teardown_mtp_draft, ~LlamaCppBackend, do_unload, shutdown, ~ModelOrchestrator |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | generate_after_prefill, run_sampling_loop, do_generate_text_only, do_generate_streaming_text_only, generate_mtp, generate_streaming |
| REQ-INFER-006 | Sampler chain construction has a fixed order and default-preserving gating | to_common_sampling |
| REQ-INFER-007 | Named grammars are registered, validated, and resolved by a fixed precedence | resolve_grammar_key |
| REQ-INFER-008 | Every declared grammar source reaches the sampler and exactly one wins | apply_grammar_source, to_common_sampling |
| REQ-INFER-009 | Tool staging drives a native render whose parse context is captured | render_common_chat, render_prompt, set_active_tools, render_with_tools, parse_response, stage_active_tools |
| REQ-INFER-010 | One template-first / adapter-second rule parses every raw emission | diagnose_empty_content, explain_empty_content, apply_adapter_parse, warn_if_content_vanished, warn_turn_diagnostics |
| REQ-INFER-011 | Reasoning-block markers are declared once per model family | strip_think_blocks, generate_streaming |
| REQ-INFER-012 | Adapter registry resolves a family parser with a safe generic fallback | parse_tagged_tool_calls, recover_action_envelope_calls, apply_action_envelope_recovery, coerce_string_typed_args, try_recover_json |
| REQ-INFER-013 | Sequential tool-call mode hard-stops at the first closed call | tool_call_close_marker, effective_stop |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | mtp_guard, generate_mtp, resolve_mtp_effective, try_mtp_route, mtp_head_guard_fires |
| REQ-INFER-020 | Tier routing, handoff rules and secondary model roles | build_routing_tables, activate_router, activate_draft, shutdown, route, loaded_models |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | generate_multimodal |
| REQ-TYPE-004 | Sentinel count members bound enum validity and force exhaustive wiring | do_supports |

**Total: 14 requirement(s) affected, 48 function(s) changed**
