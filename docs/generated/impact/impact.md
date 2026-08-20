## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | shutdown, ~ModelOrchestrator |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | generate_streaming |
| REQ-INFER-007 | Named grammars are registered, validated, and resolved by a fixed precedence | resolve_grammar_key |
| REQ-INFER-009 | Tool staging drives a native render whose parse context is captured | stage_active_tools, resolve_and_stage |
| REQ-INFER-010 | One template-first / adapter-second rule parses every raw emission | apply_adapter_parse, warn_if_content_vanished, warn_turn_diagnostics |
| REQ-INFER-011 | Reasoning-block markers are declared once per model family | generate_streaming |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | resolve_mtp_effective, try_mtp_route, mtp_head_guard_fires |
| REQ-INFER-019 | Model pool dedup and VRAM residency policy govern which tier is resident | log_fit_recommendation, resolve_vram_budget_bytes, footprint_inputs_for, estimate_footprint_bytes |
| REQ-INFER-020 | Tier routing, handoff rules and secondary model roles | build_routing_tables, activate_router, activate_draft, shutdown, route, classify_task, last_routing_result, loaded_models, can_handoff, clear_all_prompt_caches |
| REQ-INFER-021 | Per-tier sampler configuration applies only where the caller did not decide | apply_tier_sampler_overrides, apply_tier_sampler_defaults |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | has_vision_capable_tier, select_vision_tier |

**Total: 11 requirement(s) affected, 31 function(s) changed**
