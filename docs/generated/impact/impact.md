## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-COMPACT-002 | Fill-gated tool-result pruning and persistent context anchors | run |
| REQ-IDEN-001 | Tier resolution contract and per-tier loop overrides | run_turn_as, seed_system_prompt_for_tier, build_params_json |
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | do_unload |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | generate_mtp |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | mtp_guard, generate_mtp |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | generate_multimodal |
| REQ-LOOP-001 | Agent state machine with observable, dual-channel transitions | run, run_turn, run_turn |
| REQ-LOOP-002 | Bounded loop termination with synthetic completion on cap | run |
| REQ-LOOP-007 | Engine-authored transcript shaping keeps the system prompt bit-stable | prepare_prompts |

**Total: 9 requirement(s) affected, 14 function(s) changed**
