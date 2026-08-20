## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | teardown_mtp_draft |
| REQ-INFER-008 | Every declared grammar source reaches the sampler and exactly one wins | apply_grammar_source |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | effective_n_draft |
| REQ-INFER-019 | Model pool dedup and VRAM residency policy govern which tier is resident | log_fit_recommendation, resolve_vram_budget_bytes, footprint_inputs_for, estimate_footprint_bytes |

**Total: 4 requirement(s) affected, 7 function(s) changed**
