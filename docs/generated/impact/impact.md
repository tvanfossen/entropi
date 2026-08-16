## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | teardown_mtp_draft |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | run_sampling_loop |
| REQ-INFER-006 | Sampler chain construction has a fixed order and default-preserving gating | to_common_sampling |
| REQ-INFER-008 | Every declared grammar source reaches the sampler and exactly one wins | apply_grammar_source, to_common_sampling |
| REQ-INFER-009 | Tool staging drives a native render whose parse context is captured | render_common_chat, render_prompt, set_active_tools, render_with_tools |
| REQ-INFER-013 | Sequential tool-call mode hard-stops at the first closed call | tool_call_close_marker |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | effective_n_draft, mtp_guard |

**Total: 7 requirement(s) affected, 12 function(s) changed**
