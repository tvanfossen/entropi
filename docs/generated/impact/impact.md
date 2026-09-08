## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | do_unload |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | run_sampling_loop, generate_mtp |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | mtp_guard, generate_mtp |
| REQ-INFER-025 | Multimodal input is bounded, tier-gated and degrades gracefully | generate_multimodal |

**Total: 4 requirement(s) affected, 6 function(s) changed**
