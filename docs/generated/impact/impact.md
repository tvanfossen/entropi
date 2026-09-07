## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-INFER-002 | Teardown releases every model, context and adapter handle | do_unload |
| REQ-INFER-005 | Every decode path honours cooperative cancellation within one token | run_sampling_loop |
| REQ-INFER-015 | MTP speculative decode fails loud and never silently falls back | mtp_guard |

**Total: 3 requirement(s) affected, 3 function(s) changed**
