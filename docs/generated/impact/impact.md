## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-ABI-001 | Pure C at every .so boundary — opaque handles and explicit ownership | entropic_context_count |
| REQ-API-005 | Uniform precondition guard on every exported entry point | entropic_context_get, entropic_context_count |
| REQ-API-008 | Single cross-boundary allocator pair and explicit ownership transfer | entropic_context_get |
| REQ-SAFE-001 | Untrusted bytes are sanitized at ingress, never at egress | entropic_context_get |

**Total: 4 requirement(s) affected, 5 function(s) changed**
