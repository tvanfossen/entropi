## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-MCP-001 | MCPServerBase holds the shared logic; concrete servers override only deltas | execute |
| REQ-MCP-002 | Tool results always cross boundaries as a ServerResponse JSON envelope | execute, dispatch |
| REQ-MCP-003 | ToolRegistry dispatch is defensive about unknown and null tools | dispatch |
| REQ-MCP-013 | Tool arguments are validated against the declared JSON schema before dispatch | normalize_args, serialize_args |
| REQ-MCP-015 | Duplicates, errors, and denials return corrective guidance, not bare failure | tool_call_key |
| REQ-MCP-024 | Engine-level entropic.* tools validate their arguments and emit typed directives | execute |

**Total: 6 requirement(s) affected, 8 function(s) changed**
