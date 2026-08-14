## Change Impact Report

| REQ | Name | Functions Changed |
|-----|------|-------------------|
| REQ-MCP-001 | MCPServerBase holds the shared logic; concrete servers override only deltas | anchor_key, FilesystemServer, register_fs_tools |
| REQ-MCP-015 | Duplicates, errors, and denials return corrective guidance, not bare failure | skip_duplicate_check |
| REQ-MCP-021 | Filesystem tools are root-confined, read-before-write gated, and size-bounded | record_read, was_read, build_read_result, check_read_before_write, collect_glob_matches, collect_entries, do_str_replace, apply_edit, anchor_key, check_read_gates, execute, execute, compile_grep_or_error, execute, execute, compute_max_read_bytes, FilesystemServer, skip_duplicate_check, set_working_dir, root_dir, max_read_bytes, resolve_path |
| REQ-MCP-022 | Glob, grep, and read honour .gitignore plus .explorerignore semantics | classify_glob_entry, collect_glob_matches, grep_file, check_read_gates, grep_search, execute, FilesystemServer, set_working_dir, ignore |

**Total: 4 requirement(s) affected, 35 function(s) changed**
