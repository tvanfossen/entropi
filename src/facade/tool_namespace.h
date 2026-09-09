// SPDX-License-Identifier: Apache-2.0
/**
 * @file tool_namespace.h
 * @brief gh#145: the bridge's tool namespace, as a pure pair of functions.
 *
 * A consumer app hosting this engine presents its OWN tools. The engine is a
 * substrate, not the product — `librentropic.so` is deliberately consumable by
 * more than one app, so the tools a given host exposes are properly that
 * host's identity, not entropic's.
 *
 * Extracted to a private facade header for the same reason `final_text.h` was
 * (gh#130, v2.10.2): `ExternalBridge::dispatch` is private and the tools/list
 * payload is only reachable over a real unix socket, so the naming rule would
 * otherwise be testable only through an integration path. The rule is pure —
 * two strings in, one string out — and belongs where a CPU unit test can reach
 * it.
 *
 * `qualify` and `strip_tool_prefix` are inverses. That property is asserted
 * directly in tool_namespace_test.cpp, because the defect this replaces was
 * exactly a build-name and a match-name drifting apart: before v2.12.0 the
 * five tool names were written out twice, ~470 lines apart in
 * external_bridge.cpp, with nothing tying them together.
 *
 * @version 2.12.0
 */

#pragma once

#include <string>

namespace entropic {
namespace facade {

/**
 * @brief Qualify a bare tool suffix with the configured namespace.
 *
 * @param prefix Configured `mcp.external.tool_prefix`. Empty means "no
 *               namespace" and yields the bare suffix — a deliberate escape
 *               hatch for a host that wants unqualified tool names.
 * @param suffix Bare tool suffix ("ask", "status", ...).
 * @return "<prefix>.<suffix>", or `suffix` when `prefix` is empty.
 * @req REQ-BRIDGE-001
 * @version 2.12.0
 */
inline std::string qualify_tool_name(const std::string& prefix,
                                     const std::string& suffix) {
    if (prefix.empty()) {
        return suffix;
    }
    return prefix + "." + suffix;
}

/**
 * @brief Recover the bare suffix from a wire name. Inverse of qualify.
 *
 * Matches the WHOLE configured prefix rather than stripping up to the first
 * dot, and both halves of that matter:
 *   - a host configured as "acme.docs" must still dispatch its own tools;
 *   - a caller sending "entropic.ask" to a bridge configured as "sumac" must
 *     NOT silently resolve. That is a genuinely unknown tool there, and it is
 *     returned unchanged so the unknown-tool branch can report the name the
 *     caller actually sent rather than a mangled fragment.
 *
 * @param prefix Configured `mcp.external.tool_prefix`.
 * @param name Fully-qualified tool name as received on the wire.
 * @return The bare suffix, or `name` unchanged when the prefix does not match.
 * @req REQ-BRIDGE-001
 * @version 2.12.0
 */
inline std::string strip_tool_prefix(const std::string& prefix,
                                     const std::string& name) {
    if (prefix.empty()) {
        return name;
    }
    const std::string qualified = prefix + ".";
    if (name.rfind(qualified, 0) != 0) {
        return name;
    }
    return name.substr(qualified.size());
}

} // namespace facade
} // namespace entropic
