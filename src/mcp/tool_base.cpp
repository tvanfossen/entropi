// SPDX-License-Identifier: Apache-2.0
/**
 * @file tool_base.cpp
 * @brief ToolBase implementation + load_tool_definition.
 * @version 1.8.5
 */

#include <entropic/mcp/tool_base.h>
#include <entropic/mcp/server_base.h>
#include <entropic/types/logging.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

static auto logger = entropic::log::get("mcp.tool_base");

namespace entropic {

/**
 * @brief Construct with a pre-built definition.
 * @param def Tool definition.
 * @dg_internal
 * @version 1.8.5
 */
ToolBase::ToolBase(ToolDefinition def)
    : definition_(std::move(def)) {}

/**
 * @brief Get the tool name.
 * @return Tool name from definition.
 * @dg_internal
 * @version 1.8.5
 */
const std::string& ToolBase::name() const {
    return definition_.name;
}

/**
 * @brief Get the full tool definition.
 * @return Tool definition reference.
 * @dg_internal
 * @version 1.8.5
 */
const ToolDefinition& ToolBase::definition() const {
    return definition_;
}

/**
 * @brief Default anchor_key — no anchoring.
 *
 * MCPServerBase injects a context_anchor directive only for a non-empty
 * key, so the default leaves the envelope's directives array empty.
 *
 * @param args_json Tool call arguments (unused in the default).
 * @return Empty string — no ContextAnchor is injected for this tool.
 * @req REQ-MCP-001
 * @version 1.8.5
 */
std::string ToolBase::anchor_key(
    const std::string& /*args_json*/) const {
    return "";
}

/**
 * @brief Default required access level — WRITE (safe default).
 *
 * A newly added tool is treated as privileged until someone deliberately
 * relaxes it; read-only tools override this to READ.
 *
 * @return MCPAccessLevel::WRITE for every tool that does not override.
 * @req REQ-MCP-011
 * @version 1.9.4
 */
MCPAccessLevel ToolBase::required_access_level() const {
    return MCPAccessLevel::WRITE;
}

/**
 * @brief Load a tool definition from a JSON file.
 *
 * Reads the bundled `data/tools/<server>/<tool>.json` descriptor, whose
 * `inputSchema` (camelCase) is the schema ToolExecutor later validates
 * arguments against.
 *
 * @param tool_name Tool name (e.g., "read_file").
 * @param server_prefix Server directory name (e.g., "filesystem"); an
 *                      empty prefix reads directly from data_dir.
 * @param data_dir Base directory for tool JSON files.
 * @return Parsed ToolDefinition carrying name, description and the
 *         serialised inputSchema.
 * @throws std::runtime_error when the descriptor file cannot be opened;
 *         nlohmann parse errors propagate for malformed JSON or a
 *         missing `name`/`inputSchema` key.
 * @req REQ-MCP-013
 * @version 1.8.5
 */
ToolDefinition load_tool_definition(
    const std::string& tool_name,
    const std::string& server_prefix,
    const std::string& data_dir) {

    std::string path = data_dir;
    if (!server_prefix.empty()) {
        path += "/" + server_prefix;
    }
    path += "/" + tool_name + ".json";

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(
            "Tool definition not found: " + path);
    }

    auto json = nlohmann::json::parse(file);

    ToolDefinition def;
    def.name = json.at("name").get<std::string>();
    def.description = json.value("description", "");
    def.input_schema = json.at("inputSchema").dump();

    logger->info("Loaded tool definition: {} from {}",
                 def.name, path);
    return def;
}

} // namespace entropic
