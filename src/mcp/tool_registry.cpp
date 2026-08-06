// SPDX-License-Identifier: Apache-2.0
/**
 * @file tool_registry.cpp
 * @brief ToolRegistry implementation.
 * @version 1.8.5
 */

#include <entropic/mcp/tool_registry.h>
#include <entropic/mcp/server_base.h>
#include <entropic/types/logging.h>

#include <nlohmann/json.hpp>

static auto logger = entropic::log::get("mcp.tool_registry");

namespace entropic {

/**
 * @brief Register a tool instance.
 *
 * Defensive by design: registration runs inside server constructors,
 * where a failed load_tool_definition or a copy-paste name collision
 * would otherwise surface as a crash at the model's first tool call.
 * A nullptr is a logged no-op (never a stored null entry) and a
 * duplicate name logs a warning before replacing the entry.
 *
 * @param tool Non-owning pointer to a ToolBase; the server retains
 *             ownership. NULL is tolerated and logged.
 * @req REQ-MCP-003
 * @version 1.8.5
 */
void ToolRegistry::register_tool(ToolBase* tool) {
    if (tool == nullptr) {
        logger->warn("Attempted to register null tool");
        return;
    }
    const auto& name = tool->name();
    if (tools_.count(name) > 0) {
        logger->warn("Tool '{}' already registered — replacing", name);
    }
    tools_[name] = tool;
    logger->info("Registered tool: {}", name);
}

/**
 * @brief Check if a tool is registered by name.
 * @param name Tool name (without server prefix).
 * @return true when a tool of that name is registered, false for any
 *         unknown name.
 * @req REQ-MCP-003
 * @version 1.8.5
 */
bool ToolRegistry::has_tool(const std::string& name) const {
    return tools_.count(name) > 0;
}

/**
 * @brief Get all registered tool definitions as JSON array.
 *
 * Emits `inputSchema` (camelCase), matching both the bundled
 * data/tools/&#42;/&#42;.json descriptors and the plugin C ABI, so the
 * schema lookup path is uniform across server kinds.
 *
 * @return JSON array string with one `{name, description, inputSchema}`
 *         object per registered tool; "[]" for an empty registry.
 * @req REQ-MCP-003
 * @req REQ-MCP-008
 * @version 1.8.5
 */
std::string ToolRegistry::get_tools_json() const {
    auto arr = nlohmann::json::array();
    for (const auto& [name, tool] : tools_) {
        nlohmann::json entry;
        entry["name"] = tool->definition().name;
        entry["description"] = tool->definition().description;
        entry["inputSchema"] = nlohmann::json::parse(
            tool->definition().input_schema);
        arr.push_back(std::move(entry));
    }
    return arr.dump();
}

/**
 * @brief Get all registered tool definitions.
 * @return Exactly one non-owning ToolDefinition pointer per registered
 *         tool; an empty vector for an empty registry. Pointers stay
 *         valid for as long as the owning server holds its tools.
 * @req REQ-MCP-003
 * @version 1.8.5
 */
std::vector<const ToolDefinition*> ToolRegistry::get_definitions() const {
    std::vector<const ToolDefinition*> defs;
    defs.reserve(tools_.size());
    for (const auto& [name, tool] : tools_) {
        defs.push_back(&tool->definition());
    }
    return defs;
}

/**
 * @brief Dispatch a tool call to the registered tool.
 *
 * An unregistered name is answered with an error ServerResponse — the
 * missing tool is never dereferenced.
 *
 * @param name Tool name (without server prefix).
 * @param args_json JSON arguments string.
 * @return The tool's own ServerResponse when the name resolves;
 *         otherwise a response whose `result` is
 *         "Error: Unknown tool '<name>'" and whose directives are empty.
 * @req REQ-MCP-003
 * @req REQ-MCP-002
 * @version 2.0.0
 */
ServerResponse ToolRegistry::dispatch(
    const std::string& name,
    const std::string& args_json) {
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        logger->warn("Unknown tool: {}", name);
        ServerResponse resp;
        resp.result = "Error: Unknown tool '" + name + "'";
        return resp;
    }
    logger->info("Dispatch: tool='{}'", name);
    return it->second->execute(args_json);
}

/**
 * @brief Get a registered tool by name.
 * @param name Tool name (without server prefix).
 * @return Non-owning tool pointer, or nullptr when the name is not
 *         registered — callers must null-check before use.
 * @req REQ-MCP-003
 * @version 1.8.5
 */
ToolBase* ToolRegistry::get_tool(const std::string& name) const {
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        return nullptr;
    }
    return it->second;
}

} // namespace entropic
