// SPDX-License-Identifier: Apache-2.0
/**
 * @file bash.cpp
 * @brief BashServer implementation — shell command execution with safety.
 * @version 1.8.5
 */

#include <entropic/mcp/servers/bash.h>
#include <entropic/mcp/tool_base.h>
#include <entropic/mcp/server_base.h>
#include <entropic/types/logging.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

static auto logger = entropic::log::get("mcp.bash");

namespace entropic {

// ── working_dir validation ──────────────────────────────────────

/**
 * @brief Reject working_dir values that would smuggle shell syntax.
 *
 * The bash tool's security model is operator approval (the engine's
 * tool-call gate), not pattern matching against the command. But
 * `working_dir` is concatenated into a shell `cd` clause, so any
 * shell metacharacter in cwd would escape the approval surface (the
 * operator approves `command`, not the constructed shell string).
 * This check rejects the obvious smuggling vectors and requires the
 * cwd be an existing directory.
 *
 * @param cwd Caller-supplied working directory string.
 * @return true only when cwd contains none of `;&|`$<>`, newline,
 *         quotes, globs or brackets AND names an existing directory;
 *         false otherwise, which rejects the call before any shell runs.
 * @req REQ-MCP-023
 * @version 2.1.1-rc1
 */
static bool is_safe_cwd(const std::string& cwd) {
    static constexpr const char* unsafe_chars = ";&|`$<>\n\r\\\"'*?(){}[]";
    if (cwd.find_first_of(unsafe_chars) != std::string::npos) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::is_directory(cwd, ec);
}

/**
 * @brief Run a shell command and capture output.
 * @param full_cmd Command string including cd prefix and the `2>&1`
 *                 redirect that merges stderr into stdout.
 * @return The combined stdout+stderr text paired with the process exit
 *         code; ("Failed to open process", -1) when the pipe could not
 *         be opened, so a nonexistent command yields an error result
 *         rather than a throw.
 * @req REQ-MCP-023
 * @version 1.8.5
 */
static std::pair<std::string, int> run_popen(
    const std::string& full_cmd) {

    FILE* pipe = popen(full_cmd.c_str(), "r");  // NOLINT
    if (pipe == nullptr) {
        return {"Failed to open process", -1};
    }

    std::string output;
    std::array<char, 4096> buf{};
    while (fgets(buf.data(), buf.size(), pipe) != nullptr) {
        output += buf.data();
    }

    int status = pclose(pipe);  // NOLINT
    int exit_code = WEXITSTATUS(status);
    return {output, exit_code};
}

// ── ExecuteTool ─────────────────────────────────────────────────

/**
 * @brief Tool for executing shell commands.
 * @dg_internal
 * @version 1.8.5
 */
class ExecuteTool : public ToolBase {
public:
    /**
     * @brief Construct from tool definition with server ref.
     * @param def Tool definition loaded from JSON.
     * @param server Owning BashServer reference.
     * @dg_internal
     * @version 1.8.5
     */
    ExecuteTool(ToolDefinition def, BashServer& server)
        : ToolBase(std::move(def)), server_(server) {}

    /**
     * @brief Execute a shell command.
     * @param args_json JSON with "command" and optional "working_dir".
     * @return ServerResponse with stdout/stderr or error.
     * @dg_internal
     * @version 1.8.5
     */
    ServerResponse execute(const std::string& args_json) override;

private:
    BashServer& server_;
};

/**
 * @brief Execute a shell command. Operator approval is the gate.
 *
 * Security model: the engine's tool-call approval flow gates whether
 * this method runs at all. There is no command-content allowlist or
 * denylist here; the operator approves a specific `command` value
 * shown in the prompt, and that's what runs. The only validation we
 * do is on `working_dir`, because that field is concatenated into a
 * shell `cd` clause and could smuggle commands the operator never
 * saw.
 *
 * @param args_json JSON with "command" and optional "working_dir"
 *                  (defaulting to the server's own working directory).
 * @return A ServerResponse with no directives whose result is either a
 *         JSON object carrying the process exit_code and combined
 *         stdout+stderr output, or the working_dir rejection error when
 *         the cwd failed validation — in which case no shell ran.
 * @req REQ-MCP-023
 * @version 2.1.1-rc1
 */
ServerResponse ExecuteTool::execute(const std::string& args_json) {
    auto args = nlohmann::json::parse(args_json);
    std::string command = args.at("command").get<std::string>();

    std::string cwd = args.value(
        "working_dir", server_.working_dir().string());

    logger->info("[bash.execute] cmd='{}' cwd='{}'", command, cwd);

    if (!is_safe_cwd(cwd)) {
        logger->warn("Rejected unsafe working_dir: '{}'", cwd);
        return {"Error: working_dir is not an existing directory or "
                "contains shell metacharacters", {}};
    }

    std::string full_cmd =
        "cd " + cwd + " && " + command + " 2>&1";

    auto [output, exit_code] = run_popen(full_cmd);
    logger->info("Bash: exit={}, stdout={} chars, cmd='{}'",
                 exit_code, output.size(), command);

    nlohmann::json result;
    result["exit_code"] = exit_code;
    result["output"] = output;
    return {result.dump(), {}};
}

// ── BashServer ──────────────────────────────────────────────────

/**
 * @brief Construct with working directory and data dir.
 *
 * A thin MCPServerBase subclass: it builds its one tool, registers it,
 * and overrides only get_permission_pattern and set_working_dir.
 *
 * @param working_dir Default working directory for commands.
 * @param data_dir Path to bundled data directory.
 * @param timeout Command timeout in seconds.
 * @req REQ-MCP-001
 * @req REQ-MCP-023
 * @version 1.8.5
 */
BashServer::BashServer(
    const std::filesystem::path& working_dir,
    const std::string& data_dir,
    int timeout)
    : MCPServerBase("bash")
    , working_dir_(working_dir)
    , timeout_(timeout) {

    auto def = load_tool_definition(
        "execute", "bash", data_dir + "/tools");

    execute_tool_ = std::make_unique<ExecuteTool>(
        std::move(def), *this);

    register_tool(execute_tool_.get());

    logger->info("BashServer initialized: cwd='{}' timeout={}s",
                 working_dir_.string(), timeout_);
}

/**
 * @brief Destructor.
 * @dg_internal
 * @version 1.8.5
 */
BashServer::~BashServer() = default;

/**
 * @brief Permission pattern: "execute:{base_cmd} *".
 *
 * Approval granularity matters here: keying on the base command means
 * an operator's "always allow" applies to a command family rather than
 * to every shell invocation.
 *
 * @param tool_name Tool name.
 * @param args_json Arguments JSON.
 * @return "<tool>:<base command> *" — for `python -m x`, the pattern
 *         names `python`. Unparseable arguments degrade to a base
 *         command of "unknown" (logged), never a wildcard.
 * @req REQ-MCP-023
 * @req REQ-MCP-009
 * @version 1.8.5
 */
std::string
BashServer::get_permission_pattern(const std::string& tool_name, const std::string& args_json) const {

    std::string base_cmd = "unknown";
    try {
        auto args = nlohmann::json::parse(args_json);
        std::string cmd = args.at("command").get<std::string>();
        auto space = cmd.find(' ');
        base_cmd = (space != std::string::npos)
            ? cmd.substr(0, space) : cmd;
    } catch (const std::exception& e) {
        logger->warn("Failed to parse command for permission: {}",
                     e.what());
    }
    return tool_name + ":" + base_cmd + " *";
}

/**
 * @brief Set the working directory.
 *
 * Re-targets the server so a sandbox swap does not require
 * reconstructing it.
 *
 * @param path New working directory.
 * @return true — the re-target always succeeds; the path itself is
 *         validated per call by is_safe_cwd.
 * @req REQ-MCP-023
 * @version 1.8.5
 */
bool BashServer::set_working_dir(const std::string& path) {
    working_dir_ = path;
    logger->info("Working directory set to: {}", path);
    return true;
}

/**
 * @brief Get the working directory.
 * @return The directory commands default to when a call omits
 *         `working_dir`.
 * @req REQ-MCP-023
 * @version 1.8.5
 */
const std::filesystem::path& BashServer::working_dir() const {
    return working_dir_;
}

/**
 * @brief Get command timeout.
 * @return The configured command timeout in seconds.
 * @req REQ-MCP-023
 * @version 1.8.5
 */
int BashServer::timeout() const {
    return timeout_;
}

} // namespace entropic
