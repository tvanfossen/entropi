// SPDX-License-Identifier: Apache-2.0
/**
 * @file transport.h
 * @brief Abstract transport for external MCP server communication.
 *
 * Internal to librentropic-mcp.so. Not exposed across .so boundaries.
 * Implementations handle the wire protocol; ExternalMCPClient handles
 * the MCP protocol logic on top.
 *
 * @version 1.8.7
 */

#pragma once

#include <cstdint>
#include <string>

namespace entropic {

/**
 * @brief Abstract transport for external MCP communication.
 *
 * Internal to librentropic-mcp.so. Not exposed across .so boundaries.
 * Two implementations: StdioTransport (fork/exec + pipes) and
 * SSETransport (cpp-httplib SSE stream + HTTP POST).
 *
 * @version 1.8.7
 */
class Transport {
public:
    virtual ~Transport() = default;

    /**
     * @brief Open the transport connection.
     * @return true on success.
     * @version 1.8.7
     */
    virtual bool open() = 0;

    /**
     * @brief Close the transport connection.
     * @version 1.8.7
     */
    virtual void close() = 0;

    /**
     * @brief Send a JSON-RPC request and wait for response.
     * @param request_json JSON-RPC request string.
     * @param timeout_ms Timeout in milliseconds (0 = default 30s).
     * @return JSON-RPC response string, or empty on error/timeout.
     * @version 1.8.7
     */
    virtual std::string send_request(
        const std::string& request_json,
        uint32_t timeout_ms = 0) = 0;

    /**
     * @brief Check if transport is connected.
     * @return true if connected and ready.
     * @version 1.8.7
     */
    virtual bool is_connected() const = 0;

    /**
     * @brief Abort any in-flight send_request() call ASAP.
     *
     * Called from another thread (typically the engine's interrupt
     * path) so tool dispatches to external MCP servers do not run to
     * completion after Ctrl+C. Default is a no-op for transports with
     * no blocking calls. Must wake pending reads within ~100ms.
     * (P1-10, 2.0.6-rc16)
     *
     * @utility
     * @version 2.0.6-rc16
     */
    virtual void interrupt() {}

    /**
     * @brief Release an interrupt so the transport can be used again.
     *
     * The counterpart to interrupt(). gh#150: without one, a transport
     * that latches on interrupt() stays disabled for the lifetime of the
     * process, because AgentEngine::reset_interrupt() clears only the
     * engine's own flag and never reaches here.
     *
     * Default is a no-op, matching interrupt(): a transport that does not
     * latch has nothing to release.
     *
     * @req REQ-MCP-025
     * @version 2.12.1
     */
    virtual void clear_interrupt() {}

    /**
     * @brief Whether an interrupt is currently suppressing this transport.
     *
     * gh#150: a suppressed transport returns an empty response, which a
     * caller cannot distinguish from a legitimate empty result — the
     * reporting consumer saw `status=ok` on every call while every call
     * was in fact being dropped. This makes the state answerable.
     *
     * @return true when calls are being short-circuited by an interrupt.
     * @req REQ-MCP-025
     * @version 2.12.1
     */
    virtual bool is_interrupted() const { return false; }
};

} // namespace entropic
