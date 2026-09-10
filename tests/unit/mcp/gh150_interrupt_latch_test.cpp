// SPDX-License-Identifier: Apache-2.0
/**
 * @file gh150_interrupt_latch_test.cpp
 * @brief gh#150: an interrupt must not permanently disable a transport.
 *
 * StdioTransport::cancel_flag_ was a LATCH, not a flag. `store(true)`
 * appeared exactly once in the tree and nothing ever stored false, so the
 * first AgentEngine::interrupt() disabled every external MCP stdio server
 * for the lifetime of the process:
 *
 *   AgentEngine::interrupt()  -> external_interrupt_cb_
 *                             -> ServerManager::interrupt_external_tools
 *                             -> StdioTransport::interrupt()
 *                             -> cancel_flag_.store(true)   [never cleared]
 *
 *   AgentEngine::reset_interrupt() -> interrupt_flag_.store(false)
 *                                     ^ clears the ENGINE flag only; the
 *                                       transports are never told.
 *
 * Reported by a hosted consumer serving several clients over the external
 * bridge, where one client disconnect is enough to trip it and take every
 * external server away from all of the others until the host restarts.
 *
 * The interrupt() doc claimed the flag was "cleared implicitly by a
 * successful open()". It was not: open() early-returns when already
 * connected, and no path in the tree cleared the flag at all. That comment
 * sent the reporter looking in the wrong place, so it is asserted here
 * rather than merely corrected.
 *
 * @version 2.12.1
 */

#include <entropic/core/engine.h>
#include <entropic/mcp/external_client.h>
#include <entropic/mcp/tool_result_classify.h>
#include <entropic/mcp/transport_stdio.h>

#include "../core/mock_inference.h"

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace entropic;
using namespace entropic::test;

namespace {

/// @brief Stands in for a transport's latch, driven by the engine callbacks.
struct FakeTransportLatch {
    bool latched = false;
};

/// @brief Mirrors ServerManager::interrupt_external_tools.
void latch_cb(void* user_data) {
    static_cast<FakeTransportLatch*>(user_data)->latched = true;
}

/// @brief Mirrors the clearing counterpart this issue adds.
void unlatch_cb(void* user_data) {
    static_cast<FakeTransportLatch*>(user_data)->latched = false;
}

}  // namespace

SCENARIO("gh#150: clearing the engine interrupt also clears the transports",
         "[mcp][gh150][2.12.1]") {
    GIVEN("an engine wired to an external transport latch") {
        MockInference mock;
        auto iface = make_mock_interface(mock);
        LoopConfig lc;
        CompactionConfig cc;
        AgentEngine engine(iface, lc, cc);

        FakeTransportLatch latch;
        engine.set_external_interrupt(latch_cb, &latch);
        engine.set_external_reset(unlatch_cb, &latch);

        WHEN("the engine is interrupted") {
            engine.interrupt();

            THEN("the transport latch trips, as it always has") {
                CHECK(latch.latched);
            }

            AND_WHEN("the interrupt is reset for the next run") {
                engine.reset_interrupt();

                // THE bug. reset_interrupt() cleared interrupt_flag_ and
                // stopped there, so every subsequent external tool call
                // short-circuited to an empty response forever.
                THEN("the transport latch is released too") {
                    CHECK_FALSE(latch.latched);
                }
            }
        }
    }
}

SCENARIO("gh#150: a stdio transport's interrupt is observable and reversible",
         "[mcp][gh150][2.12.1]") {
    GIVEN("an unopened stdio transport") {
        StdioTransport t("probe", "/usr/bin/env",
                         std::vector<std::string>{"cat"}, {}, 30000U);

        THEN("it starts un-interrupted") {
            CHECK_FALSE(t.is_interrupted());
        }

        WHEN("it is interrupted") {
            t.interrupt();

            // A consumer could not previously tell this had happened:
            // send_request returned an empty string and execute() reported
            // status=ok, so the failure was indistinguishable from success.
            THEN("the state is observable") {
                CHECK(t.is_interrupted());
            }

            AND_WHEN("the interrupt is cleared") {
                t.clear_interrupt();
                THEN("the transport is usable again") {
                    CHECK_FALSE(t.is_interrupted());
                }
            }
        }
    }
}

SCENARIO("gh#150: open() clears the interrupt, as its doc always claimed",
         "[mcp][gh150][2.12.1]") {
    GIVEN("an interrupted transport whose command exists") {
        StdioTransport t("probe", "/usr/bin/env",
                         std::vector<std::string>{"cat"}, {}, 30000U);
        t.interrupt();
        REQUIRE(t.is_interrupted());

        WHEN("open() succeeds") {
            const bool opened = t.open();
            REQUIRE(opened);

            THEN("the flag is clear, so the reopened transport is usable") {
                CHECK_FALSE(t.is_interrupted());
            }

            t.close();
        }
    }
}

SCENARIO("gh#150: a suppressed external call is REPORTED as a failure",
         "[mcp][gh150][2.12.1]") {
    // The other half of the severity. The transport dropping calls was
    // survivable; the transport dropping calls while the logs said
    // status=ok is what made it undiagnosable from outside. status comes
    // from classify_tool_result -> looks_like_tool_error, which routes on
    // the leading text, and these messages did not lead with "Error".
    GIVEN("an external client whose transport is not connected") {
        auto transport = std::make_unique<StdioTransport>(
            "probe", "/usr/bin/env",
            std::vector<std::string>{"cat"},
            std::map<std::string, std::string>{}, 30000U);
        ExternalMCPClient client("probe", std::move(transport));

        WHEN("a tool is called") {
            const auto envelope = client.execute("search", "{}");

            THEN("the envelope classifies as an error, not as ok") {
                CHECK(mcp::looks_like_tool_error(
                    nlohmann::json::parse(envelope).value("result", "")));
            }
        }
    }

    GIVEN("a connected transport that has been interrupted") {
        auto owned = std::make_unique<StdioTransport>(
            "probe", "/usr/bin/env",
            std::vector<std::string>{"cat"},
            std::map<std::string, std::string>{}, 30000U);
        StdioTransport* raw = owned.get();
        REQUIRE(raw->open());
        raw->interrupt();
        ExternalMCPClient client("probe", std::move(owned));

        WHEN("a tool is called") {
            const auto envelope = client.execute("search", "{}");
            const auto text =
                nlohmann::json::parse(envelope).value("result", "");

            THEN("it classifies as an error") {
                CHECK(mcp::looks_like_tool_error(text));
            }
            // A 0 ms "timed out" is a contradiction, and chasing it cost
            // the reporter real time. The interrupted case now says so.
            THEN("it names the interrupt rather than claiming a timeout") {
                INFO("text=[" << text << "]");
                CHECK(text.find("interrupted") != std::string::npos);
                CHECK(text.find("timed out") == std::string::npos);
            }
        }

        raw->close();
    }
}
