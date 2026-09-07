// SPDX-License-Identifier: Apache-2.0
/**
 * @file tool_namespace_test.cpp
 * @brief gh#145: a consumer app hosting the engine presents its own tools.
 *
 * Before v2.12.0 the five bridge tool names were string literals written out
 * TWICE in external_bridge.cpp — once in tool_definitions() and again in
 * dispatch_tool()'s compare chain, ~470 lines apart, with nothing tying them
 * together. Any consumer-configurable naming has to keep those two in step, so
 * the property under test here is that qualify and strip are exact inverses.
 *
 * Reachable as a CPU unit test only because the rule lives in the private
 * facade header — ExternalBridge::dispatch is private and tools/list is
 * otherwise only observable over a real unix socket. Same rationale, and the
 * same precedent, as final_text.h (gh#130, v2.10.2).
 *
 * @version 2.12.0
 */

#include "tool_namespace.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using entropic::facade::qualify_tool_name;
using entropic::facade::strip_tool_prefix;

namespace {

/// @brief The suffixes the bridge advertises, mirroring tool_suffix::.
const std::vector<std::string>& suffixes() {
    static const std::vector<std::string> kAll = {
        "ask", "ask_status", "status", "context_clear", "context_count"};
    return kAll;
}

} // namespace

SCENARIO("gh#145: the default namespace is byte-identical to 2.11.1",
         "[tool_namespace][gh145][regression][2.12.0]") {
    GIVEN("the default tool_prefix") {
        const std::string prefix = "entropic";

        THEN("every advertised name matches the pre-2.12.0 literal") {
            CHECK(qualify_tool_name(prefix, "ask") == "entropic.ask");
            CHECK(qualify_tool_name(prefix, "ask_status")
                  == "entropic.ask_status");
            CHECK(qualify_tool_name(prefix, "status") == "entropic.status");
            CHECK(qualify_tool_name(prefix, "context_clear")
                  == "entropic.context_clear");
            CHECK(qualify_tool_name(prefix, "context_count")
                  == "entropic.context_count");
        }
    }
}

SCENARIO("gh#145: qualify and strip are inverses for every suffix",
         "[tool_namespace][gh145][2.12.0]") {
    GIVEN("a range of prefixes a consumer might configure") {
        // "acme.docs" is the case that rules out "strip to the first dot".
        const std::vector<std::string> prefixes = {
            "entropic", "sumac", "acme.docs", "a", ""};

        THEN("strip(qualify(x)) == x, always") {
            for (const auto& prefix : prefixes) {
                for (const auto& suffix : suffixes()) {
                    const auto wire = qualify_tool_name(prefix, suffix);
                    CHECK(strip_tool_prefix(prefix, wire) == suffix);
                }
            }
        }
    }
}

SCENARIO("gh#145: a foreign namespace does not resolve",
         "[tool_namespace][gh145][2.12.0]") {
    GIVEN("a bridge configured as 'sumac'") {
        const std::string prefix = "sumac";

        WHEN("a caller sends the stock entropic.* name") {
            const auto out = strip_tool_prefix(prefix, "entropic.ask");

            THEN("it is returned unchanged, so dispatch reports it unknown") {
                // Returning "ask" here would silently resolve another host's
                // tool name. Returning the original lets the unknown-tool
                // branch echo exactly what the caller sent.
                CHECK(out == "entropic.ask");
            }
        }

        WHEN("a caller sends a prefix that is only a partial match") {
            THEN("it does not resolve") {
                // "sumacx.ask" shares the first five characters.
                CHECK(strip_tool_prefix(prefix, "sumacx.ask")
                      == "sumacx.ask");
                // Bare "sumac" with no dot is not a qualified name.
                CHECK(strip_tool_prefix(prefix, "sumac") == "sumac");
            }
        }
    }
}

SCENARIO("gh#145: an empty prefix means unqualified names",
         "[tool_namespace][gh145][2.12.0]") {
    GIVEN("a host that wants bare tool names") {
        THEN("qualify is identity and strip is identity") {
            CHECK(qualify_tool_name("", "ask") == "ask");
            CHECK(strip_tool_prefix("", "ask") == "ask");
            // And a dotted name is left alone rather than being split.
            CHECK(strip_tool_prefix("", "entropic.ask") == "entropic.ask");
        }
    }
}
