// SPDX-License-Identifier: Apache-2.0
/**
 * @file turn_exclusion_test.cpp
 * @brief gh#144: only one turn at a time may run on a handle.
 *
 * `ENTROPIC_ERROR_ALREADY_RUNNING` has been declared in the enum, carried
 * in the error-string table and documented on two run entry points since
 * v1.8.9 — and returned from nowhere in the tree. Meanwhile gh#109 removed
 * `api_mutex` from all six run entry points so a long turn could not block
 * `entropic_interrupt()`, and the external bridge serves each client on its
 * own thread. Two concurrent asks therefore did not queue: they raced into
 * `run_turn`, concurrently mutating the shared `conversation_` vector and
 * concurrently decoding on one `llama_context`.
 *
 * The RED here is a MISSING GUARD rather than a wrong value — the same
 * shape as gh#143's `serialize_args`, which had no test at all. These
 * scenarios do not compile against v2.11.1 because `try_begin_turn` did not
 * exist, and the invariant they assert was unenforced.
 *
 * Kept at engine level and CPU-only: the claim is a compare-exchange on an
 * atomic, so it is fully testable against MockInference with no model, no
 * GPU, and no facade.
 *
 * @version 2.12.0
 */

#include <entropic/core/engine.h>
#include "mock_inference.h"
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>

using namespace entropic;
using namespace entropic::test;

namespace {

/// @brief Build an engine over a mock inference interface.
AgentEngine make_engine(InferenceInterface& iface) {
    static LoopConfig lc;
    static CompactionConfig cc;
    return AgentEngine(iface, lc, cc);
}

} // namespace

SCENARIO("gh#144: a second turn cannot start while one is in flight",
         "[engine][gh144][concurrency][2.12.0]") {
    GIVEN("an engine with no turn running") {
        MockInference mock;
        auto iface = make_mock_interface(mock);
        auto engine = make_engine(iface);

        REQUIRE_FALSE(engine.is_running());

        WHEN("a turn is claimed") {
            REQUIRE(engine.try_begin_turn());

            THEN("the engine reports itself running") {
                CHECK(engine.is_running());
            }
            AND_THEN("a second claim is refused") {
                // This is the whole contract: the loser is told, rather
                // than proceeding into a shared conversation vector.
                CHECK_FALSE(engine.try_begin_turn());
            }
            AND_THEN("the claim is reusable once released") {
                engine.end_turn();
                CHECK_FALSE(engine.is_running());
                CHECK(engine.try_begin_turn());
                engine.end_turn();
            }
        }
    }
}

SCENARIO("gh#144: exactly one of many racing claimants wins",
         "[engine][gh144][concurrency][2.12.0]") {
    GIVEN("an engine and eight threads racing to start a turn") {
        MockInference mock;
        auto iface = make_mock_interface(mock);
        auto engine = make_engine(iface);

        constexpr int kThreads = 8;
        std::atomic<int> winners{0};
        std::atomic<bool> go{false};

        WHEN("they all claim simultaneously") {
            std::vector<std::thread> threads;
            for (int i = 0; i < kThreads; ++i) {
                threads.emplace_back([&] {
                    while (!go.load()) { /* spin to tighten the race */ }
                    if (engine.try_begin_turn()) {
                        winners.fetch_add(1);
                    }
                });
            }
            go.store(true);
            for (auto& t : threads) { t.join(); }

            THEN("exactly one claim succeeds") {
                CHECK(winners.load() == 1);
            }
        }
    }
}

SCENARIO("gh#144: run_turn nested under an outer claim does not release it",
         "[engine][gh144][concurrency][2.12.0]") {
    GIVEN("an outer claim held by the facade") {
        MockInference mock;
        auto iface = make_mock_interface(mock);
        auto engine = make_engine(iface);

        REQUIRE(engine.try_begin_turn());

        WHEN("run_turn runs underneath it") {
            // The facade claims, then calls run_turn. run_turn must NOT
            // clear the flag on its way out — otherwise a second thread
            // could claim while the facade is still serialising results
            // out of the same conversation vector, which is precisely the
            // race gh#144 reported.
            engine.run_turn("Hello");

            THEN("the outer claim still holds") {
                CHECK(engine.is_running());
                CHECK_FALSE(engine.try_begin_turn());
            }
            AND_THEN("releasing the outer claim frees the engine") {
                engine.end_turn();
                CHECK_FALSE(engine.is_running());
            }
        }
    }
}

SCENARIO("gh#144: an unclaimed run_turn still reports itself running",
         "[engine][gh144][concurrency][2.12.0]") {
    GIVEN("an engine driven directly, with no outer claim") {
        MockInference mock;
        auto iface = make_mock_interface(mock);
        auto engine = make_engine(iface);

        WHEN("run_turn is called with no claim held") {
            engine.run_turn("Hello");

            THEN("it claims and releases on its own") {
                // Preserves the gh#40 contract that is_running() gates
                // entropic_queue_user_message, for callers that reach the
                // engine without going through the facade guard.
                CHECK_FALSE(engine.is_running());
                CHECK(engine.try_begin_turn());
                engine.end_turn();
            }
        }
    }
}
