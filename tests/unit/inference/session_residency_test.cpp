// SPDX-License-Identifier: Apache-2.0
/**
 * @file session_residency_test.cpp
 * @brief gh#144: per-session KV residency, asserted without a GPU.
 *
 * The load-bearing scenario is the one that is WRONG against v2.11.1's single
 * shared resident_tokens_ vector: two sessions sharing a system prompt must
 * NOT let one reuse the other's KV. Against a single vector the prefix scan
 * returns a non-zero common length and warm-keep takes its destructive reuse
 * branch. Per-slot residency is what makes the answer zero.
 *
 * @version 2.12.0
 */

#include "../../../src/inference/session_residency.h"
#include "../../../src/inference/warm_keep_util.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using entropic::SessionResidency;
using entropic::kNoSessionSlot;
using entropic::warm_keep_cut;

namespace {

/// @brief A shared system prefix, then session-specific content.
std::vector<int> convo(int marker, int len) {
    std::vector<int> t = {1, 2, 3, 4, 5};  // shared system prompt
    for (int i = 0; i < len; ++i) { t.push_back(marker * 1000 + i); }
    return t;
}

} // namespace

SCENARIO("gh#144: one session's KV is never reused for another",
         "[session_residency][gh144][regression][2.12.0]") {
    GIVEN("two sessions that share a system prompt") {
        SessionResidency<int> res(3);
        std::string evicted;

        const int slot_a = res.acquire("repo-a", &evicted);
        const int slot_b = res.acquire("repo-b", &evicted);
        REQUIRE(slot_a != slot_b);

        const auto a_tokens = convo(1, 40);
        res.set_resident(slot_a, a_tokens);

        WHEN("session B asks what it can reuse") {
            const auto b_tokens = convo(2, 40);
            const auto cut = warm_keep_cut(
                res.resident(slot_b), b_tokens,
                static_cast<long>(res.resident(slot_b).size()));

            THEN("it can reuse nothing, because its own slot is empty") {
                // Against v2.11.1's single shared vector this asked A's
                // tokens instead, got a non-zero common prefix (the shared
                // system prompt), and took the reuse branch — seq_rm'ing
                // A's tail and skipping the prompt cache. Worse than no
                // warm-keep at all.
                CHECK(cut == 0);
            }
        }

        AND_WHEN("session A asks about its own continuation") {
            auto a_next = a_tokens;
            a_next.push_back(9999);
            const auto cut = warm_keep_cut(
                res.resident(slot_a), a_next,
                static_cast<long>(a_tokens.size()));

            THEN("it reuses its whole resident prefix bar the last token") {
                CHECK(cut == a_tokens.size());
            }
        }
    }
}

SCENARIO("gh#144: slots are stable and reacquired, not reshuffled",
         "[session_residency][gh144][2.12.0]") {
    GIVEN("a pool of three") {
        SessionResidency<int> res(3);
        std::string evicted;

        const int a = res.acquire("a", &evicted);
        const int b = res.acquire("b", &evicted);
        const int c = res.acquire("c", &evicted);

        THEN("each session keeps its slot across lookups") {
            CHECK(res.acquire("a", &evicted) == a);
            CHECK(res.acquire("b", &evicted) == b);
            CHECK(res.acquire("c", &evicted) == c);
            CHECK(evicted.empty());
            CHECK(res.assigned_count() == 3);
        }
        AND_THEN("an unknown session has no slot until it acquires one") {
            CHECK(res.slot_for("d") == kNoSessionSlot);
        }
    }
}

SCENARIO("gh#144: a full pool evicts the least recently used session",
         "[session_residency][gh144][2.12.0]") {
    GIVEN("three sessions filling a pool of three") {
        SessionResidency<int> res(3);
        std::string evicted;
        const int a = res.acquire("a", &evicted);
        res.acquire("b", &evicted);
        res.acquire("c", &evicted);
        res.set_resident(a, convo(1, 10));

        WHEN("b and c are used again, then a fourth session arrives") {
            res.acquire("b", &evicted);
            res.acquire("c", &evicted);
            const int d = res.acquire("d", &evicted);

            THEN("the least recently used session is the one evicted") {
                CHECK(evicted == "a");
                CHECK(d == a);           // d takes a's slot
                CHECK(res.slot_for("a") == kNoSessionSlot);
                CHECK(res.slot_for("b") != kNoSessionSlot);
                CHECK(res.slot_for("c") != kNoSessionSlot);
            }
            AND_THEN("the caller is told, so it can clear that slot's KV") {
                // Eviction drops KV only. History stays authoritative and
                // re-prefillable, so the evicted session's next turn costs
                // a cold prefill — exactly what every turn costs today.
                CHECK_FALSE(evicted.empty());
            }
        }
    }
}

SCENARIO("gh#144: invalidation clears residency without losing the slot",
         "[session_residency][gh144][2.12.0]") {
    GIVEN("a session with resident tokens") {
        SessionResidency<int> res(2);
        std::string evicted;
        const int slot = res.acquire("a", &evicted);
        res.set_resident(slot, convo(1, 20));
        REQUIRE_FALSE(res.resident(slot).empty());

        WHEN("its KV is invalidated") {
            res.invalidate(slot);

            THEN("residency is empty but the assignment survives") {
                CHECK(res.resident(slot).empty());
                CHECK(res.slot_for("a") == slot);
            }
        }

        WHEN("the whole context is cleared out of band") {
            const int other = res.acquire("b", &evicted);
            res.set_resident(other, convo(2, 20));
            res.invalidate_all();

            THEN("no slot claims residency any more") {
                // Some paths still clear every sequence unconditionally.
                // After one of those, believing any slot is resident is how
                // a stale prefix gets reused.
                CHECK(res.resident(slot).empty());
                CHECK(res.resident(other).empty());
            }
        }
    }
}

SCENARIO("gh#144: a degenerate pool size still yields one usable slot",
         "[session_residency][gh144][2.12.0]") {
    GIVEN("a pool constructed with zero slots") {
        SessionResidency<int> res(0);
        std::string evicted;

        THEN("it floors at one rather than rejecting every acquire") {
            CHECK(res.slots() == 1);
            CHECK(res.acquire("a", &evicted) == 0);
        }
        AND_THEN("an out-of-range slot reads as empty, never out of bounds") {
            CHECK(res.resident(7).empty());
            CHECK(res.resident(-1).empty());
        }
    }
}
