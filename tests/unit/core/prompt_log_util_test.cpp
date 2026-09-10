// SPDX-License-Identifier: Apache-2.0
/**
 * @file prompt_log_util_test.cpp
 * @brief gh#152: the prompt log shows each body once, not once per turn.
 *
 * log_prompt wrote every message on every turn, and the message list grows
 * monotonically — so a body emitted at turn 5 was re-logged by every turn
 * after it. A consumer counting finding-shaped lines across two A/B arms
 * read 147 against 380 and took it for one arm looping 2.6x more. It was
 * re-emission, and they went looking in their own code first.
 *
 * @version 2.12.2
 */

#include "../../../src/core/prompt_log_util.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

using entropic::PromptLogDedup;
using entropic::prompt_log_reference;
using entropic::prompt_log_unabridged;

SCENARIO("v2.12.2: a growing conversation logs each body exactly once",
         "[prompt_log][gh152][2.12.2]") {
    GIVEN("a conversation that grows by one message per turn") {
        PromptLogDedup dedup;
        // Turn 1 sees [a]; turn 2 sees [a,b]; turn 3 sees [a,b,c].
        // Pre-fix that is 1 + 2 + 3 = 6 full bodies logged for 3 distinct
        // messages — the over-count that made 147 look like 380.
        const size_t a = 111, b = 222, c = 333;
        int full_logs = 0;

        for (auto turn : {1, 2, 3}) {
            if (dedup.note(a)) { ++full_logs; }
            if (turn >= 2 && dedup.note(b)) { ++full_logs; }
            if (turn >= 3 && dedup.note(c)) { ++full_logs; }
        }

        THEN("three distinct bodies produce three full logs, not six") {
            CHECK(full_logs == 3);
            CHECK(dedup.shown() == 3);
        }
    }
}

SCENARIO("v2.12.2: the reference line can still place the message",
         "[prompt_log][gh152][2.12.2]") {
    GIVEN("a body that was shown earlier") {
        const auto line = prompt_log_reference(4, "assistant", 812, 0xabcdULL);

        // Eliding is only acceptable if a reader can still tell WHICH
        // message occupies the slot and match it to its full rendering.
        THEN("it names the index, the role and the size") {
            CHECK(line.find("[4]") != std::string::npos);
            CHECK(line.find("role=assistant") != std::string::npos);
            CHECK(line.find("812 chars") != std::string::npos);
        }
        AND_THEN("it carries the hash that ties it to the full body") {
            CHECK(line.find("abcd") != std::string::npos);
        }
    }
}

SCENARIO("v2.12.2: an operator can still demand the unabridged dump",
         "[prompt_log][gh152][2.12.2]") {
    // The full prompt is what you want when diagnosing what the model
    // actually saw, so eliding must never be the only option.
    GIVEN("no override set") {
        unsetenv("ENTROPIC_LOG_FULL_PROMPT");
        THEN("the log is abridged") { CHECK_FALSE(prompt_log_unabridged()); }
    }
    GIVEN("the override set") {
        setenv("ENTROPIC_LOG_FULL_PROMPT", "1", 1);
        THEN("the log is unabridged") { CHECK(prompt_log_unabridged()); }
        unsetenv("ENTROPIC_LOG_FULL_PROMPT");
    }
}
