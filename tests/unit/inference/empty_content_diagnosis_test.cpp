// SPDX-License-Identifier: Apache-2.0
/**
 * @file empty_content_diagnosis_test.cpp
 * @brief gh#137: an empty turn must be explained by its ACTUAL cause.
 *
 * @par What this pins
 * gh#137 reported `Generate complete (batch): finish=stop, 154 chars` delivering
 * a turn with 0 chars. The engine's response was to tell the operator to raise
 * max_tokens — advice that is only correct when the generation was TRUNCATED.
 * With `finish_reason == "stop"` the model ended the turn itself while still
 * inside a reasoning block; more budget changes nothing, and the message sent
 * the reporter looking in the wrong place.
 *
 * The underlying gh#137 defect (why the model reasons from token 0 on a tier
 * with `enable_thinking: false`) did NOT reproduce on the hardware available —
 * two GPU runs on gemma-4 E2B QAT emitted no channel markers at all and
 * delivered 100% of their content. What IS provable without a reproduction is
 * that the diagnostic misattributes the cause, so that is what this pins.
 *
 * @version 2.11.0
 */

#include "empty_content_diagnosis.h"

#include <catch2/catch_test_macros.hpp>

using entropic::diagnose_empty_content;
using entropic::EmptyContentCause;
using entropic::explain_empty_content;

SCENARIO("gh#137 an empty turn is attributed to its real cause",
         "[inference][gh137][diagnostics][cpu]")
{
    GIVEN("content survived the strip") {
        THEN("there is nothing to explain, whatever the finish reason") {
            CHECK(diagnose_empty_content(false, true, "stop")
                  == EmptyContentCause::not_empty);
            CHECK(diagnose_empty_content(false, true, "length")
                  == EmptyContentCause::not_empty);
            CHECK(explain_empty_content(EmptyContentCause::not_empty).empty());
        }
    }

    GIVEN("the model produced nothing at all") {
        THEN("empty content is not a strip problem and is not reported as one") {
            CHECK(diagnose_empty_content(true, false, "stop")
                  == EmptyContentCause::not_empty);
        }
    }

    GIVEN("tokens were produced, content is empty, finish_reason is length") {
        auto cause = diagnose_empty_content(true, true, "length");

        THEN("it is a budget problem") {
            CHECK(cause == EmptyContentCause::budget_truncated);
        }
        THEN("the advice names max_tokens, because that IS the fix") {
            auto msg = explain_empty_content(cause);
            CHECK(msg.find("max_tokens") != std::string::npos);
        }
    }

    GIVEN("the gh#137 shape — tokens produced, content empty, finish=stop") {
        auto cause = diagnose_empty_content(true, true, "stop");

        THEN("it is the model stopping, not a budget ceiling") {
            CHECK(cause == EmptyContentCause::model_stopped);
        }

        THEN("the advice must NOT tell the operator to raise max_tokens") {
            // The regression that matters. The pre-2.11.0 message said
            // "Raise max_tokens." on exactly this input, which cannot help a
            // turn the model chose to end, and is what sent gh#137's reporter
            // down the wrong path.
            auto msg = explain_empty_content(cause);
            CHECK(msg.find("raising max_tokens will not help")
                  != std::string::npos);
            CHECK(msg.find("finish_reason is 'stop'") != std::string::npos);
        }
    }

    GIVEN("an unattributable finish reason") {
        auto cause = diagnose_empty_content(true, true, "error");

        THEN("the engine admits it cannot explain the turn") {
            CHECK(cause == EmptyContentCause::unknown);
            auto msg = explain_empty_content(cause);
            CHECK_FALSE(msg.empty());
            // Must not guess at a cause it does not have evidence for.
            CHECK(msg.find("max_tokens") == std::string::npos);
        }
    }
}
