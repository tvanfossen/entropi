// SPDX-License-Identifier: Apache-2.0
/**
 * @file generation_summary_test.cpp
 * @brief gh#151: both decode paths report throughput in the same shape.
 *
 * The speculative path computed throughput_tok_s (gh#108) but never logged
 * it, and logged a differently shaped line instead. So a consumer measuring
 * an MTP-vs-plain A/B from logs could read throughput on the plain arm and
 * not on the speculative one — the single comparison speculation exists to
 * be judged by. They resorted to ESTIMATING the MTP arm's output volume from
 * drafted/accepted arithmetic, and had to report a result as an upper bound
 * because of it.
 *
 * These assert the shared formatter, which is what makes the two lines
 * incapable of drifting apart.
 *
 * @version 2.12.2
 */

#include "../../../src/inference/generation_summary.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using entropic::GenerationSummary;
using entropic::format_generation_summary;

SCENARIO("v2.12.2: a plain decode summary keeps its long-standing shape",
         "[generation_summary][gh151][2.12.2]") {
    GIVEN("a completed plain generation") {
        GenerationSummary s;
        s.token_count = 87;
        s.finish_reason = "stop";
        s.generation_time_ms = 3040.0;
        s.throughput_tok_s = 28.5;

        THEN("it formats exactly as it always has") {
            CHECK(format_generation_summary(s)
                  == "Generated: 87 tokens, finish=stop, 3040ms, 28.5 tok/s");
        }

        // A plain line printing drafted=0/accepted=0 on every generation
        // would be noise, and would make "did speculation run at all"
        // ambiguous in exactly the logs used to answer that question.
        AND_THEN("it carries no speculative clause") {
            const auto out = format_generation_summary(s);
            CHECK(out.find("drafted") == std::string::npos);
            CHECK(out.find("accept_rate") == std::string::npos);
        }
    }
}

SCENARIO("v2.12.2: a speculative summary reports throughput in the SAME field",
         "[generation_summary][gh151][2.12.2]") {
    GIVEN("a completed speculative generation") {
        GenerationSummary s;
        s.token_count = 87;
        s.finish_reason = "stop";
        s.generation_time_ms = 3040.0;
        s.throughput_tok_s = 28.5;
        s.n_drafted = 1396;
        s.n_accepted = 906;

        const auto out = format_generation_summary(s);

        // THE property. Whatever else differs, an A/B must be able to read
        // the same token count and the same tok/s off both arms.
        THEN("the plain prefix is byte-identical to the plain path's line") {
            CHECK(out.rfind(
                "Generated: 87 tokens, finish=stop, 3040ms, 28.5 tok/s", 0)
                  == 0);
        }

        AND_THEN("the draft accounting is appended, not substituted") {
            CHECK(out.find("drafted=1396") != std::string::npos);
            CHECK(out.find("accepted=906") != std::string::npos);
            CHECK(out.find("accept_rate=0.649") != std::string::npos);
        }

        // The consumer estimated ~2546 generated tokens from drafted/accepted
        // arithmetic because they could not read the real count. It is right
        // here, and it is not derivable from the draft numbers.
        AND_THEN("generated is reported directly, not implied by the drafts") {
            CHECK(out.find("87 tokens") != std::string::npos);
        }
    }
}

SCENARIO("v2.12.2: degenerate generations still format without lying",
         "[generation_summary][gh151][2.12.2]") {
    GIVEN("a run that produced nothing") {
        GenerationSummary s;
        s.finish_reason = "error";

        THEN("it reports zeros rather than dividing by them") {
            CHECK(format_generation_summary(s)
                  == "Generated: 0 tokens, finish=error, 0ms, 0.0 tok/s");
        }
    }

    GIVEN("a speculative run where nothing was accepted") {
        GenerationSummary s;
        s.token_count = 1;
        s.finish_reason = "stop";
        s.generation_time_ms = 100.0;
        s.throughput_tok_s = 10.0;
        s.n_drafted = 4;
        s.n_accepted = 0;

        THEN("the accept rate is 0, not a division fault") {
            CHECK(format_generation_summary(s).find("accept_rate=0.000")
                  != std::string::npos);
        }
    }
}
