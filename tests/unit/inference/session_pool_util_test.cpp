// SPDX-License-Identifier: Apache-2.0
/**
 * @file session_pool_util_test.cpp
 * @brief gh#144: the session-pool geometry, asserted without a GPU.
 *
 * The rules decide whether a deployment fits on the card, so they are a pure
 * function and this test needs no model — same rationale as
 * warm_keep_util_test. The arithmetic being pinned here was verified against
 * the vendored llama.cpp: total KV is `n_ctx` cells whether or not the cache
 * is unified, so unified is not cheaper; it differs by giving one shared pool
 * instead of n_seq_max private guaranteed streams.
 *
 * @version 2.12.0
 */

#include "../../../src/inference/session_pool_util.h"

#include <catch2/catch_test_macros.hpp>

using namespace entropic;

namespace {

/// @brief A tier config with the consumer's real shape.
ModelConfig sumac_like() {
    ModelConfig cfg;
    cfg.context_length = 32768;
    cfg.n_parallel = 1;
    cfg.max_sessions = 1;
    return cfg;
}

} // namespace

SCENARIO("gh#144: a single session is bit-identical to pre-2.12.0",
         "[session_pool][gh144][regression][2.12.0]") {
    GIVEN("a tier that asks for no pool") {
        auto cfg = sumac_like();

        THEN("the geometry is exactly llama.cpp's single-sequence default") {
            auto g = derive_pool_geometry(cfg);
            CHECK(g.n_seq_max == 1);
            CHECK_FALSE(g.kv_unified);
            CHECK(g.n_ctx == 32768);
            CHECK(g.temp_seq_base == 1);
            CHECK(session_pool_conflict_reason(cfg).empty());
        }
    }
}

SCENARIO("gh#144: a pool allocates context_length PER SESSION",
         "[session_pool][gh144][2.12.0]") {
    GIVEN("three sessions at 32768 each") {
        auto cfg = sumac_like();
        cfg.max_sessions = 3;

        THEN("n_ctx is the total, not the per-session window") {
            auto g = derive_pool_geometry(cfg);
            // The consumer's requirement is 32k EACH. Setting n_ctx to
            // 32768 with n_seq_max 3 would silently give ~10.9k per
            // session under non-unified — a clamp, not a pool.
            CHECK(g.n_ctx == 98304);
            CHECK(g.n_seq_max == 3);
        }
        AND_THEN("the cache is NOT unified, so each stream is guaranteed") {
            CHECK_FALSE(derive_pool_geometry(cfg).kv_unified);
        }
        AND_THEN("temp seq ids start above the session slots") {
            // Sessions own [0,3); allocate_temp_seq_id would otherwise
            // hand out 1 and 2 and collide.
            CHECK(derive_pool_geometry(cfg).temp_seq_base == 3);
        }
    }
}

SCENARIO("gh#144: gh#98 batch fan-out keeps its unified buffer",
         "[session_pool][gh144][gh98][regression][2.12.0]") {
    GIVEN("a tier configured for batching and no pool") {
        auto cfg = sumac_like();
        cfg.n_parallel = 4;

        THEN("kv_unified stays on, exactly as before 2.12.0") {
            // seq_cp asserts on per-sequence buffers, so this must not
            // regress while adding the pool.
            auto g = derive_pool_geometry(cfg);
            CHECK(g.kv_unified);
            CHECK(g.n_seq_max == 4);
            CHECK(g.n_ctx == 32768);
        }
    }
}

SCENARIO("gh#144: a pool and batching together is a typed error",
         "[session_pool][gh144][2.12.0]") {
    GIVEN("a tier asking for both") {
        auto cfg = sumac_like();
        cfg.max_sessions = 3;
        cfg.n_parallel = 4;

        THEN("the conflict is named rather than silently resolved") {
            // Fail loud. Picking one for the operator would mean either
            // the pool has no private streams or run_batch asserts at
            // decode time — both silent, both worse than refusing.
            auto why = session_pool_conflict_reason(cfg);
            REQUIRE_FALSE(why.empty());
            CHECK(why.find("max_sessions") != std::string::npos);
            CHECK(why.find("n_parallel") != std::string::npos);
        }
    }

    GIVEN("a pool with no context_length") {
        auto cfg = sumac_like();
        cfg.max_sessions = 3;
        cfg.context_length = 0;

        THEN("that is refused too, since n_ctx would be zero") {
            CHECK_FALSE(session_pool_conflict_reason(cfg).empty());
        }
    }
}

SCENARIO("gh#144: degenerate values do not produce a broken geometry",
         "[session_pool][gh144][2.12.0]") {
    GIVEN("zero or negative counts") {
        auto cfg = sumac_like();
        cfg.max_sessions = 0;
        cfg.n_parallel = -3;

        THEN("both floor at 1 rather than yielding n_ctx 0 or a negative "
             "sequence count") {
            auto g = derive_pool_geometry(cfg);
            CHECK(g.n_seq_max == 1);
            CHECK(g.n_ctx == 32768);
            CHECK(g.temp_seq_base == 1);
        }
    }
}
