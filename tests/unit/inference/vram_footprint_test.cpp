// SPDX-License-Identifier: Apache-2.0
/**
 * @file vram_footprint_test.cpp
 * @brief gh#142: a VRAM estimate honest enough to gate on.
 *
 * @par The RED this was written against
 * `test-gh108-agentic-benchmark` does not fail — it SIGABRTs:
 * @code
 *   ggml_backend_cuda_buffer_type_alloc_buffer: allocating 857.61 MiB on
 *     device 0: cudaMalloc failed: out of memory
 *   ggml-backend.cpp:179: GGML_ASSERT(buffer) failed
 * @endcode
 * The engine had an admission gate for exactly this (`residency_admits` →
 * `ENTROPIC_ERROR_TIER_MODEL_TOO_LARGE`) but it is gated on
 * `vram_budget_bytes_ > 0`, and that was resolved ONLY from an undocumented
 * env var — so on every default deployment the gate was dead and llama.cpp
 * aborted the host process instead.
 *
 * @par Why the estimate had to be fixed before the gate could be switched on
 * The v2.2.4 estimate counts the whole weights file regardless of `gpu_layers`,
 * prices KV at a flat 16 KiB/token regardless of `cache_type`, and ignores the
 * vision projector entirely — which is the buffer that actually blew up. Turning
 * the gate on with that estimate trades an abort for FALSE REFUSALS: the
 * Qwen3.6-35B-A3B IQ3_XXS (~13 GB) runs today at `gpu_layers=15` on an 11 GB
 * card, and a gate that scores it as fully resident would refuse a config that
 * demonstrably works. Every case below exists to stop one of those.
 *
 * @version 2.11.0
 */

#include "vram_footprint.h"

#include <catch2/catch_test_macros.hpp>

using entropic::estimate_vram_footprint;
using entropic::FootprintInputs;
using entropic::kv_scale_for_cache_type;
using entropic::recommend_context_length;

namespace {

constexpr uint64_t kGiB = 1024ull * 1024ull * 1024ull;
constexpr uint64_t kMiB = 1024ull * 1024ull;

/// @brief The gh#108 agentic benchmark's E4B-Q8 arm, which aborted the process.
FootprintInputs agentic_e4b_q8() {
    FootprintInputs in;
    in.weights_bytes = 8192172032ull;  // gemma-4-E4B-it-Q8_0.gguf, 7.63 GiB
    in.mmproj_bytes = 899265152ull;    // the 857.61 MiB buffer that failed
    in.gpu_layers = 99;                // benchmark asks for full offload
    in.context_length = 131072;        // "true 128k, max VRAM contention"
    in.cache_type_k = "f16";
    in.cache_type_v = "f16";
    in.vram_reserve_mb = 512;
    return in;
}

}  // namespace

SCENARIO("gh#142 KV is priced by its cache type, not a flat rate",
         "[inference][gh142][vram][cpu]")
{
    GIVEN("the f16 baseline") {
        THEN("it is the unit the other types are expressed against") {
            CHECK(kv_scale_for_cache_type("f16") == 1.0);
        }
    }

    GIVEN("quantized KV cache types") {
        THEN("each costs proportionally less than f16 per element") {
            // ggml block sizes, 32 elements per block:
            //   q8_0 = 34 B -> 1.0625 B/elem -> 0.53125 x f16(2 B/elem)
            //   q4_0 = 18 B -> 0.5625 B/elem -> 0.28125 x f16
            CHECK(kv_scale_for_cache_type("q8_0") == 0.53125);
            CHECK(kv_scale_for_cache_type("q4_0") == 0.28125);
            CHECK(kv_scale_for_cache_type("q5_0") == 0.34375);
        }
        THEN("f32 costs double, not half") {
            CHECK(kv_scale_for_cache_type("f32") == 2.0);
        }
    }

    GIVEN("an unrecognised cache type") {
        THEN("it is priced as f16 — the estimate never silently under-counts") {
            CHECK(kv_scale_for_cache_type("wat") == 1.0);
        }
    }

    GIVEN("the same context at f16 versus q4_0") {
        FootprintInputs f16 = agentic_e4b_q8();
        FootprintInputs q4 = agentic_e4b_q8();
        q4.cache_type_k = "q4_0";
        q4.cache_type_v = "q4_0";

        auto a = estimate_vram_footprint(f16);
        auto b = estimate_vram_footprint(q4);

        THEN("both are computable") {
            REQUIRE(a.known);
            REQUIRE(b.known);
        }
        THEN("the q4_0 arm is materially smaller — the flat rate hid this") {
            // 128k x 16 KiB = 2 GiB at f16; q4_0 is ~0.28 of that.
            CHECK(b.bytes < a.bytes);
            CHECK((a.bytes - b.bytes) > 1400ull * kMiB);
        }
    }
}

SCENARIO("gh#142 offload placement decides whether weights count against VRAM",
         "[inference][gh142][vram][cpu]")
{
    GIVEN("full offload requested as -1") {
        FootprintInputs in = agentic_e4b_q8();
        in.gpu_layers = -1;
        auto est = estimate_vram_footprint(in);

        THEN("the whole weights file is resident and counted") {
            REQUIRE(est.known);
            CHECK(est.bytes > in.weights_bytes);
        }
    }

    GIVEN("full offload requested as 99, llama.cpp's 'all layers' convention") {
        auto est = estimate_vram_footprint(agentic_e4b_q8());

        THEN("it is treated identically to -1") {
            REQUIRE(est.known);
            CHECK(est.bytes > agentic_e4b_q8().weights_bytes);
        }
    }

    GIVEN("a CPU-resident tier, gpu_layers = 0") {
        FootprintInputs in = agentic_e4b_q8();
        in.gpu_layers = 0;
        auto est = estimate_vram_footprint(in);

        THEN("weights cost no VRAM at all") {
            REQUIRE(est.known);
            CHECK(est.bytes < in.weights_bytes);
        }
    }

    GIVEN("PARTIAL offload — the Qwen3.6-35B-A3B at gpu_layers=15 case") {
        // This is the false-refusal guard. A ~13 GB model runs today at
        // gpu_layers=15 on an 11 GB card. How much of it lands on the GPU
        // depends on the model's layer count, which is not knowable without
        // reading GGUF metadata. The estimate must therefore DECLINE to answer
        // rather than guess — an unknown estimate leaves the gate open, where a
        // guess would refuse a configuration that demonstrably works.
        FootprintInputs in;
        in.weights_bytes = 13ull * kGiB;
        in.gpu_layers = 15;
        in.context_length = 8192;
        in.cache_type_k = "q8_0";
        in.cache_type_v = "q8_0";
        in.vram_reserve_mb = 512;

        auto est = estimate_vram_footprint(in);

        THEN("the estimate reports itself unknown, and does not fabricate one") {
            CHECK_FALSE(est.known);
            CHECK(est.bytes == 0);
        }
        THEN("it says why, so the log explains the un-enforced gate") {
            CHECK(std::string(est.reason).find("partial") != std::string::npos);
        }
    }
}

SCENARIO("gh#142 the vision projector is counted — it is what actually failed",
         "[inference][gh142][vram][cpu]")
{
    GIVEN("a tier with a vision mmproj, fully offloaded") {
        FootprintInputs with = agentic_e4b_q8();
        FootprintInputs without = agentic_e4b_q8();
        without.mmproj_bytes = 0;

        auto a = estimate_vram_footprint(with);
        auto b = estimate_vram_footprint(without);

        THEN("its bytes are in the estimate") {
            REQUIRE(a.known);
            REQUIRE(b.known);
            CHECK(a.bytes - b.bytes == with.mmproj_bytes);
        }
    }

    GIVEN("a CPU-resident tier with an mmproj") {
        FootprintInputs in = agentic_e4b_q8();
        in.gpu_layers = 0;
        FootprintInputs no_proj = in;
        no_proj.mmproj_bytes = 0;

        THEN("the projector costs no VRAM either — it follows the weights") {
            CHECK(estimate_vram_footprint(in).bytes
                  == estimate_vram_footprint(no_proj).bytes);
        }
    }
}

SCENARIO("gh#142 the config that aborted the process is refused instead",
         "[inference][gh142][vram][cpu]")
{
    GIVEN("the E4B-Q8 @ 128k full-offload arm and a 1080 Ti's free VRAM") {
        // Measured on the failing run: "10121 MiB free" of 11162 MiB total.
        const uint64_t available = 10121ull * kMiB;
        auto est = estimate_vram_footprint(agentic_e4b_q8());

        THEN("the estimate exceeds what the card has") {
            REQUIRE(est.known);
            CHECK(est.bytes > available);
        }

        THEN("a context that DOES fit is recommended, not just a refusal") {
            int ctx = recommend_context_length(agentic_e4b_q8(), available);
            CHECK(ctx > 0);
            CHECK(ctx < 131072);

            FootprintInputs fitted = agentic_e4b_q8();
            fitted.context_length = ctx;
            auto refit = estimate_vram_footprint(fitted);
            REQUIRE(refit.known);
            CHECK(refit.bytes <= available);
        }
    }

    GIVEN("a request that already fits") {
        FootprintInputs in = agentic_e4b_q8();
        in.context_length = 4096;
        const uint64_t available = 10121ull * kMiB;

        THEN("the recommendation never inflates what was asked for") {
            CHECK(recommend_context_length(in, available) <= 4096);
        }
    }

    GIVEN("a model whose weights alone exceed the card") {
        FootprintInputs in = agentic_e4b_q8();
        in.weights_bytes = 20ull * kGiB;
        const uint64_t available = 10121ull * kMiB;

        THEN("no context fits, and the caller is told so with 0") {
            CHECK(recommend_context_length(in, available) == 0);
        }
    }

    GIVEN("an unknowable partial-offload tier") {
        FootprintInputs in;
        in.weights_bytes = 13ull * kGiB;
        in.gpu_layers = 15;
        in.context_length = 8192;
        in.vram_reserve_mb = 512;

        THEN("no recommendation is invented either") {
            CHECK(recommend_context_length(in, 10121ull * kMiB) == 0);
        }
    }
}

// ── gh#144: session pool multiplies the KV term ──────────

SCENARIO("gh#144: a session pool multiplies the KV estimate",
         "[vram_footprint][gh144][2.12.0]") {
    GIVEN("a tier that fits comfortably at one session") {
        FootprintInputs one = agentic_e4b_q8();
        one.context_length = 32768;
        one.max_sessions = 1;
        const auto base = entropic::estimate_vram_footprint(one);
        REQUIRE(base.known);

        WHEN("the same tier asks for three resident sessions") {
            FootprintInputs three = one;
            three.max_sessions = 3;
            const auto pooled = entropic::estimate_vram_footprint(three);

            THEN("the KV term triples rather than being counted once") {
                // Without this the gate under-counts by exactly N and can
                // admit a config that then aborts the host inside
                // llama.cpp — the failure gh#142 built the gate to prevent.
                REQUIRE(pooled.known);
                const uint64_t kv_one =
                    static_cast<uint64_t>(32768.0
                        * entropic::kv_bytes_per_token(one));
                CHECK(pooled.bytes == base.bytes + 2 * kv_one);
            }
        }
    }
}

SCENARIO("gh#144: the fit recommendation is per session, not total",
         "[vram_footprint][gh144][2.12.0]") {
    GIVEN("a budget sized so ONE session exactly fits the request") {
        FootprintInputs one = agentic_e4b_q8();
        one.context_length = 32768;
        one.max_sessions = 1;
        // recommend_context_length never inflates a request that already
        // fits, so a generous budget returns the requested length for any
        // session count and proves nothing. Pin the budget to the
        // one-session estimate so the divide is actually load-bearing.
        const auto base = entropic::estimate_vram_footprint(one);
        REQUIRE(base.known);
        const uint64_t budget = base.bytes;

        const int fits_one =
            entropic::recommend_context_length(one, budget);
        REQUIRE(fits_one > 0);

        WHEN("three sessions are requested against the same budget") {
            FootprintInputs three = one;
            three.max_sessions = 3;
            const int fits_three =
                entropic::recommend_context_length(three, budget);

            THEN("the recommended PER-SESSION window shrinks") {
                // Returning the single-session number here would hand the
                // operator a config three times too large, from the very
                // function whose job is to offer one that works.
                CHECK(fits_three < fits_one);
            }
            AND_THEN("it is near a third, allowing for 512-rounding") {
                CHECK(fits_three <= (fits_one / 3) + 512);
            }
        }
    }
}

SCENARIO("gh#144: max_sessions of 0 or 1 is priced identically",
         "[vram_footprint][gh144][regression][2.12.0]") {
    GIVEN("a tier with the field left at its default") {
        FootprintInputs def = agentic_e4b_q8();
        def.context_length = 8192;

        THEN("an explicit 1 and a degenerate 0 both match the default") {
            FootprintInputs one = def;  one.max_sessions = 1;
            FootprintInputs zero = def; zero.max_sessions = 0;
            const auto a = entropic::estimate_vram_footprint(def);
            const auto b = entropic::estimate_vram_footprint(one);
            const auto c = entropic::estimate_vram_footprint(zero);
            CHECK(a.bytes == b.bytes);
            CHECK(a.bytes == c.bytes);
        }
    }
}
