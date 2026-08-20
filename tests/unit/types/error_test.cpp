// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_error.cpp
 * @brief BDD tests for entropic_error_t and related functions.
 * @version 1.8.0
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <entropic/types/error.h>
#include <cstring>

SCENARIO("Error codes have human-readable names", "[error][types]") {
    GIVEN("Each defined error code") {
        auto code = GENERATE(
            ENTROPIC_OK,
            ENTROPIC_ERROR_INVALID_ARGUMENT,
            ENTROPIC_ERROR_INVALID_CONFIG,
            ENTROPIC_ERROR_INVALID_STATE,
            ENTROPIC_ERROR_MODEL_NOT_FOUND,
            ENTROPIC_ERROR_LOAD_FAILED,
            ENTROPIC_ERROR_GENERATE_FAILED,
            ENTROPIC_ERROR_TOOL_NOT_FOUND,
            ENTROPIC_ERROR_PERMISSION_DENIED,
            ENTROPIC_ERROR_PLUGIN_VERSION_MISMATCH,
            ENTROPIC_ERROR_PLUGIN_LOAD_FAILED,
            ENTROPIC_ERROR_TIMEOUT,
            ENTROPIC_ERROR_CANCELLED,
            ENTROPIC_ERROR_OUT_OF_MEMORY,
            ENTROPIC_ERROR_IO,
            ENTROPIC_ERROR_INTERNAL
        );

        WHEN("entropic_error_name is called") {
            const char* name = entropic_error_name(code);

            THEN("it returns a non-null, non-empty string") {
                REQUIRE(name != nullptr);
                REQUIRE(std::strlen(name) > 0);
            }

            THEN("the string starts with ENTROPIC_") {
                REQUIRE(std::strncmp(name, "ENTROPIC_", 9) == 0);
            }
        }
    }
}

SCENARIO("ENTROPIC_OK is zero", "[error][types]") {
    GIVEN("The success code") {
        THEN("it evaluates to false in boolean context") {
            REQUIRE(ENTROPIC_OK == 0);
        }
    }
}

// SCENARIO("entropic_last_error returns empty string initially") moved
// to tests/unit/api/multi_handle_test.cpp in v2.2.9 — the function's
// implementation moved from src/types/error.cpp to src/facade/entropic.cpp
// in v2.2.6 (gh#58 follow-up), so the symbol is no longer linkable from
// the types-only test target.

SCENARIO("Error callback registration rejects NULL handle", "[error][types]") {
    GIVEN("A NULL handle") {
        WHEN("setting an error callback") {
            auto result = entropic_set_error_callback(nullptr, nullptr, nullptr);

            THEN("it returns INVALID_ARGUMENT") {
                REQUIRE(result == ENTROPIC_ERROR_INVALID_ARGUMENT);
            }
        }
    }
}

SCENARIO("Error callback registration reports that it is not implemented",
         "[error][types][failloud]") {
    // Through v2.10.4 this returned ENTROPIC_OK while discarding both the
    // callback and user_data behind (void) casts, and entropic_error_callback_t
    // is invoked from nowhere in src/, include/ or python/src/. A consumer got
    // success and then silence — the engine claiming a capability it does not
    // have. Same shape as gh#133, where i_mcp_server.h documented a dlopen
    // plugin contract the engine never implemented.
    //
    // Not removed: the symbol is ENTROPIC_EXPORT and reaches the generated
    // Python wrapper, so deleting it is an ABI break. Reporting
    // NOT_IMPLEMENTED makes the gap visible at the call site instead, which is
    // what the error code was added for.
    GIVEN("a non-NULL handle") {
        // A non-null pointer is enough: the stub checks only for NULL, and
        // never dereferences the handle.
        auto* fake = reinterpret_cast<entropic_handle_t>(0x1);

        WHEN("an error callback is registered") {
            auto result = entropic_set_error_callback(fake, nullptr, nullptr);

            THEN("the engine admits the feature does not exist") {
                REQUIRE(result == ENTROPIC_ERROR_NOT_IMPLEMENTED);
                REQUIRE(result != ENTROPIC_OK);
            }
        }
    }
}
