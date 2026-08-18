// SPDX-License-Identifier: Apache-2.0
/**
 * @file app_context_content_test.cpp
 * @brief gh#141: app_context supplied as inline content, not only as a path.
 *
 * @par What was reported
 * `app_context` could only ever name a file on disk. A consumer holding the
 * text in memory — a parent-authored teaching statement living in the app's own
 * SQLite database — had no supported way to deliver it. The obvious workaround
 * (render it to a temp file, pass the path) was closed off for two stated
 * reasons that are worth preserving rather than arguing with:
 *   - the file is a PROVENANCE boundary: what the app can rewrite at runtime it
 *     can rewrite wrongly and silently after the operator last reviewed it, so
 *     keeping it read-only means what is on disk is what a human put there;
 *   - on an Android target the writable location is not a stable path across
 *     launches, so "write a file and pass its path" has no answer even in
 *     principle.
 *
 * @par The shape chosen
 * The reporter's option (1), an object form carrying the text:
 * @code{.yaml}
 *   app_context:
 *     content: |
 *       This family follows a Scandinavian, play-first approach...
 * @endcode
 * This adds NO `entropic.h` surface — a C setter (their option 2) is a public
 * ABI decision and is deliberately not taken here. Every pre-existing spelling
 * keeps its meaning, which the regression cases below pin: a bare string is
 * still a path, `false` still disables, absent is still opt-out.
 *
 * @version 2.11.0
 */

#include <catch2/catch_test_macros.hpp>
#include <entropic/config/bundled_models.h>
#include <entropic/config/loader.h>
#include <entropic/prompts/manager.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

/// @brief Write a config YAML to a unique temp file and return its path.
std::filesystem::path write_config(const std::string& body,
                                   const std::string& tag) {
    auto dir = std::filesystem::temp_directory_path()
        / ("entropic-gh141-" + tag);
    std::filesystem::create_directories(dir);
    auto path = dir / "config.yaml";
    std::ofstream out(path);
    out << body;
    out.close();
    return path;
}

/// @brief Parse a config body, requiring success.
entropic::ParsedConfig parse(const std::string& body, const std::string& tag) {
    entropic::config::BundledModels registry;
    REQUIRE(registry.load(std::filesystem::path(TEST_DATA_DIR)
                          / "bundled_models.yaml").empty());
    entropic::ParsedConfig config;
    auto err = entropic::config::parse_config_file(
        write_config(body, tag), registry, config);
    REQUIRE(err.empty());
    return config;
}

}  // namespace

SCENARIO("gh#141 app_context carries inline content", "[config][gh141][cpu]") {
    GIVEN("an app_context object with a content block") {
        auto config = parse(
            "app_context:\n"
            "  content: |\n"
            "    This family follows a Scandinavian, play-first approach.\n"
            "    Lessons are short.\n",
            "inline");

        THEN("the text is carried on the config") {
            REQUIRE(config.app_context_content.has_value());
            CHECK(config.app_context_content->find("play-first")
                  != std::string::npos);
            CHECK(config.app_context_content->find("Lessons are short")
                  != std::string::npos);
        }
        THEN("no path is implied, and it is not treated as disabled") {
            CHECK_FALSE(config.app_context.has_value());
            CHECK_FALSE(config.app_context_disabled);
        }
    }

    GIVEN("inline content and the prompt loader") {
        std::string body;
        auto err = entropic::prompts::load_app_context(
            std::nullopt, std::optional<std::string>("Inline body text."),
            /*disabled=*/false, std::filesystem::path("/nonexistent"), body);

        THEN("the content is used without touching the filesystem") {
            CHECK(err.empty());
            CHECK(body == "Inline body text.");
        }
    }

    GIVEN("inline content on a tier that is explicitly disabled") {
        std::string body = "sentinel";
        auto err = entropic::prompts::load_app_context(
            std::nullopt, std::optional<std::string>("should not appear"),
            /*disabled=*/true, std::filesystem::path("/nonexistent"), body);

        THEN("disabled still wins — opt-out is not overridden by content") {
            CHECK(err.empty());
            CHECK(body.empty());
        }
    }

    GIVEN("both a path and inline content") {
        std::string body;
        auto err = entropic::prompts::load_app_context(
            std::filesystem::path("/nonexistent/never-read.md"),
            std::optional<std::string>("content wins"),
            /*disabled=*/false, std::filesystem::path("/nonexistent"), body);

        THEN("content wins and the unreadable path is never opened") {
            // If the path were consulted this would be a "file not found".
            CHECK(err.empty());
            CHECK(body == "content wins");
        }
    }
}

SCENARIO("gh#141 every existing app_context spelling keeps its meaning",
         "[config][gh141][cpu]") {
    GIVEN("a bare string") {
        auto config = parse("app_context: /tmp/some-context.md\n", "path");

        THEN("it is still a path, and no content is invented") {
            REQUIRE(config.app_context.has_value());
            CHECK(config.app_context->string() == "/tmp/some-context.md");
            CHECK_FALSE(config.app_context_content.has_value());
            CHECK_FALSE(config.app_context_disabled);
        }
    }

    GIVEN("an explicit false") {
        auto config = parse("app_context: false\n", "off");

        THEN("it is still disabled") {
            CHECK(config.app_context_disabled);
            CHECK_FALSE(config.app_context.has_value());
            CHECK_FALSE(config.app_context_content.has_value());
        }
    }

    GIVEN("no app_context key at all") {
        auto config = parse("log_level: DEBUG\n", "absent");

        THEN("it stays opt-out, with neither path nor content") {
            CHECK_FALSE(config.app_context.has_value());
            CHECK_FALSE(config.app_context_content.has_value());
            CHECK_FALSE(config.app_context_disabled);
        }
    }
}
