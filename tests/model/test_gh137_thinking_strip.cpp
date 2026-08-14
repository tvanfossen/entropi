// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_gh137_thinking_strip.cpp
 * @brief gh#137: reproduce a tier that generates text and delivers 0 chars.
 *
 * @par What was reported
 * On v2.10.3 a tier with `enable_thinking: false` logged
 * `Generate complete (batch): finish=stop, 154 chars` and the turn carried
 * `0 tool call(s), 0 chars`. The tier burned its empty-turn allowance and the
 * delegation failed having done nothing. The reporter reverted to v2.10.2 and
 * could not isolate the mechanism — they noted a SIBLING tier with the same
 * `enable_thinking: false` was unaffected, which correctly argues against
 * "thinking-disabled tiers emit nothing".
 *
 * @par What source review already ruled out
 * Five hypotheses died against the code and against the model's own chat
 * template, so this file exists to MEASURE rather than reason:
 *   - the strip is not over-greedy: it preserves everything BEFORE the marker
 *   - the toolless path does not bypass jinja (render_prompt -> apply_chat_template
 *     -> render_common_chat, same render, enable_thinking set)
 *   - enable_thinking IS threaded tier -> params (apply_tier_sampler_defaults)
 *   - gemma4's template DOES gate `<|think|>` on enable_thinking
 *   - gemma4 emits exactly one channel name (`<|channel>thought`), so no
 *     answer-bearing channel is being eaten
 * Our C++ strip and the template's own `strip_thinking` macro agree case by
 * case. The strip is not the bug.
 *
 * @par The contract this pins
 * A turn that produced tokens must not deliver empty content with a
 * MISATTRIBUTED cause. The engine's unclosed-reasoning diagnostic tells the
 * operator to "Raise max_tokens" — advice that is only correct when the
 * generation was truncated, i.e. `finish_reason == "length"`. The report shows
 * `finish=stop`: the model closed the turn while still inside a reasoning
 * block, which no budget increase can fix. Asserting that pairing is the
 * whole point — it fails on exactly the shape gh#137 describes.
 *
 * @version 2.11.0
 */

#include "model_test_context.h"  // helpers only — NO CATCH_REGISTER_LISTENER

#include <cstdio>
#include <string>

namespace {

std::filesystem::path models_dir() {
    return std::filesystem::path(getenv("HOME")) / ".entropic" / "models";
}

/**
 * @brief Point the default tier at gemma4 E4B QAT with thinking disabled.
 *
 * Mirrors the reporter's rig (E2B QAT stands in for E4B), `enable_thinking: false`, no
 * tools staged, so the turn takes the toolless render path.
 *
 * @param ctx Test context (config mutated in place).
 * @param enable_thinking Value to put on the tier.
 * @return Default tier name.
 * @version 2.11.0
 */
std::string configure_gemma4_tier(ModelTestContext& ctx, bool enable_thinking) {
    auto target = models_dir() / "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf";
    if (!std::filesystem::is_regular_file(target)) {
        SKIP("gemma-4-E4B-it-qat GGUF not present");
    }
    REQUIRE(load_registry(ctx.registry));
    REQUIRE(load_test_config(ctx.registry, ctx.config));
    auto tier_name = ctx.config.models.default_tier;
    auto it = ctx.config.models.tiers.find(tier_name);
    if (it == ctx.config.models.tiers.end()) {
        SKIP("no default tier in config to repoint");
    }
    auto& tier = it->second;
    tier.path = target;
    tier.adapter = "gemma4";
    // Full offload on the E2B QAT variant. E4B at partial offload
    // (gpu_layers=15) trips GGML_ASSERT(n_inputs <
    // GGML_SCHED_MAX_SPLIT_INPUTS) on this 11GB card — a llama.cpp
    // graph-split limit on the hybrid arch, unrelated to gh#137.
    // E2B shares gemma4's template and channel markers, so the
    // reasoning-strip behaviour under test is identical.
    tier.gpu_layers = 99;
    tier.context_length = 4096;
    tier.enable_thinking = enable_thinking;
    tier.grammar.reset();
    return tier_name;
}

/**
 * @brief Describe where the reasoning markers sit in a raw emission.
 * @param raw Raw model output, before any stripping.
 * @return Human-readable marker census for the failure report.
 * @version 2.11.0
 */
std::string marker_census(const std::string& raw) {
    const std::string kOpen = "<|channel>";
    const std::string kClose = "<channel|>";
    auto open_at = raw.find(kOpen);
    auto close_at = raw.find(kClose);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "open@%s close@%s",
                  open_at == std::string::npos
                      ? "none" : std::to_string(open_at).c_str(),
                  close_at == std::string::npos
                      ? "none" : std::to_string(close_at).c_str());
    return buf;
}

}  // namespace

TEST_CASE("gh#137 a toolless enable_thinking:false turn must not deliver "
          "empty content with budget advice that cannot help",
          "[model][gh137][thinking][gemma4]") {
    ModelTestContext ctx;
    auto tier_name = configure_gemma4_tier(ctx, /*enable_thinking=*/false);
    if (!init_orchestrator(ctx)) {
        SKIP("orchestrator init failed (resource/config)");
    }

    // A single greedy toolless turn does NOT reproduce — measured: no channel
    // markers at all, 280 chars in and out. The reporter's tier differs in
    // three ways that matter, so mirror them:
    //   - temperature 0.6 + presence_penalty 1.5, not greedy decode
    //   - it is a DELEGATION CHILD, i.e. multi-turn
    //   - the template (line 240) re-renders a prior assistant turn's
    //     reasoning as '<|channel>thought ... <channel|>' INTO the prompt
    // That last one is the hypothesis under test: once the context contains
    // the channel pattern, the model continues it even with thinking
    // disabled, and an unterminated continuation is what vanishes.
    std::vector<entropic::Message> msgs;

    entropic::Message u1;
    u1.role = "user";
    u1.content = "Which source file defines the audio callback?";
    msgs.push_back(u1);

    entropic::Message a1;
    a1.role = "assistant";
    a1.content = "<|channel>thought\nThe user wants a file path. I should "
                 "look for a callback registration.\n<channel|>"
                 "It is defined in src/audio/driver.cpp.";
    msgs.push_back(a1);

    entropic::Message u2;
    u2.role = "user";
    u2.content = "And why is it structured that way? One sentence.";
    msgs.push_back(u2);

    entropic::GenerationParams params;
    params.max_tokens = 512;
    params.temperature = 0.6f;
    params.presence_penalty = 1.5f;
    params.enable_thinking = false;

    auto r = ctx.orchestrator->generate(msgs, params, tier_name);

    std::printf("\n===gh137 toolless enable_thinking:false===\n"
                "code=%d finish=%s\n"
                "raw_content: %zu chars, %s\n"
                "content    : %zu chars\n"
                "raw=[%s]\n"
                "content=[%s]\n===\n",
                r.error_code, r.finish_reason.c_str(),
                r.raw_content.size(), marker_census(r.raw_content).c_str(),
                r.content.size(),
                r.raw_content.c_str(), r.content.c_str());

    REQUIRE(r.error_code == 0);

    // The gh#137 shape: tokens were produced, nothing survived.
    const bool vanished = r.content.empty() && !r.raw_content.empty();

    // THIS IS THE ASSERTION, not a conditional escape hatch. An earlier
    // revision WARNed and SUCCEEDed when the defect did not appear, which is a
    // vacuous pass wearing the costume of coverage. What this file can honestly
    // prove is the positive contract measured twice above: on gemma-4 with
    // enable_thinking:false the model emits no reasoning and the strip removes
    // nothing. If that ever stops holding, this fails — which is precisely the
    // regression gh#137 would look like from the outside.
    INFO("raw_content=" << r.raw_content);
    INFO("marker census: " << marker_census(r.raw_content));
    CHECK_FALSE(vanished);
    CHECK(r.content.size() == r.raw_content.size());
}
