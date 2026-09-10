// SPDX-License-Identifier: Apache-2.0
/**
 * @file generation_summary.h
 * @brief One post-generation summary line, formatted identically on every
 *        decode path.
 *
 * gh#151: the plain decode paths logged
 *
 *     Generated: 87 tokens, finish=stop, 3040ms, 28.5 tok/s
 *
 * and the speculative path logged a differently shaped line that carried no
 * throughput at all. Both paths computed the value — `spec_finalize` has
 * populated `throughput_tok_s` since gh#108 — so this was never a missing
 * measurement in the engine. It was a measurement a reader of the LOGS could
 * take on one arm and not on the other, which is exactly the comparison
 * speculative decoding exists to be judged by.
 *
 * The formatter is shared rather than duplicated because two hand-rolled
 * lines drifting apart IS the defect being fixed here; a second one would
 * reintroduce it the first time either changed.
 *
 * Pure and vendor-free so the format is CPU-unit-testable with no model and
 * no GPU — same reason warm_keep_util.h and partial_offload.h are.
 *
 * @version 2.12.2
 */

#pragma once

#include <string>

namespace entropic {

/**
 * @brief The numbers a finished generation reports, whatever produced it.
 *
 * `n_drafted`/`n_accepted` are zero on a plain decode; the formatter omits
 * the speculative clause entirely in that case rather than printing zeros,
 * so a plain line stays byte-identical to what it has always been and
 * existing log parsers keep working.
 *
 * @version 2.12.2
 */
struct GenerationSummary {
    int token_count = 0;           ///< Tokens actually generated.
    std::string finish_reason;     ///< stop / length / error / cancel.
    double generation_time_ms = 0.0;  ///< Wall clock for the generation.
    double throughput_tok_s = 0.0;    ///< token_count / seconds.
    int n_drafted = 0;             ///< Speculative: tokens proposed (0 = plain).
    int n_accepted = 0;            ///< Speculative: tokens accepted.
};

/**
 * @brief Format the one-line generation summary.
 *
 * Plain decode reads:
 *   "Generated: 87 tokens, finish=stop, 3040ms, 28.5 tok/s"
 *
 * Speculative decode APPENDS the draft accounting, so an A/B between the
 * two arms can be read off the same field in both:
 *   "Generated: 87 tokens, finish=stop, 3040ms, 28.5 tok/s,
 *    drafted=1396, accepted=906, accept_rate=0.649"
 *
 * @param s The finished generation's numbers.
 * @return The formatted line, without a trailing newline.
 * @utility
 * @version 2.12.2
 */
inline std::string format_generation_summary(const GenerationSummary& s) {
    char buf[256];
    int n = std::snprintf(
        buf, sizeof(buf),
        "Generated: %d tokens, finish=%s, %.0fms, %.1f tok/s",
        s.token_count, s.finish_reason.c_str(),
        s.generation_time_ms, s.throughput_tok_s);
    std::string out(buf, n > 0 ? static_cast<size_t>(n) : 0U);

    // Only when speculation actually ran. A plain decode printing
    // "drafted=0, accepted=0, accept_rate=0.000" would be noise on every
    // line, and would make "did speculation run" ambiguous in the logs.
    if (s.n_drafted > 0) {
        const double rate = static_cast<double>(s.n_accepted)
                          / static_cast<double>(s.n_drafted);
        n = std::snprintf(buf, sizeof(buf),
                          ", drafted=%d, accepted=%d, accept_rate=%.3f",
                          s.n_drafted, s.n_accepted, rate);
        out.append(buf, n > 0 ? static_cast<size_t>(n) : 0U);
    }
    return out;
}

}  // namespace entropic
