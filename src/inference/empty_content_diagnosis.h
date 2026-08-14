// SPDX-License-Identifier: Apache-2.0
/**
 * @file empty_content_diagnosis.h
 * @brief gh#137: explain an empty turn without misattributing its cause.
 *
 * @par Why this exists
 * A turn can produce tokens and still deliver zero characters of content: the
 * model opened a reasoning block and never closed it, so the reasoning strip
 * correctly removed everything. That is legitimate — surfacing raw reasoning as
 * if it were the answer would be worse — but the operator needs to know WHICH
 * problem they have, and the two are fixed differently.
 *
 * The engine previously gave one message for both cases, telling the operator to
 * "Raise max_tokens". That advice is sound only when the generation was
 * truncated. gh#137 reported `finish=stop, 154 chars` delivering 0 chars — the
 * model ended the turn itself while still inside a reasoning block, and no
 * budget increase can fix that. The engine was pointing at the wrong knob, which
 * is worse than saying nothing.
 *
 * Kept as a pure function over a `finish_reason` string, deliberately free of
 * vendor types, so the decision is unit-testable on CPU. The judgment here is
 * the part worth pinning; the logging around it is not.
 *
 * @version 2.11.0
 */

#pragma once

#include <string>

namespace entropic {

/// @brief Why a turn produced tokens but delivered no content.
enum class EmptyContentCause {
    not_empty,          ///< Content survived; nothing to explain.
    budget_truncated,   ///< finish_reason == "length": ran out of tokens.
    model_stopped,      ///< Model ended the turn while still reasoning.
    unknown,            ///< Empty for a reason this rule cannot attribute.
};

/**
 * @brief Classify an empty-content turn from what the decode reported.
 *
 * @param content_empty Whether the post-strip content is empty.
 * @param produced_tokens Whether the model emitted anything at all.
 * @param finish_reason Decode finish reason ("stop", "length", "error", ...).
 * @return The cause, which decides what advice is honest to give.
 * @req REQ-INFER-010
 * @version 2.11.0
 */
inline EmptyContentCause diagnose_empty_content(bool content_empty,
                                                bool produced_tokens,
                                                const std::string& finish_reason) {
    EmptyContentCause cause = EmptyContentCause::unknown;
    if (!content_empty || !produced_tokens) {
        cause = EmptyContentCause::not_empty;
    } else if (finish_reason == "length") {
        cause = EmptyContentCause::budget_truncated;
    } else if (finish_reason == "stop") {
        cause = EmptyContentCause::model_stopped;
    }
    return cause;
}

/**
 * @brief Operator-facing explanation for an empty-content turn.
 *
 * @param cause Result of diagnose_empty_content.
 * @return Advice matching the actual cause, empty when there is nothing to say.
 * @req REQ-INFER-010
 * @version 2.11.0
 */
inline std::string explain_empty_content(EmptyContentCause cause) {
    // The gh#137 case deliberately does NOT mention max_tokens: the model chose
    // to stop, so more budget changes nothing and naming it misdirects.
    static const std::string kBudget =
        "The generation hit its token budget while still inside a reasoning "
        "block, so no answer was ever produced and the content is empty. This "
        "is a budget problem: raise max_tokens for this tier.";
    static const std::string kStopped =
        "The model ended the turn while still inside an unterminated reasoning "
        "block, so the content is empty. finish_reason is 'stop', NOT 'length' "
        "— this is not a budget problem and raising max_tokens will not help. "
        "The prompt or tier configuration is not converging on an answer.";
    static const std::string kUnknown =
        "The turn produced tokens but delivered no content, and the finish "
        "reason does not explain why. This is not a parse error; report it "
        "with the raw generation attached.";

    std::string out;
    if (cause == EmptyContentCause::budget_truncated) {
        out = kBudget;
    } else if (cause == EmptyContentCause::model_stopped) {
        out = kStopped;
    } else if (cause == EmptyContentCause::unknown) {
        out = kUnknown;
    }
    return out;
}

}  // namespace entropic
