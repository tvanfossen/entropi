// SPDX-License-Identifier: Apache-2.0
/**
 * @file prompt_log_util.h
 * @brief Which prompt messages to log in full, and which to reference.
 *
 * gh#152: `log_prompt` wrote every message in the list on every turn. The
 * list grows monotonically, so a message emitted at turn 5 is re-logged by
 * turns 6, 7, 8 … and anyone counting model output by grepping the log
 * over-counts by roughly the number of remaining turns.
 *
 * That is not hypothetical. A consumer counting finding-shaped lines across
 * two arms of an A/B got 147 against 380 and read it as the second arm
 * looping 2.6x more — a serious defect in their own code, which is where
 * they went looking. It was re-emission: 36 and 38 "End prompt" markers
 * against 36 and 38 generation turns. Their conclusion is the right one:
 * the log is not a transcript, and it reads like one.
 *
 * The rule already existed for the system message, which is hashed and
 * elided when unchanged. It was applied to the one message known to be
 * large and invariant, and never to the ones that ACCUMULATE — which are
 * precisely the ones a reader is trying to count. This generalises it.
 *
 * Pure so the decision is CPU-unit-testable with no logger, no model and no
 * conversation — the same reason warm_keep_util.h and partial_offload.h are.
 *
 * @version 2.12.2
 */

#pragma once

#include <cstdlib>
#include <string>
#include <unordered_set>

namespace entropic {

/**
 * @brief Remembers which message bodies a session's log has already shown.
 *
 * Keyed by content hash, not by index: a message's index shifts as the
 * conversation is compacted, but its content is what a reader would be
 * double-counting.
 *
 * @version 2.12.2
 */
class PromptLogDedup {
public:
    /**
     * @brief Record a message body and say whether to log it in full.
     *
     * First sighting logs in full; every later sighting is a reference.
     *
     * @param content_hash Hash of the message body.
     * @return true the first time this body is seen, false afterwards.
     * @utility
     * @version 2.12.2
     */
    bool note(size_t content_hash) {
        return seen_.insert(content_hash).second;
    }

    /**
     * @brief Forget everything, so the next turn logs in full again.
     *
     * Used when the conversation itself is replaced (a cleared context, a
     * different session), where re-showing the bodies is correct.
     *
     * @utility
     * @version 2.12.2
     */
    void reset() { seen_.clear(); }

    /**
     * @brief How many distinct bodies have been shown.
     * @return Count of full logs emitted so far.
     * @utility
     * @version 2.12.2
     */
    size_t shown() const { return seen_.size(); }

private:
    std::unordered_set<size_t> seen_;
};

/**
 * @brief Whether the operator has demanded unabridged prompt logs.
 *
 * The full prompt is genuinely valuable when diagnosing what the model
 * actually saw, so eliding it must never be the only option.
 * `ENTROPIC_LOG_FULL_PROMPT=1` restores the pre-gh#152 behaviour verbatim.
 *
 * @return true when every message should be logged in full, every turn.
 * @utility
 * @version 2.12.2
 */
inline bool prompt_log_unabridged() {
    const char* v = std::getenv("ENTROPIC_LOG_FULL_PROMPT");
    return v != nullptr && v[0] == '1';
}

/**
 * @brief The one-line stand-in for a body already shown earlier in the log.
 *
 * Carries enough to reconstruct the sequence — role, size and hash — so a
 * reader can still tell WHICH message occupies the slot and match it to its
 * earlier full rendering.
 *
 * @param index Position in the message list.
 * @param role Message role.
 * @param size_chars Body length in characters.
 * @param content_hash Hash of the body.
 * @return The reference line, without a trailing newline.
 * @utility
 * @version 2.12.2
 */
inline std::string prompt_log_reference(size_t index,
                                        const std::string& role,
                                        size_t size_chars,
                                        size_t content_hash) {
    char buf[160];
    int n = std::snprintf(
        buf, sizeof(buf),
        "[%zu] role=%s [shown above, %zu chars, hash=%016zx]",
        index, role.c_str(), size_chars, content_hash);
    return std::string(buf, n > 0 ? static_cast<size_t>(n) : 0U);
}

}  // namespace entropic
