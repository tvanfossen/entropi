// SPDX-License-Identifier: Apache-2.0
/**
 * @file session_pool_util.h
 * @brief gh#144: derive a tier's KV geometry from max_sessions, purely.
 *
 * @par Why this is a pure header
 * The same reason `warm_keep_util.h` is: the decision is a handful of
 * integer rules that decide whether a deployment fits on the card, and it
 * must be assertable by a CPU unit test with no model, no GPU and no
 * llama.cpp context. `build_cparams` applies the result; it does not decide
 * it.
 *
 * @par The geometry, and why non-unified
 * Verified against the vendored llama.cpp: total KV memory is `n_ctx` worth
 * of cells EITHER WAY (`llama-kv-cache.cpp:98,247` — the tensors are
 * `(n_embd_k_gqa, kv_size, n_stream)` with `n_stream = unified ? 1 :
 * n_seq_max`, and `llama-context.cpp:229-243` sets `n_ctx_seq = n_ctx` when
 * unified, `n_ctx / n_seq_max` when not). So unified is NOT cheaper. The
 * difference is one shared pool versus `n_seq_max` PRIVATE guaranteed
 * streams.
 *
 * Non-unified is chosen on fail-fast grounds, not memory. Under unified a
 * session that grows toward the full window starves the others and
 * `llama_decode` returns "could not find a KV slot" — a runtime failure with
 * a poor diagnostic and no attribution to a session. Non-unified converts
 * that into a per-session context overflow the existing compaction path
 * already handles against `context_length`.
 *
 * It also keeps `n_ctx_seq` inside the model's trained context. For three
 * 32768 sessions, unified would report `n_ctx_seq = 98304` and trip
 * llama.cpp's training-overflow warning (`llama-context.cpp:263`);
 * non-unified reports 32768 per stream.
 *
 * An operator who genuinely wants the cheap shared variant can still have
 * it by setting `max_sessions: 3, context_length: 12288` — the pool then
 * costs exactly today's allocation.
 *
 * @version 2.12.0
 */

#pragma once

#include <entropic/types/config.h>

#include <algorithm>
#include <string>

namespace entropic {

/// @brief Context geometry derived from a tier's session-pool request.
struct PoolGeometry {
    int n_seq_max = 1;        ///< cparams.n_seq_max
    bool kv_unified = false;  ///< cparams.kv_unified
    int n_ctx = 0;            ///< cparams.n_ctx (TOTAL cells, all streams)
    int temp_seq_base = 1;    ///< First seq id the batch fan-out may use
};

/**
 * @brief Derive the context geometry a tier's config implies.
 *
 * @param cfg Tier model config.
 * @return The geometry `build_cparams` should apply.
 * @req REQ-INFER-019
 * @version 2.12.0
 */
inline PoolGeometry derive_pool_geometry(const ModelConfig& cfg) {
    const int sessions = std::max(1, cfg.max_sessions);
    const int parallel = std::max(1, cfg.n_parallel);

    PoolGeometry g;
    g.n_seq_max = std::max(parallel, sessions);
    // gh#98's same-prefix batch fan-out REQUIRES a unified buffer (seq_cp
    // asserts on per-sequence buffers), so preserve that exactly — but a
    // session pool must NOT be unified, per the header. The two are made
    // mutually exclusive at configure time, so this can never be asked to
    // satisfy both.
    g.kv_unified = (parallel > 1) && (sessions <= 1);
    // context_length is PER SESSION.
    g.n_ctx = cfg.context_length * sessions;
    // Sessions own seq ids [0, sessions). The batch fan-out's temp ids must
    // start above them: allocate_temp_seq_id hands out 1, 2, 3... which
    // would otherwise collide with session slots. The configure-time
    // exclusion already makes this unreachable; raising the base is defence
    // in depth against a future change that removes the exclusion silently.
    g.temp_seq_base = std::max(1, sessions);
    return g;
}

/**
 * @brief Why this tier's session-pool request cannot be honoured.
 *
 * Returns a reason rather than a bool so the configure-time error names the
 * conflicting keys — a typed error the operator can act on, never a silent
 * clamp or fallback.
 *
 * @param cfg Tier model config.
 * @return Empty when the config is coherent; otherwise the reason.
 * @req REQ-INFER-019
 * @version 2.12.0
 */
inline std::string session_pool_conflict_reason(const ModelConfig& cfg) {
    std::string reason;
    if (cfg.max_sessions > 1) {
        if (cfg.n_parallel > 1) {
            reason = "max_sessions > 1 conflicts with n_parallel > 1: a "
                     "session pool needs private per-session KV streams, "
                     "while the same-prefix batch fan-out "
                     "(entropic_run_batch) needs a unified buffer. Set one "
                     "or the other, not both.";
        } else if (cfg.context_length <= 0) {
            reason = "max_sessions > 1 requires a positive context_length: "
                     "the pool allocates context_length * max_sessions "
                     "cells.";
        }
    }
    return reason;
}

}  // namespace entropic
