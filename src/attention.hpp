// src/attention.hpp
#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "types.hpp" // [SLLM MODIFIED] Use the project-wide types
#include <vector>
#include <utility> 

namespace attention {

// This type alias can remain for other potential uses
using AttentionOutput = std::pair<SDR, SDR>;

// This function can remain for other potential uses
AttentionOutput getSparseAttentionSdr(int current_sdr_idx, const std::vector<SDR>& all_sdrs, int k);

/**
 * @brief [SLLM ADDED] Creates a weighted "resonance" vector from a history of SDRs.
 * This represents the consensus or "gist" of the recent context.
 *
 * @param history A vector of recent SDRs (e.g., winner cell patterns).
 * @param k The number of nearest neighbors to consider for the consensus.
 * @return A VectorXf where each element's value represents its strength in the consensus.
 */
VectorXf getResonanceVector(const std::vector<SDR>& history, int k);

} // namespace attention

#endif // ATTENTION_HPP