// src/attention.cpp
#include "attention.hpp"
#include "rdse.hpp" // <-- [SLLM FIX] Include rdse.hpp to get the declaration for overlap()
#include <vector>
#include <numeric>
#include <algorithm>

namespace attention {

// [SLLM FIX] The redundant 'overlap' function definition has been removed from this file.

// The existing getSparseAttentionSdr function is left unchanged for potential future use.
AttentionOutput getSparseAttentionSdr(int current_sdr_idx, const std::vector<SDR>& all_sdrs, int k) {
    if (all_sdrs.empty() || all_sdrs.size() <= 1) {
        int sdr_size = all_sdrs.empty() ? 0 : all_sdrs[0].size();
        return { SDR(sdr_size, 0), SDR(sdr_size, 0) };
    }

    const SDR& target_sdr = all_sdrs[current_sdr_idx];
    int num_sdrs = all_sdrs.size();
    int sdr_size = target_sdr.size();

    std::vector<int> overlaps;
    overlaps.reserve(num_sdrs);
    for (int i = 0; i < num_sdrs; ++i) {
        if (i == current_sdr_idx) {
            overlaps.push_back(-1);
        } else {
            overlaps.push_back(overlap(target_sdr, all_sdrs[i]));
        }
    }

    int num_neighbors = num_sdrs - 1;
    int actual_k = std::min(k, num_neighbors);
    if (actual_k <= 0) {
        return { SDR(sdr_size, 0), SDR(sdr_size, 0) };
    }
    
    std::vector<int> indices(num_sdrs);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                      [&](int a, int b) { return overlaps[a] > overlaps[b]; });

    std::vector<int> top_k_indices;
    for(int i = 0; i < actual_k; ++i) {
        if (overlaps[indices[i]] > 0) {
            top_k_indices.push_back(indices[i]);
        }
    }

    if (top_k_indices.empty()) {
        return { SDR(sdr_size, 0), SDR(sdr_size, 0) };
    }
    
    std::vector<int> bit_counts(sdr_size, 0);
    for (int neighbor_idx : top_k_indices) {
        for (int bit = 0; bit < sdr_size; ++bit) {
            if (all_sdrs[neighbor_idx][bit] == 1) {
                bit_counts[bit]++;
            }
        }
    }
    SDR resonance_sdr(sdr_size, 0);
    int consensus_threshold = 1; 
    for(int bit = 0; bit < sdr_size; ++bit) {
        if (bit_counts[bit] > consensus_threshold) {
            resonance_sdr[bit] = 1;
        }
    }
    SDR union_sdr(sdr_size, 0);
    for (int neighbor_idx : top_k_indices) {
        for (int bit = 0; bit < sdr_size; ++bit) {
            union_sdr[bit] = union_sdr[bit] | all_sdrs[neighbor_idx][bit];
        }
    }
    SDR dissonance_sdr(sdr_size, 0);
    for (int bit = 0; bit < sdr_size; ++bit) {
        if (union_sdr[bit] == 1 && resonance_sdr[bit] == 0) {
            dissonance_sdr[bit] = 1;
        }
    }
    return { resonance_sdr, dissonance_sdr };
}


// --- [SLLM] Implementation of getResonanceVector (Unchanged) ---
VectorXf getResonanceVector(const std::vector<SDR>& history, int k) {
    if (history.empty() || history.size() <= 1) {
        int sdr_size = history.empty() ? 0 : history[0].size();
        return VectorXf::Zero(sdr_size);
    }

    const SDR& target_sdr = history.back(); // Use the most recent SDR as the query
    int num_sdrs = history.size();
    int sdr_size = target_sdr.size();

    // Calculate overlaps with the rest of the history
    std::vector<int> overlaps;
    overlaps.reserve(num_sdrs - 1);
    for (int i = 0; i < num_sdrs - 1; ++i) { // Exclude the target itself
        overlaps.push_back(overlap(target_sdr, history[i]));
    }

    // Find the top k most similar historical SDRs
    int num_neighbors = num_sdrs - 1;
    int actual_k = std::min(k, num_neighbors);
    if (actual_k <= 0) {
        return VectorXf::Zero(sdr_size);
    }
    
    std::vector<int> indices(num_neighbors);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                      [&](int a, int b) { return overlaps[a] > overlaps[b]; });

    // Build the resonance vector from the top_k neighbors
    VectorXf resonance_vector = VectorXf::Zero(sdr_size);
    int valid_neighbors = 0;
    for(int i = 0; i < actual_k; ++i) {
        if (overlaps[indices[i]] > 0) { // Only consider meaningful overlaps
            const SDR& neighbor_sdr = history[indices[i]];
            for(int bit = 0; bit < sdr_size; ++bit) {
                if(neighbor_sdr[bit] == 1) {
                    resonance_vector(bit) += 1.0f;
                }
            }
            valid_neighbors++;
        }
    }

    // Normalize the vector by the number of neighbors that contributed
    if (valid_neighbors > 0) {
        resonance_vector /= valid_neighbors;
    }

    return resonance_vector;
}

} // namespace attention