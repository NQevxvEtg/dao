// src/dao_model.hpp
#ifndef DAO_MODEL_HPP
#define DAO_MODEL_HPP

#include <torch/torch.h>
#include <vector>
#include <string>

// [SLLM] Full definitions are now included here to give the model ownership
// and to facilitate easier serialization of the entire model state.
#include "spatial_pooler.hpp"
#include "resonance_layer.hpp"
#include "temporal_memory.hpp"

struct DaoModel {
    // Model Components
    std::vector<SpatialPooler> spatial_poolers;
    std::vector<ResonanceLayer> resonance_layers;
    std::vector<TemporalMemory> temporal_memories;
    torch::Tensor vocab_matrix;

    // Member function declarations
    void save(const std::string& path);
    void load(const std::string& path, torch::Device device);
};

#endif // DAO_MODEL_HPP