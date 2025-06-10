// src/temporal_memory.hpp
#ifndef TEMPORAL_MEMORY_HPP
#define TEMPORAL_MEMORY_HPP

#include "types.hpp"
#include <torch/torch.h>

class TemporalMemory {
public:
    TemporalMemory() = default;
    TemporalMemory(int rdr_input_size, int num_cells, torch::Device device);

    void process(const torch::Tensor& rdr);
    void resetStates();
    const torch::Tensor& getPredictiveState() const;
    int getNumCells() const;
    
    // [SLLM ADDED] Methods to save/load all weight tensors
    void save(torch::serialize::OutputArchive& archive) const;
    void load(torch::serialize::InputArchive& archive, torch::Device device);

    // [SLLM ADDED] Collect all parameters for the optimizer
    std::vector<torch::Tensor> getParameters();

private:
    int _num_cells;
    int _rdr_input_size;
    torch::Device _device;

    torch::Tensor _input_weights;
    torch::Tensor _recurrent_weights;
    torch::Tensor _bias;
    torch::Tensor _cell_activations;
};

#endif // TEMPORAL_MEMORY_HPP