// src/resonance_layer.hpp
#ifndef RESONANCE_LAYER_HPP
#define RESONANCE_LAYER_HPP

#include "types.hpp"
#include <torch/torch.h> // [SLLM ADDED] Include the main LibTorch header
#include <cereal/cereal.hpp>
// [SLLM NOTE] Cereal does not have native support for torch::Tensor.
// We will omit its serialization for now and handle model saving/loading later.

class ResonanceLayer {
public:
    ResonanceLayer() = default;
    ResonanceLayer(int basis_sdr_size, int rdr_size, torch::Device device);

    // [SLLM MODIFIED] Now returns a torch::Tensor
    torch::Tensor process(const SDR& basis_sdr);

    const torch::Tensor& getWeights() const { return _weights; }
    torch::Tensor& getWeights() { return _weights; } // Non-const version for updates

private:
    int _basis_sdr_size;
    int _rdr_size;
    torch::Device _device; // The device (CPU or CUDA) where tensors live

    // [SLLM MODIFIED] The weights are now a torch::Tensor
    torch::Tensor _weights;
};

#endif // RESONANCE_LAYER_HPP