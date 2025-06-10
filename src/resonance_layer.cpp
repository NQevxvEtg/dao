// src/resonance_layer.cpp
#include "resonance_layer.hpp"
#include <stdexcept>

ResonanceLayer::ResonanceLayer(int basis_sdr_size, int rdr_size, torch::Device device)
    : _basis_sdr_size(basis_sdr_size), 
      _rdr_size(rdr_size),
      _device(device) {
    
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(_device)
                       .requires_grad(true);

    // [SLLM FIX] Initialize weights correctly to be a leaf tensor.
    // First create an empty tensor, then initialize it in-place.
    _weights = torch::empty({rdr_size, basis_sdr_size}, options);
    torch::nn::init::normal_(_weights, 0.0, 0.01); // Mean 0.0, Stddev 0.01
}

torch::Tensor ResonanceLayer::process(const SDR& basis_sdr) {
    if (basis_sdr.size() != _basis_sdr_size) {
        throw std::invalid_argument("Input basis_sdr has incorrect size for ResonanceLayer.");
    }

    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor basis_vector = torch::zeros({_basis_sdr_size, 1}, cpu_options);
    for(size_t i = 0; i < basis_sdr.size(); ++i) {
        if(basis_sdr[i] > 0) {
            basis_vector[i] = 1.0f;
        }
    }

    basis_vector = basis_vector.to(_device);
    torch::Tensor rdr = torch::matmul(_weights, basis_vector);
    
    return rdr;
}