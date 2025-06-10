// src/temporal_memory.cpp
#include "temporal_memory.hpp"
#include <stdexcept>

TemporalMemory::TemporalMemory(int rdr_input_size, int num_cells, torch::Device device)
    : _num_cells(num_cells),
      _rdr_input_size(rdr_input_size),
      _device(device) {

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(_device)
                       .requires_grad(true);

    _input_weights = torch::empty({_num_cells, _rdr_input_size}, options);
    _recurrent_weights = torch::empty({_num_cells, _num_cells}, options);
    _bias = torch::empty({_num_cells, 1}, options);

    torch::nn::init::normal_(_input_weights, 0.0, 0.01);
    torch::nn::init::normal_(_recurrent_weights, 0.0, 0.01);
    torch::nn::init::normal_(_bias, 0.0, 0.01);
    
    resetStates();
}

void TemporalMemory::resetStates() {
    _cell_activations = torch::zeros({_num_cells, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(_device));
}

void TemporalMemory::process(const torch::Tensor& rdr) {
    if (rdr.size(0) != _rdr_input_size) {
        throw std::runtime_error("RDR input tensor has incorrect size for TemporalMemory.");
    }
    
    auto prev_activations = _cell_activations.detach();

    auto rdr_on_device = rdr.to(_device);
    auto weighted_input = torch::matmul(_input_weights, rdr_on_device);
    auto weighted_recurrent = torch::matmul(_recurrent_weights, prev_activations);
    
    _cell_activations = torch::tanh(weighted_input + weighted_recurrent + _bias);
}

const torch::Tensor& TemporalMemory::getPredictiveState() const {
    return _cell_activations;
}

int TemporalMemory::getNumCells() const { return _num_cells; }

void TemporalMemory::save(torch::serialize::OutputArchive& archive) const {
    archive.write("tm_input_weights", _input_weights.to(torch::kCPU));
    archive.write("tm_recurrent_weights", _recurrent_weights.to(torch::kCPU));
    archive.write("tm_bias", _bias.to(torch::kCPU));
}

void TemporalMemory::load(torch::serialize::InputArchive& archive, torch::Device device) {
    archive.read("tm_input_weights", _input_weights);
    archive.read("tm_recurrent_weights", _recurrent_weights);
    archive.read("tm_bias", _bias);
    
    // [SLLM FIX] Use the correct in-place setter function with a trailing underscore.
    _input_weights = _input_weights.to(device).requires_grad_(true);
    _recurrent_weights = _recurrent_weights.to(device).requires_grad_(true);
    _bias = _bias.to(device).requires_grad_(true);
}

std::vector<torch::Tensor> TemporalMemory::getParameters() {
    return {_input_weights, _recurrent_weights, _bias};
}