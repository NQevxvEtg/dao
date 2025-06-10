// src/dao_model.cpp
#include "dao_model.hpp"
#include <iostream>
#include <filesystem>
#include <stdexcept>

void DaoModel::save(const std::string& path) {
    try {
        torch::serialize::OutputArchive archive;
        
        if (!resonance_layers.empty()) {
            archive.write("resonance_weights_0", this->resonance_layers[0].getWeights().to(torch::kCPU));
        }
        if (!temporal_memories.empty()) {
            this->temporal_memories[0].save(archive);
        }
        archive.write("vocab_matrix", this->vocab_matrix.to(torch::kCPU));

        archive.save_to(path);
        std::cout << "Model saved to " << path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }
}

void DaoModel::load(const std::string& path, torch::Device device) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model file not found at: " + path);
    }
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);

        // [SLLM FIX] Correctly load weights into existing layers.
        // The calling function is responsible for creating the layers first.
        if (this->resonance_layers.empty() || this->temporal_memories.empty()) {
             throw std::runtime_error("Model layers must be constructed before loading state.");
        }
        
        torch::Tensor resonance_weights;
        archive.read("resonance_weights_0", resonance_weights);
        this->resonance_layers[0].getWeights() = resonance_weights.to(device);
        
        this->temporal_memories[0].load(archive, device);
        
        archive.read("vocab_matrix", this->vocab_matrix);
        this->vocab_matrix = this->vocab_matrix.to(device).requires_grad_(true);

        std::cout << "Model loaded from " << path << " and moved to " << device << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
}