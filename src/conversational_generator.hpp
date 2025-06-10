// src/conversational_generator.hpp
#ifndef CONVERSATIONAL_GENERATOR_HPP
#define CONVERSATIONAL_GENERATOR_HPP

#include "text_sdr_encoder.hpp"
#include "spatial_pooler.hpp"
#include "temporal_memory.hpp"
#include "resonance_layer.hpp"
#include "grid_cell_encoder.hpp"
#include "emotion.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>

class ConversationalGenerator {
public:
    ConversationalGenerator(
        TextSdrEncoder* text_encoder,
        std::vector<SpatialPooler*>* spatial_poolers,
        std::vector<ResonanceLayer*>* resonance_layers,
        std::vector<TemporalMemory*>* temporal_memories,
        torch::Tensor& vocab_matrix,
        const EmotionConfig& emotion_config,
        torch::Device device
    );

    std::string respondTo(const std::string& prompt_text, int max_new_tokens = 50);
    void startNewConversation();

private:
    void feedInput(int token_id);
    torch::Tensor getPrediction();
    
    // [SLLM MODIFIED] The signature is updated to allow for banning specific tokens during generation.
    int decodePrediction(const torch::Tensor& prediction_tensor, const std::vector<int>& banned_tokens = {});

    TextSdrEncoder* text_enc_;
    std::vector<SpatialPooler*>* sps_;
    std::vector<ResonanceLayer*>* rls_;
    std::vector<TemporalMemory*>* tms_;
    
    GridCellEncoder grid_encoder_;
    std::vector<double> coordinates_;

    torch::Tensor& vocab_matrix_;
    EmotionConfig emotion_config_;
    torch::Device device_;
};

#endif // CONVERSATIONAL_GENERATOR_HPP