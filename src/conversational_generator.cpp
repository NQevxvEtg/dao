// src/conversational_generator.cpp
#include "conversational_generator.hpp"
#include <iostream>
#include <limits>
#include <iomanip>

ConversationalGenerator::ConversationalGenerator(
    TextSdrEncoder* text_encoder,
    std::vector<SpatialPooler*>* spatial_poolers,
    std::vector<ResonanceLayer*>* resonance_layers,
    std::vector<TemporalMemory*>* temporal_memories,
    torch::Tensor& vocab_matrix, const EmotionConfig& emotion_config, torch::Device device)
    : text_enc_(text_encoder),
      sps_(spatial_poolers),
      rls_(resonance_layers),
      tms_(temporal_memories),
      vocab_matrix_(vocab_matrix),
      emotion_config_(emotion_config),
      device_(device),
      coordinates_({0.0, 0.0}) {
    
    const int position_sdr_size = 2048;
    grid_encoder_ = GridCellEncoder(position_sdr_size, static_cast<int>(position_sdr_size * 0.02));
    grid_encoder_.addModule(50.0, 101);
}

void ConversationalGenerator::startNewConversation() {
    for(auto& tm : *tms_) {
        tm->resetStates();
    }
    coordinates_ = {0.0, 0.0};
    std::cout << "New conversation started." << std::endl;
}

void ConversationalGenerator::feedInput(int token_id) {
    SDR token_sdr = text_enc_->encodeSingleToken(token_id);
    coordinates_[0] += 1.0;
    coordinates_[1] += 1.0;
    SDR position_sdr = grid_encoder_.encode(coordinates_);
    SDR concatenated_sdr = token_sdr;
    concatenated_sdr.insert(concatenated_sdr.end(), position_sdr.begin(), position_sdr.end());

    SpatialPooler& sp = *(*sps_)[0];
    ResonanceLayer& rl = *(*rls_)[0];
    TemporalMemory& tm = *(*tms_)[0];

    SDR basis_sdr = sp.process(concatenated_sdr, false);
    torch::Tensor rdr = rl.process(basis_sdr);
    tm.process(rdr);
}

torch::Tensor ConversationalGenerator::getPrediction() {
    return (*tms_)[0]->getPredictiveState();
}

std::string ConversationalGenerator::respondTo(const std::string& prompt_text, int max_new_tokens) {
    torch::NoGradGuard no_grad;
    
    std::vector<int> prompt_token_ids = text_enc_->tokenize(prompt_text);
    if (!prompt_token_ids.empty()) {
        for(const auto& token_id : prompt_token_ids) {
            feedInput(token_id);
        }
    }

    std::vector<int> generated_ids;
    for (int i = 0; i < max_new_tokens; ++i) {
        torch::Tensor prediction_tensor = getPrediction();
        int next_token_id;

        if (i == 0) {
            next_token_id = decodePrediction(prediction_tensor, {text_enc_->getUnkId(), 2, 3});
        } else {
            next_token_id = decodePrediction(prediction_tensor);
        }

        if (next_token_id == text_enc_->getUnkId() || next_token_id >= text_enc_->getVocabSize() || next_token_id == 2) {
            break;
        }
        generated_ids.push_back(next_token_id);
        feedInput(next_token_id);
    }

    std::string response = text_enc_->decode(generated_ids);
    const std::string sp_space = "\xE2\x96\x81";
    size_t pos = 0;
    while ((pos = response.find(sp_space, pos)) != std::string::npos) {
        response.replace(pos, sp_space.length(), " ");
    }
    if (!response.empty() && response.front() == ' ') {
        response = response.substr(1);
    }
    return response;
}

int ConversationalGenerator::decodePrediction(const torch::Tensor& prediction_tensor, const std::vector<int>& banned_tokens) {
    torch::Tensor logits = torch::matmul(vocab_matrix_, prediction_tensor).squeeze();
    logits = torch::clamp(logits, -15.0f, 15.0f);

    for (const auto& token_id : banned_tokens) {
        if (token_id >= 0 && token_id < logits.size(0)) {
            logits[token_id] = -std::numeric_limits<float>::infinity();
        }
    }

    // [SLLM FIX] Replace the fragile top_k logic with a robust `masked_fill_` implementation.
    if (emotion_config_.top_k > 0 && emotion_config_.top_k < logits.size(0)) {
        // Find the value of the k-th largest logit.
        auto k_th_value = std::get<0>(torch::topk(logits, emotion_config_.top_k))[-1];
        // Create a mask where logits are less than this threshold.
        auto mask = logits < k_th_value;
        // Set all values that are True in the mask to -infinity (in-place).
        logits.masked_fill_(mask, -std::numeric_limits<float>::infinity());
    }

    logits /= emotion_config_.temp;
    auto probs = torch::softmax(logits, 0);

    if (torch::any(torch::isnan(probs)).item<bool>()) {
        // This block should no longer be reachable, but is kept as a safeguard.
        return text_enc_->getUnkId();
    }

    int next_token_id = torch::multinomial(probs, 1).item<int>();
    
    return next_token_id;
}