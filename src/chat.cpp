// src/chat.cpp
#include "dao_model.hpp"
#include "conversational_generator.hpp"
#include "emotion.hpp"
#include "text_sdr_encoder.hpp"
#include "trainer.hpp"
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>

std::string find_first_txt_file(const std::string& directory) {
    if (!std::filesystem::exists(directory)) return "";
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            return entry.path().string();
        }
    }
    return "";
}

int main() {
    // [SLLM MODIFIED] Add more detailed CUDA logging.
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        device = torch::kCUDA;
        std::cout << " - Detected " << torch::cuda::device_count() << " CUDA device(s)." << std::endl;
    } else {
        std::cout << "Running on CPU. (torch::cuda::is_available() returned false)" << std::endl;
    }

    const std::string model_path = "./model.bin";
    const std::string vocab_path = "./tokenizer.model";
    
    const int token_sdr_size = 2048;
    const int token_active_bits = static_cast<int>(token_sdr_size * 0.02);

    DaoModel model;
    TextSdrEncoder encoder(vocab_path, token_sdr_size, token_active_bits);

    if (std::filesystem::exists(model_path)) {
        train_model(model, encoder, {}, {}); // Pass empty vectors to trigger model loading path
    } else {
        std::cout << "--- Model file not found. Beginning one-time assimilation process. ---" << std::endl;
        
        const std::string train_dir = "./data/train/";
        std::string corpus_path = find_first_txt_file(train_dir);
        if (corpus_path.empty()) {
            std::cerr << "Error: No training corpus (.txt file) found in '" << train_dir << "'." << std::endl;
            return 1;
        }
        std::ifstream file(corpus_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        auto corpus_token_ids = encoder.tokenize(buffer.str());

        const std::string validate_dir = "./data/validate/";
        std::string validation_corpus_path = find_first_txt_file(validate_dir);
        if (validation_corpus_path.empty()) {
            std::cerr << "Error: No validation corpus (.txt file) found in '" << validate_dir << "'." << std::endl;
            return 1;
        }
        std::ifstream val_file(validation_corpus_path);
        std::stringstream val_buffer;
        val_buffer << val_file.rdbuf();
        auto validation_token_ids = encoder.tokenize(val_buffer.str());

        train_model(model, encoder, corpus_token_ids, validation_token_ids);
    }

    EmotionConfig emotion_config;
    std::vector<SpatialPooler*> sp_ptrs;
    for (auto &sp : model.spatial_poolers) sp_ptrs.push_back(&sp);
    std::vector<ResonanceLayer*> rl_ptrs;
    for (auto &rl : model.resonance_layers) rl_ptrs.push_back(&rl);
    std::vector<TemporalMemory*> tm_ptrs;
    for (auto &tm : model.temporal_memories) tm_ptrs.push_back(&tm);

    ConversationalGenerator generator(&encoder, &sp_ptrs, &rl_ptrs, &tm_ptrs, model.vocab_matrix, emotion_config, device);

    std::string user_input;
    generator.startNewConversation();
    while (true) {
        std::cout << "\n\n> ";
        std::getline(std::cin, user_input);
        if (user_input == "quit" || user_input == "exit") break;
        if (user_input == "new") {
            generator.startNewConversation();
            continue;
        }
        std::cout << "DAO: " << std::flush;
        std::string response = generator.respondTo(user_input);
        std::cout << response << std::endl;
    }

    return 0;
}