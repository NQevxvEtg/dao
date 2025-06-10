// src/trainer.cpp
#include "trainer.hpp"
#include "dao_model.hpp"
#include "spatial_pooler.hpp"
#include "resonance_layer.hpp"
#include "temporal_memory.hpp"
#include "grid_cell_encoder.hpp"
#include "progress_bar.hpp"
#include "text_sdr_encoder.hpp"

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <vector>

// evaluate_model is unchanged ...
double evaluate_model(DaoModel &model, TextSdrEncoder &encoder, const std::vector<int> &validation_token_ids, torch::Device device, bool verbose) {
    if (model.spatial_poolers.empty()) return 0.0;
    
    torch::NoGradGuard no_grad;
    for (auto& tm : model.temporal_memories) tm.resetStates();

    const int position_sdr_size = 2048;
    GridCellEncoder position_encoder(position_sdr_size, static_cast<int>(position_sdr_size * 0.02));
    position_encoder.addModule(50.0, 101);
    
    int correct_predictions = 0;
    int total_predictions = validation_token_ids.size() - 1;
    if (total_predictions <= 0) return 0.0;

    ProgressBar eval_bar(total_predictions, "  Evaluating");

    for (size_t i = 0; i < validation_token_ids.size() - 1; ++i) {
        int current_token_id = validation_token_ids[i];
        int actual_next_token_id = validation_token_ids[i + 1];

        torch::Tensor predictive_state = model.temporal_memories[0].getPredictiveState();
        torch::Tensor logits = torch::matmul(model.vocab_matrix, predictive_state).squeeze();
        auto prediction = torch::argmax(logits).item<int>();

        if (prediction == actual_next_token_id) {
            correct_predictions++;
        }

        SDR token_sdr = encoder.encodeSingleToken(current_token_id);
        std::vector<double> coordinates = {static_cast<double>(i), static_cast<double>(i)};
        SDR position_sdr = position_encoder.encode(coordinates);
        SDR concatenated_sdr = token_sdr;
        concatenated_sdr.insert(concatenated_sdr.end(), position_sdr.begin(), position_sdr.end());
        
        SDR basis_sdr = model.spatial_poolers[0].process(concatenated_sdr, false);
        torch::Tensor rdr = model.resonance_layers[0].process(basis_sdr);
        model.temporal_memories[0].process(rdr);
        
        eval_bar.update(i + 1);
    }
    eval_bar.done();
    
    double accuracy = (total_predictions > 0) ? (static_cast<double>(correct_predictions) / total_predictions) * 100.0 : 0.0; 
    if (verbose) {
        std::cout << "\n--- Validation Results ---" << std::endl;
        std::cout << "   - Correct Predictions: " << correct_predictions << " / " << total_predictions << std::endl;
        std::cout << "   - Top-1 Accuracy:      " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    }
    
    return accuracy;
}

void train_model(DaoModel &model, TextSdrEncoder &encoder, const std::vector<int> &corpus_token_ids, const std::vector<int> &validation_token_ids) {
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }
    // Constants for model structure
    const int column_count = 4096;
    const int token_sdr_size = 2048;
    const int position_sdr_size = 2048;
    const int concatenated_input_size = token_sdr_size + position_sdr_size;
    
    // Construct the model structure regardless of training or loading
    if (model.spatial_poolers.empty()) {
        model.spatial_poolers.emplace_back(concatenated_input_size, column_count, 0);
        model.resonance_layers.emplace_back(column_count, column_count, device);
        model.temporal_memories.emplace_back(column_count, column_count, device);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(true);
        model.vocab_matrix = torch::empty({encoder.getVocabSize(), column_count}, options);
        torch::nn::init::normal_(model.vocab_matrix, 0.0, 0.01);
    }
    
    if (corpus_token_ids.empty()) {
        model.load("./model.bin", device);
        return; 
    }
    
    std::cout << "Training on " << device << "." << std::endl;
    
    const int epochs = 20; // [SLLM] Increased default epochs
    const float learning_rate = 1e-4;

    // --- [SLLM] Early stopping parameters ---
    const int patience = 2; // Stop after 2 epochs with no improvement
    int epochs_without_improvement = 0;
    double best_validation_accuracy = -1.0;
    // ---

    std::vector<torch::Tensor> parameters;
    parameters.push_back(model.resonance_layers[0].getWeights());
    auto tm_params = model.temporal_memories[0].getParameters();
    parameters.insert(parameters.end(), tm_params.begin(), tm_params.end());
    parameters.push_back(model.vocab_matrix);

    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(learning_rate));
    auto criterion = torch::nn::CrossEntropyLoss();

    GridCellEncoder position_encoder(position_sdr_size, static_cast<int>(position_sdr_size * 0.02));
    position_encoder.addModule(50.0, 101);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "\n--- Epoch " << epoch + 1 << "/" << epochs << " ---" << std::endl;
        model.temporal_memories[0].resetStates();
        double total_loss = 0.0;
        int processed_tokens = 0;

        ProgressBar train_bar(corpus_token_ids.size() - 1, "  Training");

        for (size_t i = 0; i < corpus_token_ids.size() - 1; ++i) {
            int current_token_id = corpus_token_ids[i];
            int target_token_id = corpus_token_ids[i+1];

            SDR token_sdr = encoder.encodeSingleToken(current_token_id);
            std::vector<double> coordinates = {static_cast<double>(i), static_cast<double>(i)};
            SDR position_sdr = position_encoder.encode(coordinates);
            SDR concatenated_sdr = token_sdr;
            concatenated_sdr.insert(concatenated_sdr.end(), position_sdr.begin(), position_sdr.end());
            
            SDR basis_sdr = model.spatial_poolers[0].process(concatenated_sdr, false);
            torch::Tensor rdr = model.resonance_layers[0].process(basis_sdr);
            model.temporal_memories[0].process(rdr);
            
            torch::Tensor predictive_state = model.temporal_memories[0].getPredictiveState();
            torch::Tensor logits = torch::matmul(model.vocab_matrix, predictive_state).squeeze();
            logits = torch::clamp(logits, -15.0f, 15.0f);

            torch::Tensor target = torch::tensor({target_token_id}, torch::TensorOptions().dtype(torch::kLong).device(device));
            torch::Tensor loss = criterion(logits.unsqueeze(0), target);
            
            if (torch::isnan(loss).item<bool>()) {
                std::cerr << "Warning: NaN loss detected at step " << i << ". Skipping update." << std::endl;
                continue;
            }

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(parameters, 1.0);
            optimizer.step();

            total_loss += loss.item<double>();
            processed_tokens++;
            train_bar.update(i + 1);
        }
        train_bar.done();
        if(processed_tokens == 0) continue;

        std::cout << "   - Average Training Loss: " << std::fixed << std::setprecision(4) << (total_loss / processed_tokens) << std::endl;
        
        // --- [SLLM] Early stopping logic ---
        double current_accuracy = evaluate_model(model, encoder, validation_token_ids, device, true);

        if (current_accuracy > best_validation_accuracy) {
            best_validation_accuracy = current_accuracy;
            epochs_without_improvement = 0;
            std::cout << "   -> New best validation accuracy. Saving checkpoint to model_best.bin" << std::endl;
            model.save("./model_best.bin"); // Save the best model checkpoint
        } else {
            epochs_without_improvement++;
            std::cout << "   -> Validation accuracy did not improve for "
                      << epochs_without_improvement << " epoch(s)." << std::endl;
        }

        if (epochs_without_improvement >= patience) {
            std::cout << "\n--- Early stopping triggered after " << patience
                      << " epochs without improvement. ---" << std::endl;
            break; // Exit the epoch loop
        }
        // ---
    }
    
    // --- [SLLM] Load the best model before final save ---
    std::cout << "\n--- Assimilation Complete. Loading best model and saving final state. ---" << std::endl;
    model.load("./model_best.bin", device);
    model.save("./model.bin");
}