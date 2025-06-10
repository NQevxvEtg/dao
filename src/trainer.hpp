// src/trainer.hpp
#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "dao_model.hpp"
#include "text_sdr_encoder.hpp"
#include <vector>
#include <string>

// [SLLM REFACTORED] The main training function, now simplified for the new architecture.
void train_model(
    DaoModel &model,
    TextSdrEncoder &encoder,
    const std::vector<int> &corpus_token_ids,
    const std::vector<int> &validation_token_ids
);

// [SLLM REFACTORED] The evaluation function, adapted for the RDR architecture.
double evaluate_model(
    DaoModel &model,
    TextSdrEncoder &encoder,
    const std::vector<int> &validation_token_ids,
    bool verbose = true
);

#endif // TRAINER_HPP