// src/text_sdr_encoder.hpp
#ifndef TEXT_SDR_ENCODER_HPP
#define TEXT_SDR_ENCODER_HPP

#include "rdse.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <cereal/cereal.hpp>
#include "cereal/types/Eigen.hpp"

// Forward declaration for SentencePiece processor
namespace sentencepiece { class SentencePieceProcessor; }

class TextSdrEncoder {
public:
    TextSdrEncoder(); // Default constructor
    TextSdrEncoder(const std::string& tokenizer_model_path, int sdr_size, int sdr_active_bits);
    
    ~TextSdrEncoder(); // Destructor declared
    TextSdrEncoder(TextSdrEncoder&&) noexcept; // Move constructor
    TextSdrEncoder& operator=(TextSdrEncoder&&) noexcept; // Move assignment

    TextSdrEncoder(const TextSdrEncoder&) = delete;
    TextSdrEncoder& operator=(const TextSdrEncoder&) = delete;

    // This method now just calls tokenize and then loops through encodeSingleToken
    std::vector<SDR> encode(const std::string& text);
    
    // The core function to encode one token
    SDR encodeSingleToken(int token_id);

    // New function to get token IDs, allowing the progress bar to be external
    std::vector<int> tokenize(const std::string& text) const;

    std::string decode(const std::vector<int>& ids) const;
    int getUnkId() const;

    int getVocabSize() const;
    std::string idToPiece(int id) const;

private:
    RDSEInstance token_rdse_;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor_;
};

#endif // TEXT_SDR_ENCODER_HPP