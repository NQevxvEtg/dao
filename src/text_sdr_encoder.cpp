// src/text_sdr_encoder.cpp
#include "text_sdr_encoder.hpp"
#include "sentencepiece_processor.h"
#include "progress_bar.hpp" 
#include <stdexcept>
#include <utility> 
#include <iostream>

TextSdrEncoder::TextSdrEncoder() = default;

TextSdrEncoder::TextSdrEncoder(const std::string& tokenizer_model_path, int sdr_size, int sdr_active_bits) {
    sp_processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = sp_processor_->Load(tokenizer_model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
    }
    
    token_rdse_ = create_rdse(sdr_size, sdr_active_bits, sp_processor_->GetPieceSize(), 42);
}

TextSdrEncoder::~TextSdrEncoder() = default;
TextSdrEncoder::TextSdrEncoder(TextSdrEncoder&& other) noexcept = default;
TextSdrEncoder& TextSdrEncoder::operator=(TextSdrEncoder&& other) noexcept = default;

// New tokenize implementation
std::vector<int> TextSdrEncoder::tokenize(const std::string& text) const {
    std::vector<int> ids;
    if (sp_processor_) {
        sp_processor_->Encode(text, &ids);
    }
    return ids;
}

std::vector<SDR> TextSdrEncoder::encode(const std::string& text) {
    std::vector<int> ids = this->tokenize(text);
    std::vector<SDR> sdr_sequence;
    sdr_sequence.reserve(ids.size());
    for (int id : ids) {
        sdr_sequence.push_back(encodeSingleToken(id));
    }
    return sdr_sequence;
}

SDR TextSdrEncoder::encodeSingleToken(int token_id) {
    return encode_scalar(token_rdse_, static_cast<double>(token_id));
}

std::string TextSdrEncoder::decode(const std::vector<int>& ids) const {
    if (!sp_processor_) return "";
    // Note: The SentencePiece API uses DecodeIds, not Decode
    std::string decoded_text;
    sp_processor_->Decode(ids, &decoded_text);
    return decoded_text;
}

int TextSdrEncoder::getUnkId() const {
    if (!sp_processor_) return 0;
    return sp_processor_->unk_id();
}

int TextSdrEncoder::getVocabSize() const {
    if (!sp_processor_) return 0;
    return sp_processor_->GetPieceSize();
}

std::string TextSdrEncoder::idToPiece(int id) const {
    if (!sp_processor_) return "";
    return sp_processor_->IdToPiece(id);
}