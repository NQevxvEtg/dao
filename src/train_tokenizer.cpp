// src/train_tokenizer.cpp
#include <sentencepiece_trainer.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_corpus_file>" << std::endl;
        return 1;
    }

    std::string corpus_path = argv[1];
    std::ifstream t(corpus_path);
    if (!t) {
        std::cerr << "Error opening corpus file: " << corpus_path << std::endl;
        return 1;
    }
    std::string corpus_data((std::istreambuf_iterator<char>(t)),
                             std::istreambuf_iterator<char>());

    // Standardized model name
    const std::string model_prefix = "tokenizer";
    const std::string model_filename = model_prefix + ".model";

    // SentencePiece Trainer arguments
    // Note: Adjust vocab_size, model_type, etc. as needed for your specific corpus
    const std::string sp_args =
        "--input=" + corpus_path + " " +
        "--model_prefix=" + model_prefix + " " +
        "--vocab_size=992 " +
        "--character_coverage=1.0 " +
        "--model_type=bpe";

    // Train the model
    const auto status = sentencepiece::SentencePieceTrainer::Train(sp_args);

    if (!status.ok()) {
        std::cerr << "SentencePiece training failed: " << status.ToString() << std::endl;
        return 1;
    }

    std::cout << "SentencePiece model trained successfully." << std::endl;
    std::cout << " -> Model file: " << model_filename << std::endl;
    std::cout << " -> Vocab file: " << model_prefix << ".vocab" << std::endl;

    return 0;
}