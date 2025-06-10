// src/emotion.hpp
#ifndef EMOTION_HPP
#define EMOTION_HPP

struct EmotionConfig {
    // Decoding parameters for sampling
    float temp = 0.7f;
    int top_k = 40;
    
    // [SLLM ADDED] Penalty for repeating tokens. 1.0 means no penalty.
    // Higher values discourage repetition more strongly.
    float repetition_penalty = 1.2f;
    int repetition_lookback = 30; // How many recent tokens to consider for the penalty.

    // --- FINAL SYNTHESIS: A model of Stillness and Harmony ---
    float global_weight = 0.0073f;
    float yin_resonance_weight = 0.618f;
    float yin_dissonance_weight = 0.382f;
    float yang_weight = 1.0f;
};

#endif // EMOTION_HPP