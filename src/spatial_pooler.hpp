// src/spatial_pooler.hpp
#ifndef SPATIAL_POOLER_HPP
#define SPATIAL_POOLER_HPP

#include "types.hpp" // <-- ADDED
#include <random>
#include "progress_bar.hpp"

class SpatialPooler {
public:
    SpatialPooler() = default;
    SpatialPooler(int input_size, int num_columns, int layer_index, float potential_ratio = 0.5f,
                  float syn_perm_active_inc = 0.01f, float syn_perm_inactive_dec = 0.005f,
                  float syn_perm_connected = 0.5f, int num_active_cols_per_inhib = 10,
                  int stimulus_threshold = 5, int boost_strength = 1);

    SDR process(const SDR& input_sdr, bool learn);
    int getNumColumns() const;
    int getLayerIndex() const;
    void enablePlasticity(float active_inc, float inactive_dec);
    void disablePlasticity();
    
    template<class Archive>
    void serialize(Archive& ar) {
        ar(_input_size, _num_columns, _layerIndex, _permanences, _boost_factors, _active_duty_cycle, _overlap_duty_cycle);
    }

private:
    void initializePermanences(float potential_ratio);
    VectorXf calculateOverlap(const SDR& input_sdr);
    std::vector<int> getActiveColumns(const VectorXf& overlaps);
    void updatePermanences(const SDR& input_sdr, const std::vector<int>& active_columns);
    void boostColumns(const std::vector<int>& active_columns);

    int _input_size;
    int _num_columns;
    int _layerIndex;
    MatrixXf _permanences;
    VectorXf _boost_factors;
    VectorXf _active_duty_cycle;
    VectorXf _overlap_duty_cycle;
    float _syn_perm_active_inc;
    float _syn_perm_inactive_dec;
    float _syn_perm_connected;
    int _num_active_cols_per_inhib;
    int _stimulus_threshold;
    int _boost_strength;
    bool _plasticity_enabled;
    std::mt19937 _gen;
};

#endif // SPATIAL_POOLER_HPP