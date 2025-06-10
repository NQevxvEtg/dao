// src/spatial_pooler.cpp
#include "spatial_pooler.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

SpatialPooler::SpatialPooler(int input_size, int num_columns, int layer_index, float potential_ratio,
                             float syn_perm_active_inc, float syn_perm_inactive_dec,
                             float syn_perm_connected, int num_active_cols_per_inhib,
                             int stimulus_threshold, int boost_strength)
    : _input_size(input_size), _num_columns(num_columns), _layerIndex(layer_index),
      _permanences(num_columns, input_size), _boost_factors(num_columns),
      _active_duty_cycle(num_columns), _overlap_duty_cycle(num_columns),
      _syn_perm_active_inc(syn_perm_active_inc),
      _syn_perm_inactive_dec(syn_perm_inactive_dec),
      _syn_perm_connected(syn_perm_connected),
      _num_active_cols_per_inhib(num_active_cols_per_inhib),
      _stimulus_threshold(stimulus_threshold),
      _boost_strength(boost_strength), _plasticity_enabled(true),
      _gen(std::random_device{}()) {

    initializePermanences(potential_ratio);
    _boost_factors.setOnes();
    _active_duty_cycle.setZero();
    _overlap_duty_cycle.setZero();
}

void SpatialPooler::initializePermanences(float potential_ratio) {
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < _num_columns; ++i) {
        for (int j = 0; j < _input_size; ++j) {
            if (dist(_gen) < potential_ratio) {
                _permanences(i, j) = std::uniform_real_distribution<float>(
                    _syn_perm_connected - 0.1f, _syn_perm_connected + 0.1f)(_gen);
            } else {
                _permanences(i, j) = 0.0f;
            }
        }
    }
}

int SpatialPooler::getLayerIndex() const {
    return _layerIndex;
}

void SpatialPooler::enablePlasticity(float active_inc, float inactive_dec) {
    _plasticity_enabled = true;
    _syn_perm_active_inc = active_inc;
    _syn_perm_inactive_dec = inactive_dec;
}

void SpatialPooler::disablePlasticity() {
    _plasticity_enabled = false;
}

VectorXf SpatialPooler::calculateOverlap(const SDR& input_sdr) {
    VectorXf input_sdr_vector(_input_size);
    for (size_t i = 0; i < input_sdr.size(); ++i) {
        input_sdr_vector(i) = (input_sdr[i] > 0) ? 1.0f : 0.0f;
    }

    VectorXf overlaps = _permanences * input_sdr_vector;
    overlaps = overlaps.array() * _boost_factors.array();
    return overlaps;
}

std::vector<int> SpatialPooler::getActiveColumns(const VectorXf& overlaps) {
    std::vector<int> active_columns;
    std::vector<std::pair<float, int>> sorted_overlaps;
    for (int i = 0; i < _num_columns; ++i) {
        if (overlaps(i) > _stimulus_threshold) {
            sorted_overlaps.push_back({-overlaps(i), i});
        }
    }
    std::sort(sorted_overlaps.begin(), sorted_overlaps.end());
    int count = std::min((int)sorted_overlaps.size(), _num_active_cols_per_inhib);

    for (int i = 0; i < count; ++i) {
        active_columns.push_back(sorted_overlaps[i].second);
    }
    return active_columns;
}

void SpatialPooler::updatePermanences(const SDR& input_sdr, const std::vector<int>& active_columns) {
    if (!_plasticity_enabled) return;

    for (int col_idx : active_columns) {
        for (int i = 0; i < _input_size; ++i) {
            if (input_sdr[i] > 0) {
                _permanences(col_idx, i) += _syn_perm_active_inc;
                _permanences(col_idx, i) = std::min(1.0f, _permanences(col_idx, i));
            } else {
                _permanences(col_idx, i) -= _syn_perm_inactive_dec;
                _permanences(col_idx, i) = std::max(0.0f, _permanences(col_idx, i));
            }
        }
    }
}

void SpatialPooler::boostColumns(const std::vector<int>& active_columns) {
    // Simplified boosting logic for now
}

SDR SpatialPooler::process(const SDR& input_sdr, bool learn) {
    VectorXf overlaps = calculateOverlap(input_sdr);
    std::vector<int> active_columns_indices = getActiveColumns(overlaps);
    if (learn) {
        updatePermanences(input_sdr, active_columns_indices);
        boostColumns(active_columns_indices);
    }

    SDR output_sdr(_num_columns, 0);
    for (int idx : active_columns_indices) {
        output_sdr[idx] = 1;
    }
    return output_sdr;
}

int SpatialPooler::getNumColumns() const {
    return _num_columns;
}