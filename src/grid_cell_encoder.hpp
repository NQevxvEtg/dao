// src/grid_cell_encoder.hpp
#ifndef GRID_CELL_ENCODER_HPP
#define GRID_CELL_ENCODER_HPP

#include "rdse.hpp"
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

struct GridModule {
    RDSEInstance x_rdse;
    RDSEInstance y_rdse;
    template <class Archive> void serialize(Archive & ar) { ar(CEREAL_NVP(x_rdse), CEREAL_NVP(y_rdse)); }
};

class GridCellEncoder {
public:
    GridCellEncoder() = default; // Default constructor for Cereal
    GridCellEncoder(int sdr_size, int sdr_active_bits);

    void addModule(double resolution, int seed);
    SDR encode(const std::vector<double>& coordinates);

private:
    friend class cereal::access;
    template <class Archive>
    void serialize(Archive & ar) {
        ar(CEREAL_NVP(sdr_size_), CEREAL_NVP(sdr_active_bits_), CEREAL_NVP(modules_));
    }

    int sdr_size_;
    int sdr_active_bits_;
    std::vector<GridModule> modules_;
};

#endif // GRID_CELL_ENCODER_HPP