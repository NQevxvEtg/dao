// src/grid_cell_encoder.cpp
#include "grid_cell_encoder.hpp"
#include <stdexcept>

GridCellEncoder::GridCellEncoder(int sdr_size, int sdr_active_bits)
    : sdr_size_(sdr_size), sdr_active_bits_(sdr_active_bits) {
    // Constructor initializes member variables
}

void GridCellEncoder::addModule(double resolution, int seed) {
    // Each module gets half the active bits for its X and Y encoders
    int module_active_bits = sdr_active_bits_ / 2;
    
    RDSEInstance x_rdse = create_rdse(sdr_size_, module_active_bits, resolution, seed);
    RDSEInstance y_rdse = create_rdse(sdr_size_, module_active_bits, resolution, seed + 1);
    
    modules_.push_back({x_rdse, y_rdse});
}

SDR GridCellEncoder::encode(const std::vector<double>& coordinates) {
    if (coordinates.size() != 2) {
        throw std::invalid_argument("Coordinates must be a 2D vector [x, y].");
    }

    SDR composite_sdr(sdr_size_, 0);

    for (const auto& module : modules_) {
        // Encode the x and y coordinates separately
        SDR sdr_x = encode_scalar(module.x_rdse, coordinates[0]);
        SDR sdr_y = encode_scalar(module.y_rdse, coordinates[1]);

        // Combine the module's SDRs with the composite SDR using a bitwise OR
        for (int i = 0; i < sdr_size_; ++i) {
            composite_sdr[i] = composite_sdr[i] | sdr_x[i] | sdr_y[i];
        }
    }

    return composite_sdr;
}