// src/rdse.hpp
#ifndef RDSE_HPP
#define RDSE_HPP
#include <vector>
#include <cstdint>
#include <random>
#include <cereal/cereal.hpp> // Include Cereal
#include <cereal/types/vector.hpp>


// Using a vector of 8-bit integers for the SDR, similar to numpy's int8 array
using SDR = std::vector<int8_t>;

// A struct to hold the RDSE parameters, replacing the Python dictionary
struct RDSEInstance {
    int size;
    int active_bits;
    double resolution;
    std::vector<double> prototypes;

    template <class Archive>
    void serialize(Archive & ar) {
        ar(CEREAL_NVP(size), CEREAL_NVP(active_bits), CEREAL_NVP(resolution), CEREAL_NVP(prototypes));
    }
};

/**
 * @brief Creates a Random Distributed Scalar Encoder (RDSE) instance.
 * @param n The total number of bits in the SDR.
 * @param w The number of active bits (the sparsity).
 * @param resolution The range of the scalar values.
 * @param seed An optional seed for the random number generator.
 * @return A fully configured RDSEInstance.
 */
RDSEInstance create_rdse(int n, int w, double resolution, int seed = -1);

/**
 * @brief Encodes a scalar value into a Sparse Distributed Representation (SDR).
 * @param rdse_instance The RDSE configuration to use.
 * @param value The scalar value to encode.
 * @return The resulting SDR.
 */
SDR encode_scalar(const RDSEInstance& rdse_instance, double value);

/**
 * @brief Computes the overlap (dot product) between two binary SDRs.
 * @param sdr1 The first SDR.
 * @param sdr2 The second SDR.
 * @return The integer overlap score.
 */
int overlap(const SDR& sdr1, const SDR& sdr2);

#endif // RDSE_HPP