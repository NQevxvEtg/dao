// src/rdse.cpp
#include "rdse.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

RDSEInstance create_rdse(int n, int w, double resolution, int seed) {
    RDSEInstance instance;
    instance.size = n;
    instance.active_bits = w;
    instance.resolution = resolution;
    
    std::mt19937 gen;
    if (seed != -1) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    std::uniform_real_distribution<> distrib(0.0, resolution);
    
    instance.prototypes.reserve(n);
    for (int i = 0; i < n; ++i) {
        instance.prototypes.push_back(distrib(gen));
    }
    
    return instance;
}

SDR encode_scalar(const RDSEInstance& rdse_instance, double value) {
    if (rdse_instance.active_bits > rdse_instance.size) {
        // Handle error or adjust, for now, we assume valid input
    }
    
    std::vector<double> distances(rdse_instance.size);
    for(int i = 0; i < rdse_instance.size; ++i) {
        distances[i] = std::abs(rdse_instance.prototypes[i] - value);
    }
    
    std::vector<int> indices(rdse_instance.size);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

    // Partially sort to find the indices of the `active_bits` smallest distances
    std::partial_sort(indices.begin(), indices.begin() + rdse_instance.active_bits, indices.end(),
                      [&distances](int a, int b) {
                          return distances[a] < distances[b];
                      });
                      
    SDR sdr(rdse_instance.size, 0);
    for (int i = 0; i < rdse_instance.active_bits; ++i) {
        sdr[indices[i]] = 1;
    }
    
    return sdr;
}

int overlap(const SDR& sdr1, const SDR& sdr2) {
    int sum = 0;
    // Assuming sdr1 and sdr2 are the same size
    for (size_t i = 0; i < sdr1.size(); ++i) {
        sum += sdr1[i] * sdr2[i];
    }
    return sum;
}