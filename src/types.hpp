// src/types.hpp
#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Dense>
#include <vector>

// Define all common types here for consistency across the project.
using SDR = std::vector<int8_t>;

// Using RowMajor is often better for matrix multiplication performance in this context.
using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXf = Eigen::VectorXf;

#endif // TYPES_HPP