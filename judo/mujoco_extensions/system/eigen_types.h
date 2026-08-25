// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#pragma once

#include <vector>

#include <Eigen/Dense>

namespace EigenTypes {
using VectorT = Eigen::VectorXd;
using MatrixT = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorTList = std::vector<VectorT>;
using MatrixTList = std::vector<MatrixT>;

}  // namespace EigenTypes
