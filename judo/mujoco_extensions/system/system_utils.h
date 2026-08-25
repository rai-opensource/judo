// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#pragma once

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "system/eigen_types.h"

namespace SystemUtils {

using EigenTypes::VectorT;

/** @brief Loads a Mujoco model from an XML
 *
 * @param model_filepath filepath to the mujoco model
 */
mjModel* loadModel(const std::string& model_filepath);

/** @brief Set the state of the MuJoCo model.
 *
 * This function sets the positions and velocities of the MuJoCo model based on the provided state vector.
 *
 * @param model Pointer to the MuJoCo model.
 * @param data Pointer to the MuJoCo data.
 * @param state Vector containing the state (positions and velocities).
 */
void setState(const mjModel* model, mjData* data, const VectorT& state);

}  // namespace SystemUtils
