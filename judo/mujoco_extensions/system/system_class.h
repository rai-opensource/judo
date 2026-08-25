// Copyright (c) 2024-2025 Boston Dynamics AI Institute LLC. All rights reserved.
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <unsupported/Eigen/CXX11/Tensor>

#include "onnx_interface/onnx_interface.h"
#include "system/system_utils.h"

namespace SystemClass {

const double kInfiniteTime = std::numeric_limits<double>::infinity();

using EigenTypes::MatrixT;
using EigenTypes::MatrixTList;
using EigenTypes::VectorT;
using EigenTypes::VectorTList;

class System {
 public:
  mjModel* model;
  mjData* data;

  OnnxInterface::Policy policy;

  const std::string model_filepath;
  const std::string policy_filepath;

  Eigen::PermutationMatrix<12> orbit_to_mujoco_legs;
  Eigen::PermutationMatrix<12> mujoco_to_orbit_legs;
  Eigen::PermutationMatrix<19> orbit_to_mujoco;
  Eigen::PermutationMatrix<19> mujoco_to_orbit;

  VectorT observation;
  VectorT policy_output;

  int base_qpos_start_idx;  // base position address
  int base_qvel_start_idx;  // base velocity address
  int leg_qpos_start_idx;   // leg position address
  int leg_qvel_start_idx;   // leg velocity address

  System(const std::string& model_filepath_, const std::string& policy_filepath_);
  System(const std::string& policy_filepath_, const mjModel* reference_model,
         std::shared_ptr<OnnxInterface::Session> reference_session);
  ~System();

  /** @brief  Resets the Mujoco data and the sensor data
   *
   * @param reset_last_output whether or not to reset the state's last output
   */
  void reset(const bool reset_last_output = true);

  /** @brief  Loads a inference policy from a file
   *
   * Requires the policy_filepath be defined before calling
   *
   * @param policy_filepath filepath to the policy model
   */
  void loadPolicy(const std::string& policy_filepath_, std::shared_ptr<OnnxInterface::Session> reference_session);

  /** @brief Sets the observation vector of a system given a command
   *
   * @param command VectorT containing a command to the system
   */
  void setObservation(const VectorT& command);

  /** @brief Calls the inference on the policy of the system
   *
   * Operates based on the current observation of the system. Requires that loadPolicy() has been called
   */
  void policyInference();

  void zeroOutVectors();

  /** @brief Retrieves the last control vector from the System
   *
   * @returns A vector containing the control input
   */
  VectorT getControl();

  /** @brief Retrieves the current state vector from the System
   *
   * @returns A vector containing the current state
   */
  VectorT getState();

  /**
   * @brief Roll out the neural network policy on a MuJoCo model over a series of commands.
   *
   * This function simulates the system's dynamics and executes the neural network policy
   * on the provided initial state and sequence of commands. The function returns the resulting
   * states, sensor readings, and accumulated reward.
   *
   * This differs from the SystemUtils::physicsRollout function in that internal mujoco data is not updated here.
   *
   * @param state The initial state vector of the system, typically including positions and velocities.
   * @param command A matrix where each row represents a command to be applied to the system.
   *                The number of rows corresponds to the number of commands, and the columns
   *                represent the command parameters.
   * @param physics_substeps The number of substeps to take in the MuJoCo physics engine for each command.
   *
   * Example:
   * - Mujoco physics timestep = 0.01 -> 100Hz physics
   * - physics_substeps = 2 -> low-level control period = 0.01 * 2 = 0.02 -> 50Hz low-level control
   * - num_commands = 10 -> total simulated period = 0.01 * 100 = 1 second
   *
   * @return A tuple containing:
   *         - A `MatrixT` of states, where each row represents the state of the system (positions and velocities) at
   * each time step.
   *         - A `MatrixT` of sensor readings, where each row represents the sensor data at each time step.
   *         - A `double` representing the accumulated reward over the rollout.
   */
  std::tuple<MatrixT, MatrixT> rollout(const VectorT& state, const MatrixT& command, const int physics_substeps = 2,
                                       const bool reset_last_output = true, const double cutoff_time = kInfiniteTime);

 private:
  VectorT policy_input_;

  VectorT control_;
  VectorT default_joint_pos_;
  VectorT joint_pos_;
  VectorT joint_vel_;

  void setStateIndices();
  void initializeSystemIndices();
};

/** @brief Creates a bunch of systems from a configuration file
 *
 * @param model_filepath a filepath to the Mujoco model for the system
 * @param policy_filepath a file path to the control policy for the system
 * @param num_systems how many systems to create

 * @returns A vector of shared pointers pointing to systems
*/
std::vector<std::shared_ptr<SystemClass::System>> create_systems_vector(const mjModel* reference_model,
                                                                        const std::string& policy_filepath,
                                                                        const int num_systems);

/** @brief Rolls out a simulation for a set of systems
 *
 * @param systems a vector of shared pointers to different systems
 * @param states a vector of initial starting states, one for each system
 * @param command a vector containing a command input for each system
 * @param num_threads the number of threads we allocate for the rollout
 * @param physics_substeps how many substeps to take along the rollout

 * @returns A vector of {states, sensors, reward}
*/
std::tuple<MatrixTList, MatrixTList, VectorTList> threadedRollout(const std::vector<std::shared_ptr<System>>& systems,
                                                                  const VectorTList& states, const MatrixTList& command,
                                                                  const VectorTList& last_policy_output,
                                                                  const int num_threads, const int physics_substeps,
                                                                  const double cutoff_time = kInfiniteTime);
}  // namespace SystemClass
