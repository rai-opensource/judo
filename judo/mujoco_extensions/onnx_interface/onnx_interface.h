// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <onnxruntime_cxx_api.h>

namespace OnnxInterface {
// ONNX at the moment only supports floats :(
using VectorT = Eigen::VectorXf;
using MatrixT = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Session = Ort::Session;

class Policy {
 public:
  std::string policy_path;

  std::shared_ptr<Ort::Session> session;

  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int64_t input_size;
  int64_t output_size;
  size_t num_input_nodes;

  /** @brief  Empty initializer that creates a policy. Used for no particular reason */
  Policy();

  /** @brief Creates a policy from an ONNX file
   *
   * This constructor creates an Ort environment and session that is stored inside its session pointer.
   *
   * @param policy_path Path to the ONNX policy file
   */
  explicit Policy(const std::string& policy_path);

  /** @brief Creates a policy from an ONNX file
   *
   * This constructor creates an Ort environment and session that is stored inside its session pointer.
   *
   * @param policy_path Path to the ONNX policy file
   * @param session shared pointer to an existing Ort::Session that is allocated in a different location.
   */
  Policy(const std::string& policy_path, std::shared_ptr<Ort::Session> session_);

  ~Policy();

  /** @brief Equality or copy operation from an existing policy
   *
   * Copies the shared pointer of the other policy and policy path, then initializes the tensor data for this policy.
   *
   * @param policy An existing OnnxInterface policy
   */
  Policy& operator=(const Policy& policy);

  /** @brief Returns the input dimensions of the Onnx policy */
  std::vector<int64_t> getInputShape();

  /** @brief Returns the output dimensions of the Onnx policy */
  std::vector<int64_t> getOutputShape();

  /** @brief Returns the name of the input to the Onnx policy */
  std::string getInputName();

  /** @brief Returns the name of the output from the Onnx policy */
  std::string getOutputName();

  /** @brief Creates a tensor  */
  Ort::Value createTensorFromVector(VectorT* input_vector, const std::vector<int64_t>& vector_shape);
  Ort::Value createZeroTensor(int64_t tensor_length, const std::vector<int64_t>& tensor_shape);

  VectorT policyInference(VectorT* input_vector);

 private:
  std::string input_name_;
  std::string output_name_;

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;

  Ort::Value input_tensor_;

  Ort::MemoryInfo memory_info_;
  Ort::RunOptions run_options_;

  void initializePolicy();
};

std::shared_ptr<Ort::Session> allocateOrtSession(const std::string& policy_filepath);

std::shared_ptr<Ort::Session> getNullSession();

}  // namespace OnnxInterface
