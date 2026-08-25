// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

#include "system/system_utils.h"

namespace SystemUtils {

mjModel* loadModel(const std::string& model_filepath) {
  char error_buffer[1000] = "Could not load XML file";

  // Attempt to load the model from the XML file
  mjModel* model = mj_loadXML(model_filepath.c_str(), nullptr, error_buffer, sizeof(error_buffer));

  // Check if the model failed to load
  if (!model) {
    throw std::runtime_error("Failed to load model XML file: " + model_filepath + " - " + error_buffer);
  }
  return model;
}

void setState(const mjModel* model, mjData* data, const VectorT& state) {
  int nq = model->nq;
  int nv = model->nv;

  // Check that the state size is correct
  if (state.size() != nq + nv) {
    std::cerr << "Error: State size (" << state.size() << ") does not match "
              << "the expected size (" << nq + nv << ")." << std::endl;
    throw std::runtime_error("State size mismatch");
  }

  for (int i = 0; i < nq; ++i) {
    data->qpos[i] = state[i];
  }
  for (int i = 0; i < nv; ++i) {
    data->qvel[i] = state[nq + i];
  }
}

}  // namespace SystemUtils
