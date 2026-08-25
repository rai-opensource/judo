/* Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved. */
#include <mujoco/mujoco.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "system/system_class.h"
#include "system/system_utils.h"

#include <memory>
#include <vector>

namespace mujoco_extensions::pybind::policy_rollout {

namespace py = pybind11;

/**
 * Setup openmp environment.
 */
constexpr std::size_t OMP_NUM_THREADS{8};

using EigenTypes::MatrixTList;
using EigenTypes::VectorT;
using EigenTypes::VectorTList;

/**
 * Bindings for `policy_rollout` submodule.
 */
void bindPolicyRollout(const std::reference_wrapper<py::module>& root) {
  using pybind11::literals::operator""_a;

  Eigen::setNbThreads(OMP_NUM_THREADS);
  // Create `policy_rollout` submodule.
  auto python_module = root.get().def_submodule("policy_rollout");

  python_module.def(
      "set_state",
      [](std::shared_ptr<SystemClass::System> systemObject, const VectorT& state) -> void {
        SystemUtils::setState(systemObject->model, systemObject->data, state);
      },
      "Sets the state of a system object", py::arg("system"), py::arg("state"));

  // SystemClass
  py::class_<SystemClass::System, std::shared_ptr<SystemClass::System>>(python_module, "System")
      .def(py::init<const std::string&, const std::string&>(), "model_filepath"_a, "policy_filepath"_a)
      .def_readwrite("observation", &SystemClass::System::observation)
      .def_property(
          "policy_output",
          [](SystemClass::System& system) {
            return system.policy_output;
          },
          [](SystemClass::System& system, const VectorT& value) {
            system.policy_output.resize(value.size());
            std::copy(value.begin(), value.end(), system.policy_output.begin());
          })
      .def("reset", &SystemClass::System::reset)
      .def("load_policy", &SystemClass::System::loadPolicy)
      .def("set_observation", &SystemClass::System::setObservation, "command"_a)
      .def("policy_inference", &SystemClass::System::policyInference)
      .def("get_control", &SystemClass::System::getControl)
      .def("get_state", &SystemClass::System::getState)
      .def("rollout", &SystemClass::System::rollout, "state"_a, "command"_a, "physics_substeps"_a = 2,
           "reset_last_output"_a = true, "cutoff_time"_a = SystemClass::kInfiniteTime);  // Match C++

  python_module.def("threaded_rollout", &SystemClass::threadedRollout,
                    "Threaded policy rollout with shared pointers to System objects.", "systems"_a, "states"_a,
                    "command"_a, "last_policy_output"_a, "num_threads"_a, "physics_substeps"_a,
                    "cutoff_time"_a = SystemClass::kInfiniteTime);

  python_module.def(
      "create_systems_vector",
      [](const py::object& python_model, const std::string& policy_filepath,
         const int num_systems) -> std::vector<std::shared_ptr<SystemClass::System>> {
        const auto modelPointer = python_model.attr("_address").cast<std::uintptr_t>();
        const mjModel* reference_model = reinterpret_cast<const mjModel*>(modelPointer);

        return SystemClass::create_systems_vector(reference_model, policy_filepath, num_systems);
      },
      "model"_a, "policy_filepath"_a, "num_systems"_a);
}

}  // namespace mujoco_extensions::pybind::policy_rollout
