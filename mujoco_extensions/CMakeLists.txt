# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

cmake_minimum_required(VERSION 3.15)

# Set policy CMP0148 to NEW to use FindPython instead of deprecated FindPythonInterp/FindPythonLibs
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

project(mujoco_extensions)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use RPATH settings to avoid library path conflicts
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

# Find Python FIRST (before pybind11) to avoid deprecated FindPythonInterp
# Use CONDA_PREFIX to avoid picking up venv/system Python headers
if(DEFINED ENV{CONDA_PREFIX})
  set(Python_ROOT_DIR $ENV{CONDA_PREFIX})
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Tell pybind11 to use modern FindPython
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
  OUTPUT_VARIABLE MJ_PKG_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(GLOB MUJOCO_LIB_FILES "${MJ_PKG_ROOT}/libmujoco.so*" "${MJ_PKG_ROOT}/libmujoco.*.dylib")
if(MUJOCO_LIB_FILES)
  list(GET MUJOCO_LIB_FILES 0 MUJOCO_LIB)
else()
  message(FATAL_ERROR "MuJoCo library not found in ${MJ_PKG_ROOT}")
endif()
set(MJINC_DIR "${MJ_PKG_ROOT}/include")

# Find site-packages directory for output
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE SITE_PACKAGES_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find ONNX Runtime from conda/pixi environment
if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
  
  # Check for both Linux (.so) and macOS (.dylib) extensions
  file(GLOB ONNXRUNTIME_LIB_FILES 
       "${CONDA_PREFIX}/lib/libonnxruntime.so*"
       "${CONDA_PREFIX}/lib/libonnxruntime*.dylib") # Added for Mac
       
  if(ONNXRUNTIME_LIB_FILES)
    list(GET ONNXRUNTIME_LIB_FILES 0 ONNXRUNTIME_LIB)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
  else()
    message(FATAL_ERROR "ONNX Runtime library not found in ${CONDA_PREFIX}/lib")
  endif()

  # Resolve include directory across possible package layouts.
  # Common locations include:
  # - ${CONDA_PREFIX}/include
  # - ${CONDA_PREFIX}/include/onnxruntime
  # - ${CONDA_PREFIX}/include/onnxruntime/core/session
  find_path(ONNXRUNTIME_INCLUDE
    NAMES onnxruntime_cxx_api.h
    HINTS
      "${CONDA_PREFIX}/include"
      "${CONDA_PREFIX}/include/onnxruntime"
    PATH_SUFFIXES
      ""
      "onnxruntime"
      "onnxruntime/core/session"
  )

  if(NOT ONNXRUNTIME_INCLUDE)
    message(FATAL_ERROR
      "ONNX Runtime headers not found. Expected onnxruntime_cxx_api.h under ${CONDA_PREFIX}/include")
  endif()
else()
  message(FATAL_ERROR "CONDA_PREFIX not set. Please run within pixi environment.")
endif()

message(STATUS "MuJoCo library: ${MUJOCO_LIB}")
message(STATUS "MuJoCo include: ${MJINC_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE}")
message(STATUS "Site packages: ${SITE_PACKAGES_DIR}")

include_directories(
  ${CMAKE_SOURCE_DIR}
  ${MJINC_DIR}
  ${ONNXRUNTIME_INCLUDE}
)

# --- ONNX Interface Library ---
add_library(onnx_interface STATIC
  onnx_interface/onnx_interface.cpp
)
target_include_directories(onnx_interface PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${ONNXRUNTIME_INCLUDE}
)
target_link_libraries(onnx_interface PUBLIC
  Eigen3::Eigen
  ${ONNXRUNTIME_LIB}
)

# --- System Library ---
add_library(system STATIC
  system/system_class.cpp
  system/system_utils.cpp
)
target_include_directories(system PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${MJINC_DIR}
)
target_link_libraries(system PUBLIC
  Eigen3::Eigen
  ${MUJOCO_LIB}
  onnx_interface
)

# --- Policy Rollout pybind module ---
pybind11_add_module(policy_rollout_pybind
  policy_rollout/pybind/pybind.cpp
  policy_rollout/pybind/policy_rollout.cpp
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(policy_rollout_pybind PRIVATE
    -mtune=native
  )
else()
  target_compile_options(policy_rollout_pybind PRIVATE
    -fopenmp
    -mtune=native
  )
endif()
target_link_libraries(policy_rollout_pybind PRIVATE
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  system
  ${MUJOCO_LIB}
)
set_target_properties(policy_rollout_pybind PROPERTIES
  BUILD_RPATH "$ORIGIN/../../mujoco"
  INSTALL_RPATH "$ORIGIN/../../mujoco"
  BUILD_RPATH_USE_ORIGIN TRUE
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/policy_rollout"
)
