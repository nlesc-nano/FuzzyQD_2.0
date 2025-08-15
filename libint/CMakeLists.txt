cmake_minimum_required(VERSION 3.16)
project(libint_fuzzy LANGUAGES CXX)

# -------------------- Build type & C++ standard ----------------------------
# If the caller didn't set a build type, default to Release for speed.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen: turn off runtime assertions; keep MPL2-only header mode
add_compile_definitions(EIGEN_MPL2_ONLY EIGEN_NO_DEBUG)

# Prefer shared (pybind11_add_module already sets PIC)
set(BUILD_SHARED_LIBS ON)

# -------------------- Conda prefix (your original) -------------------------
if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# -------------------- Dependencies -----------------------------------------
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3   CONFIG REQUIRED)
find_package(libint2  CONFIG REQUIRED)

# --- OpenMP (required for the tiled projector) ---
find_package(OpenMP REQUIRED)

# Optional: MKL / vendor BLAS acceleration for Eigen GEMMs.
# Enable with -DUSE_MKL=ON at configure time (or set USE_BLAS=ON and link OpenBLAS).
option(USE_MKL  "Use Intel MKL with Eigen for GEMM" OFF)
option(USE_BLAS "Use generic BLAS (e.g., OpenBLAS) with Eigen" OFF)

if(USE_MKL)
  # With oneAPI: 'cmake -DUSE_MKL=ON -DMKL_DIR=$MKLROOT ..'
  find_package(MKL REQUIRED)
  add_compile_definitions(EIGEN_USE_MKL_ALL)
  message(STATUS "Using MKL with Eigen (EIGEN_USE_MKL_ALL)")
elseif(USE_BLAS)
  find_package(BLAS REQUIRED)
  add_compile_definitions(EIGEN_USE_BLAS)
  message(STATUS "Using external BLAS with Eigen (EIGEN_USE_BLAS)")
endif()

# -------------------- Module target ----------------------------------------
pybind11_add_module(libint_fuzzy
  bindings.cpp
  integrals_core.cpp
)

# Core libs
target_link_libraries(libint_fuzzy PRIVATE
  Libint2::cxx
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

# Optional BLAS backends
if(USE_MKL)
  target_link_libraries(libint_fuzzy PRIVATE MKL::MKL)
elseif(USE_BLAS)
  target_link_libraries(libint_fuzzy PRIVATE BLAS::BLAS)
endif()

# -------------------- Optimization flags -----------------------------------
# Prefer explicit per-target flags so we don't affect other projects.
if(MSVC)
  # /O2 /arch:AVX2 (if available); adjust as needed
  target_compile_options(libint_fuzzy PRIVATE /O2 /DNDEBUG)
else()
  # -O3 + -march=native : best speed on local CPU
  # You can add -ffast-math if acceptable; I’m avoiding it by default for safety.
  target_compile_options(libint_fuzzy PRIVATE -O3 -march=native -DNDEBUG)
  # Linker gets -O3 too (occasionally helps LTO pipelines)
  target_link_options(libint_fuzzy PRIVATE -O3)
endif()

# -------------------- Nice-to-have diagnostics -----------------------------
# Show where Libint came from (your original line)
get_target_property(LIBINT2_LOC Libint2::int2 LOCATION)
message(STATUS "Libint2 location: ${LIBINT2_LOC}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")


