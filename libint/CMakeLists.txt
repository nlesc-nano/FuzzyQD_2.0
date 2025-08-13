cmake_minimum_required(VERSION 3.16)
project(libint_fuzzy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(EIGEN_MPL2_ONLY EIGEN_NO_DEBUG)
set(BUILD_SHARED_LIBS ON)

# Prefer the active conda environment for find_package
if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3   CONFIG REQUIRED)
find_package(libint2  CONFIG REQUIRED)

pybind11_add_module(libint_fuzzy
    bindings.cpp
    integrals_core.cpp
)

# Link against Libint’s exported target and Eigen
# Libint2::cxx brings correct compile defs and link flags
# Eigen3::Eigen is header-only but keep it explicit

target_link_libraries(libint_fuzzy PRIVATE
    Libint2::cxx
    Eigen3::Eigen
)

# Optional debug output to confirm the resolved shared library
get_target_property(LIBINT2_LOC Libint2::int2 LOCATION)
message(STATUS "Libint2 location: ${LIBINT2_LOC}")



