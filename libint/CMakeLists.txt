cmake_minimum_required(VERSION 3.16)
project(libint_fuzzy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(EIGEN_MPL2_ONLY EIGEN_NO_DEBUG)

# Force dynamic libraries (prefer .dylib over .a)
set(BUILD_SHARED_LIBS ON)

# Always use the conda environment's install location
# (set this before any find_package)
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

# --- CORRECTED LINE ---
# Use the correct imported target 'Libint2::cxx' which includes all dependencies.
target_link_libraries(libint_fuzzy PRIVATE
    Libint2::cxx
    Eigen3::Eigen
)

# --- CORRECTED LINE ---
# Also use the correct target name to get the library location for debugging.
# We query the actual shared library component 'Libint2::int2'.
get_target_property(LIBINT2_LOC Libint2::int2 LOCATION)
message(STATUS "Libint2 location: ${LIBINT2_LOC}")

