cmake_minimum_required(VERSION 3.17.0)
cmake_policy(VERSION 3.6...3.17.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(glb VERSION 0.1 LANGUAGES CXX CUDA )

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_FLAGS -fopenmp)
add_executable(
        glb
        src/glb.cpp
        src/orb.cpp
        src/comm/MPIMessaging.cpp
        src/services/countService.cpp
        src/services/countLeftService.cpp
        src/services/localReshuffleService.cpp
        src/services/buildTreeService.cpp
        analysis/iter-performance/main.cpp)
set_target_properties(glb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(glb ${MPI_LIBRARIES})

set(blitz_DIR /usr/local/lib/cmake)
find_package(blitz REQUIRED)
