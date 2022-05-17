cmake_minimum_required(VERSION 3.17.0)
cmake_policy(VERSION 3.6...3.17.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(glb VERSION 0.1 LANGUAGES CXX CUDA )

message(DEBUG ${MPI_INCLUDE_PATH})
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_FLAGS -fopenmp)
add_executable(glb src/glb.cpp src/IO.cpp src/orb.cpp)
target_link_libraries(glb ${MPI_LIBRARIES})

set(blitz_DIR /usr/local/lib/cmake)
find_package(blitz REQUIRED)


