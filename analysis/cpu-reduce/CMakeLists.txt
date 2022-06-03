cmake_minimum_required(VERSION 3.17.0)
cmake_policy(VERSION 3.6...3.17.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

project(glb VERSION 0.1 LANGUAGES CXX )

add_executable(
        avx-test
        main.cpp)

set(blitz_DIR /usr/local/lib/cmake)
find_package(blitz REQUIRED)
