cmake_minimum_required(VERSION 3.17.0)
cmake_policy(VERSION 3.6...3.17.0)

INCLUDE(CheckCSourceCompiles)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDAToolkit_ROOT /usr/bin)

project(glb VERSION 0.1 LANGUAGES C CXX CUDA )

# Check for restrict keyword
# Builds the macro A_C_RESTRICT form automake
foreach(ac_kw __restrict __restrict__ _Restrict restrict)
  check_c_source_compiles(
  "
  typedef int * int_ptr;
  int foo (int_ptr ${ac_kw} ip) {
    return ip[0];
  }
  int main(){
    int s[1];
    int * ${ac_kw} t = s;
    t[0] = 0;
    return foo(t);
  }
  "
  RESTRICT)
  if(RESTRICT)
    set(ac_cv_c_restrict ${ac_kw})
    break()
  endif()
endforeach()
if(RESTRICT)
  add_definitions("-Drestrict=${ac_cv_c_restrict}")
else()
  add_definitions("-Drestrict=")
endif()

find_package(MPI REQUIRED)
#find_package(FFTW3 REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Set architecture flag
add_subdirectory(blitz)
add_subdirectory(mdl2)

set(CMAKE_C_FLAGS -fopenmp)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Release build with debug info selected")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

#set(CMAKE_C_FLAGS -DBZ_DEBUG)
#add_executable(
#     CMAKE_C_FLAGS   glb
#        src/glb.cpp
#        src/orb.cpp
#        src/comm/MPIMessaging.cpp
#        src/services/countService.cpp
#        src/services/countLeftService.cpp
#        src/services/localReshuffleService.cpp
#        src/services/buildTreeService.cpp)
#        analysis/cpu-reduce.cu/main.cpp)
#set_target_properties(glb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(glb ${MPI_LIBRARIES} blitz)

add_executable(
        orbit
        src/utils/condReduce.cu
        src/orbit.cpp
        src/services/setadd.cpp
        src/services/TraversePST.cpp
        src/services/countLeft.cpp
        src/services/count.cpp
        src/services/init.cpp
        src/services/init.cu
        src/services/copyToDevice.cu
        src/services/countLeft.cu
        src/services/reshuffle.cpp
        src/services/finalize.cu
        src/services/makeAxis.cpp
        src/constants.h
        src/data/tipsy.h)

target_include_directories(orbit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(orbit PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(orbit mdl2 blitz)
