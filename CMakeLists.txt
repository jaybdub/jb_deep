cmake_minimum_required(VERSION 3.5)
project(jb_deep)

#add_library(jb_deep)

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR})


# JB_DEEP LIBRARY

add_library(jb_deep src/tensor.cc)

# TESTS

add_executable(test_tensor test/test_tensor.cc)
target_link_libraries(test_tensor jb_deep)
