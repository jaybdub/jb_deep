cmake_minimum_required(VERSION 3.5)
project(jb_deep)

#add_library(jb_deep)

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR})


# JB_DEEP LIBRARY

# TESTS

add_executable(test_tensor test/test_tensor.cc)

add_executable(test_op test/test_op.cc)
