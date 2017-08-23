cmake_minimum_required(VERSION 3.5)
project(jb_deep)

#add_library(jb_deep)

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(mnist mnist.cc)
