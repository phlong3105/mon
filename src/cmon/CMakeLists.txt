cmake_minimum_required(VERSION 3.27)
project(cmon)

set(CMAKE_CXX_STANDARD 23)

add_library(cmon SHARED library.cpp)
