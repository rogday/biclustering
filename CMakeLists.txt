cmake_minimum_required (VERSION 3.10)

project (biclustering)

set(CMAKE_CXX_STANDARD 17)

set(TARGET_SOURCES biclustering.cpp)
add_executable(biclustering ${TARGET_SOURCES})