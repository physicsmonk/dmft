cmake_minimum_required(VERSION 3.16.1)

project(dimerHubbardMag
    LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER mpiicpc)

# Add the dmft library
add_subdirectory(dmft)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE dmft)
