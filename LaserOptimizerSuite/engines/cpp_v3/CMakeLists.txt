cmake_minimum_required(VERSION 3.14)
project(LaserSpeedPlanner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if(NOT MSVC)
    add_compile_options(-O3 -march=native -fopenmp -ffast-math)
    link_libraries(-fopenmp)
endif()

include(FetchContent)

# Eigen (Math)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# fmt (Printing)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)

add_executable(laser_planner src/main.cpp)
target_include_directories(laser_planner PRIVATE ${eigen_SOURCE_DIR} src)
target_link_libraries(laser_planner PRIVATE fmt::fmt)
