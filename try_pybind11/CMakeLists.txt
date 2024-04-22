cmake_minimum_required(VERSION 3.16)
project(try_pybind11)

set(PYTHON_EXECUTABLE "/root/autodl-tmp/anaconda3/envs/nope/bin/python3.9")
set(PYTHON_INCLUDE_DIRECTORY "/root/autodl-tmp/anaconda3/envs/nope/include/python3.9")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_subdirectory(extern/pybind11)
pybind11_add_module(try_pybind11 main.cpp)
target_link_libraries(try_pybind11 PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})

