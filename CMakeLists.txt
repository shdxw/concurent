cmake_minimum_required(VERSION 3.15)
project(concurent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")
execute_process(COMMAND pip install -r requirements.txt)
file(COPY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_BINARY_DIR})

add_executable(concurent main.cpp)