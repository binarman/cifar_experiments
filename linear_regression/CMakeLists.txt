cmake_minimum_required(VERSION 3.10)

add_executable(regression linear_fit.cpp)
target_link_libraries(regression alglib)
