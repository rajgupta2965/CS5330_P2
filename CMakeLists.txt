cmake_minimum_required(VERSION 3.20)
project(Project2)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_executable(Project2 main.cpp)

target_link_libraries(Project2 ${OpenCV_LIBS})
