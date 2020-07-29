cmake_minimum_required(VERSION 3.5)
project(ID_card)

set(CMAKE_CXX_STANDARD 11)

add_executable(ID_card main.cpp
#        IDcard.cpp IDcard.h
        IDcard2.cpp IDcard2.h MyLayer.hpp)

find_package(OpenCV)
include_directories(${opencv_INCLUDE_DIRS})
target_link_libraries(ID_card ${OpenCV_LIBS})