cmake_minimum_required(VERSION 3.10)

project(Odometry VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(external/yaml-cpp) 
add_subdirectory(external/raylib) 

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

add_library(lib
    src/lib/config.cpp
    src/lib/vo.cpp
)

target_include_directories(lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/lib
    ${OpenCV_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIR}
    external/yaml-cpp/include
    external/raylib/src
)

add_executable(odometry
    src/main.cpp
)

target_link_libraries(odometry
    PRIVATE
    lib
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
    yaml-cpp
    raylib
)

