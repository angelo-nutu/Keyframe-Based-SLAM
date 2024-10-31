cmake_minimum_required(VERSION 3.10)

# Project name
project(ZeroMQExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the ZeroMQ package
find_package(cppzmq REQUIRED)

# Include directories for ZeroMQ
include_directories(${ZEROMQ_INCLUDE_DIRS})

# Source files
set(SOURCES
    example.cpp
)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the ZeroMQ library
target_link_libraries(${PROJECT_NAME}
    cppzmq
)
