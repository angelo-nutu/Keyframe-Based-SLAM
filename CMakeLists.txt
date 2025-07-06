cmake_minimum_required(VERSION 3.10)
project(VO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMMON_WARNINGS -Wshadow)
set(COMMON_DEBUG_FLAGS -g -Og -fsanitize=address)
set(COMMON_RELEASE_FLAGS -O3 -march=native -flto)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
include(FetchContent)
FetchContent_Declare(rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

add_subdirectory(external/as-serializers)

file(GLOB LIB_SRCS "src/lib/*.cpp")

add_library(VO_LIB STATIC ${LIB_SRCS})
target_link_libraries(VO_LIB
    PUBLIC as-serializers
           ${ZMQ_LIBRARIES}
           rerun_sdk
)

target_include_directories(VO_LIB
    PUBLIC include/lib
           ${OpenCV_INCLUDE_DIRS}
           ${ZMQ_INCLUDE_DIRS}
)

add_executable(vo src/executable/main.cpp)
target_link_libraries(vo
    PRIVATE VO_LIB
            ${OpenCV_LIBS}
            as-serializers
            ${ZMQ_LIBRARIES}
            rerun_sdk
            pthread
)

target_include_directories(vo
    PRIVATE ${OpenCV_INCLUDE_DIRS}
            ${ZMQ_INCLUDE_DIRS}
            ${EAGLE_AS_SERIALIZERS_LIB}
)

target_compile_options(VO_LIB PRIVATE
    ${COMMON_WARNINGS}
    $<$<CONFIG:Debug>:${COMMON_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${COMMON_RELEASE_FLAGS}>
)

target_compile_options(vo PRIVATE
    ${COMMON_WARNINGS}
    $<$<CONFIG:Debug>:${COMMON_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${COMMON_RELEASE_FLAGS}>
)

target_compile_definitions(vo PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

target_link_options(vo PRIVATE
    -fsanitize=address
)

target_link_options(VO_LIB PRIVATE
    -fsanitize=address
)

target_compile_definitions(VO_LIB PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

install(TARGETS vo RUNTIME DESTINATION bin)
install(TARGETS VO_LIB ARCHIVE DESTINATION lib) 
install(DIRECTORY include/ DESTINATION include)
