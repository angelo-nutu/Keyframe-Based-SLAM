cmake_minimum_required(VERSION 3.16)

project(Odometry VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Wextra -Werror -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -w -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(BUILD_SHARED_LIBS TRUE)

set(USER_HOME $ENV{HOME})
set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# canlib definitions
add_compile_definitions("CANLIB_PARKING=")
add_compile_definitions(CANLIB_TIMESTAMP)

find_package(absl)

if(absl_FOUND)
  set(absl_libraries
    absl::algorithm
    absl::base
    absl::bind_front
    absl::bits
    absl::btree
    absl::cleanup
    absl::cord
    absl::core_headers
    absl::debugging
    absl::dynamic_annotations
    absl::flags
    absl::flat_hash_map
    absl::flat_hash_set
    absl::function_ref
    absl::hash
    absl::layout
    absl::log_internal_check_op
    absl::log_severity
    absl::memory
    absl::node_hash_map
    absl::node_hash_set
    absl::optional
    absl::span
    absl::status
    absl::statusor
    absl::strings
    absl::synchronization
    absl::time
    absl::type_traits
    absl::utility
    absl::variant
  )
  
  foreach(iter_lib ${absl_libraries})
    string(REPLACE "absl::" "absl_" temp ${iter_lib})
    find_library(FOUND ${temp})
    if(FOUND)
    list(APPEND protobuf_ABSL_USED_TARGETS ${iter_lib})
    endif()
    unset(FOUND CACHE)
  endforeach()
endif()

find_package(CURL REQUIRED) # HTTP stuff
find_package(Protobuf REQUIRED)


include_directories(${PROTOBUF_INCLUDE_DIR})
add_subdirectory(${DIR}/protobuffer)

# Definition of a command that must run when ${proto_files} are modified
# In this way the generation of the protobuffer is automatic
add_custom_command(
  OUTPUT ${proto_cpp} ${proto_hpp}
  COMMAND ${DIR}/protobuffer/compile.sh
  DEPENDS ${proto_files}
  VERBATIM
  COMMENT "Generating protobuffer files"
)

# The command above will run only if some target depends on it
# Definition of a dummy target (proto_generation) to use it as a dependency for other targets
# The target (proto_generation) has as dependencies the .pb.cc and .pb.h files
# https://stackoverflow.com/questions/12913077/cmake-add-dependency-to-add-custom-command-dynamically
add_custom_target(proto_generation DEPENDS ${proto_cpp} ${proto_hpp})
add_library(proto SHARED ${proto_cpp})
target_link_libraries(proto PUBLIC ${Protobuf_LIBRARIES} ${protobuf_ABSL_USED_TARGETS})

# include directories created from proto generation
include_directories(
  ${DIR}

  ${DIR}/.generated/
  ${DIR}/.generated/protobuffer/
  ${DIR}/.generated/external/

  ${proto_folders}
)

# run subfolders CMakeLists.txt
add_subdirectory(${DIR}/external)

add_subdirectory(external/yaml-cpp) 
add_subdirectory(external/raylib) 

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)


include_directories(
  src/ 
  src/telemetry/
  ${LIBS_DIR} 
  ${EXTERNAL_DIRS}
)
  
add_library(
  telemetry
  src/telemetry/communication.cpp
)

target_link_libraries(telemetry PUBLIC serializers communication mqtt-topics ${OpenCV_LIBS})

add_library(lib
  src/lib/config.cpp
  src/lib/plot.cpp
  src/lib/cameraRealSense.cpp
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
  telemetry
)

target_link_libraries(lib PUBLIC telemetry)
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
    telemetry
)