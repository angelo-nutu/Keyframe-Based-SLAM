message("[External]")

# define helper variable (CDIR is Current DIRectoty)
set(CDIR ${CMAKE_CURRENT_LIST_DIR})

set(EXTERNAL_DIRS
  ${CDIR}
  ${CDIR}/serializers/inc
  ${CDIR}/lib-communication/inc
  ${CDIR}/mqtt-topics/inc
)
include_directories(PUBLIC ${EXTERNAL_DIRS})

# --- LIB COMMUNICATION --- #
add_subdirectory(${CDIR}/lib-communication)
include_directories(${communication_SOURCE_DIRS})
target_compile_options(communication PRIVATE -w) # disable warnings

# --- MQTT TOPICS --- #
add_subdirectory(${CDIR}/mqtt-topics)
include_directories(${mqtt-topics_SOURCE_DIRS})
target_compile_options(mqtt-topics PRIVATE -w) # disable warnings

# --- SERIALIZERS --- #
add_subdirectory(${CDIR}/serializers)
target_link_libraries(serializers ${Protobuf_LIBRARIES} proto)
target_compile_options(serializers PRIVATE -fPIC)

# Dependencies
add_dependencies(serializers proto_generation)

# Pass EXTERNAL_DIRS to the parent CMakeLists.txt
set(EXTERNAL_DIRS ${EXTERNAL_DIRS} PARENT_SCOPE)