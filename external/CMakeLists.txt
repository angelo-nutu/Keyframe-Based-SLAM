message("[External]")

# define helper variable (CDIR is Current DIRectoty)
set(CDIR ${CMAKE_CURRENT_LIST_DIR})

set(EXTERNAL_DIRS
  ${CDIR}
)
include_directories(PUBLIC ${EXTERNAL_DIRS})

# Pass EXTERNAL_DIRS to the parent CMakeLists.txt
set(EXTERNAL_DIRS ${EXTERNAL_DIRS} PARENT_SCOPE)