# Auto generation of proto files.
# 1. find all the .proto in the project folder
# 2. write a file ".proto_sources.txt" used later with the list of .proto files
# 3. create the variables with the paths of the generated .cc and .h proto_files
# 4. create the command for the generation.
#
# HOW DOES IT WORK
# Need to compile the .proto files into .cc and .h with the protoc command.
# The compilation should be done only if the .proto files changed (even if cahnged only the content)
# Cmake does this "watch" for every source file, so here I'm using the same approach to watch the .proto files
# The generation should be done only if some target need the .cc or .h, so we can define a dependency tree.
# "proto_generation" is the target that can be used to start the generation.
# Ex: I have a target executable (with add_executable()) and it needs the proto files,
#     I can use add_dependency(my_executable_target proto_generation).
#     In this way I am sayng that the executable need the target proto_generation, and that should be generated
#     before my_executable_target.
# So now how can I link the proto_generation with the actual compilation script?
# By using add_custom_command() that will define some OUTPUT from the given COMMAND.
# So proto_generation will depend on the OUTPUT of the new command, and so cmake will execute it.
# And the thing of watching if files changed? that is controlled by DEPENDS of add_custom_command.
# So the custom command will be called if:
# - some target needs .cc and .h files generated from it
# - if no .cc and .h exists
# - if .proto files are added/changed
#
# COMPILATION SCRIPT
# The actual compilation script is compile.sh that reads ".proto_sources_out.txt"
# and will use protoc for each .proto in the file.

set(proto_cpp "")
set(proto_hpp "")
set(proto_folders "")
set(proto_files "")
set(proto_sources_out "${DIR}/protobuffer/.proto_sources.txt")

# specify here paths where to search for .proto files
set(proto_search_paths
  ${DIR}/external/serializers/*.proto
)

# list all files and save to file called ${proto_source_out}
file(GLOB_RECURSE proto_files ${proto_search_paths})
file(GLOB_RECURSE proto_files_relative RELATIVE ${DIR} ${proto_search_paths})
list(JOIN proto_files_relative "\n" proto_files_str)
write_file(${proto_sources_out} ${proto_files_str})

# Definition of generated files .pb.cc and .pb.h
# Replacing .proto with .pb.cc and .pb.h
# Appending the folder /.generated after ${CMAKE_CURRENT...}
foreach(iter_item  ${proto_files})
  string(REPLACE ${DIR} ${DIR}/.generated/ iter_item ${iter_item})
  string(REPLACE .proto .pb.cc tmp_cpp ${iter_item})
  string(REPLACE .proto .pb.h  tmp_hpp ${iter_item})
  # Get the folder of the .proto file
  get_filename_component(folder ${iter_item} DIRECTORY)
  # Adding the path
  list  (APPEND  proto_cpp             ${tmp_cpp})
  list  (APPEND  proto_hpp             ${tmp_hpp})
  list  (APPEND  proto_folders         ${folder} )
endforeach()


set(proto_folders ${proto_folders} PARENT_SCOPE)
set(proto_files ${proto_files} PARENT_SCOPE)
set(proto_cpp ${proto_cpp} PARENT_SCOPE)
set(proto_hpp ${proto_hpp} PARENT_SCOPE)