cmake_minimum_required(VERSION 2.8)

project(lu)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif (OPENMP_FOUND)

aux_source_directory(src SRC_DIR)
link_libraries(m)
add_executable(lu ${SRC_DIR})
