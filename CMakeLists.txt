cmake_minimum_required(VERSION 3.20)

project(BS-SOLCTRA LANGUAGES C)
option(USE_GPU "Build with GPU support" OFF)
option(VECTOR "Build for vector processor" OFF)
option(FAST_MATH "Use fast-math flag" OFF)
option(TREAT_WARNINGS_AS_ERRORS "Treat all warnings as errors" OFF)
option(ACTIVATE_SANITIZERS "Turn on the sanitizers" OFF)
option(GENERATE_OPTIMIZATION_REPORTS "Generates optimization reports" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(COMMON_FLAGS "-Wall -Wpedantic -Wextra -Wshadow -Wold-style-cast")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)
if(USE_GPU)
enable_language(CUDA)
endif()

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_subdirectory(main)
