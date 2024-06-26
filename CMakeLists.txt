cmake_minimum_required(VERSION 3.20)

project(BS-SOLCTRA LANGUAGES CUDA CXX)
option(VECTOR "Build for vector processor" OFF)
option(FAST_MATH "Use fast-math flag" OFF)
option(TREAT_WARNINGS_AS_ERRORS "Treat all warnings as errors" OFF)
option(ACTIVATE_SANITIZERS "Turn on the sanitizers" OFF)
option(GENERATE_OPTIMIZATION_REPORTS "Generates optimization reports" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(COMMON_FLAGS "-Wall -Wpedantic -Wextra -Wshadow -Wold-style-cast")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fstrict-aliasing ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -march=native -fstrict-aliasing ${COMMON_FLAGS}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_subdirectory(main)
