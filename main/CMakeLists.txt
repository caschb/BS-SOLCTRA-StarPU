set(EXECUTABLE_TARGET bs_solctra)

# add_executable(${EXECUTABLE_TARGET} src/main_multinode.cpp src/solctra_multinode.cpp src/utils.cpp 
#   src/solctra_cuda.cu src/argument_parsers.cpp src/gpu_functions.cu)
add_executable(${EXECUTABLE_TARGET} src/main.c src/argument_parsers.c src/utils.c src/cpu_functions.c src/solctra.c)
target_include_directories(${EXECUTABLE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(MPI REQUIRED)
target_link_libraries(${EXECUTABLE_TARGET} PRIVATE MPI::MPI_C)

find_package(OpenMP REQUIRED)
target_link_libraries(${EXECUTABLE_TARGET} PRIVATE OpenMP::OpenMP_C)

find_package(Threads REQUIRED)
target_link_libraries(${EXECUTABLE_TARGET} PRIVATE Threads::Threads)

find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.4 starpumpi-1.4)
if(STARPU_FOUND)
    target_include_directories(${EXECUTABLE_TARGET} SYSTEM PRIVATE ${STARPU_INCLUDE_DIRS})
    target_link_directories(${EXECUTABLE_TARGET} PRIVATE ${STARPU_STATIC_LIBRARY_DIRS})
    target_link_libraries(${EXECUTABLE_TARGET} PRIVATE ${STARPU_STATIC_LIBRARIES})
else(STARPU_FOUND)
    message(FATAL_ERROR "StarPU not found")
endif()

if(${GENERATE_OPTIMIZATION_REPORTS})
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -Rpass="*" -Rpass-missed="*" -Rpass-analysis="*")
    endif()

    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -fopt-info-vec)
    endif()
endif()
