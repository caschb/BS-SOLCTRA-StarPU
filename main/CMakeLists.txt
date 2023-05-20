set(EXECUTABLE_TARGET bs_solctra)

add_executable(${EXECUTABLE_TARGET} src/main_multinode.cpp src/solctra_multinode.cpp src/utils.cpp)

find_package(MPI REQUIRED)
target_include_directories(${EXECUTABLE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(${EXECUTABLE_TARGET} PRIVATE MPI::MPI_CXX)

find_package(Threads REQUIRED)
target_link_libraries(${EXECUTABLE_TARGET} PRIVATE Threads::Threads)

find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.4)
if(STARPU_FOUND)
    target_include_directories(${EXECUTABLE_TARGET} SYSTEM PRIVATE ${STARPU_INCLUDE_DIRS})
    target_link_directories(${EXECUTABLE_TARGET} PRIVATE ${STARPU_STATIC_LIBRARY_DIRS})
    target_link_libraries(${EXECUTABLE_TARGET} PRIVATE ${STARPU_STATIC_LIBRARIES})
else(STARPU_FOUND)
    message(FATAL_ERROR "StarPU not found")
endif()



if(${TREAT_WARNINGS_AS_ERRORS})
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -Werror)
endif()

if(${VECTOR})
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -march=knl -mavx512f -mavx512pf -mavx512er -mavx512cd)
endif()

if(${FAST_MATH})
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -ffast-math)
endif()
