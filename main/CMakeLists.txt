set(EXECUTABLE_TARGET bs_solctra)

add_executable(${EXECUTABLE_TARGET} src/main_multinode.cpp src/solctra_multinode.cpp src/utils.cpp)


target_link_libraries(
  ${EXECUTABLE_TARGET}
  PRIVATE bs_solctra::bs_solctra_options
          bs_solctra::bs_solctra_warnings)


target_include_directories(${EXECUTABLE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(MPI REQUIRED)
target_link_system_libraries(${EXECUTABLE_TARGET} PRIVATE MPI::MPI_CXX)

find_package(OpenMP REQUIRED)
target_link_system_libraries(${EXECUTABLE_TARGET} PRIVATE OpenMP::OpenMP_CXX)

find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.4)
if(STARPU_FOUND)
    target_include_directories(${EXECUTABLE_TARGET} SYSTEM PRIVATE ${STARPU_INCLUDE_DIRS})
    target_link_directories(${EXECUTABLE_TARGET} PRIVATE ${STARPU_STATIC_LIBRARY_DIRS})
    target_link_libraries(${EXECUTABLE_TARGET} PRIVATE ${STARPU_STATIC_LIBRARIES})
else(STARPU_FOUND)
    message(FATAL_ERROR "StarPU not found")
endif()


if(bs_solctra_solctra_ENABLE_KNL_OPTS)
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -mavx512f -mavx512pf -mavx512er -mavx512cd -march=knl)
endif()
if(bs_solctra_ENABLE_FAST_MATH)
    target_compile_options(${EXECUTABLE_TARGET} PRIVATE -ffast-math)
endif()

