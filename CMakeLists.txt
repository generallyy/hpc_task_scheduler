cmake_minimum_required(VERSION 3.24)
project(hpc_task_scheduler VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

option(HPC_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(HPC_BUILD_INFERENCE "Build the ONNX Runtime inference benchmark" ON)
option(HPC_FETCH_PYBIND11 "Fetch pybind11 when no installed package is found" OFF)
option(HPC_NATIVE_ARCH "Optimize for the build machine CPU" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an ONNX Runtime C/C++ distribution")

add_library(cluster_env STATIC src/cluster_env.cpp)
target_include_directories(cluster_env PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(cluster_env PUBLIC cxx_std_20)
set_target_properties(cluster_env PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(cluster_env PRIVATE /W4 /permissive- /O2 /EHsc)
    if(HPC_NATIVE_ARCH)
        target_compile_options(cluster_env PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(cluster_env PRIVATE -Wall -Wextra -Wpedantic -O3)
    if(HPC_NATIVE_ARCH)
        target_compile_options(cluster_env PRIVATE -march=native)
    endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cluster_env PUBLIC OpenMP::OpenMP_CXX)
endif()

if(HPC_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND AND HPC_FETCH_PYBIND11)
        FetchContent_Declare(
            pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    elseif(NOT pybind11_FOUND)
        message(FATAL_ERROR "pybind11 not found; install it or set HPC_FETCH_PYBIND11=ON")
    endif()
    pybind11_add_module(_cluster_env MODULE src/bindings.cpp)
    target_link_libraries(_cluster_env PRIVATE cluster_env)
    target_compile_features(_cluster_env PRIVATE cxx_std_20)
    install(TARGETS _cluster_env LIBRARY DESTINATION hpc_scheduler RUNTIME DESTINATION hpc_scheduler)
endif()

if(HPC_BUILD_INFERENCE)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
              PATH_SUFFIXES include include/onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
                 HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
                 PATH_SUFFIXES lib lib64)
    find_file(ONNXRUNTIME_SHARED_LIBRARY
              NAMES onnxruntime.dll libonnxruntime.so libonnxruntime.dylib
              HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
              PATH_SUFFIXES lib lib64 bin)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime C++ SDK not found. Set ONNXRUNTIME_ROOT or configure "
            "with -DHPC_BUILD_INFERENCE=OFF for a Python-only build.")
    endif()
    add_executable(hpc_inference src/main_inference.cpp)
    target_include_directories(hpc_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(hpc_inference PRIVATE ${ONNXRUNTIME_LIBRARY})
    target_compile_features(hpc_inference PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(hpc_inference PRIVATE /W4 /permissive- /O2 /EHsc)
    else()
        target_compile_options(hpc_inference PRIVATE -Wall -Wextra -Wpedantic -O3)
    endif()
    if(ONNXRUNTIME_SHARED_LIBRARY)
        add_custom_command(
            TARGET hpc_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ONNXRUNTIME_SHARED_LIBRARY} $<TARGET_FILE_DIR:hpc_inference>
            COMMENT "Copying the ONNX Runtime shared library beside hpc_inference"
        )
        install(FILES ${ONNXRUNTIME_SHARED_LIBRARY} DESTINATION bin)
    endif()
    install(TARGETS hpc_inference RUNTIME DESTINATION bin)
endif()

if(BUILD_TESTING)
    add_executable(cluster_env_test tests/test_cluster_env.cpp)
    target_link_libraries(cluster_env_test PRIVATE cluster_env)
    target_compile_features(cluster_env_test PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(cluster_env_test PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(cluster_env_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME cluster_env_test COMMAND cluster_env_test)
endif()
