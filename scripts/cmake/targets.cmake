# Included by the patched MNN engine CMakeLists after the llm target exists.
# The source is built afresh; no archive/object substitution is used here.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "These experiment targets are intended for Linux RK3588")
endif()
if(NOT MNN_BUILD_LLM OR NOT MNN_BUILD_LLM_OMNI OR NOT MNN_OPENCL
   OR MNN_BUILD_SHARED_LIBS OR MNN_SEP_BUILD)
    message(FATAL_ERROR "Use scripts/build_rk3588.py for the required MNN configuration")
endif()

option(KERNEL_BENCH_WITH_RKNN "Build the CPU+NPU+GPU audio path" ON)
if(KERNEL_BENCH_WITH_RKNN)
    if(NOT EXISTS "${KERNEL_BENCH_RKNN_INCLUDE}/rknn_api.h"
       OR NOT EXISTS "${KERNEL_BENCH_RKNN_LIBRARY}")
        message(FATAL_ERROR "Supply the existing RKNN 2.3.2 include directory and runtime library")
    endif()
    target_include_directories(llm PRIVATE "${KERNEL_BENCH_RKNN_INCLUDE}")
    target_compile_definitions(llm PRIVATE MNN_OMNI_RKNN_PREFIX)
    # The enclosing MNN CMakeLists attaches these to the merged static target.
    list(APPEND MNN_EXTRA_DEPENDS "${KERNEL_BENCH_RKNN_LIBRARY}")
endif()
if(NOT EXISTS "${KERNEL_BENCH_OPENCL_LIBRARY}")
    message(FATAL_ERROR "Supply the board's existing libOpenCL.so path")
endif()
find_package(Threads REQUIRED)

foreach(app IN ITEMS bigvgan_replay vcb_eval omni_ab)
    add_executable(${app} "${KERNEL_BENCH_ROOT}/src/apps/${app}.cpp")
    target_include_directories(${app} PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/transformers/llm/engine/include"
        "${CMAKE_SOURCE_DIR}/transformers/llm/engine/app")
    target_compile_definitions(${app} PRIVATE MNN_BUILD_STATIC_LIBS LLM_SUPPORT_AUDIO LLM_SUPPORT_VISION)
    # Match the original standalone runner compilation; MNN itself retains its flags.
    target_compile_options(${app} PRIVATE -fexceptions -frtti)
    set_target_properties(${app} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_link_libraries(${app} MNN Threads::Threads ${CMAKE_DL_LIBS})
endforeach()

add_executable(bench_deconv "${KERNEL_BENCH_ROOT}/kernels/bench_deconv.cpp")
target_include_directories(bench_deconv PRIVATE "${CMAKE_SOURCE_DIR}/3rd_party/OpenCLHeaders")
target_compile_options(bench_deconv PRIVATE -fexceptions -frtti)
set_target_properties(bench_deconv PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(bench_deconv "${KERNEL_BENCH_OPENCL_LIBRARY}" Threads::Threads)
