cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_multicore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_multicore_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_multicore_v2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_multicore_v2_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_multicore_v3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_multicore_v3_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_alive_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_dup1_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_getval_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fps_custom_dynamic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/group_points_dynamic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ball_query_dynamic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ball_query_tiling_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ball_query_debug.cpp
)

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

add_executable(fps_kernellaunch ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_compile_options(fps_kernellaunch PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_link_libraries(fps_kernellaunch PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
)

install(TARGETS fps_kernellaunch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Python-callable host wrapper library (NPU mode only)
if("${RUN_MODE}" STREQUAL "npu" OR "${RUN_MODE}" STREQUAL "sim")
    add_library(fps_host SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host.cpp)
    target_compile_options(fps_host PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(fps_host PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Multi-core host wrapper
    add_library(fps_host_multicore SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_multicore.cpp)
    target_compile_options(fps_host_multicore PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(fps_host_multicore PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_multicore
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Debug multi-core host wrapper
    add_library(fps_host_mc_debug SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_mc_debug.cpp)
    target_compile_options(fps_host_mc_debug PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(fps_host_mc_debug PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_mc_debug
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Multi-core v2 host wrapper (DataCopy-based cross-core communication)
    add_library(fps_host_multicore_v2 SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_multicore_v2.cpp)
    target_compile_options(fps_host_multicore_v2 PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(fps_host_multicore_v2 PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_multicore_v2
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Multi-core v2 debug host wrapper
    add_library(fps_host_mc_v2_dbg SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_mc_v2_debug.cpp)
    target_compile_options(fps_host_mc_v2_dbg PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(fps_host_mc_v2_dbg PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_mc_v2_dbg
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Multi-core v3 host wrapper (pure vector+DMA cross-core)
    add_library(fps_host_multicore_v3 SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_multicore_v3.cpp)
    target_compile_options(fps_host_multicore_v3 PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(fps_host_multicore_v3 PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_multicore_v3
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Multi-core v3 debug host wrapper
    add_library(fps_host_mc_v3_dbg SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_mc_v3_debug.cpp)
    target_compile_options(fps_host_mc_v3_dbg PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(fps_host_mc_v3_dbg PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(fps_host_mc_v3_dbg PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_mc_v3_dbg
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Alive check host wrapper (multi-core Duplicate+DataCopy verify)
    add_library(fps_host_alive_check SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_alive_check.cpp)
    target_compile_options(fps_host_alive_check PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(fps_host_alive_check PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(fps_host_alive_check PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_alive_check
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Duplicate count=1 check
    add_library(fps_host_dup1_check SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_dup1_check.cpp)
    target_compile_options(fps_host_dup1_check PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(fps_host_dup1_check PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(fps_host_dup1_check PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_dup1_check
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # GetValue on GM check
    add_library(fps_host_getval_check SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_getval_check.cpp)
    target_compile_options(fps_host_getval_check PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(fps_host_getval_check PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(fps_host_getval_check PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_getval_check
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Dynamic-shape multi-core FPS
    add_library(fps_host_dynamic SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fps_host_dynamic.cpp)
    target_compile_options(fps_host_dynamic PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(fps_host_dynamic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(fps_host_dynamic PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS fps_host_dynamic
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Dynamic-shape multi-core GroupPoints
    add_library(group_points_host_dynamic SHARED ${CMAKE_CURRENT_SOURCE_DIR}/group_points_host_dynamic.cpp)
    target_compile_options(group_points_host_dynamic PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(group_points_host_dynamic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(group_points_host_dynamic PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS group_points_host_dynamic
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Dynamic-shape multi-core Ball Query
    add_library(ball_query_host_dynamic SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ball_query_host_dynamic.cpp)
    target_compile_options(ball_query_host_dynamic PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(ball_query_host_dynamic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(ball_query_host_dynamic PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS ball_query_host_dynamic
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Tiling check debug kernel
    add_library(ball_query_tiling_check SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ball_query_tiling_check_host.cpp)
    target_compile_options(ball_query_tiling_check PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(ball_query_tiling_check PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(ball_query_tiling_check PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS ball_query_tiling_check
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Full debug kernel with checkpoints
    add_library(ball_query_debug SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ball_query_debug_host.cpp)
    target_compile_options(ball_query_debug PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(ball_query_debug PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/out/include/ascendc_kernels_npu)
    target_link_libraries(ball_query_debug PRIVATE
        host_intf_pub
        ascendc_kernels_${RUN_MODE}
    )
    install(TARGETS ball_query_debug
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
