####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(util_name acc-gpu-runner)

if (NOT Vulkan_FOUND)
    message(WARNING "Vulkan library not found, ${util_name} won't be an available build target")
else()
    message(STATUS "Vulkan library found at ${Vulkan_LIBRARY}, ${util_name} will be an available build target")

    get_target_property(MLIR_RUNNER_UTILS_PATH mlir_runner_utils LOCATION)

    add_compile_definitions(MLIR_RUNNER_UTILS_PATH="${MLIR_RUNNER_UTILS_PATH}")

    set(VULKAN_RUNTIME_WRAPPERS_PATH $<TARGET_FILE:acc-vulkan-runtime-wrappers>)
    add_compile_definitions(VULKAN_RUNTIME_WRAPPERS_PATH="${VULKAN_RUNTIME_WRAPPERS_PATH}")

    set(util_src
        src/ACCGPURunnerMain.cpp
        src/ACCJITRunner.cpp
    )
    set(util_include)

    source_group("src" FILES ${util_src})
    source_group("include" FILES ${util_include})

    add_executable(${util_name} ${util_src} ${util_include})
    target_include_directories(${util_name} PRIVATE ${ACCERA_ROOT}/accera include)

    set(mlir_libs
        LLVMCore
        LLVMSupport
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRIR
        MLIRJitRunner
        MLIRLLVMIR
        MLIRParser
        MLIRSPIRVTransforms
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRTranslation
        MLIROptLib
        ${Vulkan_LIBRARY}
    )

    set(accera_libs
        utilities
        ir
        transforms
        value
        mlirHelpers
    )

    target_link_libraries(
        ${util_name}
        PRIVATE
            ${mlir_libs}
            ${accera_libs}
    )
    copy_shared_libraries(${util_name})

    set_property(TARGET ${util_name} PROPERTY FOLDER "accera")

    add_dependencies(${util_name} acc-vulkan-runtime-wrappers)
    InstallAcceraRuntime(${util_name})

    #
    # Deploy python acc-gpu-runner tools
    #

    set(ROOT_BIN_DIR ${CMAKE_BINARY_DIR}/bin)

    set(configure_scripts
        acc_gpu_runner_config.py
    )

    set(copy_scripts
        acc-gpu-runner.py
    )

    set(deployed_scripts )
    foreach(script ${configure_scripts})
        configure_file(${script}.in ${ROOT_BIN_DIR}/${script} @ONLY)
        set(deployed_scripts ${deployed_scripts} ${ROOT_BIN_DIR}/${script})
    endforeach()

    foreach(script ${copy_scripts})
        configure_file(${script} ${ROOT_BIN_DIR}/${script} COPYONLY)
        set(deployed_scripts ${deployed_scripts} ${ROOT_BIN_DIR}/${script})
    endforeach()

    #
    # Install accc scripts and CMakeLists templates
    #
    install(PROGRAMS ${deployed_scripts} TYPE BIN)
endif()
