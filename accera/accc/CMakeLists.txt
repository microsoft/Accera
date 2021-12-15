####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(ROOT_BIN_DIR ${CMAKE_BINARY_DIR}/bin)

set_property(GLOBAL PROPERTY ACCC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/bin")
set_property(GLOBAL PROPERTY ACCC_BUILD_PATH "${ROOT_BIN_DIR}")

# accc_config.py.in, CMakeLists.txt.*.in
set(ACCERA_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
set(ACCC_TEMPLATE_DIR_NAME "accc_templates")
set(GENERATOR_CMAKELIST ${ACCC_TEMPLATE_DIR_NAME}/CMakeLists.txt.generator.in)
set(EMITTED_LIB_CMAKELIST ${ACCC_TEMPLATE_DIR_NAME}/CMakeLists.txt.emitted_library.in)
set(MAIN_CMAKELIST ${ACCC_TEMPLATE_DIR_NAME}/CMakeLists.txt.main.in)
set(MAIN_DEPLOY_DIR_NAME "deploy")
set(LIBRARY_NAME_TAG $LIBRARY_NAME$)
set(MAIN_BASENAME_TAG $MAIN_BASENAME$)
set(PROGRAM_NAME_TAG $PROGRAM_NAME$)
set(MAIN_DEPLOY_TARGET_TYPE_TAG $MAIN_DEPLOY_TARGET_TYPE$)
set(DSL_FILE_BASENAME_TAG $DSL_FILE_BASENAME$)

# build_config.py.in
set(SHARED_LIB_EXTENSION ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(STATIC_LIB_EXTENSION ${CMAKE_STATIC_LIBRARY_SUFFIX})
set(SHARED_LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(STATIC_LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
set(OBJ_EXTENSION .o)
set(ASM_EXTENSION .s)
set(EXE_EXTENSION )
set(CONFIG_IN_BUILT_PATH False)
set(ADDITIONAL_CMAKE_INIT_ARGS )
get_property(RC_VULKAN_RUNTIME_WRAPPER_REL_PATH GLOBAL PROPERTY RC_VULKAN_RUNTIME_WRAPPER_REL_PATH)
set(VULKAN_LIBRARY_PATH ${Vulkan_LIBRARY})
set(OPENMP_LIBRARY_PATH ${LLVMOpenMP_LIBRARY})

if(MSVC)
    set(OBJ_EXTENSION .obj)
    set(EXE_EXTENSION .exe)
    set(CONFIG_IN_BUILT_PATH True)
    set(ADDITIONAL_CMAKE_INIT_ARGS "-G \"Visual Studio 16 2019\" -A x64 -T host=x64")
elseif(XCODE)
    set(CONFIG_IN_BUILT_PATH True)
endif()

# Blas
if(USE_MKL)
    set(ADDITIONAL_CMAKE_INIT_ARGS "${ADDITIONAL_CMAKE_INIT_ARGS} -DUSE_MKL=1")
endif()
if(USE_BLAS)
    set(ADDITIONAL_CMAKE_INIT_ARGS "${ADDITIONAL_CMAKE_INIT_ARGS} -DUSE_BLAS=1")
endif()

set(copy_scripts
    accc.py
    utilities.py

    # TODO: Move this somewhere better
    ../onnx-emitter/onnx_emitter.py
)

set(configure_files
    accc_config.py
    build_config.py
)

set(copy_files
    parameters.py
)

set(configure_build_files
    CMakeLists.txt.emitted_library.in
    CMakeLists.txt.generator.in
    CMakeLists.txt.main.in
)

set(deployed_scripts )
set(deployed_files )

foreach(file ${configure_files})
    configure_file(${file}.in ${ROOT_BIN_DIR}/${file} @ONLY)
    set(deployed_files ${deployed_files} ${ROOT_BIN_DIR}/${file})
endforeach()

foreach(script ${copy_scripts})
    configure_file(${script} ${ROOT_BIN_DIR}/${script} COPYONLY)
    set(deployed_scripts ${deployed_scripts} ${ROOT_BIN_DIR}/${script})
endforeach()

foreach(file ${copy_files})
    configure_file(${file} ${ROOT_BIN_DIR}/${file} COPYONLY)
    set(deployed_files ${deployed_files} ${ROOT_BIN_DIR}/${file})
endforeach()

set(cmake_configured_build_files )
foreach(build_file ${configure_build_files})
    configure_file(${build_file} ${ROOT_BIN_DIR}/${ACCC_TEMPLATE_DIR_NAME}/${build_file} @ONLY)
    set(cmake_configured_build_files ${cmake_configured_build_files} ${ROOT_BIN_DIR}/${ACCC_TEMPLATE_DIR_NAME}/${build_file})
endforeach()

#
# Install accc scripts and CMakeLists templates
#

file(INSTALL
        ${LLC_EXECUTABLE} ${OPT_EXECUTABLE} ${MLIR_TRANSLATE_EXECUTABLE}
    DESTINATION
        ${CMAKE_BINARY_DIR}/bin
    USE_SOURCE_PERMISSIONS
    )
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    install(
            PROGRAMS ${LLC_EXECUTABLE} ${OPT_EXECUTABLE} ${MLIR_TRANSLATE_EXECUTABLE}
            CONFIGURATIONS ${CONFIG}
            TYPE BIN
    )
    install(
            PROGRAMS ${LLC_EXECUTABLE} ${OPT_EXECUTABLE} ${MLIR_TRANSLATE_EXECUTABLE}
            COMPONENT accera-llvm
            CONFIGURATIONS ${CONFIG}
            DESTINATION "accera/bin"
    )
    install(FILES ${cmake_configured_build_files}
            COMPONENT accera-compilers
            DESTINATION "accera/bin"
    )
endforeach(CONFIG ${CMAKE_CONFIGURATION_TYPES})

install(
        PROGRAMS ${LLC_EXECUTABLE} ${OPT_EXECUTABLE} ${MLIR_TRANSLATE_EXECUTABLE}
        COMPONENT accera-llvm
        DESTINATION "accera/bin"
)
install(
        PROGRAMS ${deployed_scripts}
        COMPONENT accera-compilers
        DESTINATION "accera"
)

install(FILES ${deployed_files}
        COMPONENT accera-compilers
        DESTINATION "accera"
)
install(FILES ${cmake_configured_build_files}
        COMPONENT accera-compilers
        DESTINATION "accera/bin"
)

install(PROGRAMS ${deployed_scripts} ${LLC_EXECUTABLE} ${OPT_EXECUTABLE} ${MLIR_TRANSLATE_EXECUTABLE} TYPE BIN)
install(FILES ${deployed_files} TYPE BIN)
install(FILES ${cmake_configured_build_files} DESTINATION bin/${ACCC_TEMPLATE_DIR_NAME})
