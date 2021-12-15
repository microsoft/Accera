# Builds LLVM for features needed by Accera
set(LLVM_VERSION "13.0.0")

# Only release builds to save space, comment this out if you want both release and debug
set(VCPKG_BUILD_TYPE release)

# BUILD_SHARED_LIBS option is not supported on Windows
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

# Note: to get the SHA512 after updating REF, run once after changing REF and the computed SHA512 will be printed in the error spew
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO llvm/llvm-project
    REF bf9f21a28be171dc500cc68b4cb1fcd3fc33f22
    SHA512 b80817d31d7123b8c86708d9f73278850908da9b7e177564bf024f1565935c5ddd42003e8519bcce4387995c30ba876f4bc8ed53b713a55181d2215b2b71988b
    HEAD_REF main
    PATCHES
        0001-Merged-PR-2213-mlir-Plumb-OpenMP-dialect-attributes-.patch
        0002-Merged-PR-2237-Improved-codegen-of-vpmaddwd-instruct.patch
        0003-Fix-bad-merge.patch
        0004-fix-install-paths.patch # cf. https://github.com/microsoft/vcpkg/blob/master/ports/llvm
)

vcpkg_find_acquire_program(PYTHON3)
get_filename_component(PYTHON3_DIR ${PYTHON3} DIRECTORY)
vcpkg_add_to_path(${PYTHON3_DIR})

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}/llvm
    PREFER_NINJA
    OPTIONS
        -DLLVM_APPEND_VC_REV=OFF
        -DLLVM_ENABLE_BINDINGS=OFF
        -DLLVM_CCACHE_BUILD=OFF
        -DLLVM_ENABLE_OCAMLDOC=OFF
        -DLLVM_ENABLE_TERMINFO=OFF
        -DLLVM_ENABLE_WARNINGS=OFF
        -DLLVM_INCLUDE_EXAMPLES=OFF
        -DLLVM_INCLUDE_TESTS=OFF
        -DLLVM_INCLUDE_DOCS=OFF
        -DLLVM_BUILD_EXAMPLES=OFF
        -DLLVM_BUILD_UTILS=ON # FileCheck
        -DLLVM_BUILD_TOOLS=ON # opt, llc, mlir-translate
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DLLVM_ENABLE_EH=ON
        -DLLVM_ENABLE_RTTI=ON
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_INSTALL_UTILS=ON # FileCheck
        "-DLLVM_ENABLE_PROJECTS=mlir;lld"
        "-DLLVM_TARGETS_TO_BUILD=host;X86;ARM;NVPTX;AMDGPU"
        -DPACKAGE_VERSION=${LLVM_VERSION}
        # Force TableGen to be built with optimization. This will significantly improve build time.
        # cf. https://github.com/microsoft/vcpkg/blob/master/ports/llvm
        -DLLVM_OPTIMIZED_TABLEGEN=ON
        # Limit the maximum number of concurrent link jobs to 1. This should fix low amount of memory issue for link.
        # cf. https://github.com/microsoft/vcpkg/blob/master/ports/llvm
        -DLLVM_PARALLEL_LINK_JOBS=1
        # Disable build LLVM-C.dll (Windows only) due to doesn't compile with CMAKE_DEBUG_POSTFIX
        # cf. https://github.com/microsoft/vcpkg/blob/master/ports/llvm
        -DLLVM_BUILD_LLVM_C_DYLIB=OFF
        # Path for binary subdirectory (defaults to 'bin')
        -DLLVM_TOOLS_INSTALL_DIR=tools/llvm
    OPTIONS_DEBUG
        -DCMAKE_DEBUG_POSTFIX=d
)

vcpkg_install_cmake()

vcpkg_fixup_cmake_targets(CONFIG_PATH "share/llvm" TARGET_PATH "share/llvm")
file(INSTALL ${SOURCE_PATH}/llvm/LICENSE.TXT DESTINATION ${CURRENT_PACKAGES_DIR}/share/llvm RENAME copyright)
file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/llvm_usage DESTINATION ${CURRENT_PACKAGES_DIR}/share/llvm RENAME usage)

vcpkg_fixup_cmake_targets(CONFIG_PATH "share/mlir" TARGET_PATH "share/mlir" DO_NOT_DELETE_PARENT_CONFIG_PATH)
file(INSTALL ${SOURCE_PATH}/mlir/LICENSE.TXT DESTINATION ${CURRENT_PACKAGES_DIR}/share/mlir RENAME copyright)
file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/mlir_usage DESTINATION ${CURRENT_PACKAGES_DIR}/share/mlir RENAME usage)

vcpkg_fixup_cmake_targets(CONFIG_PATH "share/lld" TARGET_PATH "share/lld" DO_NOT_DELETE_PARENT_CONFIG_PATH)
file(INSTALL ${SOURCE_PATH}/lld/LICENSE.TXT DESTINATION ${CURRENT_PACKAGES_DIR}/share/lld RENAME copyright)
file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/lld_usage DESTINATION ${CURRENT_PACKAGES_DIR}/share/lld RENAME usage)

file(INSTALL ${SOURCE_PATH}/llvm/LICENSE.TXT DESTINATION ${CURRENT_PACKAGES_DIR}/share/accera-llvm RENAME copyright)

vcpkg_copy_tool_dependencies(${CURRENT_PACKAGES_DIR}/tools/${PORT})

# Post-build validation does not like duplication in the /debug hierarchy
if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "debug")
    file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)
    file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/share)
    file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/tools)
endif()

# Post-build validation warnings: LLVM still generates a few DLLs in the static build
# * libclang.dll
# * LTO.dll
# * Remarks.dll
set(VCPKG_POLICY_DLLS_IN_STATIC_LIBRARY enabled)