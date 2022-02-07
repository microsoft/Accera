####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# setup for using LLVM and MLIR
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

get_filename_component(TRANSFORMS_INCLUDE include ABSOLUTE)
get_filename_component(IR_INCLUDE ../ir/include ABSOLUTE)
add_subdirectory(include)
add_subdirectory(src)

set(src src/AcceraPasses.cpp)

set(rcvalue_src
    src/value/FunctionPointerResolutionPass.cpp
    src/value/ValueFuncToTargetPass.cpp
    src/value/ValueSimplifyPass.cpp
    src/value/ValueToLLVMLoweringPass.cpp
    src/value/ValueToStandardLoweringPass.cpp
)

set(rcvalue_include
    include/value/FunctionPointerResolutionPass.h
    include/value/ValueFuncToTargetPass.h
    include/value/ValueSimplifyPass.h
    include/value/ValueToLLVMLoweringPass.h
    include/value/ValueToStandardLoweringPass.h
)

set(rcnest_src
  src/nest/LoopNestPasses.cpp
  src/nest/LoopNestToValue.cpp src/nest/LoopNestToValueFunc.cpp)

set(include include/AcceraPasses.h)

set(rcnest_include
  include/nest/LoopNestPasses.h
  include/nest/LoopNestToValue.h include/nest/LoopNestToValueFunc.h)

set(rcexec_src src/exec/ExecutionPlanToAffineLoweringPass.cpp)

set(rcexec_include include/exec/ExecutionPlanToAffineLoweringPass.h)

set(rcgpu_src
  src/gpu/AcceraToGPUPass.cpp
  src/gpu/ConvertLaunchFuncToVulkanCalls.cpp
  src/gpu/EmitVulkanWrappers.cpp
  src/gpu/SerializeToHSACO.cpp

  # Disabled
  # src/gpu/AcceraToSPIRVPass.cpp
)

set(rcgpu_include
  include/gpu/AcceraToGPUPass.h
  include/gpu/AcceraVulkanPasses.h

  # Disabled
  # include/gpu/AcceraToSPIRVPass.h
)

set(rcaccera_src src/accera/AcceraLoweringPass.cpp)

set(rcaccera_include include/accera/AcceraLoweringPass.h)

set(util_src
  src/util/MathUtilities.cpp
  src/util/SnapshotUtilities.cpp
  src/util/VectorizationUtil.cpp
  src/util/VectorizedOp.cpp

  # Disabled
  # src/util/DebugUtilities.cpp
)

set(util_include
  include/util/MathUtilities.h
  include/util/SnapshotUtilities.h
  include/util/VectorizationUtil.h
  include/util/VectorizedOp.h

  # Disabled
  # include/util/DebugUtilities.h
)


set(library_name transforms)
set(src
    ${src}
    ${rcvalue_src}
    ${rcnest_src}
    ${rcexec_src}
    ${rcgpu_src}
    ${rcaccera_src}
    ${util_src}
)

set(include
    ${include}
    ${rcvalue_include}
    ${rcnest_include}
    ${rcexec_include}
    ${rcgpu_include}
    ${rcaccera_include}
    ${util_include}
)

add_library(${library_name} ${src} ${include})
target_include_directories(
  ${library_name} PRIVATE include ${ACCERA_LIBRARIES_DIR}
   PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
   $<INSTALL_INTERFACE:include/accera/transforms/include>
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
   $<INSTALL_INTERFACE:include/accera/transforms/src>
)
target_include_directories(${library_name} SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  ${library_name}
  PUBLIC utilities
         ir
         value
         lldELF
         MLIRGPUOps
         MLIRROCDLIR
         MLIRROCDLToLLVMIRTranslation
         MLIRStandardToLLVM
         MLIRSCFToStandard
         MLIRAffineToStandard
         MLIRAffineTransforms
         MLIRLinalgToLLVM
         MLIRLinalgTransforms
         MLIRTargetLLVMIRExport
         MLIRSupport
         MLIRIR
         MLIRAnalysis
         MLIRLinalg
         MLIRLLVMIR
         MLIRParser
         MLIRPass
         MLIRTransforms
)

add_dependencies(
  ${library_name}
  ValueConversionIncGen
  LoopNestToValueIncGen
  AcceraConversionIncGen
)

#
# Install headers and library
#
InstallAcceraHeaders(
  INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
               ${CMAKE_CURRENT_LIST_DIR}/include
               ${CMAKE_CURRENT_BINARY_DIR}/src
)
InstallAcceraLibrary(${library_name})
