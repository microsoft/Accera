####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(library_name value)

set(src
    src/AcceraParameters.cpp
    src/Array.cpp
    src/ArrayOperations.cpp
    src/Cache.cpp
    src/CompilerOptions.cpp
    src/Debugging.cpp
    src/Emittable.cpp
    src/EmitterContext.cpp
    src/FastMath.cpp
    src/FunctionDeclaration.cpp
    src/Kernel.cpp
    src/KernelPredicate.cpp
    src/MLIREmitterContext.cpp
    src/Matrix.cpp
    src/MatrixFragment.cpp
    src/MatrixOperations.cpp
    src/MLOperations.cpp
    src/Nest.cpp
    src/Plan.cpp
    src/Pointer.cpp
    src/Profiling.cpp
    src/Scalar.cpp
    src/ScalarOperations.cpp
    src/Schedule.cpp
    src/TargetDevice.cpp
    src/Tensor.cpp
    src/TensorOperations.cpp
    src/Value.cpp
    src/ValueOperations.cpp
    src/Vector.cpp
    src/VectorOperations.cpp
)

set(include
    include/AcceraParameters.h
    include/Array.h
    include/ArrayOperations.h
    include/Cache.h
    include/CompilerOptions.h
    include/Debugging.h
    include/Emittable.h
    include/EmitterContext.h
    include/EmitterException.h
    include/EmitterTypes.h
    include/ExecutionOptions.h
    include/FastMath.h
    include/FunctionDeclaration.h
    include/Index.h
    include/IterationDomain.h
    include/Kernel.h
    include/KernelPredicate.h
    include/LLVMUtilities.h
    include/LoopNests.h
    include/Matrix.h
    include/MatrixFragment.h
    include/MatrixOperations.h
    include/MLIREmitterContext.h
    include/MLOperations.h
    include/Nest.h
    include/Plan.h
    include/Pointer.h
    include/Profiling.h
    include/Range.h
    include/Scalar.h
    include/ScalarDimension.h
    include/ScalarIndex.h
    include/ScalarOperations.h
    include/Schedule.h
    include/TargetDevice.h
    include/Tensor.h
    include/TensorOperations.h
    include/Value.h
    include/ValueOperations.h
    include/ValueType.h
    include/Vector.h
    include/VectorOperations.h
)

set(doc doc/README.md)

source_group("src" FILES ${src})
source_group("include" FILES ${include})
source_group("doc" FILES ${doc})

add_library(${library_name} ${src} ${include})
target_include_directories(
  ${library_name} PRIVATE include include/legacy ${ACCERA_LIBRARIES_DIR}
)
target_include_directories(${library_name} SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  ${library_name}
  PUBLIC ${LLVM_LIBS}
         ir
         mlirHelpers
         utilities
)

target_compile_options(${library_name} PUBLIC ${LLVM_COMPILE_OPTIONS})
set_property(TARGET ${library_name} PROPERTY FOLDER "accera")

# TODO: port value test to use MLIR and catch2

#
# Install headers and library
#
InstallAcceraHeaders(
  INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
               ${CMAKE_CURRENT_LIST_DIR}/include
)
InstallAcceraLibrary(${library_name})
