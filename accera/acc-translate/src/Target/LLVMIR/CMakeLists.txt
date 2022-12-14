add_mlir_translation_library(IntrinsicToLLVMIRTranslation
  IntrinsicToLLVMIRTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${ACCERA_BIN_DIR}/accera/ir/include

  DEPENDS
  MLIRAcceraIntrinsics
  AcceraIntrinsicsConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAcceraIntrinsics
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )

target_include_directories(IntrinsicToLLVMIRTranslation PUBLIC
    ${ACCERA_BIN_DIR}/ir/include
)
