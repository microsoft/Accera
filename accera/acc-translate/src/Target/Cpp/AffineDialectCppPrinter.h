////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef AFFINE_DIALECT_CPP_PRINTER_H_
#define AFFINE_DIALECT_CPP_PRINTER_H_

#include <cassert>
#include <mlir/Dialect/Affine/IR/AffineOps.h>

#include "CppPrinter.h"

namespace mlir
{
namespace cpp_printer
{

    struct AffineDialectCppPrinter : public DialectCppPrinter
    {
        AffineDialectCppPrinter(CppPrinter* printer_) :
            DialectCppPrinter(printer_), needAffineMemCpy(false) {}

        std::string getName() override { return "Affine"; }

        LogicalResult printPrologue() override;

        /// print Operation from Affine Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;

        LogicalResult printAffineApplyOp(AffineApplyOp affineApplyOp);

        LogicalResult printAffineStoreOp(AffineStoreOp affineStoreOp);

        LogicalResult printAffineLoadOp(AffineLoadOp affineLoadOp);

        LogicalResult printAffineVectorLoadOp(AffineVectorLoadOp affineVecLoadOp);

        LogicalResult printAffineVectorStoreOp(AffineVectorStoreOp affineVecStoreOp);

        LogicalResult printAffineForOp(AffineForOp affineForOp);

        LogicalResult printAffineMapFunc(AffineMap map, StringRef funcName);

        LogicalResult printAffineExpr(AffineExpr affineExpr);

        LogicalResult runPrePrintingPasses(Operation* op) override;

        llvm::DenseMap<AffineMap, std::string>& getAffineMapToFuncBaseName()
        {
            return affineMapToFuncBaseName;
        }

        static constexpr const char* affineIdxTypeStr = "int64_t";

        static constexpr const char* affineCeilDivStr = "affine_ceildiv";

        static constexpr const char* affineMapFuncPrefix = "affine_map_func_";

        std::string makeAffineIdxFuncName(StringRef funcBaseName, int idx)
        {
            return funcBaseName.str() + "_i" + std::to_string(idx);
        }

        llvm::StringRef getFuncBaseName(AffineMap map)
        {
            auto iter = affineMapToFuncBaseName.find(map);
            assert(iter != affineMapToFuncBaseName.end());
            return iter->second;
        }

    private:
        void checkAffineMemCpyPass(Operation* op);

        void checkDeadAffineOpsPass(Operation* op);

        void collectAffineMapsPass(Operation* op);

        void printAffineMapResultIndices(AffineMap map,
                                         Operation::operand_range origIndices,
                                         llvm::SmallVector<StringRef, 4>& memIdxVars);

        LogicalResult
        printMemRefAccessPtr(Value memRef,
                             const llvm::SmallVector<StringRef, 4>& memIdxVars,
                             std::string& srcMemRefPtr);

        LogicalResult
        printMemRefAccessValue(Value memRef,
                               const llvm::SmallVector<StringRef, 4>& memIdxVars,
                               std::string& memRefVal);

        bool needAffineMemCpy;

        // a map from an AffineMap to the base name of its corresponding function,
        // where the base name will be used to create affine_map_func for individual
        // indices.
        llvm::DenseMap<AffineMap, std::string> affineMapToFuncBaseName;
    };

} // namespace cpp_printer
} // namespace mlir

#endif
