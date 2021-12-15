////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Index.h"
#include "IterationDomain.h"
#include "LoopIndexInfo.h"
#include "LoopNestAttributes.h"
#include "LoopNestTypes.h"
#include "LoopVisitSchedule.h"
#include "Util.h"

#include <ir/include/value/ValueEnums.h>

#include <utilities/include/FunctionUtils.h>

#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <array>
#include <functional>
#include <optional>
#include <vector>

namespace accera::ir
{
// mlir-tblgen currently creates files with the assumption that the following
// symbols are present in the current namespace, so we have to import them
// explicitly
using llvm::APInt;
using llvm::ArrayRef;
using llvm::cast;
using llvm::iterator_range;
using llvm::Optional;
using llvm::SmallVectorImpl;
using llvm::StringRef;

using mlir::AffineMap;
using mlir::AffineValueMap;
using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::DictionaryAttr;
using mlir::ElementsAttr;
using mlir::FlatSymbolRefAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::LoopLikeOpInterface;
using mlir::MemoryEffectOpInterface;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::OwningRewritePatternList;
using mlir::ParseResult;
using mlir::Region;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::success;
using mlir::SymbolOpInterface;
using mlir::SymbolRefAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;
using mlir::ValueRange;

namespace SideEffects = mlir::SideEffects;
namespace MemoryEffects = mlir::MemoryEffects;
namespace OpTrait = mlir::OpTrait;

namespace loopnest
{
    class TerminatorOp;

    template <typename T>
    struct SplitIndexT
    {
        T outer;
        T inner;
    };
    using KernelId = std::string;

#include "nest/LoopNestExportedInterfaces.h.inc"
#include "nest/LoopNestInterfaces.h.inc"
} // namespace loopnest
} // namespace accera::ir

// Include the auto-generated header file containing the declarations of the loopnest operations.
#define GET_OP_CLASSES
#include "nest/LoopNestDialect.h.inc"
#include "nest/LoopNestOps.h.inc"

namespace accera::ir::loopnest
{

using SplitSymbolicIndex = SplitIndexT<SymbolicIndexOp>;

//
// Utility functions and EDSC-type intrinsics
//
NestOp MakeNest(mlir::OpBuilder& builder, const IterationDomain& domain);
NestOp MakeNest(mlir::OpBuilder& builder, mlir::ArrayRef<int64_t> sizes);
NestOp MakeNest(mlir::OpBuilder& builder, mlir::ArrayRef<mlir::Value> sizes);

//
// ScheduleOp fusing
//

/// Fuse two schedules, destroying them and creating a new, equivalent schedule
/// Returns the new schedule and the "fusing" index. The fusing index is the outermost loop.
std::tuple<ScheduleOp, Index> Fuse(mlir::OpBuilder& builder, ScheduleOp schedule1, ScheduleOp schedule2);

/// Fuse two schedules, destroying them and creating a new, equivalent schedule
///
/// `indexCorrespondences` is a list of index pairs, indicating which indices are being fused together. To indicate an index
/// without a correspondence, use the "null" index (`Index::none`) for the other half of the pair.
///
/// Returns the new schedule and the "fusing" index. The fusing index is the outermost loop, and the rest
/// follow the order in `indexCorrespondences`. Any indices in the original nests not in `indexCorrespondences` will be added
/// to the end (they will be the innermost loops), starting with the ones from the first nest, and then the second.
std::tuple<ScheduleOp, Index> Fuse(mlir::OpBuilder& builder, ScheduleOp schedule1, ScheduleOp schedule2, const std::vector<std::pair<Index, Index>>& indexCorrespondences);

std::tuple<ScheduleOp, Index> Fuse(
    mlir::OpBuilder& builder,
    const std::vector<ScheduleOp>& schedules,
    const std::vector<std::vector<Index>>& indexCorrespondences);

// Kernels
std::string GetUniqueKernelId(std::string id, mlir::Region* region);

KernelOp MakeKernel(mlir::OpBuilder& builder, std::function<void(mlir::OpBuilder&, mlir::Location)> body);
KernelOp MakeKernel(mlir::OpBuilder& builder, std::string id, std::function<void(mlir::OpBuilder&, mlir::Location)> body);

ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, KernelPredicateOpInterface predicate);
ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, KernelPredicateOpInterface predicate, KernelPredicateOpInterface placementPredicate);
ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, EvaluatablePredicateOpInterface predicate);

template <typename PredType>
ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, PredType predicate);

template <typename PredType1, typename PredType2>
ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, PredType1 predicate, PredType2 placementPredicate);

// Predicates
KernelPredicateOpInterface First(mlir::OpBuilder& builder, SymbolicIndexOp index);
KernelPredicateOpInterface First(mlir::OpBuilder& builder, Index index);
KernelPredicateOpInterface Last(mlir::OpBuilder& builder, SymbolicIndexOp index);
KernelPredicateOpInterface Last(mlir::OpBuilder& builder, Index index);
KernelPredicateOpInterface IndexAt(mlir::OpBuilder& builder, SymbolicIndexOp index, int64_t value);
KernelPredicateOpInterface IndexAt(mlir::OpBuilder& builder, Index index, int64_t value);
KernelPredicateOpInterface EndBoundary(mlir::OpBuilder& builder, SymbolicIndexOp index);
KernelPredicateOpInterface EndBoundary(mlir::OpBuilder& builder, Index index);
KernelPredicateOpInterface Before(mlir::OpBuilder& builder, SymbolicIndexOp index);
KernelPredicateOpInterface Before(mlir::OpBuilder& builder, Index index);
KernelPredicateOpInterface After(mlir::OpBuilder& builder, SymbolicIndexOp index);
KernelPredicateOpInterface After(mlir::OpBuilder& builder, Index index);
KernelPredicateOpInterface IsDefined(mlir::OpBuilder& builder, SymbolicIndexOp index);
KernelPredicateOpInterface IsDefined(mlir::OpBuilder& builder, Index index);
KernelPredicateOpInterface InRange(mlir::OpBuilder& builder, Index index, Range range);
KernelPredicateOpInterface InRange(mlir::OpBuilder& builder, SymbolicIndexOp index, Range range);

KernelPredicateOpInterface Conjunction(mlir::OpBuilder& builder, KernelPredicateOpInterface lhs, KernelPredicateOpInterface rhs);
KernelPredicateOpInterface Disjunction(mlir::OpBuilder& builder, KernelPredicateOpInterface lhs, KernelPredicateOpInterface rhs);

//
// Templated method implementations
//
template <int N>
std::array<SymbolicIndexOp, N> NestOp::getIndices(OpBuilder& builder)
{
    auto domain = getDomain().getValue();
    assert(domain.NumDimensions() >= N);
    auto dimensions = domain.GetDimensions();

    std::array<SymbolicIndexOp, N> result;

    for (int i = 0; i < N; ++i)
    {
        result[i] = getOrCreateSymbolicIndex(builder, dimensions[i]);
    }

    return result;
}

template <int N>
std::array<SymbolicIndexOp, N> ScheduleOp::getIndices(OpBuilder& builder)
{
    auto domain = getDomain().getValue();
    assert(domain.NumDimensions() >= N);
    auto dimensions = domain.GetDimensions();

    std::array<SymbolicIndexOp, N> result;

    for (int i = 0; i < N; ++i)
    {
        result[i] = getOrCreateSymbolicIndex(builder, dimensions[i]);
    }

    return result;
}

template <typename PredType>
ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, PredType predicate)
{
    if (auto kernelPredicate = mlir::dyn_cast_or_null<KernelPredicateOpInterface>(predicate.getOperation()))
    {
        return MakeKernel(builder, kernel, kernelPredicate);
    }
    else
    {
        return MakeKernel(builder, kernel, mlir::dyn_cast_or_null<EvaluatablePredicateOpInterface>(predicate.getOperation()));
    }
}

template <typename PredType1, typename PredType2>
ScheduledKernelOp MakeKernel(mlir::OpBuilder& builder, KernelOp kernel, PredType1 predicate, PredType2 placementPredicate)
{
    return MakeKernel(builder, kernel, mlir::dyn_cast_or_null<KernelPredicateOpInterface>(predicate.getOperation()), mlir::dyn_cast_or_null<KernelPredicateOpInterface>(placementPredicate.getOperation()));
}

} // namespace accera::ir::loopnest
