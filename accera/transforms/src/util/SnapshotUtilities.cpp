////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "util/SnapshotUtilities.h"

#include <mlir/IR/Builders.h>
#include <mlir/Transforms/LocationSnapshot.h>

#include <llvm/Support/FormatVariadic.h>

namespace accera::transforms
{
static std::atomic<size_t> _nextPipeIdx = 0;

void IRSnapshot(const std::string& filename, mlir::Operation* op, const std::string& fileExtension)
{
    auto fullFileNameStr = filename + fileExtension;
    (void)generateLocationsFromIR(fullFileNameStr, op, mlir::OpPrintingFlags{}.printGenericOpForm());
}

IRSnapshotter::IRSnapshotter(const IRSnapshotter& other)
{
    _nextIdx.store(other._nextIdx.load());
    _options = other._options;
}

void IRSnapshotter::Snapshot(const std::string& nameSuffix, mlir::Operation* op)
{
    if (_options.EnableIntraPassSnapshots)
    {
        size_t index = _nextIdx++;
        auto combinedPrefixSuffix = llvm::formatv("{0}_{1}_{2}", _options.FileNamePrefix, index, nameSuffix);
        auto combinedPrefixSuffixStr = combinedPrefixSuffix.str();
        IRSnapshot(combinedPrefixSuffixStr, op);
    }
}

IRSnapshotter IRSnapshotter::MakeSnapshotPipe()
{
    size_t pipeIdx = _nextPipeIdx++;
    auto combinedNamePrefix = llvm::formatv("{0}_{1}", _options.FileNamePrefix, pipeIdx);
    auto combinedNamePrefixStr = combinedNamePrefix.str();
    IntraPassSnapshotOptions options{ _options.EnableIntraPassSnapshots, combinedNamePrefixStr };
    IRSnapshotter pipe(options);
    return pipe;
}
} // namespace accera::transforms
