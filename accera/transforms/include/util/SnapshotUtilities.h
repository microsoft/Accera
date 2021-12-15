////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <atomic>
#include <string>

namespace mlir
{
class Operation;
}

namespace accera::transforms
{
// Utility function for snapshotting the IR of an operation to a file
void IRSnapshot(const std::string& filename, mlir::Operation* op, const std::string& fileExtension = ".mlir");

struct IntraPassSnapshotOptions
{
    IntraPassSnapshotOptions(bool enable = false, const std::string& fileNamePrefix = "") :
        EnableIntraPassSnapshots(enable), FileNamePrefix(fileNamePrefix) {}

    bool EnableIntraPassSnapshots;
    std::string FileNamePrefix;
};

class IRSnapshotter
{
public:
    IRSnapshotter(const IntraPassSnapshotOptions& options) :
        _options(options),
        _nextIdx(0)
    {}

    IRSnapshotter(const IRSnapshotter& other);

    IRSnapshotter MakeSnapshotPipe();
    void Snapshot(const std::string& nameSuffix, mlir::Operation* op);

private:
    IntraPassSnapshotOptions _options;
    std::atomic<size_t> _nextIdx;
};

} // namespace accera::transforms
