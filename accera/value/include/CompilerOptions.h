////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Umesh Madan, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TargetDevice.h"

#include <utilities/include/PropertyBag.h>
#include <utilities/include/StringUtil.h>

#include <optional>

namespace accera
{
namespace value
{
    /// <summary> Standard compiler switches. </summary>
    struct CompilerOptions
    {
        CompilerOptions() = default;

        /// <summary> Constructor from a property bag. </summary>
        explicit CompilerOptions(const utilities::PropertyBag& properties);

        /// <summary> Create a new `CompilerOptions` by adding or overriding the options in the given `PropertyBag` </summary>
        [[nodiscard]] CompilerOptions AppendOptions(const utilities::PropertyBag& properties) const;

        // global options
        /// <summary> Optimize output code using LLVM. </summary>
        bool optimize = true;

        /// <summary> Generate position independent code (equivalent to -fPIC). </summary>
        std::optional<bool> positionIndependentCode;

        /// <summary> Emit profiling code. </summary>
        bool profile = false;

        /// <summary> Allow emitting more efficient code that isn't necessarily IEEE-754 compatible. </summary>
        bool useFastMath = true;

        /// <summary> Allow printing of diagnostic messages from the compiled model. </summary>
        bool includeDiagnosticInfo = false;

        /// <summary> Name of the target device. </summary>
        TargetDevice targetDevice = { "host" };

        // Options that can be changed during code generation (e.g., per function)
        /// <summary> Emit code that calls an external BLAS library. </summary>
        bool useBlas = true;

        /// <summary> Explicitly unroll loops in certain cases. </summary>
        bool unrollLoops = false;

        /// <summary> Emit inline code for common operations. </summary>
        bool inlineOperators = true;

        /// <summary> Enable ELL's vectorization </summary>
        bool allowVectorInstructions = false;

        /// <summary> Size of vector units. </summary>
        int vectorWidth = 4;

        /// <summary> Emit debug code. </summary>
        bool debug = false;

        /// <summary> The name of the file being compiled. </summary>
        std::string modelFile;

        /// <summary> The byte alignment to use for global values. </summary>
        unsigned globalValueAlignment = 32;

        /// <summary> Whether to bare pointer style declarations for defined functions. </summary>
        bool useBarePtrCallConv = false;

        /// <summary> Whether to emit C wrapper declarations for defined functions. </summary>
        bool emitCWrapperDecls = false;

        /// <summary> The function name prefix to give to C wrapper declarations. </summary>
        std::string cWrapperPrefix = "_mlir_ciface_";

    private:
        void AddOptions(const utilities::PropertyBag& properties);
    };

} // namespace value
} // namespace accera
