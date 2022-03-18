////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilerOptions.h"

#include <llvm/ADT/StringSwitch.h>

#define ADD_TO_STRING_ENTRY(NAMESPACE, ENTRY) \
    case NAMESPACE::ENTRY:                    \
        return #ENTRY;
#define BEGIN_FROM_STRING if (false)
#define ADD_FROM_STRING_ENTRY(NAMESPACE, ENTRY) else if (s == #ENTRY) return NAMESPACE::ENTRY

namespace accera
{
namespace value
{
    static ExecutionRuntime GetExecutionRuntime(std::string runtimeName)
    {
        return ::llvm::StringSwitch<ExecutionRuntime>(runtimeName)
            .Case("Default", ExecutionRuntime::DEFAULT)
            .Case("Vulkan", ExecutionRuntime::VULKAN)
            .Case("Rocm", ExecutionRuntime::ROCM)
            .Case("CUDA", ExecutionRuntime::CUDA)
            .Case("None", ExecutionRuntime::NONE)
            .Case("OpenMP", ExecutionRuntime::OPENMP)
            .Default(ExecutionRuntime::DEFAULT);
    }

    /// <summary> Constructor from a property bag </summary>
    CompilerOptions::CompilerOptions(const utilities::PropertyBag& properties)
    {
        AddOptions(properties);
    }

    CompilerOptions CompilerOptions::AppendOptions(const utilities::PropertyBag& properties) const
    {
        CompilerOptions result = *this;
        result.AddOptions(properties);
        return result;
    }

    void CompilerOptions::AddOptions(const utilities::PropertyBag& properties)
    {
        if (properties.HasEntry("positionIndependentCode"))
        {
            positionIndependentCode = properties.GetOrParseEntry<bool>("positionIndependentCode");
        }

        optimize = properties.GetOrParseEntry("optimize", optimize);
        unrollLoops = properties.GetOrParseEntry("unrollLoops", unrollLoops);
        inlineOperators = properties.GetOrParseEntry<bool>("inlineOperators", inlineOperators);
        allowVectorInstructions = properties.GetOrParseEntry<bool>("allowVectorInstructions", allowVectorInstructions);
        vectorWidth = properties.GetOrParseEntry<int>("vectorWidth", vectorWidth);
        useBlas = properties.GetOrParseEntry<bool>("useBlas", useBlas);
        profile = properties.GetOrParseEntry<bool>("profile", profile);
        includeDiagnosticInfo = properties.GetOrParseEntry<bool>("includeDiagnosticInfo", includeDiagnosticInfo);
        useFastMath = properties.GetOrParseEntry<bool>("useFastMath", useFastMath);
        debug = properties.GetOrParseEntry<bool>("debug", debug);
        gpu_only = properties.GetOrParseEntry<bool>("gpu_only", gpu_only);
        globalValueAlignment = properties.GetOrParseEntry<int>("globalValueAlignment", globalValueAlignment);

        if (properties.HasEntry("deviceName"))
        {
            targetDevice = GetTargetDevice(properties.GetEntry<std::string>("deviceName"));
        }
        if (properties.HasEntry("runtime"))
        {
            executionRuntime = GetExecutionRuntime(properties.GetEntry<std::string>("runtime"));
        }
    }

} // namespace value

} // namespace accera
