////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EmitterContext.h"
#include "FunctionDeclaration.h"
#include "Matrix.h"
#include "MatrixFragment.h"
#include "Scalar.h"
#include "TargetDevice.h"
#include "Value.h"
#include "Vector.h"

#include <utilities/include/Exception.h>

#include <iostream>
#include <tuple>
#include <utility>

namespace accera
{
namespace value
{
    using namespace utilities;

    EmitterContext::IfContextImpl::~IfContextImpl() = default;

    EmitterContext::IfContext::IfContext(std::unique_ptr<EmitterContext::IfContextImpl> impl) :
        _impl(std::move(impl))
    {}

    EmitterContext::IfContext&& EmitterContext::IfContext::ElseIf(Scalar test, std::function<void()> fn) &&
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch, "test variable for ElseIf must be of type Boolean but got " + ToString(test.GetType()) + " instead.");
        }

        _impl->ElseIf(test, fn);

        return std::move(*this);
    }

    EmitterContext::IfContext& EmitterContext::IfContext::ElseIf(Scalar test, std::function<void()> fn) &
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch, "test variable for ElseIf must be of type Boolean but got " + ToString(test.GetType()) + " instead.");
        }

        _impl->ElseIf(test, fn);

        return *this;
    }

    void EmitterContext::IfContext::Else(std::function<void()> fn) &&
    {
        _impl->Else(fn);
    }

    void EmitterContext::IfContext::Else(std::function<void()> fn) &
    {
        _impl->Else(fn);
    }

    EmitterContext::EmitterContext(const CompilerOptions& compilerOptions) :
        _compilerOptions(compilerOptions)
    {
        CompleteTargetDevice(_compilerOptions.targetDevice);
    }

    EmitterContext::~EmitterContext() = default;

    Value EmitterContext::Allocate(ValueType type, size_t size, size_t align, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes)
    {
        return Allocate(type, MemoryLayout(MemoryShape{ (int)size }), align, flags, /*runtimeSizes=*/{});
    }

    Value EmitterContext::Allocate(ValueType type, MemoryLayout layout, size_t align, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes)
    {
        return AllocateImpl(type, layout, align, flags, runtimeSizes);
    }

    Value EmitterContext::StaticAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        if (auto globalValue = GetGlobalValue(GlobalAllocationScope::Function, name, layout))
        {
            return *globalValue;
        }

        return GlobalAllocateImpl(GlobalAllocationScope::Function, name, type, layout, flags);
    }

    Value EmitterContext::GlobalAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        if (auto globalValue = GetGlobalValue(GlobalAllocationScope::Global, name, layout))
        {
            return *globalValue;
        }

        return GlobalAllocateImpl(GlobalAllocationScope::Global, name, type, layout, flags);
    }

    std::optional<Value> EmitterContext::GetGlobalValue(GlobalAllocationScope scope, std::string name, MemoryLayout layout)
    {
        if (auto globalValue = GetGlobalValue(scope, name))
        {
            Value value = globalValue.value();
            const auto globalMemSize = value.GetLayout().GetMemorySize();
            if (layout.GetMemorySize() > globalMemSize)
            {
                throw InputException(InputExceptionErrors::invalidSize, "Size of global value (" + std::to_string(globalMemSize) + ") is less than that of the requested layout (" + std::to_string(layout.GetMemorySize()) + ").");
            }
            value.SetLayout(layout);

            return value;
        }

        return std::nullopt;
    }

    detail::ValueTypeDescription EmitterContext::GetType(Emittable emittable)
    {
        return GetTypeImpl(emittable);
    }

    EmitterContext::DefinedFunction EmitterContext::CreateFunction(FunctionDeclaration decl, EmitterContext::DefinedFunction fn)
    {
        return CreateFunctionImpl(decl, fn);
    }

    EmitterContext::DefinedFunction EmitterContext::DeclareExternalFunction(FunctionDeclaration decl)
    {
        return DeclareExternalFunctionImpl(decl);
    }

    bool EmitterContext::IsFunctionDefined(FunctionDeclaration decl) const
    {
        return IsFunctionDefinedImpl(decl);
    }

    Value EmitterContext::StoreConstantData(ConstantData data, MemoryLayout layout, const std::string& name)
    {
        return StoreConstantDataImpl(data, layout, name);
    }

    bool EmitterContext::IsConstantData(Value v) const
    {
        if (!v.IsDefined() || v.IsEmpty()) return false;

        if (!std::holds_alternative<Emittable>(v.GetUnderlyingData())) return true;

        return IsConstantDataImpl(v);
    }

    Value EmitterContext::ResolveConstantDataReference(Value source)
    {
        return ResolveConstantDataReferenceImpl(source);
    }

    void EmitterContext::For(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name)
    {
        if (layout.NumElements() == 0)
        {
            return;
        }

        return ForImpl(layout, fn, name);
    }

    void EmitterContext::For(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name)
    {
        if (!(start.GetType() == stop.GetType() && start.GetType() == step.GetType()))
        {
            throw InputException(InputExceptionErrors::typeMismatch, "start/stop/step types must match");
        }

        if (start.GetType() == ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "start/stop/step must not be boolean");
        }

        return ForImpl(start, stop, step, fn, name);
    }

    void EmitterContext::MoveData(Value& source, Value& destination)
    {
        return MoveDataImpl(source, destination);
    }

    void EmitterContext::CopyData(const Value& source, Value& destination)
    {
        return CopyDataImpl(source, destination);
    }

    void EmitterContext::Store(const Value& source, Value& destination, const std::vector<int64_t>& indices)
    {
        return StoreImpl(source, destination, indices);
    }

    Value EmitterContext::View(Value source, const std::vector<Scalar>& offsets, const std::vector<Scalar>& newShape, const std::vector<Scalar>& strides)
    {
        return ViewImpl(source, offsets, newShape, strides);
    }

    Value EmitterContext::Slice(Value source, std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets)
    {
        return SliceImpl(source, slicedDimensions, sliceOffsets);
    }

    Value EmitterContext::MergeDimensions(Value source, int64_t dim1, int64_t dim2)
    {
        return MergeDimensionsImpl(source, dim1, dim2);
    }

    Value EmitterContext::SplitDimension(Value source, int64_t dim, Scalar size)
    {
        return SplitDimensionImpl(source, dim, size);
    }

    Value EmitterContext::Reshape(Value source, const MemoryLayout& layout)
    {
        return ReshapeImpl(source, layout);
    }

    Value EmitterContext::ReinterpretCast(Value source, ValueType type)
    {
        return ReinterpretCastImpl(source, type);
    }

    Value EmitterContext::Reorder(Value source, const utilities::DimensionOrder& order)
    {
        return ReorderImpl(source, order);
    }

    Value EmitterContext::UnaryOperation(ValueUnaryOperation op, Value value)
    {
        return UnaryOperationImpl(op, value);
    }

    Value EmitterContext::BinaryOperation(ValueBinaryOperation op, Value source1, Value source2)
    {
        return BinaryOperationImpl(op, source1, source2);
    }

    Value EmitterContext::LogicalOperation(ValueLogicalOperation op, Value source1, Value source2)
    {
        if (!source1.IsDefined() || !source2.IsDefined())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Both operands must be defined before performing logical operation.");
        }

        if (source1.GetBaseType() != source2.GetBaseType() && !IsLogicalComparable(source1.GetBaseType(), source2.GetBaseType()))
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Both operands must have logically comparable base types (" + ToString(source1.GetBaseType()) + ", " + ToString(source2.GetBaseType()) + ").");
        }

        return LogicalOperationImpl(op, source1, source2);
    }

    Value EmitterContext::MMALoadSync(const Matrix& source, const int64_t rowOffset, const int64_t colOffset, const MatrixFragment& target)
    {
        return MMALoadSyncImpl(source, rowOffset, colOffset, target);
    }

    void EmitterContext::MMAStoreSync(const MatrixFragment& source, Matrix& target, const int64_t rowOffset, const int64_t colOffset)
    {
        if (source.GetType() != target.GetValue().GetBaseType() || source.GetFragmentType() != MatrixFragment::Type::Acc)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Invalid argument passed for MMA store.");
        }
        MMAStoreSyncImpl(source, target, rowOffset, colOffset);
    }

    Value EmitterContext::MMAComputeSync(const MatrixFragment& A, const MatrixFragment& B, const MatrixFragment& C, uint32_t cbsz, uint32_t abid, uint32_t blgp)
    {
        if (A.GetFragmentShape() != B.GetFragmentShape() || A.GetFragmentShape() != C.GetFragmentShape() || A.GetFragmentType() != MatrixFragment::Type::A || B.GetFragmentType() != MatrixFragment::Type::B || C.GetFragmentType() != MatrixFragment::Type::Acc)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Invalid argument passed for MMA compute.");
        }

        return MMAComputeSyncImpl(A, B, C, cbsz, abid, blgp);
    }

    Scalar EmitterContext::Cast(Scalar value, ValueType type)
    {
        return CastImpl(value, type);
    }

    Scalar EmitterContext::Round(Scalar value)
    {
        return RoundImpl(value);
    }

    bool EmitterContext::IsImplicitlyCastable(ValueType source, ValueType target) const
    {
        return IsImplicitlyCastableImpl(source, target);
    }

    Scalar EmitterContext::Bitcast(Scalar value, ValueType type)
    {
        return BitcastImpl(value, type);
    }

    EmitterContext::IfContext EmitterContext::If(Scalar test, std::function<void()> fn)
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch, "test variable for If must be of type Boolean but got " + ToString(test.GetType()) + " instead.");
        }

        return IfImpl(test, fn);
    }

    void EmitterContext::While(Scalar test, std::function<void()> fn)
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch, "test variable for While must be of type Boolean but got " + ToString(test.GetType()) + " instead.");
        }

        return WhileImpl(test, fn);
    }

    std::optional<Value> EmitterContext::Call(FunctionDeclaration func, std::vector<ViewAdapter> args)
    {
        std::vector<Value> valueArgs(args.begin(), args.end());
        return CallImpl(func, valueArgs);
    }

    void EmitterContext::Prefetch(Value data, PrefetchType type, PrefetchLocality locality)
    {
        PrefetchImpl(data, type, locality);
    }

    void EmitterContext::DebugBreak()
    {
        DebugBreakImpl();
    }

    void EmitterContext::DebugDump(Value value, std::string tag, std::ostream* stream) const
    {
        std::ostream& outStream = stream != nullptr ? *stream : std::cerr;

        if (value.IsDefined())
        {
            DebugDumpImpl(value, tag, outStream);
        }
        else
        {
            outStream << "Value is undefined";
            if (!tag.empty())
            {
                outStream << "[tag = " << tag << "]";
            }
            outStream << "\n";
        }
    }

    void EmitterContext::DebugDump(FunctionDeclaration fn, std::string tag, std::ostream* stream) const
    {
        std::ostream& outStream = stream != nullptr ? *stream : std::cerr;

        DebugDumpImpl(fn, tag, outStream);
    }

    void EmitterContext::Print(ViewAdapter value, bool toStderr)
    {
        PrintImpl(value, toStderr);
    }

    void EmitterContext::Print(const std::string& message, bool toStderr)
    {
        PrintImpl(message, toStderr);
    }

    void EmitterContext::PrintRawMemory(ViewAdapter value)
    {
        PrintRawMemoryImpl(value);
    }

    void EmitterContext::DebugPrint(std::string message)
    {
        DebugPrintImpl(message);
    }

    void EmitterContext::SetName(const Value& value, const std::string& name)
    {
        SetNameImpl(value, name);
    }

    std::string EmitterContext::GetName(const Value& value) const
    {
        return GetNameImpl(value);
    }

    void EmitterContext::ImportCodeFile(std::string file)
    {
        ImportCodeFileImpl(file);
    }

    std::string EmitterContext::UniqueName(const std::string& prefix)
    {
        auto uniqueId = _uniqueNames[prefix]++;
        return prefix + "_" + std::to_string(uniqueId);
    }

    Scalar EmitterContext::Max(Vector input)
    {
        return MaxImpl(input);
    }

    Scalar EmitterContext::Sum(Vector input)
    {
        return SumImpl(input);
    }

    void swap(EmitterContext& l, EmitterContext& r) noexcept
    {
        std::swap(l._uniqueNames, r._uniqueNames);
    }

    namespace
    {
        EmitterContext* s_context = nullptr;
    }

    EmitterContext& GetContext()
    {
        if (s_context == nullptr)
        {
            throw LogicException(LogicExceptionErrors::illegalState, "EmitterContext is not set!");
        }

        return *s_context;
    }

    void SetContext(EmitterContext& context)
    {
        s_context = &context;
    }

    void ClearContext() noexcept
    {
        s_context = nullptr;
    }

    ContextGuard<>::ContextGuard(EmitterContext& context) :
        _oldContext(s_context)
    {
        SetContext(context);
    }

    ContextGuard<>::~ContextGuard()
    {
        _oldContext ? SetContext(*_oldContext) : ClearContext();
    }

    Value Allocate(ValueType type, size_t size, size_t align, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes)
    {
        return GetContext().Allocate(type, size, align, flags, runtimeSizes);
    }

    Value Allocate(ValueType type, MemoryLayout layout, size_t align, AllocateFlags flags, const std::vector<ScalarDimension>& runtimeSizes)
    {
        return GetContext().Allocate(type, layout, align, flags, runtimeSizes);
    }

    Value StaticAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        return GetContext().StaticAllocate(name, type, layout, flags);
    }

    Value GlobalAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        return GetContext().GlobalAllocate(name, type, layout, flags);
    }

    EmitterContext::IfContext If(Scalar test, std::function<void()> fn)
    {
        return GetContext().If(test, fn);
    }

    void While(Scalar test, std::function<void()> fn)
    {
        return GetContext().While(test, fn);
    }

    void ForRange(Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(std::string{}, end, fn);
    }

    void ForRange(const std::string& name, Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(name, Scalar{ int64_t{ 0 } }, end, fn);
    }

    void ForRange(Scalar start, Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(std::string{}, start, end, fn);
    }

    void ForRange(const std::string& name, Scalar start, Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(name, start, end, int64_t{ 1 }, fn);
    }

    void ForRange(Scalar start, Scalar end, Scalar step, std::function<void(Scalar)> fn)
    {
        ForRange(std::string{}, start, end, step, fn);
    }

    void ForRange(const std::string& name, Scalar start, Scalar end, Scalar step, std::function<void(Scalar)> fn)
    {
        GetContext().For(start, end, step, fn, name);
    }

    void ForRanges(std::vector<Scalar> range_ends, std::function<void(std::vector<Scalar>)> fn)
    {
        const auto N = range_ends.size();
        std::vector<Scalar> iter_idxs(N);
        for (auto idx = 0u; idx < N; ++idx)
        {
            ForRange(range_ends[idx], [&, idx](Scalar s) {
                iter_idxs[idx] = s;
                if (idx == N - 1)
                {
                    fn(iter_idxs);
                }
            });
        }
    }

    void DebugBreak()
    {
        GetContext().DebugBreak();
    }

    void DebugDump(FunctionDeclaration fn, std::string tag, std::ostream* stream)
    {
        GetContext().DebugDump(fn, tag, stream);
    }

    void DebugDump(Value value, std::string tag, std::ostream* stream)
    {
        GetContext().DebugDump(value, tag, stream);
    }

    void DebugPrint(std::string message)
    {
        GetContext().DebugPrint(message);
    }

    void MemCopy(ViewAdapter dest, ViewAdapter source, std::optional<Scalar> length)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "MemCopy not implemented");
    }

    void MemMove(ViewAdapter dest, ViewAdapter source, std::optional<Scalar> length)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "MemMove not implemented");
    }

    void MemSet(ViewAdapter dest, Scalar data, std::optional<Scalar> length)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "MemSet not implemented");
    }

    void MemZero(ViewAdapter dest, std::optional<Scalar> length)
    {
        // As of 9/11/2019, when compiling with C++17, `Value{ int8_t{} }` in place of `Value(int8_t{})`
        // triggers the `std::initializer_list<T>` ctor for Value, instead of the `Value<T>(T)` ctor.
        // This might change in C++20.
        MemSet(dest, Value(int8_t{}), length);
    }

    void Print(ViewAdapter value, bool toStderr)
    {
        GetContext().Print(value, toStderr);
    }

    void Print(const std::string& message, bool toStderr)
    {
        GetContext().Print(message, toStderr);
    }

    void PrintRawMemory(ViewAdapter value)
    {
        GetContext().PrintRawMemory(value);
    }

    std::string UniqueName(const std::string& prefix)
    {
        return GetContext().UniqueName(prefix);
    }

} // namespace value
} // namespace accera
