////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Emittable.h"
#include "Scalar.h"
#include "Value.h"
#include "ValueType.h"

#include <value/include/CompilerOptions.h>

#include <utilities/include/Boolean.h>
#include <utilities/include/EnumFlagHelpers.h>
#include <utilities/include/FunctionUtils.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TupleUtils.h>

#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace accera
{
namespace value
{
    class FunctionDeclaration;
    class Array;
    class Matrix;
    class Vector;
    enum class PrefetchType;
    enum class PrefetchLocality;

    enum class AllocateFlags : uint64_t
    {
        None = 0,
        ThreadLocal = 1 << 0,
        Stack = 1 << 1,
    };
    ACCERA_DEFINE_ENUM_FLAG_OPERATORS(AllocateFlags);

    /// <summary> An interface describing the global context that's used by the Value library </summary>
    /// <remarks> This class employs the non-virtual interface pattern to provide an easy to use API while
    /// minimizing the functions needed to be overloaded. </remarks>
    class EmitterContext
    {
    protected:
        using MemoryLayout = utilities::MemoryLayout;
        using MemoryCoordinates = utilities::MemoryCoordinates;

        class IfContextImpl
        {
        public:
            virtual ~IfContextImpl();
            virtual void ElseIf(Scalar, std::function<void()>) = 0;
            virtual void Else(std::function<void()>) = 0;
        };

        enum class GlobalAllocationScope
        {
            Global,
            Function
        };

    public:
        using DefinedFunction = std::function<std::optional<Value>(std::vector<Value>)>;

        class IfContext
        {
        public:
            IfContext(std::unique_ptr<IfContextImpl> impl);
            IfContext(const IfContext&) = delete;
            IfContext(IfContext&&) = default;

            IfContext&& ElseIf(Scalar, std::function<void()>) &&;
            void Else(std::function<void()>) &&;

            IfContext& ElseIf(Scalar, std::function<void()>) &;
            void Else(std::function<void()>) &;

        private:
            std::unique_ptr<IfContextImpl> _impl;
        };

        /// <summary> Describes the type that can be used to represent constant C++ data </summary>
        using ConstantData = detail::ConstantData;

        EmitterContext(const CompilerOptions& compilerOptions = {});

        virtual ~EmitterContext();

        /// <summary> Allocates data with the specified type and size </summary>
        /// <param name="type"> The type of the data to allocate </param>
        /// <param name="size"> The size of the allocation, in number of elements </param>
        /// <param name="alignment"> The byte alignment to use for the allocated value. </summary>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        Value Allocate(ValueType type, size_t size, size_t alignment = 0, AllocateFlags flags = AllocateFlags::None);

        /// <summary> Allocates data with the specified type and size </summary>
        /// <param name="type"> The type of the data to allocate </param>
        /// <param name="layout"> The memory layout of the allocation, in number of elements </param>
        /// <param name="alignment"> The byte alignment to use for the allocated value. </summary>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        Value Allocate(ValueType type, MemoryLayout layout, size_t alignment = 0, AllocateFlags flags = AllocateFlags::None);

        /// <summary> Allocates function static data </summary>
        /// <param name="name"> The name of the variable </param>
        /// <param name="type"> The type of the data </param>
        /// <param name="layout"> The layout of the data </param>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        Value StaticAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags = AllocateFlags::None);

        /// <summary> Allocates function static data </summary>
        /// <param name="name"> The name of the variable </param>
        /// <param name="data"> The data </param>
        /// <param name="layout"> The layout of the data </param>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
        Value StaticAllocate(std::string name, const std::vector<T>& data, std::optional<utilities::MemoryLayout> layout = {}, AllocateFlags flags = AllocateFlags::None)
        {
            if (auto globalValue = GetGlobalValue(GlobalAllocationScope::Function, name))
            {
                return globalValue.value();
            }

            auto optionalLayout = utilities::MemoryLayout(utilities::MemoryShape{ static_cast<int64_t>(data.size()) });
            return GlobalAllocateImpl(GlobalAllocationScope::Function, name, data, layout.value_or(optionalLayout), flags);
        }

        /// <summary> Allocates scalar function static data </summary>
        /// <param name="name"> The name of the variable </param>
        /// <param name="data"> The data </param>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
        Value StaticAllocate(std::string name, T t, AllocateFlags flags = AllocateFlags::None)
        {
            return this
                ->template StaticAllocate(name,
                                          std::vector<
                                              std::conditional_t<std::is_same_v<T, bool>, utilities::Boolean, T>>{ t },
                                          flags);
        }

        /// <summary> Allocates global data </summary>
        /// <param name="name"> The name of the variable </param>
        /// <param name="type"> The type of the data </param>
        /// <param name="layout"> The layout of the data </param>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        Value GlobalAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags = AllocateFlags::None);

        /// <summary> Allocates global data </summary>
        /// <param name="name"> The name of the variable </param>
        /// <param name="data"> The data </param>
        /// <param name="layout"> The layout of the data </param>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
        Value GlobalAllocate(std::string name, const std::vector<T>& data, std::optional<utilities::MemoryLayout> layout = {}, AllocateFlags flags = AllocateFlags::None)
        {
            if (auto globalValue = GetGlobalValue(GlobalAllocationScope::Global, name))
            {
                return globalValue.value();
            }

            auto optionalLayout = utilities::MemoryLayout(utilities::MemoryShape{ static_cast<int64_t>(data.size()) });
            return GlobalAllocateImpl(GlobalAllocationScope::Global, name, data, layout.value_or(optionalLayout), flags);
        }

        /// <summary> Allocates scalar global data </summary>
        /// <param name="name"> The name of the variable </param>
        /// <param name="data"> The data </param>
        /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
        Value GlobalAllocate(std::string name, T t, AllocateFlags flags = AllocateFlags::None)
        {
            return this
                ->template GlobalAllocate(name,
                                          std::vector<
                                              std::conditional_t<std::is_same_v<T, bool>, utilities::Boolean, T>>{ t },
                                          flags);
        }

        /// <summary> Gets the type information contained in an instance of Emittable </summary>
        /// <param name="emittable"> The instance of Emittable to be queried </param>
        /// <returns> A std::pair instance describing the fundamental type of data, along with the number of pointers </returns>
        detail::ValueTypeDescription GetType(Emittable emittable);

        /// <summary> Creates a callable function </summary>
        /// <param name="decl"> The function declaration describing the function </param>
        /// <param name="fn"> The function that defines the function body to be executed when the callable function is called </param>
        /// <returns> A callable function that executes the body described by fn </returns>
        DefinedFunction CreateFunction(FunctionDeclaration decl, DefinedFunction fn);

        /// <summary> Declares an external function </summary>
        /// <param name="decl"> The function declaration describing the function </param>
        /// <returns> A callable function declaration that calls the external function </returns>
        DefinedFunction DeclareExternalFunction(FunctionDeclaration decl);

        /// <summary> Returns true if function is defined for this context, false otherwise </summary>
        /// <param name="decl"> The function declaration describing the function </param>
        bool IsFunctionDefined(FunctionDeclaration decl) const;

        /// <summary> Stores data known ahead of time in the form of a std::vector of one of the fundamental types </summary>
        /// <param name="data"> The data that is to be stored by the context instance </param>
        /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
        Value StoreConstantData(ConstantData data, MemoryLayout layout, const std::string& name);

        /// <summary> Makes a reference to a constant data source. This data source may originate from another context </summary>
        /// <param name="constantDataSource"> The constant data source that will be referenced from this context </param>
        /// <returns> An instance of Value that contains a reference to the source Value </returns>
        Value ResolveConstantDataReference(Value constantDataSource);

        /// <summary> Creates a for loop over the memory pointed to with the given layout </summary>
        /// <param name="layout"> The layout used to describe the iteration characteristics. Only active elements are iterated over </param>
        /// <param name="fn"> The function to be called for each coordinate where there is an active element </param>
        /// <param name="name"> Optional, a name that can be used by the emitter context to tag this loop in the emitted code </param>
        void For(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name = "");

        /// <summary> Creates a for loop beggining at `start`, ending at `stop`, and incrementing by `step` </summary>
        /// <param name="start"> The value used to initialize the loop counter </param>
        /// <param name="stop"> The terminal value of the loop </param>
        /// <param name="step"> The value by which the loop counter is incremented </param>
        /// <param name="fn"> The function to be called for each coordinate where there is an active element </param>
        /// <param name="name"> Optional, a name that can be used by the emitter context to tag this loop in the emitted code </param>
        void For(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name = "");

        /// <summary> Moves the data from one location to another </summary>
        /// <param name="source"> The source of the memory to be moved </param>
        /// <param name="destination"> The destination of the memory to be moved </param>
        /// <remarks> The source and destination instances must be constrained and have matching MemoryLayouts </remarks>
        void MoveData(Value& source, Value& destination);

        /// <summary> Copies the data from one location to another </summary>
        /// <param name="source"> The source of the memory to be copied </param>
        /// <param name="destination"> The destination of the memory to be copied </param>
        /// <remarks> The source and destination instances must be constrained and have matching MemoryLayouts </remarks>
        void CopyData(const Value& source, Value& destination);

        /// <summary> Returns a view of a portion of a memory buffer </summary>
        /// <param name="source"> The source location </param>
        /// <param name="offsets"> The origin of the view --- the indices of the first entry in the subarray </param>
        /// <param name="newShape"> The shape of the view </param>
        /// <param name="strides"> The strides of the view </param>
        /// <returns> A Value instance that refers to the sub-part of the input </returns>
        /// <remarks> source must be constrained and the number of items in offset must match the degree of source </remarks>
        Value View(Value source, const std::vector<Scalar>& offsets, const utilities::MemoryShape& newShape, const std::vector<int64_t>& strides);

        /// <summary> Returns a slice of a portion of a memory buffer </summary>
        /// <param name="source"> The source location </param>
        /// <param name="slicedDimensions"> The dimensions to remove from the domain. </param>
        /// <param name="sliceOffsets"> The index of the slice to keep for each of the sliced dimensions. </param>
        /// <returns> A Value instance that refers to the sliced portion of the input </returns>
        /// <remarks> source must be constrained </remarks>
        Value Slice(Value source, std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets);

        /// <summary> Get a view of a memory buffer where 2 (contiguous) dimensions are merged </summary>
        /// <param name="source"> The source location </param>
        /// <param name="dim1"> One of the dimensions to merge </param>
        /// <param name="dim2"> The other dimension to merge </param>
        /// <remarks>
        /// The given dimensions must be adjacent in memory, and the smaller dimension to be merged
        /// must have the full extent of the underlying memory
        /// </remarks>
        Value MergeDimensions(Value source, int64_t dim1, int64_t dim2);

        /// <summary> Get a view of a memory buffer where a dimension is split into 2 dimensions </summary>
        /// <param name="source"> The source location </param>
        /// <param name="dim"> The dimension to split </param>
        /// <param name="size"> The extent of the new inner dimension </param>
        /// <remarks>
        /// The new dimension will be placed immediately after the split dimension
        /// The dimension being split must have full extent (and not be a sub-view of some other array)
        /// The extent (and size) of the dimension being split must be a multiple of the split size
        /// </remarks>
        Value SplitDimension(Value source, int64_t dim, int64_t size);

        /// <summary> Returns a view of a memory buffer using a different layout </summary>
        /// <param name="source"> The source location </param>
        /// <param name="layout"> The memory layout for the new result view to use </param>
        /// <remarks> The total memory size of the original layout and the new layout must match </remarks>
        Value Reshape(Value source, const MemoryLayout& layout);

        /// <summary> Get a view of a memory buffer using a different logical ordering of dimensions </summary>
        /// <param name="source"> The source location </param>
        /// <param name="order"> The order for the new view to use </param>
        /// <remarks> This operation doesn't alter any memory: it just returns a view of the array with a different logical ordering. </remarks>
        Value Reorder(Value source, const utilities::DimensionOrder& order);

        /// <summary> Performs a unary operation </summary>
        /// <param name="op"> The unary operation to perform </param>
        /// <param name="source"> The data over which to perform the operation </param>
        /// <returns> An instance of Value pointing to the result </returns>
        Value UnaryOperation(ValueUnaryOperation op, Value source);

        /// <summary> Performs a binary operation </summary>
        /// <param name="op"> The binary operation to perform </param>
        /// <param name="source1"> The data which is considered the first operand </param>
        /// <param name="source2"> The data which is considered the second operand </param>
        /// <returns> An instance of Value pointing to the result </returns>
        Value BinaryOperation(ValueBinaryOperation op, Value source1, Value source2);

        Value LogicalOperation(ValueLogicalOperation op, Value source1, Value source2);

        /// <summary> Performs matrix multiply load operation.
        /// There are restrictions on the input types and sizes. </summary>
        /// <param name="source"> The input memref </param>
        /// <param name="shape"> The shape of the load </param>
        /// <param name="operand"> The kind of the mfma matrix </param>
        Matrix MFMALoad(Value source, const std::vector<int64_t> & shape, const std::string & operand);

        /// <summary> Performs matrix multiply store operation.
        /// There are restrictions on the source type. </summary>
        /// <param name="source"> The input mfma matrix </param>
        /// <param name="target"> The target memref </param>
        void MFMAStore(Matrix source, Value target);

        /// <summary> Performs matrix multiply accumulate compute operation D = A.B + C.
        /// This operation assumes that A, B, C, and D have been loaded using the MFMALoad operation.
        /// There are restrictions on the input types and sizes. </summary>
        /// <param name="A"> The input A mfma matrix </param>
        /// <param name="B"> The input B mfma matrix </param>
        /// <param name="C"> The input C mfma matrix </param>
        /// <returns> The result destination mfma matrix </returns>
        Matrix MFMACompute(Matrix A, Matrix B, Matrix C); 

        Scalar Max(Vector input);

        Scalar Sum(Vector input);

        Scalar Cast(Scalar value, ValueType type);

        Scalar UnsignedCast(Scalar value, ValueType type);

        Scalar Bitcast(Scalar value, ValueType type);

        IfContext If(Scalar test, std::function<void()> fn);

        void While(Scalar test, std::function<void()> fn);

        std::optional<Value> Call(FunctionDeclaration func, std::vector<ViewAdapter> args);

        void Prefetch(Value data, PrefetchType type, PrefetchLocality locality);

        void DebugBreak();
        void DebugDump(Value value, std::string tag, std::ostream* stream) const;
        void DebugDump(FunctionDeclaration fn, std::string tag, std::ostream* stream) const;

        /// <summary> Returns a unique name based on the prefix provided </summary>
        /// <param name="prefix"> The prefix for the unique name desired </param>
        /// <returns> A unique name for this instance </returns>
        std::string UniqueName(const std::string& prefix);

        /// <summary> Print a value </summary>
        /// <param name="value"> The value to print </param>
        /// <param name="toStderr"> Prints to stderr if true, otherwise prints to stdout </param>
        void Print(ViewAdapter value, bool toStderr = false);

        /// <summary> Print a string </summary>
        /// <param name="message"> The message to print </param>
        /// <param name="toStderr"> Prints to stderr if true, otherwise prints to stdout </param>
        void Print(const std::string& message, bool toStderr = false);

        /// <summary> Print a value's raw memory </summary>
        /// <param name="value"> The value to print </param>
        void PrintRawMemory(ViewAdapter value);

        /// <summary> Emit a debug print message.  This assumes the application
        /// on the target platform implements a "void DebugPrint(char* message)" function.  This function will be
        /// defined for you when running in JIT or Compute mode.  </summary>
        void DebugPrint(std::string message);

        /// <summary> Set the name for the value instance </summary>
        /// <param name="value"> The Value instance </param>
        /// <param name="name"> The name </param>
        void SetName(const Value& value, const std::string& name);

        /// <summary> Gets the name for the value instance. If `SetName` has not been called, this will return the name
        /// chosen by the emitter context, if any. </summary>
        /// <param name="value"> The Value instance </param>
        [[nodiscard]] std::string GetName(const Value& value) const;

        void ImportCodeFile(std::string filename);

        [[nodiscard]] const CompilerOptions& GetCompilerOptions() const { return _compilerOptions; }
        [[nodiscard]] const TargetDevice& GetTargetDevice() const { return _compilerOptions.targetDevice; }

        /// <summary> Set the MemoryLayout for the Value instance </summary>
        /// <param name="v"> The Value instance to update </param>
        /// <param name="l"> The MemoryLayout to be set on the Value instance </param>
        void SetLayout(Value& v, const MemoryLayout& l)
        {
            SetLayoutImpl(v, l);
            v._layout = l;
        }

        virtual ViewAdapter ReduceN(Scalar begin, Scalar end, Scalar increment, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(Scalar, std::vector<ViewAdapter>)>) = 0;
        virtual ViewAdapter Reduce(Array a, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(ViewAdapter, std::vector<ViewAdapter>)>) = 0;
        virtual ViewAdapter MapReduce(Array a, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(ViewAdapter)> mapFn, std::function<ViewAdapter(ViewAdapter, std::vector<ViewAdapter>)> reduceFn) = 0;

        virtual void ReturnValue(ViewAdapter view) = 0;

        virtual Scalar GetTime() = 0;
        virtual void EnterProfileRegion(const std::string& regionName) = 0;
        virtual void ExitProfileRegion(const std::string& regionName) = 0;
        virtual void PrintProfileResults() = 0;

    protected:
        std::optional<Value> GetGlobalValue(GlobalAllocationScope, std::string, MemoryLayout);

        std::map<std::string, int> _uniqueNames;

    private:
        virtual void SetLayoutImpl(Value&, const MemoryLayout&) {}
        virtual Value AllocateImpl(ValueType, MemoryLayout, size_t alignment, AllocateFlags flags) = 0;

        virtual std::optional<Value> GetGlobalValue(GlobalAllocationScope scope, std::string name) = 0;
        virtual Value GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ConstantData data, MemoryLayout layout, AllocateFlags flags) = 0;
        virtual Value GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ValueType type, MemoryLayout layout, AllocateFlags flags) = 0;

        virtual detail::ValueTypeDescription GetTypeImpl(Emittable) = 0;

        virtual DefinedFunction CreateFunctionImpl(FunctionDeclaration decl, DefinedFunction fn) = 0;
        virtual DefinedFunction DeclareExternalFunctionImpl(FunctionDeclaration decl) = 0;
        virtual bool IsFunctionDefinedImpl(FunctionDeclaration decl) const = 0;

        virtual Value StoreConstantDataImpl(ConstantData data, MemoryLayout layout, const std::string& name) = 0;
        virtual Value ResolveConstantDataReferenceImpl(Value constantDataSource) = 0;

        virtual void ForImpl(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name) = 0;
        virtual void ForImpl(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name) = 0;

        virtual void MoveDataImpl(Value& source, Value& destination) = 0;

        virtual void CopyDataImpl(const Value& source, Value& destination) = 0;

        virtual Value ViewImpl(Value source, const std::vector<Scalar>& offsets, const utilities::MemoryShape& newShape, const std::vector<int64_t>& strides) = 0;

        virtual Value SliceImpl(Value source, std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets) = 0;

        virtual Value MergeDimensionsImpl(Value source, int64_t dim1, int64_t dim2) = 0;

        virtual Value SplitDimensionImpl(Value source, int64_t dim, int64_t size) = 0;

        virtual Value ReshapeImpl(Value source, const MemoryLayout& layout) = 0;

        virtual Value ReorderImpl(Value source, const utilities::DimensionOrder& order) = 0;

        virtual Value UnaryOperationImpl(ValueUnaryOperation op, Value source) = 0;

        virtual Value BinaryOperationImpl(ValueBinaryOperation op, Value source1, Value source2) = 0;

        virtual Value LogicalOperationImpl(ValueLogicalOperation op, Value source1, Value source2) = 0;

        virtual Matrix MFMALoadImpl(Value source, const std::vector<int64_t> & shape, const std::string & operand) = 0;

        virtual void MFMAStoreImpl(Matrix source, Value target) = 0;

        virtual Matrix MFMAComputeImpl(Matrix A, Matrix B, Matrix C) = 0;

        virtual Scalar MaxImpl(Vector input) = 0;

        virtual Scalar SumImpl(Vector input) = 0;

        virtual Scalar CastImpl(Scalar value, ValueType type) = 0;

        virtual Scalar UnsignedCastImpl(Scalar value, ValueType type) = 0;

        virtual Scalar BitcastImpl(Scalar value, ValueType type) = 0;

        virtual IfContext IfImpl(Scalar test, std::function<void()> fn) = 0;

        virtual void WhileImpl(Scalar test, std::function<void()> fn) = 0;

        virtual std::optional<Value> CallImpl(FunctionDeclaration func, std::vector<Value> args) = 0;

        virtual void PrefetchImpl(Value data, PrefetchType type, PrefetchLocality locality) = 0;

        virtual void DebugBreakImpl() = 0;

        virtual void DebugDumpImpl(Value value, std::string tag, std::ostream& stream) const = 0;
        virtual void DebugDumpImpl(FunctionDeclaration fn, std::string tag, std::ostream& stream) const = 0;

        virtual void PrintImpl(ViewAdapter value, bool toStderr) = 0;
        virtual void PrintImpl(const std::string& value, bool toStderr) = 0;

        virtual void PrintRawMemoryImpl(ViewAdapter value) = 0;

        virtual void DebugPrintImpl(std::string message) = 0;

        virtual void SetNameImpl(const Value& value, const std::string& name) = 0;
        virtual std::string GetNameImpl(const Value& value) const = 0;

        virtual void ImportCodeFileImpl(std::string filename) = 0;

        friend void swap(EmitterContext&, EmitterContext&) noexcept;

        CompilerOptions _compilerOptions;
    };

    /// <summary> Returns the global instance of EmitterContext </summary>
    /// <remarks> This is the instance used by all operations in the Value library. As such, it must be set before using
    /// any of the APIs provided in the Value library </summary>
    EmitterContext& GetContext();

    /// <summary> Sets the global instance of EmitterContext </summary>
    /// <param name="context"> The context to set as the global instance </param>
    void SetContext(EmitterContext& context);

    /// <summary> Clears the global instance of EmitterContext </summary>
    void ClearContext() noexcept;

    namespace detail
    {
        template <bool TakesContext, typename Fn, typename ContextType>
        struct InvokeForContextHelper
        {
            using ResultType = std::invoke_result_t<Fn, ContextType&>;
        };

        template <typename Fn, typename ContextType>
        struct InvokeForContextHelper<false, Fn, ContextType>
        {
            using ResultType = std::invoke_result_t<Fn>;
        };

        template <typename Fn, typename ContextType>
        using InvokeForContextHelperT = typename InvokeForContextHelper<std::is_invocable_v<Fn, ContextType&>, Fn, ContextType>::ResultType;
    } // namespace detail

    /// <summary> Invokes the provided function object if the GlobalContext is of the provided ContextType </summary>
    /// <typeparam name="ContextType"> The specific context derived from EmitterContext </typeparam>
    /// <param name="fn"> The function object to call, which takes a lvalue-reference of ContextType </param>
    /// <returns> The return value of `fn`, wrapped in a `std::optional` </returns>
    template <typename ContextType, typename Fn, typename ReturnType = detail::InvokeForContextHelperT<Fn, ContextType>>
    auto InvokeForContext(Fn&& fn) -> std::conditional_t<std::is_same_v<ReturnType, void>, void, std::optional<ReturnType>>;

    /// <summary> A helper RAII class to set a particular EmitterContext instance as the global context and unset it at the end of scope </summary>
    template <typename T = void, bool b = std::is_base_of_v<EmitterContext, T>>
    struct ContextGuard;

    template <>
    struct ContextGuard<void, false>
    {
        /// <summary> Constructor </summary>
        /// <param name="context"> The instance of EmitterContext to set as the global context </param>
        ContextGuard(EmitterContext& context);

        /// <summary> Destructor for the instance. Sets the global context to nullptr </summary>
        ~ContextGuard();

        ContextGuard(const ContextGuard&) = delete;
        ContextGuard(ContextGuard&&) = delete;
        ContextGuard& operator=(const ContextGuard&) = delete;
        ContextGuard& operator=(ContextGuard&&) = delete;

    private:
        EmitterContext* _oldContext;
    };

    template <typename T>
    struct ContextGuard<T, true> : private ContextGuard<>
    {
        template <typename... Args>
        ContextGuard(Args&&... args) :
            ContextGuard<>(_context),
            _context(std::forward<Args>(args)...)
        {}

        T& GetContext() { return _context; }

    private:
        T _context;
    };

    inline const CompilerOptions& GetContextCompilerOptions()
    {
        return GetContext().GetCompilerOptions();
    }

    inline const TargetDevice& GetContextTargetDevice()
    {
        return GetContext().GetTargetDevice();
    }

    /// <summary> Allocates data with the specified type and size </summary>
    /// <param name="type"> The type of the data to allocate </param>
    /// <param name="size"> The size of the allocation, in number of elements </param>
    /// <param name="alignment"> The byte alignment to use for the allocated value. </summary>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
    Value Allocate(ValueType type, size_t size, size_t alignment = 0, AllocateFlags flags = AllocateFlags::None);

    inline Value Allocate(ValueType type, size_t size, AllocateFlags flags)
    {
        return Allocate(type, size, 0, flags);
    }

    /// <summary> Allocates data with the specified type and size </summary>
    /// <param name="type"> The type of the data to allocate </param>
    /// <param name="layout"> The memory layout of the allocation, in number of elements </param>
    /// <param name="alignment"> The byte alignment to use for the allocated value. </summary>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
    Value Allocate(ValueType type, utilities::MemoryLayout layout, size_t alignment = 0, AllocateFlags flags = AllocateFlags::None);

    /// <summary> Allocates data with the specified type and size </summary>
    /// <param name="type"> The type of the data to allocate </param>
    /// <param name="layout"> The memory layout of the allocation, in number of elements </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
    inline Value Allocate(ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        return Allocate(type, layout, 0, flags);
    }

    /// <summary> Allocates data with the specified type and size </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="size"> The size of the allocation, in number of elements </param>
    /// <param name="alignment"> The byte alignment to use for the allocated value. </summary>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
    template <typename T>
    Value Allocate(size_t size, size_t alignment = 0, AllocateFlags flags = AllocateFlags::None)
    {
        return Allocate(GetValueType<T>(), size, alignment, flags);
    }

    /// <summary> Allocates data with the specified type and size </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="size"> The size of the allocation, in number of elements </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
    template <typename T>
    Value Allocate(size_t size, AllocateFlags flags)
    {
        return Allocate<T>(size, 0, flags);
    }

    /// <summary> Allocates data with the specified type and size </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="layout"> The memory layout of the allocation, in number of elements </param>
    /// <param name="alignment"> The byte alignment to use for the allocated value. </summary>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    /// <returns> An instance of Value that contains a reference to the allocated memory </returns>
    template <typename T>
    Value Allocate(utilities::MemoryLayout layout, size_t alignment = 0, AllocateFlags flags = AllocateFlags::None)
    {
        return Allocate(GetValueType<T>(), layout, alignment, flags);
    }

    /// <summary> Allocates data with the specified type and size </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="layout"> The memory layout of the allocation, in number of elements </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    template <typename T>
    Value Allocate(utilities::MemoryLayout layout, AllocateFlags flags)
    {
        return Allocate<T>(layout, 0, flags);
    }

    /// <summary> Allocates function static data </summary>
    /// <param name="name"> The name of the variable </param>
    /// <param name="type"> The type of the data </param>
    /// <param name="layout"> The layout of the data </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    Value StaticAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags = AllocateFlags::None);

    /// <summary> Allocates function static data </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="name"> The name of the variable </param>
    /// <param name="data"> The data </param>
    /// <param name="layout"> The layout of the data </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
    Value StaticAllocate(std::string name, const std::vector<T>& data, std::optional<utilities::MemoryLayout> layout = {}, AllocateFlags flags = AllocateFlags::None)
    {
        return GetContext().StaticAllocate(name, data, layout, flags);
    }

    /// <summary> Allocates scalar function static data </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="name"> The name of the variable </param>
    /// <param name="value"> The variable's value </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
    Scalar StaticAllocate(std::string name, T value, AllocateFlags flags = AllocateFlags::None)
    {
        return StaticAllocate(name,
                              std::vector<std::conditional_t<std::is_same_v<T, bool>, utilities::Boolean, T>>{ value },
                              utilities::ScalarLayout,
                              flags);
    }

    /// <summary> Allocates global data </summary>
    /// <param name="name"> The name of the variable </param>
    /// <param name="type"> The type of the data </param>
    /// <param name="layout"> The layout of the data </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    Value GlobalAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags = AllocateFlags::None);

    /// <summary> Allocates global data </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="name"> The name of the variable </param>
    /// <param name="layout"> The layout of the data </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
    Value GlobalAllocate(std::string name, utilities::MemoryLayout layout, AllocateFlags flags = AllocateFlags::None)
    {
        return GlobalAllocate(name, GetValueType<T>(), layout, flags);
    }

    /// <summary> Allocates global data </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="name"> The name of the variable </param>
    /// <param name="data"> The data </param>
    /// <param name="layout"> The layout of the data </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
    Value GlobalAllocate(std::string name, const std::vector<T>& data, std::optional<utilities::MemoryLayout> layout = {}, AllocateFlags flags = AllocateFlags::None)
    {
        return GetContext().GlobalAllocate(name, data, layout, flags);
    }

    /// <summary> Allocates scalar global data </summary>
    /// <typeparam name="T"> The type of the data to allocate </param>
    /// <param name="name"> The name of the variable </param>
    /// <param name="flags"> Any additional flags. Not all contexts may support all flags. </summary>
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, void*> = nullptr>
    Scalar GlobalAllocate(std::string name, T t, AllocateFlags flags = AllocateFlags::None)
    {
        return GlobalAllocate(name,
                              std::vector<std::conditional_t<std::is_same_v<T, bool>, utilities::Boolean, T>>{ t },
                              utilities::ScalarLayout,
                              flags);
    }

    void DebugBreak();
    void DebugDump(FunctionDeclaration fn, std::string tag = "", std::ostream* stream = nullptr);
    void DebugDump(Value value, std::string tag = "", std::ostream* stream = nullptr);

    template <typename ViewType, std::enable_if_t<std::is_same_v<decltype(std::declval<ViewType>().GetValue()), Value>, void*> = nullptr>
    void DebugDump(ViewType value, std::string tag = "", std::ostream* stream = nullptr)
    {
        return DebugDump(value.GetValue(), tag, stream);
    }

    /// <summary> Emit a debug print message.  This assumes the application
    /// on the target platform implements a "void DebugPrint(char* message)" function.  This function will be
    /// defined for you when running in JIT or Compute mode.  </summary>
    void DebugPrint(std::string message);

    EmitterContext::IfContext If(Scalar test, std::function<void()> fn);

    void While(Scalar test, std::function<void()> fn);

    void ForRange(const std::string& name, Scalar end, std::function<void(Scalar)> fn);
    void ForRange(const std::string& name, Scalar start, Scalar end, std::function<void(Scalar)> fn);
    void ForRange(const std::string& name, Scalar start, Scalar end, Scalar step, std::function<void(Scalar)> fn);
    void ForRange(Scalar end, std::function<void(Scalar)> fn);
    void ForRange(Scalar start, Scalar end, std::function<void(Scalar)> fn);
    void ForRange(Scalar start, Scalar end, Scalar step, std::function<void(Scalar)> fn);

    void ForRanges(std::vector<Scalar> range_ends, std::function<void(std::vector<Scalar>)> fn);

    Matrix MFMALoad(Value source, const std::vector<int64_t> & shape, const std::string & operand);
    void MFMAStore(Matrix source, Value target);
    Matrix MFMACompute(Matrix A, Matrix B, Matrix C); 

    /// <summary> Runs the provided function, in parallel if possible </summary>
    /// <typeparam name="Tys..."> The types that represent the captured values. Must be `Value` or types that provide a member
    /// function called `GetValue()` which returns a `Value` instance. </typeparam>
    /// <param name="numTasks"> The number of tasks that should be created </param>
    /// <param name="captured"> A list of values to be used inside the function </param>
    /// <param name="fn"> The function that gets run for each task. The first parameter is the task number for that particular call.
    /// Subsequent parameters must match the typelist `Tys...` and will be filled in with the values provided within the `captured`
    /// parameter. In other words, the signature for the function should be `void fn(Scalar, Tys...)`. </param>
    template <typename Fn, typename... Tys>
    void Parallelize(int numTasks, std::tuple<Tys...> captured, Fn&& fn);

    /// <summary> Runs the provided function, in parallel if possible </summary>
    /// <param name="numTasks"> The number of tasks that should be created </param>
    /// <param name="captured"> A list of values to be used inside the function </param>
    /// <param name="fn"> The function that gets run for each task. The first parameter is the task number for that particular call. The second parameter
    /// will be filled in with the values provided within the `captured` parameter. </param>
    void Parallelize(int numTasks, std::vector<Value> captured, std::function<void(Scalar, std::vector<Value>)> fn);

    void MemCopy(ViewAdapter dest, ViewAdapter source, std::optional<Scalar> length = std::nullopt);
    void MemMove(ViewAdapter dest, ViewAdapter source, std::optional<Scalar> length = std::nullopt);
    void MemSet(ViewAdapter dest, Scalar data, std::optional<Scalar> length = std::nullopt);
    void MemZero(ViewAdapter dest, std::optional<Scalar> length = std::nullopt);

    /// <summary> Specifier determining if the fetch should be for a read or a write </summary>
    enum class PrefetchType
    {
        Read = 0,
        Write
    };

    /// <summary> Temporal locality specifier. Data with temporal locality, or persistence,
    /// is expected to be accessed multiple times and so should be left in a cache when it
    /// is prefetched so it will continue to be readily accessible. Accesses to data with
    /// no temporal locality are transient; the data is unlikely to be accessed multiple times
    /// and, if possible, should not be left in a cache where it would displace other data that
    /// might be needed soon. </summary>
    enum class PrefetchLocality
    {
        None = 0,
        Low,
        Moderate,
        Extreme
    };

    template <typename ViewType>
    void Prefetch(ViewType view, PrefetchType type = PrefetchType::Read, PrefetchLocality locality = PrefetchLocality::None);

    /// <summary> Returns a unique name based on the prefix provided </summary>
    /// <param name="prefix"> The prefix for the unique name desired </param>
    /// <returns> A unique name for the current EmitterContext instance </returns>
    std::string UniqueName(const std::string& prefix);

    /// <summary> Print a value </summary>
    /// <param name="value"> The value to print </param>
    /// <param name="toStderr"> Prints to stderr if true, otherwise prints to stdout </param>
    void Print(ViewAdapter value, bool toStderr = false);

    /// <summary> Print a string </summary>
    /// <param name="message"> The message to print </param>
    /// <param name="toStderr"> Prints to stderr if true, otherwise prints to stdout </param>
    void Print(const std::string& message, bool toStderr = false);

    /// <summary> Print a value's memory </summary>
    void PrintRawMemory(ViewAdapter value);

    /// <summary> Returns the passed in View type with the memory layout representative of the full view of the memory, i.e., no padding. </summary>
    template <typename ViewType>
    ViewType AsFullView(ViewType view);

    template <typename T, typename FnType = std::function<T(Scalar, T)>>
    T ReduceN(Scalar end, T init, FnType&& forFn)
    {
        auto rangeType = end.GetType();
        Scalar start = Cast(0, rangeType);
        Scalar step = Cast(1, rangeType);

        return GetContext()
            .ReduceN(
                start,
                end,
                step,
                std::vector<ViewAdapter>{ init },
                [&](Scalar index, std::vector<ViewAdapter> iterValues) {
                    return forFn(index, T{ iterValues[0].GetValue() });
                })
            .GetValue();
    }

    template <typename T, typename FnType = std::function<T(Scalar, T)>>
    T ReduceN(Scalar start, Scalar end, T init, FnType&& forFn)
    {
        auto rangeType = end.GetType();
        Scalar step = Cast(1, rangeType);

        return GetContext()
            .ReduceN(
                start,
                end,
                step,
                std::vector<ViewAdapter>{ init },
                [&](Scalar index, std::vector<ViewAdapter> iterValues) {
                    return forFn(index, T{ iterValues[0].GetValue() });
                })
            .GetValue();
    }

    template <typename T, typename FnType = std::function<T(Scalar, T)>>
    T ReduceN(Scalar start, Scalar end, Scalar step, T init, FnType&& forFn)
    {
        return GetContext()
            .ReduceN(
                start,
                end,
                step,
                std::vector<ViewAdapter>{ init },
                [&](Scalar index, std::vector<ViewAdapter> iterValues) {
                    return forFn(index, T{ iterValues[0].GetValue() });
                })
            .GetValue();
    }

    template <typename T,
              typename FnType = std::function<T(Scalar, T)>>
    T Reduce(const Array& a, T init, FnType&& reduceFn)
    {
        return GetContext()
            .Reduce(
                a,
                std::vector<ViewAdapter>{ init },
                [&](Scalar value, std::vector<ViewAdapter> iterValues) {
                    return reduceFn(value, T{ iterValues[0].GetValue() });
                })
            .GetValue();
    }

    template <typename T,
              typename MapFnType = std::function<T(Scalar)>,
              typename ReduceFnType = std::function<T(Scalar, T)>>
    T MapReduce(const Array& a, T init, MapFnType&& mapFn, ReduceFnType&& reduceFn)
    {
        return GetContext()
            .MapReduce(
                a,
                std::vector<ViewAdapter>{ init },
                [&](Scalar value) {
                    return mapFn(value);
                },
                [&](Scalar value, std::vector<ViewAdapter> iterValues) {
                    return reduceFn(value, T{ iterValues[0].GetValue() });
                })
            .GetValue();
    }

    inline void Return(ViewAdapter view = {}) { GetContext().ReturnValue(view); }

    inline Scalar GetTime() { return GetContext().GetTime(); }
} // namespace value
} // namespace accera

#pragma region implementation

namespace accera
{
namespace value
{

    template <typename ContextType, typename Fn, typename ReturnType>
    auto InvokeForContext(Fn&& fn) -> std::conditional_t<std::is_same_v<ReturnType, void>, void, std::optional<ReturnType>>
    {
        static_assert(std::is_base_of_v<EmitterContext, std::decay_t<ContextType>>,
                      "ContextType must be derived from EmitterContext");

        if (auto ptr = dynamic_cast<ContextType*>(&GetContext()); ptr != nullptr)
        {
            if constexpr (std::is_invocable_v<Fn, decltype(*ptr)>)
            {
                return fn(*ptr);
            }
            else
            {
                return fn();
            }
        }

        if constexpr (!std::is_same_v<ReturnType, void>)
        {
            return std::nullopt;
        }
    }

    template <typename Fn, typename... Tys>
    void Parallelize(int numTasks, std::tuple<Tys...> captured, Fn&& fn)
    {
        auto capturedValues = utilities::TupleToVector<Value>([](auto view) { return detail::GetValue(view); }, captured);

        Parallelize(
            numTasks,
            capturedValues,
            [fn = std::move(fn)](Scalar threadIndex, std::vector<Value> captures) {
                auto fnArgsTuple = [&] {
                    std::tuple scalarTuple{ threadIndex };
                    if constexpr (sizeof...(Tys) > 0)
                    {
                        auto capturesTuple = utilities::VectorToTuple<Tys...>(captures);
                        return std::tuple_cat(scalarTuple, capturesTuple);
                    }
                    else
                    {
                        return scalarTuple;
                    }
                }();
                std::apply(fn, fnArgsTuple);
            });
    }

    template <typename ViewType>
    void Prefetch(ViewType view, PrefetchType type, PrefetchLocality locality)
    {
        GetContext().Prefetch(detail::GetValue(view), type, locality);
    }

    template <typename ViewType>
    ViewType AsFullView(ViewType view)
    {
        auto value = detail::GetValue(view);
        value.SetLayout(utilities::MemoryLayout{ value.GetLayout().GetExtent(), value.GetLayout().GetDimensionOrder() });
        return value;
    }

} // namespace value
} // namespace accera

#pragma endregion implementation
