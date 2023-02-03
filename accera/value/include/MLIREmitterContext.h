////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <forward_list>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>

#include "EmitterContext.h"
#include "FunctionDeclaration.h"
#include "Scalar.h"
#include "Tensor.h"
#include "Vector.h"

#include <ir/include/Metadata.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>

#include <ir/include/exec/ExecutionPlanEnums.h>

namespace mlir
{
class ModuleOp;
class Value;
} // namespace mlir

namespace accera
{
namespace value
{

    enum class GPUIndexType
    {
        BlockDim,
        BlockId,
        GridDim,
        ThreadId,
    };

    struct GPUIndex
    {
        GPUIndex(std::function<Scalar(const GPUIndexDimension)>);

        Scalar X();
        Scalar Y();
        Scalar Z();

    private:
        std::function<Scalar(const GPUIndexDimension)> _fn;
    };

    class MLIRContextBase
    {
    protected:
        MLIRContextBase(const std::string& moduleName);
        MLIRContextBase(mlir::ModuleOp& existingModule);

        struct Impl;

        std::unique_ptr<Impl> _impl;
    };

    /// <summary> A specialization of EmitterContext that emits MLIR IR </summary>
    class MLIRContext : private MLIRContextBase
        , public EmitterContext
    {
    public:
        /// <summary> Constructor </summary>
        ///
        /// <param name="moduleName"> Name of the module. </param>
        /// <param name="parameters"> Options for the compiler </param>
        MLIRContext(const std::string& moduleName, const CompilerOptions& options = {});
        MLIRContext(mlir::ModuleOp& existingModule, const CompilerOptions& options = {});

        ~MLIRContext();

        void print() const;
        void verify() const;
        void save(std::string filename) const;

        mlir::OwningOpRef<mlir::ModuleOp> cloneModule() const;

        void writeHeader(std::optional<std::string> filename = std::nullopt) const;

        void setMetadata(const std::string& key, const accera::ir::MetadataValueType& value);
        accera::ir::Metadata getFullMetadata();

        void setDataLayout(const CompilerOptions& options);

        void setTargetFeatures(const TargetDevice& targetDevice);

        void setDebugMode(bool enable);

        struct EmittableInfo
        {
            void* data;
            detail::ValueTypeDescription desc;
            bool isGlobal = false;
        };

        template<GPUIndexType>
        GPUIndex GetGPUIndex();

        friend mlir::Value Unwrap(accera::value::Value&);
        friend mlir::Value UnwrapScalar(Scalar&);
        friend ViewAdapter Wrap(mlir::Value, std::optional<accera::utilities::MemoryLayout> layout);

        mlir::OpBuilder& GetOpBuilder();

    private:
        void SetLayoutImpl(Value&, const MemoryLayout&) override;

        Value AllocateImpl(ValueType value, MemoryLayout layout, size_t alignment, AllocateFlags flags = AllocateFlags::None, const std::vector<ScalarDimension>& runtimeSizes = {}) override;

        std::optional<Value> GetGlobalValue(GlobalAllocationScope scope, std::string name) override;

        Value GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ConstantData data, MemoryLayout layout, AllocateFlags flags = AllocateFlags::None) override;
        Value GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ValueType type, MemoryLayout layout, AllocateFlags flags = AllocateFlags::None) override;

        detail::ValueTypeDescription GetTypeImpl(Emittable emittable) override;

        DefinedFunction CreateFunctionImpl(FunctionDeclaration decl, DefinedFunction fn) override;
        DefinedFunction DeclareExternalFunctionImpl(FunctionDeclaration decl) override;
        bool IsFunctionDefinedImpl(FunctionDeclaration decl) const override;

        Value StoreConstantDataImpl(ConstantData data, MemoryLayout layout, const std::string& name) override;
        bool IsConstantDataImpl(Value) const override;
        Value ResolveConstantDataReferenceImpl(Value constantDataSource) override;

        void ForImpl(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name) override;
        void ForImpl(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name) override;

        ViewAdapter ReduceN(Scalar begin, Scalar end, Scalar increment, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(Scalar, std::vector<ViewAdapter>)>) override;
        ViewAdapter Reduce(Array a, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(ViewAdapter, std::vector<ViewAdapter>)>) override;

        ViewAdapter MapReduce(Array a, std::vector<ViewAdapter> initArgs, std::function<ViewAdapter(ViewAdapter)> mapFn, std::function<ViewAdapter(ViewAdapter, std::vector<ViewAdapter>)> reduceFn) override;

        void MoveDataImpl(Value& source, Value& destination) override;

        void CopyDataImpl(const Value& source, Value& destination) override;

        void StoreImpl(const Value& source, Value& destination, const std::vector<int64_t>& indices) override;

        Value ViewImpl(Value source, const std::vector<Scalar>& offsets, const utilities::MemoryShape& shape, const std::vector<int64_t>& strides) override;

        Value SliceImpl(Value source, std::vector<int64_t> slicedDimensions, std::vector<Scalar> sliceOffsets) override;

        Value MergeDimensionsImpl(Value source, int64_t dim1, int64_t dim2) override;

        Value SplitDimensionImpl(Value source, int64_t dim, int64_t size) override;

        Value ReshapeImpl(Value source, const MemoryLayout& layout) override;

        Value ReorderImpl(Value source, const utilities::DimensionOrder& order) override;

        Value UnaryOperationImpl(ValueUnaryOperation op, Value source) override;

        Value BinaryOperationImpl(ValueBinaryOperation op, Value source1, Value source2) override;

        Value LogicalOperationImpl(ValueLogicalOperation op, Value source1, Value source2) override;

        Value MMALoadSyncImpl(const Matrix& source, const int64_t rowOffset, const int64_t colOffset, const MatrixFragment& target) override;
        void MMAStoreSyncImpl(const MatrixFragment& source, Matrix& target, const int64_t rowOffset, const int64_t colOffset) override;
        Value MMAComputeSyncImpl(const MatrixFragment& A, const MatrixFragment& B, const MatrixFragment& C, uint32_t cbsz, uint32_t abid, uint32_t blgp) override;

        Scalar CastImpl(Scalar value, ValueType type) override;

        Scalar RoundImpl(Scalar value) override;

        bool IsImplicitlyCastableImpl(ValueType source, ValueType target) const override;

        Scalar BitcastImpl(Scalar value, ValueType type) override;

        IfContext IfImpl(Scalar test, std::function<void()> fn) override;

        void WhileImpl(Scalar test, std::function<void()> fn) override;

        std::optional<Value> CallImpl(FunctionDeclaration func, std::vector<Value> args) override;

        void ReturnValue(ViewAdapter view) override;

        Scalar GetTime() override;

        void EnterProfileRegion(const std::string& regionName) override;
        void ExitProfileRegion(const std::string& regionName) override;
        void PrintProfileResults() override;

        void PrefetchImpl(Value data, PrefetchType type, PrefetchLocality locality) override;

        void PrintImpl(ViewAdapter value, bool toStderr) override;
        void PrintImpl(const std::string& value, bool toStderr) override;
        void PrintRawMemoryImpl(ViewAdapter value) override;

        void DebugBreakImpl() override;
        void DebugDumpImpl(Value value, std::string tag, std::ostream& stream) const override;
        void DebugDumpImpl(FunctionDeclaration fn, std::string tag, std::ostream& stream) const override;
        void DebugPrintImpl(std::string message) override;

        void SetNameImpl(const Value& value, const std::string& name) override;
        std::string GetNameImpl(const Value& value) const override;

        void ImportCodeFileImpl(std::string) override;

        Scalar MaxImpl(Vector input) override;

        Scalar SumImpl(Vector input) override;

        Value IntrinsicCall(FunctionDeclaration intrinsic, std::vector<Value> args);

        std::optional<Value> EmitExternalCall(FunctionDeclaration func, std::vector<Value> args);

        bool TypeCompatible(Value value1, Value value2);

        std::string GetScopeAdjustedName(GlobalAllocationScope scope, std::string name) const;
        std::string GetGlobalScopedName(std::string name) const;
        std::string GetCurrentFunctionScopedName(std::string name) const;

        Value Realize(Value value) const;
        Value EnsureEmittable(Value value);
        std::vector<Value> EnsureEmittable(std::vector<Value> values);

        EmittableInfo& StoreGlobalEmittable(EmittableInfo);
        EmittableInfo& StoreLocalEmittable(EmittableInfo);

        class IfContextImpl;
        struct FunctionScope;

        mutable std::mutex _mutex;

        std::forward_list<EmittableInfo> _globalEmittables;
        std::stack<std::forward_list<EmittableInfo>> _localEmittables;
        std::map<std::string, std::pair<Emittable, MemoryLayout>> _globals;
        std::unordered_map<FunctionDeclaration, DefinedFunction> _definedFunctions;
    };

    bool HasMLIRTypeConversion(ValueType valueType);
    mlir::Type ValueTypeToMLIRType(mlir::OpBuilder& builder, ValueType valueType);
    ValueType MLIRTypeToValueType(mlir::Type type);
    mlir::Value Unwrap(ViewAdapter);
    mlir::Value UnwrapScalar(Scalar);
    ViewAdapter Wrap(mlir::Value, std::optional<utilities::MemoryLayout> layout = std::nullopt);
    std::vector<ViewAdapter> Wrap(std::vector<mlir::Value>, std::function<utilities::MemoryLayout(mlir::Value)> layoutFn = nullptr);
    Value ResolveConstantDataReference(Value constantDataSource);

    struct GPU
    {
        enum class BarrierScope
        {
            Block = 0,
            Warp = 1,
            Threadfence = 2,
        };

        static GPUIndex BlockDim();
        static GPUIndex BlockId();
        static GPUIndex GridDim();
        static GPUIndex ThreadId();
        static void Barrier(BarrierScope scope = BarrierScope::Block);
    };

    void FillResource(ViewAdapter, Scalar);
    void PrintMemref(ViewAdapter);

    mlir::OwningOpRef<mlir::ModuleOp> GatherModules(const std::string& name, const std::vector<MLIRContext*>& contexts, mlir::MLIRContext* context);
    void SaveModule(const std::string& filename, mlir::ModuleOp moduleOp);

    void WriteHeaderForModule(const std::string& filename, mlir::ModuleOp moduleOp);
    void WriteHeaderForModules(const std::string& filename, const std::string& libraryName, const std::vector<value::MLIRContext*>& contexts);

    inline ::accera::value::MLIRContext& GetMLIRContext()
    {
        return dynamic_cast<::accera::value::MLIRContext&>(::accera::value::GetContext());
    }
} // namespace value
} // namespace accera
