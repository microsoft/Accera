////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "InPlaceUnrollInfo.h"
#include "ParallelizationInfo.h"
#include "TensorizationInfo.h"
#include "VectorizationInfo.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/Hashing.h>

using mlir::IntegerAttr;

namespace accera::ir
{

namespace executionPlan
{
    // Make storage types hashable
    llvm::hash_code hash_value(const VectorizationInfo& index);
    llvm::hash_code hash_value(const ParallelizationInfo& index);
    llvm::hash_code hash_value(const InPlaceUnrollInfo& index);

    namespace detail
    {
        //
        // TrivialStorage is a mixin class to relieve some of the burden of defining AttributeStorage subclasses for simple, statically-allocated types
        //
        // `ConcreteT` is a CRTP parameter: the AttributeStorage class being defined
        // `Storage` is the type of the underlying storage object -- this object must implement op==, and must be hashable via a `llvm::hash_code hash_value(const Storage&)` function
        template <typename ConcreteT, typename Storage>
        struct TrivialStorage
        {
            using KeyTy = Storage;

            TrivialStorage(const Storage& obj) :
                _obj(obj)
            {
            }

            // Key equality and hash functions.
            bool operator==(const KeyTy& key) const
            {
                return key == getValue();
            }

            static unsigned hashKey(const KeyTy& key)
            {
                return hash_value(key);
            }

            static ConcreteT*
            construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
            {
                return new (allocator.allocate<ConcreteT>())
                    ConcreteT(key);
            }

            // Returns the stored value.
            Storage getValue() const
            {
                return _obj;
            }

            Storage _obj;
        };

        struct VectorizationInfoAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<VectorizationInfoAttributeStorage, VectorizationInfo>
        {
            using KeyTy = VectorizationInfo;

            VectorizationInfoAttributeStorage(const VectorizationInfo& vectorizationInfo) :
                TrivialStorage<VectorizationInfoAttributeStorage, VectorizationInfo>(vectorizationInfo)
            {
            }
        };

        struct ParallelizationInfoAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<ParallelizationInfoAttributeStorage, ParallelizationInfo>
        {
            using KeyTy = ParallelizationInfo;

            ParallelizationInfoAttributeStorage(const ParallelizationInfo& parallelizationInfo) :
                TrivialStorage<ParallelizationInfoAttributeStorage, ParallelizationInfo>(parallelizationInfo)
            {
            }
        };

        struct TensorizeInfoAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<TensorizeInfoAttributeStorage, TensorizationInfo>
        {
            using KeyTy = TensorizationInfo;

            TensorizeInfoAttributeStorage(const TensorizationInfo& tensorizeInfo) :
                TrivialStorage<TensorizeInfoAttributeStorage, TensorizationInfo>(tensorizeInfo)
            {
            }
        };

        struct InPlaceUnrollInfoAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<InPlaceUnrollInfoAttributeStorage, InPlaceUnrollInfo>
        {
            using KeyTy = InPlaceUnrollInfo;

            InPlaceUnrollInfoAttributeStorage(const InPlaceUnrollInfo& inPlaceUnrollInfo) :
                TrivialStorage<InPlaceUnrollInfoAttributeStorage, InPlaceUnrollInfo>(inPlaceUnrollInfo)
            {
            }
        };
    } // namespace detail

    class VectorizationInfoAttr
        : public mlir::Attribute::AttrBase<VectorizationInfoAttr, mlir::Attribute, detail::VectorizationInfoAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = VectorizationInfo;

        static VectorizationInfoAttr get(const ValueType& vectorizationInfo, mlir::MLIRContext* context);

        ValueType getValue() const;

        static llvm::StringRef getKeyName() { return "accxp_vectorizationInfo"; }
    };

    class ParallelizationInfoAttr
        : public mlir::Attribute::AttrBase<ParallelizationInfoAttr, mlir::Attribute, detail::ParallelizationInfoAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = ParallelizationInfo;

        static ParallelizationInfoAttr get(const ValueType& parallelizationInfo, mlir::MLIRContext* context);

        ValueType getValue() const;

        static llvm::StringRef getKeyName() { return "accxp_parallelizationInfo"; }
    };

    class TensorizationInfoAttr
        : public mlir::Attribute::AttrBase<TensorizationInfoAttr, mlir::Attribute, detail::TensorizeInfoAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = TensorizationInfo;

        static TensorizationInfoAttr get(const ValueType& tensorizationInfo, mlir::MLIRContext* context);

        ValueType getValue() const;

        static llvm::StringRef getKeyName() { return "accxp_tensorizationInfo"; }
    };

    class InPlaceUnrollInfoAttr
        : public mlir::Attribute::AttrBase<InPlaceUnrollInfoAttr, mlir::Attribute, detail::InPlaceUnrollInfoAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = InPlaceUnrollInfo;

        static InPlaceUnrollInfoAttr get(const ValueType& InPlaceUnrollInfo, mlir::MLIRContext* context);

        ValueType getValue() const;

        static llvm::StringRef getKeyName() { return "accxp_inPlaceUnrollInfo"; }
    };

    //
    // Parse and print functions
    //
    VectorizationInfoAttr parseVectorizationInfo(mlir::DialectAsmParser& os);
    ParallelizationInfoAttr parseParallelizationInfo(mlir::DialectAsmParser& os);
    TensorizationInfoAttr parseTensorizationInfo(mlir::DialectAsmParser& os);
    InPlaceUnrollInfoAttr parseInPlaceUnrollInfo(mlir::DialectAsmParser& os);

    void print(VectorizationInfoAttr attr, mlir::DialectAsmPrinter& os);
    void print(ParallelizationInfoAttr attr, mlir::DialectAsmPrinter& os);
    void print(TensorizationInfoAttr attr, mlir::DialectAsmPrinter& os);
    void print(InPlaceUnrollInfoAttr attr, mlir::DialectAsmPrinter& os);

} // namespace executionPlan
} // namespace accera::ir
