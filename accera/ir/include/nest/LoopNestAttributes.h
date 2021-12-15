////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/Hashing.h>

#include "nest/LoopNestEnums.h.inc"
#include "Index.h"
#include "IndexRange.h"
#include "IterationDomain.h"
#include "OperandIndex.h"
#include "Range.h"
#include "TransformedDomain.h"


using mlir::IntegerAttr;
using mlir::StringAttr;


namespace accera::ir
{

namespace loopnest
{
    // Make storage types hashable
    llvm::hash_code hash_value(const Index& index);
    llvm::hash_code hash_value(const IndexRange& indexRange);
    llvm::hash_code hash_value(const IterationDomain& domain);
    llvm::hash_code hash_value(const Range& r);
    llvm::hash_code hash_value(const SplitIndex& splitIndex);
    llvm::hash_code hash_value(const TransformedDomain& domain);
    llvm::hash_code hash_value(const OperandIndex& index);

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

        struct IndexAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<IndexAttributeStorage, Index>
        {
            using KeyTy = Index;

            IndexAttributeStorage(const Index& index) :
                TrivialStorage<IndexAttributeStorage, Index>(index)
            {
            }
        };

        struct IndexRangeAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<IndexRangeAttributeStorage, IndexRange>
        {
            using KeyTy = IndexRange;

            IndexRangeAttributeStorage(const IndexRange& indexRange) :
                TrivialStorage<IndexRangeAttributeStorage, IndexRange>(indexRange)
            {
            }
        };

        struct IterationDomainAttributeStorage : public mlir::AttributeStorage
        {
            using KeyTy = mlir::ArrayRef<IndexRange>;

            IterationDomainAttributeStorage(const mlir::ArrayRef<IndexRange>& value);
            IterationDomainAttributeStorage(const IterationDomain& value);

            // Key equality function.
            bool operator==(const KeyTy& key) const;

            IterationDomain getValue() const { return value; }

            static KeyTy getKey(const IterationDomain& domain);

            // Construct a new storage instance.
            static IterationDomainAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
            {
                return new (allocator.allocate<IterationDomainAttributeStorage>())
                    IterationDomainAttributeStorage(allocator.copyInto(key));
            }

            IterationDomain value;
        };

        struct RangeAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<RangeAttributeStorage, Range>
        {
            using KeyTy = Range;

            RangeAttributeStorage(const Range& range) :
                TrivialStorage<RangeAttributeStorage, Range>(range)
            {
            }
        };

        struct SplitIndexAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<SplitIndexAttributeStorage, SplitIndex>
        {
            using KeyTy = SplitIndex;

            SplitIndexAttributeStorage(const SplitIndex& index) :
                TrivialStorage<SplitIndexAttributeStorage, SplitIndex>(index)
            {
            }
        };

        struct TransformedDomainAttributeStorage : public mlir::AttributeStorage
        {
            using KeyTy = TransformedDomain::AttributeKey;

            TransformedDomainAttributeStorage(const TransformedDomain& value);
            TransformedDomainAttributeStorage(const KeyTy& value);

            // Key equality function.
            bool operator==(const KeyTy& key) const;

            TransformedDomain getValue() const { return value; }

            static KeyTy getKey(const TransformedDomain& domain);

            static unsigned hashKey(const KeyTy& key)
            {
                return hash_value(key);
            }

            // Construct a new storage instance.
            static TransformedDomainAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key);

            TransformedDomain value;
        };

        struct OperandIndexAttributeStorage final
            : public mlir::AttributeStorage
            , public TrivialStorage<OperandIndexAttributeStorage, OperandIndex>
        {
            using KeyTy = OperandIndex;

            OperandIndexAttributeStorage(const OperandIndex& operandIndex) :
                TrivialStorage<OperandIndexAttributeStorage, OperandIndex>(operandIndex)
            {
            }
        };

    } // namespace detail

    // TODO: make another mixin class to take care of trivial get()?
    class IndexAttr
        : public mlir::Attribute::AttrBase<IndexAttr, mlir::Attribute, detail::IndexAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = Index;

        static IndexAttr get(const ValueType& index, mlir::MLIRContext* context);

        ValueType getValue() const;
    };

    class IndexRangeAttr
        : public mlir::Attribute::AttrBase<IndexRangeAttr, mlir::Attribute, detail::IndexRangeAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = IndexRange;

        static IndexRangeAttr get(const ValueType& value, mlir::MLIRContext* context);

        ValueType getValue() const;
    };

    class IterationDomainAttr
        : public mlir::Attribute::AttrBase<IterationDomainAttr, mlir::Attribute, detail::IterationDomainAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = IterationDomain;

        static IterationDomainAttr get(const ValueType& value, mlir::MLIRContext* context);

        ValueType getValue() const;
    };

    class RangeAttr
        : public mlir::Attribute::AttrBase<RangeAttr, mlir::Attribute, detail::RangeAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = Range;

        static RangeAttr get(const ValueType& value, mlir::MLIRContext* context);
        static RangeAttr get(int begin, int end, int increment, mlir::MLIRContext* context);
        static RangeAttr get(int begin, OperandIndex end, int increment, mlir::MLIRContext* context);
        static RangeAttr get(int begin, mlir::Value end, int increment, mlir::MLIRContext* context);

        ValueType getValue() const;

        // static mlir::LogicalResult verifyConstructionInvariants(mlir::Optional<mlir::Location> loc,
        //                                                         mlir::MLIRContext* ctx,
        //                                                         const ValueType& range);

        // static mlir::LogicalResult verifyConstructionInvariants(mlir::Optional<mlir::Location> loc,
        //                                                         mlir::MLIRContext* ctx,
        //                                                         int begin,
        //                                                         int end,
        //                                                         int increment);
    };

    // TODO: make this subclass from TypedArrayAttrBase (?)
    class SplitIndexAttr
        : public mlir::Attribute::AttrBase<SplitIndexAttr, mlir::Attribute, detail::SplitIndexAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = SplitIndex;

        static SplitIndexAttr get(const ValueType& value, mlir::MLIRContext* context);

        ValueType getValue() const;
    };

    class TransformedDomainAttr
        : public mlir::Attribute::AttrBase<TransformedDomainAttr, mlir::Attribute, detail::TransformedDomainAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = TransformedDomain;

        static TransformedDomainAttr get(const ValueType& value, mlir::MLIRContext* context);

        ValueType getValue() const;
    };

    class FragmentPredicateAttr : public mlir::DictionaryAttr
    {
    public:
        using DictionaryAttr::DictionaryAttr;
        static FragmentPredicateAttr get(
            IndexAttr index,
            IntegerAttr fragmentType,
            mlir::MLIRContext* context);

        IndexAttr index() const;
        IntegerAttr fragmentType() const;
    };

    class OperandIndexAttr
        : public mlir::Attribute::AttrBase<OperandIndexAttr, mlir::Attribute, detail::OperandIndexAttributeStorage>
    {
    public:
        using Base::Base;
        using ValueType = OperandIndex;

        static OperandIndexAttr get(const ValueType& index, mlir::MLIRContext* context);

        ValueType getValue() const;
    };

    //
    // Parse and print functions
    //
    IndexAttr parseIndex(mlir::DialectAsmParser& os);
    IndexRangeAttr parseIndexRange(mlir::DialectAsmParser& os);
    IterationDomainAttr parseIterationDomain(mlir::DialectAsmParser& os);
    RangeAttr parseRange(mlir::DialectAsmParser& os);
    SplitIndexAttr parseSplitIndex(mlir::DialectAsmParser& os);
    TransformedDomainAttr parseTransformedDomain(mlir::DialectAsmParser& os);
    OperandIndexAttr parseOperandIndex(mlir::DialectAsmParser& os);

    void print(IndexAttr attr, mlir::DialectAsmPrinter& os);
    void print(IndexRangeAttr attr, mlir::DialectAsmPrinter& os);
    void print(IterationDomainAttr attr, mlir::DialectAsmPrinter& os);
    void print(RangeAttr attr, mlir::DialectAsmPrinter& os);
    void print(SplitIndexAttr attr, mlir::DialectAsmPrinter& os);
    void print(TransformedDomainAttr attr, mlir::DialectAsmPrinter& os);
    void print(OperandIndexAttr attr, mlir::DialectAsmPrinter& os);

} // namespace loopnest
} // namespace accera::ir
