////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestAttributes.h"

#include <llvm/Support/ErrorHandling.h>
#include <utilities/include/Exception.h>
#include <utilities/include/TypeTraits.h>

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>

#include <iostream>
#include <stdexcept>
#include <variant>

#include "nest/LoopNestAttrs.cpp.inc"
#include "nest/LoopNestEnums.cpp.inc"

namespace accera::ir
{
namespace loopnest
{
    //
    // DSL type printers
    //
    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, Index index)
    {
        printer << "{" << index.GetName() << "," << index.GetId() << '}';
        return printer;
    }
    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, OperandIndex opIndex)
    {
        printer << "{op_idx:" << opIndex.GetIndex() << '}';
        return printer;
    }

    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, Range range)
    {
        printer << '{';
        printer << range.Begin();
        printer << ':';
        if (range.HasConstantEnd())
        {
            printer << range.End();
        }
        else if (range.HasIndexEnd())
        {
            printer << range.EndIndex();
        }
        else if (range.HasVariableEnd())
        {
            auto arg = range.VariableEnd().dyn_cast<mlir::BlockArgument>();
            printer << arg;
        }
        else
        {
            printer << range.EndOperandIndex();
        }
        printer << ':';
        printer << range.Increment();
        printer << '}';
        return printer;
    }

    mlir::DialectAsmPrinter& operator<<(mlir::DialectAsmPrinter& printer, IndexRange indexRange)
    {
        printer << indexRange.GetIndex() << '='
                << indexRange.GetRange();
        // printer << "{"
        //         << indexRange.GetIndex() << ','
        //         << indexRange.GetRange() << '}';
        return printer;
    }

    //
    // OperandIndexAttr
    //
    OperandIndexAttr OperandIndexAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    OperandIndex OperandIndexAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    OperandIndexAttr parseOperandIndex(mlir::DialectAsmParser& parser)
    {
        // Parse an OperandIndex attribute in the following form:
        //   operand-index-attr ::= `{` `op_idx` :` idx `}`

        if (failed(parser.parseLBrace()))
            return {};

        llvm::StringRef opIdxStr;
        if (failed(parser.parseKeyword(&opIdxStr)))
            return {};

        int idx;
        if (failed(parser.parseInteger(idx)))
            return {};

        if (failed(parser.parseRBrace()))
            return {};

        return OperandIndexAttr::get(OperandIndex{ idx }, parser.getBuilder().getContext());
    }

    void print(OperandIndexAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "operand_index";
        auto index = attr.cast<OperandIndexAttr>().getValue();
        printer << index;
    }

    //
    // IndexAttr
    //
    IndexAttr IndexAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    Index IndexAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    IndexAttr parseIndex(mlir::DialectAsmParser& parser)
    {
        // Parse an Index attribute in the following form:
        //   index-attr ::= `{` name `,` id `}`

        if (failed(parser.parseLBrace()))
            return {};

        llvm::StringRef name;
        if (failed(parser.parseKeyword(&name)))
            return {};

        if (failed(parser.parseComma()))
            return {};

        int id;
        if (failed(parser.parseInteger(id)))
            return {};

        if (failed(parser.parseRBrace()))
            return {};

        return IndexAttr::get(Index{ std::string(name), id }, parser.getBuilder().getContext());
    }

    void print(IndexAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "index";
        auto index = attr.cast<IndexAttr>().getValue();
        printer << index;
    }

    std::variant<OperandIndexAttr, IndexAttr> parseIndexOrOperandIndex(mlir::DialectAsmParser& parser)
    {
        // Parse either an OperandIndex attribute or an Index attribute
        //   operand-index-attr ::= `{` `op_idx` `,` id `}`
        //   index-attr ::= `{` name `,` id `}`

        if (failed(parser.parseLBrace()))
            return {};

        llvm::StringRef name;
        if (failed(parser.parseKeyword(&name)))
            return {};

        if (failed(parser.parseComma()))
            return {};

        int id;
        if (failed(parser.parseInteger(id)))
            return {};

        if (failed(parser.parseRBrace()))
            return {};

        if (name == "op_idx")
        {
            return OperandIndexAttr::get(OperandIndex{ id }, parser.getBuilder().getContext());
        }
        else
        {
            return IndexAttr::get(Index{ std::string(name), id }, parser.getBuilder().getContext());
        }
    }

    //
    // IndexRangeAttr
    //
    IndexRangeAttr IndexRangeAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    IndexRange IndexRangeAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    // {i_0,0}={0:8:1}
    IndexRangeAttr parseIndexRange(mlir::DialectAsmParser& parser)
    {
        IndexAttr indexAttr = parseIndex(parser);
        if (!indexAttr)
        {
            return {};
        }

        if (failed(parser.parseEqual()))
            return {};

        RangeAttr rangeAttr = parseRange(parser);
        if (!rangeAttr)
            return {};

        return IndexRangeAttr::get(IndexRange{ indexAttr.getValue(), rangeAttr.getValue() }, parser.getBuilder().getContext());
    }

    // {i_0,0}={0:8:1}
    void print(IndexRangeAttr attr, mlir::DialectAsmPrinter& printer)
    {
        printer << "indexrange";
        auto indexRange = attr.getValue();
        printer << indexRange;
    }

    //
    // IterationDomainAttr
    //
    namespace detail
    {
        IterationDomainAttributeStorage::IterationDomainAttributeStorage(const IterationDomain& value_) :
            value(value_)
        {
        }

        IterationDomainAttributeStorage::IterationDomainAttributeStorage(const mlir::ArrayRef<IndexRange>& value_) :
            value(std::vector<IndexRange>(std::begin(value_), std::end(value_)))
        {
        }

        IterationDomainAttributeStorage::KeyTy IterationDomainAttributeStorage::getKey(const IterationDomain& domain)
        {
            return domain.GetRanges();
        }

        bool IterationDomainAttributeStorage::operator==(const KeyTy& key) const
        {
            auto ranges = getValue().GetRanges();
            if (key.size() != ranges.size())
            {
                return false;
            }
            auto pairs = llvm::zip(ranges, key);
            return std::all_of(std::begin(pairs), std::end(pairs), [](auto x) { return std::get<0>(x) == std::get<1>(x); });
        }

    } // namespace detail

    IterationDomainAttr IterationDomainAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    IterationDomain IterationDomainAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    // { {i_0,0}={0:8:1}, {i_1,1}={0:10:2} }
    IterationDomainAttr parseIterationDomain(mlir::DialectAsmParser& parser)
    {
        if (failed(parser.parseLBrace()))
            return {};

        if (succeeded(parser.parseOptionalRBrace())) // nothing inside
            return IterationDomainAttr::get(IterationDomain{}, parser.getBuilder().getContext());

        std::vector<IndexRange> ranges;
        do
        {
            IndexRangeAttr range = parseIndexRange(parser);
            if (!range)
            {
                return {};
            }
            ranges.push_back(range.getValue());
        } while (succeeded(parser.parseOptionalComma()));

        if (failed(parser.parseRBrace()))
            return {};

        return IterationDomainAttr::get(IterationDomain(ranges), parser.getBuilder().getContext());
    }

    // { {i_0,0}={0:8:1}, {i_1,1}={0:10:2} }
    void print(IterationDomainAttr attr, mlir::DialectAsmPrinter& printer)
    {
        auto domain = attr.cast<IterationDomainAttr>().getValue();
        printer << "idomain{";
        llvm::interleaveComma(domain.GetRanges(), printer); // interleaveComma relies on op<<
        printer << '}';
    }

    //
    // RangeAttr
    //
    RangeAttr RangeAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    RangeAttr RangeAttr::get(int begin, int end, int increment, mlir::MLIRContext* context)
    {
        return Base::get(context, begin, end, increment);
    }

    RangeAttr RangeAttr::get(int begin, OperandIndex endOpIndex, int increment, mlir::MLIRContext* context)
    {
        return Base::get(context, begin, endOpIndex, increment);
    }

    RangeAttr RangeAttr::get(int begin, mlir::Value end, int increment, mlir::MLIRContext* context)
    {
        return Base::get(context, begin, end, increment);
    }

    Range RangeAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    // mlir::LogicalResult RangeAttr::verifyConstructionInvariants(mlir::Optional<mlir::Location> loc,
    //                                                             mlir::MLIRContext* context,
    //                                                             const ValueType& range)
    // {
    //     auto thisLoc = loc.getValueOr(mlir::UnknownLoc::get(context));
    //     if (range.End() < range.Begin())
    //     {
    //         return mlir::emitError(thisLoc) << "Range with negative extent";
    //     }

    //     if (range.Increment() == 0)
    //     {
    //         return mlir::emitError(thisLoc) << "Range with increment of 0";
    //     }

    //     if (range.Increment() < 0)
    //     {
    //         return mlir::emitError(thisLoc) << "Range with negative increment";
    //     }

    //     return mlir::success();
    // }

    // mlir::LogicalResult RangeAttr::verifyConstructionInvariants(mlir::Optional<mlir::Location> loc,
    //                                                             mlir::MLIRContext* context,
    //                                                             int begin,
    //                                                             int end,
    //                                                             int increment)
    // {
    //     return verifyConstructionInvariants(loc, context, Range{ begin, end, increment });
    // }

    RangeAttr parseRange(mlir::DialectAsmParser& parser)
    {
        // Parse a range attribute in the following form:
        //   range-attr ::= `{` begin `:` end `:` step `}`

        if (failed(parser.parseLBrace()))
            return {};

        int begin;
        if (failed(parser.parseInteger(begin)))
            return {};

        if (failed(parser.parseColon()))
            return {};

        // The end value is an attribute containing either be an integer constant or an index
        Index endIndex;
        OperandIndex endOperandIndex;
        int endInt;
        bool isEndInt = false;
        bool isEndIndex = false;
        bool isEndOperandIndex = false;
        if (parser.parseOptionalInteger(endInt).hasValue())
        {
            isEndInt = true;
        }
        else
        {
            auto endIndexOrOperandIndex = parseIndexOrOperandIndex(parser);
            if (std::holds_alternative<IndexAttr>(endIndexOrOperandIndex))
            {
                isEndIndex = true;
                endIndex = std::visit(
                    utilities::VariantVisitor{
                        [](IndexAttr endIdx) -> Index {
                            return endIdx.getValue();
                        },
                        [](OperandIndexAttr endIdx) -> Index {
                            assert(false && "Unsupported end index type");
                            return {};
                        },
                        [](auto&& endIdx) -> Index {
                            assert(false && "Unsupported end index type");
                            return {};
                        } },
                    endIndexOrOperandIndex);
            }
            else if (std::holds_alternative<OperandIndexAttr>(endIndexOrOperandIndex))
            {
                isEndOperandIndex = true;
                endOperandIndex = std::visit(
                    utilities::VariantVisitor{
                        [](IndexAttr endOpIdx) -> OperandIndex {
                            assert(false && "Unsupported end index type");
                            return {};
                        },
                        [](OperandIndexAttr endOpIdx) -> OperandIndex {
                            return endOpIdx.getValue();
                        },
                        [](auto&& endOpIdx) -> OperandIndex {
                            assert(false && "Unsupported end index type");
                            return {};
                        } },
                    endIndexOrOperandIndex);
            }
        }

        if (failed(parser.parseColon()))
            return {};

        int increment;
        if (failed(parser.parseInteger(increment)))
            return {};

        if (failed(parser.parseRBrace()))
            return {};

        if (isEndIndex)
        {
            return RangeAttr::get(Range{ begin, endIndex, increment }, parser.getBuilder().getContext());
        }
        else if (isEndOperandIndex)
        {
            return RangeAttr::get(Range{ begin, endOperandIndex, increment }, parser.getBuilder().getContext());
        }
        else if (isEndInt)
        {
            return RangeAttr::get(Range{ begin, endInt, increment }, parser.getBuilder().getContext());
        }

        llvm_unreachable("unexpected");
        return {};
    }

    void print(RangeAttr attr, mlir::DialectAsmPrinter& printer)
    {
        auto range = attr.getValue();

        printer << "range";
        printer << range;
    }

    //
    // SplitIndexAttr
    //
    SplitIndexAttr SplitIndexAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    SplitIndex SplitIndexAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    SplitIndexAttr parseSplitIndex(mlir::DialectAsmParser& parser)
    {
        if (failed(parser.parseLBrace()))
            return {};

        auto outerAttr = parseIndex(parser);
        if (!outerAttr)
        {
            return {};
        }

        if (failed(parser.parseComma()))
            return {};

        auto innerAttr = parseIndex(parser);
        if (!innerAttr)
        {
            return {};
        }

        if (failed(parser.parseRBrace()))
            return {};

        return SplitIndexAttr::get(SplitIndex{ outerAttr.getValue(), innerAttr.getValue() }, parser.getBuilder().getContext());
    }

    void print(SplitIndexAttr attr, mlir::DialectAsmPrinter& printer)
    {
        auto splitIndex = attr.getValue();
        printer << "splitindex{"
                << "{" << splitIndex.outer.GetName() << ',' << splitIndex.outer.GetId() << "},"
                << "{" << splitIndex.inner.GetName() << ',' << splitIndex.inner.GetId() << "}}";
    }

    //
    // TransformedDomainAttr
    //
    namespace detail
    {
        TransformedDomainAttributeStorage::TransformedDomainAttributeStorage(const TransformedDomain& value_) :
            value(value_)
        {
        }

        TransformedDomainAttributeStorage::TransformedDomainAttributeStorage(const KeyTy& value_) :
            value(value_)
        {
        }

        TransformedDomainAttributeStorage::KeyTy TransformedDomainAttributeStorage::getKey(const TransformedDomain& domain)
        {
            KeyTy result;
            result.dimensions = domain.GetDimensions();

            auto indices = domain.GetIndices();
            for (const auto& index : indices)
            {
                auto expr = domain.GetIndexExpr(index);
                auto range = domain.GetIndexRange(index);
                auto padding = domain.GetIndexPadding(index);
                result.indices.push_back({ index, expr, range, padding });
            }

            return result;
        }

        bool TransformedDomainAttributeStorage::operator==(const KeyTy& key) const
        {
            auto thisKey = TransformedDomainAttributeStorage::getKey(getValue());
            return key == thisKey;
        }

        TransformedDomainAttributeStorage* TransformedDomainAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
        {
            return new (allocator.allocate<TransformedDomainAttributeStorage>()) TransformedDomainAttributeStorage(key);
        }
    } // namespace detail

    TransformedDomainAttr TransformedDomainAttr::get(const ValueType& value, mlir::MLIRContext* context)
    {
        return Base::get(context, value);
    }

    TransformedDomain TransformedDomainAttr::getValue() const
    {
        return getImpl()->getValue();
    }

    TransformedDomainAttr parseTransformedDomain(mlir::DialectAsmParser& parser)
    {
        std::vector<Index> dimensions;
        if (failed(parser.parseLBrace()))
            return {};

        if (failed(parser.parseKeyword("dims")))
            return {};
        if (failed(parser.parseColon()))
            return {};
        if (failed(parser.parseLBrace()))
            return {};
        if (failed(parser.parseOptionalRBrace())) // something inside
        {
            do
            {
                auto dim = parseIndex(parser);
                dimensions.push_back(dim.getValue());
            } while (succeeded(parser.parseOptionalComma()));
            if (failed(parser.parseRBrace()))
                return {};
        }

        if (failed(parser.parseComma()))
            return {};

        std::vector<std::tuple<Index, AffineExpression, Range, std::vector<int64_t>>> indexInfos;
        if (failed(parser.parseKeyword("indices")))
            return {};
        if (failed(parser.parseColon()))
            return {};

        if (failed(parser.parseLBrace()))
            return {};
        if (failed(parser.parseOptionalRBrace())) // something inside
        {
            do
            {
                if (failed(parser.parseLBrace()))
                    return {};

                auto index = parseIndex(parser).getValue();
                if (failed(parser.parseColon()))
                    return {};
                auto range = parseRange(parser).getValue();
                AffineExpression expr;
                if (succeeded(parser.parseOptionalEqual()))
                {
                    // Example: {i,14} : {0:16:1} = {(d0, d1) -> (d0 + d1), {{i_o,19}, {i_i,20}}}
                    mlir::AffineMap map;
                    std::vector<Index> exprIndices;

                    if (failed(parser.parseLBrace()))
                        return {};
                    if (failed(parser.parseAffineMap(map)))
                        return {};
                    if (failed(parser.parseComma()))
                        return {};

                    // parse indices
                    if (failed(parser.parseLBrace()))
                        return {};
                    if (failed(parser.parseOptionalRBrace())) // something inside
                    {
                        do
                        {
                            auto exprIndex = parseIndex(parser);
                            exprIndices.push_back(exprIndex.getValue());
                        } while (succeeded(parser.parseOptionalComma()));
                        if (failed(parser.parseRBrace()))
                            return {};
                    }

                    if (failed(parser.parseRBrace()))
                        return {};

                    // put the expr from the map and the indices into an AffineExpression
                    if (map.getNumResults() != 1)
                    {
                        parser.emitError(parser.getCurrentLocation(), "Invalid map in affine expression attribute");
                        return {};
                    }

                    expr = { map.getResult(0), exprIndices };
                }

                std::vector<int64_t> padding;
                if (succeeded(parser.parseOptionalComma()))
                {
                    if (failed(parser.parseLBrace()))
                        return {};
                    do
                    {
                        int padValue = 0;
                        parser.parseInteger(padValue);
                        padding.push_back(padValue);
                    } while (succeeded(parser.parseOptionalComma()));
                    if (failed(parser.parseRBrace()))
                        return {};
                }

                if (failed(parser.parseRBrace()))
                    return {};

                indexInfos.emplace_back(index, expr, range, padding);
            } while (succeeded(parser.parseOptionalComma()));
            if (failed(parser.parseRBrace()))
                return {};
        }

        if (failed(parser.parseRBrace()))
            return {};

        TransformedDomain::AttributeKey key;
        key.dimensions = dimensions;
        key.indices = indexInfos;
        return TransformedDomainAttr::get({ key }, parser.getBuilder().getContext());
    }

    void print(TransformedDomainAttr attr, mlir::DialectAsmPrinter& printer)
    {
        auto domain = attr.getValue();
        printer << "xfdomain{";

        // dimensions
        auto dims = domain.GetDimensions();
        printer << "dims: {";
        llvm::interleaveComma(dims, printer); // interleaveComma relies on op<<
        printer << "}, ";

        // all indices
        auto numIndices = domain.NumIndices();
        auto indices = domain.GetIndices();
        printer << "indices: {";
        for (int i = 0; i < numIndices; ++i)
        {
            // index
            auto index = indices[i];
            printer << "{";
            printer << index;

            // range
            printer << " : " << domain.GetIndexRange(index);

            // expr
            // print expression: expr:{<expr>, <indices>}
            auto expr = domain.GetIndexExpr(index);
            if (!expr.IsIdentity())
            {
                printer << " = {";

                // printer << expr.GetAffineExpr() << ",";
                // There isn't any public API for parsing AffineExprs, so we have to encode the expr in a map.
                auto exprIndices = expr.GetIndices();
                auto map = mlir::AffineMap::get(exprIndices.size(), 0, expr.GetAffineExpr());
                printer << map << ", ";

                printer << "{";
                llvm::interleaveComma(exprIndices, printer); // interleaveComma relies on op<<
                printer << "}";
                printer << "}";
            }

            // padding
            auto padding = domain.GetIndexPadding(index);
            if (!padding.empty())
            {
                printer << ", ";
                printer << "{";
                llvm::interleaveComma(padding, printer);
                printer << "}";
            }

            printer << "}";
            if (i < numIndices - 1)
            {
                printer << ", ";
            }
        }
        printer << "}";
        printer << "}";
    }

    //
    // Hash functions for storage key types
    //
    llvm::hash_code hash_value(const std::string& s)
    {
        return std::hash<std::string>()(s);
    }

    llvm::hash_code hash_value(const Index& index)
    {
        return std::hash<Index>()(index);
    }

    llvm::hash_code hash_value(const OperandIndex& index)
    {
        return std::hash<OperandIndex>()(index);
    }

    llvm::hash_code hash_value(const IndexRange& indexRange)
    {
        return llvm::hash_combine(indexRange.GetIndex(), indexRange.GetRange());
    }

    llvm::hash_code hash_value(const IterationDomain& domain)
    {
        auto result = llvm::hash_value("IterationDomain");
        for (const auto& range : domain.GetRanges())
        {
            result = llvm::hash_combine(result, hash_value(range));
        }
        return result;
    }

    llvm::hash_code hash_value(const Range& range)
    {
        if (range.HasConstantEnd())
        {
            return llvm::hash_combine(range.Begin(), range.End(), range.Increment());
        }
        else if (range.HasIndexEnd())
        {
            return llvm::hash_combine(range.Begin(), range.EndIndex(), range.Increment());
        }
        else if (range.HasOperandIndexEnd())
        {
            return llvm::hash_combine(range.Begin(), range.EndOperandIndex(), range.Increment());
        }
        else if (range.HasVariableEnd())
        {
            return llvm::hash_combine(range.Begin(), hash_value(range.VariableEnd()), range.Increment());
        }
        llvm_unreachable("Unhandled Range case");
    }

    llvm::hash_code hash_value(const SplitIndex& splitIndex)
    {
        return llvm::hash_combine(splitIndex.outer, splitIndex.inner);
    }

    llvm::hash_code hash_value(const TransformedDomain& domain)
    {
        llvm::hash_code result = hash_value("domain");
        for (auto d : domain.GetDimensions())
        {
            result = llvm::hash_combine(result, d);
        }

        for (auto i : domain.GetIndices())
        {
            auto range = domain.GetIndexRange(i);
            auto expr = domain.GetIndexExpr(i);
            result = llvm::hash_combine(result, i);
            result = llvm::hash_combine(result, range);
            auto exprIndices = expr.GetIndices();
            for (const auto& exprIndex : exprIndices)
            {
                result = llvm::hash_combine(result, exprIndex);
            }
            result = llvm::hash_combine(result, expr.GetAffineExpr());
        }

        return result;
    }
} // namespace loopnest
} // namespace accera::ir
