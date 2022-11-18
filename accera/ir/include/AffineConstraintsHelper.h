////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>

#include <optional>
#include <stack>
#include <unordered_map>

namespace accera::ir
{
namespace util
{

    class AffineConstraintsHelper
    {
    public:
        struct IdWrapper
        {
            enum class Type
            {
                Dimension,
                Symbol,
                Local
            };

            IdWrapper(unsigned id_, Type type_) :
                typeId(id_),
                type(type_) {}

            static IdWrapper GetDimId(unsigned id);
            static IdWrapper GetSymbolId(unsigned id);
            static IdWrapper GetLocalId(unsigned id);
            static IdWrapper FromFullId(unsigned fullId, const mlir::FlatAffineConstraints& cst);

            unsigned GetFullId(const mlir::FlatAffineConstraints& cst) const;
            mlir::AffineExpr GetExpr(mlir::MLIRContext* context) const;

            bool operator<(const IdWrapper& other) const { return typeId < other.typeId; }

            unsigned typeId;
            Type type;
        };

        AffineConstraintsHelper(mlir::MLIRContext* context) :
            _context(context), _debugPrinting(false) {}
        AffineConstraintsHelper(const mlir::FlatAffineValueConstraints& constraints, mlir::MLIRContext* context) :
            _context(context), _cst(constraints), _debugPrinting(false) {}

        AffineConstraintsHelper(const AffineConstraintsHelper& other) = default;

        AffineConstraintsHelper Clone() const;
        mlir::MLIRContext* GetContext() const;
        const mlir::FlatAffineValueConstraints& GetConstraints() const;
        bool IsEmpty() const;
        void SetDebugPrinting(bool enabled);

        IdWrapper AddDim();
        IdWrapper AddSymbol();
        IdWrapper AddSymbol(mlir::Value value);
        IdWrapper AddSymbol(const mlir::AffineValueMap& valueMap, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        IdWrapper AddConstant(int64_t constant);

        // Manage value handles for ids
        void SetValue(const IdWrapper& id, mlir::Value value);
        std::optional<mlir::Value> GetValue(const IdWrapper& id) const;

        std::optional<IdWrapper> GetIdOfValue(mlir::Value val) const;

        // Set equalities
        void SetEqual(const IdWrapper& id, int64_t constant);
        void SetEqual(const IdWrapper& id, const IdWrapper& other);
        void SetEqualExpr(const IdWrapper& id, mlir::AffineExpr expr, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void SetEqualMap(const IdWrapper& id, mlir::AffineMap map, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void SetEqualMap(const IdWrapper& id, mlir::AffineMap unalignedMap, mlir::ValueRange operands, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void SetEqualMap(const IdWrapper& id, mlir::AffineValueMap unalignedValueMap, std::optional<mlir::AffineExpr> localExpr = std::nullopt);

        // Add lower and upper bounds

        // Add lower/upper bound to constant
        void AddLowerBound(const IdWrapper& id, int64_t constantLowerBound);
        void AddUpperBound(const IdWrapper& id, int64_t constantUpperBound, bool exclusive = true);

        // Add lower/upper bound to other id
        void AddLowerBound(const IdWrapper& id, const IdWrapper& bound);
        void AddUpperBound(const IdWrapper& id, const IdWrapper& bound, bool exclusive = true);

        // Add lower/upper bound to aligned expr
        void AddLowerBoundExpr(const IdWrapper& id, mlir::AffineExpr lowerBoundExpr, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void AddUpperBoundExpr(const IdWrapper& id, mlir::AffineExpr upperBoundExpr, bool exclusive = true, std::optional<mlir::AffineExpr> localExpr = std::nullopt);

        // Add lower/upper bound to aligned map
        void AddLowerBoundMap(const IdWrapper& id, mlir::AffineMap lowerBoundMap, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void AddUpperBoundMap(const IdWrapper& id, mlir::AffineMap upperBoundMap, bool exclusive = true, std::optional<mlir::AffineExpr> localExpr = std::nullopt);

        // Add lower/upper bound to unaligned map
        void AddLowerBoundMap(const IdWrapper& id, mlir::AffineMap unalignedLowerBoundMap, mlir::ValueRange operands, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void AddUpperBoundMap(const IdWrapper& id, mlir::AffineMap unalignedUpperBoundMap, mlir::ValueRange operands, bool exclusive = true, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void AddLowerBoundMap(const IdWrapper& id, mlir::AffineValueMap unalignedLowerBoundValueMap, std::optional<mlir::AffineExpr> localExpr = std::nullopt);
        void AddUpperBoundMap(const IdWrapper& id, mlir::AffineValueMap unalignedUpperBoundValueMap, bool exclusive = true, std::optional<mlir::AffineExpr> localExpr = std::nullopt);

        void ProjectOut(const IdWrapper& id);
        void ProjectOut(const std::vector<IdWrapper>& ids);

        void Simplify();

        void ResolveSymbolsToAffineApplyOps(mlir::OpBuilder& builder, mlir::Location loc);

        std::pair<mlir::AffineValueMap, mlir::AffineValueMap> GetLowerAndUpperBound(const IdWrapper& id, mlir::OpBuilder& builder, mlir::Location loc, const std::vector<IdWrapper>& idsToProjectOut = {}) const;

        mlir::AffineMap AlignAffineValueMap(mlir::AffineValueMap& affineValueMap) const;
        mlir::AffineMap GetMap(const std::vector<mlir::AffineExpr>& exprs) const;
        mlir::AffineMap GetMap(mlir::AffineExpr expr) const;

    protected:
        mlir::MLIRContext* _context;

    private:
        std::vector<mlir::AffineExpr> GetLocalExprsVec() const;
        std::vector<mlir::Value> GetConstraintValuesForDimId(const IdWrapper& id) const;

        bool IsExprResolvable(const mlir::AffineExpr& expr);

        mlir::Value ResolveLocalExpr(mlir::OpBuilder& builder, mlir::Location loc, const mlir::AffineExpr& expr, mlir::Value noneFillValue);

        unsigned GetNumLocals() const;

        void MaybeDebugPrint() const;

        void AddBound(const mlir::IntegerPolyhedron::BoundType& type, const IdWrapper& id, int64_t value);
        void AddBound(const mlir::IntegerPolyhedron::BoundType& type, const IdWrapper& id, mlir::AffineMap map, std::optional<mlir::AffineExpr> localExpr = std::nullopt);

        std::stack<mlir::AffineExpr> _localExprs; // New local exprs get added at the beginning of the sequence of local exprs, so hold them in a stack for simplicity
        std::unordered_map<int64_t, IdWrapper> _heldConstants;
        mlir::FlatAffineValueConstraints _cst;
        bool _debugPrinting;
    };

} // namespace util
} // namespace accera::ir
