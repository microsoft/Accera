////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AffineConstraintsHelper.h"

#include <utilities/include/Exception.h>

#include <queue>

using namespace accera::utilities;

namespace accera::ir
{
namespace util
{
    using IdWrapper = AffineConstraintsHelper::IdWrapper;

    IdWrapper IdWrapper::GetDimId(unsigned id)
    {
        return IdWrapper(id, IdWrapper::Type::Dimension);
    }
    IdWrapper IdWrapper::GetSymbolId(unsigned id)
    {
        return IdWrapper(id, IdWrapper::Type::Symbol);
    }
    IdWrapper IdWrapper::GetLocalId(unsigned id)
    {
        return IdWrapper(id, IdWrapper::Type::Local);
    }

    IdWrapper IdWrapper::FromFullId(unsigned fullId, const mlir::FlatAffineConstraints& cst)
    {
        // Columns are ordered as [ dims..., syms..., locals... ]
        // The "full id" is the absolute column index, but we want to keep track of what the index within the
        // column segment is rather than the absolute index.
        // For example, if we have 1 dim and 2 symbols, we have columns [0, 1, 2] mapping to [d0, s0, s1]
        // and if we are given full id = 1, then that is referring to the 0'th index symbol s0.
        // If we were to then add a dimension to the system, we would have columns [0, 1, 2, 3] mapping to [d0, d1, s0, s1] and 
        // full id = 1 would then be referencing the newly created 1st index dim d1.
        // This is why we choose to hold onto the information "s0" rather than "1", so we can add dimensions without concern about how it
        // shifts the absolute column position of other ids. (and similarly for locals wrt dims and syms)

        if (fullId >= cst.getNumIds())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Invalid id for the given constraints");
        }

        bool local = fullId >= cst.getNumDimAndSymbolIds();
        bool symbol = !local && (fullId >= cst.getNumDimIds());
        bool dim = !local && !symbol;
        if (dim)
        {
            return IdWrapper(fullId, IdWrapper::Type::Dimension);
        }
        else if (symbol)
        {
            auto symId = fullId - cst.getNumDimIds();
            return IdWrapper(symId, IdWrapper::Type::Symbol);
        }
        else if (local)
        {
            auto localId = fullId - cst.getNumDimAndSymbolIds();
            return IdWrapper(localId, IdWrapper::Type::Local);
        }
        else
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Attempting to create an IdWrapper that has an in-range absolute column id but is not a dim, sym, or local");
        }
    }

    unsigned IdWrapper::GetFullId(const mlir::FlatAffineConstraints& cst) const
    {
        switch (type)
        {
        case IdWrapper::Type::Dimension:
            return typeId;
        case IdWrapper::Type::Symbol:
            return cst.getNumDimIds() + typeId;
        case IdWrapper::Type::Local:
            return cst.getNumDimAndSymbolIds() + typeId;
        default:
            throw LogicException(LogicExceptionErrors::illegalState, "Invalid IdWrapper state");
        }
    }

    mlir::AffineExpr IdWrapper::GetExpr(mlir::MLIRContext* context) const
    {
        switch (type)
        {
        case IdWrapper::Type::Dimension:
            return mlir::getAffineDimExpr(typeId, context);
        case IdWrapper::Type::Symbol:
            return mlir::getAffineSymbolExpr(typeId, context);
        case IdWrapper::Type::Local:
            throw LogicException(LogicExceptionErrors::illegalState, "Local ids are neither dimensions nor symbols in the constraint system so an expr cannot be generated");
        default:
            throw LogicException(LogicExceptionErrors::illegalState, "Invalid IdWrapper state");
        }
    }

    AffineConstraintsHelper AffineConstraintsHelper::Clone() const
    {
        return AffineConstraintsHelper(*this);
    }

    mlir::MLIRContext* AffineConstraintsHelper::GetContext() const
    {
        return _context;
    }

    const mlir::FlatAffineValueConstraints& AffineConstraintsHelper::GetConstraints() const
    {
        return _cst;
    }

    bool AffineConstraintsHelper::IsEmpty() const
    {
        return _cst.isEmpty();
    }

    void AffineConstraintsHelper::SetDebugPrinting(bool enabled)
    {
        _debugPrinting = enabled;
    }

    IdWrapper AffineConstraintsHelper::AddDim()
    {
        unsigned dimId = _cst.appendDimId();
        auto idWrapper = IdWrapper::GetDimId(dimId);
        MaybeDebugPrint();
        return idWrapper;
    }

    IdWrapper AffineConstraintsHelper::AddSymbol()
    {
        unsigned symId = _cst.appendSymbolId();
        auto idWrapper = IdWrapper::GetSymbolId(symId);
        MaybeDebugPrint();
        return idWrapper;
    }

    IdWrapper AffineConstraintsHelper::AddSymbol(mlir::Value value)
    {
        auto heldValOpt = GetIdOfValue(value);
        if (heldValOpt.has_value())
        {
            return *heldValOpt;
        }
        unsigned symId = _cst.appendSymbolId(mlir::ValueRange{ value });
        auto idWrapper = IdWrapper::GetSymbolId(symId);
        MaybeDebugPrint();
        return idWrapper;
    }

    IdWrapper AffineConstraintsHelper::AddSymbol(const mlir::AffineValueMap& valueMap, std::optional<mlir::AffineExpr> localExpr)
    {
        IdWrapper symId = AddSymbol();
        SetEqualMap(symId, valueMap, localExpr);
        MaybeDebugPrint();
        return symId;
    }

    IdWrapper AffineConstraintsHelper::AddConstant(int64_t constant)
    {
        auto findIter = _heldConstants.find(constant);
        if (findIter != _heldConstants.end())
        {
            return findIter->second;
        }

        auto idWrapper = AddSymbol();
        SetEqual(idWrapper, constant);
        _heldConstants.insert({ constant, idWrapper });
        MaybeDebugPrint();
        return idWrapper;
    }

    // Set value handles for ids
    void AffineConstraintsHelper::SetValue(const IdWrapper& id, mlir::Value value)
    {
        _cst.setValue(id.GetFullId(_cst), value);
    }
    std::optional<mlir::Value> AffineConstraintsHelper::GetValue(const IdWrapper& id) const
    {
        auto fullId = id.GetFullId(_cst);
        if (_cst.hasValue(fullId))
        {
            return _cst.getValue(fullId);
        }
        else
        {
            return std::nullopt;
        }
    }

    std::optional<IdWrapper> AffineConstraintsHelper::GetIdOfValue(mlir::Value val) const
    {
        if (_cst.containsId(val))
        {
            unsigned id = 0;
            bool found = _cst.findId(val, &id);
            assert(found);
            return IdWrapper::FromFullId(id, _cst);
        }
        else
        {
            return std::nullopt;
        }
    }

    // Set equalities
    void AffineConstraintsHelper::SetEqual(const IdWrapper& id, int64_t constant)
    {
        SetEqualExpr(id, mlir::getAffineConstantExpr(constant, _context));
    }
    void AffineConstraintsHelper::SetEqual(const IdWrapper& id, const IdWrapper& other)
    {
        SetEqualExpr(id, other.GetExpr(_context));
    }
    void AffineConstraintsHelper::SetEqualExpr(const IdWrapper& id, mlir::AffineExpr expr, std::optional<mlir::AffineExpr> localExpr)
    {
        SetEqualMap(id, GetMap(expr), localExpr);
    }
    void AffineConstraintsHelper::SetEqualMap(const IdWrapper& id, mlir::AffineMap map, std::optional<mlir::AffineExpr> localExpr)
    {
        AddBound(mlir::IntegerPolyhedron::BoundType::EQ, id, map, localExpr);
    }
    void AffineConstraintsHelper::SetEqualMap(const IdWrapper& id, mlir::AffineMap unalignedMap, mlir::ValueRange operands, std::optional<mlir::AffineExpr> localExpr)
    {
        SetEqualMap(id, mlir::AffineValueMap(unalignedMap, operands), localExpr);
    }
    void AffineConstraintsHelper::SetEqualMap(const IdWrapper& id, mlir::AffineValueMap unalignedValueMap, std::optional<mlir::AffineExpr> localExpr)
    {
        auto alignedMap = AlignAffineValueMap(unalignedValueMap);
        SetEqualMap(id, alignedMap, localExpr);
    }

    // Set lower and upper bounds

    // Set lower/upper bound to constant
    // Note: even though mlir::FlatAffineValueConstraints inherits mlir::IntegerPolyhedron::addBound for constants,
    //       we do not use that API and instead prefer to use the AffineMap-based bounds for consistency in how our
    //       wrapper handles inclusive/exclusive upper bounds.
    void AffineConstraintsHelper::AddLowerBound(const IdWrapper& id, int64_t constantLowerBound)
    {
        AddLowerBoundExpr(id, mlir::getAffineConstantExpr(constantLowerBound, _context));
    }
    void AffineConstraintsHelper::AddUpperBound(const IdWrapper& id, int64_t constantUpperBound, bool exclusive)
    {
        AddUpperBoundExpr(id, mlir::getAffineConstantExpr(constantUpperBound, _context), exclusive);
    }

    // Set lower/upper bound to other id
    void AffineConstraintsHelper::AddLowerBound(const IdWrapper& id, const IdWrapper& bound)
    {
        AddLowerBoundExpr(id, bound.GetExpr(_context));
    }
    void AffineConstraintsHelper::AddUpperBound(const IdWrapper& id, const IdWrapper& bound, bool exclusive)
    {
        AddUpperBoundExpr(id, bound.GetExpr(_context), exclusive);
    }

    // Set lower/upper bound to aligned expr
    void AffineConstraintsHelper::AddLowerBoundExpr(const IdWrapper& id, mlir::AffineExpr lowerBoundExpr, std::optional<mlir::AffineExpr> localExpr)
    {
        AddLowerBoundMap(id, GetMap(lowerBoundExpr), localExpr);
    }
    void AffineConstraintsHelper::AddUpperBoundExpr(const IdWrapper& id, mlir::AffineExpr upperBoundExpr, bool exclusive, std::optional<mlir::AffineExpr> localExpr)
    {
        AddUpperBoundMap(id, GetMap(upperBoundExpr), exclusive, localExpr);
    }

    // Set lower/upper bound to aligned map
    void AffineConstraintsHelper::AddLowerBoundMap(const IdWrapper& id, mlir::AffineMap lowerBoundMap, std::optional<mlir::AffineExpr> localExpr)
    {
        AddBound(mlir::IntegerPolyhedron::BoundType::LB, id, lowerBoundMap, localExpr);
    }
    void AffineConstraintsHelper::AddUpperBoundMap(const IdWrapper& id, mlir::AffineMap upperBoundMap, bool exclusive, std::optional<mlir::AffineExpr> localExpr)
    {
        if (!exclusive)
        {
            // When adding an upper bound map constraint, mlir::FlatAffineValueConstraints::addBound will interpret the given map as an exclusive upper bound
            // and subtract 1 from it before holding onto its data.
            // If we want an inclusive upper bound instead, then we need to manually add 1 to the expressions before adding them as bounds
            mlir::MutableAffineMap mutableMap(upperBoundMap);
            for (unsigned resultIdx = 0; resultIdx < mutableMap.getNumResults(); ++resultIdx)
            {
                mutableMap.setResult(resultIdx, mutableMap.getResult(resultIdx) + 1);
            }
            upperBoundMap = mutableMap.getAffineMap();
        }
        AddBound(mlir::IntegerPolyhedron::BoundType::UB, id, upperBoundMap, localExpr);
    }

    // Set lower/upper bound to unaligned map
    void AffineConstraintsHelper::AddLowerBoundMap(const IdWrapper& id, mlir::AffineMap unalignedLowerBoundMap, mlir::ValueRange operands, std::optional<mlir::AffineExpr> localExpr)
    {
        AddLowerBoundMap(id, mlir::AffineValueMap(unalignedLowerBoundMap, operands), localExpr);
    }
    void AffineConstraintsHelper::AddUpperBoundMap(const IdWrapper& id, mlir::AffineMap unalignedUpperBoundMap, mlir::ValueRange operands, bool exclusive, std::optional<mlir::AffineExpr> localExpr)
    {
        AddUpperBoundMap(id, mlir::AffineValueMap(unalignedUpperBoundMap, operands), exclusive, localExpr);
    }
    void AffineConstraintsHelper::AddLowerBoundMap(const IdWrapper& id, mlir::AffineValueMap unalignedLowerBoundValueMap, std::optional<mlir::AffineExpr> localExpr)
    {
        auto alignedMap = AlignAffineValueMap(unalignedLowerBoundValueMap);
        AddLowerBoundMap(id, alignedMap, localExpr);
    }
    void AffineConstraintsHelper::AddUpperBoundMap(const IdWrapper& id, mlir::AffineValueMap unalignedUpperBoundValueMap, bool exclusive, std::optional<mlir::AffineExpr> localExpr)
    {
        auto alignedMap = AlignAffineValueMap(unalignedUpperBoundValueMap);
        AddUpperBoundMap(id, alignedMap, exclusive, localExpr);
    }

    mlir::AffineMap AffineConstraintsHelper::AlignAffineValueMap(mlir::AffineValueMap& affineValueMap) const
    {
        auto map = affineValueMap.getAffineMap();
        assert(map.getNumResults() == 1 && "Only supports single-result maps");
        auto operands = affineValueMap.getOperands();
        auto alignedMap = _cst.computeAlignedMap(map, operands);
        assert(alignedMap.getNumResults() == 1);
        return alignedMap;
    }

    void AffineConstraintsHelper::ProjectOut(const IdWrapper& id)
    {
        _cst.projectOut(id.GetFullId(_cst));
    }

    void AffineConstraintsHelper::ProjectOut(const std::vector<IdWrapper>& ids)
    {
        // Sort the ids in descending full id order
        std::vector<IdWrapper> sortedIds = ids;
        std::sort(sortedIds.begin(), sortedIds.end(), [&](const IdWrapper& lhs, const IdWrapper& rhs) {
            return lhs.GetFullId(_cst) > rhs.GetFullId(_cst);
        });
        for (auto id : sortedIds)
        {
            ProjectOut(id);
        }
    }

    void AffineConstraintsHelper::Simplify()
    {
        _cst.removeTrivialRedundancy();
        _cst.removeRedundantInequalities();
        _cst.removeRedundantConstraints();
    }

    void AffineConstraintsHelper::ResolveSymbolsToAffineApplyOps(mlir::OpBuilder& builder, mlir::Location loc)
    {
        mlir::Value zeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        std::vector<mlir::AffineExpr> localExprs = GetLocalExprsVec();
        std::queue<unsigned> dimSymIdsToResolve;
        std::queue<unsigned> localIdsToResolve;
        for (unsigned id = 0; id < _cst.getNumDimAndSymbolIds(); ++id)
        {
            if (!_cst.hasValue(id))
            {
                dimSymIdsToResolve.push(id);
            }
        }
        for (unsigned localIdx = 0; localIdx < localExprs.size(); ++localIdx)
        {
            localIdsToResolve.push(localIdx);
        }
        while (!dimSymIdsToResolve.empty() || !localIdsToResolve.empty())
        {
            // Try to resolve as many local exprs as early as we can to speed up the process
            auto startingLocalSize = localIdsToResolve.size();
            if (!localIdsToResolve.empty())
            {
                for (unsigned localResolveCounter = 0; localResolveCounter < startingLocalSize; ++localResolveCounter)
                {
                    auto currentLocalId = localIdsToResolve.front();
                    localIdsToResolve.pop();
                    auto fullId = _cst.getNumDimAndSymbolIds() + currentLocalId;
                    if (!_cst.hasValue(fullId))
                    {
                        if (IsExprResolvable(localExprs[currentLocalId]))
                        {
                            auto localValue = ResolveLocalExpr(builder, loc, localExprs[currentLocalId], zeroVal);
                            _cst.setValue(fullId, localValue);
                        }
                        else
                        {
                            // Try to resolve it later when we have more dims and symbols resolved
                            localIdsToResolve.push(currentLocalId);
                        }
                    }
                }
            }

            // Try to handle as many ids as we can currently, if we run through every ID and handle nothing, then throw
            auto startingDimSymSize = dimSymIdsToResolve.size();
            for (unsigned idCount = 0; idCount < startingDimSymSize; ++idCount)
            {
                auto currentId = dimSymIdsToResolve.front();
                dimSymIdsToResolve.pop();
                if (_cst.hasValue(currentId))
                {
                    // Already resolved to an mlir::Value instance, so just remove this id from the queue and continue
                    continue;
                }

                // If it doesn't have a value, then it must have an equality constraint with other dimensions / symbols
                llvm::SmallVector<unsigned, 4> lbIndices;
                llvm::SmallVector<unsigned, 4> ubIndices;
                llvm::SmallVector<unsigned, 4> eqIndices;
                _cst.getLowerAndUpperBoundIndices(currentId, &lbIndices, &ubIndices, &eqIndices);

                std::vector<unsigned> eqIndicesVec(eqIndices.begin(), eqIndices.end());

                assert(!eqIndicesVec.empty() && "Requires at least one equality constraint on a symbol without a value");
                // Try to resolve one of the equalities to an affine apply op
                bool handled = false;
                for (auto eqIndex : eqIndicesVec)
                {
                    auto eqCoeffs = _cst.getEquality(eqIndex);
                    // Check if we have values for every non-zero coefficient
                    bool haveAllValuesNeeded = true;
                    mlir::AffineExpr coeffExpr = builder.getAffineConstantExpr(0);
                    // Create a new indexing that creates a map out of only symbol ids that get used.
                    // Not all symbols have values yet, so don't require getting an operand for one unless we depend on it here
                    unsigned currentScopedSymId = 0;
                    std::vector<mlir::Value> operands;
                    bool negate = false;
                    for (unsigned coeffIdx = 0; coeffIdx < eqCoeffs.size(); ++coeffIdx)
                    {
                        if (coeffIdx == currentId)
                        {
                            // Determine whether we need to negate the expression at the end
                            int64_t currentIdCoeff = eqCoeffs[coeffIdx];
                            if (currentIdCoeff != -1 && currentIdCoeff != 1)
                            {
                                throw LogicException(LogicExceptionErrors::illegalState, "Non-1 coefficients on the value being resolved aren't supported");
                            }

                            // A vector of coefficients is something like [1, 0, -2, 8] (these could be any integers)
                            // Aligned with columns, these are:
                            //   x  y  z  const
                            // [ 1  0 -2      8]
                            // And get interpretted as "(1)x + (0)y + (-2)z + (8) = 0", simplifying to "x - 2z + 8 = 0"
                            // So if we are trying to solve for x, we want to derive "x = 2z - 8" from this coefficient vector.
                            // Note: the coeffExpr we've been building up from everything except for the column we're resolving is holding
                            // the coefficients as they exist in the vector, not as they would exist once we re-arrange the equation
                            // to have our resolve column alone on one side.
                            // So, we have coeffExpr = <(d0) -> (-2*d0 + 8)>
                            // and if our resolution column's coefficient is positive we want to negate this and turn it into
                            // <(d0) -> (2*d0 - 8)>, which will match our desired "x = 2z - 8"

                            // Negate the other coefficients if our resolve column's coefficient is positive
                            negate = currentIdCoeff > 0;
                        }
                        else if (eqCoeffs[coeffIdx] != 0)
                        {
                            if (coeffIdx == _cst.getNumIds())
                            {
                                // The final coefficient is the constant offset
                                coeffExpr = coeffExpr + eqCoeffs[coeffIdx];
                            }
                            else
                            {
                                if (!_cst.hasValue(coeffIdx))
                                {
                                    // We have a non-zero coefficient, but do not have a value for that position in the constraint system yet
                                    // So try to resolve other IDs first then come back to this one
                                    haveAllValuesNeeded = false;
                                    break;
                                }
                                coeffExpr = coeffExpr + eqCoeffs[coeffIdx] * builder.getAffineSymbolExpr(currentScopedSymId++);
                                operands.push_back(_cst.getValue(coeffIdx));
                            }
                        }
                    }
                    if (haveAllValuesNeeded)
                    {
                        // We have values for every non-zero, coefficient, so construct an AffineApplyOp as a function of those values
                        if (negate)
                        {
                            coeffExpr = -1 * coeffExpr;
                        }
                        auto coeffMap = mlir::AffineMap::get(0, currentScopedSymId, coeffExpr);

                        llvm::SmallVector<mlir::Value, 4> operandsSmallVec(operands.begin(), operands.end());
                        mlir::fullyComposeAffineMapAndOperands(&coeffMap, &operandsSmallVec);
                        mlir::canonicalizeMapAndOperands(&coeffMap, &operandsSmallVec);
                        mlir::Value resolvedVal = builder.create<mlir::AffineApplyOp>(loc, coeffMap, operandsSmallVec);
                        _cst.setValue(currentId, resolvedVal);
                        handled = true;
                        break;
                    }
                }
                if (!handled)
                {
                    // Try to handle it again later when other terms are resolved
                    dimSymIdsToResolve.push(currentId);
                }
            }
            auto endingDimSymSize = dimSymIdsToResolve.size();
            auto endingLocalSize = localIdsToResolve.size();
            bool handledAtLeastOneDimSymID = endingDimSymSize < startingDimSymSize;
            bool handledAtLeastOneLocalID = endingLocalSize < startingLocalSize;
            if (!handledAtLeastOneDimSymID && !handledAtLeastOneLocalID)
            {
                throw LogicException(LogicExceptionErrors::illegalState, "Failed to resolve any ids (columns) in the constraint system to values during a resolution pass while some still remain unresolved");
            }
        }

        // Sanity check
        if (!localIdsToResolve.empty() || !dimSymIdsToResolve.empty())
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Failed to resolve all ids (columns) in the constraint system to values");
        }
    }

    std::pair<mlir::AffineValueMap, mlir::AffineValueMap> AffineConstraintsHelper::GetLowerAndUpperBound(const IdWrapper& id, mlir::OpBuilder& builder, mlir::Location loc, const std::vector<IdWrapper>& idsToProjectOut) const
    {
        auto tmpResolveCst = Clone();
        tmpResolveCst.ProjectOut(idsToProjectOut);
        tmpResolveCst.Simplify();
        tmpResolveCst.ResolveSymbolsToAffineApplyOps(builder, loc);

        auto& cst = tmpResolveCst._cst;

        unsigned dimId = id.GetFullId(cst);
        std::vector<mlir::AffineExpr> localExprs = tmpResolveCst.GetLocalExprsVec();

        // Note: Internally in mlir::FlatAffineValueConstraints::getLowerAndUpperBound, the 'offset' arg is used as the id of the
        //       constraint column to solve for, while the 'pos' arg is only used in `pos + offset` expressions.
        //       In mlir::MemRefRegion::getLowerAndUpperBound, this function is used as though 'pos' gives a column position and 'offset' gives
        //       an offset from that position, whereas in mlir::FlatAffineConstraints::getSliceBounds, the opposite appears to be done...
        //       It's not clear if this is a bug or just a bad choice of arg names.
        auto [lbMap, ubMap] = cst.getLowerAndUpperBound(0 /* pos */, dimId /* offset */, 1 /* num */, cst.getNumDimIds(), localExprs, _context);

        lbMap = mlir::removeDuplicateExprs(lbMap);
        ubMap = mlir::removeDuplicateExprs(ubMap);

        // TODO : we may need to write our own custom lower / upper bound resolution that ports portions of the builtin tools
        // For now we depend on the builtin tools and some workarounds
        // In some situations, seemingly more common with split sizes of 1, the constraints can simplify away some inequalities that are
        // useful for detecting lower and upper bounds via the rather simple checks that getLowerAndUpperBound() performs.
        //      (Put another way: the code that simplifies constraints is smarter than the code that detects lower/upper bounds,
        //       and if the latter was smarter then a single bound might have been detected)
        // When a single lower bound or a single upper bound isn't found, then multiple can be returned. This can cause issues for how
        // Accera uses these bounds.
        // As a workaround, if we have multiple results we attempt to find a single constant result and use that instead
        if (lbMap.getNumResults() > 1)
        {
            auto lbConstOpt = cst.getConstantBound(mlir::IntegerPolyhedron::BoundType::LB, dimId);
            if (lbConstOpt.hasValue())
            {
                auto lbConstExpr = mlir::getAffineConstantExpr(lbConstOpt.getValue(), _context);
                lbMap = mlir::AffineMap::get(lbMap.getNumDims(), lbMap.getNumSymbols(), lbConstExpr);
            }
        }

        if (ubMap.getNumResults() > 1)
        {
            auto ubConstOpt = cst.getConstantBound(mlir::IntegerPolyhedron::BoundType::UB, dimId);
            if (ubConstOpt.hasValue())
            {
                auto ubConstExpr = mlir::getAffineConstantExpr(ubConstOpt.getValue(), _context);
                ubMap = mlir::AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(), ubConstExpr);
            }
        }

        auto constraintOperands = tmpResolveCst.GetConstraintValuesForDimId(id);
        auto simplifiedLBMap = lbMap;
        auto simplifiedUBMap = ubMap;
        llvm::SmallVector<mlir::Value, 4> simplifiedLBOperands(constraintOperands.begin(), constraintOperands.end());
        llvm::SmallVector<mlir::Value, 4> simplifiedUBOperands(constraintOperands.begin(), constraintOperands.end());

        mlir::fullyComposeAffineMapAndOperands(&simplifiedLBMap, &simplifiedLBOperands);
        mlir::fullyComposeAffineMapAndOperands(&simplifiedUBMap, &simplifiedUBOperands);
        mlir::canonicalizeMapAndOperands(&simplifiedLBMap, &simplifiedLBOperands);
        mlir::canonicalizeMapAndOperands(&simplifiedUBMap, &simplifiedUBOperands);

        mlir::AffineValueMap lbValueMap(simplifiedLBMap, simplifiedLBOperands);
        mlir::AffineValueMap ubValueMap(simplifiedUBMap, simplifiedUBOperands);

        return std::make_pair(lbValueMap, ubValueMap);
    }

    mlir::AffineMap AffineConstraintsHelper::GetMap(const std::vector<mlir::AffineExpr>& exprs) const
    {
        return mlir::AffineMap::get(_cst.getNumDimIds(), _cst.getNumSymbolIds(), exprs, _context);
    }

    mlir::AffineMap AffineConstraintsHelper::GetMap(mlir::AffineExpr expr) const
    {
        return GetMap(std::vector<mlir::AffineExpr>{ expr });
    }

    std::vector<mlir::AffineExpr> AffineConstraintsHelper::GetLocalExprsVec() const
    {
        std::vector<mlir::AffineExpr> localExprsVec;
        localExprsVec.reserve(_localExprs.size());
        std::stack<mlir::AffineExpr> localExprsCopy(_localExprs);
        while (!localExprsCopy.empty())
        {
            localExprsVec.push_back(localExprsCopy.top());
            localExprsCopy.pop();
        }
        return localExprsVec;
    }

    std::vector<mlir::Value> AffineConstraintsHelper::GetConstraintValuesForDimId(const IdWrapper& id) const
    {
        llvm::SmallVector<mlir::Value, 4> constraintVals;
        llvm::SmallVector<mlir::Value, 4> constraintValsAfterDim;
        unsigned dimId = id.GetFullId(_cst);
        _cst.getValues(0, dimId, &constraintVals);
        _cst.getValues(dimId + 1, _cst.getNumDimAndSymbolIds(), &constraintValsAfterDim);
        constraintVals.append(constraintValsAfterDim);
        std::vector<mlir::Value> constraintValsVec(constraintVals.begin(), constraintVals.end());
        return constraintValsVec;
    }

    bool AffineConstraintsHelper::IsExprResolvable(const mlir::AffineExpr& expr)
    {
        // Check if every dim and sym id in the expression has a value in the constraint system
        for (unsigned id = 0; id < _cst.getNumDimAndSymbolIds(); ++id)
        {
            bool isSymId = id >= _cst.getNumDimIds();
            auto idSimple = isSymId ? id - _cst.getNumDimIds() : id;
            bool isFunctionOfId = isSymId ? expr.isFunctionOfSymbol(idSimple) : expr.isFunctionOfDim(idSimple);
            if (isFunctionOfId && !_cst.hasValue(id))
            {
                return false;
            }
        }
        return true;
    }

    mlir::Value AffineConstraintsHelper::ResolveLocalExpr(mlir::OpBuilder& builder, mlir::Location loc, const mlir::AffineExpr& expr, mlir::Value noneFillValue)
    {
        mlir::FlatAffineValueConstraints tmpCst(_cst); // Make a temporary constraint system so we can set values we don't need to null
        for (unsigned id = 0; id < tmpCst.getNumIds(); ++id)
        {
            if (!tmpCst.hasValue(id))
            {
                tmpCst.setValue(id, noneFillValue);
            }
        }
        llvm::SmallVector<mlir::Value, 4> operands;
        tmpCst.getAllValues(&operands);
        // drop the local values from the operands
        operands.pop_back_n(tmpCst.getNumLocalIds());
        auto map = mlir::AffineMap::get(tmpCst.getNumDimIds(), tmpCst.getNumSymbolIds(), expr);
        mlir::fullyComposeAffineMapAndOperands(&map, &operands);
        mlir::canonicalizeMapAndOperands(&map, &operands);
        return builder.create<mlir::AffineApplyOp>(loc, map, operands);
    }

    unsigned AffineConstraintsHelper::GetNumLocals() const
    {
        return _cst.getNumLocalIds();
    }

    void AffineConstraintsHelper::MaybeDebugPrint() const
    {
        if (_debugPrinting)
        {
            _cst.dump();
        }
    }

    void AffineConstraintsHelper::AddBound(const mlir::IntegerPolyhedron::BoundType& type, const IdWrapper& id, mlir::AffineMap map, std::optional<mlir::AffineExpr> localExpr)
    {
        unsigned localsBefore = GetNumLocals();
        [[maybe_unused]] auto addBoundResult = _cst.addBound(type, id.GetFullId(_cst), map);
        assert(mlir::succeeded(addBoundResult));
        unsigned localsAfter = GetNumLocals();
        assert(localsAfter >= localsBefore && "Bad state: can't erase locals by adding bounds");
        if (localsAfter > localsBefore)
        {
            if ((localsAfter - localsBefore) != 1)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "A new bound can only introduce at most 1 local, but the given map added more than 1");
            }
            if (!localExpr.has_value())
            {
                throw InputException(InputExceptionErrors::invalidArgument, "A bound introduced a local, but a local expr was not provided");
            }
            _localExprs.push(*localExpr);
        }
        MaybeDebugPrint();
    }

} // namespace util
} // namespace accera::ir