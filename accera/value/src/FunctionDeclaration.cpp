////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FunctionDeclaration.h"

#include <utilities/include/Hash.h>

#include <cctype>

namespace accera
{
namespace value
{
    using namespace accera::utilities;

    FunctionDeclaration::FunctionDeclaration(std::string name) :
        _originalFunctionName(name),
        _isEmpty(false)
    {
        if (!std::isalpha(_originalFunctionName[0]) && _originalFunctionName[0] != '_')
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Function names must begin with an _ or alphabetical character");
        }
    }

    FunctionDeclaration& FunctionDeclaration::DefineFromFile(std::string file)
    {
        CheckNonEmpty();

        _importedSource = file;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Returns(ViewAdapter returnType)
    {
        CheckNonEmpty();

        _returnType = returnType;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Parameters(std::vector<ViewAdapter> paramTypes, std::optional<std::vector<FunctionParameterUsage>> paramUsages)
    {
        CheckNonEmpty();
        if (paramUsages.has_value())
        {
            if(paramUsages->size() != paramTypes.size())
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Parameter usages, if specified, must match the number of parameter types");
            }
            _paramUsages.assign(paramUsages->begin(), paramUsages->end());
        }
        else
        {
            // assume input/output if not specified
            std::fill_n(std::back_inserter(_paramUsages), paramTypes.size(), FunctionParameterUsage::inputOutput);
        }
        _paramTypes.assign(paramTypes.begin(), paramTypes.end());

        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Decorated(bool shouldDecorate)
    {
        CheckNonEmpty();

        _isDecorated = shouldDecorate;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Public(bool isPublic)
    {
        _isPublic = isPublic;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Inlined(FunctionInlining shouldInline)
    {
        CheckNonEmpty();

        _inlineState = shouldInline;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Target(ExecutionOptions target)
    {
        CheckNonEmpty();

        _execTarget = target;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::UseMemRefDescriptorArgs(bool useMemRefDescriptorArgs)
    {
        CheckNonEmpty();

        _useMemRefDescriptorArgs = useMemRefDescriptorArgs;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::External(bool isExternal)
    {
        CheckNonEmpty();

        _external = isExternal;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::CWrapper(bool emitCWrapper)
    {
        CheckNonEmpty();

        _emitCWrapper = emitCWrapper;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::HeaderDecl(bool emitHeaderDecl)
    {
        CheckNonEmpty();

        _emitHeaderDecl = emitHeaderDecl;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::RawPointerAPI(bool rawPointerAPI)
    {
        CheckNonEmpty();

        _rawPointerAPI = rawPointerAPI;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::AddTag(const std::string& tag)
    {
        CheckNonEmpty();

        _tags.push_back(tag);
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::BaseName(const std::string& baseName)
    {
        CheckNonEmpty();

        _baseName = baseName;
        return *this;
    }

    std::optional<Value> FunctionDeclaration::Call(std::vector<ViewAdapter> arguments) const
    {
        CheckNonEmpty();

        if (!_importedSource.empty() && !IsDefined())
        {
            GetContext().ImportCodeFile(_importedSource);
        }

        return GetContext().Call(*this, arguments);
    }

    const std::string& FunctionDeclaration::GetFunctionName() const
    {
        CheckNonEmpty();

        if (_isDecorated)
        {
            if (!_decoratedFunctionName)
            {
                size_t hash = 0;
                if (_returnType)
                {
                    HashCombine(hash, static_cast<int>(_returnType->GetBaseType()));
                    HashCombine(hash, _returnType->PointerLevel());
                    if (_returnType->IsConstrained())
                    {
                        HashCombine(hash, _returnType->GetLayout());
                    }
                }
                for (auto p : _paramTypes)
                {
                    HashCombine(hash, static_cast<int>(p.GetBaseType()));
                    HashCombine(hash, p.PointerLevel());
                    if (p.IsConstrained())
                    {
                        HashCombine(hash, p.GetLayout());
                    }
                }
                _decoratedFunctionName = _originalFunctionName + "_" + std::to_string(hash);
            }
            return *_decoratedFunctionName;
        }
        else
        {
            return _originalFunctionName;
        }
    }

    const std::vector<FunctionParameterUsage>& FunctionDeclaration::GetParameterUsages() const
    {
        CheckNonEmpty();

        return _paramUsages;
    }

    const std::vector<Value>& FunctionDeclaration::GetParameterTypes() const
    {
        CheckNonEmpty();

        return _paramTypes;
    }

    const std::optional<Value>& FunctionDeclaration::GetReturnType() const
    {
        CheckNonEmpty();

        return _returnType;
    }

    bool FunctionDeclaration::IsPublic() const
    {
        CheckNonEmpty();

        return _isPublic;
    }

    bool FunctionDeclaration::IsDefined() const
    {
        CheckNonEmpty();

        return GetContext().IsFunctionDefined(*this);
    }

    bool FunctionDeclaration::IsImported() const
    {
        CheckNonEmpty();

        return !_importedSource.empty();
    }

    bool FunctionDeclaration::IsEmpty() const { return _isEmpty; }

    FunctionInlining FunctionDeclaration::InlineState() const
    {
        CheckNonEmpty();
        return _inlineState;
    }

    void FunctionDeclaration::CheckNonEmpty() const
    {
        if (_isEmpty)
        {
            throw LogicException(LogicExceptionErrors::notInitialized, "FunctionDeclaration is empty");
        }
    }

    FunctionDeclaration DeclareFunction(std::string name)
    {
        return FunctionDeclaration(name);
    }

    /*extern*/ FunctionDeclaration AbsFunctionDeclaration = DeclareFunction("Abs").Decorated(false);
    /*extern*/ FunctionDeclaration CosFunctionDeclaration = DeclareFunction("Cos").Decorated(false);
    /*extern*/ FunctionDeclaration CopySignFunctionDeclaration = DeclareFunction("CopySign").Decorated(false);
    /*extern*/ FunctionDeclaration ExpFunctionDeclaration = DeclareFunction("Exp").Decorated(false);
    /*extern*/ FunctionDeclaration LogFunctionDeclaration = DeclareFunction("Log").Decorated(false);
    /*extern*/ FunctionDeclaration Log10FunctionDeclaration = DeclareFunction("Log10").Decorated(false);
    /*extern*/ FunctionDeclaration Log2FunctionDeclaration = DeclareFunction("Log2").Decorated(false);
    /*extern*/ FunctionDeclaration MaxNumFunctionDeclaration = DeclareFunction("MaxNum").Decorated(false);
    /*extern*/ FunctionDeclaration MinNumFunctionDeclaration = DeclareFunction("MinNum").Decorated(false);
    /*extern*/ FunctionDeclaration PowFunctionDeclaration = DeclareFunction("Pow").Decorated(false);
    /*extern*/ FunctionDeclaration SinFunctionDeclaration = DeclareFunction("Sin").Decorated(false);
    /*extern*/ FunctionDeclaration SqrtFunctionDeclaration = DeclareFunction("Sqrt").Decorated(false);
    /*extern*/ FunctionDeclaration TanhFunctionDeclaration = DeclareFunction("Tanh").Decorated(false);
    /*extern*/ FunctionDeclaration RoundFunctionDeclaration = DeclareFunction("Round").Decorated(false);
    /*extern*/ FunctionDeclaration FloorFunctionDeclaration = DeclareFunction("Floor").Decorated(false);
    /*extern*/ FunctionDeclaration CeilFunctionDeclaration = DeclareFunction("Ceil").Decorated(false);
    /*extern*/ FunctionDeclaration FmaFunctionDeclaration = DeclareFunction("Fma").Decorated(false);

    /*extern*/ FunctionDeclaration MemCopyFunctionDeclaration = DeclareFunction("MemCpy").Decorated(false);
    /*extern*/ FunctionDeclaration MemMoveFunctionDeclaration = DeclareFunction("MemMove").Decorated(false);
    /*extern*/ FunctionDeclaration MemSetFunctionDeclaration = DeclareFunction("MemSet").Decorated(false);

} // namespace value
} // namespace accera

size_t std::hash<::accera::value::FunctionDeclaration>::operator()(const ::accera::value::FunctionDeclaration& value) const
{
    // The function name already encodes the return values and parameter names. And for undecorated functions, we
    // don't want to support overloading
    return std::hash<std::string>{}(value.GetFunctionName());
}
