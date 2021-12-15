////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraParameters.h"

namespace accera::parameter
{

namespace detail
{
    /*extern*/ llvm::cl::OptionCategory CommonCategory("Accera Common Options");
    /*extern*/ llvm::cl::OptionCategory CustomCategory("Accera Custom Generator Options");

    SupportsRequiredSize::SupportsRequiredSize() :
        _requiredSize(0) {}

    void SupportsRequiredSize::SetRequiredSize(size_t size)
    {
        _requiredSize = size;
    }

    size_t SupportsRequiredSize::GetRequiredSize() const
    {
        return _requiredSize;
    }

    bool SupportsRequiredSize::IsRequiredSizeSet() const
    {
        return _requiredSize != 0;
    }

    bool SupportsRequiredSize::IsValidSize(size_t size) const
    {
        return !IsRequiredSizeSet() || (_requiredSize == size);
    }

    SupportsRequiredElementSize::SupportsRequiredElementSize() :
        _requiredElementSize(0) {}

    void SupportsRequiredElementSize::SetRequiredElementSize(size_t size)
    {
        _requiredElementSize = size;
    }

    size_t SupportsRequiredElementSize::GetRequiredElementSize() const
    {
        return _requiredElementSize;
    }

    bool SupportsRequiredElementSize::IsRequiredElementSizeSet() const
    {
        return _requiredElementSize != 0;
    }

    bool SupportsRequiredElementSize::IsValidElementSize(size_t size) const
    {
        return !IsRequiredElementSizeSet() || (_requiredElementSize == size);
    }

    StringListParser::StringListParser(llvm::cl::Option& O, const char* delimiter) :
        llvm::cl::parser<std::vector<std::string>>(O),
        Delimiter(delimiter)
    {}

    bool StringListParser::parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<std::string>& V)
    {
        // Returns true on parsing error
        llvm::SmallVector<llvm::StringRef, 8> valueStrs;
        llvm::StringRef(Arg).split(valueStrs, Delimiter, -1, false);
        for (auto& valueStr : valueStrs)
        {
            V.push_back(std::string(valueStr));
        }
        if (!IsValidSize(V.size()))
        {
            return O.error("Must have exactly " + std::to_string(GetRequiredSize()) + " values");
        }
        return false;
    }

    StringList::StringList() :
        ParameterTypeBase<std::vector<std::string>>("list<string>", "S1,S2,S3,...")
    {}

    void StringList::operator=(ValueType val)
    {
        ParameterTypeBase<ValueType>::operator=(val);
    }

    std::string StringList::ValueToString()
    {
        std::string interleavedCommas;
        llvm::raw_string_ostream accumulationStream(interleavedCommas);
        std::vector<std::string> valueStrs;
        llvm::interleaveComma(this->value, accumulationStream);
        return "[" + interleavedCommas + "]";
    }

    String::String() :
        ParameterTypeBase<std::string>("string", "string") {}

    void String::operator=(ValueType val)
    {
        ParameterTypeBase<ValueType>::operator=(val);
    }

    std::string String::ValueToString()
    {
        return this->value;
    }

    Boolean::Boolean() :
        ParameterTypeBase<bool>("bool", "bool")
    {}

    void Boolean::operator=(ValueType val)
    {
        ParameterTypeBase<ValueType>::operator=(val);
    }

    std::string Boolean::ValueToString()
    {
        return this->value ? "true" : "false";
    }

} // namespace detail
void ParseAcceraCommandLineOptions(int argc, const char** argv)
{
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0], true);

    // Hide non-Accera parameters inherited from LLVM due to using llvm::cl
    llvm::cl::HideUnrelatedOptions(detail::AcceraCategories);
    llvm::cl::ParseCommandLineOptions(argc, argv, "Accera Generator\n");
}

DomainParameter::DomainParameter() :
    CommonParameter<detail::IntegerList<int64_t>>("domain")
{}
DomainListParameter::DomainListParameter() :
    CommonParameter<detail::IntegerListList<int64_t>>("domains")
{}
LibraryNameParameter::LibraryNameParameter() :
    CommonParameter<detail::String>("library-name")
{}

Size::Size(size_t sizeValue) :
    value(sizeValue)
{}

InnerSize::InnerSize(size_t sizeValue) :
    value(sizeValue)
{}

} // namespace accera::parameter
