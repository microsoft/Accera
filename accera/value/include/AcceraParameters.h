////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Signals.h>

#include <utilities/include/Boolean.h>
#include <utilities/include/FunctionUtils.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace accera
{
namespace parameter
{
    /// Usage example
    /*
    // In Accera cpp file
    const size_t Dims = 3;
    static DomainParameter<Dims> Domain;
    static CustomParameter<int> InnerSplitK("innerSplitK", Default(2), MinValue(1), MaxValue(8));
    static CustomParameter<std::vector<int>> InnerSplitsJ("innerSplitsJ");
    static CustomParameter<std::vector<int>> InnerSplitsI("innerSplitsI", Size(2));
    static CustomParameter<std::vector<int>, Occurrences::Required> MyRequiredVecParam("myRequiredVecParam", Size(5));

    const size_t PermutationSize = 6;
    static CustomParameter<Permutation<PermutationSize>> ScheduleOrder("scheduleOrder", Default(Permutation<PermutationSize>::CanonicalPermutation()));

    void AcceraFunction(args...)
    {
        ...
        std::vector<int> domain = Domain.value;
        int innerSplitKValue = InnerSplitK.value;
        std::vector<Index> defaultIndexOrder = { ... }; // default ordering of indices
        std::vector<Index> permutedIndices = ScheduleOrder.parameter.ApplyPermutation(defaultIndexOrder);
        schedule.SetOrder(permutedIndices);
        ...
    }
    ...
    int main(int argc, const char** argv)
    {
        ParseAcceraCommandLineOptions(argc, argv);

        ...
    }

    // Command line usage, supposing the above Accera cpp file is built into generator.exe:
    > path/to/generator.exe --help
    > path/to/generator.exe --domain=256,256,256 --innerSplitK=4 --innerSplitsJ=16,8,4 --innerSplitsI=32,16 --myRequiredVecParam=2,4,8,16,32 --scheduleOrder=3,0,2,1,4,5
    */

    using Occurrences = llvm::cl::NumOccurrencesFlag;

    namespace detail
    {
        extern llvm::cl::OptionCategory CommonCategory;
        extern llvm::cl::OptionCategory CustomCategory;

        static std::vector<const llvm::cl::OptionCategory*> AcceraCategories{
            &detail::CommonCategory,
            &detail::CustomCategory
        };

        // Parser base classes

        class SupportsRequiredSize
        {
        public:
            SupportsRequiredSize();

            void SetRequiredSize(size_t size);
            size_t GetRequiredSize() const;
            bool IsRequiredSizeSet() const;

        protected:
            bool IsValidSize(size_t size) const;

        private:
            size_t _requiredSize;
        };

        class SupportsRequiredElementSize
        {
        public:
            SupportsRequiredElementSize();

            void SetRequiredElementSize(size_t size);
            size_t GetRequiredElementSize() const;
            bool IsRequiredElementSizeSet() const;

        protected:
            bool IsValidElementSize(size_t size) const;

        private:
            size_t _requiredElementSize;
        };

        template <typename DataType>
        class SupportsMinMaxStep
        {
        public:
            SupportsMinMaxStep();

            void SetMaxValue(DataType max);
            void SetMinValue(DataType min);
            void SetStepValue(DataType step);

        protected:
            bool IsErrorValue(llvm::cl::Option& option, DataType val) const;

            DataType _maxValue;
            DataType _minValue;
            DataType _stepValue;
        };

        // Custom parser classes

        class StringListParser : public llvm::cl::parser<std::vector<std::string>>
            , public SupportsRequiredSize
        {
        public:
            const char* Delimiter;
            StringListParser(llvm::cl::Option& O, const char* delimiter = ",");

            bool parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<std::string>& V);
        };

        template <class IntType>
        class IntegerListParser : public llvm::cl::parser<std::vector<IntType>>
            , public SupportsRequiredSize
        {
        public:
            const char* Delimiter = ",";
            IntegerListParser(llvm::cl::Option& O);

            bool parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<IntType>& V);
        };

        // TODO : generalize custom list parser so this becomes something like ListParser<ListParser<IntType>>
        template <class IntType>
        class IntegerListListParser : public llvm::cl::parser<std::vector<std::vector<IntType>>>
            , public SupportsRequiredSize
            , public SupportsRequiredElementSize
        {
        public:
            const char* Delimiter = ":";
            IntegerListListParser(llvm::cl::Option& O);

            bool parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<std::vector<IntType>>& V);
        };

        // Some template specializations of llvm::cl::parser<> in llvm\include\llvm\Support\CommandLine.h for number data types are marked final
        // so wrap those parsers instead of subclassing them
        template <class DataType>
        class NumberParserWrapper : public llvm::cl::basic_parser<DataType>
            , public SupportsMinMaxStep<DataType>
        {
        public:
            using typename llvm::cl::basic_parser<DataType>::parser_data_type;
            using typename llvm::cl::basic_parser<DataType>::OptVal;
            NumberParserWrapper(llvm::cl::Option& O);

            bool parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, DataType& V);

            operator llvm::cl::parser<DataType>() const { return _wrappedParser; }

        private:
            llvm::cl::parser<DataType> _wrappedParser;
        };

        template <size_t Count>
        class PermutationParser : public IntegerListParser<size_t>
        {
        public:
            PermutationParser(llvm::cl::Option& O);

            bool parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<size_t>& V);
            bool CheckPermutation(llvm::cl::Option& option, const std::vector<size_t>& list);
        };

        // Parameter Type classes

        template <typename DataType>
        class ParameterTypeBase
        {
        public:
            DataType value;

            ParameterTypeBase(const std::string& description, const std::string& valueDescription);

            const std::string& Description() const;
            const std::string& ValueDescription() const;
            void SetDescription(const std::string& description);
            void SetValueDescription(const std::string& valueDescription);

            void operator=(const DataType& val);

            virtual std::string ValueToString() = 0;

        private:
            std::string _description;
            std::string _valueDescription;
        };

        template <typename IntType>
        class Integer : public ParameterTypeBase<IntType>
            , public SupportsMinMaxStep<IntType>
        {
        public:
            using ValueType = IntType;
            using ParserType = NumberParserWrapper<IntType>;

            Integer();

            void operator=(ValueType val);

            std::string ValueToString() override;
        };

        class StringList : public ParameterTypeBase<std::vector<std::string>>
            , public SupportsRequiredSize
        {
        public:
            using ValueType = std::vector<std::string>;
            using ParserType = StringListParser;

            StringList();

            void operator=(ValueType val);

            std::string ValueToString() override;
        };

        template <typename IntType>
        class IntegerList : public ParameterTypeBase<std::vector<IntType>>
            , public SupportsRequiredSize
        {
        public:
            using ValueType = std::vector<IntType>;
            using ParserType = IntegerListParser<IntType>;

            IntegerList();

            void operator=(ValueType val);

            std::string ValueToString() override;
        };

        template <size_t Count>
        class Permutation : public IntegerList<size_t>
        {
        public:
            using ValueType = IntegerList<size_t>::ValueType;
            using ParserType = PermutationParser<Count>;

            Permutation();

            void operator=(ValueType val);

            static std::vector<size_t> CanonicalPermutation();

            template <typename ElementType>
            std::vector<ElementType> ApplyPermutation(const std::vector<ElementType>& input);
        };

        // TODO : generalize lists so this becomes something like List<List<IntType>>
        template <typename IntType>
        class IntegerListList : public ParameterTypeBase<std::vector<std::vector<IntType>>>
            , public SupportsRequiredElementSize
            , public SupportsRequiredSize
        {
        public:
            using ValueType = std::vector<std::vector<IntType>>;
            using ParserType = IntegerListListParser<IntType>;

            IntegerListList();

            void operator=(ValueType val);

            std::string ValueToString() override;
        };

        class String : public ParameterTypeBase<std::string>
        {
        public:
            using ValueType = std::string;
            using ParserType = llvm::cl::parser<std::string>;

            String();

            void operator=(ValueType val);
            std::string ValueToString() override;
        };

        class Boolean : public ParameterTypeBase<bool>
        {
        public:
            using ValueType = bool;
            using ParserType = llvm::cl::parser<bool>;

            Boolean();

            void operator=(ValueType val);
            std::string ValueToString() override;
        };
    } // namespace detail

    // Parameter Constraints
    template <typename DataType>
    struct Default
    {
        DataType value;

        Default(const DataType& defaultValue);

        template <typename ParameterType>
        void ApplyToParameter(ParameterType& parameter) const;

        template <typename ParserClass>
        void ApplyToParser(ParserClass& parser) const;
    };

    template <typename DataType>
    struct MaxValue
    {
        DataType value;

        MaxValue(const DataType& maxValue);

        template <typename ParameterType>
        void ApplyToParameter(ParameterType& parameter) const;

        template <typename ParserClass>
        void ApplyToParser(ParserClass& parser) const;
    };

    template <typename DataType>
    struct MinValue
    {
        DataType value;

        MinValue(const DataType& minValue);

        template <typename ParameterType>
        void ApplyToParameter(ParameterType& parameter) const;

        template <typename ParserClass>
        void ApplyToParser(ParserClass& parser) const;
    };

    template <typename DataType>
    struct StepValue
    {
        DataType value;

        StepValue(const DataType& stepValue);

        template <typename ParameterType>
        void ApplyToParameter(ParameterType& parameter) const;

        template <typename ParserClass>
        void ApplyToParser(ParserClass& parser) const;
    };

    // Required list size
    struct Size
    {
        size_t value;

        Size(size_t sizeValue);

        template <typename ParameterType>
        void ApplyToParameter(ParameterType& parameter) const;

        template <typename ParserClass>
        void ApplyToParser(ParserClass& parser) const;
    };

    // Required size for inner lists in a list-of-lists
    struct InnerSize
    {
        size_t value;

        InnerSize(size_t sizeValue);

        template <typename ParameterType>
        void ApplyToParameter(ParameterType& parameter) const;

        template <typename ParserClass>
        void ApplyToParser(ParserClass& parser) const;
    };

    // Parameters
    namespace detail
    {
        template <typename ParameterType, bool StorageFlag, typename ParserClass, typename... Modifiers>
        struct ModifierApplicator
        {
            ModifierApplicator(Modifiers&&... modifiers);

            void apply(llvm::cl::opt<ParameterType, StorageFlag, ParserClass>& O) const;

            std::tuple<Modifiers...> _modifiers;
        };

        template <typename ParameterType>
        class ParameterHolder
        {
        public:
            ParameterType parameter;
            typename ParameterType::ValueType& value;
            template <typename... Modifiers>
            ParameterHolder(Modifiers&&... modifiers);

            operator typename ParameterType::ValueType() { return value; }

        private:
            template <typename Modifier>
            void ApplyModifier(Modifier&& mod);

            template <typename Modifier, typename... Modifiers>
            void ApplyModifiers(Modifier&& mod, Modifiers&&... mods);
        };

        template <typename ParameterType, Occurrences occurrences>
        class BaseParameter : public ParameterHolder<ParameterType>
            , public llvm::cl::opt<ParameterType, true, typename ParameterType::ParserType>
        {
        public:
            template <typename... Modifiers>
            BaseParameter(const char* name, llvm::cl::OptionCategory& category, Modifiers&&... modifiers) :
                ParameterHolder<ParameterType>(std::forward<Modifiers>(modifiers)...),
                llvm::cl::opt<ParameterType, true, typename ParameterType::ParserType>(llvm::cl::desc(this->parameter.Description()),
                                                                                       llvm::cl::value_desc(this->parameter.ValueDescription()),
                                                                                       llvm::cl::location(this->parameter),
                                                                                       llvm::cl::cat(category),
                                                                                       occurrences,
                                                                                       ModifierApplicator<ParameterType, true, typename ParameterType::ParserType, Modifiers...>(std::forward<Modifiers>(modifiers)...))
            {
                llvm::cl::Option::setArgStr(name);
            }

            std::string GetArgName()
            {
                return this->ArgStr.str();
            }
        };

// Use a macro to specialize BaseParameter type mappings since type aliasing with partial specialization is not allowed
#define BASE_PARAMETER_SPECIALIZATION(ParameterAliasType, RealParameterType)                                                                                                                                                         \
    template <Occurrences occurrences>                                                                                                                                                                                               \
    class BaseParameter<ParameterAliasType, occurrences> : public ParameterHolder<RealParameterType>                                                                                                                                 \
        , public llvm::cl::opt<RealParameterType, true, typename RealParameterType::ParserType>                                                                                                                                      \
    {                                                                                                                                                                                                                                \
    public:                                                                                                                                                                                                                          \
        template <typename... Modifiers>                                                                                                                                                                                             \
        BaseParameter(const char* name, llvm::cl::OptionCategory& category, Modifiers&&... modifiers) :                                                                                                                              \
            ParameterHolder<RealParameterType>(std::forward<Modifiers>(modifiers)...),                                                                                                                                               \
            llvm::cl::opt<RealParameterType, true, typename RealParameterType::ParserType>(llvm::cl::desc(this->parameter.Description()),                                                                                            \
                                                                                           llvm::cl::value_desc(this->parameter.ValueDescription()),                                                                                 \
                                                                                           llvm::cl::location(this->parameter),                                                                                                      \
                                                                                           llvm::cl::cat(category),                                                                                                                  \
                                                                                           occurrences,                                                                                                                              \
                                                                                           ModifierApplicator<RealParameterType, true, typename RealParameterType::ParserType, Modifiers...>(std::forward<Modifiers>(modifiers)...)) \
        {                                                                                                                                                                                                                            \
            llvm::cl::Option::setArgStr(name);                                                                                                                                                                                       \
        }                                                                                                                                                                                                                            \
        std::string GetArgName()                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                            \
            return this->ArgStr.str();                                                                                                                                                                                               \
        }                                                                                                                                                                                                                            \
    };

        BASE_PARAMETER_SPECIALIZATION(int, Integer<int>);
        BASE_PARAMETER_SPECIALIZATION(int64_t, Integer<int64_t>);
        BASE_PARAMETER_SPECIALIZATION(std::vector<int>, IntegerList<int>);
        BASE_PARAMETER_SPECIALIZATION(std::vector<int64_t>, IntegerList<int64_t>);
        BASE_PARAMETER_SPECIALIZATION(bool, Boolean);
        BASE_PARAMETER_SPECIALIZATION(std::string, String);
        BASE_PARAMETER_SPECIALIZATION(std::vector<std::string>, StringList);
        BASE_PARAMETER_SPECIALIZATION(std::vector<utilities::Boolean>, Boolean);

#undef BASE_PARAMETER_SPECIALIZATION

    } // namespace detail

    template <typename ParameterType, Occurrences occurrences = Occurrences::Required>
    class CommonParameter : public detail::BaseParameter<ParameterType, occurrences>
    {
    public:
        template <typename... Modifiers>
        CommonParameter(const char* name, Modifiers&&... modifiers) :
            detail::BaseParameter<ParameterType, occurrences>(name, detail::CommonCategory, std::forward<Modifiers>(modifiers)...)
        {}
    };

    class DomainParameter : public CommonParameter<detail::IntegerList<int64_t>>
    {
    public:
        DomainParameter();
    };

    class DomainListParameter : public CommonParameter<detail::IntegerListList<int64_t>>
    {
    public:
        DomainListParameter();
    };

    class LibraryNameParameter : public CommonParameter<detail::String>
    {
    public:
        LibraryNameParameter();
    };

    template <typename ParameterType, Occurrences occurrences = Occurrences::Optional>
    class CustomParameter : public detail::BaseParameter<ParameterType, occurrences>
    {
    public:
        template <typename... Modifiers>
        CustomParameter(const char* name, Modifiers&&... modifiers) :
            detail::BaseParameter<ParameterType, occurrences>(name, detail::CustomCategory, std::forward<Modifiers>(modifiers)...)
        {}
    };

    template <size_t Count>
    using Permutation = detail::Permutation<Count>;

    void ParseAcceraCommandLineOptions(int argc, const char** argv);

} // namespace parameter
} // namespace accera

#pragma region implementation

namespace accera
{
namespace parameter
{
    namespace detail
    {

        template <typename DataType>
        SupportsMinMaxStep<DataType>::SupportsMinMaxStep() :
            _maxValue(std::numeric_limits<DataType>::max()),
            _minValue(std::numeric_limits<DataType>::min()),
            _stepValue(static_cast<DataType>(1))
        {}

        template <typename DataType>
        void SupportsMinMaxStep<DataType>::SetMaxValue(DataType max)
        {
            _maxValue = max;
        }

        template <typename DataType>
        void SupportsMinMaxStep<DataType>::SetMinValue(DataType min)
        {
            _minValue = min;
        }

        template <typename DataType>
        void SupportsMinMaxStep<DataType>::SetStepValue(DataType step)
        {
            _stepValue = step;
        }

        template <typename DataType>
        bool SupportsMinMaxStep<DataType>::IsErrorValue(llvm::cl::Option& option, DataType val) const
        {
            // Returns true on error
            if (val > _maxValue)
            {
                return option.error("Given value is larger than registered maximum: " + std::to_string(_maxValue));
            }
            else if (val < _minValue)
            {
                return option.error("Given value is smaller than registered minimum: " + std::to_string(_minValue));
            }
            else if (val % _stepValue != 0)
            {
                return option.error("Given value is not a multiple of the registered step size: " + std::to_string(_stepValue));
            }
            return false;
        }

        template <typename IntType>
        IntegerListParser<IntType>::IntegerListParser(llvm::cl::Option& O) :
            llvm::cl::parser<std::vector<IntType>>(O)
        {}

        template <typename IntType>
        bool IntegerListParser<IntType>::parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<IntType>& V)
        {
            // Returns true on parsing error
            llvm::SmallVector<llvm::StringRef, 8> valueStrs;
            llvm::StringRef(Arg).split(valueStrs, Delimiter, -1, false);
            for (auto& valueStr : valueStrs)
            {
                auto str = std::string(valueStr);
                V.push_back(static_cast<IntType>(std::stoll(str)));
            }
            if (!IsValidSize(V.size()))
            {
                return O.error("Must have exactly " + std::to_string(GetRequiredSize()) + " values");
            }
            return false;
        }

        template <typename IntType>
        IntegerListListParser<IntType>::IntegerListListParser(llvm::cl::Option& O) :
            llvm::cl::parser<std::vector<std::vector<IntType>>>(O)
        {}

        template <typename IntType>
        bool IntegerListListParser<IntType>::parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<std::vector<IntType>>& V)
        {
            // Returns true on parsing error
            llvm::SmallVector<llvm::StringRef, 8> valueStrs;
            llvm::StringRef(Arg).split(valueStrs, Delimiter, -1, false);
            for (auto& valueStr : valueStrs)
            {
                std::vector<IntType> innerList;
                IntegerListParser<IntType> innerParser(O);
                if (IsRequiredElementSizeSet())
                {
                    innerParser.SetRequiredSize(GetRequiredElementSize());
                }
                bool innerParseError = innerParser.parse(O, ArgName, valueStr, innerList);
                if (innerParseError)
                {
                    return innerParseError;
                }
                V.push_back(innerList);
            }
            if (!IsValidSize(V.size()))
            {
                return O.error("Must have exactly " + std::to_string(GetRequiredSize()) + " values");
            }
            return false;
        }

        template <typename DataType>
        NumberParserWrapper<DataType>::NumberParserWrapper(llvm::cl::Option& O) :
            llvm::cl::basic_parser<DataType>(O),
            _wrappedParser(O)
        {}

        template <typename DataType>
        bool NumberParserWrapper<DataType>::parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, DataType& V)
        {
            // Returns true on parsing error
            bool baseParseResult = _wrappedParser.parse(O, ArgName, Arg, V);
            return baseParseResult || this->IsErrorValue(O, V);
        }

        template <size_t Count>
        PermutationParser<Count>::PermutationParser(llvm::cl::Option& O) :
            IntegerListParser<size_t>(O)
        {
            this->SetRequiredSize(Count);
        }

        template <size_t Count>
        bool PermutationParser<Count>::parse(llvm::cl::Option& O, llvm::StringRef ArgName, llvm::StringRef Arg, std::vector<size_t>& V)
        {
            // Returns true on parsing error
            return IntegerListParser<size_t>::parse(O, ArgName, Arg, V) || CheckPermutation(O, V);
        }

        template <size_t Count>
        bool PermutationParser<Count>::CheckPermutation(llvm::cl::Option& option, const std::vector<size_t>& list)
        {
            // Check that all values are in [0, Count) with no repeats
            // return true on error
            std::vector<size_t> canonicalPermutation = Permutation<Count>::CanonicalPermutation();
            if (list.size() != Count)
            {
                return option.error("Permutation requires " + std::to_string(Count) + " elements");
            }
            else if (!std::is_permutation(list.begin(), list.end(), canonicalPermutation.begin()))
            {
                return option.error("Permutation given is not a valid permutation");
            }
            return false;
        }

        template <typename DataType>
        ParameterTypeBase<DataType>::ParameterTypeBase(const std::string& description, const std::string& valueDescription) :
            _description(description),
            _valueDescription(valueDescription)
        {}

        template <typename DataType>
        const std::string& ParameterTypeBase<DataType>::Description() const
        {
            return _description;
        }

        template <typename DataType>
        const std::string& ParameterTypeBase<DataType>::ValueDescription() const
        {
            return _valueDescription;
        }

        template <typename DataType>
        void ParameterTypeBase<DataType>::SetDescription(const std::string& description)
        {
            _description = description;
        }

        template <typename DataType>
        void ParameterTypeBase<DataType>::SetValueDescription(const std::string& valueDescription)
        {
            _valueDescription = valueDescription;
        }

        template <typename DataType>
        void ParameterTypeBase<DataType>::operator=(const DataType& val)
        {
            value = val;
        }

        template <typename IntType>
        Integer<IntType>::Integer() :
            ParameterTypeBase<IntType>("int", "N")
        {}

        template <typename IntType>
        void Integer<IntType>::operator=(ValueType val)
        {
            ParameterTypeBase<ValueType>::operator=(val);
        }

        template <typename IntType>
        std::string Integer<IntType>::ValueToString()
        {
            return std::to_string(this->value);
        }

        template <typename IntType>
        IntegerList<IntType>::IntegerList() :
            ParameterTypeBase<std::vector<IntType>>("list<int>", "N1,N2,N3,...")
        {}

        template <typename IntType>
        void IntegerList<IntType>::operator=(ValueType val)
        {
            ParameterTypeBase<ValueType>::operator=(val);
        }

        template <typename IntType>
        std::string IntegerList<IntType>::ValueToString()
        {
            std::string interleavedCommas;
            llvm::raw_string_ostream accumulationStream(interleavedCommas);
            std::vector<std::string> valueStrs;
            std::transform(this->value.begin(), this->value.end(), std::back_inserter(valueStrs), [](IntType val) { return std::to_string(val); });
            llvm::interleaveComma(valueStrs, accumulationStream);
            return "[" + interleavedCommas + "]";
        }

        template <size_t Count>
        Permutation<Count>::Permutation()
        {
            this->SetDescription("permutation(" + std::to_string(Count) + ")");
            this->SetValueDescription("0,1,2,...");
        }

        template <size_t Count>
        void Permutation<Count>::operator=(ValueType val)
        {
            IntegerList<size_t>::operator=(val);
        }

        template <size_t Count>
        std::vector<size_t> Permutation<Count>::CanonicalPermutation()
        {
            std::vector<size_t> result(Count);
            std::iota(result.begin(), result.end(), 0);
            return result;
        }

        template <size_t Count>
        template <typename ElementType>
        std::vector<ElementType> Permutation<Count>::ApplyPermutation(const std::vector<ElementType>& input)
        {
            if (this->value.size() == input.size())
            {
                std::vector<ElementType> result;
                result.reserve(input.size());
                for (size_t permuteIdx = 0; permuteIdx < this->value.size(); ++permuteIdx)
                {
                    result.push_back(input[this->value[permuteIdx]]);
                }
                return result;
            }
            return std::vector<ElementType>();
        }

        template <typename IntType>
        IntegerListList<IntType>::IntegerListList() :
            ParameterTypeBase<std::vector<std::vector<IntType>>>("list<list<int>>", "A1,A2,A3,...;B1,B2,B3,...;...")
        {}

        template <typename IntType>
        void IntegerListList<IntType>::operator=(ValueType val)
        {
            ParameterTypeBase<ValueType>::operator=(val);
        }

        template <typename IntType>
        std::string IntegerListList<IntType>::ValueToString()
        {
            std::string interleavedSemicolons;
            llvm::raw_string_ostream accumulationStream(interleavedSemicolons);
            std::vector<std::string> valueStrs;
            std::transform(this->value.begin(), this->value.end(), std::back_inserter(valueStrs), [](std::vector<IntType>& val) {
                IntegerList<IntType> innerList;
                innerList = val;
                return innerList.ValueToString();
            });
            llvm::interleave(valueStrs, accumulationStream, ":");
            return "[" + interleavedSemicolons + "]";
        }

        template <typename ParameterType, bool StorageFlag, typename ParserClass, typename... Modifiers>
        ModifierApplicator<ParameterType, StorageFlag, ParserClass, Modifiers...>::ModifierApplicator(Modifiers&&... modifiers) :
            _modifiers(std::forward_as_tuple(modifiers...))
        {}

        template <typename ParameterType, bool StorageFlag, typename ParserClass, typename... Modifiers>
        void ModifierApplicator<ParameterType, StorageFlag, ParserClass, Modifiers...>::apply(llvm::cl::opt<ParameterType, StorageFlag, ParserClass>& O) const
        {
            ParserClass& parser = O.getParser();
            std::apply([&](const auto&... modifiers) { ((modifiers.ApplyToParser(parser)), ...); }, _modifiers);
        }

        template <typename ParameterType>
        template <typename... Modifiers>
        ParameterHolder<ParameterType>::ParameterHolder(Modifiers&&... modifiers) :
            value(parameter.value)
        {
            if constexpr (sizeof...(modifiers) > 0)
            {
                ApplyModifiers(modifiers...);
            }
        }

        template <typename ParameterType>
        template <typename Modifier>
        void ParameterHolder<ParameterType>::ApplyModifier(Modifier&& mod)
        {
            mod.ApplyToParameter(parameter);
        }

        template <typename ParameterType>
        template <typename Modifier, typename... Modifiers>
        void ParameterHolder<ParameterType>::ApplyModifiers(Modifier&& mod, Modifiers&&... mods)
        {
            ApplyModifier(mod);

            if constexpr (sizeof...(mods) > 0)
            {
                ApplyModifiers(mods...);
            }
        }
    } // namespace detail

    // Parameter Constraints
    template <typename DataType>
    Default<DataType>::Default(const DataType& defaultValue) :
        value(defaultValue)
    {}

    template <typename DataType>
    template <typename ParameterType>
    void Default<DataType>::ApplyToParameter(ParameterType& parameter) const
    {
        static_assert(std::is_same<typename ParameterType::ValueType, DataType>::value, "Default value type must be the same as the parameter type");
        parameter.value = value;
        parameter.SetDescription(parameter.Description() + " default(" + parameter.ValueToString() + ")");
    }

    template <typename DataType>
    template <typename ParserClass>
    void Default<DataType>::ApplyToParser(ParserClass& parser) const
    {}

    template <typename DataType>
    MaxValue<DataType>::MaxValue(const DataType& maxValue) :
        value(maxValue)
    {}

    template <typename DataType>
    template <typename ParameterType>
    void MaxValue<DataType>::ApplyToParameter(ParameterType& parameter) const
    {
        static_assert(std::is_same<typename ParameterType::ValueType, DataType>::value, "Max value type must be the same as the parameter type");
        parameter.SetDescription(parameter.Description() + " max(" + std::to_string(value) + ")");
    }

    template <typename DataType>
    template <typename ParserClass>
    void MaxValue<DataType>::ApplyToParser(ParserClass& parser) const
    {
        auto castParser = static_cast<detail::SupportsMinMaxStep<DataType>*>(&parser);
        castParser->SetMaxValue(value);
    }

    template <typename DataType>
    MinValue<DataType>::MinValue(const DataType& minValue) :
        value(minValue)
    {}

    template <typename DataType>
    template <typename ParameterType>
    void MinValue<DataType>::ApplyToParameter(ParameterType& parameter) const
    {
        static_assert(std::is_same<typename ParameterType::ValueType, DataType>::value, "Min value type must be the same as the parameter type");
        parameter.SetDescription(parameter.Description() + " min(" + std::to_string(value) + ")");
    }

    template <typename DataType>
    template <typename ParserClass>
    void MinValue<DataType>::ApplyToParser(ParserClass& parser) const
    {
        auto castParser = static_cast<detail::SupportsMinMaxStep<DataType>*>(&parser);
        castParser->SetMinValue(value);
    }

    template <typename DataType>
    StepValue<DataType>::StepValue(const DataType& stepValue) :
        value(stepValue)
    {}

    template <typename DataType>
    template <typename ParameterType>
    void StepValue<DataType>::ApplyToParameter(ParameterType& parameter) const
    {
        static_assert(std::is_same<typename ParameterType::ValueType, DataType>::value, "Step value type must be the same as the parameter type");
        parameter.SetDescription(parameter.Description() + " step(" + std::to_string(value) + ")");
    }

    template <typename DataType>
    template <typename ParserClass>
    void StepValue<DataType>::ApplyToParser(ParserClass& parser) const
    {
        auto castParser = static_cast<detail::SupportsMinMaxStep<DataType>*>(&parser);
        castParser->SetStepValue(value);
    }

    template <typename ParameterType>
    void Size::ApplyToParameter(ParameterType& parameter) const
    {
        parameter.SetDescription(parameter.Description() + " size(" + std::to_string(value) + ")");
    }

    template <typename ParserClass>
    void Size::ApplyToParser(ParserClass& parser) const
    {
        auto castParser = static_cast<detail::SupportsRequiredSize*>(&parser);
        castParser->SetRequiredSize(value);
    }

    template <typename ParameterType>
    void InnerSize::ApplyToParameter(ParameterType& parameter) const
    {
        parameter.SetDescription(parameter.Description() + " innersize(" + std::to_string(value) + ")");
    }

    template <typename ParserClass>
    void InnerSize::ApplyToParser(ParserClass& parser) const
    {
        auto castParser = static_cast<detail::SupportsRequiredElementSize*>(&parser);
        castParser->SetRequiredElementSize(value);
    }

} // namespace parameter
} // namespace accera

#pragma endregion implementation
