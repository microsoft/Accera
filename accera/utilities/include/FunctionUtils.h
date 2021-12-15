////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <functional>
#include <tuple>

namespace accera
{
namespace utilities
{

    /// <summary>
    /// InOrderFunctionEvaluator() is a template function that evaluates a number of zero-argument functions in order.
    /// Usage:
    ///
    /// InOrderFunctionEvaluator(f1, f2, f3);
    ///
    /// The above is equivalent to:
    ///
    /// f1(); f2(); f3()
    ///
    /// </summary>

    /// <summary> Recursive base case with zero functions. Does nothing. </summary>
    inline void InOrderFunctionEvaluator() {}

    /// <summary> Invokes a series of zero-argument functions. </summary>
    ///
    /// <param name="function"> The first function to evaluate </param>
    /// <param name="functions"> The rest of the functions to evaluate </param>
    template <typename Function, typename... Functions>
    void InOrderFunctionEvaluator(Function&& function, Functions&&... functions);

    /// <summary>
    /// ApplyToEach() is a template function that applies a single-argument function to each
    /// of a number of arguments.
    /// Usage:
    ///
    /// ApplyToEach(f, arg1, arg2, arg3);
    ///
    /// The above is equivalent to:
    ///
    /// f(arg1); f(arg2); f(arg3);
    ///
    /// </summary>

    /// <summary> Recursive base case with zero arguments. Does nothing. </summary>
    template <typename FunctionType>
    inline void ApplyToEach(FunctionType&& /*function*/)
    {
    }

    /// <summary> Applies a single-argument function to each of a number of arguments. </summary>
    ///
    /// <param name="function"> The function to apply </param>
    /// <param name="arg"> The first argument to apply the function to </param>
    /// <param name="args"> The rest of the arguments to apply the function to </param>
    template <typename FunctionType, typename Arg, typename... Args>
    void ApplyToEach(FunctionType&& function, Arg&& arg, Args&&... args);

    namespace detail
    {
        template <typename FunctionType, typename Tuple, size_t... I>
        void ApplyToEach(FunctionType&& function, Tuple&& tuple, std::index_sequence<I...>)
        {
            (function(std::get<I>(tuple)), ...);
        }
    } // namespace detail

    template <typename FunctionType, typename... Args>
    void ApplyToEach(FunctionType&& function, std::tuple<Args...>& tuple)
    {
        detail::ApplyToEach(
            std::forward<FunctionType>(function),
            std::forward<std::tuple<Args...>>(tuple),
            std::make_index_sequence<sizeof...(Args)>());
    }

    //
    // FunctionTraits
    //

    /// <summary> FunctionTraits: A type-traits-like way to get the return type and argument types of a function </summary>
    ///
    template <typename T>
    struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> { }; // generic base template

    // Function pointers
    template <typename ReturnT, typename... Args>
    struct FunctionTraits<ReturnT(Args...)>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    template <typename ReturnT, typename... Args>
    struct FunctionTraits<ReturnT (*)(Args...)>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    // std::function
    template <typename ReturnT, typename... Args>
    struct FunctionTraits<std::function<ReturnT(Args...)>>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    // const std::function
    template <typename ReturnT, typename... Args>
    struct FunctionTraits<const std::function<ReturnT(Args...)>>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    // class member
    template <typename ReturnT, typename Class, typename... Args>
    struct FunctionTraits<ReturnT (Class::*)(Args...)>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    template <typename ReturnT, typename Class, typename... Args>
    struct FunctionTraits<ReturnT (Class::*)(Args...)&>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    template <typename ReturnT, typename Class, typename... Args>
    struct FunctionTraits<ReturnT (Class::*)(Args...) const>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    template <typename ReturnT, typename Class, typename... Args>
    struct FunctionTraits<ReturnT (Class::*)(Args...) const&>
    {
        using Type = ReturnT(Args...);
        using ReturnType = ReturnT;
        using ArgTypes = std::tuple<Args...>;
        static constexpr size_t NumArgs = typename std::tuple_size<ArgTypes>();
    };

    // Handy type aliases
    template <typename F>
    using FunctionType = typename FunctionTraits<F>::Type;

    template <typename FunctionType>
    using FunctionReturnType = typename FunctionTraits<FunctionType>::ReturnType;

    template <typename FunctionType>
    constexpr bool HasReturnValue()
    {
        return !std::is_same_v<void, FunctionReturnType<FunctionType>>;
    }

    template <typename FunctionType>
    using FunctionArgTypes = typename FunctionTraits<FunctionType>::ArgTypes;

    /// <summary> Returns a default-constructed tuple of types the given function expects as arguments </summary>
    template <typename FunctionType>
    FunctionArgTypes<FunctionType> GetFunctionArgTuple(FunctionType& function);
} // namespace utilities
} // namespace accera

#pragma region implementation

namespace accera
{
namespace utilities
{
    template <typename Function, typename... Functions>
    void InOrderFunctionEvaluator(Function&& function, Functions&&... functions)
    {
        function();
        InOrderFunctionEvaluator(std::forward<Functions>(functions)...);
    }

    template <typename FunctionType, typename ArgType, typename... ArgTypes>
    void ApplyToEach(FunctionType&& function, ArgType&& arg, ArgTypes&&... args)
    {
        function(std::forward<ArgType>(arg));
        ApplyToEach(std::forward<FunctionType>(function), std::forward<ArgTypes>(args)...);
    }

    template <size_t Index>
    struct IndexTag
    {
        static constexpr size_t index = Index;
    };

    template <size_t Index>
    constexpr size_t GetTagIndex(IndexTag<Index> tag)
    {
        return Index;
    }
} // namespace utilities
} // namespace accera

#pragma endregion implementation
