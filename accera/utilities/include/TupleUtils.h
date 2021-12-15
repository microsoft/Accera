////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tuple>
#include <utility> // for integer_sequence
#include <vector>

namespace accera
{
namespace utilities
{
    //
    // Extracting the tail of a tuple
    //

    template <typename T>
    struct TupleTailImpl; // undefined

    template <typename FirstType, typename... RestTypes>
    struct TupleTailImpl<std::tuple<FirstType, RestTypes...>>
    {
        typedef std::tuple<RestTypes...> type;
    };

    template <typename TupleType>
    using TupleTailType = typename TupleTailImpl<TupleType>::type;

    //
    // General "wrap tuple types" mechanism
    //
    template <template <typename> class WrapperType, typename... Types>
    struct TupleOfWrappedElements
    {
        using type = std::tuple<WrapperType<Types>...>;
    };

    template <typename TupleType, template <typename> class WrapperType, size_t... Sequence>
    static auto MakeWrappedTupleHelper(const TupleType& tuple, std::index_sequence<Sequence...>)
    {
        // STYLE discrepancy
        // fails if Wrapper<T> has no copy constructor
        return typename TupleOfWrappedElements<WrapperType, typename std::tuple_element<Sequence, TupleType>::type...>::type{};
    }

    template <typename TupleType, template <typename> class WrapperType>
    static auto MakeWrappedTuple(const TupleType& tuple)
    {
        // STYLE discrepancy
        // Note: fails if Wrapper<T> has no copy constructor
        return MakeWrappedTupleHelper<TupleType, WrapperType>(tuple, std::make_index_sequence<std::tuple_size<TupleType>::value>());
    }

    template <typename TupleType, template <typename> class WrapperType>
    struct TupleTypeWrapper
    {
        using type = decltype(MakeWrappedTuple<TupleType, WrapperType>(TupleType{}));
    };

    template <typename TupleType, template <typename> class WrapperType>
    using WrappedTuple = typename TupleTypeWrapper<TupleType, WrapperType>::type;

    //
    // Unwrapping tuples
    //

    template <typename WrappedType, template <typename> class WrapperType>
    auto UnwrapType(WrapperType<WrappedType>* x)
    {
        // STYLE discrepancy
        return WrappedType{};
    }

    template <typename WrappedType, template <typename> class WrapperType>
    auto UnwrapType(WrapperType<WrappedType> x)
    {
        // STYLE discrepancy
        return WrappedType{};
    }

    template <typename... WrappedTypes>
    auto UnwrapTuple(const std::tuple<WrappedTypes...>& elements)
    {
        // STYLE discrepancy
        return std::tuple<decltype(UnwrapType(WrappedTypes{}))...>{};
    }

    template <typename WrappedTupleType>
    struct UnwrappedTuple
    {
        using type = decltype(UnwrapTuple(WrappedTupleType{}));
    };

    template <typename WrappedTupleType>
    using UnwrappedTupleType = typename UnwrappedTuple<WrappedTupleType>::type;

    namespace detail
    {
        template <typename T, typename Fn, typename... Types, size_t... I>
        std::vector<T> TupleToVector(Fn&& fn, std::tuple<Types...> tuple, std::index_sequence<I...>)
        {
            std::vector<T> v;
            v.reserve(sizeof...(Types));
            (v.push_back(fn(std::get<I>(tuple))), ...);
            return v;
        }
    } // namespace detail

    template <typename T, typename Fn, typename... Types>
    std::vector<T> TupleToVector(Fn&& fn, std::tuple<Types...> tuple)
    {
        static_assert(
            std::conjunction_v<std::is_invocable_r<T, Fn, Types>...>,
            "Tranformation function Fn does not accept all types within tuple");
        return detail::TupleToVector<T>(
            std::forward<Fn>(fn),
            std::forward<std::tuple<Types...>>(tuple),
            std::make_index_sequence<sizeof...(Types)>());
    }

    namespace detail
    {
        // Helper structs to repeat a single type in a tuple N times
        // e.g. RepeatTuple<int, 5> is the type std::tuple<int, int, int, int, int>
        template <typename RepeatedT, std::size_t Index>
        using Repeated = RepeatedT;

        template <typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
        struct RepeatTupleImpl;

        template <typename T, std::size_t N, std::size_t... Indices>
        struct RepeatTupleImpl<T, N, std::index_sequence<Indices...>>
        {
            using type = std::tuple<Repeated<T, Indices>...>;
            using value_type = T;
        };
    } // namespace detail

    template <typename T, std::size_t N>
    using RepeatTuple = typename detail::RepeatTupleImpl<T, N>::type;

    namespace detail
    {
        template <typename... Args, typename T, size_t... I>
        std::tuple<Args...> VectorToTuple(const std::vector<T>& t, std::index_sequence<I...>)
        {
            return { t[I]... };
        }

        template <size_t N, typename T, size_t... I>
        RepeatTuple<T, N> VectorToRepeatTuple(const std::vector<T>& t, std::index_sequence<I...>)
        {
            return { t[I]... };
        }
    } // namespace detail

    /// <summary> Converts a vector to a specified tuple of Args... types </summary>
    template <typename... Args, typename T>
    std::tuple<Args...> VectorToTuple(const std::vector<T>& t)
    {
        return detail::VectorToTuple<Args...>(t, std::make_index_sequence<sizeof...(Args)>());
    }

    /// <summary> Converts a vector to a specified tuple of size N </summary>
    template <size_t N, typename T>
    RepeatTuple<T, N> VectorToTuple(const std::vector<T>& t)
    {
        return detail::VectorToRepeatTuple<N>(t, std::make_index_sequence<N>());
    }

} // namespace utilities
} // namespace accera
