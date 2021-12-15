////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <iterator>
#include <set>
#include <type_traits>

namespace accera::ir
{
namespace util
{
    template <typename Range>
    bool IsZeroBasedIntRangePermutation(const Range& range)
    {
        size_t count = 0;

        std::set<typename Range::value_type> seen;
        for (auto i : range)
        {
            seen.insert(i);
            ++count;
        }

        // ensures uniqueness
        if (seen.size() != count)
        {
            return false;
        }

        auto [min_el, max_el] = std::minmax_element(begin(range), end(range));
        return *min_el == 0 && *max_el == static_cast<typename Range::value_type>(count - 1);
    }

    template <typename T>
    struct IntIterator
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = T;
        using pointer = T;
        using reference = T;

        IntIterator(T value) :
            _value(value) {}

        IntIterator operator++(int) /* postfix */ { return _value++; }
        IntIterator& operator++() /* prefix */
        {
            ++_value;
            return *this;
        }
        reference operator*() const { return _value; }
        pointer operator->() const { return _value; }
        IntIterator operator+(difference_type v) const { return _value + v; }
        bool operator==(const IntIterator& rhs) const { return _value == rhs._value; }
        bool operator!=(const IntIterator& rhs) const { return _value != rhs._value; }
        T _value;
    };

    template <typename T>
    struct SizeRange
    {
        T size;
    };

    template <typename T>
    IntIterator<T> begin(const SizeRange<T>& r)
    {
        return 0;
    }

    template <typename T>
    IntIterator<T> end(const SizeRange<T>& r)
    {
        return r.size;
    }

    template <typename T>
    SizeRange<T> int_range(T size)
    {
        static_assert(std::is_integral<T>());
        return { size };
    }

} // namespace util
} // namespace accera::ir
