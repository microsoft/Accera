////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "EmitterContext.h"

#include <utilities/include/MemoryLayout.h>

namespace accera
{
namespace value
{
    /// <summary> A View type that wraps a Value instance that points to Value data instances </summary>
    class Pointer
    {
    public:
        Pointer();

        /// <summary> Constructor that wraps the provided instance of Value </summary>
        /// <param name="value"> The Value instance to wrap </param>
        /// <param name="name"> An optional name for the emitted construct </param>
        /// <remarks> "value" is the pointer, not the data that it points </remarks>
        Pointer(Value value, const std::string& name = "");

        Pointer(const Pointer&);
        Pointer(Pointer&&) noexcept;
        ~Pointer();

        Value& operator*();

        /// <summary> Gets the underlying wrapped Value (pointer) instance </summary>
        Value GetValue() const;

        /// <summary> Returns the memory layout of the data </summary>
        utilities::MemoryLayout GetDataLayout() const;

        /// <summary> Retrieves the type of the data </summary>
        ValueType GetDataType() const;

        /// <summary> Stores data at this pointer instance </summary>
        void Store(ViewAdapter data);

        void SetName(const std::string& name);
        std::string GetName() const;

    private:
        Value _value;
    };
} // namespace value
} // namespace accera