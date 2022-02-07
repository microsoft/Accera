////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <variant>

namespace accera
{
namespace value
{
    namespace targets
    {
        //  <summary> A struct encapsulating x, y, z indices for a GPU processor </summary>
        struct Dim3
        {
            /// <summary> The x index </summary>
            int64_t x;
            /// <summary> The y index </summary>
            int64_t y;
            /// <summary> The z index </summary>
            int64_t z;

            Dim3(int64_t x_ = 1, int64_t y_ = 1, int64_t z_ = 1) :
                x(x_), y(y_), z(z_) {}
        };

        /// <summary> The CPU execution options </summary>
        struct CPU
        {};

        /// <summary> The GPU execution options </summary>
        struct GPU
        {
            /// <summary> Indicates the grid </summary>
            Dim3 grid;

            /// <summary> Indicates the block </summary>
            Dim3 block;

            GPU(Dim3 grid_ = Dim3(1, 1, 1), Dim3 block_ = Dim3(1, 1, 1)) :
                grid(grid_), block(block_){};
        };

        using Target = std::variant<CPU, GPU>;
        enum class Runtime : int
        {
            Default,
            Vulkan,
            Rocm,
            CUDA
        };

    } // namespace targets

    using ExecutionTarget = targets::Target;
    using ExecutionRuntime = targets::Runtime;
} // namespace value
} // namespace accera