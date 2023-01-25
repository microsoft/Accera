####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import copy
import cpuinfo
import re
from typing import List, Union
from dataclasses import dataclass, field, fields
from enum import Enum, auto

from ._lang_python import ScalarType, _GetKnownDeviceNames
from ._lang_python._lang import (
    BLOCK_X, BLOCK_Y, BLOCK_Z, THREAD_X, THREAD_Y, THREAD_Z, WARP_X, WARP_Y, _MemorySpace, _MMAShape, _ExecutionRuntime as Runtime
)


class Category(Enum):
    CPU = auto()
    GPU = auto()


class Architecture(Enum):
    HOST = auto()
    ARM = auto()
    X86 = auto()
    X86_64 = auto()
    AARCH64 = auto()


# Branding is currently unused
KNOWN_CPUS_HEADER = \
    ["Model", "Family", "Branding", "Base Freq (GHz)", "Turbo Freq (GHz)", "Cores", "Threads", "Cache Lines", "Cache Sizes (KB)", "Vector Bytes", "Vector Registers", "Extensions", "ISA", "Runtime"]

# yapf: disable
# NOTE: When updating this table, please update docs/Reference/classes/Target/Model.md by following the instructions in that file
KNOWN_CPUS = [

    # Intel Skylake
    # ref: https://en.wikipedia.org/wiki/Skylake_(microarchitecture)
    # Mainstream desktop processors
    ["Intel 6700K", "Skylake", "Core i7", 4.0, {1: 4.2, 2: 4.0, 4: 4.0}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6785R", "Skylake", "Core i7", 3.3, {1: 3.9, 2: 3.8, 4: 3.5}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"], # Has 128 MB L4, unaccounted/untested
    ["Intel 6700",  "Skylake", "Core i7", 3.4, {1: 4.0, 2: 3.9, 4: 3.7}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6700T", "Skylake", "Core i7", 2.8, {1: 3.6, 2: 3.5, 4: 3.4}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # ref: https://www.cpu-world.com/CPUs/Core_i7/Intel-Core%20i7-6820HQ%20Mobile%20processor.html
    ["Intel 6820HQ", "Skylake", "Core i7", 2.7, {1: 3.6, 2: 3.4, 4: 3.2}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP" ],

    ["Intel 6600K", "Skylake", "Core i5", 3.5, {1: 3.9, 2: 3.8, 4: 3.6}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6685R", "Skylake", "Core i5", 3.2, {1: 3.8, 2: 3.7, 4: 3.3}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6600",  "Skylake", "Core i5", 3.3, {1: 3.9, 2: 3.8, 4: 3.6}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6585R", "Skylake", "Core i5", 2.8, {1: 3.6, 2: 3.5, 4: 3.1}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6500",  "Skylake", "Core i5", 3.2, {1: 3.6, 2: 3.5, 4: 3.3}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6600T", "Skylake", "Core i5", 2.7, {1: 3.5, 2: 3.4, 4: 3.3}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6500T", "Skylake", "Core i5", 2.5, {1: 3.1, 2: 3.0, 4: 2.8}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6402P", "Skylake", "Core i5", 2.8, {1: 3.4, 2: 3.4, 4: 3.2}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6400T", "Skylake", "Core i5", 2.2, {1: 2.8, 2: 2.7, 4: 2.5}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6400",  "Skylake", "Core i5", 2.7, {1: 3.3, 2: 3.3, 4: 3.1}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 6320",  "Skylake", "Core i3", 3.9,                       {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6300",  "Skylake", "Core i3", 3.8,                       {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6100",  "Skylake", "Core i3", 3.7,                       {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6300T", "Skylake", "Core i3", 3.3,                       {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6100T", "Skylake", "Core i3", 3.2,                       {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6098P", "Skylake", "Core i3", 3.6,                       {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel G4520",   "Skylake", "Pentium", 3.6,                       {}, 2, 2, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G4500",   "Skylake", "Pentium", 3.5,                       {}, 2, 2, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G4500T",  "Skylake", "Pentium", 3.0,                       {}, 2, 2, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G4400",   "Skylake", "Pentium", 3.3,                       {}, 2, 2, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G4400T",  "Skylake", "Pentium", 2.9,                       {}, 2, 2, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G4400TE", "Skylake", "Pentium", 2.4,                       {}, 2, 2, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],

    ["Intel G3920",   "Skylake", "Celeron", 2.9,                       {}, 2, 2, [32, 256, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G3900",   "Skylake", "Celeron", 2.8,                       {}, 2, 2, [32, 256, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G3900TE", "Skylake", "Celeron", 2.3,                       {}, 2, 2, [32, 256, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],
    ["Intel G3900T",  "Skylake", "Celeron", 2.6,                       {}, 2, 2, [32, 256, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2"], "X86_64", "OPENMP"],

    # High-end desktop processors (Skylake-X)
    # 7th generation Skylake-X high-end desktop CPUs
    ["Intel 7980XE", "Skylake-X", "Core i9", 2.6, {2: 4.2, 1: 4.4}, 18, 36, [32, 1024, 1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 7960X",  "Skylake-X", "Core i9", 2.8, {2: 4.2, 1: 4.4}, 16, 32, [32, 1024, 1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 7940X",  "Skylake-X", "Core i9", 3.1, {2: 4.3, 1: 4.4}, 14, 28, [32, 1024, 1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 7920X",  "Skylake-X", "Core i9", 2.9, {2: 4.3, 1: 4.4}, 12, 24, [32, 1024, 1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 7900X",  "Skylake-X", "Core i9", 3.3, {2: 4.3, 1: 4.5}, 10, 20, [32, 1024, 1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    ["Intel 7820X",  "Skylake-X", "Core i7", 3.6, {2: 4.3, 1: 4.5},  8, 26, [32,  1024,  1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 7800X",  "Skylake-X", "Core i7", 3.5,        {1: 4.0},  6, 12, [32,  1024,  1408], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # 9th generation Skylake-X high-end desktop CPUs
    ["Intel 9990XE", "Skylake-X", "Core i9", 4.0, {2: 5.0, 1: 5.0}, 14, 28, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 9980XE", "Skylake-X", "Core i9", 3.0, {2: 4.4, 1: 4.5}, 18, 36, [32, 1024, 24.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 9960X",  "Skylake-X", "Core i9", 3.1, {2: 4.4, 1: 4.5}, 16, 32, [32, 1024, 22.00 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 9940X",  "Skylake-X", "Core i9", 3.3, {2: 4.4, 1: 4.5}, 14, 32, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 9920X",  "Skylake-X", "Core i9", 3.5, {2: 4.4, 1: 4.5}, 12, 24, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 9900X",  "Skylake-X", "Core i9", 3.5, {2: 4.4, 1: 4.5}, 10, 20, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 9820X",  "Skylake-X", "Core i9", 3.3, {2: 4.1, 1: 4.2}, 10, 20, [32, 1024, 16.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    ["Intel 9800X",  "Skylake-X", "Core i7", 3.8, {2: 4.4, 1: 4.5},  8, 16, [32, 1024, 16.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # Xeon High-end desktop processors (Skylake-X)
    ["Intel W-3175X", "Skylake-X",   "Xeon", 3.1, {2: 3.8, 1: 4.3}, 28, 56, [32, 1024, 38.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # Xeon desktop processors (Skylake-W)
    # ref: https://en.wikipedia.org/wiki/List_of_Intel_Skylake-based_Xeon_microprocessors#%22Skylake-W%22_(14_nm)
    ["Intel W-2102", "Skylake-W", "Xeon", 2.9, 0.0, 4,  4,  [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2104", "Skylake-W", "Xeon", 3.2, 0.0, 4,  4,  [32, 1024, 24.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2123", "Skylake-W", "Xeon", 3.6, 3.9, 4,  8,  [32, 1024, 22.00 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2125", "Skylake-W", "Xeon", 4.0, 4.5, 4,  8,  [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2133", "Skylake-W", "Xeon", 3.6, 3.9, 6,  12, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2135", "Skylake-W", "Xeon", 3.7, 4.5, 6,  12, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2140B","Skylake-W", "Xeon", 3.2, 4.2, 8,  16, [32, 1024, 16.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-2150B","Skylake-W", "Xeon", 3.0, 4.5, 10, 20, [32, 1024, 27.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # TODO: Fill in Mobile, Workstation, Server, Skylake-SP Processors

    # Xeon Skylake-SP
    # ref: https://en.wikipedia.org/wiki/List_of_Intel_Skylake-based_Xeon_microprocessors
    ["Intel 4108",  "Skylake", "Xeon Silver", 1.8, 3.0, 8,  16, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 4109T", "Skylake", "Xeon Silver", 2.0, 3.0, 8,  16, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 4110",  "Skylake", "Xeon Silver", 2.1, 3.0, 8,  16, [32, 1024, 19.25 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 4112",  "Skylake", "Xeon Silver", 2.6, 3.0, 4,  8,  [32, 1024, 16.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 4114",  "Skylake", "Xeon Silver", 2.2, 3.0, 10, 20, [32, 1024, 27.50 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    # Intel Kaby Lake
    # ref: https://en.wikipedia.org/wiki/Kaby_Lake
    # Desktop processors
    ["Intel 7740X", "Kaby Lake", "Core i7", 4.3, {1: 4.5, 2: 4.5, 4: 4.5}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7700K", "Kaby Lake", "Core i7", 4.2, {1: 4.5, 2: 4.4, 4: 4.5}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7700",  "Kaby Lake", "Core i7", 3.6, {1: 4.2, 2: 4.1, 4: 4.0}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7700T", "Kaby Lake", "Core i7", 2.9, {1: 3.8, 2: 3.7, 4: 3.6}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 7640X", "Kaby Lake", "Core i5", 4.0, {1: 4.2, 2: 4.2, 4: 4.0}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7600K", "Kaby Lake", "Core i5", 3.8, {1: 4.2, 2: 4.1, 4: 4.0}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7600",  "Kaby Lake", "Core i5", 3.5, {1: 4.1, 2: 4.0, 4: 3.9}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7600T", "Kaby Lake", "Core i5", 2.8, {1: 3.7, 2: 3.6, 4: 3.5}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7500",  "Kaby Lake", "Core i5", 3.4, {1: 3.8, 2: 3.7, 4: 3.6}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7500T", "Kaby Lake", "Core i5", 2.7, {1: 3.3, 2: 3.2, 4: 3.1}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7400",  "Kaby Lake", "Core i5", 3.0, {1: 3.5, 2: 3.4, 4: 3.3}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7400T", "Kaby Lake", "Core i5", 2.4, {1: 3.0, 2: 2.9, 4: 2.7}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 7350K",  "Kaby Lake", "Core i3", 4.2,                   {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7320",   "Kaby Lake", "Core i3", 4.1,                   {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7300",   "Kaby Lake", "Core i3", 4.0,                   {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7300T",  "Kaby Lake", "Core i3", 3.5,                   {}, 2, 4, [32, 256, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7100",   "Kaby Lake", "Core i3", 3.9,                   {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7100T",  "Kaby Lake", "Core i3", 3.4,                   {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7101E",  "Kaby Lake", "Core i3", 3.9,                   {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 7101TE", "Kaby Lake", "Core i3", 3.4,                   {}, 2, 4, [32, 256, 3 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # TODO: Fill in Pentium, Celeron Processors

    # TODO: Fill in Mobile Processors

    # Server/workstation Xeon processors
    ["Intel E3-1285 v6", "Kaby Lake", "Xeon", 4.1, {1: 4.5}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1280 v6", "Kaby Lake", "Xeon", 3.9, {1: 4.2}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1275 v6", "Kaby Lake", "Xeon", 3.8, {1: 4.2}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1270 v6", "Kaby Lake", "Xeon", 3.8, {1: 4.2}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1245 v6", "Kaby Lake", "Xeon", 3.7, {1: 4.1}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1240 v6", "Kaby Lake", "Xeon", 3.7, {1: 4.1}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1230 v6", "Kaby Lake", "Xeon", 3.5, {1: 3.9}, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel E3-1225 v6", "Kaby Lake", "Xeon", 3.3, {1: 3.7}, 4, 4, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E3-1220 v6", "Kaby Lake", "Xeon", 3.0, {1: 3.5}, 4, 4, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # Intel Kaby Lake Refresh
    # https://en.wikipedia.org/wiki/List_of_Intel_Core_i7_processors
    ["Intel 8550U", "Kaby Lake R", "Core i7", 1.8, 4.0, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8650U", "Kaby Lake R", "Core i7", 1.9, 4.2, 4, 8, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # TODO: Fill in remaining Kaby Lake data

    # Intel Coffee Lake
    # ref: https://en.wikipedia.org/wiki/Coffee_Lake
    # Desktop processors (Coffee Lake S)
    ["Intel 8086K", "Coffee Lake", "Core i7", 4.0, {1: 5.0, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.4, 6: 4.3}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8700K", "Coffee Lake", "Core i7", 3.7, {1: 4.7, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.4, 6: 4.3}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8700",  "Coffee Lake", "Core i7", 3.2, {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.3, 6: 4.3}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8700T", "Coffee Lake", "Core i7", 2.4, {1: 4.0, 2: 4.0, 3: 3.9, 4: 3.9, 5: 3.8, 6: 3.8}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 8600K", "Coffee Lake", "Core i5", 3.6, {1: 4.3, 2: 4.2, 3: 4.2, 4: 4.2, 5: 4.1, 6: 4.1}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8600",  "Coffee Lake", "Core i5", 3.1, {1: 4.3, 2: 4.2, 3: 4.2, 4: 4.2, 5: 4.1, 6: 4.1}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8600T", "Coffee Lake", "Core i5", 2.3, {1: 3.7, 2: 3.6, 3: 3.6, 4: 3.6, 5: 3.5, 6: 3.5}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8500",  "Coffee Lake", "Core i5", 3.0, {1: 4.1, 2: 4.0, 3: 4.0, 4: 4.0, 5: 3.9, 6: 3.9}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8500T", "Coffee Lake", "Core i5", 2.1, {1: 3.5, 2: 3.4, 3: 3.3, 4: 3.3, 5: 3.2, 6: 3.2}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8400",  "Coffee Lake", "Core i5", 2.8, {1: 4.0, 2: 3.9, 3: 3.9, 4: 3.9, 5: 3.8, 6: 3.8}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8400T", "Coffee Lake", "Core i5", 1.7, {1: 3.3, 2: 3.2, 3: 3.1, 4: 3.1, 5: 3.0, 6: 3.0}, 6, 6, [32, 256, 9 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 8350K", "Coffee Lake", "Core i3", 4.0,                                         {}, 4, 4, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8300",  "Coffee Lake", "Core i3", 3.7,                                         {}, 4, 4, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8300T", "Coffee Lake", "Core i3", 3.2,                                         {}, 4, 4, [32, 256, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8100",  "Coffee Lake", "Core i3", 3.6,                                         {}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8100F", "Coffee Lake", "Core i3", 3.6,                                         {}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 8100T", "Coffee Lake", "Core i3", 3.1,                                         {}, 4, 4, [32, 256, 6 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # TODO: Fill in Pentium, Celeron Processors

    # Workstation processors (Coffee Lake S)
    ["Intel 2186G", "Coffee Lake", "Xeon E", 3.8, {i + 1:4.7 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2176G", "Coffee Lake", "Xeon E", 3.7, {i + 1:4.7 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2146G", "Coffee Lake", "Xeon E", 3.5, {i + 1:4.5 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2136",  "Coffee Lake", "Xeon E", 3.3, {i + 1:4.5 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2126G", "Coffee Lake", "Xeon E", 3.3, {i + 1:4.5 for i in range(6)}, 6,  6, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2174G", "Coffee Lake", "Xeon E", 3.8, {i + 1:4.7 for i in range(6)}, 4,  8, [32, 256,  8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2144G", "Coffee Lake", "Xeon E", 3.6, {i + 1:4.5 for i in range(6)}, 4,  8, [32, 256,  8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2134",  "Coffee Lake", "Xeon E", 3.5, {i + 1:4.5 for i in range(6)}, 4,  8, [32, 256,  8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2124G", "Coffee Lake", "Xeon E", 3.4, {i + 1:4.5 for i in range(6)}, 4,  4, [32, 256,  8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2124",  "Coffee Lake", "Xeon E", 3.3, {i + 1:4.3 for i in range(6)}, 4,  4, [32, 256,  8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 2104G", "Coffee Lake", "Xeon E", 3.2,                            {}, 4,  4, [32, 256,  8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # TODO: Fill in remaining Coffee Lake data

    # Intel Comet Lake
    # https://en.wikipedia.org/wiki/Comet_Lake_(microprocessor)
    # Desktop processors
    ["Intel 10900K",  "Comet Lake", "Core i9", 3.7, {i+1:4.8 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10900KF", "Comet Lake", "Core i9", 3.7, {i+1:4.8 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10910",   "Comet Lake", "Core i9", 3.6, {i+1:4.7 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10900",   "Comet Lake", "Core i9", 2.8, {i+1:4.5 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10900F",  "Comet Lake", "Core i9", 2.8, {i+1:4.5 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10900T",  "Comet Lake", "Core i9", 1.9, {i+1:3.7 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10850K",  "Comet Lake", "Core i9", 3.6, {i+1:4.7 for i in range(10)}, 10, 20, [32, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 10700K",  "Comet Lake", "Core i7", 3.8, {i+1:4.7 for i in range(8)}, 8, 16, [32, 256, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10700KF", "Comet Lake", "Core i7", 3.8, {i+1:4.7 for i in range(8)}, 8, 16, [32, 256, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10700",   "Comet Lake", "Core i7", 2.9, {i+1:4.6 for i in range(8)}, 8, 16, [32, 256, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10700F",  "Comet Lake", "Core i7", 2.9, {i+1:4.6 for i in range(8)}, 8, 16, [32, 256, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10700T",  "Comet Lake", "Core i7", 2.0, {i+1:3.7 for i in range(8)}, 8, 16, [32, 256, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 10600K",  "Comet Lake", "Core i5", 4.1, {i+1:4.5 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10600KF", "Comet Lake", "Core i5", 4.1, {i+1:4.5 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10600",   "Comet Lake", "Core i5", 3.3, {i+1:4.4 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10600T",  "Comet Lake", "Core i5", 2.4, {i+1:3.7 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10500",   "Comet Lake", "Core i5", 3.1, {i+1:4.2 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10500T",  "Comet Lake", "Core i5", 2.3, {i+1:3.5 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10400",   "Comet Lake", "Core i5", 2.9, {i+1:4.0 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10400F",  "Comet Lake", "Core i5", 2.9, {i+1:4.0 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10400T",  "Comet Lake", "Core i5", 2.0, {i+1:3.2 for i in range(6)}, 6, 12, [32, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    ["Intel 10320",  "Comet Lake", "Core i3", 3.8, {i+1:4.4 for i in range(4)}, 4, 8, [32, 356, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10300",  "Comet Lake", "Core i3", 3.7, {i+1:4.2 for i in range(4)}, 4, 8, [32, 356, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10300T", "Comet Lake", "Core i3", 3.0, {i+1:3.6 for i in range(4)}, 4, 8, [32, 356, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10100",  "Comet Lake", "Core i3", 3.6, {i+1:4.1 for i in range(4)}, 4, 8, [32, 356, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10100F", "Comet Lake", "Core i3", 3.6, {i+1:4.1 for i in range(4)}, 4, 8, [32, 356, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 10100T", "Comet Lake", "Core i3", 3.0, {i+1:3.5 for i in range(4)}, 4, 8, [32, 356, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # TODO: Fill in Pentium, Celeron Processors

    # Workstation processors
    ["Intel W-1290P", "Comet Lake", "Xeon W", 3.7, {i+1:4.8 for i in range(10)}, 10, 20, [32, 356, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel W-1290",  "Comet Lake", "Xeon W", 3.2, {i+1:4.6 for i in range(10)}, 10, 20, [32, 356, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel W-1290T", "Comet Lake", "Xeon W", 1.9, {i+1:3.8 for i in range(10)}, 10, 20, [32, 356, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel W-1270P", "Comet Lake", "Xeon W", 3.8, {i+1:4.7 for i in range(10)}, 10, 20, [32, 356, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel W-1270",  "Comet Lake", "Xeon W", 3.4, {i+1:4.7 for i in range(10)}, 10, 20, [32, 356, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel W-1250P", "Comet Lake", "Xeon W", 4.1, {i+1:4.5 for i in range(10)}, 10, 20, [32, 356, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel W-1250",  "Comet Lake", "Xeon W", 3.3, {i+1:4.4 for i in range(10)}, 10, 20, [32, 356, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # TODO: Fill in remaining Comet Lake data

    # Intel Rocket Lake
    # https://en.wikipedia.org/wiki/Rocket_Lake
    # Desktop processors
    ["Intel 11600T",  "Rocket Lake", "Core i5", 1.7, {**{i+1:3.5 for i in range(6)}, **{1: 4.1},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11600KF", "Rocket Lake", "Core i5", 3.9, {**{i+1:4.6 for i in range(6)}, **{1: 4.9},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11600K",  "Rocket Lake", "Core i5", 3.9, {**{i+1:4.6 for i in range(6)}, **{1: 4.9},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11600",   "Rocket Lake", "Core i5", 2.8, {**{i+1:4.3 for i in range(6)}, **{1: 4.8},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11500T",  "Rocket Lake", "Core i5", 1.5, {**{i+1:3.4 for i in range(6)}, **{1: 3.9},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11500",   "Rocket Lake", "Core i5", 2.7, {**{i+1:4.2 for i in range(6)}, **{1: 4.6},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11400T",  "Rocket Lake", "Core i5", 1.3, {**{i+1:3.3 for i in range(6)}, **{1: 3.7},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11400F",  "Rocket Lake", "Core i5", 2.6, {**{i+1:4.2 for i in range(6)}, **{1: 4.4},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11400",   "Rocket Lake", "Core i5", 2.6, {**{i+1:4.2 for i in range(6)}, **{1: 4.4},          }, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11700T",  "Rocket Lake", "Core i7", 1.4, {**{i+1:3.6 for i in range(8)}, **{1: 4.5}, **{2: 4.6}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11700KF", "Rocket Lake", "Core i7", 3.6, {**{i+1:4.6 for i in range(8)}, **{1: 4.9}, **{2: 5.0}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11700K",  "Rocket Lake", "Core i7", 3.6, {**{i+1:4.6 for i in range(8)}, **{1: 4.9}, **{2: 5.0}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11700F",  "Rocket Lake", "Core i7", 2.5, {**{i+1:4.4 for i in range(8)}, **{1: 4.8}, **{2: 4.9}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11700",   "Rocket Lake", "Core i7", 2.5, {**{i+1:4.4 for i in range(8)}, **{1: 4.8}, **{2: 4.9}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11900T",  "Rocket Lake", "Core i9", 1.5, {**{i+1:3.7 for i in range(8)}, **{1: 4.8}, **{2: 4.9}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11900KF", "Rocket Lake", "Core i9", 3.5, {**{i+1:4.8 for i in range(8)}, **{1: 5.1}, **{2: 5.2}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11900K",  "Rocket Lake", "Core i9", 3.5, {**{i+1:4.8 for i in range(8)}, **{1: 5.1}, **{2: 5.2}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11900F",  "Rocket Lake", "Core i9", 2.5, {**{i+1:4.7 for i in range(8)}, **{1: 5.0}, **{2: 5.1}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 11900",   "Rocket Lake", "Core i9", 2.5, {**{i+1:4.7 for i in range(8)}, **{1: 5.0}, **{2: 5.1}}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # Workstation processors
    ["Intel W-1350",  "Rocket Lake", "Xeon W", 3.3, {i+1:5.0 for i in range(6)}, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-1350P", "Rocket Lake", "Xeon W", 4.0, {i+1:5.1 for i in range(6)}, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-1370",  "Rocket Lake", "Xeon W", 2.9, {i+1:5.1 for i in range(8)}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-1370P", "Rocket Lake", "Xeon W", 3.6, {i+1:5.2 for i in range(8)}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-1390",  "Rocket Lake", "Xeon W", 2.8, {i+1:5.2 for i in range(8)}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-1390P", "Rocket Lake", "Xeon W", 3.5, {i+1:5.3 for i in range(8)}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel W-1390T", "Rocket Lake", "Xeon W", 1.5, {i+1:4.9 for i in range(8)}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # Server processors
    ["Intel 2314",  "Rocket Lake", "Xeon E", 2.8, {1: 4.5}, 4, 4, [48, 512, 8 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2324G", "Rocket Lake", "Xeon E", 3.1, {1: 4.6}, 4, 4, [48, 512, 8 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2334",  "Rocket Lake", "Xeon E", 3.4, {1: 4.8}, 4, 8, [48, 512, 8 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2336",  "Rocket Lake", "Xeon E", 2.9, {1: 4.8}, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2356G", "Rocket Lake", "Xeon E", 3.2, {1: 5.0}, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2374G", "Rocket Lake", "Xeon E", 3.7, {1: 5.0}, 4, 8, [48, 512, 8 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2378",  "Rocket Lake", "Xeon E", 2.6, {1: 4.8}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2378G", "Rocket Lake", "Xeon E", 2.8, {1: 5.1}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2386G", "Rocket Lake", "Xeon E", 3.5, {1: 5.1}, 6, 12, [48, 512, 12 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 2388G", "Rocket Lake", "Xeon E", 3.2, {1: 5.1}, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],

    # Intel Ice Lake
    # ref: https://en.wikipedia.org/wiki/Ice_Lake_(microprocessor)
    # ref: https://en.wikichip.org/wiki/intel/microarchitectures/ice_lake_(client)
    ["Intel 1000G1", "Ice Lake", "Core i3", 1.1, {1: 3.2, 2: 3.2       }, 2, 4, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1000G4", "Ice Lake", "Core i3", 1.1, {1: 3.2, 2: 3.2       }, 2, 4, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1005G1", "Ice Lake", "Core i3", 1.2, {1: 3.4, 2: 3.4       }, 2, 4, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1030G4", "Ice Lake", "Core i5", 0.7, {1: 3.5,        4: 3.2}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1030G7", "Ice Lake", "Core i5", 0.8, {1: 3.5,        4: 3.2}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1035G1", "Ice Lake", "Core i5", 1.0, {1: 3.6,        4: 3.3}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1035G4", "Ice Lake", "Core i5", 1.1, {1: 3.7,        4: 3.3}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1035G7", "Ice Lake", "Core i5", 1.2, {1: 3.7,        4: 3.3}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1060G7", "Ice Lake", "Core i7", 1.0, {1: 3.8,        4: 3.4}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1065G7", "Ice Lake", "Core i7", 1.3, {1: 3.9, 2: 3.8, 4: 3.5}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 1068G7", "Ice Lake", "Core i7", 2.3, {1: 4.1,        4: 3.6}, 4, 8, [48, 512, 2 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],

    ["Intel 8351N", "Ice Lake", "Xeon Platinum", 2.40, {36: 3.10}, 36, 72, [48, 512, 54 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8352S", "Ice Lake", "Xeon Platinum", 2.20, {32: 2.80}, 32, 64, [48, 512, 48 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8352V", "Ice Lake", "Xeon Platinum", 2.10, {36: 2.50}, 36, 72, [48, 512, 54 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8352Y", "Ice Lake", "Xeon Platinum", 2.20, {32: 2.80}, 32, 64, [48, 512, 48 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8358",  "Ice Lake", "Xeon Platinum", 2.60, {32: 3.30}, 32, 64, [48, 512, 48 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8358P", "Ice Lake", "Xeon Platinum", 2.60, {32: 3.20}, 32, 64, [48, 512, 48 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8360Y", "Ice Lake", "Xeon Platinum", 2.40, {36: 3.10}, 36, 72, [48, 512, 54 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8362",  "Ice Lake", "Xeon Platinum", 2.80, {32: 3.50}, 32, 64, [48, 512, 48 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8368",  "Ice Lake", "Xeon Platinum", 2.40, {38: 3.20}, 38, 76, [48, 512, 57 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8368Q", "Ice Lake", "Xeon Platinum", 2.60, {38: 3.30}, 38, 76, [48, 512, 57 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8380",  "Ice Lake", "Xeon Platinum", 2.30, {40: 3.00}, 40, 80, [48, 512, 60 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],

    # Intel Cascade Lake
    # ref: https://en.wikipedia.org/wiki/Cascade_Lake_(microarchitecture)
    # ref: https://en.wikichip.org/wiki/intel/microarchitectures/cascade_lake
    # ref: https://en.wikipedia.org/wiki/List_of_Intel_Xeon_processors_(Cascade_Lake-based)
    ["Intel 6209U",   "Cascade Lake", "Xeon Gold", 2.1, {20: 3.9}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6210U",   "Cascade Lake", "Xeon Gold", 2.5, {20: 3.9}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6212U",   "Cascade Lake", "Xeon Gold", 2.4, {24: 3.9}, 24, 48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3223",  "Cascade Lake", "Xeon W",    3.5, { 8: 4.0},  8, 16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3225",  "Cascade Lake", "Xeon W",    3.7, { 8: 4.3},  8, 16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3235",  "Cascade Lake", "Xeon W",    3.3, {12: 4.4}, 12, 24, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3245",  "Cascade Lake", "Xeon W",    3.2, {16: 4.4}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3245M", "Cascade Lake", "Xeon W",    3.2, {16: 4.4}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3265",  "Cascade Lake", "Xeon W",    2.7, {24: 4.4}, 24, 48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3265M", "Cascade Lake", "Xeon W",    2.7, {24: 4.4}, 24, 48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3275",  "Cascade Lake", "Xeon W",    2.5, {28: 4.4}, 28, 56, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel W-3275M", "Cascade Lake", "Xeon W",    2.5, {28: 4.4}, 28, 56, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],

    ["Intel 3204",  "Cascade Lake", "Xeon Bronze",   1.9,        {},  6,   6, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5218R", "Cascade Lake", "Xeon Gold",     2.1, {20: 4.0}, 20,  40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5220R", "Cascade Lake", "Xeon Gold",     2.2, {24: 4.0}, 24,  48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6226R", "Cascade Lake", "Xeon Gold",     2.9, {16: 3.9}, 16,  32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6230R", "Cascade Lake", "Xeon Gold",     2.1, {26: 4.0}, 26,  52, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6238R", "Cascade Lake", "Xeon Gold",     2.2, {28: 4.0}, 28,  56, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6240R", "Cascade Lake", "Xeon Gold",     2.4, {24: 4.0}, 24,  48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6242R", "Cascade Lake", "Xeon Gold",     3.1, {20: 4.1}, 20,  40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6246R", "Cascade Lake", "Xeon Gold",     3.4, {16: 4.1}, 16,  32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6248R", "Cascade Lake", "Xeon Gold",       3, {24: 4.0}, 24,  48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6258R", "Cascade Lake", "Xeon Gold",     2.7, {28: 4.0}, 28,  56, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 9221",  "Cascade Lake", "Xeon Platinum", 2.1, {32: 3.7}, 32,  64, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 9222",  "Cascade Lake", "Xeon Platinum", 2.3, {32: 3.7}, 32,  64, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 9242",  "Cascade Lake", "Xeon Platinum", 2.3, {48: 3.8}, 48,  96, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 9282",  "Cascade Lake", "Xeon Platinum", 2.6, {56: 3.8}, 56, 112, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4208",  "Cascade Lake", "Xeon Silver",   2.1, { 8: 3.2},  8,  16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4209T", "Cascade Lake", "Xeon Silver",   2.2, { 8: 3.2},  8,  16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4210",  "Cascade Lake", "Xeon Silver",   2.2, {10: 3.2}, 10,  20, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4210R", "Cascade Lake", "Xeon Silver",   2.4, {10: 3.2}, 10,  20, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4214",  "Cascade Lake", "Xeon Silver",   2.2, {12: 3.2}, 12,  24, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4214R", "Cascade Lake", "Xeon Silver",   2.4, {12: 3.5}, 12,  24, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4214Y", "Cascade Lake", "Xeon Silver",   2.2, {12: 3.2}, 12,  24, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4215",  "Cascade Lake", "Xeon Silver",   2.5, { 8: 3.5},  8,  16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4215R", "Cascade Lake", "Xeon Silver",   3.2, { 8: 4.0},  8,  16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 4216",  "Cascade Lake", "Xeon Silver",   2.1, {16: 3.2}, 16,  32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],

    ["Intel 5215",  "Cascade Lake", "Xeon Gold", 2.5, {10: 3.4}, 10, 20, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5215L", "Cascade Lake", "Xeon Gold", 2.5, {10: 3.4}, 10, 20, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5215M", "Cascade Lake", "Xeon Gold", 2.5, {10: 3.4}, 10, 20, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5217",  "Cascade Lake", "Xeon Gold", 3.0, { 8: 3.7},  8, 16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5218",  "Cascade Lake", "Xeon Gold", 2.3, {16: 3.9}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5218B", "Cascade Lake", "Xeon Gold", 2.3, {16: 3.9}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5218N", "Cascade Lake", "Xeon Gold", 2.3, {16: 3.7}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5218T", "Cascade Lake", "Xeon Gold", 2.1, {16: 3.8}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5220",  "Cascade Lake", "Xeon Gold", 2.2, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5220S", "Cascade Lake", "Xeon Gold", 2.7, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5220T", "Cascade Lake", "Xeon Gold", 1.9, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 5222",  "Cascade Lake", "Xeon Gold", 3.8, { 4: 3.9},  4,  8, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6222V", "Cascade Lake", "Xeon Gold", 1.8, {20: 3.6}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6226",  "Cascade Lake", "Xeon Gold", 2.7, {12: 3.7}, 12, 24, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6230",  "Cascade Lake", "Xeon Gold", 2.1, {20: 3.9}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6230N", "Cascade Lake", "Xeon Gold", 2.3, {20: 3.9}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6230T", "Cascade Lake", "Xeon Gold", 2.1, {20: 3.9}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6234",  "Cascade Lake", "Xeon Gold", 3.3, { 8: 4.0},  8, 16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6238",  "Cascade Lake", "Xeon Gold", 2.1, {22: 3.7}, 22, 44, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6238L", "Cascade Lake", "Xeon Gold", 2.1, {22: 3.7}, 22, 44, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6238M", "Cascade Lake", "Xeon Gold", 2.1, {22: 3.7}, 22, 44, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6238T", "Cascade Lake", "Xeon Gold", 1.9, {22: 3.7}, 22, 44, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6240",  "Cascade Lake", "Xeon Gold", 2.6, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6240L", "Cascade Lake", "Xeon Gold", 2.6, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6240M", "Cascade Lake", "Xeon Gold", 2.6, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6240Y", "Cascade Lake", "Xeon Gold", 2.6, {18: 3.9}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6242",  "Cascade Lake", "Xeon Gold", 2.8, {16: 3.9}, 16, 32, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6244",  "Cascade Lake", "Xeon Gold", 3.6, { 8: 4.4},  8, 16, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6246",  "Cascade Lake", "Xeon Gold", 3.3, {12: 4.2}, 12, 24, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6248",  "Cascade Lake", "Xeon Gold", 2.5, {20: 3.9}, 20, 40, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6252",  "Cascade Lake", "Xeon Gold", 2.1, {24: 3.7}, 24, 48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6252N", "Cascade Lake", "Xeon Gold", 2.3, {24: 3.6}, 24, 48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6254",  "Cascade Lake", "Xeon Gold", 3.1, {18: 4.0}, 18, 36, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 6262V", "Cascade Lake", "Xeon Gold", 1.9, {24: 3.6}, 24, 48, [32, 1024, 1.375 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],

    ["Intel 8253",  "Cascade Lake", "Xeon Platinum", 2.2, {16: 3.0}, 16, 32, [32, 1024, 22 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8256",  "Cascade Lake", "Xeon Platinum", 3.8, { 4: 3.9},  4,  8, [32, 1024, 16.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8260",  "Cascade Lake", "Xeon Platinum", 2.4, {24: 3.9}, 24, 48, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8260L", "Cascade Lake", "Xeon Platinum", 2.4, {24: 3.9}, 24, 48, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8260M", "Cascade Lake", "Xeon Platinum", 2.4, {24: 3.9}, 24, 48, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8260Y", "Cascade Lake", "Xeon Platinum", 2.4, {24: 3.9}, 24, 48, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8268",  "Cascade Lake", "Xeon Platinum", 2.9, {24: 3.9}, 24, 48, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8270",  "Cascade Lake", "Xeon Platinum", 2.7, {26: 4.0}, 26, 52, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8272CL",  "Cascade Lake", "Xeon Platinum", 2.6, {26: 3.4}, 26, 52, [32, 1024, 35.75 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8273CL",  "Cascade Lake", "Xeon Platinum", 2.2, {28: 3.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8276",  "Cascade Lake", "Xeon Platinum", 2.2, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8276L", "Cascade Lake", "Xeon Platinum", 2.2, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8276M", "Cascade Lake", "Xeon Platinum", 2.2, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8280",  "Cascade Lake", "Xeon Platinum", 2.7, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8280L", "Cascade Lake", "Xeon Platinum", 2.7, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8280M", "Cascade Lake", "Xeon Platinum", 2.7, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],
    ["Intel 8284",  "Cascade Lake", "Xeon Platinum", 3.0, {28: 4.0}, 28, 56, [32, 1024, 38.5 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512", "AVX-VNNI"], "X86_64", "OPENMP"],

    # Intel Tiger Lake
    # ref: https://en.wikipedia.org/wiki/Tiger_Lake
    # Desktop processors
    ["Intel 11900KB", "Tiger Lake", "Core i9", 3.3, 4.9, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 11850H",  "Tiger Lake", "Core i7", 3.2, 4.8, 8, 16, [48, 512, 24 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 11700B",  "Tiger Lake", "Core i7", 3.2, 4.8, 8, 16, [48, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 11500B",  "Tiger Lake", "Core i5", 3.3, 4.6, 6, 12, [48, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 11100B",  "Tiger Lake", "Core i3", 3.6, 4.4, 4, 8,  [48, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # Mobile processors
    ["Intel 1195G7",  "Tiger Lake", "Core i7", 2.9, 5.0, 4, 8, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1185G7",  "Tiger Lake", "Core i7", 3.0, 4.8, 4, 8, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1165G7",  "Tiger Lake", "Core i7", 2.8, 4.7, 4, 8, [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1155G7",  "Tiger Lake", "Core i5", 2.5, 4.5, 4, 8,  [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1145G7",  "Tiger Lake", "Core i5", 2.6, 4.4, 4, 8,  [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1135G7",  "Tiger Lake", "Core i5", 2.4, 4.2, 4, 8,  [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1125G7",  "Tiger Lake", "Core i3", 2.0, 3.7, 4, 8,  [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 1115G7",  "Tiger Lake", "Core i3", 3.0, 4.1, 2, 4,  [48, 512, 16 * 1024], [64, 64, 64], 64, 32, ["SSE4.1", "SSE4.2", "AVX2", "AVX512"], "X86_64", "OPENMP"],
    ["Intel 7505",    "Tiger Lake", "Pentium Gold", 2.0, 3.5, 2, 4,  [48, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel 6035",    "Tiger Lake", "Celeron", 1.8, 0.0, 2, 2,  [48, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # Intel Ivy Bridge
    # https://en.wikipedia.org/wiki/List_of_Intel_Xeon_processors_(Ivy_Bridge-based)
    ["Intel E5-1607 v2",  "Ivy Bridge", "Xeon E5", 3.0, 3.0, 4, 8, [48, 256, 10 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-1620 v2",  "Ivy Bridge", "Xeon E5", 3.7, 3.9, 4, 8, [48, 256, 10 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-1650 v2",  "Ivy Bridge", "Xeon E5", 3.5, 3.9, 6, 12, [48, 256, 12 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-1660 v2",  "Ivy Bridge", "Xeon E5", 3.7, 4.0, 6, 12, [48, 256, 15 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-1680 v2",  "Ivy Bridge", "Xeon E5", 3.0, 3.9, 8, 16, [48, 256, 25 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # Intel Haswell
    # ref: https://en.wikipedia.org/wiki/List_of_Intel_Haswell-based_Xeon_microprocessors
    ["Intel E5-1650 v3",  "Haswell", "Xeon E5", 3.5, 3.8, 6, 12, [48, 256, 15 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-1660 v3",  "Haswell", "Xeon E5", 3.0, 3.5, 8, 16, [48, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-1680 v3",  "Haswell", "Xeon E5", 3.2, 3.8, 8, 16, [48, 256, 20 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["Intel E5-2620 v3",  "Haswell", "Xeon E5", 2.4, 3.2, 6, 12, [48, 256, 15 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # AMD Zen
    # ref: https://en.wikipedia.org/wiki/Zen_(first_generation)
    # ref: https://en.wikichip.org/wiki/amd/microarchitectures/zen
    ["AMD 200GE", "Zen", "Athlon", 3.2, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 220GE", "Zen", "Athlon", 3.4, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 240GE", "Zen", "Athlon", 3.5, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 300U", "Zen", "Athlon", 2.4, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3150U", "Zen", "Athlon Gold", 2.4, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 200GE", "Zen", "Athlon", 3.2, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3050U", "Zen", "Athlon Silver", 2.3, {}, 2, 2, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7351P", "Zen", "EPYC", 2.4, {1: 2.9}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7401P", "Zen", "EPYC", 2, {1: 3}, 24, 48, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7551P", "Zen", "EPYC", 2, {1: 3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3101", "Zen", "EPYC Embedded", 2.1, {1: 2.9}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3151", "Zen", "EPYC Embedded", 2.7, {1: 2.9}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3201", "Zen", "EPYC Embedded", 1.5, {1: 3.1}, 8, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3251", "Zen", "EPYC Embedded", 2.5, {1: 3.1}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3255", "Zen", "EPYC Embedded", 2.5, {}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3301", "Zen", "EPYC Embedded", 2, {1: 3}, 12, 12, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3351", "Zen", "EPYC Embedded", 1.9, {1: 3}, 12, 24, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3401", "Zen", "EPYC Embedded", 1.85, {1: 3}, 16, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3451", "Zen", "EPYC Embedded", 2.15, {1: 3}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD FireFlight", "Zen", "", 3, {}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1200", "Zen", "Ryzen 3", 3.1, {1: 3.4}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1300X", "Zen", "Ryzen 3", 3.5, {1: 3.7}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2200G", "Zen", "Ryzen 3", 3.5, {1: 3.7}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2200GE", "Zen", "Ryzen 3", 3.2, {1: 3.6}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2200U", "Zen", "Ryzen 3", 2.5, {1: 3.4}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2300U", "Zen", "Ryzen 3", 2, {1: 3.4}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3250U", "Zen", "Ryzen 3", 2.6, {}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 1200", "Zen", "Ryzen 3", 3.1, {1: 3.4}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 1300", "Zen", "Ryzen 3", 3.5, {1: 3.7}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2200G", "Zen", "Ryzen 3", 3.5, {1: 3.7}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2200GE", "Zen", "Ryzen 3", 3.2, {1: 3.6}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2300U", "Zen", "Ryzen 3", 2, {1: 3.4}, 4, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1400", "Zen", "Ryzen 5", 3.2, {1: 3.4}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1500X", "Zen", "Ryzen 5", 3.5, {1: 3.7}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1600", "Zen", "Ryzen 5", 3.2, {1: 3.6}, 6, 12, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1600X", "Zen", "Ryzen 5", 3.6, {1: 4}, 6, 12, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2400G", "Zen", "Ryzen 5", 3.6, {1: 3.9}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2400GE", "Zen", "Ryzen 5", 3.2, {1: 3.8}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2500U", "Zen", "Ryzen 5", 2, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2600H", "Zen", "Ryzen 5", 3.2, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 1500", "Zen", "Ryzen 5", 3.5, {1: 3.7}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 1600", "Zen", "Ryzen 5", 3.2, {1: 3.6}, 6, 12, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2400G", "Zen", "Ryzen 5", 3.6, {1: 3.9}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2400GE", "Zen", "Ryzen 5", 3.2, {1: 3.8}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2500U", "Zen", "Ryzen 5", 2, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1700", "Zen", "Ryzen 7", 3, {1: 3.7}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1700X", "Zen", "Ryzen 7", 3.4, {1: 3.8}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1800X", "Zen", "Ryzen 7", 3.6, {1: 4}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2700U", "Zen", "Ryzen 7", 2.2, {1: 3.8}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2800H", "Zen", "Ryzen 7", 3.3, {1: 3.8}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 1700", "Zen", "Ryzen 7", 3, {1: 3.7}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 1700X", "Zen", "Ryzen 7", 3.4, {1: 3.8}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2700U", "Zen", "Ryzen 7", 2.2, {1: 3.8}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD R1102G", "Zen", "Ryzen Embedded", 1.2, {1: 2.6}, 2, 2, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD R1305G", "Zen", "Ryzen Embedded", 1.5, {1: 2.8}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD R1505G", "Zen", "Ryzen Embedded", 2.4, {1: 3.3}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD R1606G", "Zen", "Ryzen Embedded", 2.6, {1: 3.5}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1202B", "Zen", "Ryzen Embedded", 2.3, {1: 3.2}, 2, 4, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1404I", "Zen", "Ryzen Embedded", 2, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1500B", "Zen", "Ryzen Embedded", 2.2, {}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1605B", "Zen", "Ryzen Embedded", 2, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1756B", "Zen", "Ryzen Embedded", 3.25, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1780B", "Zen", "Ryzen Embedded", 3.35, {1: 3.6}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V1807B", "Zen", "Ryzen Embedded", 3.35, {1: 3.8}, 4, 8, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1900X", "Zen", "Ryzen Threadripper", 3.8, {1: 4}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1920X", "Zen", "Ryzen Threadripper", 3.5, {1: 4}, 12, 24, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 1950X", "Zen", "Ryzen Threadripper", 3.4, {1: 4}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7251", "Zen", "EPYC", 2.1, {1: 2.9}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7261", "Zen", "EPYC", 2.5, {1: 2.9}, 8, 16, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7281", "Zen", "EPYC", 2.1, {1: 2.7}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7301", "Zen", "EPYC", 2.2, {1: 2.7}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7351", "Zen", "EPYC", 2.4, {1: 2.9}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7371", "Zen", "EPYC", 3.1, {1: 3.8}, 16, 32, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7401", "Zen", "EPYC", 2, {1: 3}, 24, 48, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7451", "Zen", "EPYC", 2.3, {1: 3.2}, 24, 48, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7501", "Zen", "EPYC", 2, {1: 3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7551", "Zen", "EPYC", 2, {1: 3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7601", "Zen", "EPYC", 2.2, {1: 3.2}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # AMD Zen+
    # ref: https://en.wikichip.org/wiki/amd/microarchitectures/zen%2B#All_Zen.2B_Chips
    ["AMD 3000G", "Zen+", "Athlon", 3.5, {}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 300GE", "Zen+", "Athlon", 3.4, {}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 300U", "Zen+", "Athlon", 2.4, {1: 3.3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2300X", "Zen+", "Ryzen 3", 3.5, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3200G", "Zen+", "Ryzen 3", 3.6, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3200U", "Zen+", "Ryzen 3", 2.6, {1: 3.5}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3300U", "Zen+", "Ryzen 3", 2.1, {1: 3.5}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3200G", "Zen+", "Ryzen 3", 3.6, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3200GE", "Zen+", "Ryzen 3", 3.3, {1: 3.8}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3300U", "Zen+", "Ryzen 3", 2.1, {1: 3.5}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2500X", "Zen+", "Ryzen 5", 3.6, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2600", "Zen+", "Ryzen 5", 3.4, {1: 3.9}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2600E", "Zen+", "Ryzen 5", 3.1, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2600X", "Zen+", "Ryzen 5", 3.6, {1: 4.2}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3400G", "Zen+", "Ryzen 5", 3.7, {1: 4.2}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3500U", "Zen+", "Ryzen 5", 2.1, {1: 3.7}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3550H", "Zen+", "Ryzen 5", 2.1, {1: 3.7}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3580U", "Zen+", "Ryzen 5", 2.1, {1: 3.7}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2600", "Zen+", "Ryzen 5", 3.4, {1: 3.9}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3400G", "Zen+", "Ryzen 5", 3.7, {1: 4.2}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3400GE", "Zen+", "Ryzen 5", 3.3, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3500U", "Zen+", "Ryzen 5", 2.1, {1: 3.7}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2700", "Zen+", "Ryzen 7", 3.2, {1: 4.1}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2700E", "Zen+", "Ryzen 7", 2.8, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2700X Gold Edition", "Zen+", "Ryzen 7", 3.7, {1: 4.3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2700X", "Zen+", "Ryzen 7", 3.7, {1: 4.3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3700U", "Zen+", "Ryzen 7", 2.3, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3750H", "Zen+", "Ryzen 7", 2.3, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3780U", "Zen+", "Ryzen 7", 2.3, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2700", "Zen+", "Ryzen 7", 3.2, {1: 4.1}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 2700X", "Zen+", "Ryzen 7", 3.6, {1: 4.1}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3700U", "Zen+", "Ryzen 7", 2.3, {1: 4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2920X", "Zen+", "Ryzen Threadripper", 3.5, {1: 4.3}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2950X", "Zen+", "Ryzen Threadripper", 3.5, {1: 4.4}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2970WX", "Zen+", "Ryzen Threadripper", 3, {1: 4.2}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 2990WX", "Zen+", "Ryzen Threadripper", 3, {1: 4.2}, 32, 64, [32, 512, 2 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # AMD Zen2
    # ref: https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
    ["AMD 7232P", "Zen2", "EPYC", 3.1, {1: 3.2}, 8, 16, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7302P", "Zen2", "EPYC", 3, {1: 3.3}, 16, 32, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7402P", "Zen2", "EPYC", 2.8, {1: 3.35}, 24, 48, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7502P", "Zen2", "EPYC", 2.5, {1: 3.35}, 32, 64, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7702P", "Zen2", "EPYC", 2, {1: 3.35}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4300G", "Zen2", "Ryzen 3", 3.8, {1: 4}, 4, 8, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4300GE", "Zen2", "Ryzen 3", 3.5, {1: 4}, 4, 8, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4300U", "Zen2", "Ryzen 3", 2.7, {1: 3.7}, 4, 4, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5300U", "Zen2", "Ryzen 3", 2.6, {1: 3.8}, 4, 8, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4350G", "Zen2", "Ryzen 3", 3.8, {1: 4}, 4, 8, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4350GE", "Zen2", "Ryzen 3", 3.5, {1: 4}, 4, 8, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4450U", "Zen2", "Ryzen 3", 2.5, {1: 3.7}, 4, 8, [32, 2 * 1024, 4 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3500", "Zen2", "Ryzen 5", 3.6, {1: 4.1}, 6, 6, [32, 2 * 1024, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3500X", "Zen2", "Ryzen 5", 3.6, {1: 4.1}, 6, 6, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3600", "Zen2", "Ryzen 5", 3.6, {1: 4.2}, 6, 12, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3600X", "Zen2", "Ryzen 5", 3.8, {1: 4.4}, 6, 12, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3600XT", "Zen2", "Ryzen 5", 3.8, {1: 4.5}, 6, 12, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4500U", "Zen2", "Ryzen 5", 2.3, {1: 4}, 6, 6, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4600G", "Zen2", "Ryzen 5", 3.7, {1: 4.2}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4600GE", "Zen2", "Ryzen 5", 3.3, {1: 4.2}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4600H", "Zen2", "Ryzen 5", 3, {1: 4}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4600HS", "Zen2", "Ryzen 5", 3, {1: 4}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4600U", "Zen2", "Ryzen 5", 2.1, {1: 4}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4680U", "Zen2", "Ryzen 5", 2.1, {1: 4}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5500U", "Zen2", "Ryzen 5", 2.1, {1: 4}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3600", "Zen2", "Ryzen 5", 3.6, {1: 4.2}, 6, 12, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4650G", "Zen2", "Ryzen 5", 3.7, {1: 4.2}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4650GE", "Zen2", "Ryzen 5", 3.3, {1: 4.2}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4650U", "Zen2", "Ryzen 5", 2.1, {1: 4}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3700X", "Zen2", "Ryzen 7", 3.6, {1: 4.4}, 8, 16, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3800X", "Zen2", "Ryzen 7", 3.9, {1: 4.5}, 8, 16, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3800XT", "Zen2", "Ryzen 7", 3.9, {1: 4.7}, 8, 16, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4700G", "Zen2", "Ryzen 7", 3.6, {1: 4.4}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4700GE", "Zen2", "Ryzen 7", 3.1, {1: 4.3}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4700U", "Zen2", "Ryzen 7", 2, {1: 4.1}, 8, 8, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4800H", "Zen2", "Ryzen 7", 2.9, {1: 4.2}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4800HS", "Zen2", "Ryzen 7", 2.9, {1: 4.2}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4800U", "Zen2", "Ryzen 7", 1.8, {1: 4.2}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4980U", "Zen2", "Ryzen 7", 2, {1: 4.4}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5700U", "Zen2", "Ryzen 7", 1.8, {1: 4.3}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3700", "Zen2", "Ryzen 7", 3.6, {1: 4.4}, 8, 16, [32, 2 * 1024, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4750G", "Zen2", "Ryzen 7", 3.6, {1: 4.4}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4750GE", "Zen2", "Ryzen 7", 3.1, {1: 4.3}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 4750U", "Zen2", "Ryzen 7", 1.7, {1: 4.1}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3900", "Zen2", "Ryzen 9", 3.1, {1: 4.3}, 12, 24, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3900X", "Zen2", "Ryzen 9", 3.8, {1: 4.6}, 12, 24, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3900XT", "Zen2", "Ryzen 9", 3.8, {1: 4.7}, 12, 24, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3950X", "Zen2", "Ryzen 9", 3.5, {1: 4.7}, 16, 32, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4900H", "Zen2", "Ryzen 9", 3.3, {1: 4.4}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 4900HS", "Zen2", "Ryzen 9", 3, {1: 4.3}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 3900", "Zen2", "Ryzen 9", 3.1, {1: 4.3}, 12, 24, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V2516", "Zen2", "Ryzen Embedded", 2.1, {}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V2546", "Zen2", "Ryzen Embedded", 3, {}, 6, 12, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V2718", "Zen2", "Ryzen Embedded", 1.7, {}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD V2748", "Zen2", "Ryzen Embedded", 2.9, {}, 8, 16, [32, 2 * 1024, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3960X", "Zen2", "Ryzen Threadripper", 3.8, {1: 4.5}, 24, 48, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3970X", "Zen2", "Ryzen Threadripper", 3.7, {1: 4.5}, 32, 64, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3980X", "Zen2", "Ryzen Threadripper", 3.2, {1: 4.5}, 48, 96, [32, 2 * 1024, ], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 3990X", "Zen2", "Ryzen Threadripper", 2.9, {1: 4.3}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7252", "Zen2", "EPYC", 3.1, {1: 3.2}, 8, 16, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7262", "Zen2", "EPYC", 3.2, {1: 3.4}, 8, 16, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7272", "Zen2", "EPYC", 2.9, {1: 3.2}, 12, 24, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7282", "Zen2", "EPYC", 2.8, {1: 3.2}, 16, 32, [32, 2 * 1024, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7302", "Zen2", "EPYC", 3, {1: 3.3}, 16, 32, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7352", "Zen2", "EPYC", 2.3, {1: 3.2}, 24, 48, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7402", "Zen2", "EPYC", 2.8, {1: 3.35}, 24, 48, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7452", "Zen2", "EPYC", 2.35, {1: 3.35}, 32, 64, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7502", "Zen2", "EPYC", 2.5, {1: 3.35}, 32, 64, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7532", "Zen2", "EPYC", 2.4, {1: 3.3}, 32, 64, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7542", "Zen2", "EPYC", 2.9, {1: 3.4}, 32, 64, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7552", "Zen2", "EPYC", 2.2, {1: 3.35}, 48, 96, [32, 2 * 1024, 192 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7642", "Zen2", "EPYC", 2.3, {1: 3.3}, 48, 96, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7662", "Zen2", "EPYC", 2, {1: 3.3}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7702", "Zen2", "EPYC", 2, {1: 3.35}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7742", "Zen2", "EPYC", 2.25, {1: 3.4}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7F32", "Zen2", "EPYC", 3.7, {1: 3.9}, 8, 16, [32, 2 * 1024, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7F52", "Zen2", "EPYC", 3.5, {1: 3.9}, 16, 32, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7F72", "Zen2", "EPYC", 3.2, {1: 3.7}, 24, 48, [32, 2 * 1024, 192 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7H12", "Zen2", "EPYC", 2.6, {1: 3.3}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7V12", "Zen2", "EPYC", 2.45, {1: 3.3}, 64, 128, [32, 2 * 1024, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # AMD Zen3
    # ref: https://en.wikichip.org/wiki/amd/microarchitectures/zen_3
    ["AMD 7313P", "Zen3", "Milan", 3, {1: 3.7}, 16, 32, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7443P", "Zen3", "Milan", 2.85, {1: 4}, 24, 48, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7543P", "Zen3", "Milan", 2.8, {1: 3.7}, 32, 64, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7713P", "Zen3", "Milan", 2, {1: 3.675}, 64, 128, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5300G", "Zen3", "Cezanne", 4, {1: 4.2}, 4, 8, [32, 512, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5300GE", "Zen3", "Cezanne", 3.6, {1: 4.2}, 4, 8, [32, 512, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5400U", "Zen3", "Cezanne", 2.6, {1: 4}, 4, 8, [32, 512, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5350G", "Zen3", "Cezanne", 4, {1: 4.2}, 4, 8, [32, 512, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5350GE", "Zen3", "Cezanne", 3.6, {1: 4.2}, 4, 8, [32, 512, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5450U", "Zen3", "Cezanne", 2.6, {1: 4}, 4, 8, [32, 512, 8 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5600G", "Zen3", "Cezanne", 3.9, {1: 4.4}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5600GE", "Zen3", "Cezanne", 3.4, {1: 4.4}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5600H", "Zen3", "Cezanne", 3.3, {1: 4.2}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5600HS", "Zen3", "Cezanne", 3, {1: 4.2}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5600U", "Zen3", "Cezanne", 2.3, {1: 4.2}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5600X", "Zen3", "Vermeer", 3.7, {1: 4.6}, 6, 12, [32, 512, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5650G", "Zen3", "Cezanne", 3.9, {1: 4.4}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5650GE", "Zen3", "Cezanne", 3.4, {1: 4.4}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5650U", "Zen3", "Cezanne", 2.3, {1: 4.2}, 6, 12, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5700G", "Zen3", "Cezanne", 3.8, {1: 4.6}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5700GE", "Zen3", "Cezanne", 3.2, {1: 4.6}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5800", "Zen3", "Vermeer", 3.4, {1: 4.6}, 8, 16, [32, 512, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5800H", "Zen3", "Cezanne", 3.2, {1: 4.4}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5800HS", "Zen3", "Cezanne", 2.8, {1: 4.4}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5800U", "Zen3", "Cezanne", 1.9, {1: 4.4}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5800X", "Zen3", "Vermeer", 3.8, {1: 4.7}, 8, 16, [32, 512, 32 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5750G", "Zen3", "Cezanne", 3.8, {1: 4.6}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5750GE", "Zen3", "Cezanne", 3.2, {1: 4.6}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD PRO 5850U", "Zen3", "Cezanne", 1.9, {1: 4.4}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5900", "Zen3", "Vermeer", 3, {1: 4.7}, 12, 24, [32, 512, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5900HS", "Zen3", "Cezanne", 3, {1: 4.6}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5900HX", "Zen3", "Cezanne", 3.3, {1: 4.6}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5900X", "Zen3", "Vermeer", 3.7, {1: 4.8}, 12, 24, [32, 512, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5950X", "Zen3", "Vermeer", 3.4, {1: 4.9}, 16, 32, [32, 512, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5980HS", "Zen3", "Cezanne", 3, {1: 4.8}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 5980HX", "Zen3", "Cezanne", 3.3, {1: 4.8}, 8, 16, [32, 512, 16 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 72F3", "Zen3", "Milan", 3.7, {1: 4.1}, 8, 16, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7313", "Zen3", "Milan", 3, {1: 3.7}, 16, 32, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7343", "Zen3", "Milan", 3.2, {1: 3.9}, 16, 32, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 73F3", "Zen3", "Milan", 3.5, {1: 4}, 16, 32, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7413", "Zen3", "Milan", 2.65, {1: 3.6}, 24, 48, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7443", "Zen3", "Milan", 2.85, {1: 4}, 24, 48, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7453", "Zen3", "Milan", 2.75, {1: 3.45}, 28, 56, [32, 512, 64 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 74F3", "Zen3", "Milan", 3.2, {1: 4}, 24, 48, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7513", "Zen3", "Milan", 2.6, {1: 3.65}, 32, 64, [32, 512, 128 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7543", "Zen3", "Milan", 2.8, {1: 3.7}, 32, 64, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 75F3", "Zen3", "Milan", 2.95, {1: 4}, 32, 64, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7643", "Zen3", "Milan", 2.3, {1: 3.6}, 48, 96, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7663", "Zen3", "Milan", 2, {1: 3.5}, 56, 112, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7713", "Zen3", "Milan", 2, {1: 3.675}, 64, 128, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],
    ["AMD 7763", "Zen3", "Milan", 2.45, {1: 3.5}, 64, 128, [32, 512, 256 * 1024], [64, 64, 64], 32, 16, ["SSE4.1", "SSE4.2", "AVX2"], "X86_64", "OPENMP"],

    # Apple
    # ref: https://en.wikipedia.org/wiki/Apple_M1_Pro_and_M1_Max (incomplete information)
    ["Apple M1 Max", "M1 Max", "M1 Max", 2.0, {1: 3.2}, 10, 10, [24*1024, 48*1024], [], 0, 0, [], "AARCH64", "OPENMP"],

    # Raspberry Pi
    # ref: https://www.raspberrypi.org/app/uploads/2012/02/BCM2835-ARM-Peripherals.pdf
    # ref: http://sandsoftwaresound.net/raspberry-pi/raspberry-pi-gen-1/memory-hierarchy/
    ["Raspberry Pi Zero", "Pi0", "Broadcom BCM2835", 0.7, {1: 1}, 1, 2, [16], [32], 0, 0, [], "ARM", ""], # pi0 has a 128 KB L2, but it's usually reserved for the GPU
    # ref: https://patchwork.kernel.org/project/linux-arm-kernel/patch/20211218200009.16856-1-rs@noreya.tech/
    ["Raspberry Pi 3B", "Pi3", "Broadcom BCM2837B0", 1.4, {}, 4, 8, [32, 512], [64, 64], 0, 0, [], "ARM", "OPENMP"], # pi3 has a 128 KB L2, but it's usually reserved for GPU
    # ref: https://patchwork.kernel.org/project/linux-arm-kernel/patch/20211221224830.16746-1-rs@noreya.tech/
    ["Raspberry Pi 4B", "Pi4", "Broadcom BCM2711", 1.5, {}, 4, 8, [32, 1024], [64, 64], 0, 0, [], "ARM", "OPENMP"],

    ["ARM Cortex-M4", "Cortex-M4", "ARM Cortex-M4", .008, {}, 1, 1, [], [], 0, 0, [], "ARM", ""],
    ["ARM Cortex-M4F", "Cortex-M4", "ARM Cortex-M4F", .008, {}, 1, 1, [], [], 0, 0, ["fpu"], "ARM", ""],
]
# yapf: enable


@dataclass(frozen=True, eq=True)
class TensorCoreInformationEntry:
    shape: _MMAShape
    inType: ScalarType
    outType: ScalarType


@dataclass(frozen=True)
class TensorCoreInformation:
    entries: List[TensorCoreInformationEntry] = field(default_factory=list)

    def supports(
        self, input_type: ScalarType, output_type: ScalarType, shape: _MMAShape, num_total_passes: int,
        num_fused_passes: int
    ) -> bool:
        if not (num_total_passes >= 1 and (num_fused_passes == -1 or num_total_passes % num_fused_passes == 0)):
            return False

        for entry in self.entries:
            if input_type == entry.inType and output_type == entry.outType and entry.shape == shape:
                return True
        return False

    def mma_shape_to_tuple(self, mma_shape: _MMAShape):
        return {
            _MMAShape.M64xN64xK1_B4 : (64, 64, 1),
            _MMAShape.M64xN64xK1_B2 : (64, 64, 1),
            _MMAShape.M32xN32xK2_B1 : (32, 32, 2),
            _MMAShape.M16xN16xK4_B1 : (16, 16, 4),
            _MMAShape.M64xN64xK4_B4 : (64, 64, 4),
            _MMAShape.M64xN64xK4_B2 : (64, 64, 4),
            _MMAShape.M32xN32xK8_B1 : (32, 32, 8),
            _MMAShape.M16xN16xK16_B1 : (16, 16, 16),
            _MMAShape.M64xN64xK2_B4: (64, 64, 2),
            _MMAShape.M64xN64xK2_B2: (64, 64, 2),
            _MMAShape.M32xN32xK4_B1: (32, 32, 4),
            _MMAShape.M16xN16xK8_B1: (16, 16, 8),
            _MMAShape.M32xN8xK16_B1: (32, 8, 16),
            _MMAShape.M8xN32xK16_B1: (8, 32, 16)
        }[mma_shape]

    def compute_tensor_splits(self, mma_shape: _MMAShape, num_total_passes: int = 1):
        tensor_splits = self.mma_shape_to_tuple(mma_shape)
        mutable_tensor_splits = list(tensor_splits)
        mutable_tensor_splits[2] *= num_total_passes
        return tuple(mutable_tensor_splits)


MI100_TENSORCORE_INFO = TensorCoreInformation([
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK4_B1, inType=ScalarType.float32, outType=ScalarType.float32),    # maps to the 16x16x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK2_B1, inType=ScalarType.float32, outType=ScalarType.float32),    # maps to the 32x32x2 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK1_B2, inType=ScalarType.float32, outType=ScalarType.float32),    # maps to the 32x32x1 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK1_B4, inType=ScalarType.float32, outType=ScalarType.float32),    # maps to the 16x16x1 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.float16, outType=ScalarType.float16),    # maps to the 16x16x16 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK8_B1, inType=ScalarType.float16, outType=ScalarType.float16),    # maps to the 32x32x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B2, inType=ScalarType.float16, outType=ScalarType.float16),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B4, inType=ScalarType.float16, outType=ScalarType.float16),    # maps to the 16x16x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.float16, outType=ScalarType.float32),    # maps to the 16x16x16 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK8_B1, inType=ScalarType.float16, outType=ScalarType.float32),    # maps to the 32x32x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B2, inType=ScalarType.float16, outType=ScalarType.float32),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B4, inType=ScalarType.float16, outType=ScalarType.float32),    # maps to the 16x16x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.int8, outType=ScalarType.int32),    # maps to the 16x16x16 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK8_B1, inType=ScalarType.int8, outType=ScalarType.int32),    # maps to the 32x32x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B2, inType=ScalarType.int8, outType=ScalarType.int32),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B4, inType=ScalarType.int8, outType=ScalarType.int32),    # maps to the 16x16x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.int8, outType=ScalarType.int16),    # maps to the 16x16x16 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK8_B1, inType=ScalarType.int8, outType=ScalarType.int16),    # maps to the 32x32x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B2, inType=ScalarType.int8, outType=ScalarType.int16),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B4, inType=ScalarType.int8, outType=ScalarType.int16),    # maps to the 16x16x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.int8, outType=ScalarType.int8),    # maps to the 16x16x16 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK8_B1, inType=ScalarType.int8, outType=ScalarType.int8),    # maps to the 32x32x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B2, inType=ScalarType.int8, outType=ScalarType.int8),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK4_B4, inType=ScalarType.int8, outType=ScalarType.int8),    # maps to the 16x16x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK8_B1, inType=ScalarType.bfloat16, outType=ScalarType.float32),    # maps to the 16x16x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK4_B1, inType=ScalarType.bfloat16, outType=ScalarType.float32),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK2_B2, inType=ScalarType.bfloat16, outType=ScalarType.float32),    # maps to the 32x32x2 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK2_B4, inType=ScalarType.bfloat16, outType=ScalarType.float32),    # maps to the 16x16x2 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK8_B1, inType=ScalarType.bfloat16, outType=ScalarType.bfloat16),    # maps to the 16x16x8 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M32xN32xK4_B1, inType=ScalarType.bfloat16, outType=ScalarType.bfloat16),    # maps to the 32x32x4 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK2_B2, inType=ScalarType.bfloat16, outType=ScalarType.bfloat16),    # maps to the 32x32x2 mfma instruction
    TensorCoreInformationEntry(shape=_MMAShape.M64xN64xK2_B4, inType=ScalarType.bfloat16, outType=ScalarType.bfloat16),    # maps to the 16x16x2 mfma instruction
])

# https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#tensor-operations
NV_AMPERE_TENSORCORE_INFO = TensorCoreInformation([
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.float16, outType=ScalarType.float16),
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.float16, outType=ScalarType.float32),
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.bfloat16, outType=ScalarType.float32),
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.int8, outType=ScalarType.int32),
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK16_B1, inType=ScalarType.uint8, outType=ScalarType.int32),
    TensorCoreInformationEntry(shape=_MMAShape.M32xN8xK16_B1, inType=ScalarType.float16, outType=ScalarType.float16),
    TensorCoreInformationEntry(shape=_MMAShape.M32xN8xK16_B1, inType=ScalarType.float16, outType=ScalarType.float32),
    TensorCoreInformationEntry(shape=_MMAShape.M32xN8xK16_B1, inType=ScalarType.bfloat16, outType=ScalarType.float32),
    TensorCoreInformationEntry(shape=_MMAShape.M32xN8xK16_B1, inType=ScalarType.int8, outType=ScalarType.int32),
    TensorCoreInformationEntry(shape=_MMAShape.M32xN8xK16_B1, inType=ScalarType.uint8, outType=ScalarType.int32),
    TensorCoreInformationEntry(shape=_MMAShape.M8xN32xK16_B1, inType=ScalarType.float16, outType=ScalarType.float16),
    TensorCoreInformationEntry(shape=_MMAShape.M8xN32xK16_B1, inType=ScalarType.float16, outType=ScalarType.float32),
    TensorCoreInformationEntry(shape=_MMAShape.M8xN32xK16_B1, inType=ScalarType.bfloat16, outType=ScalarType.float32),
    TensorCoreInformationEntry(shape=_MMAShape.M8xN32xK16_B1, inType=ScalarType.int8, outType=ScalarType.int32),
    TensorCoreInformationEntry(shape=_MMAShape.M8xN32xK16_B1, inType=ScalarType.uint8, outType=ScalarType.int32),
    TensorCoreInformationEntry(shape=_MMAShape.M16xN16xK8_B1, inType=ScalarType.float32, outType=ScalarType.float32),
])

# Tensor Cores is current unused
KNOWN_GPUS_HEADER = [
    "Runtime", "Model", "Branding", "Family", "Cores", "MaxThreadsPerBlock", "MaxBlockSize", "MaxStaticSharedMemoryPerBlock",
    "MaxSharedMemoryPerBlock", "WarpSize", "Base Freq (GHz)", "MaxRegistersPerBlock", "Vector Bytes", "TensorCoreInformation"
]
KNOWN_GPUS = [
    # NVIDIA
    # Pascal Tuning Guide: https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#shared-memory-capacity
    ["CUDA", "NVidia P100", "Pascal", "sm60", 56, 1024, [1024, 1024, 64], 49152, 49152, 32, 1.328500, 65536, 0, None],    # TODO : get the real values for the vector register sizes in bytes

    # Volta Tuning Guide: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#l1-cache
    ["CUDA", "NVidia V100", "Volta", "sm70", 80, 1024, [1024, 1024, 64], 49152, 98304, 32, 1.380000, 65536, 0, None],

    # https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
    # Ampere Tuning Guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#l1-cache
    # Devices of compute capability 8.6 have 2x more FP32 operations per cycle per SM than devices of compute capability 8.0.
    # While a binary compiled for 8.0 will run as is on 8.6, it is recommended to compile explicitly for 8.6 to benefit from the increased FP32 throughput.
    ["CUDA", "NVidia A100", "Ampere", "sm80", 108, 1024, [1024, 1024, 64], 49152, 166912, 32, 1.410000, 65536, 0, NV_AMPERE_TENSORCORE_INFO],
    ["CUDA", "NVidia RTX A6000", "Ampere", "sm86", 108, 1024, [1024, 1024, 64], 49152, 101376, 32, 1.410000, 65536, 0, NV_AMPERE_TENSORCORE_INFO],

    # AMD
    ["ROCM", "AMD Radeon7", "Vega20", "gfx906", 60, 1024, [1024, 1024, 1024], 65536, 65536, 64, 1.801000, 65536, 0, None],
    ["ROCM", "AMD MI50", "Vega20", "gfx906", 60, 1024, [1024, 1024, 1024], 65536, 65536, 64, 1.725000, 65536, 0, None],

    # The MI100 can move up to Up to 4 DWORDs per instruction, so we set the vector size to 16 bytes - https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf
    ["ROCM", "AMD MI100", "Arcturus", "gfx908", 120, 1024, [1024, 1024, 1024], 65536, 65536, 64, 1.502000, 65536, 16, MI100_TENSORCORE_INFO],
    ["ROCM", "AMD MI200", "Aldebaran", "gfx90a", 220, 1024, [1024, 1024, 1024], 65536, 65536, 64, 1.700000, 65536, 0, None]
]
# yapf: enable


@dataclass
class _TargetContainer:
    "Generic container class for target information"
    architecture: Architecture = None
    cache_lines: List[int] = field(default_factory=list)
    cache_sizes: List[int] = field(default_factory=list)
    category: Category = None
    runtime: Runtime = Runtime.NONE
    extensions: List[str] = field(default_factory=list)
    family: str = ""
    frequency_GHz: float = 0.0
    name: str = ""
    num_cores: int = 0
    num_threads: int = 0
    tensor_core_info: TensorCoreInformation = field(default_factory=TensorCoreInformation)
    turbo_frequency_GHz: dict = field(default_factory=dict)    # Dictionary of number of cores needed => Turbo frequency
    vector_bytes: int = 0
    vector_registers: int = 0
    warp_size: int = 0
    max_threads_per_block: int = 0
    max_block_size: List[int] = field(default_factory=list)
    max_static_shared_memory_per_block: int = 0
    max_shared_memory_per_block: int = 0
    max_registers_per_block: int = 0

    _device_name: str = "host"    # used internally for emitting known targets

    _full_extensions: List[str] = field(default_factory=list)

    _max_frequency_GHz: float = 0.0
    _max_num_cores: int = 0
    _max_num_threads: int = 0
    _max_vector_bytes: int = 0
    _max_vector_registers: int = 0

    def __post_init__(self):

        device_name = \
            self.family.lower() if self.family else (self.name.lower() if self.name else self._device_name)
        if device_name in _GetKnownDeviceNames():
            self._device_name = device_name

        self._full_extensions = self.extensions

        self._max_frequency_GHz = self.frequency_GHz
        self._max_num_cores = self.num_cores
        self._max_num_threads = self.num_threads
        self._max_turbo_frequency_GHz = self.turbo_frequency_GHz
        self._max_vector_bytes = self.vector_bytes
        self._max_vector_registers = self.vector_registers

    @property
    def vectorization_info(self):
        from ._lang_python._lang import _VectorizationInfo

        return _VectorizationInfo(vector_bytes=self.vector_bytes, vector_units=self.vector_registers, unroll_only=False)


KNOWN_DEVICES = {
    Category.CPU: {},
    Category.GPU: {}
}

_MODEL_TRANSLATION_DICT = str.maketrans({c: '_'
                                         for c in " -."})


class _Models_enum_str:

    def __str__(self):
        return self.value


Model = None


def _recompute_known_devices():
    model_names = []
    for device in KNOWN_CPUS:
        device = {v: device[i]
                  for i, v in enumerate(KNOWN_CPUS_HEADER)}
        target = _TargetContainer(
            architecture=Architecture[device["ISA"]],
            cache_lines=device["Cache Lines"],
            cache_sizes=device["Cache Sizes (KB)"],
            category=Category.CPU,
            extensions=device["Extensions"],
            family=device["Family"],
            frequency_GHz=device["Base Freq (GHz)"],
            name=device["Model"],
            num_cores=device["Cores"],
            num_threads=device["Threads"],
            runtime=Runtime.__members__[device["Runtime"]] if device["Runtime"] else Runtime.NONE,
            turbo_frequency_GHz=device["Turbo Freq (GHz)"],
            vector_bytes=device["Vector Bytes"],
            vector_registers=device["Vector Registers"],
        )
        KNOWN_DEVICES[target.category][target.name] = target
        model_names.append((target.name, target.name))
        model_names.append((target.name.upper().translate(_MODEL_TRANSLATION_DICT), target.name))

    for device in KNOWN_GPUS:
        device = {v: device[i]
                  for i, v in enumerate(KNOWN_GPUS_HEADER)}
        target = _TargetContainer(
            category=Category.GPU,
            runtime=Runtime.__members__[device["Runtime"]],
            family=device["Family"],
            name=device["Model"],
            num_cores=device["Cores"],
            warp_size=device["WarpSize"],
            max_threads_per_block=device["MaxThreadsPerBlock"],
            max_block_size=device["MaxBlockSize"],
            max_static_shared_memory_per_block=device["MaxStaticSharedMemoryPerBlock"],
            max_shared_memory_per_block=device["MaxSharedMemoryPerBlock"],
            frequency_GHz=device["Base Freq (GHz)"],
            max_registers_per_block=device["MaxRegistersPerBlock"],
            tensor_core_info=device["TensorCoreInformation"],
            vector_bytes=device["Vector Bytes"],
            vector_registers=
            1,    # Setting this to 1 will enable vectorization but prevents unroll-and-jamming cache filling. TODO : get the right value for this
        )
        KNOWN_DEVICES[target.category][target.name] = target
        model_names.append((target.name, target.name))
        model_names.append((target.name.upper().translate(_MODEL_TRANSLATION_DICT), target.name))

    # This will raise an error if there's a duplicate enum name being added
    # It may need to be addressed in the future, but leaving it for when it
    # becomes necessary. It'll be the first device that can be run in both CPU and GPU modes
    global Model
    Model = Enum('Model', model_names, type=_Models_enum_str)


_recompute_known_devices()


class GridUnits(Enum):
    # Not expected to change from GPU to GPU
    BLOCK_X = BLOCK_X
    BLOCK_Y = BLOCK_Y
    BLOCK_Z = BLOCK_Z
    THREAD_X = THREAD_X
    THREAD_Y = THREAD_Y
    THREAD_Z = THREAD_Z
    WARP_X = WARP_X
    WARP_Y = WARP_Y


class Target(_TargetContainer):
    "Factory-like class for target information"

    Category = Category
    Architecture = Architecture
    Runtime = Runtime
    Model = Model

    def __init__(
        self,
        known_name: Union[str, Model] = "HOST",
        category: Category = None,
        architecture: Architecture = None,
        runtime: Runtime = None,
        name: str = None,
        family: str = None,
        extensions: List[str] = None,
        num_threads: int = None,
        num_cores: int = None,
        vector_bytes: int = 0,
        vector_registers: int = None,
        frequency_GHz: float = None,
        tensor_core_info: TensorCoreInformation = None,
        turbo_frequency_GHz: float = None,
        cache_sizes: List[int] = None,
        cache_lines: List[int] = None
    ):
        "Factory-like constructor that uses the model parameter to fill-in known defaults"

        super().__init__()

        if known_name:
            known_name = self._try_get_known_name(known_name)

            if known_name == "HOST":

                super().__init__(
                    category=category or Target.Category.CPU,
                    architecture=Target.Architecture["HOST"],
                    vector_bytes=32,    # There are 32-bytes per full SIMD register
                    vector_registers=16,    # There are 16 YMM registers
                    extensions=[
                        "MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE4", "SSE4.1", "SSE4.2", "AVX", "AVX2", "FMA3"
                    ]
                )
            else:

                if not category:
                    # Known devices are partitioned by device type, to allow for collisions in device names and to be
                    # able to select the GPU instead of CPU, but this might be unnecessary
                    # If category isn't provided, we check all categories and if there's only a single match, it's used
                    # If there are multiple matches, we raise an error because it's ambiguous

                    potential_devices = list(filter(None, [KNOWN_DEVICES[c].get(known_name) for c in Category]))
                    if not potential_devices:
                        raise RuntimeError(f"No known devices that match {known_name}")
                    elif len(potential_devices) > 1:
                        raise RuntimeError(f"Multiple devices known by name {known_name}, please specify category")
                    else:
                        device = potential_devices[0]
                else:
                    device = KNOWN_DEVICES[category].get(known_name)

                if not device:
                    if category != Category.GPU:    # TODO: GPUs characteristics are not fully fleshed out
                        raise Exception(
                            f"Unknown device name for {category}. If a new device has been added at runtime, _recompute_known_devices must be called"
                        )
                else:
                    for f in fields(device):
                        setattr(self, f.name, getattr(device, f.name))

        # override with user-specified values, if any
        # KEEP THIS SORTED
        self.architecture = architecture or self.architecture
        self.cache_lines = cache_lines or self.cache_lines
        self.cache_sizes = cache_sizes or self.cache_sizes
        self.category = category or self.category
        self.runtime = runtime or self.runtime
        self.extensions = extensions or self.extensions
        self.family = family or self.family
        self.frequency_GHz = frequency_GHz or self.frequency_GHz
        self.name = name or self.name
        self.num_cores = num_cores or self.num_cores
        self.num_threads = num_threads or self.num_threads
        self.tensor_core_info = tensor_core_info or self.tensor_core_info
        self.turbo_frequency_GHz = turbo_frequency_GHz or self.turbo_frequency_GHz
        self.vector_bytes = vector_bytes or self.vector_bytes
        self.vector_registers = vector_registers or self.vector_registers
        # TODO: inspect target characteristics of HOST rather than assuming these defaults
        if self.category == Target.Category.GPU:
            self.GridUnit = copy.deepcopy(GridUnits)
            self.MemorySpace = copy.deepcopy(_MemorySpace)

        # If known_name was provided, we should override the internal fields too
        if known_name:
            super().__post_init__()

    def _try_get_known_name(self, known_name):
        if known_name == "HOST":
            cpu_info = cpuinfo.get_cpu_info()

            # use regular expression to match the names in known devices with the model name from cpuinfo,
            # the regex looks like ^.*?\b[word1]\b.*?\b[word2]\b.*?\b[word3]\b.*? ... \b[wordN]\b.*?
            # for example, given name strings "Intel" and "W-2123", which can match the following string,
            # "Intel(R) Xeon(R) W-2123 CPU @ 3.60GHz", this is case insensitive match.
            for m in Model:
                name_info = re.split(r'\s', m.name)

                regex_match = r'^.*?'
                for info in name_info:
                    regex_match = regex_match + r'\b' + info + r'\b.*?'

                match = re.match(regex_match, cpu_info['brand_raw'], re.IGNORECASE)
                if match:
                    return m.name

            # print a warning for unknown host if nothing matched
            from termcolor import colored
            print(
                colored(
                    f"""Warning: Your host machine "{cpu_info['brand_raw']}" is not a known target model. To generate optimal code, we recommend that you inspect the accera.Target.Models enumeration to find the closest matching target model,
and create a Target using that model. You may also define a custom target if there is no closest match.
For more details please refer to this link: https://microsoft.github.io/Accera/Reference/classes/Target/Target/#known-device-names""",
                    'yellow'
                )
            )

        # in case a value from the Model enum is used
        return str(known_name)

    def is_compatible_with(self, other: "Target"):
        return all([
            self.name == other.name,
            self.category == other.category,
            self.runtime == other.runtime,
            self.architecture == other.architecture,
            all(e in other._full_extensions for e in self.extensions),
            other._max_frequency_GHz == 0 or self.frequency_GHz <= other._max_frequency_GHz,
            other._max_num_cores == 0 or self.num_cores <= other._max_num_cores,
            other._max_num_threads == 0 or self.num_threads <= other._max_num_threads,
            other._max_vector_bytes == 0 or self.vector_bytes <= other._max_vector_bytes,
            other._max_vector_registers == 0 or self.vector_registers <= other._max_vector_registers,
        ])


# for convenience
Target.HOST = Target()

# Amazing resource: https://en.wikichip.org
#         example: https://en.wikichip.org/wiki/intel/microarchitectures/coffee_lake
#
# Intel Turbo Boost -
# https://www.thinkwiki.org/wiki/Turbo_Boost
# https://www.tomshardware.com/features/intel-thermal-velocity-boost-glossary-definition-tvb
# https://www.tomshardware.com/reference/intel-favored-cpu-cores-turbo-boost-max-technology-3.0
#
# Raspberry Pi 4 8 GB Potential Speed Boost
# https://www.tomshardware.com/news/raspberry-pi-os-bullseye-released
# https://www.tomshardware.com/news/pi-4-gets-updated-soc
