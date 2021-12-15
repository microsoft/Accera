####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import List, Any
from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass
class _TargetContainer:
    "Generic container class for target information"
    architecture: Any = None
    cache_lines: List[int] = field(default_factory=list)
    cache_sizes: List[int] = field(default_factory=list)
    category: Any = None
    default_block_size: int = 0
    extensions: List[str] = field(default_factory=list)
    family: str = ""
    frequency_GHz: float = 0.0
    name: str = ""
    num_cores: int = 0
    num_threads: int = 0
    turbo_frequency_GHz: float = 0.0
    vector_bytes: int = 0
    vector_registers: int = 0

    _device_name: str = "host"    # used internally for emitting known targets

    _full_extensions: List[str] = field(default_factory=list)

    _max_frequency_GHz: float = 0.0
    _max_num_cores: int = 0
    _max_num_threads: int = 0
    _max_turbo_frequency_GHz: float = 0.0
    _max_vector_bytes: int = 0
    _max_vector_registers: int = 0

    def __post_init__(self):
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

        return _VectorizationInfo(vector_bytes=self.vector_bytes,
                                  vector_units=self.vector_registers,
                                  unroll_only=False)


class Target(_TargetContainer):
    "Factory-like class for target information"

    class Model(Enum):
        HOST = auto()
        INTEL_CORE_GENERATION_7 = auto()
        INTEL_CORE_GENERATION_8_AND_9 = auto()
        INTEL_CORE_GENERATION_10 = auto()
        INTEL_CORE_GENERATION_11 = auto()
        INTEL_CORE_GENERATION_12 = auto()
        NVIDIA_TESLA_V100 = auto()
        RASPBERRY_PI3 = auto()
        RASPBERRY_PI0 = auto()

    class Category(Enum):
        CPU = auto()
        GPU = auto()

    class Architecture(Enum):
        HOST = auto()
        ARM = auto()
        X86 = auto()
        X86_64 = auto()
        # AARCH64 = auto()

    model: Model

    def __init__(self,
                 model: Model = Model.HOST,
                 category: Category = None,
                 architecture: Architecture = None,
                 name: str = None,
                 family: str = None,
                 extensions: List[str] = None,
                 num_threads: int = None,
                 num_cores: int = None,
                 vector_bytes: int = 0,
                 vector_registers: int = None,
                 frequency_GHz: float = None,
                 turbo_frequency_GHz: float = None,
                 cache_sizes: List[int] = None,
                 cache_lines: List[int] = None,
                 default_block_size: int = None):
        "Factory-like constructor that uses the model parameter to fill-in known defaults"

        self.model = model

        # initialize known defaults based on the model
        if self.model == Target.Model.HOST:
            super().__init__(
                category=Target.Category.CPU,
                architecture=Target.Architecture.HOST,
                vector_bytes=32,  # There are 32-bytes per full SIMD register
                vector_registers=16,  # There are 16 YMM registers
                extensions=["MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE4", "SSE4.1", "SSE4.2", "AVX", "AVX2", "FMA3"])

        elif self.model == Target.Model.INTEL_CORE_GENERATION_7:
            super().__init__(
                category=Target.Category.CPU,
                architecture=Target.Architecture.X86_64,
                family="Skylake-X",
                vector_bytes=32, # There are 32-bytes per SIMD register
                vector_registers=16, # There are 16 YMM registers
                extensions=["SSE4.1", "SSE4.2", "AVX2"],
                _device_name="avx512")

        elif self.model == Target.Model.INTEL_CORE_GENERATION_8_AND_9:
            super().__init__(
                category=Target.Category.CPU,
                architecture=Target.Architecture.X86_64,
                family="Coffee Lake",
                vector_bytes=32,
                vector_registers=16,
                extensions=["SSE4.1", "SSE4.2", "AVX2"],
                _device_name="avx512")

        elif self.model == Target.Model.INTEL_CORE_GENERATION_10:
            super().__init__(
                category=Target.Category.CPU,
                architecture=Target.Architecture.X86_64,
                family="Comet Lake",
                vector_bytes=32,
                vector_registers=16,
                extensions=["SSE4.1", "SSE4.2", "AVX2"],
                _device_name="avx512")

        elif self.model == Target.Model.INTEL_CORE_GENERATION_11:
            super().__init__(
                category=Target.Category.CPU,
                architecture=Target.Architecture.X86_64,
                family="Rocket Lake",
                vector_bytes=64, # Each SIMD register is 512 bits (64 bytes) wide
                vector_registers=32, # There are 32 ZMM registers
                extensions=["SSE4.1", "SSE4.2", "AVX2", "AVX512"],
                _device_name="avx512")

        elif self.model == Target.Model.INTEL_CORE_GENERATION_12:
            super().__init__(
                category=Target.Category.CPU,
                architecture=Target.Architecture.X86_64,
                family="Alder Lake",
                vector_bytes=32,
                vector_registers=16,
                extensions=["SSE4.1", "SSE4.2", "AVX2"],
                _device_name="avx512")

        elif self.model == Target.Model.RASPBERRY_PI3:
            super().__init__(category=Target.Category.CPU,
                             architecture=Target.Architecture.ARM,
                             family="Broadcom BCM2837B0",
                             num_cores=4,
                             num_threads=8,
                             frequency_GHz=1.4,
                             _device_name="pi3")

        elif self.model == Target.Model.RASPBERRY_PI0:
            super().__init__(category=Target.Category.CPU,
                             architecture=Target.Architecture.ARM,
                             family="Broadcom BCM2835",
                             num_cores=1,
                             num_threads=2,
                             frequency_GHz=1.0,
                             _device_name="pi0")

        elif self.model == Target.Model.NVIDIA_TESLA_V100:
            super().__init__(category=Target.Category.GPU, default_block_size=16)

        # override with user-specified values, if any
        # KEEP THIS SORTED
        self.architecture = architecture or self.architecture
        self.cache_lines = cache_lines or self.cache_lines
        self.cache_sizes = cache_sizes or self.cache_sizes
        self.category = category or self.category
        self.default_block_size = default_block_size or self.default_block_size
        self.extensions = extensions or self.extensions
        self.family = family or self.family
        self.frequency_GHz = frequency_GHz or self.frequency_GHz
        self.name = name or self.name
        self.num_cores = num_cores or self.num_cores
        self.num_threads = num_threads or self.num_threads
        self.turbo_frequency_GHz = turbo_frequency_GHz or self.turbo_frequency_GHz
        self.vector_bytes = vector_bytes or self.vector_bytes
        self.vector_registers = vector_registers or self.vector_registers

    def is_compatible_with(self, other: "Target"):
        return all([
            self.name == other.name,
            self.category == other.category,
            self.architecture == other.architecture,
            all(e in other._full_extensions for e in self.extensions),
            other._max_frequency_GHz == 0 or self.frequency_GHz <= other._max_frequency_GHz,
            other._max_num_cores == 0 or self.num_cores <= other._max_num_cores,
            other._max_num_threads == 0 or self.num_threads <= other._max_num_threads,
            other._max_turbo_frequency_GHz == 0 or self.turbo_frequency_GHz <= other._max_turbo_frequency_GHz,
            other._max_vector_bytes == 0 or self.vector_bytes <= other._max_vector_bytes,
            other._max_vector_registers == 0 or self.vector_registers <= other._max_vector_registers,
        ])


# for convenience
Target.HOST = Target(model=Target.Model.HOST)
