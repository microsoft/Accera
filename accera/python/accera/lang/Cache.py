####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from dataclasses import dataclass
from typing import Any, Tuple, Union

from .Array import Array
from .LoopIndex import LoopIndex
from .._lang_python._lang import (
    CacheIndexing, _CacheAllocation, _CacheStrategy, _MemorySpace, _MemoryAffineCoefficients, _DimensionOrder
)


@dataclass
class Cache:
    plan: "accera.Plan"
    target: Union[Array, Any]
    index: LoopIndex = None
    trigger_index: LoopIndex = None
    level: int = None
    trigger_level: int = None
    element_type: "accera.ScalarType" = None
    layout: Union[Array.Layout, Tuple[int]] = None
    max_elements: int = None
    thrifty: bool = False
    double_buffer: bool = False
    double_buffer_location: _MemorySpace = _MemorySpace.NONE
    offset: int = 0
    native_cache: Any = None
    vectorize: bool = False
    location: _MemorySpace = _MemorySpace.NONE
    strategy: _CacheStrategy = _CacheStrategy.STRIPED
    indexing: CacheIndexing = CacheIndexing.GLOBAL_TO_PHYSICAL
    allocation: _CacheAllocation = _CacheAllocation.AUTO
    shared_memory_offset: int = 0

    @property
    def target_shape(self):
        if isinstance(self.target, Cache):
            return self.target.target_shape
        else:
            return self.target.shape
    @property
    def target_role(self):
        if isinstance(self.target, Cache):
            return self.target.target_role
        else:
            return self.target.role

    @property
    def memory_map(self):
        if isinstance(self.layout, tuple):
            from .Layout import MemoryMapLayout

            mmap_layout = MemoryMapLayout(self.layout, self.target_shape, self.offset)
            return _MemoryAffineCoefficients(mmap_layout.coefficients, mmap_layout.offset)
        return None

    @property
    def dimension_permutation(self):
        if isinstance(self.layout, Array.Layout) and self.layout is not Array.Layout.DEFERRED:
            first_major = list(range(len(self.target_shape)))
            dim_orders = {
                Array.Layout.FIRST_MAJOR: first_major,
                Array.Layout.LAST_MAJOR: list(reversed(first_major)),
            }
            return _DimensionOrder(dim_orders[self.layout])
        return None


@dataclass
class DelayedCache(Cache):
    completed: bool = False
    enqueue_command: bool = True

    def complete(self, cache: Cache):
        self.index = cache.index
        self.trigger_index = cache.trigger_index
        self.level = cache.level
        self.trigger_level = cache.trigger_level
        self.element_type = cache.element_type
        self.layout = cache.layout
        self.max_elements = cache.max_elements
        self.thrifty = cache.thrifty
        self.double_buffer = cache.double_buffer
        self.double_buffer_location = cache.double_buffer_location
        self.offset = cache.offset
        self.native_cache = cache.native_cache
        self.vectorize = cache.vectorize
        self.location = cache.location
        self.indexing = cache.indexing
        self.allocation = cache.allocation

        self.completed = True
