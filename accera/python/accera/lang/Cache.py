####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from dataclasses import dataclass
from typing import Any, Tuple, Union
from .Array import Array
from .LoopIndex import LoopIndex
from .._lang_python._lang import CacheIndexing, _CacheAllocation, _MemorySpace, _MemoryAffineCoefficients, _DimensionOrder


@dataclass(frozen=True)
class Cache:
    plan: "accera.ActionPlan"
    target: Array
    index: LoopIndex = None
    trigger_index: LoopIndex = None
    layout: Union[Array.Layout, Tuple[int]] = None
    max_elements: int = None
    thrifty: bool = False
    location: Any = None    # TODO - MemoryLocation, reconcile with _MemorySpace
    offset: int = 0
    memory_space: _MemorySpace = _MemorySpace.NONE
    indexing: CacheIndexing = CacheIndexing.GLOBAL_TO_PHYSICAL
    allocation: _CacheAllocation = _CacheAllocation.AUTO

    @property
    def memory_map(self):
        if isinstance(self.layout, tuple):
            from .Layout import MemoryMapLayout

            mmap_layout = MemoryMapLayout(self.layout, self.target.shape, self.offset)
            return _MemoryAffineCoefficients(mmap_layout.coefficients, mmap_layout.offset)
        return None

    @property
    def dimension_permutation(self):
        if isinstance(self.layout, Array.Layout) and self.layout is not Array.Layout.DEFERRED:
            first_major = list(range(len(self.target.shape)))
            dim_orders = {
                Array.Layout.FIRST_MAJOR: first_major,
                Array.Layout.LAST_MAJOR: list(reversed(first_major)),
            }
            return _DimensionOrder(dim_orders[self.layout])
        return None
