####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Union


class Layout(Enum):
    "Defines a standard Array layout"

    #: Specifies a memory layout where the first major axis is in contiguous memory. For example, in a matrix, this corresponds to "row-major".
    FIRST_MAJOR = auto()

    #: Specifies a memory layout where the last major axis is in contiguous memory. For example, in a matrix, this corresponds to "column-major".
    LAST_MAJOR = auto()

    #: Defer specifying the memory layout for a `Role.CONST` array until a cache is created.
    DEFERRED = auto()

    def to_numpy_order(self):
        mapping = {
            Layout.FIRST_MAJOR:
                "C",    # Numpy uses "C" for C-like arrays where the last logical dimension is the fastest moving in memory
            Layout.LAST_MAJOR:
                "F"    # Numpy uses "F" for Fortran-like arrays where the first logical dimension is the fastest moving in memory
        }
        return mapping.get(
            self, None
        )    # All mappings except for first-major and last-major don't have a numpy default setting and must use manually-specified strides


def get_coefficients_for_layout(layout: Layout, shape: Tuple[int]):
    from functools import reduce
    from operator import mul

    if layout == Layout.DEFERRED:
        raise (ValueError("No coefficients available for Layout.DEFERRED"))

    # An affine memory map computes a scalar index into flattened memory (s_index) by
    # performing a vector dot product of the shape vector (v_shape)
    # with the affine memory map (v_memory_map). The general formula is:
    #
    #     s_index = v_shape.dot(v_memory_map) + s_offset
    # where s_offset=0 in the current implementation.
    #
    # Given a 4-dimensional shape vector v_shape = (s0, s1, s2, s3):
    #    First-major: (s0xs1xs2, s0xs1, s2, 1)
    #    Last-major: (1, s0, s0xs1, s0xs1xs2)
    # In both cases, the last dimension (s3) is not used in computing the affine memory map.
    ndim = len(shape)
    coeffs = [1] + [reduce(mul, shape[:i]) for i in range(1, ndim)]    # last major
    return coeffs[::-1] if layout == Layout.FIRST_MAJOR else coeffs


@dataclass(frozen=True)
class MemoryMapLayout:
    layout: Union[Tuple[int], Layout]
    shape: Tuple[int]
    offset: int = 0

    @property
    def coefficients(self):
        return get_coefficients_for_layout(self.layout, self.shape) if isinstance(self.layout, Layout) else self.layout

    @property
    def order(self):
        result = None
        ndim = len(self.shape)
        if isinstance(self.layout, Tuple):
            # infer the layout from coefficients
            first_major_coeffs = get_coefficients_for_layout(Layout.FIRST_MAJOR, self.shape)
            if self.layout == first_major_coeffs:
                result = range(ndim)
            elif self.layout == first_major_coeffs[::-1]:
                result = range(ndim)[::-1]
        elif isinstance(self.layout, Layout):
            if self.layout == Layout.FIRST_MAJOR:
                result = range(ndim)
            elif self.layout == Layout.LAST_MAJOR:
                result = range(ndim)[::-1]
        return result
