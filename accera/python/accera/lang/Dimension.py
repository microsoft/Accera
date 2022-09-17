####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import *
from enum import Enum, auto
from varname import varname

from .._lang_python import ScalarType, _MemoryLayout
from .._lang_python._lang import Scalar, Allocate


class Dimension:
    "A dimension of a multi-dimensional array"

    class Role(Enum):
        "Defines the Dimension role"
        INPUT = (auto())   #:  An input dimension (immutable and provided as an Accera function argument).
        OUTPUT = auto()    #: An output dimension (mutable and updated by an Accera function).

    def __init__(
        self,
        name: str = None,
        role: "accera.Dimension.Role"=Role.INPUT,
        value: Union["Dimension", int] = None
    ):
        self._name = name
        self._native_dim = None
        self._role = role

        if value:
            if self._role != Dimension.Role.OUTPUT:
                raise ValueError("Only output dimension can accept the optional value to initialize itself")
            self._native_dim = value._native_dim if isinstance(value, Dimension) else Scalar(value)

        self._create_native_dim()

    @property
    def role(self):
        return self._role

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        return self._native_dim + (other._native_dim if isinstance(other, Dimension) else other)

    def __sub__(self, other):
        return self._native_dim - (other._native_dim if isinstance(other, Dimension) else other)

    def __mul__(self, other):
        return self._native_dim * (other._native_dim if isinstance(other, Dimension) else other)

    def __truediv__(self, other):
        return self._native_dim / (other._native_dim if isinstance(other, Dimension) else other)

    def __mod__(self, other):
        return self._native_dim % (other._native_dim if isinstance(other, Dimension) else other)

    def __rmul__(self, other):
        return self._native_dim * other

    def _create_native_dim(self):
        if not self._native_dim:
            # TODO: use ScalarDimension once it is aliased to Scalar(type=ScalarType.index)
            self._native_dim = Scalar(type=ScalarType.index)


def create_dimensions():
    try:
        names = varname(multi_vars=True)
        return (
            tuple([Dimension(name=n) for n in names])
            if len(names) > 1
            else Dimension(name=names[0])
        )
    except Exception as e:
        raise RuntimeError(
            "Caller didn't assign the return value(s) of create_dimensions() directly to any variable(s)"
        )
