####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from enum import Enum
from typing import Callable
import unittest

import numpy as np

from accera import ScalarType


class FailedReason(Enum):
    NOT_IN_CORE = "Not yet implemented (core)"
    NOT_IN_PY = "Not yet implemented (python)"
    UNKNOWN = "Unknown failure"
    BUG = "Bug"
    INVALID = "Invalid"


def expectedFailure(reason: FailedReason, msg: str, condition: bool = True) -> Callable:
    "Extends the unittest.expectedFailure decorator to print failure details and takes an optional condition"

    def _decorator(func):
        @unittest.expectedFailure
        def _wrapper(x):
            print(f"\n{reason.value}: {msg}")
            try:
                return func(x)
            except Exception as e:
                print(f"\t{e}\n")
                raise (e)

        return _wrapper if condition else func

    return _decorator


def get_type_str(datatype: ScalarType):
   return datatype.name


def accera_to_np_type(datatype: ScalarType):
    if datatype == ScalarType.float64:
        return np.float64

    if datatype == ScalarType.float32:
        return np.float32

    if datatype == ScalarType.float16:
        return np.float16

    if datatype == ScalarType.int32:
        return np.int32

    if datatype == ScalarType.int16:
        return np.int16

    if datatype == ScalarType.int8:
        return np.int8

    return None
