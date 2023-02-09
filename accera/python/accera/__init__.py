####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

try:
    from ._version import __version__
except:
    # CMake-driven builds do not generate _version.py yet
    __version__ = None

from .Targets import Target, KNOWN_DEVICES, KNOWN_CPUS, KNOWN_GPUS
from .Parameter import DelayedParameter, create_parameters, create_parameter_grid
from .Constants import *
from .Package import Package

from .lang import *
from ._lang_python import CompilerOptions, ScalarType, _GetTargetDeviceFromName, AllocateFlags, Role
from ._lang_python import (
    abs, max, min, ceil, floor, sqrt, exp, log, log10, log2, sin, cos, tan, sinh, cosh, tanh, logical_and, logical_or,
    logical_not, cast, round, remainderf
)

# Global initialization
Package._init_default_module()
