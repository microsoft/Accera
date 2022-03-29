####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

try:
    from ._version import __version__
except:
    # CMake-driven builds do not generate _version.py yet
    __version__ = None
