####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import *
from varname import varname

from .._lang_python import Role
from .._lang_python._lang import Dimension

def create_dimensions(role=Role.INPUT):
    try:
        names = varname(multi_vars=True)
        return (
            tuple([Dimension(name=n, role=role) for n in names])
            if len(names) > 1
            else Dimension(name=names[0], role=role)
        )
    except Exception as e:
        raise RuntimeError(
            "Caller didn't assign the return value(s) of create_dimensions() directly to any variable(s)"
        )
