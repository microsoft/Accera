####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class NativeLoopNestContext:
    from .._lang_python._lang import _Nest, _Schedule, _ExecutionPlan

    function_args: list
    runtime_args: list
    nest: _Nest = None
    schedule: _Schedule = None
    plan: _ExecutionPlan = None
    options: Any = None

    # a mapping that gets updated to keep track of python objects with their native counterparts
    # current is used to keep track of Value and LoopIndex instances
    mapping: Mapping[int, Any] = field(default_factory=dict)    # default_factory is needed because `dict` is a mutable
