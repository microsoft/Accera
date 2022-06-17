####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import Union

from ..lang import *
from ..lang.Array import *
from ..lang.Layout import *
from ..lang.LoopIndex import *
from ..lang.Plan import *
from ..Parameter import *

from enum import Flag, Enum, auto

class NoneCacheHeuristics:
    "None cache heuristics is a brute force approach to synthesize cache arguments with a parameterized list of values."

    def __init__(
        self,
        source: Array = None,
        index: LoopIndex = None,
        layout: Array.Layout = None
    ):
        self._source = source
        self._index = index
        self._layout = layout
        self._params_list = []

    # This is an interface function between an execution plan
    # and a hueristic. It creates a list of parameters with possible values.
    def create_parameterized_args(self, plan):
        # Pass the handle for plan to heuristics object
        self._plan = plan

        if self._layout is None:
            layout_values = [i for i in Layout if i != Array.Layout.DEFERRED]
            layout_params = create_and_set_parameters(layout_values)
            self._layout = layout_params
            self._params_list.append(layout_params)
        if self._index is None:
            index_values = self._plan._sched._indices
            index_params = create_and_set_parameters(index_values)
            self._index = index_params
            self._params_list.append(index_params)

    def invoke_cache_dsl_command(self, plan):
        plan.cache(self._source, index = self._index, layout = self._layout)
