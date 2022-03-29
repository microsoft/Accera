####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import List
from varname import varname

class DelayedParameter:
    def __init__(self, name=None):
        self._value = None
        self._name = name

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value


def create_parameters(count: int):
    if count < 1:
        raise ValueError("Invalid parameters count")
    names = varname(multi_vars=True)
    return (tuple([DelayedParameter(name) for name in names])
            if count > 1 else DelayedParameter(names[0]))


def get_parameters_from_grid(parameter_grid: dict) -> List[dict]:
    """Get a list of parameters combinations from the parameter grid.

    Args:
        parameter_grid: A set of different values for each parameter, which will be used to generate a list of all valid parameter combinations.
    """
    import itertools

    choices = []
    keys = []
    combinations_list = []

    for key, value in parameter_grid.items():
        try:
            _ = iter(value)
        except TypeError:
            value = [value]
        choices.append(value)
        keys.append(key)

    choice_variants = itertools.product(*choices)
    for variant in choice_variants:
        combinations_list.append(dict(zip(keys, variant)))

    return combinations_list
