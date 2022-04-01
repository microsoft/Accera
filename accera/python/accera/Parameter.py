####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import List, Callable
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


def create_parameter_grid(
    parameter_choices: dict, 
    filter_func: Callable = None, 
    sample: int = 0
) -> List[dict]:
    """
    Create a parameter grid from a dictionary that maps each parameter to its possible values,
    with/without a self-defined filter and the number of sample.

        Returns a list of a dictionary or a dictionary of {DelayedParameter: value}.

        Args:
            parameter_choices: A dictionary that maps each parameter to its possible values, e.g. 
                                        P0, P1, P2, P3, P4 = create_parameters(5)
                                        parameter_choices = {
                                            P0: [8, 16],
                                            P1: [16, 32],
                                            P2: [16],
                                            P3: [1.0, 2.0],
                                            P4: [3, 5, 7]
                                        }

            filter_func: A callable to filter parameter_choices which returns a bool to indicate whether a given parameter combination should be included in the grid.
            sample: A number to limit the number of parameter grid.
    """
    import itertools
    import random

    choices = []
    keys = []

    for key, value in parameter_choices.items():
        try:
            _ = iter(value)
        except TypeError:
            value = [value]
        choices.append(value)
        keys.append(key)

    choice_variants = itertools.product(*choices)

    filtered_choice_variants = list(filter(filter_func, choice_variants))
    if sample > 0 and sample < len(filtered_choice_variants):
        filtered_choice_variants = random.sample(filtered_choice_variants, sample)

    return [dict(zip(keys, variant)) for variant in filtered_choice_variants]
