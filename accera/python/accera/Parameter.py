####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
import operator as ops
from typing import Any, List, Callable, Union
from varname import varname

# TODO: rename DelayedParameter to Parameter
class DelayedParameter:
    def __init__(
        self,
        name: str = None,
        operand1: Union["DelayedParameter", int, float] = None,
        operand2: Union["DelayedParameter", int, float] = None,
        operation: Callable[["DelayedParameter", "DelayedParameter"], Any] = None,
        possible_values: list = None,
    ):
        self._value = None
        self._name = name
        self._operand1 = operand1
        self._operand2 = operand2
        self._operation = operation
        self._possible_values = possible_values

    def get_value(self):
        """
        get_value method either returns the value of a normal DelayedParameter if it has been set,
        or works as the delay evaluation of resultant DelayedParameter calculated from arithmetic operation,
        it recusively calls the operand's get_value method, then does arithmetic operation with the value of the operands.
        """
        if self._operation:
            operand1 = (
                self._operand1.get_value()
                if isinstance(self._operand1, DelayedParameter)
                else self._operand1
            )
            operand2 = (
                self._operand2.get_value()
                if self._operand2 and isinstance(self._operand2, DelayedParameter)
                else self._operand2
            )
            self._value = (
                self._operation(operand1, operand2)
                if operand2
                else self._operation(operand1)
            )

        return self._value
 
    def get_possible_values(self):
        if self._possible_values != None:
            return self._possible_values
        else:
            raise Exception("Heuristic parameter must have a possible value")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def set_value(self, value):
        self._value = value

    def __add__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__add__)

    def __sub__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__sub__)

    def __mul__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__mul__)

    def __matmul__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__matmul__)

    def __truediv__(self, other):
        return DelayedParameter(
            operand1=self, operand2=other, operation=ops.__truediv__
        )

    def __floordiv__(self, other):
        return DelayedParameter(
            operand1=self, operand2=other, operation=ops.__floordiv__
        )

    def __mod__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__mod__)

    def __rmul__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__mul__)

    # TODO: __pow__ accepts an optional arg for modulo
    def __pow__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__pow__)

    def __lshift__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__lshift__)

    def __rshift__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__rshift__)

    def __and__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__and__)

    def __xor__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__xor__)

    def __or__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__or__)

    def __lt__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__lt__)

    def __le__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__le__)

    def __ne__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__ne__)

    def __ge__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__ge__)

    def __gt__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=ops.__gt__)

    def __divmod__(self, other):
        return DelayedParameter(operand1=self, operand2=other, operation=divmod)

    def __abs__(self):
        return DelayedParameter(operand1=self, operand2=None, operation=ops.__abs__)

    def __neg__(self):
        return DelayedParameter(operand1=self, operand2=None, operation=ops.__neg__)

    def __pos__(self):
        return DelayedParameter(operand1=self, operand2=None, operation=ops.__pos__)

    def __invert__(self):
        return DelayedParameter(operand1=self, operand2=None, operation=ops.__invert__)

    def __abs__(self):
        return DelayedParameter(operand1=self, operand2=None, operation=ops.__abs__)


def create_parameters():
    try:
        names = varname(multi_vars=True)
        return (
            tuple([DelayedParameter(name) for name in names])
            if len(names) > 1
            else DelayedParameter(names[0])
        )
    except Exception as e:
        raise RuntimeError(
            "Caller didn't assign the return value(s) of create_parameters() directly to any variable(s)"
        )

def create_and_set_parameters(possible_values: List, append_suffix_to_varnames: bool = False):
    name = varname()
    if append_suffix_to_varnames:
         name += token_hex(4)
    return DelayedParameter(name=name, possible_values=possible_values)

def create_parameter_grid(
    parameter_choices: dict, filter_func: Callable = None, sample: int = 0, seed=None
) -> List[dict]:
    """
    Create a parameter grid from a dictionary that maps each parameter to its possible values,
    with/without a self-defined filter and the number of sample.

        Returns a list of a dictionary or a dictionary of {DelayedParameter: value}.

        Args:
            parameter_choices: A dictionary that maps each parameter to its possible values, e.g.
                                        P0, P1, P2, P3, P4 = create_parameters()
                                        parameter_choices = {
                                            P0: [8, 16],
                                            P1: [16, 32],
                                            P2: [16],
                                            P3: [1.0, 2.0],
                                            P4: [3, 5, 7]
                                        }

            filter_func: A callable to filter parameter_choices which returns a bool to indicate whether a given parameter combination should be included in the grid.
            sample: A number to limit the number of parameter grid.
            seed: A number as the seed value for the generator to start with to generate a random number.
    """
    import itertools
    import random
    from .lang import LoopIndex

    choices = []
    keys = []

    for key, value in parameter_choices.items():
        try:
            _ = iter(value)
        except TypeError:
            value = [value]

        # if the parameter is a loop order, we permute the indices
        if all(isinstance(v, LoopIndex) for v in value):
            value = list(itertools.permutations(value, len(value)))

        choices.append(value)
        keys.append(key)

    choice_variants = itertools.product(*choices)

    filtered_choice_variants = list(filter(filter_func, choice_variants))
    if sample > 0 and sample < len(filtered_choice_variants):
        if seed:
            random.seed(seed)
        filtered_choice_variants = random.sample(filtered_choice_variants, sample)

    return [dict(zip(keys, variant)) for variant in filtered_choice_variants]

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
