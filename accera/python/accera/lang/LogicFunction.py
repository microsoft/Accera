####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from functools import singledispatch, partialmethod
from inspect import getclosurevars

from .Array import Array
from .LoopIndex import LoopIndex


class LogicFunction:
    _SPECIAL_TYPES = (Array, LoopIndex)

    def __init__(self, func):
        self.func = func
        self.__name__ = self.func.__name__
        vars = getclosurevars(func)
        self.func_globals = vars.globals
        self.func_nonlocals = vars.nonlocals

    def get_captures(self, _type=None):
        args_dict = {}
        for d in [self.func_globals, self.func_nonlocals]:
            for k, v in d.items():
                if isinstance(v, ((_type, ) if _type else ()) + LogicFunction._SPECIAL_TYPES):
                    args_dict[k] = v
                    continue

                try:
                    it = iter(v)
                except TypeError:
                    if not _type or isinstance(v, _type):
                        args_dict[k] = v
                else:
                    if not _type or all(isinstance(elem, _type) for elem in it):
                        args_dict[k] = v
        return args_dict

    get_args = partialmethod(get_captures, Array)
    get_indices = partialmethod(get_captures, LoopIndex)

    def __call__(self, **kwargs):
        # get the mapping of global and nonlocal variable captures used by the function
        globals_to_update = [k for k in kwargs if k in self.func_globals]
        nonlocals_to_update = [k for k in kwargs if k in self.func_nonlocals]
        # make sure there's no overlap between the keys. might be impossible to hit
        assert set(globals_to_update).isdisjoint(nonlocals_to_update)

        # for both sets of dictionaries, store their current values
        current_globals = {}
        current_nonlocals = {}
        for k in globals_to_update:
            # func.__globals__ is a dictionary of the globals from the module of func
            # it exists at least as of CPython3.7
            current_globals[k] = self.func.__globals__[k]
            self.func.__globals__[k] = kwargs[k]

        # func.__code__ is the byte compiled code object of func
        # co_freevars is the  of nonlocal captured variables that need values ("free variables")
        # __closure__ is the list of values to these free variables, in the same order
        # their state of existence is locked for our purposes
        assert bool(self.func.__code__.co_freevars) == bool(self.func.__closure__)
        for k, v in zip(self.func.__code__.co_freevars, self.func.__closure__ or []):

            if k not in nonlocals_to_update:
                continue

            # these values are "Cell" types, with the property "cell_contents" which holds the actual value
            # so you can replace it on the fly
            current_nonlocals[k] = v.cell_contents
            v.cell_contents = kwargs[k]

        try:
            # run the function in a try block so that no matter what, we undo our changes
            self.func()
        except Exception as e:
            raise e
        finally:
            # need to create proper unwinding infra

            # we undo the changes that we made to the func object
            for k, v in zip(self.func.__code__.co_freevars, self.func.__closure__ or []):
                if k not in nonlocals_to_update:
                    continue

                v.cell_contents = current_nonlocals[k]

            for k in globals_to_update:
                self.func.__globals__[k] = current_globals[k]


@singledispatch
def logic_function(func):
    return LogicFunction(func)


@logic_function.register
def _(func: LogicFunction):
    return func
