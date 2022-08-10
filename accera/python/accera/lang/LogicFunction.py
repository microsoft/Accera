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
                if isinstance(
                    v, ((_type,) if _type else ()) + LogicFunction._SPECIAL_TYPES
                ):
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
        current_globals, current_nonlocals = self._update_globals_locals(**kwargs)

        try:
            # run the function in a try block so that no matter what, we undo our changes
            self.func()
        except Exception as e:
            raise e
        finally:
            # need to create proper unwinding infra
            self._undo_update_globals_locals(current_globals, current_nonlocals, **kwargs)

    def _update_globals_locals(self, **kwargs):
        import types

        # for both sets of dictionaries, store their current values
        current_globals = {}
        current_nonlocals = {}

        # get the mapping of global and nonlocal variable captures used by the function
        globals_to_update = [k for k in kwargs if k in self.func_globals]
        nonlocals_to_update = [k for k in kwargs if k in self.func_nonlocals]
        
        # make sure there's no overlap between the keys. might be impossible to hit
        assert set(globals_to_update).isdisjoint(nonlocals_to_update)

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
            if isinstance(v.cell_contents, types.FunctionType):
                v_func = logic_function(v.cell_contents)
                globals, nonlocals = v_func._update_globals_locals(**kwargs)
                current_globals.update(globals)
                current_nonlocals.update(nonlocals)
                continue
            if k not in nonlocals_to_update:
                continue

            # these values are "Cell" types, with the property "cell_contents" which holds the actual value
            # so you can replace it on the fly
            if id(v.cell_contents) != id(kwargs[k]):
                current_nonlocals[k] = v.cell_contents
                v.cell_contents = kwargs[k]
        
        return current_globals, current_nonlocals


    def _undo_update_globals_locals(self, current_globals, current_nonlocals, **kwargs):
        import types

        # get the mapping of global and nonlocal variable captures used by the function
        globals_to_update = [k for k in kwargs if k in self.func_globals]
        nonlocals_to_update = [k for k in kwargs if k in self.func_nonlocals]

        # we undo the changes that we made to the func object
        for k, v in zip(self.func.__code__.co_freevars, self.func.__closure__ or []):
            if isinstance(v.cell_contents, types.FunctionType):
                v_func = logic_function(v.cell_contents)
                v_func._undo_update_globals_locals(current_globals, current_nonlocals, **kwargs)
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
