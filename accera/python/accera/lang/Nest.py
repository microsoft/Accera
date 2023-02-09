####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import Callable, List, Union, Tuple
from functools import partial
from varname import varname

from .Array import Array
from .LoopIndex import LoopIndex
from .LogicFunction import logic_function, LogicFunction
from .NativeLoopNestContext import NativeLoopNestContext
from ..Parameter import DelayedParameter
from .. import Target
from .._lang_python import Role
from .._lang_python._lang import Scalar, Dimension, Array as NativeArray
from ..Constants import k_dynamic_size


class Nest:
    "Represents an iteration space"

    def __init__(self, shape: List[Union[int, DelayedParameter, Dimension]]):
        """Creates a Nest

        Args:
            shape: a list of dimensions
        """
        self._commands = []
        self._delayed_calls = {}
        self._logic_fns = []
        self._shape = [(dim, LoopIndex(self)) for dim in shape]

        if any([isinstance(s, DelayedParameter) for s in shape]):
            self._delayed_calls[partial(self._init_delayed)] = tuple([s for s in shape])

    def create_schedule(self) -> "accera.Schedule":
        "Creates a schedule for shaping the iteration space"

        from .Schedule import Schedule

        return Schedule(self)

    def create_plan(self, target: "accera.Target" = Target.HOST) -> "accera.Plan":
        """Creates a plan using the default schedule for the nest.

        Args:
            target: Optional target specification. Defaults to the HOST
        """

        return self.create_schedule().create_plan(target)

    def get_shape(self) -> List[int]:
        """Gets the iteration space extents

        Returns:
            A list of extents (N-dimensional iteration space)
        """
        return [idx for idx, _ in self._shape]

    def get_indices(self) -> Union[List[LoopIndex], LoopIndex]:
        """Gets the iteration space indices

        Returns:
            A list of indices (multi-dimensional iteration space)
            or a single index (1-dimensional iteration space)
        """
        try:
            names = varname(multi_vars=True)
        except:
            names = None

        if names:
            indices = [idx for _, idx in self._shape]
            if len(self._shape) > 1:
                zipped_name_index = zip(names, indices)
                for name, index in zipped_name_index:
                    if not index.name:
                        index.name = name
            else:
                self._shape[0][1].name = names[0]

        return (
            [idx for _, idx in self._shape]
            if len(self._shape) > 1
            else self._shape[0][1]
        )

    def iteration_logic(self, logic: Callable = None, predicate=None, placement=None):
        """Adds iteration logic to the nest

        Args:
            logic: Python function that represents the logic to run in the innermost loop of the nest.
            predicate: The predicate that determine when the logic code should run.
            placement: The predicate that determines where the logic code should be placed.

        Remarks: this can be invoked as a decorator, where the logic function will be the first argument:

            @nest.iteration_logic
            def _():
                # logic function implementation

            The decorator invocation pattern only applies when the additional arguments are using the default
            values. To use non-default values (for `predicate`, for example), call this like a standard
            method:

            def fn():
                # logic function implementation

            nest.iteration_logic(fn, predicate=my_predicate)

        """
        wrapped_logic = logic_function(logic)
        self._logic_fns.append(wrapped_logic)

        self._commands.append(
            partial(self._add_iteration_logic, wrapped_logic, predicate, placement)
        )

    def _add_iteration_logic(
        self, logic_fn, pred, placement, context: NativeLoopNestContext
    ):
        from .._lang_python._lang import _Logic

        captures_to_replace = self._get_captures_to_replace(logic_fn, context)

        def logic_fn_wrapper():
            logic_fn(**captures_to_replace)

        logic = _Logic(logic_fn.__name__, logic_fn_wrapper)

        context.schedule.add_kernel(logic, pred, placement)

    def _get_captures_to_replace(self, logic_fn, context: NativeLoopNestContext):
        import types

        captures_to_replace = {}
        
        for k, v in logic_fn.get_captures().items():
            value_id = id(v)
            if value_id in context.mapping:
                captures_to_replace[k] = context.mapping[value_id]
            else:
                from .._lang_python._lang import Allocate
                if isinstance(v, Array):
                    from .._lang_python import _ResolveConstantDataReference

                    if v.role == Role.TEMP:
                        runtimeSizes = [x for x in v._shape if isinstance(x, Dimension)]
                        temp_array = NativeArray(
                            Allocate(type=v.element_type, layout=v.layout, flags=v.flags, runtimeSizes=runtimeSizes)
                        )
                        captures_to_replace[k] = context.mapping[value_id] = temp_array
                    elif v.role == Role.CONST:
                        const_ref_array = NativeArray(
                            _ResolveConstantDataReference(v._value)
                        )
                        captures_to_replace[k] = context.mapping[
                            value_id
                        ] = const_ref_array
                    continue
                elif isinstance(v, (Scalar, Dimension)) and v.role == Role.TEMP:
                    temp_scalar = Scalar(Allocate(type=v.type), role=v.role)
                    captures_to_replace[k] = context.mapping[value_id] = temp_scalar
                    continue
                elif isinstance(v, LoopIndex):
                    continue
                elif isinstance(v, DelayedParameter):
                    captures_to_replace[k] = v.get_value()
                    continue
                elif isinstance(v, types.FunctionType):
                    captures_to_replace.update(self._get_captures_to_replace(logic_function(v), context))
                    continue

                try:
                    it = iter(v)
                except TypeError:
                    continue
                else:
                    replaced_values = []
                    is_scalar = False
                    for elem in it:
                        if isinstance(elem, Scalar):
                            is_scalar = True
                            break
                        replacement = context.mapping.get(id(elem))
                        if replacement:
                            replaced_values.append(replacement)
                    if not is_scalar:
                        captures_to_replace[k] = tuple(replaced_values)

        return captures_to_replace

    def _build_native_context(self, context: NativeLoopNestContext):
        from .._lang_python._lang import _Nest

        try:
            args_iter = iter(*context.runtime_args)
        except TypeError:
            args_iter = iter(context.runtime_args)

        logic_args = {}
        for x, y in zip(context.function_args, args_iter):
            if isinstance(x, Array):
                logic_args[id(x)] = NativeArray(y)
            elif isinstance(x, Dimension):
                x._value = y
                logic_args[id(x)] = Scalar(y, name=x.name, role=x.role)
            elif isinstance(x, Scalar):
                logic_args[id(x)] = Scalar(y, role=x.role)
            else:
                logic_args[id(x)] = y

        nest_shape = [k_dynamic_size if isinstance(x, Dimension) else x for x, _ in self._shape]
        nest_rt_sizes = [x for x, _ in self._shape if isinstance(x, Dimension)]
        context.nest = _Nest(shape=nest_shape, runtime_sizes=nest_rt_sizes)

        native_indices = context.nest.get_indices()

        # fake index => native index
        index_handles_to_native_index = dict(
            zip([id(x) for _, x in self._shape], native_indices)
        )
        logic_args.update(index_handles_to_native_index)

        context.mapping.update(logic_args)

    def _build_with_native_context(self, context: NativeLoopNestContext):
        for cmd in self._commands:
            cmd(context)

    def _init_delayed(self, shape: List[int]):
        resolved_shape = [(shape[i], self._shape[i][1]) for i in range(0, len(shape))]
        self._shape = resolved_shape

    def _replay_delayed_calls(self):
        """
        This method is called once per adding function, so it can be called multiple times when
        multiple functions get added. In order for the functions to be added correctly, we need to make sure all
        the residual states are cleared between different method calls.

        For example, in Schedule class, we identify that Schedule._index_map can have residual states, so we need to reset self._index_map
        before we replay the delayed methods.

        If there is no residual state between different method calls, no need to reset.
        """
        for delayed_call in self._delayed_calls:
            params = self._delayed_calls[delayed_call]

            if isinstance(params, Tuple):
                resolved_param_list = []
                for p in params:
                    if isinstance(p, DelayedParameter):
                        resolved_param_list.append(p.get_value())
                    else:
                        resolved_param_list.append(p)
                delayed_call(resolved_param_list)
            else:
                delayed_call(params.get_value())

    def get_logic(self) -> List[LogicFunction]:
        return self._logic_fns
