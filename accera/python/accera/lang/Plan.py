####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import *
from functools import partial

from ..Parameter import DelayedParameter

from .Array import Array
from .Schedule import Schedule, IndexTransform
from .Cache import Cache
from .Function import Function
from .LoopIndex import LoopIndex
from .NativeLoopNestContext import NativeLoopNestContext
from ..Targets import Target

from .._lang_python._lang import CacheIndexing

class ActionPlan:
    def __init__(self, schedule: Schedule, target: Target = Target.HOST):
        self._sched = schedule
        self._target = target
        self._commands = []
        self._delayed_calls = {}
        self._index_attrs: Mapping[LoopIndex, List[str]] = {}

    def _add_index_attr(self, index: LoopIndex, attr: str):
        attrs = self._index_attrs.get(index, [])
        attrs.append(attr)
        self._index_attrs[index] = attrs

    def print(self):
        self._sched.print(lambda index: self._index_attrs.get(index, []))

    def unroll(self, index: Union[LoopIndex, DelayedParameter]):
        """Unrolls the loop along a dimension

        Args:
            index: The dimension to unroll
        """
        if isinstance(index, DelayedParameter):
            self._delayed_calls[partial(self.unroll)] = index
            return None

        self._add_index_attr(index, "unrolled")
        self._commands.append(partial(self._unroll, index))

    def _unroll(self, index, context: NativeLoopNestContext):
        native_index = context.mapping[id(index)]

        # TODO: Move to final location depending on where unroll should be
        context.schedule.unroll(native_index)

    def vectorize(self, index: Union[LoopIndex, DelayedParameter]):
        """Only available for targets that have SIMD registers and support vector instructions. Marks a dimension of the iteration-space for vectorization.
        Args:
            index: The index to vectorize
        """
        if isinstance(index, DelayedParameter):
            self._delayed_calls[partial(self.vectorize)] = index
            return None

        if not self._target.vectorization_info:
            raise RuntimeError("The target does not support vectorization")

        self._add_index_attr(index, "vectorized")

        self._commands.append(
            partial(self._vectorize, index, self._target.vectorization_info)
        )

    def _vectorize(self, index, vectorization_info, context: NativeLoopNestContext):
        context.plan.vectorize(context.mapping[id(index)], vectorization_info)

    def parallelize(self, indices: Union[LoopIndex, Tuple[LoopIndex], DelayedParameter], pin: Union[Tuple[Any], DelayedParameter]=None, policy: Union[str, DelayedParameter]="static"):
        """Performs one or more loops in parallel on multiple cores or processors.
        Only available for targets with multiple cores or processors. 

        Args:
            indices: The iteration-space dimensions to run in parallel. 
            pin: Pin the computation to a subset of cores or processors.
            policy: The scheduling policy to apply ("dynamic" or "static").
        """
        if isinstance(indices, DelayedParameter) or isinstance(pin, DelayedParameter) or isinstance(policy, DelayedParameter):
            self._delayed_calls[partial(self.parallelize)] = {"indices" : indices, "pin" : pin, "policy" : policy}
            return None

        indices = [indices] if isinstance(indices, LoopIndex) else list(indices)

        # ensure the indices are contiguous and follow the Schedule ordering
        start = self._sched._indices.index(indices[0])
        end = start + len(indices)
        if end > len(self._sched._indices) or indices != self._sched._indices[start:end]:
            raise ValueError("indices must be contiguous in the Schedule dimension order")

        for index in indices:
            self._add_index_attr(index, "parallelized")

        self._commands.append(
            partial(self._parallelize, indices, policy, self._target.num_threads)
        )

    def _parallelize(self, indices, policy, num_threads, context: NativeLoopNestContext):
        from .._lang_python._lang import _ParallelizationPolicy
        idxs = [context.mapping[id(index)] for index in indices]
        context.plan.parallelize(idxs, num_threads, 
            _ParallelizationPolicy.DYNAMIC if policy == "dynamic" else _ParallelizationPolicy.STATIC)

    def cache(self,
              source: Union[Array, Cache],
              index: Union[LoopIndex, DelayedParameter] = None,
              trigger_index: Union[LoopIndex, DelayedParameter] = None,
              layout: Array.Layout = None,
              max_elements: int = None,
              thrifty: bool = None,
              location: Any = None,
              level: Union[int, DelayedParameter] = None,
              trigger_level: Union[int, DelayedParameter] = None):
        """Adds a cache for a view target

        Args:
            source: The array or cache from which this cache is copied.
            index: The index used to determine the cache level. Specify one and only one of `index`, `level`, `max_elements`.
            trigger_index: The index used to determine what level to fill the cache at. `trigger_index` can't come after `index` in the schedule order, and will default to `index` if not specified. Specify at most one of `trigger_index` or `trigger_level`.
            layout: The affine memory map, if different from the source.
            level: The key-slice level to cache (the number of wildcard dimensions in a key-slice). Specify one and only one of `index`, `level`, `max_elements`.
            trigger_level: The key-slice level to fill the cache at. `trigger_level` can't be smaller than `level`, and will default to `level` if not specified. Specify at most one of `trigger_index` or `trigger_level`.
            max_elements: The maximum elements to include in the cached region. Specify one and only one of `index`, `level`, `max_elements`.
            thrifty: Use thrifty caching (copy data into a cache only if the cached data differs from the original active block).
        """
        if any([isinstance(arg, DelayedParameter) for arg in (index, trigger_index, level, trigger_level)]):
            self._delayed_calls[
                partial(
                    self.cache,
                    source=source,
                    layout=layout,
                    max_elements=max_elements,
                    thrifty=thrifty,
                    location=location
                )
            ] = {
                    "index" : index,
                    "trigger_index" : trigger_index,
                    "level" : level,
                    "trigger_level" : trigger_level
                }
            return None

        if isinstance(source, Array):
            target = source
        else:
            raise NotImplementedError(
                "Hierarchical caching is not yet implemented"
            )  # TODO

        if thrifty:
            raise NotImplementedError("Thrifty caching is not yet implemented")  # TODO

        if sum(i is not None for i in [index, level, max_elements]) != 1:
            raise ValueError(
                "Specify one and only one of index, level, or max_elements"
            )
        
        if (trigger_level or trigger_index):
            if isinstance(source, Array) and source.role not in [Array.Role.CONST, Array.Role.INPUT]:
                raise ValueError("Multicaching is only supported for CONST and INPUT arrays")
            if layout is None:
                # Multi-caching is only supported for manual caching, so if a multicache is requested but a manual cache layout isn't provided, take the input array layout as the cache layout
                # TODO : merge "active element" automatic caching and "active block" manual caching so this isn't needed
                layout = source._requested_layout

        if level:
            # the level of the key-slices is the count of right-aligned wildcards, e.g. level 2 = (i[0], ..., *, *)
            # therefore the index is at position -level, e.g. (i[0], ..., index, *)
            # Note: this takes a snapshot of the schedule ordering
            index = self._sched._indices[-level]
        if trigger_level:
            # the trigger level is the level of the loopnest to fill the cache at. Must be the same as level or precede it
            # Note: this takes a snapshot of the schedule ordering
            trigger_index = self._sched._indices[-trigger_level]
        if index:
            self._add_index_attr(index, "cache")
        if trigger_index:
            self._add_index_attr(trigger_index, "trigger")

        cache = Cache(self, target, index, trigger_index, layout, max_elements, thrifty, location)
        self._commands.append(partial(self._add_cache, cache))
        return cache

    def _add_cache(self, cache, context: NativeLoopNestContext):
        from ..Targets import Target

        last_in_index = context.mapping[id(cache.index)] if cache.index else None

        trigger_index = context.mapping[id(cache.trigger_index)] if cache.trigger_index else last_in_index

        target = context.mapping[id(cache.target)]

        # TODO: support layout, location, thrifty
        if (isinstance(self._target, Target) and self._target.category == Target.Category.GPU):
            context.plan.add_cache(target, last_in_index, trigger_index, cache.max_elements, MemorySpace.NONE)
        else:
            context.plan.add_cache(target=target,
                                   index=last_in_index,
                                   trigger_index=trigger_index,
                                   max_elements=cache.max_elements,
                                   indexing=cache.indexing,
                                   allocation=cache.allocation,
                                   memory_space=cache.memory_space,
                                   memory_map=cache.memory_map,
                                   dim_order=cache.dimension_permutation)

    def pack_and_embed_buffer(self,
                              target,
                              wrapper_fn_name,
                              packed_buffer_name="",
                              indexing=CacheIndexing.GLOBAL_TO_PHYSICAL):
        """Emits a packing function for the given target and rewrites the loopnest to assume the given input is packed

        Args:
            target: The target being cached (e.g Array, Matrix, etc)
            data: The constant data to pack
            wrapper_fn_name: The name to give the wrapping function
            packed_buffer_name: The name to give the packed constant buffer
            indexing: The cache indexing
        """
        # TODO: Make this work with multiple kernels, fused schedules

        if target.role != Array.Role.CONST:
            raise ValueError("Can only pack and embed constant data buffers")

        self._commands.append(
            partial(self._pack_and_embed_buffer, target, wrapper_fn_name, packed_buffer_name, indexing))

    def emit_runtime_init_pack(self,
                               target,
                               packing_func_name,
                               packed_buf_size_func_name,
                               indexing=CacheIndexing.GLOBAL_TO_PHYSICAL):
        """Emits a packing function for the given target and rewrites the loopnest to assume the given input is packed

        Args:
            target: The target being cached (e.g Array, Matrix, etc)
            packing_func_name: The name of the packing function to emit
            packed_buf_size_func_name: The name of the function giving the packed buffer size to emit
            indexing: The cache indexing
        """
        # TODO: Make this work with multiple kernels, fused schedules
        self._commands.append(
            partial(
                self._emit_runtime_init_packing,
                target,
                packing_func_name,
                packed_buf_size_func_name,
                indexing,
            )
        )

    def _pack_and_embed_buffer(
        self,
        target,
        wrapper_fn_name,
        packed_buffer_name,
        indexing,
        context: NativeLoopNestContext,
    ):
        constant_data_buffer = target
        target = context.mapping[id(target)]
        context.plan.pack_and_embed_buffer(
            target,
            constant_data_buffer,
            wrapper_fn_name,
            packed_buffer_name,
            indexing,
        )

    def _emit_runtime_init_packing(
        self,
        target,
        packing_func_name,
        packed_buf_size_func_name,
        indexing,
        context: NativeLoopNestContext,
    ):
        target = context.mapping[id(target)]
        context.plan.emit_runtime_init_packing(
            target, packing_func_name, packed_buf_size_func_name, indexing
        )

    def bind(self, indices: Tuple[LoopIndex], grid: Tuple):
        """Binds iteration space dimensions to GPU execution units

        Args:
            indices: Sequence of indices to bind
            grid: Sequence of GPU thread or block identifiers, in the same order as indices
        """

        assert len(indices) == len(grid)

        if self._target is not None and self._target.category == Target.Category.GPU:
            self._commands.append(partial(self._bind, indices, grid))
        else:
            raise ValueError("Only supported on plans with GPU targets")

    def _bind(self, indices, grid, context: NativeLoopNestContext):
        for index, proc in zip(indices, grid):
            index = context.mapping[id(index)]
            context.plan.map_index_to_processor(index, proc)

    def kernelize(
        self, unroll_indices: Union[Tuple[LoopIndex], DelayedParameter], vectorize_indices: Union[Tuple[LoopIndex], LoopIndex, DelayedParameter] = None
    ):
        """Performs automatic kernelization.

        This is a sequence of unrolls, followed by an optional vectorize instruction:

            plan.kernelize(unroll_indices(i, j), vectorize_indices=k)

        is shorthand for:

            plan.unroll(i)

            plan.unroll(j)

            plan.vectorize(k)

        Args:
            unroll_indices: a list of indices to unroll
            vectorize_indices: Optional indices to vectorize
        """
        if isinstance(unroll_indices, DelayedParameter) or isinstance(vectorize_indices, DelayedParameter):
            self._delayed_calls[partial(self.kernelize)] = {"unroll_indices" : unroll_indices, "vectorize_indices" : vectorize_indices}
            return None

        vindices = [vectorize_indices] if isinstance(vectorize_indices, LoopIndex) else list(vectorize_indices)
        for vidx in vindices:
            if vidx in unroll_indices:
                raise ValueError("vectorize_indices cannot be one of the unroll indices")

        if not self._target.vectorization_info:
            raise RuntimeError("The target does not support vectorization")

        for idx in unroll_indices:
            self.unroll(idx)

        for vidx in vindices:
            self.vectorize(vidx)

    def _build_native_context(self, context: NativeLoopNestContext):

        target = self._target
        sched = self._sched
        nest = sched._nest

        if target and target.category == Target.Category.GPU:
            from .._lang_python._lang import _GPU, _Dim3

            BASE_BLOCK_DIM = 16
            block_dims = [BASE_BLOCK_DIM, BASE_BLOCK_DIM]
            grid_dims = [1, 1]

            # lookup the split factors for each loop index
            assert isinstance(self._sched, Schedule)
            
            index_to_splitfactor_map = {i: self._sched.get_index_transform(i)[1] 
                            for i in self._sched.get_indices() 
                                if self._sched.get_index_transform(i) and 
                                    self._sched.get_index_transform(i)[0] is IndexTransform.SPLIT}

            n = min(len(block_dims), len(nest._shape))

            # infer the block dimension from the split factor (if specified)
            for i, x in enumerate(nest._shape[:n]):
                shape, index = x
                if index in index_to_splitfactor_map:
                    block_dims[i] = index_to_splitfactor_map[index]

                # compute the grid size based on the block size
                grid_dims[i], remainder = divmod(shape, block_dims[i])
                if remainder != 0:
                    # TODO: remove this restriction
                    raise (
                        RuntimeError(
                            f"Shape {shape} must be a multiple of split factor {block_dims[i]}"
                        )
                    )

            context.options = _GPU(
                grid=_Dim3(*grid_dims, 1), block=_Dim3(*block_dims, 1)
            )
            context.plan = context.schedule.create_gpu_action_plan(context.options)
        else:
            context.plan = context.schedule.create_action_plan()

    def _build_with_native_context(self, context: NativeLoopNestContext):
        for cmd in self._commands:
            cmd(context)

    def _replay_delayed_calls(self):
        for delayed_call in self._delayed_calls:
            params = self._delayed_calls[delayed_call]
            if isinstance(params, dict):
                resolved_params = {key : params[key].get_value() if isinstance(params[key], DelayedParameter) else params[key] for key in params}
                delayed_call(**resolved_params)
            else:
                resolved_params = [param.get_value() if isinstance(param, DelayedParameter) else param for param in params]
                delayed_call(*resolved_params)


def _build_native_nest(plan: "ActionPlan", nest_args: List[Array]):
    from .._lang_python._lang import _Valor

    sched = plan._sched
    nest = sched._nest

    for array_arg in nest_args:
        array_arg._replay_delayed_calls()

    def build_array_native_context(ctx):
        for array_arg in nest_args:
            array_arg._build_native_context(ctx)

    def build_loopnest_native_context(ctx):
        nest._build_native_context(ctx)
        sched._build_native_context(ctx)
        plan._build_native_context(ctx)

    def nest_wrapper_fn(*args: List[List[_Valor]]):
        # wrapper function is responsible for mapping
        # the args to the loopnest logic function
        nest._replay_delayed_calls()
        sched._replay_delayed_calls()
        plan._replay_delayed_calls()

        loopnest_context = NativeLoopNestContext(
            function_args=nest_args, runtime_args=args
        )
        build_array_native_context(loopnest_context)
        build_loopnest_native_context(loopnest_context)

        nest._build_with_native_context(loopnest_context)
        sched._build_with_native_context(loopnest_context)
        plan._build_with_native_context(loopnest_context)

    return nest_wrapper_fn


def _create_function(
    plan: "ActionPlan", args: List[Array], public: bool = True, no_inline: bool = False
) -> Function:
    from secrets import token_hex

    name = f"nest_impl_{token_hex(16)}"

    return Function(
        name=name,
        args=args,
        public=public,
        definition=_build_native_nest(plan, args),
        no_inline=no_inline,
        target=plan._target,
    )


ActionPlan._create_function = _create_function
