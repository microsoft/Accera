####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import logging
from typing import *
from functools import partial, reduce
from collections.abc import Iterable

from ..Parameter import DelayedParameter
from .Array import Array
from .Schedule import Schedule, IndexTransform
from .Cache import Cache, DelayedCache
from .Function import Function
from .LoopIndex import LoopIndex
from .NativeLoopNestContext import NativeLoopNestContext
from ..Targets import GridUnits, Target
from ..Platforms import LibraryDependency
from ..Constants import AUTO
from .._lang_python import ScalarType, Role
from accera._lang_python._lang import Dimension

from .._lang_python._lang import (
    CacheIndexing,
    _MemorySpace,
    _MMASchedulingPolicy,
    _MMAShape,
    _CacheStrategy,
    _MMAFragmentOp
)
from ..algorithms import *


def _ceildiv(x, y):
    return ((x - 1) // y) + 1

class Plan:
    def __init__(self, schedule: Schedule, target: Target, _dynamic_shared_memory_size: int, _blocks_per_SM: int):
        self._sched = schedule
        self._target = target
        self._dynamic_shared_memory_size = _dynamic_shared_memory_size
        self._blocks_per_SM = _blocks_per_SM
        self._commands = []
        self._delayed_calls = {}
        self._index_attrs: Mapping[LoopIndex, List[str]] = {}
        self._dynamic_dependencies = set()
        self._bindings = {}
        self._heuristic_params = []

        if (
            target.category == Target.Category.GPU
            and target.runtime == Target.Runtime.VULKAN
        ):
            self._dynamic_dependencies.add(LibraryDependency.VULKAN)

    def _add_index_attr(self, index: LoopIndex, attr: str):
        attrs = self._index_attrs.get(index, [])
        attrs.append(attr)
        self._index_attrs[index] = attrs

    def print(self):
        self._sched.print(lambda index: self._index_attrs.get(index, []))

    def _get_heuristic_parameters(self):
        return self._heuristic_params

    def auto(
        self,
        *heuristics
    ):
        """Invokes an AutoPlanner to synthesize transofrmations such as, cache, vectorize, parallelize, unroll, bind, etc.
 
        Args:
        List of heuristics in algorithms module.
        """
        for heuristic in heuristics:
            # TODO: replace this check with an abstract base class of heuristics
            if not(isinstance(heuristic, NoneCacheHeuristics)):
                raise TypeError("Expected a NoneCacheHeuristic instance to be passed to plan.auto()")
            heuristic.create_parameterized_args(self)
            self._heuristic_params.append(heuristic._params_list)
            heuristic.invoke_cache_dsl_command(self)
        self._commands.append(partial(self._auto, heuristics))

    def _auto(
        self,
        heuristics: List,
        context: NativeLoopNestContext
    ):
        if not all(isinstance(heuristic, NoneCacheHeuristics) for heuristic in heuristics):
            raise TypeError("Heuristic must be an instance of NoneCacheHeuristics\n")

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

    def parallelize(
        self,
        indices: Union[LoopIndex, Tuple[LoopIndex], DelayedParameter],
        pin: Union[Tuple[Any], DelayedParameter] = None,
        policy: Union[str, DelayedParameter] = "static",
        max_threads: Union[int, DelayedParameter] = None
    ):
        """Executes one or more loops in parallel on multiple cores or processors.
        Only available for targets with multiple cores or processors.

        Args:
            indices: The iteration-space dimensions to run in parallel.
                To assign multiple threads to an index, first split that index,
                then parallelize its split indices.

                Unsplit indices will be assigned one thread each, split indices
                will be assigned threads based on the number of split blocks.
                This is limited by the number of threads supported by the target.
            pin: Pin the computation to a subset of cores or processors.
            policy: The scheduling policy to apply ("dynamic" or "static").
            max_threads: The maximum number of threads to use when distributing the workload.
        """
        if self._target.category == Target.Category.CPU:
            self._dynamic_dependencies.add(LibraryDependency.OPENMP)

        if any([isinstance(arg, DelayedParameter) for arg in [indices, pin, policy, max_threads]]):
            self._delayed_calls[partial(self.parallelize)] = {
                "indices": indices,
                "pin": pin,
                "policy": policy,
                "max_threads": max_threads
            }
            return None

        indices = [indices] if isinstance(indices, LoopIndex) else list(indices)

        # ensure the indices are contiguous and follow the Schedule ordering
        start = self._sched._indices.index(indices[0])
        end = start + len(indices)
        if (
            end > len(self._sched._indices)
            or indices != self._sched._indices[start:end]
        ):
            raise ValueError(
                "indices must be contiguous in the Schedule dimension order"
            )

        for index in indices:
            self._add_index_attr(index, "parallelized")

        self._commands.append(partial(self._parallelize, indices, policy, max_threads))

    def _parallelize(self, indices, policy, max_threads, context: NativeLoopNestContext):
        from .._lang_python._lang import _ParallelizationPolicy

        if max_threads is None:
            max_threads = self._target.num_threads
        elif max_threads <= 0:
            raise ValueError("max_threads must be a positive (greater than 0) integer.")

        # num_threads = number of iterations, clamped by the number of threads supported by this target and the user provided limit (if any)
        num_threads = min(max_threads, self._target.num_threads, self._sched._get_num_split_iterations(indices))
        logging.debug(f"Parallelizing with {num_threads} thread(s)")

        idxs = [context.mapping[id(index)] for index in indices]

        context.plan.parallelize(
            idxs,
            num_threads,
            _ParallelizationPolicy.DYNAMIC
            if policy == "dynamic"
            else _ParallelizationPolicy.STATIC,
        )

    def tensorize(
        self,
        indices: Union[LoopIndex, Tuple[LoopIndex]],
        mma_shape: _MMAShape,
        num_total_passes: int = 1,
        use_static_offsets: bool = False,
        num_fused_passes: int = None,
        scheduling_policy: _MMASchedulingPolicy = _MMASchedulingPolicy.PASS_ORDER,
        prologue_op: _MMAFragmentOp = _MMAFragmentOp.NONE,
        prologue_arg: float = 0.0,
        epilogue_op: _MMAFragmentOp = _MMAFragmentOp.NONE,
        epilogue_arg: float = 0.0,
        _use_rocWMMA: bool = False,
    ):
        """Only available for targets with native matrix multiplication instruction (tensor core) support.
        Marks the dimensions of the iteration-space for tensorization.
        Only perfectly nested loops of the following form can be tensorized:

        for i in range(M):
            for k in range(N):
                for j in range(K):
                    C[i, j] += A[i, k] * B[k, j]

        Args:
            indices: The iteration space dimensions to tensorize.
            mma_shape: The MMA op type to use for tensorization.
            num_total_passes: This controls the number of MMA passes to run.
            use_static_offsets: This is an optimization flag, which when enabled will use precomputed offset maps stored in device constant memory.
            num_fused_passes: This controls the number of passes for which register allocation is done, higher the value more the number of registers that are allocated.
            scheduling_policy: For multi-block MMA operations, this controls whether matrix multiplication is done block-by-block or pass-by-pass (affects register usage).
            prologue_op: The pre-tensorization operation to run on matrix fragment data as a part of load, e.g. 0-init (CLEAR).
            epilogue_op: The post-tensorization operation to run on matrix fragment data as a part of store, e.g. ReLU.
        """
        if self._target.category != Target.Category.GPU:
            raise ValueError("tensorization currently only supported on GPU targets")

        indices = [indices] if isinstance(indices, LoopIndex) else list(indices)

        if len(indices) < 3:
            raise ValueError("tensorization requires at least three input indices")

        for index in indices:
            self._add_index_attr(index, "tensorized")

        self._commands.append(
            partial(
                self._tensorize,
                indices,
                mma_shape,
                num_total_passes,
                use_static_offsets,
                num_fused_passes,
                scheduling_policy,
                prologue_op,
                prologue_arg,
                epilogue_op,
                epilogue_arg,
                _use_rocWMMA,
            )
        )

    def _tensorize(
        self,
        indices,
        mma_shape,
        num_total_passes,
        use_static_offsets,
        num_fused_passes,
        scheduling_policy,
        prologue_op,
        prologue_arg,
        epilogue_op,
        epilogue_arg,
        _use_rocWMMA,
        context: NativeLoopNestContext,
    ):
        if num_fused_passes is None:
            num_fused_passes = -1
        elif num_fused_passes <= 0:
            raise ValueError(
                "Number of passes used for fusing must be a positive number greater than 0."
            )

        for index in list(map(self._sched._resolve_index, indices)):
            start, stop, step = self._sched.get_index_range(index)
            if start != 0:
                raise ValueError("The tensorization index must start at 0")
        if (
            not self._target.tensor_core_info.supports(
                input_type=ScalarType.float32,
                output_type=ScalarType.float32,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes,
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.float16,
                output_type=ScalarType.float32,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes,
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.float16,
                output_type=ScalarType.float16,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes,
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.bfloat16,
                output_type=ScalarType.bfloat16,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.bfloat16,
                output_type=ScalarType.float32,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.uint8,
                output_type=ScalarType.int32,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.int8,
                output_type=ScalarType.int32,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.int8,
                output_type=ScalarType.int16,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes
            )
            and not self._target.tensor_core_info.supports(
                input_type=ScalarType.int8,
                output_type=ScalarType.int8,
                shape=mma_shape,
                num_total_passes=num_total_passes,
                num_fused_passes=num_fused_passes
            )
            ):
            raise ValueError(
                "The target does not support the given tensorization dimensions with shape=",
                mma_shape,
            )

        idxs = [context.mapping[id(index)] for index in indices]

        context.plan.tensorize(
            indices=idxs,
            dims=mma_shape,
            numTotalPasses=num_total_passes,
            useStaticOffsets=use_static_offsets,
            numFusedPasses=num_fused_passes,
            schedulingPolicy=scheduling_policy,
            prologueOp=prologue_op,
            prologueArg=prologue_arg,
            epilogueOp=epilogue_op,
            epilogueArg=epilogue_arg,
            _useRocWMMA=_use_rocWMMA,
        )

    def cache(
        self,
        source: Union[Array, Cache],
        index: Union[LoopIndex, DelayedParameter] = None,
        trigger_index: Union[LoopIndex, DelayedParameter] = None,
        layout: Union[Array.Layout, DelayedParameter] = None,
        max_elements: int = None,
        element_type: Union[ScalarType, DelayedParameter] = None,
        thrifty: Union[bool, DelayedParameter] = None,
        location: _MemorySpace = _MemorySpace.NONE,
        level: Union[int, DelayedParameter] = None,
        trigger_level: Union[int, DelayedParameter] = None,
        double_buffer: Union[bool, DelayedParameter] = False,
        double_buffer_location: Union[object, _MemorySpace, DelayedParameter] = AUTO,
        _shared_memory_offset: Union[int, DelayedParameter] = None,
        vectorize: Union[bool, DelayedParameter, object] = AUTO,
        strategy: _CacheStrategy = AUTO,
        _delayed_cache: DelayedCache = None,
        _temp_array_multicaches: bool = False # experimental: allow multi-caching of TEMP arrays
    ):
        """Adds a cache for a view target

        Args:
            source: The array or cache from which this cache is copied.
            index: The index used to determine the cache level. Specify one and only one of `index`, `level`, `max_elements`.
            trigger_index: The index used to determine what level to fill the cache at. `trigger_index` can't come after `index` in the schedule order, and will default to `index` if not specified. Specify at most one of `trigger_index` or `trigger_level`.
            layout: The affine memory map, if different from the source.
            level: The key-slice level to cache (the number of wildcard dimensions in a key-slice). Specify one and only one of `index`, `level`, `max_elements`.
            trigger_level: The key-slice level to fill the cache at. `trigger_level` can't be smaller than `level`, and will default to `level` if not specified. Specify at most one of `trigger_index` or `trigger_level`.
            max_elements: The maximum elements to include in the cached region. Specify one and only one of `index`, `level`, `max_elements`.
            element_type: The element type to use in the cache. Defaults to the element type of the cached array.
            strategy: The thread to data mapping pattern to use when collaboratively caching by multiple threads. Defaults to AUTO which will resolve to the strategy best suited for the current target environment.
            thrifty: Use thrifty caching (copy data into a cache only if the cached data differs from the original active block). This defaults to False as it slows down compilation speed so it is intended as an opt-in feature.
            double_buffer: Make this a double buffer cache by copying data one iteration ahead and using private memory on GPU for this procedure.
            vectorize: Whether to vectorize the cache operations. Defaults to AUTO, which will behave like `vectorize=True` if the loopnest has a vectorized loop or `vectorize=False` if the loopnest has no vectorized loops.
            double_buffer_location: The memory space used for storing iteration data for the double buffer cache. Requires that double_buffer is set to True. Defaults to AUTO.
                AUTO will configure the double buffering location based on the following:
                | location            | double_buffer | double_buffer_location = `AUTO` |
                | ------------------- | ------------- | ------------------------------- |
                | MemorySpace.SHARED  | True          | MemorySpace.PRIVATE             |
                | !MemorySpace.SHARED | True          | Same value as location          |
        """
        if (
            any(
                [
                    isinstance(arg, DelayedParameter)
                    for arg in (
                        index,
                        trigger_index,
                        level,
                        trigger_level,
                        thrifty,
                        double_buffer,
                        double_buffer_location,
                        vectorize,
                        layout,
                        element_type,
                        _shared_memory_offset
                    )
                ]
            )
            or (isinstance(source, DelayedCache) and not source.completed)
        ):
            # If any of the cache level arguments are parameters, then this cache call is incomplete until those parameters
            # have values. Additionally, if this is a hierarchical cache and an outer cache is parameterized,
            # then this cache call is also incomplete until the outer cache's parameters have values

            # Create an incomplete Cache object so hierarchical caches that depend on this cache handle can
            # have an object to hold onto
            delayed_cache = DelayedCache(plan=self, target=source)
            self._delayed_calls[
                partial(
                    self.cache,
                    source=source,
                    max_elements=max_elements,
                    location=location,
                    _delayed_cache=delayed_cache,
                )
            ] = {
                "index": index,
                "trigger_index": trigger_index,
                "level": level,
                "trigger_level": trigger_level,
                "layout": layout,
                "thrifty": thrifty,
                "double_buffer": double_buffer,
                "double_buffer_location": double_buffer_location,
                "_shared_memory_offset": _shared_memory_offset,
                "vectorize": vectorize,
                "element_type": element_type
            }
            return delayed_cache

        if sum(i is not None for i in [index, level, max_elements]) != 1:
            raise ValueError(
                "Specify one and only one of index, level, or max_elements"
            )

        if max_elements is not None and max_elements <= 0:
            raise ValueError(
                "Max element count specified as a cache budget must be greater than 0"
            )

        if isinstance(source, Array):
            array_role = source.role
        elif isinstance(source, Cache):
            array_role = source.target_role

        if double_buffer and array_role not in [Role.CONST, Role.INPUT]:
            raise ValueError(
                "Double-buffering is only supported for CONST and INPUT arrays"
            )

        if not double_buffer and double_buffer_location != AUTO:
            raise ValueError(
                "double_buffer_location is only valid to specify when double_buffer is set to True"
            )

        if self._target.category == Target.Category.GPU and location == _MemorySpace.GLOBAL:
            raise ValueError("Global memory caches are not yet supported on GPU targets.")

        if self._target.category == Target.Category.CPU and strategy != AUTO:
            raise ValueError("Only AUTO strategy is supported on the CPU")

        if strategy == AUTO:
            strategy = _CacheStrategy.STRIPED

        if double_buffer and location == _MemorySpace.MMA_FRAGMENT:
            raise ValueError("Double buffering is not supported with Tensor caching")

        if _shared_memory_offset is not None and location != _MemorySpace.SHARED:
            raise ValueError("_shared_memory_offset can only be set with shared memory")

        if double_buffer_location is AUTO:
            if double_buffer:
                if (
                    self._target.category == Target.Category.GPU
                    and location == _MemorySpace.SHARED
                ):
                    double_buffer_location = _MemorySpace.PRIVATE
                else:
                    double_buffer_location = location
            else:
                double_buffer_location = _MemorySpace.NONE

        if max_elements is None:
            # Validate or set index / level values

            # Validate that if index is specified, then level and trigger_level are not
            if (index is not None) and (level is not None or trigger_level is not None):
                raise ValueError(
                    "Can't specify both a cache index and a cache level or trigger level"
                )

            # Validate that if level is specified, then index and trigger_index are not
            if (level is not None) and (index is not None or trigger_index is not None):
                raise ValueError(
                    "Can't specify both a cache level and a cache index or trigger index"
                )

            if level:
                # the level of the key-slices is the count of right-aligned wildcards, e.g. level 2 = (i[0], ..., *, *)
                # therefore the index is at position -level, e.g. (i[0], ..., index, *)
                # Note: this takes a snapshot of the schedule ordering
                index = self._sched._indices[-level]
            else:
                self._add_index_attr(index, "cache")
                index_pos = self._sched._indices.index(index)
                level = len(self._sched._indices) - index_pos

            multicaching_roles = [
                Role.CONST,
                Role.INPUT,
            ]
            if _temp_array_multicaches:
                multicaching_roles += [Role.TEMP]

            if (trigger_level or trigger_index) and array_role not in multicaching_roles:
                raise ValueError(
                    "Multicaching is only supported for CONST and INPUT arrays"
                )

            if layout is None:
                layout = source._requested_layout

            # Validate or set trigger_index / trigger_level values

            if trigger_index is not None and trigger_level is not None:
                raise ValueError(
                    "Can't specify both a trigger_index and a trigger_level"
                )

            if trigger_index is None and trigger_level is None:
                trigger_index = index
                trigger_level = level
            elif trigger_level is not None:
                # the trigger level is the level of the loopnest to fill the cache at. Must be the same as level or precede it
                # Note: this takes a snapshot of the schedule ordering
                trigger_index = self._sched._indices[-trigger_level]
            else:
                self._add_index_attr(trigger_index, "trigger")
                trigger_index_pos = self._sched._indices.index(trigger_index)
                trigger_level = len(self._sched._indices) - trigger_index_pos

            if level > trigger_level:
                raise ValueError(
                    "Cache level must be less than or equal to the cache trigger level"
                )

            if level <= 0:
                raise ValueError("Cache level must be greater than or equal to 1")

            if trigger_level <= 0:
                raise ValueError(
                    "Cache trigger level must be greater than or equal to 1"
                )

        if isinstance(source, Cache):
            # The outer cache must have a higher cache level and a higher trigger level than this cache, or a higher max element budget
            if source.max_elements is None and (
                source.level is None or source.trigger_level is None
            ):
                # If the outer cache doesn't have a max element budget, then it must have both a cache level and a cache trigger_level
                raise ValueError(
                    "Given source cache doesn't have a cache level, trigger_level, or max_elements"
                )

            if (source.max_elements is None) != (max_elements is None):
                raise ValueError(
                    "Can only create a max element hierarchical caches of other max element caches"
                )
            if source.max_elements is not None:
                if source.max_elements <= max_elements:
                    raise ValueError(
                        "Outer max element cache for a hierarchical cache must have a larger budget than the inner cache"
                    )
            else:
                if source.level <= level:
                    raise ValueError(
                        "Outer cache for a hierarchical cache must have a higher cache level than inner cache"
                    )
                if source.level < trigger_level:
                    raise ValueError(
                        "Outer cache for a hierarchical cache must have a greater or equal cache level than the inner cache's trigger_level"
                    )
        if element_type is None:
            element_type = source.element_type

        cache = Cache(
            plan=self,
            target=source,
            index=index,
            trigger_index=trigger_index,
            level=level,
            trigger_level=trigger_level,
            element_type=element_type,
            layout=layout,
            max_elements=max_elements,
            thrifty=thrifty,
            location=location,
            double_buffer=double_buffer,
            double_buffer_location=double_buffer_location,
            shared_memory_offset=_shared_memory_offset,
            vectorize=vectorize,
            strategy=strategy
        )

        if _delayed_cache:
            _delayed_cache.complete(cache)
            cache = _delayed_cache
            if _delayed_cache.enqueue_command:
                self._commands.append(partial(self._add_cache, cache))
                _delayed_cache.enqueue_command = False
        else:
            self._commands.append(partial(self._add_cache, cache))

        return cache

    def _add_cache(self, cache, context: NativeLoopNestContext):
        from ..Targets import Target

        # Resolve vectorize=AUTO to either True or False since vectorize() will have been called by this point
        if cache.vectorize is AUTO:
            cache.vectorize = False
            for attrs in self._index_attrs.values():
                if "vectorized" in attrs:
                    cache.vectorize = True

        vectorization_info = None
        if cache.vectorize:
            vectorization_info = self._target.vectorization_info

        last_in_index = context.mapping[id(cache.index)] if cache.index else None

        trigger_index = (
            context.mapping[id(cache.trigger_index)]
            if cache.trigger_index
            else last_in_index
        )

        if isinstance(cache.target, Array):
            target = context.mapping[id(cache.target)]
        else:
            target = cache.target.native_cache

        cache.native_cache = context.plan.add_cache(
            target=target,
            index=last_in_index,
            trigger_index=trigger_index,
            max_elements=cache.max_elements,
            indexing=cache.indexing,
            allocation=cache.allocation,
            location=cache.location,
            memory_map=cache.memory_map,
            dim_order=cache.dimension_permutation,
            thrifty=cache.thrifty,
            double_buffer=cache.double_buffer,
            double_buffer_location=cache.double_buffer_location,
            shared_memory_offset=cache.shared_memory_offset,
            vectorization_info=vectorization_info,
            element_type=cache.element_type,
            strategy=cache.strategy
        )

    def pack_and_embed_buffer(
        self,
        target,
        wrapper_fn_name,
        packed_buffer_name="",
        indexing=CacheIndexing.GLOBAL_TO_PHYSICAL,
    ):
        """Emits a packing function for the given target and rewrites the loopnest to assume the given input is packed

        Args:
            target: The target being cached (e.g Array, Matrix, etc)
            data: The constant data to pack
            wrapper_fn_name: The name to give the wrapping function
            packed_buffer_name: The name to give the packed constant buffer
            indexing: The cache indexing
        """
        # TODO: Make this work with multiple kernels, fused schedules

        if target.role != Role.CONST:
            raise ValueError("Can only pack and embed constant data buffers")

        self._commands.append(
            partial(
                self._pack_and_embed_buffer,
                target,
                wrapper_fn_name,
                packed_buffer_name,
                indexing,
            )
        )

    def emit_runtime_init_pack(
        self,
        target,
        packing_func_name,
        packed_buf_size_func_name,
        indexing=CacheIndexing.GLOBAL_TO_PHYSICAL,
    ):
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

    def bind(self, mapping: Mapping[Union[LoopIndex, Tuple[LoopIndex], DelayedParameter], Union[GridUnits, DelayedParameter]]):
        """Binds iteration space dimensions to GPU execution units

        Args:
            mapping: Mapping of indices to GPU thread or block identifiers
        """

        if self._target is not None and self._target.category == Target.Category.GPU:
            if any(
                [
                    isinstance(index, DelayedParameter)
                    or isinstance(proc, DelayedParameter)
                    for index, proc in mapping.items()
                ]
            ):
                self._delayed_calls[partial(self.bind)] = mapping
                return None

            self._commands.append(partial(self._bind, mapping))

            for index_or_tuple, proc in mapping.items():
                self._bindings[proc] = index_or_tuple

        else:
            raise ValueError("Only supported on plans with GPU targets")

    def _bind(self, mapping: Mapping[Union[LoopIndex, Tuple[LoopIndex]], GridUnits], context: NativeLoopNestContext):
        for index_or_tuple, proc in mapping.items():
            if isinstance(index_or_tuple, tuple):
                resolved_index_or_tuple = [context.mapping[id(index)] for index in index_or_tuple]
            else:
                resolved_index_or_tuple = [context.mapping[id(index_or_tuple)]]
            context.plan._map_index_to_processor(resolved_index_or_tuple, proc.value)

    def kernelize(
        self,
        unroll_indices: Union[Tuple[LoopIndex], DelayedParameter],
        vectorize_indices: Union[Tuple[LoopIndex], LoopIndex, DelayedParameter] = None,
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
        if isinstance(unroll_indices, DelayedParameter) or isinstance(
            vectorize_indices, DelayedParameter
        ):
            self._delayed_calls[partial(self.kernelize)] = {
                "unroll_indices": unroll_indices,
                "vectorize_indices": vectorize_indices,
            }
            return None

        vindices = (
            [vectorize_indices]
            if isinstance(vectorize_indices, LoopIndex)
            else list(vectorize_indices)
        )
        for vidx in vindices:
            if vidx in unroll_indices:
                raise ValueError(
                    "vectorize_indices cannot be one of the unroll indices"
                )

        if not self._target.vectorization_info:
            raise RuntimeError("The target does not support vectorization")

        for idx in unroll_indices:
            self.unroll(idx)

        for vidx in vindices:
            self.vectorize(vidx)

    def _calc_block_grid_dim(self):
        target = self._target
        sched = self._sched
        nest = sched._nest

        # lookup the split factors for each loop index
        index_to_splitfactor_map = {}

        for i in self._sched.get_indices():
            transform = self._sched.get_index_transform(i)
            if transform and transform[0] is IndexTransform.SPLIT:
                index_to_splitfactor_map[i] = transform[1]

        def units_to_dim(units, dims):
            def compute_index_itercount(idx):
                begin, end, step = self._sched.get_index_range(idx)
                if step == 1 and idx in index_to_splitfactor_map:
                    return index_to_splitfactor_map[idx]
                else:
                    return _ceildiv(end - begin, step)

            for i, u in enumerate(units):
                index_or_tuple = self._bindings.get(u)
                if index_or_tuple is not None:
                    if not isinstance(index_or_tuple, Iterable):
                        index_or_tuple = (index_or_tuple,)
                    dims[i] = reduce(lambda x, y: x*y, [compute_index_itercount(index) for index in index_or_tuple])

        block_dims = [1, 1, 1]
        grid_dims = [1, 1, 1]
        warp_dims = [0, 0]

        if any(proc in self._bindings for proc in target.GridUnit):
            # TODO: move this into the C++ layer
            block_units = [
                target.GridUnit.BLOCK_X,
                target.GridUnit.BLOCK_Y,
                target.GridUnit.BLOCK_Z,
            ]
            thread_units = [
                target.GridUnit.THREAD_X,
                target.GridUnit.THREAD_Y,
                target.GridUnit.THREAD_Z,
            ]
            warp_units = [
                target.GridUnit.WARP_X,
                target.GridUnit.WARP_Y,
            ]

            # Block units map to the grid specification
            units_to_dim(block_units, grid_dims)

            # Thread units map to the block specification
            units_to_dim(thread_units, block_dims)

            # Use warp units to create blockdim in case of tensorization
            units_to_dim(warp_units, warp_dims)

            warp_dim_set = any(item != 0 for item in warp_dims)

            if warp_dim_set:
                block_dims = [warp_dims[0] * target.warp_size, warp_dims[1], 1]
        else:
            BASE_BLOCK_DIM = 16
            block_dims = [BASE_BLOCK_DIM, BASE_BLOCK_DIM, 1]

            n = min(len(block_dims), len(nest._shape))

            # infer the block dimension from the split factor (if specified)
            for i, x in enumerate(nest._shape[:n]):
                shape, index = x
                if index in index_to_splitfactor_map:
                    block_dims[i] = index_to_splitfactor_map[index]

                # compute the grid size based on the block size
                grid_dims[i] = _ceildiv(shape, block_dims[i])

        return block_dims, grid_dims

    def _is_valid_block_dim(self, block_dims) -> bool:
        max_block_dim = self._target.max_block_size
        if len(max_block_dim) != 3:
            return True # Not a known GPU target (max_block_size not set by user) - bypass validation.

        for i in range(3):
            if block_dims[i] > max_block_dim[i]:
                return False

        return True

    def _is_valid_block_size(self, block_dims) -> bool:
        max_threads = self._target.max_threads_per_block
        if not max_threads:
            return True # Not a known GPU target (max_threads_per_block not set by user) - bypass validation.

        block_size = block_dims[0] * block_dims[1] * block_dims[2]
        return block_size <= max_threads

    def _erase_loops(self, indices: List[LoopIndex]):
        for index in indices:
            self._add_index_attr(index, "_erase")

        self._commands.append(
            partial(self._erase_loops_delayed, indices)
        )

    def _erase_loops_delayed(self, indices: List[LoopIndex], context: NativeLoopNestContext):
        for index in indices:
            context.plan._erase_loop(context.mapping[id(index)])

    def _build_native_context(self, context: NativeLoopNestContext):
        target = self._target

        if target and target.category == Target.Category.GPU:
            from .._lang_python._lang import _GPU, _Dim3

            assert isinstance(self._sched, Schedule)

            block_dims, grid_dims = self._calc_block_grid_dim()

            # Validate block dim sizes
            if not self._is_valid_block_dim(block_dims):
                raise ValueError (f"Invalid block dimensions: {block_dims}. Largest supported block dimensions: {target.max_block_size}.")

            if not self._is_valid_block_size(block_dims):
                raise ValueError(f"Invalid block size {block_dims}. Max threads per block: {target.max_threads_per_block}.")

            context.options = _GPU(grid=_Dim3(*grid_dims), block=_Dim3(*block_dims), dynamic_shared_memory_size=self._dynamic_shared_memory_size, blocks_per_SM=self._blocks_per_SM)
            context.plan = context.schedule.create_gpu_plan(
                gpu_options=context.options, runtime=target.runtime
            )
        else:
            context.plan = context.schedule.create_plan()

    def _build_with_native_context(self, context: NativeLoopNestContext):
        for cmd in self._commands:
            cmd(context)

    def _replay_delayed_calls(self):
        """
        This method is called once per adding function, so it can be called multiple times when
        multiple functions get added. In order for the functions to be added correctly, we need to make sure all
        the residual states are cleared between different method calls.

        For example, in Schedule class, we identify that Schedule._index_map can have residual states, so we need to reset self._index_map
        before we replay the delayed methods.

        If there is no residual states between different method calls, no need to reset.
        """
        for delayed_call in self._delayed_calls:
            params = self._delayed_calls[delayed_call]
            if isinstance(params, dict):
                resolved_params = {
                    key.get_value() if isinstance(key, DelayedParameter) \
                        else key: params[key].get_value() if isinstance(params[key], DelayedParameter) \
                        else params[key] \
                            for key in params
                }

                # Some methods package the long list of arguments into a dict parameter, like Plan.cache, Plan.parallelize and Plan.kernelize
                if delayed_call.func.__name__ == "bind":
                    delayed_call(resolved_params)
                elif delayed_call.func.__name__ == "cache" or \
                    delayed_call.func.__name__ == "parallelize" or \
                    delayed_call.func.__name__ == "kernelize":
                    delayed_call(**resolved_params)
                else:
                    raise NotImplementedError(
                        f"Please add the function Plan.{delayed_call.func.__name__} that supports parameters to the conditional list and indicate the way of dereferencing the dict parameter"
                    )
            else:
                resolved_params = [
                    param.get_value() if isinstance(param, DelayedParameter) else param
                    for param in params
                ]
                delayed_call(*resolved_params)


def _build_native_nest(plan: "Plan", nest_args: List[Array]):
    from .._lang_python._lang import _Valor

    sched = plan._sched
    nest = sched._nest

    for array_arg in nest_args:
        if isinstance(array_arg, Array):
            array_arg._replay_delayed_calls()

    def build_array_native_context(ctx):
        for array_arg in nest_args:
            if isinstance(array_arg, Array):
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
            function_args=list(nest_args), runtime_args=args
        )
        build_array_native_context(loopnest_context)
        build_loopnest_native_context(loopnest_context)

        nest._build_with_native_context(loopnest_context)
        sched._build_with_native_context(loopnest_context)
        plan._build_with_native_context(loopnest_context)

    return nest_wrapper_fn


def _create_function(
    plan: "Plan", args: List[Union[Array, Dimension]], public: bool = True, **kwargs
) -> Function:
    from secrets import token_hex

    name = f"nest_impl_{token_hex(16)}"

    return Function(
        name=name,
        args=args,
        public=public,
        definition=_build_native_nest(plan, args),
        target=plan._target,
        **kwargs
    )


Plan._create_function = _create_function
