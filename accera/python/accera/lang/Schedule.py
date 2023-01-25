####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import List, Mapping, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from varname import varname

from .NativeLoopNestContext import NativeLoopNestContext
from .Nest import Nest, LoopIndex
from ..Targets import Target
from ..Parameter import DelayedParameter
from .._lang_python._lang import _MMAShape


@dataclass
class IndexTransform(Enum):
    SPLIT = auto()
    SKEW = auto()
    PAD = auto()


@dataclass
class IndexEntry:
    stop: int
    start: int = 0
    step: int = 1
    inners: List[LoopIndex] = field(default_factory=list)
    parent: LoopIndex = None
    transform: Tuple[IndexTransform, Any] = None

    def interval(self):
        return (self.start, self.stop, self.step)

    def num_iterations(self):
        return (self.stop - self.start) // self.step


class Schedule:
    "Used for transforming an iteration space"

    def __init__(self, nest: Nest):
        self._nest = nest
        self._delayed_calls = {}
        self._parameterized_index_map = {}

        # nest.get_indices gives us a single index if there's only one index
        self._indices = nest.get_indices()
        try:
            _ = iter(self._indices)
        except TypeError:
            self._indices: List[LoopIndex] = [self._indices]

        shape = nest.get_shape()
        if any([isinstance(s, DelayedParameter) for s in shape]):
            self._delayed_calls[partial(self._init_delayed)] = nest

        self._index_map = {
            index: IndexEntry(stop=size) for index, size in zip(self._indices, shape)
        }

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def get_indices(self):
        return self._indices.copy()

    def create_plan(self, target: "accera.Target" = Target.HOST, _dynamic_shared_memory_size: int = None, _blocks_per_SM: int = None) -> "accera.Plan":
        """Creates a plan for running this schedule

        Args:
            target: Optional target specification. Defaults to the HOST
        """
        from .Plan import Plan

        if _dynamic_shared_memory_size is not None and target.category != Target.Category.GPU:
            raise ValueError("_dynamic_shared_memory_size can only be set on a GPU target")

        if _blocks_per_SM is not None and target.category != Target.Category.GPU:
            raise ValueError("_blocks_per_SM can only be set on a GPU target")

        return Plan(self, target, _dynamic_shared_memory_size if _dynamic_shared_memory_size is not None else 0, _blocks_per_SM if _blocks_per_SM is not None else 0)

    def get_index_range(self, index):
        return self._index_map[index].interval()

    def get_index_parent(self, index):
        return self._index_map[index].parent

    def get_index_transform(self, index):
        return self._index_map[index].transform

    def split(self, index: LoopIndex, size: Union[int, DelayedParameter]) -> LoopIndex:
        """The `split` transformation takes a dimension `i` and a `size`, modifies `i`, and creates a new dimension `ii`.

        Args:
            index: The dimension to split
            size: The split size

        Returns:
            The inner dimension after split
        """
        index = self._resolve_index(index)

        try:
            name = varname(multi_vars=False)
        except:
            name = None

        if isinstance(size, DelayedParameter):
            self._delayed_calls[partial(self._split_delayed, index)] = size

            inner_index = index.create_child_index()
            if name and not inner_index.name:
                inner_index.name = name
            order_pos = self._indices.index(index) + 1
            self._indices.insert(order_pos, inner_index)
            self._index_map[index].inners.insert(0, inner_index)
            self._index_map[inner_index] = IndexEntry(
                stop=0, parent=index, transform=(IndexTransform.SPLIT, 0)
            )

            return inner_index

        if not size or not float(size).is_integer() or size < 1:
            raise ValueError("Split sizes must be integers >= 1")

        inner_index = index.create_child_index()
        if name and not inner_index.name:
            inner_index.name = name

        order_pos = self._indices.index(index) + 1
        self._indices.insert(order_pos, inner_index)

        start, stop, step = self._index_map[index].interval()
        self._index_map[index].step *= size
        self._index_map[index].inners.insert(0, inner_index)
        self._index_map[inner_index] = IndexEntry(stop=size, parent=index, transform=(IndexTransform.SPLIT, size))

        return inner_index

    def skew(
        self,
        index: LoopIndex,
        reference_index: LoopIndex,
        unroll_loops_smaller_than: Union[int, DelayedParameter] = None,
    ) -> None:
        """Transforms a dimension with respect to a reference dimension into a parellelogram by padding with empty elements.

        Args:
            index: The dimension to skew
            reference_index: The reference dimension
            unroll_loops_smaller_than: Unroll loops that are smaller than this range (non-inclusive)
        """

        index = self._resolve_index(index)
        reference_index = self._resolve_index(reference_index)

        skewed_index = index.create_child_index()

        order_pos = self._indices.index(index)
        self._indices[order_pos] = skewed_index

        start, stop, _ = self._index_map[index].interval()
        ref_start, ref_stop, _ = self._index_map[reference_index].interval()

        if isinstance(unroll_loops_smaller_than, DelayedParameter):
            self._delayed_calls[
                partial(self._skew_delayed, skewed_index, reference_index)
            ] = unroll_loops_smaller_than
            self._index_map[skewed_index] = IndexEntry(
                stop=(stop - start) + (ref_stop - ref_start),
                parent=index,
                transform=(IndexTransform.SKEW, (reference_index, 0)),
            )
            return

        self._index_map[skewed_index] = IndexEntry(
            stop=(stop - start) + (ref_stop - ref_start),
            parent=index,
            transform=(
                IndexTransform.SKEW,
                (reference_index, unroll_loops_smaller_than),
            ),
        )

    def pad(
        self, index: LoopIndex, size: Union[int, DelayedParameter], _front: bool = True
    ) -> None:
        """Pads the beginning of a specified dimension of the iteration-space with empty (no-op) elements.

        Args:
            index: The dimension to pad
            size: The number of elements to pad
        """

        index = self._resolve_index(index)

        padded_index = index.create_child_index()

        order_pos = self._indices.index(index)
        self._indices[order_pos] = padded_index

        start, stop, _ = self._index_map[index].interval()

        if isinstance(size, DelayedParameter):
            self._delayed_calls[partial(self._pad_delayed, padded_index, _front)] = size
            self._index_map[padded_index] = IndexEntry(
                start=start,
                stop=stop,
                parent=index,
                transform=(IndexTransform.PAD, (0, _front)),
            )
            return

        self._index_map[padded_index] = IndexEntry(
            start=start,
            stop=stop + size,
            parent=index,
            transform=(IndexTransform.PAD, (size, _front)),
        )

    def reorder(
        self,
        order: Union[Tuple[LoopIndex], LoopIndex, DelayedParameter] = None,
        *args: LoopIndex,
    ):
        """The `reorder` transformation sets the order of the indices in the schedule.

        Args:
            order: Either the order of indices to set, or the outermost index if using variable arguments.
            args: Optional variable arguments containing subsequent indices to set

        Remarks:
        These orders are not allowed:
            1. The *outer dimension* created by a `split` transformation must always precede the corresponding *inner dimension*.
            2. The *fusing dimension* created by a `fuse` operation must always precede any *unfused dimensions*.
        """

        if isinstance(order, DelayedParameter):
            self._delayed_calls[partial(self.reorder, *args)] = order
            return

        indices = [order] + list(args) if isinstance(order, LoopIndex) else list(order)

        if len(indices) != len(self._indices):
            raise ValueError(
                f"Expected {len(self._indices)} indices, but got {len(indices)} indices instead"
            )

        indices = list(map(self._resolve_index, indices))

        visited = []
        for i in indices:
            if (
                self._index_map[i].parent
                and self._index_map[i].parent not in visited
                and self._index_map[i].transform
                and self._index_map[i].transform[0] is IndexTransform.SPLIT
            ):
                raise ValueError(
                    "An inner dimension must not be ordered before its outer dimension"
                )
            visited.append(i)

        self._indices = indices

    def tile(
        self, shape=Mapping[LoopIndex, Union[int, DelayedParameter]]
    ) -> Tuple[LoopIndex]:
        """The `tile` transformation is a convenience syntax that takes a dict of indices and sizes, and splits each index by the corresponding size.
        The indices involved in the split are then ordered such that all the outer indices precede all of their respective inner indices.

            ii, jj, kk = schedule.tile({i: 8, j: 2, k: 3})

        The tile transformation above is shorthand for the following sequence of transformations:

            ii = schedule.split(i, 8)

            jj = schedule.split(j, 2)

            kk = schedule.split(k, 3)

        Args:
            shape: Mapping of indices to tile sizes
        """
        try:
            names = varname(multi_vars=True)
        except:
            names = None

        # split for each index and it will automatically place the inner child index after its parent
        # self._indices is updated in-place.
        split_indices = [
            self.split(self._resolve_index(idx), factor)
            for idx, factor in shape.items()
        ]

        if names:
            zipped_name_index = zip(names, split_indices)
            for name, index in zipped_name_index:
                index._name = name

        return split_indices

    def is_valid_loop_order(self, loop_order: Tuple[LoopIndex]) -> bool:
        """This method is used to validate the order of parent index and inner index, inner index should precede its parant index.

        Args:
            loop_order: A tuple of loop index.

        """
        loop_order = list(loop_order)
        for index in loop_order:
            for inner_index in self._index_map[index].inners:
                if loop_order.index(index) > loop_order.index(inner_index):
                    return False
            if self._index_map[index].parent:
                if loop_order.index(index) < loop_order.index(
                    self._index_map[index].parent
                ):
                    return False
        return True

    def print(self, per_index_fn: Callable[[LoopIndex], List[str]] = None):
        indices = self.get_indices()
        for ordinal, index in enumerate(indices):
            attrs = per_index_fn(index) if per_index_fn else []
            start, stop, step = self.get_index_range(index)
            print(
                f"{'  ' * ordinal} for idx_{ordinal} in range({start}, {stop}, {step}):{'' if not attrs else (' # ' + ','.join(attrs))}"
            )

    def _build_native_context(self, context: NativeLoopNestContext):
        context.schedule = context.nest.create_schedule()

    def _build_with_native_context(self, context: NativeLoopNestContext):
        from .._lang_python._lang import Scalar

        def get_native_index(idx):
            if id(idx) in context.mapping:
                return context.mapping[id(idx)]

            if self._index_map[idx].parent:
                parent = self._index_map[idx].parent
                native_parent = get_native_index(parent)

                if self._index_map[idx].transform:
                    transform, params = self._index_map[idx].transform

                    if transform is IndexTransform.SPLIT:
                        factor = params
                        native_idx = Scalar(
                            context.schedule.split(native_parent, factor)
                        )
                    elif transform is IndexTransform.SKEW:
                        reference_index, unroll_loops_smaller_than = params
                        native_ref_idx = get_native_index(reference_index)
                        native_idx = Scalar(
                            context.schedule.skew(native_parent, native_ref_idx)
                        )
                        if unroll_loops_smaller_than:
                            context.schedule.unroll(
                                native_idx, unroll_loops_smaller_than
                            )
                            context.schedule.unroll(
                                native_ref_idx, unroll_loops_smaller_than
                            )
                    elif transform is IndexTransform.PAD:
                        size, pad_front = params
                        native_idx = Scalar(
                            context.schedule.pad(native_parent, size, pad_front)
                        )
                    else:
                        raise NotImplementedError(f"Unsupported transform {transform}")

                    context.mapping[id(idx)] = native_idx
                    return native_idx

            raise ValueError("Unknown index")

        native_idxs = list(map(get_native_index, self._indices))
        context.schedule.set_order(native_idxs)

    def _init_delayed(self, nest: Nest):
        shape = nest.get_shape()

        for index, size in zip(self._index_map, shape):
            self._index_map[index].stop = size

    def _split_delayed(self, index: LoopIndex, size: int):
        index = self._resolve_index(index)

        if not size or not float(size).is_integer() or size < 1:
            raise ValueError("Split sizes must be integers >= 1")

        inner_index = self._index_map[index].inners[0]

        start, stop, step = self._index_map[index].interval()
        interval = stop - start
        rem = interval % (step * size)
        self._index_map[index].step *= size
        self._index_map[inner_index].stop = size + rem
        self._index_map[inner_index].transform = IndexTransform.SPLIT, size

    def _pad_delayed(self, index: LoopIndex, front: bool, size: int):
        self._index_map[index].stop += size
        self._index_map[index].transform = IndexTransform.PAD, (size, front)

    def _skew_delayed(
        self,
        index: LoopIndex,
        reference_index: LoopIndex,
        unroll_loops_smaller_than: int,
    ):
        self._index_map[index].transform = IndexTransform.SKEW, (
            reference_index,
            unroll_loops_smaller_than,
        )

    # If this function is updated to return something, fused schedule needs to be updated as well
    def _replay_delayed_calls(self):
        """
        This method is called once per adding function, so it can be called multiple times when
        multiple functions get added. In order for the functions to be added correctly, we need to make sure all
        the residual states are cleared between different method calls.

        In Schedule class, we identify that Schedule._index_map can have residual states, so we need to reset self._index_map
        before we replay the delayed methods.
        """

        if self._delayed_calls:
            # Reset the index map to its pre-parameterized state before applying function-specific parameters
            if self._parameterized_index_map:
                self._index_map = self._deep_copy_index_map(
                    self._parameterized_index_map
                )
            else:
                self._parameterized_index_map = self._deep_copy_index_map(
                    self._index_map
                )

            for delayed_call in self._delayed_calls:
                params = self._delayed_calls[delayed_call]

                if isinstance(params, DelayedParameter):
                    delayed_call(params.get_value())
                else:
                    delayed_call(params)

    def _resolve_index(self, index):
        if index not in self._index_map:
            raise ValueError("Unknown index!")

        # Assume: 1:1 mapping between parent and child
        parent_child_map = {
            index_info.parent: i
            for i, index_info in self._index_map.items()
            if index_info.parent
        }

        if index not in set(self._indices) and index in parent_child_map:
            # This is an in-place transformed index (skew, pad), where the caller would
            # still hold a reference to the pre-transformed index.
            # If candidate == parent, replace with the child (=the actual transformed index)
            # BUGBUG (?): the concept of "parent" is overloaded to be the pre-transformed index
            actual_index = parent_child_map[index]

            # split does an in-place transform, but preserves the outer index's identity
            # while inserting a new index.
            assert (
                self._index_map[actual_index].transform
                and self._index_map[actual_index].transform[0]
                is not IndexTransform.SPLIT
            )
            return actual_index
        return index

    def _get_num_split_iterations(self, indices: List[LoopIndex]):
        result = 1
        for i in list(map(self._resolve_index, indices)):
            result *= self._index_map[i].num_iterations()
        return result

    def _deep_copy_index_map(self, index_map):
        index_map_copy = {}
        for index, entry in index_map.items():
            inners_copy = [idx for idx in entry.inners]
            index_map_copy[index] = IndexEntry(
                entry.stop,
                entry.start,
                entry.step,
                inners_copy,
                entry.parent,
                entry.transform,
            )

        return index_map_copy

    def _create_tensorizable_plan(self, target, block_indices, warp_indices, tensor_indices, outer_nest_order, dynamic_shared_memory_size: int = 0, blocks_per_SM: int = 0):
        i, j = block_indices
        ii, jj = warp_indices
        iii, jjj, kk = tensor_indices

        self.reorder(*outer_nest_order,
                            # note (ii, jj) together will map to which warp in the block is active
                            iii,
                            jjj,
                            kk)

        plan = self.create_plan(target=target, _dynamic_shared_memory_size=dynamic_shared_memory_size, _blocks_per_SM=blocks_per_SM)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_Y,
                j: target.GridUnit.BLOCK_X,
                ii: target.GridUnit.WARP_Y,
                jj: target.GridUnit.WARP_X
            }
        )
        return plan, tensor_indices


class FusedSchedule(Schedule):
    def __init__(self, schedules: List[Schedule], partial: int = None):

        s_indices = [s.get_indices() for s in schedules]

        # NOTE: This handles full fusing if partial == None
        partial = (
            max(len(indices) for indices in s_indices) if partial is None else partial
        )

        if any(partial > len(indices) for indices in s_indices):
            raise ValueError("Insufficient indices present in schedules for fusing")

        index_map: Mapping[LoopIndex, IndexEntry] = {}

        # fused index extent is number of schedules fused
        fusing_index = LoopIndex()
        index_map[fusing_index] = IndexEntry(stop=len(schedules))
        iteration_space = [len(schedules)]

        common_indices: List[LoopIndex] = []
        common_to_orig_map: Mapping[LoopIndex, List[LoopIndex]] = {}
        orig_to_common_map: Mapping[LoopIndex, LoopIndex] = {}

        for dim_indices in zip(*(idxs[:partial] for idxs in s_indices)):
            # compute the outer limits of the index ranges
            start = min(s.get_index_range(i)[0] for i, s in zip(dim_indices, schedules))
            stop = max(s.get_index_range(i)[1] for i, s in zip(dim_indices, schedules))
            step = schedules[0].get_index_range(dim_indices[0])[2]

            # pad indices so that all schedule dimensions are equal-sized
            for dim_idx, dim_sched in zip(dim_indices, schedules):
                start_d, stop_d, step_d = dim_sched.get_index_range(dim_idx)
                if start_d > start:
                    dim_sched.pad(dim_idx, start_d - start, _front=True)
                if stop_d < stop:
                    dim_sched.pad(dim_idx, stop - stop_d, _front=False)
                if step_d != step:
                    raise ValueError("Incompatible ranges for fusion")

            common_idx = LoopIndex()

            padded_indices = [
                s._resolve_index(i) for i, s in zip(dim_indices, schedules)
            ]
            for dim_idx in padded_indices:
                orig_to_common_map[dim_idx] = common_idx

            common_to_orig_map[common_idx] = list(padded_indices)
            common_indices.append(common_idx)

            # TODO: how do parent links propagate?

            index_map[common_idx] = IndexEntry(start=start, stop=stop, step=step)
            iteration_space.append(stop - start)

        unfused_indices: List[LoopIndex] = []
        unfused_idx_to_orig_sched_map: Mapping[LoopIndex, Schedule] = {}
        unfused_idx_to_orig_map: Mapping[LoopIndex, LoopIndex] = {}
        for s in schedules:
            for idx in s.get_indices()[partial:]:
                new_idx = LoopIndex()
                unfused_indices.append(new_idx)
                orig_to_common_map[idx] = new_idx

                start, stop, step = s.get_index_range(idx)
                orig_parent = s.get_index_parent(idx)

                # convert the common indices for transformations that involve index params
                orig_xf = s.get_index_transform(idx)
                xf = None
                if orig_xf:
                    xf = tuple()
                    for p in orig_xf:
                        xf = xf + (orig_to_common_map[p] if isinstance(p, LoopIndex) else p,)
                index_map[new_idx] = IndexEntry(
                    start=start,
                    stop=stop,
                    step=step,
                    parent=(orig_to_common_map[orig_parent]) if orig_parent else None,
                    transform=xf,
                )
                unfused_idx_to_orig_sched_map[new_idx] = s
                unfused_idx_to_orig_map[new_idx] = idx

        fused_nest = Nest(iteration_space)
        # TODO: this is hacky
        iteration_space_indices = [fusing_index] + common_indices
        for i, dim in enumerate(fused_nest._shape):
            fused_nest._shape[i] = (dim[0], iteration_space_indices[i])
        super().__init__(fused_nest)

        self._index_map = index_map
        self._schedules = schedules
        self._fusing_index = fusing_index
        self._common_indices = common_indices
        self._common_to_orig_map = common_to_orig_map
        self._orig_to_common_map = orig_to_common_map
        self._unfused_indices = unfused_indices
        self._unfused_idx_to_orig_sched_map = unfused_idx_to_orig_sched_map
        self._unfused_idx_to_orig_map = unfused_idx_to_orig_map

        self._indices = (
            [self._fusing_index] + self._common_indices + self._unfused_indices
        )

    def print(self, per_index_fn: Callable[[LoopIndex], List[str]] = None):
        # TODO
        ...

    def get_fusing_index(self):
        return self._fusing_index

    def get_fused_indices(self):
        return self._common_indices

    def get_unfused_indices(self):
        return self._unfused_indices

    def reorder(
        self, order: Union[Tuple[LoopIndex], LoopIndex] = None, *args: LoopIndex
    ):
        indices = [order] + list(args) if isinstance(order, LoopIndex) else list(order)

        # handle split unfused indices
        unfused_base_indices = [i.base_index for i in self._unfused_indices]

        visited = []
        for i in indices:
            if (
                i.base_index in unfused_base_indices
                and self._fusing_index not in visited
            ):
                raise ValueError(
                    "An unfused dimension must not be ordered before a fusing dimension"
                )
            visited.append(i)

        super().reorder(order, *args)

    def _replay_delayed_calls(self):
        for s in self._schedules:
            s._replay_delayed_calls()

        return super()._replay_delayed_calls()

    def _build_native_context(self, context: NativeLoopNestContext):
        # By the time we've reached here, our fused nest has already been created

        sched_contexts: Mapping[Schedule, NativeLoopNestContext] = {}
        for s in self._schedules:
            contained_context = NativeLoopNestContext(
                function_args=context.function_args, runtime_args=context.runtime_args
            )
            contained_context.mapping.update(context.mapping)

            s._nest._build_native_context(contained_context)
            s._build_native_context(contained_context)

            # TODO: remove these lines
            s._nest._build_with_native_context(contained_context)
            s._build_with_native_context(contained_context)

            sched_contexts[s] = contained_context

            context.mapping.update(contained_context.mapping)

        scheds = self._schedules
        native_scheds = [sched_contexts[s].schedule for s in scheds]
        mappings = [sched_contexts[s].mapping for s in scheds]

        # list of list of indices
        orig_indices = [
            self._common_to_orig_map[common_idx] for common_idx in self._common_indices
        ]
        zipped_native_indices: list[list] = []
        for zipped_list in orig_indices:
            assert len(zipped_list) == len(mappings)

            native_indices: list = []
            for index, mapping in zip(zipped_list, mappings):
                native_indices.append(mapping[id(index)])
            zipped_native_indices.append(native_indices)

        native_prime_sched, native_other_scheds = native_scheds[0], native_scheds[1:]
        fused_native_index = native_prime_sched.fuse(
            native_other_scheds, zipped_native_indices
        )
        all_indices_post_fusion = native_prime_sched.get_indices()

        if len(all_indices_post_fusion) != 1 + len(self._common_indices) + len(
            self._unfused_indices
        ):
            return ValueError("Unexpected number of indices returned from fusion")

        context.mapping[id(self._fusing_index)] = fused_native_index

        for i, common_idx in enumerate(self._common_indices):
            context.mapping[id(common_idx)] = all_indices_post_fusion[1 + i]

        unfused_begin = 1 + len(self._common_indices)
        for i, unfused in enumerate(self._unfused_indices):
            context.mapping[id(unfused)] = all_indices_post_fusion[unfused_begin + i]

        context.schedule = native_prime_sched


def fuse(
    scheds: Union[Tuple[Schedule], Schedule], *args: Schedule, partial: int = None
) -> FusedSchedule:
    """The `fuse` operation combines multiple iteration spaces into a single "fused" iteration space.
    The fused iteration space represents the union of the work in the original spaces.

    In cases where it doesn't make sense to fuse all of the iteration space dimensions, we can choose
    to fuse a prefix of the dimensions and leave the rest unfused.

    Args:
        schedules: Either the schedules to fuse if performing partial fusing, or the first schedule to fuse if fusing all dimensions
        *args: Optional variable arguments containing subsequent schedules to fuse
        partial: The number of dimensions to fuse. If not specified, all dimensions will be fused
    """
    schedules = [scheds] + list(args) if isinstance(scheds, Schedule) else list(scheds)
    return FusedSchedule(schedules, partial)
