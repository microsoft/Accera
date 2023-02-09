# Runtime-sized implementation of ReduceMean
# Prototype only - meant to illustrate future versions of Accera
# cf. onnx2exe/reduce_mean.py

import accera as acc
from typing import List, Tuple

# https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
op_type = "ReduceMean"


def _get_axes(axes, input_shape):
    if not axes:    # default is to reduce along all dimensions
        axes = range(len(input_shape))
    return axes


def _get_out_indices(indices, axes: List[int], keepdims: int, zero_idx):
    out_indices = indices.copy()
    for i in axes:
        if keepdims == 1:
            out_indices[i] = zero_idx
        else:
            out_indices[i] = -1
    else:
        out_indices = list(filter((-1).__ne__, out_indices))
        if len(out_indices) == 0:
            out_indices = [zero_idx]

    return out_indices


def _get_zero_index_nest(shape: Tuple[acc.Dimension]):
    nest = acc.Nest(shape=[1] + shape)    # padded by 1 in the beginning as a noop loop
    indices = nest.get_indices()
    zero_idx = indices[0]
    indices.pop(0)
    return nest, indices, zero_idx


def get_output_shape(input_shape: Tuple[acc.Dimension], axes: List[int], keepdims: int = 1):

    output_shape = tuple(acc.Dimension(role=acc.Role.OUTPUT, value=i) for i in input_shape)

    # assign number to each Dimension
    # list filtering (comparison operators)
    for i in axes:
        if keepdims == 1:
            output_shape[i] = 1
        else:
            output_shape[i] = -1
    else:
        output_shape = list(filter((-1).__ne__, output_shape))
        if len(output_shape) == 0:
            output_shape = [acc.Dimension(role=acc.Role.OUTPUT, value=1)]
    return output_shape


def _get_dim_prod(input_shape: Tuple[acc.Dimension], axes: List[int]):
    dim_prod = 1
    for i in axes:
        dim_prod *= input_shape[i]
    return dim_prod


def _init_buffer(buffer: acc.Array):
    nest_1 = acc.Nest(shape=buffer.shape)
    indices_1 = nest_1.get_indices()

    @nest_1.iteration_logic
    def _():
        buffer[indices_1] = acc._cast(0, buffer.element_type)

    return nest_1.create_schedule()


def _accumulate_buffer(data: acc.Array, reduced: acc.Array, axes: List[int], keepdims: int):
    nest0, indices0, zero_idx = _get_zero_index_nest(data.shape)
    out_indices = _get_out_indices(indices0, axes, keepdims, zero_idx)

    @nest0.iteration_logic
    def _():
        reduced[out_indices] += data[indices0]

    return nest0.create_schedule()


def _div_buffer(buffer: acc.Array, dim_count):
    nest1 = acc.Nest(shape=buffer.shape)
    indices1 = nest1.get_indices()

    @nest1.iteration_logic
    def _():
        buffer[indices1] /= acc._cast(dim_count, buffer.element_type)

    return nest1.create_schedule()


def _reorder_schedule(schedule, axes: List[int], keepdims: int, rank: int):
    indices = schedule.get_indices()
    if keepdims == 0 and len(indices) < rank:
        return

    non_reduced = []
    reduced = []
    if len(indices) > rank:
        # this mean we prepended a noop loop
        reduced.append(0)
        for i in range(1, len(indices)):
            if (i - 1) not in axes:
                non_reduced.append(i)
            else:
                reduced.append(i)
    else:
        for i in range(len(indices)):
            if i not in axes:
                non_reduced.append(i)
            else:
                reduced.append(i)

    reordered = []
    for i in non_reduced:
        reordered.append(indices[i])

    for i in reduced:
        reordered.append(indices[i])

    schedule.reorder(reordered)


def reduce_mean_schedule(data: acc.Array, reduced: acc.Array, axes: List[int] = None, keepdims: int = 1):
    input_shape = data.shape
    axes = _get_axes(axes, input_shape)
    dim_prod = _get_dim_prod(input_shape, axes)
    input_rank = len(input_shape)

    # TODO: optimization: move reduction dimensions to the right and fuse the non-reduced dimensions
    schedule_1 = _init_buffer(reduced)
    _reorder_schedule(schedule_1, axes, keepdims, input_rank)
    schedule0 = _accumulate_buffer(data, reduced, axes, keepdims)
    _reorder_schedule(schedule0, axes, keepdims, input_rank)
    schedule1 = _div_buffer(reduced, dim_prod)
    _reorder_schedule(schedule1, axes, keepdims, input_rank)

    # BUG: Currently, concat fusing is not supported for >2 schedules
    fused_dims = input_rank - len(axes)
    schedule00 = acc.fuse((schedule_1, schedule0, schedule1), partial=fused_dims)
    return schedule00    #acc.fuse((schedule00, schedule1), partial=0)


def reduce_mean_rank_3(dtype: acc.ScalarType, axes: List[int] = None, keepdims: int = 1):
    # Runtime sizes with fixed rank
    input_shape = tuple(acc.Dimension() for _ in range(3))

    axes = _get_axes(axes, input_shape)
    output_shape = get_output_shape(input_shape, axes, keepdims)
    data = acc.Array(role=acc.Role.INPUT, element_type=dtype, shape=input_shape)
    reduced = acc.Array(role=acc.Role.OUTPUT, element_type=dtype, shape=output_shape)

    schedule = reduce_mean_schedule(data, reduced, axes, keepdims)

    package = acc.Package()

    # Generate a function like:
    #
    # ReduceMean_rank_3(float* data, int64_t data_dim0, int64_t data_dim1, int64_t data_dim2,
    #   float** reduced, int64_t* reduced_dim0, ...);
    #
    # where the number of reduced_dimN is determined at compile time by axes and keepdims
    # (this keeps the ranks constant but the dimensions variable)
    #
    # This function dynamically allocates memory for the output pointer, which must
    # be released by the caller. The caller must also provide an implementation of:
    #     void _accera_allocate(void** memory, size_t bytes);
    #
    args = tuple(data) + input_shape + tuple(reduced) + output_shape

    package.add(schedule, args=args, base_name=f"{op_type}_rank_3")

    # TODO: package.build, etc to return a hat package
