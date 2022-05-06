####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import Sequence, NamedTuple
from accera import Array, Target


class Options(NamedTuple):
    ForceCacheBMatrix = False
    BCacheSizeThreshold = 128**2
    KUnroll = 4
    NumRowsInKernel = 6
    NumColumnsInKernelScaleFactor = 2
    BMatrixTileSize: Sequence[int] = [256, 128]


def RuntimeInitCacheMLAS(
    A: Array,
    B: Array,
    C: Array,
    pack_fn_name: str,
    packed_buffer_size_fn_name: str,
    opts=Options(),
    target=Target.HOST
):
    from accera import Nest

    if not all([len(array.shape) == 2 for array in [A, B, C]]):
        raise RuntimeError("Invalid shapes for arguments")

    _M_A, _K_A = A.shape
    _K_B, _N_B = B.shape
    _M_C, _N_C = C.shape

    if _M_A != _M_C or _K_A != _K_B or _N_B != _N_C:
        raise RuntimeError("Incompatible shapes for arguments")

    M = _M_C
    N = _N_C
    K = _K_A
    output_rows = M
    output_cols = N
    inner_dim = K

    # Schedule constants
    column_block = opts.BMatrixTileSize[1]
    inner_dim_block = opts.BMatrixTileSize[0]
    num_rows_in_kernel = opts.NumRowsInKernel
    num_cols_in_kernel = opts.NumColumnsInKernelScaleFactor * (
        target.vector_bytes // 4 or 8
    )    # (target.vector_bytes // 4) is how many 32-bit float elements can fit into the vector register

    # Apply a simple stretching to the kernel size to fit the output shape
    if num_cols_in_kernel > output_cols:
        while num_cols_in_kernel > output_cols:
            num_rows_in_kernel *= 2
            num_cols_in_kernel /= 2
    elif num_rows_in_kernel > output_rows:
        while num_rows_in_kernel > output_rows:
            num_rows_in_kernel /= 2
            num_cols_in_kernel *= 2

    # now clamp
    num_rows_in_kernel = int(min(num_rows_in_kernel, output_rows))
    num_cols_in_kernel = int(min(num_cols_in_kernel, output_cols))

    # Apply a simple stretching to the block sizes to use as much of
    # the original columnBlock x innerDimensionBlock area as possible
    while column_block > output_cols:
        if (column_block / 2) < num_cols_in_kernel:
            # Don't shrink the column block smaller than num_cols_in_kernel
            break
        column_block /= 2
        inner_dim_block *= 2
    while inner_dim_block > inner_dim:
        inner_dim_block /= 2
        column_block *= 2

    # Now clamp
    column_block = int(min(column_block, output_cols))
    inner_dim_block = int(min(inner_dim_block, inner_dim))

    nest = Nest(shape=(M, N, K))
    i, j, k = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    schedule = nest.create_schedule()

    jj = schedule.split(j, column_block)
    kk = schedule.split(k, inner_dim_block)
    kkk = schedule.split(kk, opts.KUnroll)
    jjj = schedule.split(jj, num_cols_in_kernel)
    jjjj = schedule.split(
        jjj, target.vector_bytes // 4 or 8
    )    # (target.vector_bytes // 4) is how many 32-bit float elements can fit into the vector register
    ii = schedule.split(i, num_rows_in_kernel)

    schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

    plan = schedule.create_plan()

    plan.emit_runtime_init_pack(B, pack_fn_name, packed_buffer_size_fn_name)
    plan.cache(C, ii)

    plan.unroll(jjj)
    plan.unroll(ii)
    plan.vectorize(jjjj)

    return plan, (A, B, C)


def EmitTimeCacheMLAS(A: Array, B: Array, C: Array, opts=Options(), wrapper_fn_name: str = "", target=Target.HOST):
    from accera import Nest

    if not all([len(array.shape) == 2 for array in [A, B, C]]):
        raise RuntimeError("Invalid shapes for arguments")

    _M_A, _K_A = A.shape
    _K_B, _N_B = B.shape
    _M_C, _N_C = C.shape

    if _M_A != _M_C or _K_A != _K_B or _N_B != _N_C:
        raise RuntimeError("Incompatible shapes for arguments")

    M = _M_C
    N = _N_C
    K = _K_A
    output_rows = M
    output_cols = N
    inner_dim = K

    # Schedule constants
    column_block = opts.BMatrixTileSize[1]
    inner_dim_block = opts.BMatrixTileSize[0]
    num_rows_in_kernel = opts.NumRowsInKernel
    num_cols_in_kernel = opts.NumColumnsInKernelScaleFactor * (
        target.vector_bytes // 4 or 8
    )    # (target.vector_bytes // 4) is how many 32-bit float elements can fit into the vector register

    # Apply a simple stretching to the kernel size to fit the output shape
    if num_cols_in_kernel > output_cols:
        while num_cols_in_kernel > output_cols:
            num_rows_in_kernel *= 2
            num_cols_in_kernel /= 2
    elif num_rows_in_kernel > output_rows:
        while num_rows_in_kernel > output_rows:
            num_rows_in_kernel /= 2
            num_cols_in_kernel *= 2

    # now clamp
    num_rows_in_kernel = int(min(num_rows_in_kernel, output_rows))
    num_cols_in_kernel = int(min(num_cols_in_kernel, output_cols))

    # Apply a simple stretching to the block sizes to use as much of
    # the original columnBlock x innerDimensionBlock area as possible
    while column_block > output_cols:
        if (column_block / 2) < num_cols_in_kernel:
            # Don't shrink the column block smaller than num_cols_in_kernel
            break
        column_block /= 2
        inner_dim_block *= 2
    while inner_dim_block > inner_dim:
        inner_dim_block /= 2
        column_block *= 2

    # Now clamp
    column_block = int(min(column_block, output_cols))
    inner_dim_block = int(min(inner_dim_block, inner_dim))

    nest = Nest(shape=(M, N, K))
    i, j, k = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    schedule = nest.create_schedule()

    jj = schedule.split(j, column_block)
    kk = schedule.split(k, inner_dim_block)
    kkk = schedule.split(kk, opts.KUnroll)
    jjj = schedule.split(jj, num_cols_in_kernel)
    jjjj = schedule.split(
        jjj, target.vector_bytes // 4 or 8
    )    # (target.vector_bytes // 4) is how many 32-bit float elements can fit into the vector register
    ii = schedule.split(i, num_rows_in_kernel)

    schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

    plan = schedule.create_plan(target)

    # BUGBUG: TODO: need to resolve constant reference for B during _pack_and_embed_buffer
    # Assertion failed: !symbolRef.empty() && "expected valid symbol reference", file C:\dev\external\llvm-project\mlir\lib\IR\AsmPrinter.cpp, line 1256
    # plan.pack_and_embed_buffer(B, wrapper_fn_name)
    plan.cache(C, ii)

    plan.unroll(jjj)
    plan.unroll(ii)
    plan.vectorize(jjjj)

    return plan, (A, B, C)
