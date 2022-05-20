####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import math
import os
import hatlib
from dataclasses import asdict
from itertools import combinations_with_replacement, product
from typing import Any, Dict, List, Tuple
from accera import Array, Nest, Constants, ScalarType, Target, Package, Schedule, Plan
from accera._lang_python._lang import _MMAShape

from gemm_opts import GemmOpts

def _cuda_fp32(
    target: Target, schedule: Schedule, args: Tuple[Array, Array, Array], outer_tile_x: int, outer_tile_y: int
) -> Plan:
    A, B, C = args
    i, j, k = schedule.get_indices()

    ii, jj = schedule.tile({
        i: outer_tile_x,
        j: outer_tile_y
    })

    schedule.reorder((i, j, ii, jj, k))

    plan = schedule.create_plan(target=target)

    # plan.cache(
    #     A,
    #     index=ii,
    #     double_buffer=True,
    #     double_buffer_location=Constants.AUTO,
    #     vectorize=True,
    #     location=target.MemorySpace.SHARED,
    #     layout=Array.Layout.FIRST_MAJOR
    # )
    # plan.cache(
    #     B,
    #     index=ii,
    #     double_buffer=True,
    #     double_buffer_location=Constants.AUTO,
    #     vectorize=True,
    #     location=target.MemorySpace.SHARED,
    #     layout=Array.Layout.FIRST_MAJOR
    # )

    plan.bind(
        mapping={
            i: target.GridUnit.BLOCK_Y,
            j: target.GridUnit.BLOCK_X,
            ii: target.GridUnit.THREAD_Y,
            jj: target.GridUnit.THREAD_X
        }
    )
    return plan


def _rocm_tensorize_fp32(
    target: Target, schedule: Schedule, args: Tuple[Array, Array, Array], outer_tile_x: int, outer_tile_y: int,
    outer_tile_k: int, mfma_tile: int
) -> Plan:
    A, B, C = args
    i, j, k = schedule.get_indices()

    ii, jj, kk = schedule.tile({
        i: outer_tile_x,
        j: outer_tile_y,
        k: outer_tile_k
    })
    iii, jjj, kkk = schedule.tile({
        ii: mfma_tile[0],
        jj: mfma_tile[1],
        kk: mfma_tile[2]
    })

    schedule.reorder((i, j, k, ii, jj, kk, iii, jjj, kkk))

    plan = schedule.create_plan(target=target)
    plan.bind(
        mapping={
            i: target.GridUnit.BLOCK_Y,
            j: target.GridUnit.BLOCK_X,
            ii: target.GridUnit.THREAD_Y,
            jj: target.GridUnit.THREAD_X
        }
    )
    plan.tensorize(indices=(iii, jjj, kkk), mma_shape=_MMAShape.T2x2x16, num_total_passes=4)

    plan.cache(
        A,
        index=ii,
        double_buffer=True,
        double_buffer_location=Constants.AUTO,
        vectorize=True,
        location=target.MemorySpace.SHARED,
        layout=Array.Layout.FIRST_MAJOR
    )
    plan.cache(
        B,
        index=ii,
        double_buffer=True,
        double_buffer_location=Constants.AUTO,
        vectorize=True,
        location=target.MemorySpace.SHARED,
        layout=Array.Layout.FIRST_MAJOR
    )

    return plan


def create_gemm_nest_args(opts:GemmOpts):
    Layout = Array.Layout
    M = int(opts.m)
    N = int(opts.n)
    K = int(opts.k)
    alpha = float(opts.alpha)
    beta = float(opts.beta)
    transA = bool(opts.transA)
    transB = bool(opts.transB)

    A = Array(
        role=Array.Role.INPUT,
        element_type=ScalarType.float32,
        shape=(M, K),
        layout=Layout.LAST_MAJOR if transA else Layout.FIRST_MAJOR
    )
    B = Array(
        role=Array.Role.INPUT,
        element_type=ScalarType.float32,
        shape=(K, N),
        layout=Layout.LAST_MAJOR if transB else Layout.FIRST_MAJOR
    )
    C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

    nest = Nest([M, N, K])
    i, j, k = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    return nest, (A, B, C)


def benchmark_gemm(opts:GemmOpts, target:str, output_dir:str):
    """
    Interfaces with the gpu benchmark tool

    Returns a list, where the first item is a list of *additional* keys available in the subsequent dictionary objects. 
    These dictionary objects are dictionaries of benchmarking results for a number of variations of the specified GEMM config.
    The keys that are always present are:

    "m", "n", "k", "transA", "transB", "alpha", "beta", "type", "time_ms", "gflops"
    """
    target:Target = Target(target)

    FLOAT_BYTES = 4
    M = str(opts.m)
    N = str(opts.n)
    K = str(opts.k)
    alpha = str(opts.alpha).replace('.', '_')
    beta = str(opts.beta).replace('.', '_')
    transA = str(int(opts.transA))
    transB = str(int(opts.transB))

    exec_headers = ["compilable", "executable"]

    benchmark_headers = ["function_name", "mean", "median_of_means", "mean_of_small_means", "robust_mean", "min_of_means"]

    def lstr(l):
        return '/'.join(list(map(str, l)))

    if target.runtime == Target.Runtime.ROCM:
        outer_tiles = [16, 32, 64, 128]
        mfma_tiles = [(2, 2, 16), (4, 4, 32), (2, 32, 64), (4, 16, 64)]

        def valid_variant(outer_t, mfma_t):
            return (outer_t[0] % mfma_t[2] == 0) and (outer_t[1] % mfma_t[2] == 0) and \
                (2 * outer_t[0] * outer_t[1] * FLOAT_BYTES <= target.max_shared_memory_per_block)

        variants = filter(lambda v:valid_variant(v[0], v[1]), product(combinations_with_replacement(outer_tiles, 3), mfma_tiles))

    elif target.runtime == Target.Runtime.CUDA:
        outer_tiles = [16, 32, 64, 128]
        variants = combinations_with_replacement(outer_tiles, 2)

    entries:List[Dict[str, Any]] = []

    nest, (A, B, C) = create_gemm_nest_args(opts)

    for variant in variants:
        entry = {
            **asdict(opts),
            **dict.fromkeys(exec_headers, False),
            **dict.fromkeys(benchmark_headers, '-')
        }
        package = Package()

        try:
            if target.runtime == Target.Runtime.ROCM:
                (outer_tile_x, outer_tile_y, outer_tile_k), mfma_tile = variant

                entry['outer_tiles'] = lstr(variant[0])
                entry['mfma_tiles'] = lstr(mfma_tile)
                fn_name = f"ROCM_GEMM_{M}_{N}_{K}_{transA}_{transB}__{alpha}__{beta}__{outer_tile_x}_{outer_tile_y}_{outer_tile_k}_{'_'.join(list(map(str, mfma_tile)))}"
                plan = _rocm_tensorize_fp32(
                    target, nest.create_schedule(), (A, B, C), outer_tile_x, outer_tile_y, outer_tile_k, mfma_tile
                )

            elif target.runtime == Target.Runtime.CUDA:
                outer_tile_x, outer_tile_y = variant

                entry['outer_tiles'] = lstr(variant)

                fn_name = f"CUDA_GEMM_{M}_{N}_{K}_{transA}_{transB}__{alpha}__{beta}__{outer_tile_x}_{outer_tile_y}"
                plan = _cuda_fp32(target, nest.create_schedule(), (A, B, C), outer_tile_x, outer_tile_y)

            # TODO: add more supported types/values
            assert opts.type == 's', "Can only handle single-precision floats"
            assert float(opts.alpha) == 1., "alpha must be 1"
            assert float(opts.beta) == 0, "beta must be 0"

            package_name = f"gemm_benchmarks_{fn_name}"
            print(f"Adding {fn_name}")
            package.add(plan, args=(A, B, C), base_name=fn_name)
            package.build(
                package_name,
                output_dir=output_dir,
                fail_on_error=True,
                _quiet=False
            )
        except:
            print(f"Error while building {fn_name}")
            entry['compilable'] = False
        else:
            entry['compilable'] = True

            try:
                results = hatlib.run_benchmark(
                    os.path.join(output_dir, package_name + ".hat")
                )
            except:
                entry['executable'] = False
            else:
                entry['executable'] = len(results) > 0
                if len(results) > 0:
                    entry.update(results[0])

                time_secs = entry.get('mean', '-')
                if time_secs != '-':
                    time_ms = float(time_secs) * 1e3
                    flopms = (2 * opts.k + 2) * opts.m * opts.n / time_ms
                    flops = flopms * 1e3
                    gflops = flops * 1e-9

                    entry['time_ms'] = time_ms
                    entry['gflops'] = gflops

        entry.setdefault('time_ms', math.nan)
        entry.setdefault('gflops', math.nan)
        entries.append(entry)

    return entries
