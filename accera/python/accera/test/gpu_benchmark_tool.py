#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import csv
import hatlib
from io import TextIOWrapper
from itertools import combinations_with_replacement, product
import os
from typing import List, Tuple
from accera import Array, Nest, Constants, ScalarType, Target, Package, Scalar, Schedule, Plan
import argparse
from dataclasses import dataclass, asdict
import json
import sys

from requests import head


@dataclass
class GemmOpts:
    M: int = 0
    N: int = 0
    K: int = 0
    type: str = ''
    transA: bool = False
    transB: bool = False
    alpha: float = 1.
    beta: float = 0.


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
    plan.tensorize(indices=(iii, jjj, kkk))

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
    M = opts.M
    N = opts.N
    K = opts.K
    alpha = opts.alpha
    beta = opts.beta
    transA = opts.transA
    transB = opts.transB

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


def benchmark_gemm(f: TextIOWrapper, opts:GemmOpts, target:Target, output_dir:str, suffix:str):
    FLOAT_BYTES = 4
    M = opts.M
    N = opts.N
    K = opts.K

    func_def_headers = ["M", "N", "K", "transA", "transB", "alpha", "beta", "type"]
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
        func_def_headers += ["outer_tiles", "mfma_tiles"]

    elif target.runtime == Target.Runtime.CUDA:
        outer_tiles = [16, 32, 64, 128]
        variants = combinations_with_replacement(outer_tiles, 2)
        func_def_headers += ["outer_tiles"]

    headers = func_def_headers + exec_headers + benchmark_headers

    w = csv.DictWriter(f, headers, '-')
    w.writeheader()

    nest, (A, B, C) = create_gemm_nest_args(opts)

    for variant in variants:
        entry = asdict(opts)

        package = Package()

        try:
            assert opts.alpha == 1., "alpha must be 1"
            assert opts.beta == 0, "beta must be 0"

            if target.runtime == Target.Runtime.ROCM:
                (outer_tile_x, outer_tile_y, outer_tile_k), mfma_tile = variant

                entry['outer_tiles'] = lstr(variant[0])
                entry['mfma_tiles'] = lstr(mfma_tile)
                fn_name = f"ROCM_GEMM_{M}_{N}_{K}_{outer_tile_x}_{outer_tile_y}_{outer_tile_k}_{'_'.join(list(map(str, mfma_tile)))}"
                plan = _rocm_tensorize_fp32(
                    target, nest.create_schedule(), (A, B, C), outer_tile_x, outer_tile_y, outer_tile_k, mfma_tile
                )

            elif target.runtime == Target.Runtime.CUDA:
                outer_tile_x, outer_tile_y = variant

                entry['outer_tiles'] = lstr(variant)

                fn_name = f"CUDA_GEMM_{M}_{N}_{K}_{outer_tile_x}_{outer_tile_y}"
                plan = _cuda_fp32(target, nest.create_schedule(), (A, B, C), outer_tile_x, outer_tile_y)

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

        w.writerow(entry)


def benchmark_gemm_shapes(data: List[GemmOpts], target: Target, output_dir:str, prefix:str=""):
    for gemm in data:
        suffix = f"{gemm.M}_{gemm.N}_{gemm.K}_{gemm.alpha}_{gemm.beta}_{gemm.transA}_{gemm.transB}_{gemm.type}"

        with open(os.path.join(output_dir, f"results_{suffix}.csv"), "w", newline='') as results_file:
            benchmark_gemm(results_file, gemm, target, output_dir, suffix)


def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The input config file', default='gpu_benchmark_config.json')
    parser.add_argument('-t', '--target', help='The target the emitter is emitting HAT package for')
    parser.add_argument('-o', '--output', help='The output directory', default=None)
    args = parser.parse_args(args)

    with open(args.input) as f:
        data = [GemmOpts(**data) for data in json.load(f)]

    output_dir = args.output or os.getcwd()
    target = Target(args.target)

    benchmark_gemm_shapes(data, target, output_dir, "results_")


if __name__ == "__main__":
    main(sys.argv[1:])
