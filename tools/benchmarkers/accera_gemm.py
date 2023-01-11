####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import multiprocessing
import os
import psutil
from copy import deepcopy
from functools import reduce
import progressbar
from termcolor import colored
from dataclasses import dataclass, InitVar, asdict
from datetime import datetime
import time
from itertools import combinations_with_replacement, product
from typing import Any, Dict, List, Tuple
import numpy as np
import hatlib
from accera import Array, Nest, Constants, ScalarType, Target, Package
from accera._lang_python._lang import _MMASchedulingPolicy, _MMAShape, _MMAFragmentOp, _CacheStrategy
import cosmosdb
from gemm_opts import GemmOpts
import math

@dataclass
class BenchmarkResult:
    opts: InitVar[GemmOpts]
    dtype: InitVar[str]
    gpu_id: int
    commit_id: str
    commit_datetime: str
    commit_branch: str
    target_name: str
    deviceProperties: str
    M: int = -1
    N: int = -1
    K: int = -1
    alpha: float= -1.0
    beta: float= -1.0
    trans_A: bool=False
    trans_B: bool=False
    in_type: str=''
    out_type: str=''
    mma_shape: str=''
    use_static_offsets: bool=False
    cache_layout_A: str = ''
    cache_layout_B: str = ''
    cache_strategy_A: str = ''
    cache_strategy_B: str = ''
    cache_C: bool=False
    block_tile: Tuple[int, int]=(0, 0)
    thread_coarsening_factor: Tuple[int, int]=(1, 1)
    k_split: int=-1
    double_buffering: bool=False
    vectorize: bool=False
    num_total_passes: int=-1
    num_fused_passes: int=-1
    scheduling_policy: int=-1
    target_rt: str=''
    time_ms: float= 0.0
    TFlops: float= 0.0
    dt: str= datetime.now().ctime()
    compiler_version: str=''
    category: str=''
    compilable: bool=False
    executable: bool=False
    check: bool=False
    correct: bool=False
    kernelCode: str=''
    prog_out: str=''

    def __post_init__(self, opts, dtype):
        if opts is None or self.gpu_id is None or self.commit_id is None or self.commit_datetime is None \
            or self.commit_branch is None or self.target_name is None or self.deviceProperties is None:
            raise ValueError("One of the mandatory parameters were not set.")

        self.M=opts.m
        self.N=opts.n
        self.K=opts.k
        self.alpha=opts.alpha
        self.beta=opts.beta
        self.trans_A=opts.transA
        self.trans_B=opts.transB
        self.in_type=dtype
        self.out_type=dtype

    # Encode all the input arguments to gemm into the partitionKey for efficient queries
    def get_partition_key(self) -> str:
        key_info = {
            'm' : self.M,
            'n' : self.N,
            'k' : self.K,
            'a' : self.alpha,
            'b' : self.beta,
            'i' : self.in_type,
            'o' : self.out_type,
            'ta' : self.trans_A,
            'tb' : self.trans_B,
            'tg' : self.target_name.replace(' ', '_')
        }
        return "_".join([f"{key}{str(key_info[key])}" for key in key_info])

    # Encode all the optimizations into the ID for uniqueness of record
    def get_id(self) -> str:
        key_info = {
            'mma' : self.mma_shape,
            'so' : self.use_static_offsets,
            'ca' : self.cache_layout_A,
            'cb' : self.cache_layout_B,
            'sa' : self.cache_strategy_A,
            'sb' : self.cache_strategy_B,
            'cc' : self.cache_C,
            'bt0' : self.block_tile[0],
            'bt1' : self.block_tile[1],
            'tc0' : self.thread_coarsening_factor[0],
            'tc1' : self.thread_coarsening_factor[1],
            'ks' : self.k_split,
            'db' : self.double_buffering,
            'v' : self.vectorize,
            'tp' : self.num_total_passes,
            'fp' : self.num_fused_passes,
            'sp' : self.scheduling_policy,
            'git' : self.commit_id
        }
        return "_".join([f"{key}{str(key_info[key])}" for key in key_info])

    def get_result_row(self) -> Dict[str, Any]:
        d = self.__dict__
        d['id'] = self.get_id()
        d['partitionKey'] = self.get_partition_key()
        return d

def print_log(verbose: bool, msg: str, color: str = None):
    if verbose:
        if not color:
            print(msg)
        else:
            print(colored(msg, color))

def get_k(target: Target, mma_shape: _MMAShape):
    mma_shape = target.tensor_core_info.mma_shape_to_tuple(mma_shape)
    return mma_shape[2]

def get_m(target: Target, mma_shape: _MMAShape):
    mma_shape = target.tensor_core_info.mma_shape_to_tuple(mma_shape)
    return mma_shape[0]

def get_n(target: Target, mma_shape: _MMAShape):
    mma_shape = target.tensor_core_info.mma_shape_to_tuple(mma_shape)
    return mma_shape[1]

def get_layout(transpose: bool):
    return Array.Layout.LAST_MAJOR if transpose else Array.Layout.FIRST_MAJOR

def get_type(typeStr: str):
    return ScalarType.float32 if typeStr == 's' else ScalarType.float16

def single_block_mma(mma_shape: _MMAShape):
    return mma_shape != _MMAShape.M64xN64xK1_B4 and mma_shape != _MMAShape.M64xN64xK1_B2 and mma_shape != _MMAShape.M64xN64xK4_B4 and mma_shape != _MMAShape.M64xN64xK4_B2

def create_gemm_nest_args(M: int, N: int, K: int, transA: bool, transB: bool, dtype):
    datatype = get_type(dtype)

    A = Array(
        role=Array.Role.INPUT,
        element_type=datatype,
        shape=(M, K),
        layout=get_layout(transA)
    )
    B = Array(
        role=Array.Role.INPUT,
        element_type=datatype,
        shape=(K, N),
        layout=get_layout(transB)
    )
    C = Array(role=Array.Role.INPUT_OUTPUT, element_type=datatype, shape=(M, N))

    nest = Nest([M, N, K])
    i, j, k = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    return nest, (A, B, C)

def benchmark_kernel(
    target: Target, M: int, N: int, K: int, transA: bool, transB: bool, dtype, outer_tile_m: int, outer_tile_n: int,
    outer_tile_k: int, cacheA_layout, cacheB_layout, cache_C: bool, cacheA_strategy: _CacheStrategy, cacheB_strategy: _CacheStrategy,
    mma_shape, use_static_offsets, double_buffering, vectorize, num_total_passes, num_fused_passes, scheduling_policy, thread_coarsening_factor, relu):
    nest, (A, B, C) = create_gemm_nest_args(M, N, K, transA, transB, dtype)
    schedule = nest.create_schedule()

    i, j, k = schedule.get_indices()

    ii, jj, kk = schedule.tile({
        i: outer_tile_m,
        j: outer_tile_n,
        k: outer_tile_k
    })

    tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)

    iii, jjj = schedule.tile({
        ii: tensor_splits[0] * thread_coarsening_factor[0],
        jj: tensor_splits[1] * thread_coarsening_factor[1]
    })

    iiii, jjjj, kkk = schedule.tile({
        iii: tensor_splits[0],
        jjj: tensor_splits[1],
        kk: tensor_splits[2]
    })
    outer_nest_order = (i, j, k, ii, jj, iii, jjj, kk)
    block_indices = (i, j)
    warp_indices = (ii, jj)
    tensor_indices = (iiii, jjjj, kkk)

    elem_bytes = 4 if dtype == "s" else 2
    shared_mem_usage_bytes = elem_bytes * (outer_tile_m + outer_tile_n) * outer_tile_k
    use_dynamic_shared_mem = shared_mem_usage_bytes > target.max_static_shared_memory_per_block
    dynamic_shared_mem_usage_bytes = shared_mem_usage_bytes if use_dynamic_shared_mem else 0
    epilogue_op = _MMAFragmentOp.ReLU if relu else _MMAFragmentOp.NONE

    plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=block_indices, warp_indices=warp_indices, tensor_indices=tensor_indices, outer_nest_order=outer_nest_order, dynamic_shared_memory_size=dynamic_shared_mem_usage_bytes)
    plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes, use_static_offsets=use_static_offsets, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy, epilogue_op=epilogue_op)

    plan.cache(
        A,
        index=ii,
        double_buffer=double_buffering,
        double_buffer_location=Constants.AUTO,
        vectorize=vectorize,
        location=target.MemorySpace.SHARED,
        layout=cacheA_layout,
        strategy=cacheA_strategy,
        _shared_memory_offset=0 if use_dynamic_shared_mem else None
    )
    plan.cache(
        B,
        index=ii,
        double_buffer=double_buffering,
        double_buffer_location=Constants.AUTO,
        vectorize=vectorize,
        location=target.MemorySpace.SHARED,
        layout=cacheB_layout,
        strategy=cacheB_strategy,
        _shared_memory_offset=outer_tile_m * outer_tile_k if use_dynamic_shared_mem else None
    )

    if cache_C:
        out_cache_idx = k if thread_coarsening_factor == (1, 1) else kk
        plan.cache(
            C,
            index=out_cache_idx,
            vectorize=vectorize,
            location=target.MemorySpace.MMA_FRAGMENT
            # Don't specify layout so it defaults to the C array's layout
        )

    return plan, A, B, C

def get_dir(output_prefix):
    return os.path.split(output_prefix)[0] or '.'

def get_hat_path(output_prefix, package_name):
    return os.path.join(get_dir(output_prefix), package_name + ".hat")


def get_variants(opts: GemmOpts, dtype, target):
    ELEM_SIZE_BYTES = 4 if dtype == 's' else 2
    datatype = get_type(dtype)
    outer_tiles = [16, 32, 64, 128, 256]

    if target.runtime == Target.Runtime.ROCM:
        mfma_tiles = [_MMAShape.M64xN64xK1_B4, _MMAShape.M64xN64xK1_B2, _MMAShape.M32xN32xK2_B1, _MMAShape.M16xN16xK4_B1,
                      _MMAShape.M64xN64xK4_B4, _MMAShape.M64xN64xK4_B2, _MMAShape.M32xN32xK8_B1, _MMAShape.M16xN16xK16_B1]
    elif target.runtime == Target.Runtime.CUDA:
        mfma_tiles = [_MMAShape.M16xN16xK16_B1, _MMAShape.M32xN8xK16_B1, _MMAShape.M8xN32xK16_B1, _MMAShape.M16xN16xK8_B1]

    k_split = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    k_split_reduced = []
    for k in k_split:
        if opts.k % k == 0:
            k_split_reduced.append(k)
            if len(k_split_reduced) > 3:
                break # Since we start with the larger sizes, once we have enough we can skip the small k-splits

    num_total_passes = list(range(2, opts.k + 1, 1))
    num_total_passes_reduced = []
    for tp in num_total_passes:
        found = False
        for kk in k_split_reduced:
            if kk % tp == 0:
                for mma_tile in mfma_tiles:
                    if kk % (tp * get_k(target, mma_tile)) == 0:
                        num_total_passes_reduced.append(tp)
                        found = True
                        break
                if found:
                    break
    num_total_passes_reduced.append(1)
    fuse_factor = [1]#, 2, 4, 8]
    scheduling_policy = [_MMASchedulingPolicy.PASS_ORDER, _MMASchedulingPolicy.BLOCK_ORDER]
    thread_coarsening_factor_r = [1, 2, 4]
    thread_coarsening_factor_c = [1, 2, 4]
    use_static_offsets = [False]
    vectorize = [True]
    double_buffering = [True]
    cacheA_layout = [Array.Layout.LAST_MAJOR]
    cacheB_layout = [Array.Layout.FIRST_MAJOR]
    cacheA_strategy = [_CacheStrategy.BLOCKED, _CacheStrategy.STRIPED]
    cacheB_strategy = [_CacheStrategy.BLOCKED, _CacheStrategy.STRIPED]

    def valid_variant(outer_t, outer_k, mma_type, total_passes, fuse_factor, scheduling_policy, thread_coarsening_factor_r, thread_coarsening_factor_c):
        # For single block MMA shapes, we don't need to test both scheduling policies and they will
        # both effectively result in the same schedule. Just ignore BLOCK_ORDER in that case.
        if single_block_mma(mma_type) and scheduling_policy == _MMASchedulingPolicy.BLOCK_ORDER:
            return False

        thread_coarsening_factor = thread_coarsening_factor_r * thread_coarsening_factor_c
        if thread_coarsening_factor > 4:
            return False

        thread_blocks_x = math.ceil(opts.n / outer_t[1])
        thread_blocks_y = math.ceil(opts.m / outer_t[0])
        num_waves = math.ceil(thread_blocks_x * thread_blocks_y / target.num_cores) # Assuming a minimum occupancy of 1 thread block per SM
        # No point having a thread-coarsening factor which is greater than the number of waves required with single occupancy.
        if thread_coarsening_factor > num_waves:
            return False

        mma_m = get_m(target, mma_type)
        mma_n = get_n(target, mma_type)
        cache_size_a = outer_t[0] * outer_k
        cache_size_b = outer_t[1] * outer_k
        pass_width = total_passes * get_k(target, mma_type)
        warp_tile_rows = mma_m * thread_coarsening_factor_r
        warp_tile_cols = mma_n * thread_coarsening_factor_c

        return total_passes % fuse_factor == 0 and (opts.m % outer_t[0] == 0) and (opts.n % outer_t[1] == 0) and (opts.k % outer_k == 0) and \
                (outer_t[0] % warp_tile_rows == 0) and (outer_t[1] % warp_tile_cols == 0) and outer_k % pass_width == 0 and \
                ((cache_size_a + cache_size_b) * ELEM_SIZE_BYTES <= target.max_shared_memory_per_block) and \
                target.tensor_core_info.supports(datatype, datatype, mma_type, total_passes, total_passes // fuse_factor)

    return list(filter(lambda v: valid_variant(v[0], v[1], v[2], v[10], v[11], v[12], v[13], v[14]), product(combinations_with_replacement(outer_tiles, 2), k_split_reduced,
                                                                                    mfma_tiles, use_static_offsets, double_buffering, vectorize, cacheA_layout, cacheB_layout,
                                                                                    cacheA_strategy, cacheB_strategy, num_total_passes_reduced, fuse_factor, scheduling_policy,
                                                                                    thread_coarsening_factor_r, thread_coarsening_factor_c)))


def benchmark_gemm(opts: GemmOpts, dtype, batch_size: int, output_prefix: str, category: str, available_gpus, container_name, verbose_logs, compiler_ver, commit_id, commit_datetime: str, commit_branch, target_name, check_result, relu, dev_props):
    """
    Architecture Overview:
    --------------------------------------------------------------------------------------------
    This function takes a particular gemm input and produces different kernel implementations
    by permuting the possible different optimizations and the configurations e.g. vectorization,
    cache layouts etc. In order to do this, it creates a set of variations (aka "variants")
    which are then benchmarked on GPU devices.

    To do this efficiently, we decompose the task into a producer-consumer model where the
    producers ("run_variant") are responsible for building the HAT package (CPU-bound) for a
    particular kernel configuration and the consumers ("gemm_runner") are responsible for taking
    a HAT package and benchmarking them (GPU-bound). We spawn 1 producer process for each kernel
    variant (in waves of X processes where X is a constant based on the number of CPU cores on the
    system) and 1 consumer process for each GPU device available for kernel execution.
    ----------------------------------------------------------------------------------------------
    Here is the flow diagram for the benchmarking pipeline:

                            GEMM input
                                v
                        +-------------------+     +-------------------->--------------------+
                        |      Create       |     |     +------------------->------------+  |
                        |   shared queues  -+-----+-----+-----+---   --+                 |  |
                        |    (1 per GPU)    |     Q1    Q2    Q3       Qn ------->---+   |  |
                        |        |          |     ^     ^     ^        ^             |   |  |
                        |      Create       |     |     |     |        |             |   |  |
                        |    golden data    |      \    |     |       /              |   |  |
              Main      |        |          |     +-----------------------+          |   |  |
             Process    |     Generate      |     |  GPU Device selector  |          |   |  |
                        |     variants      |     +-----------------------+          |   |  |
                        |        |          |     /   |     |    \   \     \         |   |  v
                        |  Spawn producers  |    /    |     |     |   \     \        |   v  |
                        |  (1 per variant) -+---+-----+-----+-----+----+--- --+      |   |  |
                        |        |          |   P1    P2    P3    P4   P5     Pv     v   |  |
                        |  Spawn consumers  |                                        |   |  |
                        |    (1 per GPU)   -+-----+-----+-----+---   ---+            |   |  |
                        |        |          |    C1    C2    C3        Cn <----------+   |  |
                        |   Wait for all    |     ^     ^                                |  |
                        | processes to join |     |     +----------<---------------------+  |
                        +-------------------+     ----------------------<-------------------+
    ----------------------------------------------------------------------------------------------
    Producer process: "run_variant"

             i, kernel variant, golden data, sync. list of results
                                 v
                        +-------------------+
                        |      Create       |
                        |   result object   |
                        |         |         |
                        |     Build HAT     |
                        |      package      |
                        |         |         |
                        |  Push HAT pkg to  +-------------> Qi
                        |  shared queue Qi  |
                        +-------------------+
    ----------------------------------------------------------------------------------------------
    Consumer process (Ci) for shared queue Qi: "gemm_runner"

                         i, sync. list of results
                                 v
                        +-------------------+
                        | Until Qi is empty <-------------- Qi
                        |       do:         |
                        |         |         |
                        |    Pop HAT pkg.   |
                        |   from queue Di   |
                        |         |         |
                        |      Verify       |
                        |    correctness    |
                        |  vs. golden data  |
                        |         |         |
                        |   Run benchmark   |
                        |      (hatlib)     |
                        |         |         |
                        |   Upload results  |
                        |    to Cosmos DB   |
                        +-------------------+
    """
    # create shared process queue, one per gpu
    gpu_devices = []
    device_q = []
    for i in range(len(available_gpus)):
        if available_gpus[i]:
            gpu_devices.append(i)
            device_q.append(multiprocessing.Queue())

    total_gpus = len(gpu_devices)
    target = Target(target_name)
    variants = get_variants(opts, dtype, target)

    if len(variants) == 0: # this means we failed to find any valid kernel configuration for this input
        print_log(verbose_logs, 'No valid kernel configurations found.', "magenta")
        if container_name:
            result = BenchmarkResult(opts=opts, dtype=dtype, gpu_id=-1, commit_id=commit_id, commit_datetime=commit_datetime, commit_branch=commit_branch, target_name=target_name, deviceProperties='')
            result.target_rt = 'ROCM' if target.runtime == Target.Runtime.ROCM else 'CUDA'
            result.compiler_version = compiler_ver
            cosmosdb.upsert_benchmark_results([result.get_result_row()], container_name, verbose_logs)
    else:
        # Create golden data for verification if required
        golden_data = None
        if check_result:
            # Create the arrays with the appropriate layout
            datatype = get_type(dtype)
            npdatatype = np.dtype(datatype.name)
            A_test, B_test, C_test = (np.ndarray((opts.m, opts.k), dtype=npdatatype, order=get_layout(bool(opts.transA)).to_numpy_order()),
                                      np.ndarray((opts.k, opts.n), dtype=npdatatype, order=get_layout(bool(opts.transB)).to_numpy_order()),
                                      np.ndarray((opts.m, opts.n), dtype=npdatatype, order=Array.Layout.FIRST_MAJOR.to_numpy_order()))

            # Create all the random input data
            A_test_data, B_test_data, C_test_data = (np.random.random((opts.m, opts.k)).astype(npdatatype),
                                                     np.random.random((opts.k, opts.n)).astype(npdatatype),
                                                     np.random.random((opts.m, opts.n)).astype(npdatatype))

            # Assign the default-ordered input data to the appropriately-ordered arrays
            A_test[:] = A_test_data
            B_test[:] = B_test_data
            C_test[:] = C_test_data

            C_ref = (opts.beta * C_test) + (opts.alpha * (A_test @ B_test))
            golden_data = (A_test, B_test, C_test, C_ref)

        waveSize = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        for wave in range(0, len(variants), waveSize):
            processes = []
            result_rows = manager.list()
            print(f"Wave {wave // waveSize} (Completed: {wave}/{len(variants)} kernels)")
            for i in range(wave, min(wave + waveSize, len(variants))):
                gpu_idx = i % total_gpus
                gpu_id = gpu_devices[gpu_idx]
                p = multiprocessing.Process(name=f"builder{i}", target=run_variant, args=(variants[i], gpu_id, device_q[gpu_idx], opts, dtype, target, output_prefix, category, compiler_ver, commit_id, commit_datetime, commit_branch, target_name, dev_props[gpu_id], verbose_logs, check_result, relu))
                p.start()

            time.sleep(5)

            for i in range(total_gpus):
                p = multiprocessing.Process(name=f"runner{i}", target=gemm_runner, args=(gpu_devices[i], batch_size, output_prefix, device_q[i], result_rows, golden_data, verbose_logs))
                p.start()
                processes.append(p)

            if not verbose_logs:
                bar = progressbar.ProgressBar(maxval=len(processes), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()

            i = 0
            for p in processes:
                while p.is_alive():
                    print_log(verbose_logs, f"Joining process: {p.name}, {p.pid}")

                    proc = psutil.Process(p.pid)
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        print_log(verbose_logs, f"Zombie process found: {p.name}, {p.pid}, skipping...")

                        break # just move on

                    p.join(5)
                else:
                    i += 1
                    if not verbose_logs:
                        bar.update(i)
            else:
                if not verbose_logs:
                    bar.finish()

            if container_name:
                cosmosdb.upsert_benchmark_results(result_rows, container_name, verbose_logs)


def run_variant(variant, gpu_id, device_q, opts, dtype, target, output_prefix, category, compiler_ver, commit_id, commit_datetime, commit_branch, target_name, dev_props, verbose_logs, check_result, relu):
    try:
        assert float(opts.alpha) == 1., "alpha must be 1"

        (outer_tile_m, outer_tile_n), k_split, mma_shape, use_static_offsets, double_buffering, vectorize, cacheA_layout, cacheB_layout, cacheA_strategy, cacheB_strategy, num_total_passes, fuse_factor, scheduling_policy, thread_coarsening_factor_r, thread_coarsening_factor_c = variant
        result = BenchmarkResult(opts=opts, dtype=dtype, gpu_id=gpu_id, commit_id=commit_id, commit_datetime=commit_datetime, commit_branch=commit_branch, target_name=target_name, deviceProperties=dev_props)
        result.mma_shape = mma_shape.name
        result.use_static_offsets = use_static_offsets
        result.double_buffering = double_buffering
        result.vectorize = vectorize
        result.cache_layout_A = "F" if cacheA_layout == Array.Layout.FIRST_MAJOR else "L"
        result.cache_layout_B = "F" if cacheB_layout == Array.Layout.FIRST_MAJOR else "L"
        result.cache_strategy_A = "B" if cacheA_strategy == _CacheStrategy.BLOCKED else "S"
        result.cache_strategy_B = "B" if cacheB_strategy == _CacheStrategy.BLOCKED else "S"
        result.num_total_passes = num_total_passes
        result.num_fused_passes = num_fused_passes = num_total_passes // fuse_factor
        result.scheduling_policy = 0 if scheduling_policy == _MMASchedulingPolicy.PASS_ORDER else 1
        result.block_tile = (outer_tile_m, outer_tile_n)
        result.thread_coarsening_factor = (thread_coarsening_factor_r, thread_coarsening_factor_c)
        result.k_split = k_split
        result.compiler_version = compiler_ver
        result.category = category
        result.check = check_result
        result.cache_C = True
        if target.runtime == Target.Runtime.ROCM:
            result.target_rt = 'ROCM'
        elif target.runtime == Target.Runtime.CUDA:
            result.target_rt = 'CUDA'

        plan, A, B, C = benchmark_kernel(target, opts.m, opts.n, opts.k, opts.transA, opts.transB, dtype, outer_tile_m, outer_tile_n, k_split, cacheA_layout, cacheB_layout, result.cache_C,
                                        cacheA_strategy, cacheB_strategy, mma_shape, use_static_offsets, double_buffering, vectorize, num_total_passes, num_fused_passes, scheduling_policy, result.thread_coarsening_factor, relu)
        fn_name = f"{result.target_rt}_GEMM_{result.get_partition_key().replace('.', '_')}_{result.get_id().replace('.', '_')}"
        block_dims, grid_dims = plan._calc_block_grid_dim()
        is_valid_plan = plan._is_valid_block_dim(block_dims) and plan._is_valid_block_size(block_dims)
        block_size = reduce(lambda a, b: a * b, block_dims)
        tile_size_a = outer_tile_m * k_split
        tile_size_b = k_split * outer_tile_n
        wpt_a = tile_size_a // block_size
        wpt_b = tile_size_b // block_size
        submit_package = is_valid_plan and wpt_a >= 1 and wpt_b >= 1 and tile_size_a % block_size == 0 and tile_size_b % block_size == 0
        if submit_package:
            package_name = f"gemm_benchmarks_{fn_name}"
            package = Package()
            function = package.add(plan, args=(A, B, C), base_name=fn_name)
            package.build(
                name=package_name,
                output_dir=get_dir(output_prefix),
                fail_on_error=True,
                _quiet=True
            )

            result.compilable = True
        else:
            print_log(verbose_logs, f"Invalid kernel launch configuration for {fn_name}", 'grey')

    except Exception as e:
        error = f"[Fail] Error while building {fn_name} on gpu {gpu_id}: {e}"
        result.prog_out += error
        print_log(verbose_logs, error, 'red')

    finally:
        if submit_package:
            device_q.put((package_name, opts, fn_name, result))
            print_log(verbose_logs, f"Submitted package: {package_name} on gpu {gpu_id}.", 'yellow')

def gemm_runner(gpu_id: int, batch_size: int, output_prefix, device_q, result_rows, golden_data, verbose_logs):
    while True:
        # Attempt few retries before giving up
        retry = 5
        while device_q.empty():
            print_log(verbose_logs, f'Device queue for gpu {gpu_id} polling for 5 seconds...')

            time.sleep(5)
            retry = retry - 1
            if retry <= 0:
                break

        if retry <= 0:
            print_log(verbose_logs, f'Device queue for gpu {gpu_id} exiting.')
            break

        (package_name, opts, fn_name, result) = device_q.get()
        if result.compilable:
            print_log(verbose_logs, f"Running package: {package_name} on gpu {gpu_id}. Pending packages: {device_q.qsize()}", 'cyan')

            kernel_path = os.path.join(get_dir(output_prefix), package_name + ".cu")
            with open(kernel_path, 'r') as kernel_file:
                result.kernelCode = kernel_file.read()
            try:
                hat_path = get_hat_path(output_prefix, package_name)
                if result.check:
                    (A_test, B_test, C_test, C_ref) = golden_data

                    hat_package, func_map = hatlib.load(hat_path)
                    C_copy = deepcopy(C_test)
                    func_map[fn_name](A_test, B_test, C_copy, gpu_id=gpu_id)
                    np.testing.assert_allclose(C_ref, C_copy, rtol=1e-2)

                    print_log(verbose_logs, f'Passed correctness for package: {package_name} on gpu {gpu_id}', 'green')

                result.correct = True

                results = hatlib.run_benchmark(
                    hat_path,
                    batch_size=batch_size,
                    min_time_in_sec=1,
                    gpu_id=gpu_id
                )
            except Exception as e:
                if result.target_rt == 'ROCM':
                    from hatlib.pyhip.hip import hipErrorInvalidConfiguration
                    if isinstance(e, hipErrorInvalidConfiguration):
                        error = f"[Fail] Invalid kernel configuration for function {fn_name} on gpu {gpu_id}: {e}"
                        print_log(verbose_logs, error, 'red')
                    else:
                        error = f"[Fail] Error while running function {fn_name} on gpu {gpu_id}: {e}"
                        print(colored(error, 'red'))
                else:
                    error = f"[Fail] Error while running function {fn_name} on gpu {gpu_id}: {e}"
                    print(colored(error, 'red'))

                result.prog_out += error

            else: # if there is no exception
                result.executable = len(results) > 0
                result.prog_out += str([asdict(r) for r in results])
                if len(results) > 0:
                    print_log(verbose_logs, f"[Pass] {fn_name}", 'green')

                    time_s = results[0].min_of_means
                    if time_s != '-':
                        flops = 2 * opts.k * opts.m * opts.n / float(time_s)
                        tflops = flops * 1e-12

                        result.time_ms = float(time_s) * 1e3
                        result.TFlops = tflops
                        print_log(verbose_logs, f"Throughput of {fn_name}: {tflops} TFlops on gpu {gpu_id}", 'blue')

        result_rows.append(result.get_result_row())