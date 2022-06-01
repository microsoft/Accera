####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import multiprocessing
import os
import psutil
from copy import deepcopy
import progressbar
from multiprocessing import Process, Manager, Queue
from re import M
from termcolor import colored
from dataclasses import dataclass, InitVar
from datetime import datetime
import time
from itertools import combinations_with_replacement, product
from typing import Any, Dict, List, Tuple
import numpy as np
import hatlib
from accera import Array, Nest, Constants, ScalarType, Target, Package, Schedule, Plan
from accera._lang_python._lang import _MMASchedulingPolicy, _MMAShape
import cosmosdb
from gemm_opts import GemmOpts

@dataclass
class BenchmarkResult:
    opts: InitVar[GemmOpts]
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
    cache_C: bool=False
    block_tile: Tuple[int, int]=(0, 0)
    k_split: int=-1
    double_buffering: bool=False
    vectorize: bool=False
    num_total_passes: int=-1
    num_fused_passes: int=-1
    scheduling_policy: int=-1
    target_rt: str=''
    time_ms: float= -1.0
    TFlops: float= -1.0
    dt: str= datetime.now().ctime()
    compiler_version: str=''
    compilable: bool=False
    executable: bool=False
    check: bool=False
    correct: bool=False
    kernelCode: str=''
    prog_out: str=''

    def __post_init__(self, opts):
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
        self.in_type=opts.type
        self.out_type=opts.type

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
            'cc' : self.cache_C,
            'bt0' : self.block_tile[0],
            'bt1' : self.block_tile[1],
            'ks' : self.k_split,
            'db' : self.double_buffering,
            'v' : self.vectorize,
            'tp' : self.num_total_passes,
            'fp' : self.num_fused_passes,
            'sp' : self.scheduling_policy,
            'git' : self.commit_id
        }
        return "_".join([f"{key}{str(key_info[key])}" for key in key_info])

    def getResultRow(self) ->Dict[str, Any]:
        d = self.__dict__
        d['id'] = self.get_id()
        d['partitionKey'] = self.get_partition_key()
        return d

def get_k(mfma_tile: _MMAShape):
    if mfma_tile == _MMAShape.M64xN64xK4_B4 or mfma_tile == _MMAShape.M64xN64xK1_B4 or mfma_tile == _MMAShape.M64xN64xK4_B2 or mfma_tile == _MMAShape.M64xN64xK1_B2:
        if mfma_tile == _MMAShape.M64xN64xK4_B4 or mfma_tile == _MMAShape.M64xN64xK4_B2:
            return 4
        else:
            return 1
    elif mfma_tile == _MMAShape.M32xN32xK8_B1 or mfma_tile == _MMAShape.M32xN32xK2_B1:
        if mfma_tile == _MMAShape.M32xN32xK8_B1:
            return 8
        else:
            return 2
    elif mfma_tile == _MMAShape.M16xN16xK16_B1 or mfma_tile == _MMAShape.M16xN16xK4_B1:
        if mfma_tile == _MMAShape.M16xN16xK16_B1:
            return 16
        else:
            return 4

    return 0

def get_leading_dim(mfma_t: _MMAShape):
    if mfma_t == _MMAShape.M16xN16xK4_B1 or mfma_t == _MMAShape.M16xN16xK16_B1:
        return 16
    elif mfma_t == _MMAShape.M32xN32xK2_B1 or mfma_t == _MMAShape.M32xN32xK8_B1:
        return 32
    else:
        return 64

def getLayout(transpose: bool):
    return Array.Layout.LAST_MAJOR if transpose else Array.Layout.FIRST_MAJOR

def getType(typeStr: str):
    return ScalarType.float32 if typeStr == 's' else ScalarType.float16

def get_mfma_split(runtime, mfma_tile):
    if runtime == Target.Runtime.CUDA:
        return (4, 2)

    if mfma_tile == _MMAShape.M64xN64xK4_B4 or mfma_tile == _MMAShape.M64xN64xK1_B4:
        return (4, 16)
    if mfma_tile == _MMAShape.M64xN64xK4_B2 or mfma_tile == _MMAShape.M64xN64xK1_B2:
        return (2, 32)
    if mfma_tile == _MMAShape.M32xN32xK8_B1 or mfma_tile == _MMAShape.M32xN32xK2_B1:
        return (4, 4)
    return (2, 2)


def _benchmark_kernel(
    target: Target, schedule: Schedule, args: Tuple[Array, Array, Array], outer_tile_x: int, outer_tile_y: int,
    outer_tile_k: int, cacheALayout, cacheBLayout, mfma_tile, use_static_offsets, double_buffering, vectorize, num_total_passes, num_fused_passes, scheduling_policy) -> Plan:
    A, B, C = args
    i, j, k = schedule.get_indices()

    ii, jj, kk = schedule.tile({
        i: outer_tile_x,
        j: outer_tile_y,
        k: outer_tile_k
    })

    (ot_x, ot_y) = get_mfma_split(target.runtime, mfma_tile)

    iii, jjj, kkk = schedule.tile({
        ii: ot_x,
        jj: ot_y,
        kk: num_total_passes * get_k(mfma_tile)
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
    plan.tensorize(indices=(iii, jjj, kkk), mma_shape=mfma_tile, use_static_offsets=use_static_offsets, num_total_passes=num_total_passes, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy)

    if target.runtime == Target.Runtime.ROCM:
        plan.cache(
            A,
            index=ii,
            double_buffer=double_buffering,
            double_buffer_location=Constants.AUTO,
            vectorize=vectorize,
            location=target.MemorySpace.SHARED,
            layout=cacheALayout
        )
        plan.cache(
            B,
            index=ii,
            double_buffer=double_buffering,
            double_buffer_location=Constants.AUTO,
            vectorize=vectorize,
            location=target.MemorySpace.SHARED,
            layout=cacheBLayout
        )

    return plan

def get_dir(output_prefix):
    return os.path.split(output_prefix)[0] or '.'

def get_hat_path(output_prefix, package_name):
    return os.path.join(get_dir(output_prefix), package_name + ".hat")

def create_gemm_nest_args(opts: GemmOpts):
    M = int(opts.m)
    N = int(opts.n)
    K = int(opts.k)
    datatype = getType(opts.type)

    A = Array(
        role=Array.Role.INPUT,
        element_type=datatype,
        shape=(M, K),
        layout=getLayout(bool(opts.transA))
    )
    B = Array(
        role=Array.Role.INPUT,
        element_type=datatype,
        shape=(K, N),
        layout=getLayout(bool(opts.transB))
    )
    C = Array(role=Array.Role.INPUT_OUTPUT, element_type=datatype, shape=(M, N))

    nest = Nest([M, N, K])
    i, j, k = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    return nest, (A, B, C)


def get_variants(opts: GemmOpts, target):
    ELEM_SIZE_BYTES = 4 if opts.type == 's' else 2
    datatype = getType(opts.type)
    outer_tiles = [16, 32, 64, 128, 256]
    mfma_tiles_all = [_MMAShape.M64xN64xK1_B4, _MMAShape.M64xN64xK1_B2, _MMAShape.M32xN32xK2_B1, _MMAShape.M16xN16xK4_B1,
                        _MMAShape.M64xN64xK4_B4, _MMAShape.M64xN64xK4_B2, _MMAShape.M32xN32xK8_B1, _MMAShape.M16xN16xK16_B1]

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
                for mma_tile in mfma_tiles_all:
                    if kk % (tp * get_k(mma_tile)) == 0:
                        num_total_passes_reduced.append(tp)
                        found = True
                        break
                if found:
                    break
    num_total_passes_reduced.append(1)
    fuse_factor = [1]#, 2, 4, 8]
    scheduling_policy = [_MMASchedulingPolicy.PASS_ORDER, _MMASchedulingPolicy.BLOCK_ORDER]
    use_static_offsets = [False]
    vectorize = [True]
    double_buffering = [True]
    cacheALayout = [Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR]
    cacheBLayout = [Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR]

    def valid_variant(outer_t, outer_k, mfma_t, total_passes, fuse_factor):
        leadingDim = get_leading_dim(mfma_t)
        cache_size_a = outer_t[0] * outer_k
        cache_size_b = outer_t[1] * outer_k
        passWidth = total_passes * get_k(mfma_t)

        return total_passes % fuse_factor == 0 and (opts.m % leadingDim == 0) and (opts.n % leadingDim == 0) and \
                (outer_t[0] % leadingDim == 0) and (outer_t[1] % leadingDim == 0) and outer_k % passWidth == 0 and \
                ((cache_size_a + cache_size_b) * ELEM_SIZE_BYTES <= target.max_shared_memory_per_block) and \
                target.tensor_core.supports(datatype, datatype, mfma_t, total_passes, total_passes // fuse_factor)

    if target.runtime == Target.Runtime.ROCM:
        mfma_tiles = [_MMAShape.M64xN64xK1_B4, _MMAShape.M64xN64xK1_B2, _MMAShape.M32xN32xK2_B1, _MMAShape.M16xN16xK4_B1,
                      _MMAShape.M64xN64xK4_B4, _MMAShape.M64xN64xK4_B2, _MMAShape.M32xN32xK8_B1, _MMAShape.M16xN16xK16_B1]
    elif target.runtime == Target.Runtime.CUDA:
        mfma_tiles = [_MMAShape.M16xN16xK16_B1]

    return list(filter(lambda v: valid_variant(v[0], v[1], v[2], v[8], v[9]), product(combinations_with_replacement(outer_tiles, 2), k_split_reduced, mfma_tiles,
                                                                                    use_static_offsets, double_buffering, vectorize, cacheALayout,
                                                                                    cacheBLayout, num_total_passes_reduced, fuse_factor, scheduling_policy)))


def benchmark_gemm(opts: GemmOpts, output_prefix: str, available_gpus, containerName, verbose_logs, compilerVersion, commit_id, commit_datetime, commit_branch, target_name, check_result, dev_props):
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
            device_q.append(Queue())

    total_gpus = len(gpu_devices)
    target = Target(target_name)
    variants = get_variants(opts, target)

    if len(variants) == 0: # this means we failed to find any valid kernel configuration for this input
        if verbose_logs:
            print(colored('No valid kernel configurations found.', "magenta"))
        if containerName:
            result = BenchmarkResult(opts=opts, gpu_id=-1, commit_id=commit_id, commit_datetime=str(commit_datetime), commit_branch=commit_branch, target_name=target_name, deviceProperties='')
            result.target_rt = 'ROCM' if target.runtime == Target.Runtime.ROCM else 'CUDA'
            result.compiler_version = compilerVersion
            cosmosdb.upsert_benchmark_results([result.getResultRow()], containerName, verbose_logs)
    else:
        # Create golden data for verification if required
        golden_data = None
        if check_result:
            # Create the arrays with the appropriate layout
            datatype = getType(opts.type)
            npdatatype = np.dtype(datatype.name)
            A_test, B_test, C_test = (np.ndarray((opts.m, opts.k), dtype=npdatatype, order=getLayout(bool(opts.transA)).to_numpy_order()),
                                      np.ndarray((opts.k, opts.n), dtype=npdatatype, order=getLayout(bool(opts.transB)).to_numpy_order()),
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
                p = Process(name=f"builder{i}", target=run_variant, args=(variants[i], gpu_id, device_q[gpu_idx], opts, target, output_prefix, compilerVersion, commit_id, commit_datetime, commit_branch, target_name, dev_props[gpu_id], verbose_logs, check_result))
                p.start()

            time.sleep(5)

            for i in range(total_gpus):
                p = Process(name=f"runner{i}", target=gemm_runner, args=(gpu_devices[i], output_prefix, device_q[i], result_rows, golden_data, verbose_logs))
                p.start()
                processes.append(p)

            if not verbose_logs:
                bar = progressbar.ProgressBar(maxval=len(processes), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()

            i = 0
            for p in processes:
                while p.is_alive():
                    if verbose_logs:
                        print(f"Joining process: {p.name}, {p.pid}")

                    proc = psutil.Process(p.pid)
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        if verbose_logs:
                            print(f"Zombie process found: {p.name}, {p.pid}, skipping...")

                        break # just move on

                    p.join(5)
                else:
                    i += 1
                    if not verbose_logs:
                        bar.update(i)
            else:
                if not verbose_logs:
                    bar.finish()

            if containerName:
                cosmosdb.upsert_benchmark_results(result_rows, containerName, verbose_logs)

def run_variant(variant, gpu_id, device_q, opts, target, output_prefix, compilerVersion, commit_id, commit_datetime, commit_branch, target_name, dev_props, verbose_logs, check_result):
    nest, (A, B, C) = create_gemm_nest_args(opts)

    try:
        assert float(opts.alpha) == 1., "alpha must be 1"

        (outer_tile_x, outer_tile_y), k_split, mfma_tile, use_static_offsets, double_buffering, vectorize, cacheALayout, cacheBLayout, num_total_passes, fuse_factor, scheduling_policy = variant
        result = BenchmarkResult(opts=opts, gpu_id=gpu_id, commit_id=commit_id, commit_datetime=str(commit_datetime), commit_branch=commit_branch, target_name=target_name, deviceProperties=dev_props)
        result.mma_shape = mfma_tile.name
        result.use_static_offsets = use_static_offsets
        result.double_buffering = double_buffering
        result.vectorize = vectorize
        result.cache_layout_A = "F" if cacheALayout == Array.Layout.FIRST_MAJOR else "L"
        result.cache_layout_B = "F" if cacheBLayout == Array.Layout.FIRST_MAJOR else "L"
        result.num_total_passes = num_total_passes
        result.num_fused_passes = num_fused_passes = num_total_passes // fuse_factor
        result.scheduling_policy = 0 if scheduling_policy == _MMASchedulingPolicy.PASS_ORDER else 1
        result.block_tile = (outer_tile_x, outer_tile_y)
        result.k_split = k_split
        result.compiler_version = compilerVersion
        result.check = check_result
        if target.runtime == Target.Runtime.ROCM:
            result.target_rt = 'ROCM'
        elif target.runtime == Target.Runtime.CUDA:
            result.target_rt = 'CUDA'

        plan = _benchmark_kernel(target, nest.create_schedule(), (A, B, C), outer_tile_x, outer_tile_y, k_split,
                                    cacheALayout, cacheBLayout, mfma_tile, use_static_offsets, double_buffering, vectorize, num_total_passes, num_fused_passes, scheduling_policy)
        fn_name = f"{result.target_rt}_GEMM_{result.get_partition_key().replace('.', '_')}_{result.get_id().replace('.', '_')}"
        package_name = f"gemm_benchmarks_{fn_name}"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=fn_name)
        package.build(
            package_name,
            output_dir=get_dir(output_prefix),
            fail_on_error=True,
            _quiet=True
        )

        result.compilable = True

    except Exception as e:
        error = f"[Fail] Error while building {fn_name} on gpu {gpu_id}: {e}"
        print(colored(error, 'red'))
        result.prog_out += error

    device_q.put((package_name, opts, fn_name, result))
    if verbose_logs:
        print(colored(f"Submitted package: {package_name} on gpu {gpu_id}.", 'yellow'))

def gemm_runner(gpu_id, output_prefix, device_q, result_rows, golden_data, verbose_logs):
    while True:
        # Attempt few retries before giving up
        retry = 5
        while device_q.empty():
            if verbose_logs:
                print(f'Device queue for gpu {gpu_id} polling for 5 seconds...')

            time.sleep(5)
            retry = retry - 1
            if retry <= 0:
                break

        if retry <= 0:
            if verbose_logs:
                print(f'Device queue for gpu {gpu_id} exiting.')
            break

        (package_name, opts, fn_name, result) = device_q.get()
        if result.compilable:
            if verbose_logs:
                print(colored(f"Running package: {package_name} on gpu {gpu_id}. Pending packages: {device_q.qsize()}", 'cyan'))

            with open(os.path.join(get_dir(output_prefix), package_name + ".cu"), 'r') as kernel_file:
                result.kernelCode = kernel_file.read()
            try:
                if result.check:
                    (A_test, B_test, C_test, C_ref) = golden_data

                    hat_package, func_map = hatlib.load(get_hat_path(output_prefix, package_name))
                    C_copy = deepcopy(C_test)
                    func_map[fn_name](A_test, B_test, C_copy, gpu_id=gpu_id)
                    np.testing.assert_allclose(C_ref, C_copy, rtol=1e-4 if opts.type == 's' else 1e-2)

                    if verbose_logs:
                        print(colored(f'Passed correctness for package: {package_name} on gpu {gpu_id}', 'green'))

                result.correct = True

                results = hatlib.run_benchmark(
                    get_hat_path(output_prefix, package_name),
                    batch_size=2,
                    min_time_in_sec=1,
                    gpu_id=gpu_id
                )
            except hatlib.pyhip.hip.hipErrorInvalidConfiguration as e:
                error = f"[Fail] Invalid kernel configuration for function {fn_name} on gpu {gpu_id}: {e}"
                result.prog_out += error
                if verbose_logs:
                    print(colored(error, 'red'))
            except Exception as e:
                error = f"[Fail] Error while running function {fn_name} on gpu {gpu_id}: {e}"
                result.prog_out += error
                print(colored(error, 'red'))
            else:
                result.executable = len(results) > 0
                result.prog_out += str(results)
                if len(results) > 0:
                    if verbose_logs:
                        print(colored(f"[Pass] {fn_name}", 'green'))

                    time_s = results[0].get('min_of_means', '-')
                    if time_s != '-':
                        flops = 2 * opts.k * opts.m * opts.n / float(time_s)
                        tflops = flops * 1e-12

                        result.time_ms = float(time_s) * 1e3
                        result.TFlops = tflops
                        if verbose_logs:
                            print(colored(f"Throughput of {fn_name}: {tflops} TFlops on gpu {gpu_id}", 'blue'))

        result_rows.append(result.getResultRow())