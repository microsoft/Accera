#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import inspect
import logging
import os
import pathlib
import shutil
import sys
import unittest
from enum import Enum
from typing import Callable, List

import numpy as np

try:
    import cuda
except:
    CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = True

if sys.platform == 'linux':
    try:
        import ctypes
        ROCM_AVAILABLE = bool(ctypes.cdll.LoadLibrary('libamdhip64.so'))
    except:
        ROCM_AVAILABLE = False
else:
    ROCM_AVAILABLE = False

print(f"ROCM_AVAILABLE: {ROCM_AVAILABLE}")
print(f"CUDA_AVAILABLE: {CUDA_AVAILABLE}")

DEV_MODE = False
if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    DEV_MODE = True
    sys.path.insert(1, os.getcwd())

from accera._lang_python import Role
from accera._lang_python._lang import _MMAShape, _MMASchedulingPolicy, _MemorySpace, _CacheStrategy, _MMAFragmentOp
from accera.test import verifiers
from accera.test.test_utils import expectedFailure, FailedReason, get_type_str
from accera import Array, Nest, Package, ScalarType, Target, Constants
from accera.Targets import GridUnits

TEST_PACKAGE_DIR = "test_mfma"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def _matmul_nest(M, N, K, A, B, C):
    nest = Nest(shape=(M, N, K))
    i, j, k = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    return nest

def _relu_nest(M, N, C):
    import accera as acc

    nest = Nest(shape=(M, N))
    i, j = nest.get_indices()

    @nest.iteration_logic
    def _():
        C[i, j] = acc.max(0.0, C[i, j])

    return nest

def _matmul_tensor_schedule(M, N, K, A, B, C, block_tile, outer_tile_k, thread_coarsening_tile, tensor_splits):
    matmul_nest = _matmul_nest(M, N, K, A, B, C)
    matmul_sched = matmul_nest.create_schedule()
    i, j, k = matmul_sched.get_indices()
    ii, jj, kk = matmul_sched.tile({
        i: block_tile[0],
        j: block_tile[1],
        k: outer_tile_k
    })
    iii, jjj = matmul_sched.tile({
        ii: tensor_splits[0] * thread_coarsening_tile[0],
        jj: tensor_splits[1] * thread_coarsening_tile[1]
    })
    iiii, jjjj, kkk = matmul_sched.tile({
        iii: tensor_splits[0],
        jjj: tensor_splits[1],
        kk: tensor_splits[2]
    })
    matmul_sched.reorder(i, j, ii, jj, iii, jjj, k, kk, iiii, jjjj, kkk)

    return matmul_sched


class TensorizeTest(unittest.TestCase):
    PACKAGE_MODE = Package.Mode.RELEASE

    def _verify_matmul(self, function, A, B, C, verifier):
        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        C_ref = C_test + A_test @ B_test

        verifier.check_correctness(function.name, before=(A_test, B_test, C_test), after=(A_test, B_test, C_ref))


    def _check_cu_has_mfma(self, test_name, verifier):
        checker = verifier.file_checker(f"{test_name}.cu")
        checker.check_label(
            'extern "C" __global__  __launch_bounds__({{.+}}) void ' + test_name + '_{{.+}}__gpu__('
        )
        checker.check('mma_sync')
        checker.run()


    def _check_cu_has_no_mfma(self, test_name, verifier):
        checker = verifier.file_checker(f"{test_name}.cu")
        checker.check_label(
            'extern "C" __global__  __launch_bounds__({{.+}}) void ' + test_name + '_{{.+}}__gpu__('
        )
        checker.check_not('mma_sync')
        checker.run()

    def _get_np_datatype(self, p):
        from bfloat16 import bfloat16
        if p.element_type == ScalarType.bfloat16:
            return bfloat16

        return np.dtype(p.element_type.name)

    def _get_random_data(self, p):
        datatype = self._get_np_datatype(p)
        if p.element_type in [ScalarType.int8, ScalarType.int16, ScalarType.int32, ScalarType.int64]:
            return np.random.randint(-2, 2, p.shape, datatype)

        if p.element_type in [ScalarType.uint8, ScalarType.uint16, ScalarType.uint32, ScalarType.uint64]:
            return np.random.randint(0, 2, p.shape, datatype)

        return np.random.random(p.shape).astype(datatype)

    def _verify_matrix_multiplication_function(
        self,
        function: "accera.Function",
        package: Package,
        package_name: str,
        file_check_fn: Callable = None,
        check_correctness: bool = True,
        tolerance: float = 1e-5,
        file_list: List[str] = None,
        package_format: Package.Format = None,
        package_mode: Package.Mode = None,
        fail_on_error: bool = True,
        alpha = 1,
        beta = 1,
        quiet=True
    ) -> None:
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        package_format = package_format or self.PACKAGE_FORMAT
        package_mode = package_mode or self.PACKAGE_MODE

        if file_check_fn:
            package_format |= Package.Format.MLIR    # filecheck requires MLIR output

        with verifiers.VerifyPackage(self, package_name, output_dir, file_list=file_list) as v:
            package.build(name=package_name,
                          format=package_format,
                          mode=package_mode,
                          output_dir=output_dir,
                          fail_on_error=fail_on_error,
                          _quiet=quiet)

            if check_correctness:
                # Create the arrays with the appropriate layout
                A_test, B_test, C_test = (np.ndarray(p.shape, dtype=self._get_np_datatype(p), order=p.requested_layout.to_numpy_order()) for p in function.requested_args)

                # Create all the random input data
                A_test_data, B_test_data, C_test_data = (self._get_random_data(p) for p in function.requested_args)

                # Assign the default-ordered input data to the appropriately-ordered arrays
                A_test[:] = A_test_data
                B_test[:] = B_test_data
                C_test[:] = C_test_data

                C_ref = alpha * (A_test @ B_test) + (beta * C_test)

                v.check_correctness(function.name, before=(A_test, B_test, C_test), after=(A_test, B_test, C_ref), tolerance=tolerance)

            # apply optional file checks
            if file_check_fn:
                file_check_fn(v)

    def _rocm_matmul(self, test_name, M, N, K, block_tile, outer_tile_k, thread_tile=None, thread_coarsening_tile=(1, 1), inner_tile_k=None,
                     cache=(True, True, True), cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                     double_buffer=False, double_buffer_location=Constants.AUTO, vectorize=False,
                     tensorize=True, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=1, num_fused_passes=None, use_static_offsets=False,
                     scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                     bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                     array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                     array_element_types=[ScalarType.float32, ScalarType.float32, ScalarType.float32],
                     file_check_fn=None, tolerance=1e-5) -> None:
        from accera import Array, Nest, Package, Target

        target = Target(Target.Model.AMD_MI100)

        if thread_tile is not None and tensorize:
            raise ValueError("Can't specify both a thread_tile shape and tensorize")

        A = Array(role=Role.INPUT, element_type=array_element_types[0], shape=(M, K), layout=array_layouts[0])
        B = Array(role=Role.INPUT, element_type=array_element_types[1], shape=(K, N), layout=array_layouts[1])
        C = Array(role=Role.INPUT_OUTPUT, element_type=array_element_types[2], shape=(M, N), layout=array_layouts[2])

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile({
            i: block_tile[0],
            j: block_tile[1],
            k: outer_tile_k
        })

        if tensorize:
            tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)

            iii, jjj = schedule.tile({
                ii: tensor_splits[0] * thread_coarsening_tile[0],
                jj: tensor_splits[1] * thread_coarsening_tile[1]
            })

            iiii, jjjj, kkk = schedule.tile({
                iii: tensor_splits[0],
                jjj: tensor_splits[1],
                kk: tensor_splits[2]
            })
            outer_nest_order = (i, j, k, ii, jj, iii, jjj, kk)
            plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iiii, jjjj, kkk), outer_nest_order=outer_nest_order)
            plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes, use_static_offsets=use_static_offsets, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy)
            out_cache_idx = k if thread_coarsening_tile == (1, 1) else kk
        else:
            if thread_tile is None:
                thread_tile = block_tile

            if inner_tile_k is None:
                inner_tile_k = outer_tile_k

            iii, jjj, kkk = schedule.tile({
                ii: thread_tile[0],
                jj: thread_tile[1],
                kk: inner_tile_k
            })
            iiii, jjjj = schedule.tile({
                iii: thread_coarsening_tile[0],
                jjj: thread_coarsening_tile[1]
            })
            schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, iiii, jjjj, kkk)

            plan = schedule.create_plan(target=target)
            plan.bind(
                mapping={
                    i: bind_order[0],
                    j: bind_order[1],
                    iii: bind_order[2],
                    jjj: bind_order[3]
                }
            )

            out_cache_idx = k

        if cache[0]:
            plan.cache(
                A, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[0], strategy=_CacheStrategy.BLOCKED
            )
        if cache[1]:
            plan.cache(
                B, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[1], strategy=_CacheStrategy.STRIPED
            )
        if cache[2]:
            acc_loc = target.MemorySpace.MMA_FRAGMENT if tensorize else target.MemorySpace.PRIVATE
            plan.cache(
                C, index=out_cache_idx, vectorize=vectorize, location=acc_loc, layout=cache_layouts[2]
            )

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=ROCM_AVAILABLE,
            tolerance=tolerance,
            file_check_fn=file_check_fn,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR | Package.Format.DEFAULT
        )


    def _matmul_relu_plan(self, M, N, K, block_tile, thread_tile, outer_tile_k, inner_tile_k, thread_coarsening_tile, bind_order, target, A, B, C):
        import accera as acc

        matmul_nest = _matmul_nest(M, N, K, A, B, C)
        relu_nest = _relu_nest(M, N, C)

        matmul_sched = matmul_nest.create_schedule()
        relu_sched = relu_nest.create_schedule()
        schedule = acc.fuse(matmul_sched, relu_sched, partial = 2)
        f, i, j, k = schedule.get_indices()

        ii, jj, kk = schedule.tile({
            i: block_tile[0],
            j: block_tile[1],
            k: outer_tile_k
        })

        if thread_tile is None:
            thread_tile = block_tile

        if inner_tile_k is None:
            inner_tile_k = outer_tile_k

        iii, jjj, kkk = schedule.tile({
            ii: thread_tile[0],
            jj: thread_tile[1],
            kk: inner_tile_k
        })
        iiii, jjjj = schedule.tile({
            iii: thread_coarsening_tile[0],
            jjj: thread_coarsening_tile[1]
        })
        schedule.reorder(i, j, ii, jj, iii, jjj, f, k, kk, iiii, jjjj, kkk)

        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: bind_order[0],
                j: bind_order[1],
                iii: bind_order[2],
                jjj: bind_order[3]
            }
        )

        out_cache_idx = jjj
        in_cache_idx = ii

        return plan, in_cache_idx, out_cache_idx


    def _matmul_relu_tensor_plan(self, M, N, K, block_tile, outer_tile_k, thread_coarsening_tile, mma_shape, num_total_passes, use_static_offsets, num_fused_passes, scheduling_policy, target, A, B, C, prologue_op, epilogue_op):
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        schedule = _matmul_tensor_schedule(M, N, K, A, B, C, block_tile, outer_tile_k, thread_coarsening_tile, tensor_splits)
        i, j, ii, jj, iii, jjj, k, kk, iiii, jjjj, kkk = schedule.get_indices()

        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_Y,
                j: target.GridUnit.BLOCK_X,
                ii: target.GridUnit.WARP_Y,
                jj: target.GridUnit.WARP_X
            }
        )
        plan.tensorize(indices=(iiii, jjjj, kkk), mma_shape=mma_shape, num_total_passes=num_total_passes, use_static_offsets=use_static_offsets, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy, prologue_op=prologue_op, prologue_arg=0.0, epilogue_op=epilogue_op)
        out_cache_idx = jj if thread_coarsening_tile == (1, 1) else k
        in_cache_idx = ii

        return plan, in_cache_idx, out_cache_idx


    def _matmul_relu(self, test_name, M, N, K, block_tile, outer_tile_k, thread_tile=None, thread_coarsening_tile=(1, 1), inner_tile_k=None,
                    cache=(True, True, True), cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                    double_buffer=False, double_buffer_location=Constants.AUTO, vectorize=False, prologue_op=_MMAFragmentOp.SET, epilogue_op=_MMAFragmentOp.ReLU,
                    tensorize=True, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=1, num_fused_passes=None, use_static_offsets=False,
                    scheduling_policy=_MMASchedulingPolicy.PASS_ORDER, model=Target.Model.AMD_MI100,
                    bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                    array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                    array_element_types=[ScalarType.float32, ScalarType.float32, ScalarType.float32],
                    file_check_fn=None, tolerance=1e-5) -> None:
        from accera import Array, Package, Target

        target = Target(model)

        if thread_tile is not None and tensorize:
            raise ValueError("Can't specify both a thread_tile shape and tensorize")

        A = Array(role=Role.INPUT, element_type=array_element_types[0], shape=(M, K), layout=array_layouts[0])
        B = Array(role=Role.INPUT, element_type=array_element_types[1], shape=(K, N), layout=array_layouts[1])
        C = Array(role=Role.INPUT_OUTPUT, element_type=array_element_types[2], shape=(M, N), layout=array_layouts[2])

        if tensorize:
            plan, in_cache_idx, out_cache_idx = self._matmul_relu_tensor_plan(M, N, K, block_tile, outer_tile_k, thread_coarsening_tile, mma_shape, num_total_passes, use_static_offsets, num_fused_passes, scheduling_policy, target, A, B, C, prologue_op, epilogue_op)
        else:
            plan, in_cache_idx, out_cache_idx = self._matmul_relu_plan(M, N, K, block_tile, thread_tile, outer_tile_k, inner_tile_k, thread_coarsening_tile, bind_order, target, A, B, C)

        if cache[0]:
            plan.cache(
                A, index=in_cache_idx, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[0], strategy=_CacheStrategy.BLOCKED
            )
        if cache[1]:
            plan.cache(
                B, index=in_cache_idx, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[1], strategy=_CacheStrategy.STRIPED
            )
        if cache[2]:
            acc_loc = target.MemorySpace.MMA_FRAGMENT if tensorize else target.MemorySpace.PRIVATE
            plan.cache(
                C, index=out_cache_idx, vectorize=vectorize, location=acc_loc, layout=cache_layouts[2]
            )

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=ROCM_AVAILABLE if model == Target.Model.AMD_MI100 else CUDA_AVAILABLE,
            tolerance=tolerance,
            file_check_fn=file_check_fn,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR_VERBOSE | Package.Format.DEFAULT,
            beta = 0 if tensorize and cache[2] else 1
        )


    def _rocm_batch_matmul(self, test_name, batch_count, M, N, K, block_tile, outer_tile_k, b_split=None, thread_tile=None, thread_coarsening_tile=(1, 1), inner_tile_k=None,
                            cache=(True, True, True), cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                            double_buffer=False, double_buffer_location=Constants.AUTO, vectorize=False,
                            tensorize=True, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=1, num_fused_passes=None, use_static_offsets=False,
                            scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                            bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                            array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                            array_element_types=[ScalarType.float32, ScalarType.float32, ScalarType.float32],
                            file_check_fn=None, tolerance=1e-5) -> None:
        from accera import Array, Nest, Package, Target

        target = Target(Target.Model.AMD_MI100)

        if thread_tile is not None and tensorize:
            raise ValueError("Can't specify both a thread_tile shape and tensorize")

        if thread_tile is None:
            thread_tile = block_tile

        if inner_tile_k is None:
            inner_tile_k = outer_tile_k

        A = Array(role=Role.INPUT, element_type=array_element_types[0], shape=(batch_count, M, K), layout=array_layouts[0])
        B = Array(role=Role.INPUT, element_type=array_element_types[1], shape=(batch_count, K, N), layout=array_layouts[1])
        C = Array(role=Role.INPUT_OUTPUT, element_type=array_element_types[2], shape=(batch_count, M, N), layout=array_layouts[2])

        nest = Nest(shape=(batch_count, M, N, K))
        b, i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[b, i, j] += A[b, i, k] * B[b, k, j]

        schedule = nest.create_schedule()

        if b_split:
            bb, ii, jj, kk = schedule.tile({
                b: b_split,
                i: block_tile[0],
                j: block_tile[1],
                k: outer_tile_k
            })
        else:
            ii, jj, kk = schedule.tile({
                i: block_tile[0],
                j: block_tile[1],
                k: outer_tile_k
            })

        if tensorize:
            tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)

            iii, jjj, kkk = schedule.tile({
                ii: tensor_splits[0],
                jj: tensor_splits[1],
                kk: tensor_splits[2]
            })
            if b_split:
                outer_nest_order = (i, j, k, ii, jj, b, kk, bb)
            else:
                outer_nest_order = (b, i, j, k, ii, jj, kk)
            plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kkk), outer_nest_order=outer_nest_order)
            plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes, use_static_offsets=use_static_offsets, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy)
        else:
            iii, jjj, kkk = schedule.tile({
                ii: thread_tile[0],
                jj: thread_tile[1],
                kk: inner_tile_k
            })
            iiii, jjjj = schedule.tile({
                iii: thread_coarsening_tile[0],
                jjj: thread_coarsening_tile[1]
            })
            if b_split:
                schedule.reorder(i, j, k, ii, jj, b, kk, bb, iii, jjj, iiii, jjjj, kkk)
            else:
                schedule.reorder(b, i, j, k, ii, jj, kk, iii, jjj, iiii, jjjj, kkk)

            plan = schedule.create_plan(target=target)
            plan.bind(
                mapping={
                    i: bind_order[0],
                    j: bind_order[1],
                    iii: bind_order[2],
                    jjj: bind_order[3]
                }
            )

        if cache[0]:
            plan.cache(
                A, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[0]
            )
        if cache[1]:
            plan.cache(
                B, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[1]
            )
        if cache[2]:
            acc_loc = target.MemorySpace.MMA_FRAGMENT if tensorize else target.MemorySpace.PRIVATE
            plan.cache(
                C, index=k, vectorize=vectorize, location=acc_loc, layout=cache_layouts[2]
            )

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=ROCM_AVAILABLE,
            tolerance=tolerance,
            file_check_fn=file_check_fn,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR | Package.Format.DEFAULT
        )


    # This should produce MFMA instructions
    def test_rocm_tensorize_single_block_single_warp_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 16
        N = M
        K = M
        outer_tile_x = 16
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, k, ii, jj)
        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    @unittest.skip("TODO: This exposes a known bug, Chuck is working on fixing this issue.")
    def test_rocm_tensorize_single_block_single_warp_output_reordered_indices(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 16
        N = M
        K = M
        outer_tile_x = 16
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)
        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)


        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_single_block_single_warp_output_reordered_A(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 16
        N = M
        K = M
        outer_tile_x = 16
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)
        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            # TODO: re-enable test when non-row-major arrays are supported
            # if ROCM_AVAILABLE:
            #     self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_single_block_single_warp_output_reordered_A_and_indices(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 16
        N = M
        K = M
        outer_tile_x = 16
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            # TODO: re-enable test when non-row-major arrays are supported
            # if ROCM_AVAILABLE:
            #     self._verify_matmul(function, A, B, C, v)


    # This should not produce MFMA instructions
    def test_rocm_no_tensorize_multi_block_multi_warp_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })
        iii, jjj, kk = schedule.tile({
            ii: 2,
            jj: 2,
            k: 16
        })

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_Y,
                j: target.GridUnit.BLOCK_X,
                ii: target.GridUnit.THREAD_Y,
                jj: target.GridUnit.THREAD_X
            }
        )

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_no_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output_boundary(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 704
        N = 384
        K = 4096
        outer_tile_x = 128
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y,
        })

        mma_shape = _MMAShape.M64xN64xK1_B4
        num_total_passes = 64
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output_reordered_indices(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            ## This test is failing
            # if ROCM_AVAILABLE:
            #     self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output_reordered_A(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            # TODO: re-enable test when non-row-major arrays are supported
            # if ROCM_AVAILABLE:
            #     self._verify_matmul(function, A, B, C, v)


    # This should produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output_reordered_A_and_indices(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            # TODO: re-enable test when non-row-major arrays are supported
            # if ROCM_AVAILABLE:
            #     self._verify_matmul(function, A, B, C, v)

    # This should produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output_newsplits(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 1
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should not produce MFMA instructions
    def test_rocm_tensorize_same_mul_operands(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            a = A[i, k]
            b = B[k, j]
            C[i, j] += a * a

        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir,
            )

            self._check_cu_has_no_mfma(test_name, v)

            # TODO: test that this function returns A * A.transpose
            # if ROCM_AVAILABLE:
            #     self._verify_matmul(function, A, B, C, v)


    # This should not produce MFMA instructions
    def test_rocm_tensorize_bad_accum_operands(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        D = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            a = A[i, k]
            b = B[k, j]
            prod = a * b
            c = C[i, j]
            d = D[i, j]
            C[i, j] = d + prod

        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C, D), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_no_mfma(test_name, v)


    # This should succeed
    # This should produce MFMA instructions
    def test_rocm_tensorize_reversed_mul_order(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            a = A[i, k]
            b = B[k, j]
            prod = b * a
            c = C[i, j]
            C[i, j] = c + prod

        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)

        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_mfma(test_name, v)

            if ROCM_AVAILABLE:
                self._verify_matmul(function, A, B, C, v)


    # This should not produce MFMA instructions
    def test_rocm_tensorize_bad_C_indexing(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            a = A[i, k]
            b = B[k, j]
            prod = a * b
            c = C[i, j]
            C[j, i] = c + prod

        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        mma_shape = _MMAShape.M16xN16xK4_B1
        num_total_passes = 4
        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            k: tensor_splits[2]
        })

        outer_nest_order = (i, j, ii, jj, k)
        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu", f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._check_cu_has_no_mfma(test_name, v)


    def test_cpu(self) -> None:
        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            a = A[i, k]
            b = B[k, j]
            prod = a * b
            c = C[i, j]
            C[i, j] = c + prod

        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })
        iii, jjj, kk = schedule.tile({
            ii: 2,
            jj: 2,
            k: 16
        })

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        plan = schedule.create_plan()

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir) as v:
            package.build(test_name,
                format=Package.Format.MLIR_DYNAMIC,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            self._verify_matmul(function, A, B, C, v)


    def _cuda_tensorize(self, M, N, K, outer_tile_x, outer_tile_y, mma_shape, num_total_passes, tolerance=1e-5, intype=ScalarType.float16, outtype=ScalarType.float16, num_fused_passes=None, scheduling_policy=_MMASchedulingPolicy.PASS_ORDER, verify=True) -> None:
        from accera import Target
        A = Array(role=Role.INPUT, element_type=intype, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=intype, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=outtype, shape=(M, N))

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })

        target = Target(Target.Model.NVIDIA_RTX_A6000)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
        iii, jjj, kk = schedule.tile({
            ii: tensor_splits[0], # We want (ii, jj) to map to the warp in the block
            jj: tensor_splits[1],
            k: tensor_splits[2] # All threads run the k loop, the kk loop will be tensorized
        })

        outer_nest_order = (i, j, k, ii, jj)
        plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kk), outer_nest_order=outer_nest_order)
        plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy)

        package = Package()
        test_name = "test_cuda_tensorize"
        test_name += f"_{M}x{N}x{K}"
        test_name += f"_{get_type_str(intype)}"
        test_name += f"_{get_type_str(outtype)}"
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=CUDA_AVAILABLE and verify,
            tolerance=tolerance,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.DEFAULT | Package.Format.MLIR
        )

    def test_cuda_tensorize_16x16x16_fp16_fp16(self) -> None:
        self._cuda_tensorize(16, 16, 16, 16, 16, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-3)

    def test_cuda_tensorize_16x16x32_fp16_fp16(self) -> None:
        self._cuda_tensorize(16, 16, 32, 16, 16, _MMAShape.M16xN16xK16_B1, 2, tolerance=1e-2)

    def test_cuda_tensorize_16x16x384_fp16_fp32(self) -> None:
        self._cuda_tensorize(16, 16, 384, 16, 16, _MMAShape.M16xN16xK16_B1, 12, tolerance=1e-2,
                             intype=ScalarType.float16, outtype=ScalarType.float32, num_fused_passes=4)

    def test_cuda_tensorize_64x128x64_fp16_fp16(self) -> None:
        self._cuda_tensorize(64, 128, 64, 64, 64, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-2)

    def test_cuda_tensorize_1024x512x1024_fp16_fp16(self) -> None:
        self._cuda_tensorize(1024, 512, 1024, 64, 64, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-2)

    def test_cuda_tensorize_1024x1024x1024_fp16_fp32(self) -> None:
        self._cuda_tensorize(1024, 1024, 1024, 64, 64, _MMAShape.M16xN16xK16_B1, 32, tolerance=1e-2, intype=ScalarType.float16,
                             outtype=ScalarType.float32, num_fused_passes=8, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_16x16x16_fp16_fp32(self) -> None:
        self._cuda_tensorize(16, 16, 16, 16, 16, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-3, intype=ScalarType.float16,
                             outtype=ScalarType.float32)

    def test_cuda_tensorize_16x16x256_bfp16_fp32(self) -> None:
        self._cuda_tensorize(16, 16, 256, 16, 16, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-5, intype=ScalarType.bfloat16,
                             outtype=ScalarType.float32)

    def test_cuda_tensorize_16x16x128_i8_i32(self) -> None:
        self._cuda_tensorize(16, 16, 128, 16, 16, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-5, intype=ScalarType.int8,
                             outtype=ScalarType.int32)

    def test_cuda_tensorize_64x16x128_ui8_i32(self) -> None:
        self._cuda_tensorize(64, 16, 128, 16, 16, _MMAShape.M16xN16xK16_B1, 1, tolerance=1e-5, intype=ScalarType.uint8,
                             outtype=ScalarType.int32)

    def test_cuda_tensorize_32x8x16_fp16_fp16(self) -> None:
        self._cuda_tensorize(32, 8, 16, 32, 8, _MMAShape.M32xN8xK16_B1, 1, tolerance=1e-3)

    def test_cuda_tensorize_32x8x32_fp16_fp16(self) -> None:
        self._cuda_tensorize(32, 8, 32, 32, 8, _MMAShape.M32xN8xK16_B1, 2, tolerance=1e-2)

    def test_cuda_tensorize_32x8x384_fp16_fp32(self) -> None:
        self._cuda_tensorize(32, 8, 384, 32, 8, _MMAShape.M32xN8xK16_B1, 12, tolerance=1e-2, intype=ScalarType.float16,
                             outtype=ScalarType.float32, num_fused_passes=4)

    def test_cuda_tensorize_64x64x64_fp16_fp16(self) -> None:
        self._cuda_tensorize(64, 64, 64, 64, 64, _MMAShape.M32xN8xK16_B1, 1, tolerance=1e-2)

    def test_cuda_tensorize_1024x1024x2048_fp16_fp16(self) -> None:
        self._cuda_tensorize(1024, 1024, 2048, 64, 64, _MMAShape.M32xN8xK16_B1, 1, tolerance=1e-2)

    def test_cuda_tensorize_1024x1024x512_fp16_fp32(self) -> None:
        self._cuda_tensorize(1024, 1024, 512, 64, 64, _MMAShape.M32xN8xK16_B1, 32, tolerance=1e-2, intype=ScalarType.float16,
                             outtype=ScalarType.float32, num_fused_passes=8, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_32x8x192_bfp16_fp32(self) -> None:
        self._cuda_tensorize(32, 8, 192, 32, 8, _MMAShape.M32xN8xK16_B1, 12, tolerance=1e-5, intype=ScalarType.bfloat16,
                             outtype=ScalarType.float32, num_fused_passes=4)

    def test_cuda_tensorize_32x8x96_i8_i32(self) -> None:
        self._cuda_tensorize(32, 8, 96, 32, 8, _MMAShape.M32xN8xK16_B1, 6, tolerance=1e-5, intype=ScalarType.int8,
                             outtype=ScalarType.int32, num_fused_passes=2)

    def test_cuda_tensorize_32x64x48_ui8_i32(self) -> None:
        self._cuda_tensorize(32, 64, 48, 32, 8, _MMAShape.M32xN8xK16_B1, 3, tolerance=1e-5, intype=ScalarType.uint8,
                             outtype=ScalarType.int32, num_fused_passes=1)

    def test_cuda_tensorize_8x32x16_fp16_fp16(self) -> None:
        self._cuda_tensorize(8, 32, 16, 8, 32, _MMAShape.M8xN32xK16_B1, 1, tolerance=1e-3)

    def test_cuda_tensorize_8x32x32_fp16_fp16(self) -> None:
        self._cuda_tensorize(8, 32, 32, 8, 32, _MMAShape.M8xN32xK16_B1, 2, tolerance=1e-2)

    def test_cuda_tensorize_8x32x384_fp16_fp32(self) -> None:
        self._cuda_tensorize(8, 32, 384, 8, 32, _MMAShape.M8xN32xK16_B1, 12, tolerance=1e-2, intype=ScalarType.float16,
                             outtype=ScalarType.float32, num_fused_passes=4)

    def test_cuda_tensorize_128x64x64_fp16_fp16(self) -> None:
        self._cuda_tensorize(128, 64, 64, 64, 64, _MMAShape.M8xN32xK16_B1, 1, tolerance=1e-2)

    def test_cuda_tensorize_1024x1024x1024_fp16_fp16(self) -> None:
        self._cuda_tensorize(1024, 1024, 1024, 64, 64, _MMAShape.M8xN32xK16_B1, 4, tolerance=1e-2)

    def test_cuda_tensorize_1024x1024x2048_fp16_fp32(self) -> None:
        self._cuda_tensorize(1024, 1024, 2048, 64, 64, _MMAShape.M8xN32xK16_B1, 32, tolerance=1e-2, intype=ScalarType.float16,
                             outtype=ScalarType.float32, num_fused_passes=4, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_512x512x1024_bfp16_fp32(self) -> None:
        self._cuda_tensorize(512, 512, 1024, 64, 64, _MMAShape.M8xN32xK16_B1, 16, tolerance=1e-5, intype=ScalarType.bfloat16,
                             outtype=ScalarType.float32, num_fused_passes=4, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_512x512x64_i8_i32(self) -> None:
        self._cuda_tensorize(512, 512, 64, 64, 64, _MMAShape.M8xN32xK16_B1, 4, tolerance=1e-5, intype=ScalarType.int8,
                             outtype=ScalarType.int32, num_fused_passes=4, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_512x256x64_ui8_i32(self) -> None:
        self._cuda_tensorize(512, 256, 64, 64, 64, _MMAShape.M8xN32xK16_B1, 4, tolerance=1e-5, intype=ScalarType.uint8,
                             outtype=ScalarType.int32, num_fused_passes=4, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_16x16x16_fp32_fp32(self) -> None:
        self._cuda_tensorize(16, 16, 16, 16, 16, _MMAShape.M16xN16xK8_B1, 1, tolerance=1e-3, intype=ScalarType.float32,
                             outtype=ScalarType.float32)

    def test_cuda_tensorize_16x16x32_fp32_fp32(self) -> None:
        self._cuda_tensorize(16, 16, 32, 16, 16, _MMAShape.M16xN16xK8_B1, 2, tolerance=1e-3, intype=ScalarType.float32,
                             outtype=ScalarType.float32)

    def test_cuda_tensorize_16x16x384_fp32_fp32(self) -> None:
        self._cuda_tensorize(16, 16, 384, 16, 16, _MMAShape.M16xN16xK8_B1, 12, tolerance=1e-3, intype=ScalarType.float32,
                             outtype=ScalarType.float32, num_fused_passes=4)

    def test_cuda_tensorize_64x64x64_fp32_fp32(self) -> None:
        self._cuda_tensorize(64, 64, 64, 64, 64, _MMAShape.M16xN16xK8_B1, 1, tolerance=1e-3, intype=ScalarType.float32,
                             outtype=ScalarType.float32)

    def test_cuda_tensorize_1024x1024x1024_fp32_fp32(self) -> None:
        self._cuda_tensorize(1024, 1024, 1024, 64, 64, _MMAShape.M16xN16xK8_B1, 1, tolerance=1e-3, intype=ScalarType.float32,
                             outtype=ScalarType.float32)

    def test_cuda_tensorize_1024x1024x2048_fp32_fp32(self) -> None:
        self._cuda_tensorize(1024, 1024, 2048, 64, 64, _MMAShape.M16xN16xK8_B1, 32, tolerance=1e-3, intype=ScalarType.float32,
                             outtype=ScalarType.float32, num_fused_passes=8, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)


    def _cuda_cache_tensorize(self, M, N, K, outer_tile_m, outer_tile_n, outer_tile_k, test_name,
                              tensorize=True,
                              mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, cache=False, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              double_buffer=False, double_buffer_location=Constants.AUTO, vectorize=False,
                              scheduling_policy=_MMASchedulingPolicy.PASS_ORDER, epilogue_op=_MMAFragmentOp.NONE,
                              bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                              array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              element_type=ScalarType.float16, output_type=ScalarType.float16) -> None:
        from accera import Array, Nest, Package, Target

        A = Array(role=Role.INPUT, element_type=element_type, shape=(M, K), layout=array_layouts[0])
        B = Array(role=Role.INPUT, element_type=element_type, shape=(K, N), layout=array_layouts[1])
        C = Array(role=Role.INPUT_OUTPUT, element_type=output_type, shape=(M, N), layout=array_layouts[2])

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile({
            i: outer_tile_m,
            j: outer_tile_n,
            k: outer_tile_k
        })

        target = Target(Target.Model.NVIDIA_RTX_A6000)
        elem_bytes = 4 if element_type == ScalarType.float32 else 2
        shared_mem_usage_bytes = elem_bytes * (outer_tile_m + outer_tile_n) * outer_tile_k
        blocks_per_SM = 2 if (2 * shared_mem_usage_bytes) <= target.max_shared_memory_per_block else None
        use_dynamic_shared_mem = shared_mem_usage_bytes > target.max_static_shared_memory_per_block
        dynamic_shared_mem_usage_bytes = shared_mem_usage_bytes if use_dynamic_shared_mem else 0
        if tensorize:
            tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)
            iii, jjj, kkk = schedule.tile({
                ii: tensor_splits[0], # We want (ii, jj) to map to the warp in the block
                jj: tensor_splits[1],
                kk: tensor_splits[2] # All threads run the kk loop, the kkk loop will be tensorized
            })

            outer_nest_order = (i, j, k, ii, jj, kk)
            plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kkk), outer_nest_order=outer_nest_order, dynamic_shared_memory_size=dynamic_shared_mem_usage_bytes, blocks_per_SM=blocks_per_SM)
            plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes, scheduling_policy=scheduling_policy, epilogue_op=epilogue_op)
        else:
            # TODO : split this case into a different helper function as this is a tensorize helper
            default_thread_splits = (16, 16, 16)
            iii, jjj, kkk = schedule.tile({
                ii: default_thread_splits[0],
                jj: default_thread_splits[1],
                kk: default_thread_splits[2]
            })
            schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk)

            plan = schedule.create_plan(target=target, _dynamic_shared_memory_size=dynamic_shared_mem_usage_bytes, _blocks_per_SM=blocks_per_SM)
            plan.bind(
                mapping={
                    i: bind_order[0],
                    j: bind_order[1],
                    ii: bind_order[2],
                    jj: bind_order[3]
                }
            )

        if cache:
            if use_dynamic_shared_mem:
                plan.cache(
                    A, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, _shared_memory_offset=0, layout=cache_layouts[0], strategy=_CacheStrategy.BLOCKED
                )
                plan.cache(
                    B, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, _shared_memory_offset=outer_tile_m * outer_tile_k, layout=cache_layouts[1], strategy=_CacheStrategy.STRIPED
                )
            else:
                plan.cache(
                    A, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[0], strategy=_CacheStrategy.BLOCKED
                )
                plan.cache(
                    B, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[1], strategy=_CacheStrategy.STRIPED
                )

            acc_loc = target.MemorySpace.MMA_FRAGMENT if tensorize else target.MemorySpace.PRIVATE
            plan.cache(
                C, index=k, vectorize=vectorize, location=acc_loc, layout=Array.Layout.FIRST_MAJOR
            )

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)
        def file_check_dyn_mem_fn(v):
            checker = v.file_checker(f"{test_name}.cu")
            checker.check(f"{dynamic_shared_mem_usage_bytes}>>>")
            checker.run()

        def file_check_blocksPerSM_fn(v):
            checker = v.file_checker(f"{test_name}.cu")
            checker.check_label('extern "C" __global__  __launch_bounds__({{.+}}, ' + f'{blocks_per_SM}' + ') void ' + test_name + '_{{.+}}__gpu__(')
            checker.run()

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=CUDA_AVAILABLE,
            tolerance=1e-5 if element_type == ScalarType.float32 else 1e-2,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR | Package.Format.DEFAULT,
            file_check_fn=file_check_dyn_mem_fn if use_dynamic_shared_mem else (file_check_blocksPerSM_fn if blocks_per_SM > 0 else None)
        )

    def test_cuda_cache_tensorize(self) -> None:
        self._cuda_cache_tensorize(M=1024, N=1024, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_cuda_cache_tensorize")

    def test_cuda_cache_double_buffering_tensorize(self) -> None:
        self._cuda_cache_tensorize(M=1024, N=1024, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_cache_double_buffering_tensorize", tensorize=True, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_cuda_non_square_simple(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_non_square_simple", tensorize=False, cache=False, element_type=ScalarType.float32, output_type=ScalarType.float32)

    def test_cuda_non_square_last_major_inputs(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_inputs", tensorize=False,
                                   cache=False, vectorize=False, element_type=ScalarType.float32, output_type=ScalarType.float32,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_cuda_non_square_last_major_inputs_dynamic_shared(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_inputs_dynamic_shared", tensorize=False,
                                   cache=True, vectorize=False, element_type=ScalarType.float32, output_type=ScalarType.float32,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_cuda_non_square_last_major_output(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_output", tensorize=False,
                                   cache=False, vectorize=False, element_type=ScalarType.float32, output_type=ScalarType.float32,
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_non_square_last_major_inputs_output(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_inputs_output", tensorize=False,
                                   cache=False, vectorize=False, element_type=ScalarType.float32, output_type=ScalarType.float32,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_inputs(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_tensorize_non_square_last_major_inputs", tensorize=True,
                                   cache=False, vectorize=False,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_output(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_tensorize_non_square_last_major_output", tensorize=True,
                                   cache=False, vectorize=False,
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_output_dynamic_shared(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=256,
                                   test_name="test_cuda_tensorize_non_square_last_major_output_dynamic_shared", tensorize=True,
                                   cache=True, vectorize=False, double_buffer = True,
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_inputs_output(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_tensorize_non_square_last_major_inputs_output", tensorize=True,
                                   cache=False, vectorize=False,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_tensorize_non_square", tensorize=True, cache=False)

    def test_cuda_cache_non_square(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_cache_non_square", tensorize=False, cache=True, output_type=ScalarType.float32)

    def test_cuda_cache_double_buffering_non_square(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_cache_double_buffering_non_square", tensorize=False, cache=True,
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, output_type=ScalarType.float32)

    def test_cuda_cache_double_buffering_tensorize_non_square(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_cache_double_buffering_tensorize_non_square", tensorize=True, cache=True,
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_cuda_cache_double_buffering_tensorize_non_square_uneven_tile(self) -> None:
        self._cuda_cache_tensorize(M=768, N=576, K=1024, outer_tile_m=16, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_cache_double_buffering_tensorize_non_square_uneven_tile", tensorize=True, cache=True,
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_cuda_vectorized_cache_double_buffering_tensorize_non_square(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_vectorized_cache_double_buffering_tensorize_non_square", tensorize=True, cache=True,
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True)

    def test_cuda_vectorized_cache_double_buffering_tensorize_non_square_relu(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, epilogue_op=_MMAFragmentOp.ReLU,
                                    test_name="test_cuda_vectorized_cache_double_buffering_tensorize_non_square_relu", tensorize=True, cache=True,
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True)

    def test_cuda_vectorized_cache_double_buffering_tensorize_non_square_blockorder(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_vectorized_cache_double_buffering_tensorize_non_square_blockorder", tensorize=True, cache=True,
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_vectorized_cache_non_square(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square", tensorize=False,
                                   cache=True, vectorize=True, output_type=ScalarType.float32,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_cuda_vectorized_cache_non_square_last_major_inputs(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square_last_major_inputs", tensorize=False,
                                   cache=True, vectorize=True, output_type=ScalarType.float32,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_cuda_vectorized_cache_non_square_last_major_output(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square_last_major_output", tensorize=False,
                                   cache=True, vectorize=True, output_type=ScalarType.float32,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_vectorized_cache_non_square_last_major_inputs_output(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square_last_major_inputs_output", tensorize=False,
                                   cache=True, vectorize=True, output_type=ScalarType.float32,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_vectorized_cache_non_square_small_tiles(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=32,
                                   test_name="test_cuda_vectorized_cache_non_square_small_tiles", tensorize=False,
                                   cache=True, vectorize=True, output_type=ScalarType.float32,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_cuda_vectorized_cache_non_square_transpose(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square_transpose", tensorize=False,
                                   cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR], output_type=ScalarType.float32,
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_cuda_vectorized_cache_non_square_double_buffer(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square_double_buffer", tensorize=False,
                                   cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, output_type=ScalarType.float32,
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_cuda_vectorized_cache_non_square_double_buffer_small_tiles(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=32,
                                   test_name="test_cuda_vectorized_cache_non_square_double_buffer_small_tiles", tensorize=False,
                                   cache=True, vectorize=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, output_type=ScalarType.float32,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_cuda_vectorized_cache_non_square_double_buffer_transpose(self) -> None:
        self._cuda_cache_tensorize(M=1280, N=768, K=1024, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_vectorized_cache_non_square_double_buffer_transpose", tensorize=False,
                                   cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR],
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, output_type=ScalarType.float32,
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def _rocm_tensorize(self, M, N, K, outer_tile_m, outer_tile_n, outer_tile_k=None,
                        mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=1, tolerance=1e-5,
                        intype=ScalarType.float32, outtype=ScalarType.float32,
                        use_static_offsets=False, num_fused_passes=None,
                        scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                        thread_coarsening_tile=(1, 1),
                        test_name=None) -> None:

        target = Target(Target.Model.AMD_MI100)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes)

        if outer_tile_k is None:
            outer_tile_k = K

        if test_name is None:
            test_name = "test_rocm_tensorize"
            test_name += f"_{M}x{N}x{K}"
            test_name += "_" + get_type_str(intype)
            test_name += "_" + get_type_str(outtype)
            test_name += "_" + "x".join([str(dim) for dim in tensor_splits])
            test_name += "_" + mma_shape.name
            if use_static_offsets:
                test_name += "_tensormap"
            if num_fused_passes is not None:
                test_name += "_p" + str(num_fused_passes)
            if thread_coarsening_tile[0] != 1 or thread_coarsening_tile[1] != 1:
                test_name += "_t" + str(thread_coarsening_tile[0]) + "_" + str(thread_coarsening_tile[1])

        def file_check_fn(v):
            checker = v.file_checker(f"{test_name}.cu")
            checker.check("rocwmma::load_matrix_sync<1,")
            checker.check("rocwmma::store_matrix_sync<1,")
            checker.run()

        self._rocm_matmul(test_name, M, N, K,
                          block_tile=(outer_tile_m, outer_tile_n),
                          outer_tile_k=outer_tile_k,
                          tensorize=True,
                          mma_shape=mma_shape,
                          num_total_passes=num_total_passes,
                          num_fused_passes=num_fused_passes,
                          cache=(False, False, False),
                          use_static_offsets=use_static_offsets,
                          scheduling_policy=scheduling_policy,
                          thread_coarsening_tile=thread_coarsening_tile,
                          array_element_types=[intype, intype, outtype],
                          file_check_fn=file_check_fn if use_static_offsets else None,
                          tolerance=tolerance)

    def test_rocm_tensorize_16x16x16_fp32_fp32_16x16x16_M16xN16xK4_B1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

    def test_rocm_tensorize_32x32x32_fp32_fp32_32x32x32_M32xN32xK2_B1(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=16)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64)

    def test_rocm_tensorize_960x1024x1024_fp32_fp32_128x128x64_M64xN64xK1_B4(self) -> None:
        self._rocm_tensorize(960, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64)

    def test_rocm_tensorize_64x64x64_fp32_fp32_16x16x16_M16xN16xK4_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

    def test_rocm_tensorize_64x64x64_fp32_fp32_32x32x32_M32xN32xK2_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=16)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_16x16x16_M16xN16xK4_B1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_32x32x32_M32xN32xK2_B1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=16)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_64x64x64_M64xN64xK1_B2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_64x64x64_M64xN64xK1_B4(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64)

    def test_rocm_tensorize_16x16x16_fp16_fp32_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_32x32x32_fp16_fp32_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_16x16x16_bfp16_fp32_16x16x16_M16xN16xK8_B1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK8_B1, num_total_passes=2, tolerance=1e-3, intype=ScalarType.bfloat16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_64x64x64_M64xN64xK4_B2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_64x64x64_M64xN64xK4_B4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp32_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp32_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp32_64x64x64_M64xN64xK4_B2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp32_64x64x64_M64xN64xK4_B4(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_16x16x16_fp16_fp16_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_32x32x32_fp16_fp16_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp16_64x64x64_M64xN64xK4_B2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_32x32x32_bfp16_bfp16_32x32x32_M32xN32xK4_B1(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK4_B1, num_total_passes=8, tolerance=1e-2, intype=ScalarType.bfloat16, outtype=ScalarType.bfloat16)

    def test_rocm_tensorize_64x64x64_fp16_fp16_64x64x64_M64xN64xK4_B4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float16)

    # TODO: This requires tolerance to be set higher than the other tests (verify discrepancies)
    def test_rocm_tensorize_64x64x64_fp16_fp16_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp16_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B4(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16)

    def test_rocm_tensorize_16x16x16_i8_i32_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int32)

    def test_rocm_tensorize_64x64x64_i8_i32_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int32)

    def test_rocm_tensorize_64x64x64_i8_i32_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int32)

    def test_rocm_tensorize_128x128x64_i8_i8_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(128, 128, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int8)

    def test_rocm_tensorize_256x128x128_i8_i32_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(256, 128, 128, 128, 128, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int32)

    def test_rocm_tensorize_128x256x512_i8_i8_64x64x64_M64xN64xK4_B2(self) -> None:
        self._rocm_tensorize(128, 256, 512, 128, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int8)

    def test_rocm_tensorize_128x128x128_i8_i8_64x64x64_M64xN64xK4_B4(self) -> None:
        self._rocm_tensorize(128, 128, 128, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int8)

    def test_rocm_tensorize_16x16x16_i8_i16_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int16)

    def test_rocm_tensorize_64x64x64_i8_i16_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int16)

    def test_rocm_tensorize_64x64x64_i8_i16_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int16)

    def test_rocm_tensorize_128x128x64_i8_i16_16x16x16_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(128, 128, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int16)

    def test_rocm_tensorize_256x128x128_i8_i16_32x32x32_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(256, 128, 128, 128, 128, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-6, intype=ScalarType.int8, outtype=ScalarType.int16)

    # Testing precomputed index map optimization
    def test_rocm_tensorize_16x16x16_fp32_fp32_16x16x16_M16xN16xK4_B1_tensormap(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_32x32x32_fp32_fp32_32x32x32_M32xN32xK2_B1_tensormap(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B2_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B4_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_64x64x64_fp16_fp32_16x16x16_M16xN16xK16_B1_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_64x64x64_fp16_fp32_32x32x32_M32xN32xK8_B1_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_16x16x16_M16xN16xK16_B1_tensormap(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True)

    def test_rocm_tensorize_64x64x64_bfp16_fp32_64x64x64_M64xN64xK2_B2_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK2_B2, num_total_passes=32, tolerance=1e-3, intype=ScalarType.bfloat16, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_32x32x32_M32xN32xK8_B1_tensormap(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp32_64x64x64_M64xN64xK4_B2_tensormap(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-3, intype=ScalarType.float16, outtype=ScalarType.float32, use_static_offsets=True)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B4_tensormap(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True)

    # Testing tunable register usage
    def test_rocm_tensorize_16x16x16_fp32_fp32_16x16x16_M16xN16xK4_B1_tensormap_p1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True, num_fused_passes=1)

    def test_rocm_tensorize_16x16x16_fp32_fp32_16x16x16_M16xN16xK4_B1_p2(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=2, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_16x16x16_M16xN16xK16_B1_p1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=False, num_fused_passes=1)

    def test_rocm_tensorize_32x32x32_fp32_fp32_32x32x32_M32xN32xK2_B1_p1(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=1)

    def test_rocm_tensorize_32x32x32_fp32_fp32_32x32x32_M32xN32xK2_B1_tensormap_p16(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True, num_fused_passes=16, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_32x32x32_fp16_fp32_32x32x32_M32xN32xK8_B1_p2(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=2)

    def test_rocm_tensorize_32x32x32_fp16_fp16_32x32x32_M32xN32xK8_B1_tensormap_p4(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True, num_fused_passes=4)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B2_tensormap_p1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True, num_fused_passes=1, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B2_p4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=4)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B2_tensormap_p16(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True, num_fused_passes=16, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp32_fp32_64x64x64_M64xN64xK1_B2_p64(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=64)

    def test_rocm_tensorize_64x64x64_fp16_fp16_64x64x64_M64xN64xK4_B2_tensormap_p2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True, num_fused_passes=2)

    def test_rocm_tensorize_64x64x64_fp16_fp16_64x64x64_M64xN64xK4_B2_p4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=False, num_fused_passes=4, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp16_fp32_64x64x64_M64xN64xK4_B2_tensormap_p8(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float32, use_static_offsets=True, num_fused_passes=8, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp16_fp32_64x64x64_M64xN64xK4_B2_p16(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B4_tensormap_p1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True, num_fused_passes=1)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B4_p2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=False, num_fused_passes=2, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B4_tensormap_p4(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True, num_fused_passes=4, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_64x64x64_M64xN64xK4_B4_p8(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=16, tolerance=1e-2, intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=False, num_fused_passes=8)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_64x64x64_M64xN64xK1_B4_tensormap_p32(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=True, num_fused_passes=32, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_64x64x64_M64xN64xK1_B4_p64(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64, tolerance=1e-2, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=64, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    # Test arbitrary K
    def test_rocm_tensorize_16x16x4_fp32_fp32_16x16x4_M16xN16xK4_B1(self) -> None:
        self._rocm_tensorize(16, 16, 4, 16, 16, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=1)

    def test_rocm_tensorize_16x16x20_fp32_fp32_16x16x20_M16xN16xK4_B1(self) -> None:
        self._rocm_tensorize(16, 16, 20, 16, 16, mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=5)

    def test_rocm_tensorize_16x16x1264_fp16_fp32_16x16x1264_M16xN16xK16_B1(self) -> None:
        self._rocm_tensorize(16, 16, 1264, 16, 16, mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=79, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_32x32x48_fp32_fp32_32x32x48_M32xN32xK2_B1(self) -> None:
        self._rocm_tensorize(32, 32, 48, 32, 32, mma_shape=_MMAShape.M32xN32xK2_B1, num_total_passes=24)

    def test_rocm_tensorize_32x32x24_fp16_fp32_32x32x24_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(32, 32, 24, 32, 32, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=3, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_32x32x136_fp16_fp32_32x32x136_M32xN32xK8_B1(self) -> None:
        self._rocm_tensorize(32, 32, 136, 32, 32, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=17, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_64x64x1_fp32_fp32_64x64x1_M64xN64xK1_B2(self) -> None:
        self._rocm_tensorize(64, 64, 1, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B2, num_total_passes=1)

    def test_rocm_tensorize_64x64x44_fp16_fp32_64x64x44_M64xN64xK4_B2(self) -> None:
        self._rocm_tensorize(64, 64, 44, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B2, num_total_passes=11, tolerance=1e-3, intype=ScalarType.float16)

    def test_rocm_tensorize_64x64x3_fp32_fp32_64x64x3_M64xN64xK1_B4(self) -> None:
        self._rocm_tensorize(64, 64, 3, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=3)

    def test_rocm_tensorize_64x64x93_fp32_fp32_64x64x93_M64xN64xK1_B4(self) -> None:
        self._rocm_tensorize(64, 64, 93, 64, 64, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=93)

    def test_rocm_tensorize_64x64x20_fp16_fp32_64x64x20_M64xN64xK4_B4(self) -> None:
        self._rocm_tensorize(64, 64, 20, 64, 64, mma_shape=_MMAShape.M64xN64xK4_B4, num_total_passes=5, tolerance=1e-3, intype=ScalarType.float16)

    @expectedFailure(FailedReason.INVALID, "the hardware does not support the requested tensorcore shape")
    def test_rocm_tensorize_invalid_shape_output(self) -> None:
        self._rocm_tensorize(256, 256, 256, 64, 64, "test_rocm_tensorize_invalid_shape_output", False)

    def _rocm_cache_matmul(self, test_name, M, N, K,
                           block_tile,
                           outer_tile_k,
                           thread_tile=None,
                           thread_coarsening_tile=(1, 1),
                           inner_tile_k=1,
                           cache=(True, True, True),
                           cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                           double_buffer=False,
                           double_buffer_location=Constants.AUTO,
                           vectorize=False,
                           bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                           array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                           array_element_types=[ScalarType.float32, ScalarType.float32, ScalarType.float32],
                           file_check_fn=None,
                           tolerance=1e-5) -> None:
        self._rocm_matmul(test_name, M, N, K,
                          block_tile=block_tile,
                          outer_tile_k=outer_tile_k,
                          thread_tile=thread_tile,
                          thread_coarsening_tile=thread_coarsening_tile,
                          inner_tile_k=inner_tile_k,
                          tensorize=False,
                          cache=cache,
                          cache_layouts=cache_layouts,
                          double_buffer=double_buffer,
                          double_buffer_location=double_buffer_location,
                          vectorize=vectorize,
                          bind_order=bind_order,
                          array_layouts=array_layouts,
                          array_element_types=array_element_types,
                          file_check_fn=file_check_fn,
                          tolerance=tolerance)

    def _rocm_cache_tensorize(self, M, N, K,
                              block_tile,
                              outer_tile_k,
                              test_name,
                              mma_shape=_MMAShape.M16xN16xK4_B1,
                              thread_coarsening_tile=(1, 1),
                              num_total_passes=1,
                              cache=(True, True, True),
                              cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              double_buffer=False,
                              array_element_types=[ScalarType.float32, ScalarType.float32, ScalarType.float32],
                              double_buffer_location=Constants.AUTO,
                              vectorize=False,
                              use_static_offsets=False,
                              scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                              array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              tolerance=1e-5,
                              file_check_fn=None) -> None:
        self._rocm_matmul(test_name, M, N, K,
                          block_tile=block_tile,
                          outer_tile_k=outer_tile_k,
                          tensorize=True,
                          array_element_types=array_element_types,
                          mma_shape=mma_shape,
                          num_total_passes=num_total_passes,
                          cache=cache,
                          thread_coarsening_tile=thread_coarsening_tile,
                          cache_layouts=cache_layouts,
                          double_buffer=double_buffer,
                          double_buffer_location=double_buffer_location,
                          vectorize=vectorize,
                          use_static_offsets=use_static_offsets,
                          scheduling_policy=scheduling_policy,
                          array_layouts=array_layouts,
                          tolerance=tolerance,
                          file_check_fn=file_check_fn)

    def test_rocm_cache_tensorize(self) -> None:
        self._rocm_cache_tensorize(M=1024, N=1024, K=1024, block_tile=(64, 64), outer_tile_k=64, test_name="test_rocm_cache_tensorize")

    def test_rocm_cache_double_buffering_tensorize(self) -> None:
        self._rocm_cache_tensorize(M=1024, N=1024, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_cache_double_buffering_tensorize",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_rocm_non_square(self) -> None:
        self._rocm_matmul(M=1280, N=768, K=1024,
                          block_tile=(64, 64), outer_tile_k=64, inner_tile_k=32, thread_coarsening_tile=(2, 2),
                          cache=(False, False, False), tensorize=False,
                          test_name="test_rocm_non_square")

    def test_rocm_non_square_last_major_inputs(self) -> None:
        self._rocm_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                          test_name="test_rocm_non_square_last_major_inputs",
                          tensorize=False, cache=(False, False, False),
                          bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                          array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_rocm_non_square_last_major_output(self) -> None:
        self._rocm_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                          test_name="test_rocm_non_square_last_major_output",
                          tensorize=False, cache=(False, False, False),
                          bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                          array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_non_square_last_major_inputs_output(self) -> None:
        self._rocm_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                          test_name="test_rocm_non_square_last_major_inputs_output",
                          tensorize=False, cache=(False, False, False),
                          bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                          array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_tensorize_non_square(self) -> None:
        self._rocm_tensorize(M=1280, N=768, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_tensorize_non_square")

    def test_rocm_cache_non_square(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64, thread_coarsening_tile=(2, 2), test_name="test_rocm_cache_non_square")

    def test_rocm_cache_double_buffering_non_square(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64, thread_coarsening_tile=(2, 2), test_name="test_rocm_cache_double_buffering_non_square", double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_rocm_cache_double_buffering_tensorize_non_square(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_cache_double_buffering_tensorize_non_square",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True,
                                   scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_vectorized_cache_double_buffering_tensorize_blockorder(self) -> None:
        self._rocm_cache_tensorize(M=1024, N=1024, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_blockorder",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True,
                                   mma_shape=_MMAShape.M32xN32xK2_B1, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK1_B2(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK1_B2",
                                   mma_shape=_MMAShape.M64xN64xK1_B2, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK1_B4(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK1_B4",
                                   mma_shape=_MMAShape.M64xN64xK1_B4, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, scheduling_policy=_MMASchedulingPolicy.PASS_ORDER)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder_fp16(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder_fp16",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True,
                                   mma_shape=_MMAShape.M16xN16xK16_B1, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER,
                                   array_element_types=[ScalarType.float16, ScalarType.float16, ScalarType.float16], tolerance=1e-3)

    def test_rocm_vectorized_cache_double_buffering_tensorize_blockorder_fp16(self) -> None:
        self._rocm_cache_tensorize(M=1024, N=1024, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_blockorder_fp16",
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True,
                                   mma_shape=_MMAShape.M32xN32xK8_B1, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER,
                                   array_element_types=[ScalarType.float16, ScalarType.float16, ScalarType.float32], tolerance=1e-3)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK4_B2_fp16(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK4_B2_fp16",
                                   mma_shape=_MMAShape.M64xN64xK4_B2, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER,
                                   array_element_types=[ScalarType.float16, ScalarType.float16, ScalarType.float16], tolerance=1e-3)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK4_B4_fp16(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                   test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK4_B4_fp16",
                                   mma_shape=_MMAShape.M64xN64xK4_B4, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                                   array_element_types=[ScalarType.float16, ScalarType.float16, ScalarType.float32], tolerance=1e-3)

    def test_rocm_vectorized_cache_non_square(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                test_name="test_rocm_vectorized_cache_non_square",
                                vectorize=True,
                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_last_major_inputs(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                test_name="test_rocm_vectorized_cache_non_square_last_major_inputs",
                                vectorize=True,
                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_rocm_vectorized_cache_non_square_last_major_output(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                test_name="test_rocm_vectorized_cache_non_square_last_major_output",
                                vectorize=True,
                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_vectorized_cache_non_square_last_major_inputs_output(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                test_name="test_rocm_vectorized_cache_non_square_last_major_inputs_output",
                                vectorize=True,
                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_vectorized_cache_non_square_small_tiles(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=32,
                                test_name="test_rocm_vectorized_cache_non_square_small_tiles",
                                vectorize=True,
                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_transpose(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                test_name="test_rocm_vectorized_cache_non_square_transpose",
                                cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR],
                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_double_buffer(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                test_name="test_rocm_vectorized_cache_non_square_double_buffer",
                                double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_double_buffer_small_tiles(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=32,
                                test_name="test_rocm_vectorized_cache_non_square_double_buffer_small_tiles",
                                vectorize=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_double_buffer_transpose(self) -> None:
        self._rocm_cache_matmul(M=1280, N=768, K=1024, block_tile=(16, 16), outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_double_buffer_transpose",
                                   cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR],
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_tensormap(self) -> None:
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                    test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_tensormap",
                                    double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, use_static_offsets=True)

    def test_rocm_batchgemm_vectorized_cache_non_square_transpose(self) -> None:
        self._rocm_batch_matmul(batch_count=3, M=1280, N=768, K=1024, block_tile=(32, 32), outer_tile_k=64,
                                test_name="test_rocm_batchgemm_vectorized_cache_non_square_transpose", tensorize=False,
                                cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR],
                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_batchgemm_vectorized_cache_non_square_tensorize_transpose(self) -> None:
        self._rocm_batch_matmul(batch_count=3, M=1280, N=768, K=1024, block_tile=(32, 32), outer_tile_k=64,
                                test_name="test_rocm_batchgemm_vectorized_cache_non_square_tensorize_transpose", tensorize=True,
                                cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR],
                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_batchgemm_vectorized_cache_double_buffering_non_square(self) -> None:
        self._rocm_batch_matmul(batch_count=8, M=1280, N=768, K=1024, block_tile=(32, 32), outer_tile_k=64,
                                test_name="test_rocm_batchgemm_vectorized_cache_double_buffering_non_square", tensorize=False,
                                double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True)

    def test_rocm_batchgemm_vectorized_cache_double_buffering_tensorize_non_square_tensormap(self) -> None:
        self._rocm_batch_matmul(batch_count=8, M=1280, N=768, K=1024, block_tile=(64, 64), outer_tile_k=64,
                                test_name="test_rocm_batchgemm_vectorized_cache_double_buffering_tensorize_non_square_tensormap",
                                double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, use_static_offsets=True)

    def test_rocm_batchgemm_vectorized_cache_double_buffering_tensorize_square(self) -> None:
        self._rocm_batch_matmul(batch_count=1, M=64, N=64, K=64, block_tile=(64, 64), outer_tile_k=64,
                                test_name="test_rocm_batchgemm_vectorized_cache_double_buffering_tensorize_square",
                                double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, use_static_offsets=False)

    def test_rocm_batchgemm_vectorized_cache_double_buffering_tensorize_square_bsplit(self) -> None:
        self._rocm_batch_matmul(batch_count=32, b_split=2, M=128, N=128, K=128, block_tile=(16, 16), outer_tile_k=16, cache=(True, True, False),
                                test_name="test_rocm_batchgemm_vectorized_cache_double_buffering_tensorize_square_bsplit",
                                double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, use_static_offsets=False)

    # Testing thread coarsening with tensorization
    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_64x64x64_M64xN64xK1_B4_t2_2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64, tolerance=1e-2, thread_coarsening_tile=(2, 2))

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_32x32x32_M32xN32xK8_B1_tensormap_t2_1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 128, 128, mma_shape=_MMAShape.M32xN32xK8_B1, num_total_passes=4, tolerance=1e-2, thread_coarsening_tile=(2, 1), intype=ScalarType.float16, outtype=ScalarType.float16, use_static_offsets=True)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK1_B4_t2_1(self) -> None:
        test_name = "test_rocm_vectorized_cache_double_buffering_tensorize_non_square_M64xN64xK1_B4_t2_1"
        def file_check_fn(v):
            checker = v.file_checker(f"{test_name}.cu")
            checker.check("<<<dim3(12, 10, 1), dim3(64, 1, 1), 0>>>")
            checker.run()
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(128, 64), outer_tile_k=64, thread_coarsening_tile=(2, 1),
                                   mma_shape=_MMAShape.M64xN64xK1_B4, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   test_name=test_name, file_check_fn=file_check_fn, vectorize=True, scheduling_policy=_MMASchedulingPolicy.PASS_ORDER)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder_fp16_t2_2(self) -> None:
        test_name = "test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder_fp16_t2_2"
        def file_check_fn(v):
            checker = v.file_checker(f"{test_name}.cu")
            checker.check("<<<dim3(6, 10, 1), dim3(256, 4, 1), 0>>>")
            checker.run()
        self._rocm_cache_tensorize(M=1280, N=768, K=1024, block_tile=(128, 128), outer_tile_k=64, thread_coarsening_tile=(2, 2),
                                   test_name=test_name, file_check_fn=file_check_fn, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, mma_shape=_MMAShape.M16xN16xK16_B1, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER,
                                   array_element_types=[ScalarType.float16, ScalarType.float16, ScalarType.float32], tolerance=1e-3)

    def test_rocm_matmul_relu_vectorized(self) -> None:
        self._matmul_relu(M=16, N=16, K=16, block_tile=(16, 16), outer_tile_k=16,
                                test_name="test_rocm_matmul_relu_vectorized",
                                vectorize=True, cache=(True, True, False), tensorize=False)

    def test_rocm_matmul_relu_vectorized_cache(self) -> None:
        self._matmul_relu(M=16, N=16, K=16, block_tile=(16, 16), outer_tile_k=16,
                                test_name="test_rocm_matmul_relu_vectorized_cache",
                                vectorize=True, cache=(True, True, True), tensorize=False)

    def test_rocm_matmul_relu_vectorized_tensorize(self) -> None:
        self._matmul_relu(M=16, N=16, K=16, block_tile=(16, 16), outer_tile_k=16,
                                test_name="test_rocm_matmul_relu_vectorized_tensorize",
                                vectorize=True, cache=(True, True, False), tensorize=True, prologue_op=_MMAFragmentOp.NONE)

    def test_rocm_matmul_relu_vectorized_tensorize_cache(self) -> None:
        self._matmul_relu(M=128, N=128, K=16, block_tile=(16, 16), outer_tile_k=16, epilogue_op=_MMAFragmentOp.ReLU_NoConditional,
                                test_name="test_rocm_matmul_relu_vectorized_tensorize_cache",
                                vectorize=True, cache=(True, True, True), tensorize=True)

    def test_rocm_matmul_relu_vectorized_tensorize_cache_thread_coarsening(self) -> None:
        self._matmul_relu(M=128, N=128, K=16, block_tile=(64, 64), outer_tile_k=16, thread_coarsening_tile=(2, 2),
                                test_name="test_rocm_matmul_relu_vectorized_tensorize_cache_thread_coarsening",
                                vectorize=True, cache=(True, True, True), tensorize=True)

    def _test_cache_memory_order_helper(self, a_layout, a_cache_layout, double_buffer, vectorize, tensorize, element_type = ScalarType.float32,
                                        mma_shape = _MMAShape.M16xN16xK4_B1, num_total_passes = 4, model = Target.Model.AMD_MI100) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 512
        N = 512
        K = 512

        # Pick the A and B tile sizes to be smaller than the number of threads per block
        outer_tile_x = 64
        outer_tile_y = 64
        outer_tile_k = 64

        output_type = element_type if tensorize else ScalarType.float32

        A = Array(role=Role.INPUT, element_type=element_type, shape=(M, K), layout=a_layout)
        B = Array(role=Role.INPUT, element_type=element_type, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(role=Role.INPUT_OUTPUT, element_type=output_type, shape=(M, N), layout=Array.Layout.FIRST_MAJOR)

        nest = _matmul_nest(M, N, K, A, B, C)
        i, j, k = nest.get_indices()
        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y,
            k: outer_tile_k
        })

        target = Target(model)
        tensor_splits = target.tensor_core_info.compute_tensor_splits(mma_shape, num_total_passes=num_total_passes)

        iii, jjj, kkk = schedule.tile({
            ii: tensor_splits[0],
            jj: tensor_splits[1],
            kk: tensor_splits[2]
        })
        if tensorize:
            outer_nest_order = (i, j, k, ii, jj, kk)
            plan, tensorization_indices = schedule._create_tensorizable_plan(target, block_indices=(i, j), warp_indices=(ii, jj), tensor_indices=(iii, jjj, kkk), outer_nest_order=outer_nest_order)
            plan.tensorize(indices=tensorization_indices, mma_shape=mma_shape, num_total_passes=num_total_passes)
        else:
            schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk)
            plan = schedule.create_plan(target)
            plan.bind(
                mapping={
                    i: target.GridUnit.BLOCK_Y,
                    j: target.GridUnit.BLOCK_X,
                    iii: target.GridUnit.THREAD_Y,
                    jjj: target.GridUnit.THREAD_X
                }
            )

        plan.cache(A,
            index=ii,
            double_buffer=double_buffer,
            vectorize=vectorize,
            location=target.MemorySpace.SHARED,
            layout=a_cache_layout
        )

        layout_str_map = {
            Array.Layout.FIRST_MAJOR : "F",
            Array.Layout.LAST_MAJOR : "L"
        }
        bool_str_map = {
            True : "T",
            False : "F"
        }
        platform = "rocm" if model == Target.Model.AMD_MI100 else "cuda"
        name_parts = [
            f"test_{platform}_memory_order_cache_tensorized",
            layout_str_map[a_layout],
            layout_str_map[a_cache_layout],
            bool_str_map[double_buffer],
            bool_str_map[vectorize],
            bool_str_map[tensorize]
        ]
        test_name = "_".join(name_parts)
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=(platform == "rocm" and ROCM_AVAILABLE) or (platform == "cuda" and CUDA_AVAILABLE),
            tolerance=1e-5 if element_type == ScalarType.float32 else 1e-2,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.DEFAULT | Package.Format.MLIR
        )

    # FIRST-FIRST (ROCM)
    def test_rocm_memory_order_cache_tensorized_F_F_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, True)

    def test_rocm_memory_order_cache_tensorized_F_F_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, False)

    def test_rocm_memory_order_cache_tensorized_F_F_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, True)

    def test_rocm_memory_order_cache_tensorized_F_F_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, True)

    def test_rocm_memory_order_cache_tensorized_F_F_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, False)

    def test_rocm_memory_order_cache_tensorized_F_F_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, True)

    def test_rocm_memory_order_cache_tensorized_F_F_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, False)

    def test_rocm_memory_order_cache_tensorized_F_F_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, False)

    # FIRST-LAST (ROCM)
    def test_rocm_memory_order_cache_tensorized_F_L_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, True, True)

    def test_rocm_memory_order_cache_tensorized_F_L_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, True, False)

    def test_rocm_memory_order_cache_tensorized_F_L_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, False, True)

    def test_rocm_memory_order_cache_tensorized_F_L_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, True, True)

    def test_rocm_memory_order_cache_tensorized_F_L_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, False, False)

    def test_rocm_memory_order_cache_tensorized_F_L_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, False, True)

    def test_rocm_memory_order_cache_tensorized_F_L_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, True, False)

    def test_rocm_memory_order_cache_tensorized_F_L_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, False, False)

    # LAST-FIRST (ROCM)
    def test_rocm_memory_order_cache_tensorized_L_F_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, True)

    def test_rocm_memory_order_cache_tensorized_L_F_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, False)

    def test_rocm_memory_order_cache_tensorized_L_F_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, True)

    def test_rocm_memory_order_cache_tensorized_L_F_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, True)

    def test_rocm_memory_order_cache_tensorized_L_F_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, False)

    def test_rocm_memory_order_cache_tensorized_L_F_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, True)

    def test_rocm_memory_order_cache_tensorized_L_F_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, False)

    def test_rocm_memory_order_cache_tensorized_L_F_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, False)

    # LAST-LAST (ROCM)
    def test_rocm_memory_order_cache_tensorized_L_L_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, True, True)

    def test_rocm_memory_order_cache_tensorized_L_L_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, True, False)

    def test_rocm_memory_order_cache_tensorized_L_L_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, False, True)

    def test_rocm_memory_order_cache_tensorized_L_L_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, True, True)

    def test_rocm_memory_order_cache_tensorized_L_L_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, False, False)

    def test_rocm_memory_order_cache_tensorized_L_L_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, False, True)

    def test_rocm_memory_order_cache_tensorized_L_L_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, True, False)

    def test_rocm_memory_order_cache_tensorized_L_L_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, False, False)

    # FIRST-FIRST (CUDA)
    def test_cuda_memory_order_cache_tensorized_F_F_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_F_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    # FIRST-LAST (CUDA)
    def test_cuda_memory_order_cache_tensorized_F_L_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, True, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_F_L_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR, False, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    # LAST-FIRST (CUDA)
    def test_cuda_memory_order_cache_tensorized_L_F_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, True, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_F_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR, False, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    # LAST-LAST (CUDA)
    def test_cuda_memory_order_cache_tensorized_L_L_T_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_T_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_T_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_F_T_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, True, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_T_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, True, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_F_F_T(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, False, True, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_F_T_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, True, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    def test_cuda_memory_order_cache_tensorized_L_L_F_F_F(self) -> None:
        self._test_cache_memory_order_helper(Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, False, False, False, ScalarType.float16, _MMAShape.M16xN16xK16_B1, 1, Target.Model.NVIDIA_RTX_A6000)

    @unittest.skip("This test doesn't need to run in automated runs, only meant for debugging.")
    def test_benchmark(self) -> None:
        from accera import Package, Target
        from accera_gemm import benchmark_kernel

        target = Target(Target.Model.NVIDIA_RTX_A6000)
        test_name = "test_benchmark"

        plan, A, B, C = benchmark_kernel(target, 768, 576, 1024, False, False, "h", 16, 64, 256, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR,
                                         True, _CacheStrategy.BLOCKED, _CacheStrategy.STRIPED, _MMAShape.M16xN16xK16_B1, False, True, True, 1, 1, _MMASchedulingPolicy.PASS_ORDER)
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=CUDA_AVAILABLE,
            tolerance=1e-2,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR_VERBOSE | Package.Format.DEFAULT # Remove MLIR and it will break correctness
        )

if __name__ == '__main__':
    unittest.main(verbosity=10)
