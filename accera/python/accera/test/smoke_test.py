#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import sys
import unittest
import logging
import os
import pathlib
import shutil
import numpy as np
from enum import Enum
from typing import Callable, List

try:
    import cuda, pynvrtc
except:
    CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = True

DEV_MODE = False
if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    DEV_MODE = True
    sys.path.insert(1, os.getcwd())

from accera import Package, ScalarType, Nest, Array, Scalar, fuse, create_parameters
from accera.samples import MatrixMultiplication
from accera.test import verifiers

TEST_PACKAGE_DIR = "test_acccgen"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# TODO: Remove all @expectedFailure decorators as implementation converges with spec


class FailedReason(Enum):
    NOT_IN_CORE = "Not yet implemented (core)"
    NOT_IN_PY = "Not yet implemented (python)"
    UNKNOWN = "Unknown failure"
    BUG = "Bug"
    INVALID = "Invalid"


def expectedFailure(reason: FailedReason, msg: str) -> Callable:
    "Extends the unittest.expectedFailure decorator to print failure details"

    def _decorator(func):

        @unittest.expectedFailure
        def _wrapper(x):
            print(f"\n{reason.value}: {msg}")
            try:
                return func(x)
            except Exception as e:
                print(f"\t{e}\n")
                raise (e)

        return _wrapper

    return _decorator


class SmokeTest(unittest.TestCase):
    PACKAGE_FORMAT = Package.Format.MLIR_DYNAMIC if DEV_MODE else Package.Format.HAT_DYNAMIC
    PACKAGE_MODE = Package.Mode.RELEASE

    def test_full_fusion_trivial(self) -> None:
        A = Array(role=Array.Role.INPUT, shape=(16, 16))
        B = Array(role=Array.Role.INPUT, shape=(16, 16))
        C = Array(role=Array.Role.INPUT_OUTPUT, shape=(16, 16))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 16))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1
        nest1 = Nest(shape=(16, 16))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()
        schedule = fuse(schedule0, schedule1)
        f, i, j = schedule.get_indices()
        plan = schedule.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_full_fusion_trivial"
        package = Package()
        package.add(plan, args=(A, B, C), base_name="test_full_fusion_trivial")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_full_fusion_reordered(self) -> None:
        A = Array(role=Array.Role.INPUT, shape=(16, 16))
        B = Array(role=Array.Role.INPUT, shape=(16, 16))
        C = Array(role=Array.Role.INPUT_OUTPUT, shape=(16, 16))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 16))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1
        nest1 = Nest(shape=(16, 16))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()
        schedule = fuse(schedule0, schedule1)
        f, i, j = schedule.get_indices()
        schedule.reorder(i, j, f)
        plan = schedule.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_full_fusion_reordered"
        package = Package()
        package.add(plan, args=(A, B, C), base_name="test_full_fusion_reordered")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_partial_fusion_trivial(self) -> None:
        A = Array(role=Array.Role.INPUT, shape=(16, 11))
        B = Array(role=Array.Role.INPUT, shape=(11, 10))
        C = Array(role=Array.Role.INPUT_OUTPUT, shape=(16, 10))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 10, 11))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1
        nest1 = Nest(shape=(16, 10))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] = C[i1, j1] * Scalar(0.2)

        schedule1 = nest1.create_schedule()

        schedule = fuse((schedule0, schedule1), partial=2)
        f, i, j, k = schedule.get_indices()
        plan = schedule.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_partial_fusion_trivial"
        package = Package()
        package.add(plan, args=(A, B, C), base_name="test_partial_fusion_trivial")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_partial_fusion_reordered(self) -> None:
        A = Array(role=Array.Role.INPUT, shape=(16, 11))
        B = Array(role=Array.Role.INPUT, shape=(11, 10))
        C = Array(role=Array.Role.INPUT_OUTPUT, shape=(16, 10))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 10, 11))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1
        nest1 = Nest(shape=(16, 10))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] = C[i1, j1] * Scalar(0.2)

        schedule1 = nest1.create_schedule()

        schedule = fuse((schedule0, schedule1), partial=2)
        f, i, j, k = schedule.get_indices()
        schedule.reorder(i, j, f, k)
        plan = schedule.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_partial_fusion_reordered"
        package = Package()
        package.add(plan, args=(A, B, C), base_name="test_partial_fusion_reordered")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_partial_fusion_matmul3_naive(self) -> None:
        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(16, 11))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(11, 10))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(16, 10))
        D = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(10, 7))
        E = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(16, 7))

        # Create nest0 and schedule0 for C = A @ B
        nest0 = Nest(shape=(16, 10, 11))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 E += C @ D
        nest1 = Nest(shape=(16, 7, 10))
        i1, j1, k1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            E[i1, j1] += C[i1, k1] * D[k1, j1]

        schedule1 = nest1.create_schedule()
        schedule1.reorder(i1, k1, j1)

        schedule = fuse((schedule0, schedule1), partial=2)

        plan = schedule.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_partial_fusion_matmul3_naive"
        package = Package()
        package.add(plan, args=(A, B, C, D, E), base_name="test_partial_fusion_matmul3_naive")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_partial_fusion_matmul3_fancy(self) -> None:
        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(16, 11))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(11, 10))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(16, 10))
        D = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(10, 7))
        E = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(16, 7))

        # Create nest0 and schedule0 for C = A @ B
        nest0 = Nest(shape=(16, 10, 11))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 E += C @ D
        nest1 = Nest(shape=(16, 7, 10))
        i1, j1, k1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            E[i1, j1] += C[i1, k1] * D[k1, j1]

        schedule1 = nest1.create_schedule()
        schedule1.reorder(i1, k1, j1)

        schedule = fuse((schedule0, schedule1), partial=2)
        f, i, j, k, l = schedule.get_indices()
        schedule.reorder(i, j, f, k, l)

        ii, jj = schedule.tile((i, j), (4, 4))
        schedule.reorder(i, j, f, ii, jj, k, l)

        plan = schedule.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_partial_fusion_matmul3_fancy"
        package = Package()
        package.add(plan, args=(A, B, C, D, E), base_name="test_partial_fusion_matmul3_fancy")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_multischedule_fusion1(self) -> None:
        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(4, 8))
        B = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(4, ))
        accum = Array(role=Array.Role.TEMP, element_type=ScalarType.float32, shape=(1, ))

        # for i in range(4):
        #     accum[0]  = 0.0
        #     B[i] *= 2.1
        #     for j in range(8):
        #         accum[0] += A[i, j]
        #     B[i] += accum[0] * 1.2

        nest0 = Nest(shape=(4, ))
        nest1 = Nest(shape=(4, 8))
        nest2 = Nest(shape=(4, ))

        i0 = nest0.get_indices()
        i1, j1 = nest1.get_indices()
        i2 = nest2.get_indices()

        @nest0.iteration_logic
        def _():
            accum[0] = 0.0
            B[i0] *= 2.1

        @nest1.iteration_logic
        def _():
            accum[0] += A[i1, j1]

        @nest2.iteration_logic
        def _():
            B[i2] += accum[0] * 1.2

        fused = fuse((n.create_schedule() for n in [nest0, nest1, nest2]), partial=1)
        f, i, j = fused.get_indices()
        fused.reorder(i, f, j)

        plan = fused.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_multischedule_fusion1"
        package = Package()
        package.add(plan, args=(A, B), base_name="test_multischedule_fusion1")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_multischedule_fusion2(self) -> None:
        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(4, 8, 12))
        B = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(4, 8))
        accum = Array(role=Array.Role.TEMP, element_type=ScalarType.float32, shape=(1, ))

        # Goal:
        # for i in range(4):
        #     accum[0] = B[i, 0] * A[i, 0, 0]
        #     for j in range(8):
        #         for k in range(12):
        #             accum[0] += A[i, j, k] * 0.2
        #         B[i, j] *= accum[0]

        nest0 = Nest(shape=(4, ))
        nest1 = Nest(shape=(4, 8, 12))
        nest2 = Nest(shape=(4, 8))

        i0 = nest0.get_indices()
        i1, j1, k1 = nest1.get_indices()
        i2, j2 = nest2.get_indices()

        @nest0.iteration_logic
        def _():
            accum[0] = B[i0, 0] * A[i0, 0, 0]

        @nest1.iteration_logic
        def _():
            accum[0] += A[i1, j1, k1] * 0.2

        @nest2.iteration_logic
        def _():
            B[i2, j2] *= accum[0]

        fused0 = fuse((nest1.create_schedule(), nest2.create_schedule()), partial=2)
        ff0, if0, jf0, kf0 = fused0.get_indices()

        # equivalent:
        # for ff0 in range(2):
        #     if ff0 == 0:
        #         for if0 in range(4):
        #             for jf0 in range(8):
        #                 for kf0 in range(12):
        #                         accum[0] += A[if0, jf0, kf0] * 0.2
        #     if ff0 == 1:
        #         for if0 in range(4):
        #             for jf0 in range(8):
        #                 B[if0, jf0] *= accum[0]

        fused0.reorder(if0, jf0, ff0, kf0)

        # equivalent:
        # for if0 in range(4):
        #     for jf0 in range(8):
        #         for ff0 in range(2):
        #             if ff0 == 0:
        #                 for kf0 in range(12):
        #                         accum[0] += A[if0, jf0, kf0] * 0.2
        #             if ff0 == 1:
        #                 B[if0, jf0] *= accum[0]

        fused1 = fuse((nest0.create_schedule(), fused0), partial=1)
        ff1, if1, jf1, ff0f1, kf1 = fused1.get_indices()
        # equivalent:
        # for ff1 in range(2):
        #     if ff1 == 0:
        #         for if1 in range(4):
        #             accum[0] = B[if1, 0] * A[if1, 0, 0]
        #     if ff1 == 1:
        #         for if1 in range(4):
        #             for jf1 in range(8):
        #                 for ff0f1 in range(2):
        #                     if ff0f1 == 0:
        #                         for kf0 in range(12):
        #                             accum[0] += A[if0, jf0, kf0] * 0.2
        #                     if ff0f1 == 1:
        #                         B[if1, jf1] *= accum[0]

        fused1.reorder(if1, ff1, jf1, ff0f1, kf1)
        # equivalent:
        # for if1 in range(4):
        #     for ff1 in range(2):
        #         if ff1 == 0:
        #             accum[0] = B[if1, 0] * A[if1, 0, 0]
        #         if ff1 == 1:
        #             for jf1 in range(8):
        #                 for ff0f1 in range(2):
        #                     if ff0f1 == 0:
        #                         for kf1 in range(12):
        #                             accum[0] += A[if0, jf0, kf0] * 0.2
        #                     if ff0f1 == 1:
        #                         B[if1, jf1] *= accum[0]

        plan = fused1.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_multischedule_fusion2"
        package = Package()
        package.add(plan, args=(A, B), base_name="test_multischedule_fusion2")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_naive_matmul(self) -> None:
        # Define our matrix sizes
        M = 128
        N = 256
        K = 256

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()
        kk = sched.split(k, 4)

        plan = sched.create_plan()
        plan.unroll(kk)

        # Create a package and add our function definition to it
        package_name = "test_naive_matmul"
        package = Package()
        package.add(plan, args=(A, B, C), base_name="hello_matmul_py")

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_mlas_matmul(self) -> None:
        from itertools import combinations_with_replacement
        from accera.samples.MatrixMultiplication import MLAS

        domains = combinations_with_replacement([1, 31, 63, 127], 3)

        package = Package()

        opts = MatrixMultiplication.Options(UseAlphaScalingFusion=False)

        for domain in domains:
            M, N, K = domain
            A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
            B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
            C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
            package.add(*MLAS(A, B, C, alpha=0.2, zero_C=True, opts=opts), base_name=f"mlas_py_{M}_{N}_{K}")

        package_name = "test_mlas_matmul"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_emittime_cache_mlas_matmul(self) -> None:
        from accera.samples.OfflineCacheMatrixMultiplication import EmitTimeCacheMLAS

        package = Package()
        M, N, K = [31, 63, 127]
        B_data = np.array([float(x) for x in range(K * N)]).reshape(K, N).astype(np.float32)

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.CONST, element_type=ScalarType.float32, data=B_data)
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        function = package.add(*EmitTimeCacheMLAS(A, B, C), base_name=f"mlas_py_{M}_{N}_{K}")

        A_test = np.ones(A.shape, dtype=np.float32) * 3.14
        B_test = np.ones(B.shape, dtype=np.float32) * 0.42
        C_test = np.ones(C.shape, dtype=np.float32) * -1.27
        C_ref = C_test + A_test @ B_test

        package_name = "emittime_cache_mlas"
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, output_dir=output_dir, mode=self.PACKAGE_MODE, format=self.PACKAGE_FORMAT)

            # check build and correctness
            v.check_correctness(function.name, before=(A_test, B_test, C_test), after=(A_test, B_test, C_ref))

    def test_runtime_init_cache_mlas_matmul(self) -> None:
        from accera.samples.OfflineCacheMatrixMultiplication import RuntimeInitCacheMLAS

        package = Package()

        M, N, K = [31, 63, 127]
        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        package.add(
            *RuntimeInitCacheMLAS(A, B, C, pack_fn_name="pack", packed_buffer_size_fn_name="packed_size"),
            base_name=f"mlas_py_{M}_{N}_{K}"
        )

        package_name = "runtime_init_cache_mlas"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_const_array_shared_across_functions(self) -> None:
        # In this scenario we use a single CONST data matrix in two Accera functions,
        # the first will perform matmul and the second will perform elementwise add

        package = Package()

        M = 256
        N = 256
        K = 256

        const_matrix_shape = (K, N)
        data = np.random.random(const_matrix_shape).astype(np.float32)
        const_matrix = Array(role=Array.Role.CONST, element_type=ScalarType.float32, data=data)

        # Matmul function

        matmul_input_matrix = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))

        matmul_output_matrix = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        matmul_nest = Nest(shape=(M, N, K))
        i, j, k = matmul_nest.get_indices()

        @matmul_nest.iteration_logic
        def _():
            matmul_output_matrix[i, j] += matmul_input_matrix[i, k] * const_matrix[k, j]

        matmul_schedule = matmul_nest.create_schedule()
        matmul_plan = matmul_schedule.create_plan()

        package.add(matmul_plan, args=(matmul_input_matrix, matmul_output_matrix), base_name="matmul_fn")

        # Elementwise add function

        ew_add_input_matrix = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))

        ew_add_output_matrix = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(K, N))

        ew_add_nest = Nest(shape=(K, N))
        x, y = ew_add_nest.get_indices()

        @ew_add_nest.iteration_logic
        def _():
            ew_add_output_matrix[x, y] = ew_add_input_matrix[x, y] + const_matrix[x, y]

        ew_add_schedule = ew_add_nest.create_schedule()
        ew_add_plan = ew_add_schedule.create_plan()

        package.add(ew_add_plan, args=(ew_add_input_matrix, ew_add_output_matrix), base_name="ew_add_fn")

        package_name = "const_matrix_shared_between_functions"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def test_gpu_matmul(self) -> None:
        import math
        from accera import Target
        from accera._lang_python._lang import _If, as_index

        def get_clamped_block_dimensions(M, N, base_block_dim_M=16, base_block_dim_N=16):
            return min(M, base_block_dim_M), min(N, base_block_dim_N)

        def compute_grid_dimensions(M, N, blockdim_M, blockdim_N):
            return math.ceil(M / blockdim_M), math.ceil(N / blockdim_N)

        def round_up(number, multiple):
            return math.ceil(number / multiple) * multiple

        M = 128
        N = 256
        K = 256

        block_x, block_y = get_clamped_block_dimensions(M, N)
        grid_x, grid_y = compute_grid_dimensions(M, N, block_x, block_y)

        self.assertEqual(16, block_x)
        self.assertEqual(16, block_y)
        self.assertEqual(8, grid_x)
        self.assertEqual(16, grid_y)

        M_ = round_up(M, block_x)
        N_ = round_up(N, block_y)

        self.assertEqual(M_, M)
        self.assertEqual(N_, N)

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M_, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N_))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M_, N_))

        nest = Nest(shape=(M_, N_, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():

            def if_block():
                C[i, j] += A[i, k] * B[k, j]

            _If(i < as_index(M) and j < as_index(N), if_block)    # TODO: wrap implicitly

        schedule = nest.create_schedule()
        ii = schedule.split(i, block_x)
        jj = schedule.split(j, block_y)

        schedule.reorder(i, j, ii, jj, k)

        target = Target(category=Target.Category.GPU, runtime=Target.Runtime.VULKAN)
        plan = schedule.create_plan(target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )

        package = Package()
        package.add(plan, args=(A, B, C), base_name="hello_matmul_gpu")

        # BUGBUG: More than 1 Nest function cannot be added to the same GPU package (function names are now unique)
        #   17_SPIRVUpdateVCE.mlir:134:2: error: should only contain one 'spv.module' op
        #   spv.module @__spv__NestFunction_0_module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        # package.add(plan, args=(A, B, C), base_name="hello_matmul_gpu")
        format = self.PACKAGE_FORMAT if "VULKAN_SDK" in os.environ else Package.Format.HAT_STATIC
        with verifiers.VerifyPackage(self, "hello_matmul_gpu", TEST_PACKAGE_DIR):
            package.build(name="hello_matmul_gpu", format=format, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    @expectedFailure(FailedReason.NOT_IN_CORE, "function that contains multiple nests")
    def test_int8_matmul(self) -> None:
        from accera import _cast, _unsigned_cast

        # Define our matrix sizes
        M = 128
        N = 256
        K = 256

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.int8, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.int8, shape=(K, N))
        zero_point_A = Array(role=Array.Role.INPUT, element_type=ScalarType.int32, shape=(1, ))
        zero_point_B = Array(role=Array.Role.INPUT, element_type=ScalarType.int32, shape=(1, ))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, N))
        col_sums = Array(role=Array.Role.TEMP, element_type=ScalarType.int32, shape=(N, ))
        row_sums = Array(role=Array.Role.TEMP, element_type=ScalarType.int32, shape=(M, ))

        def get_compute_col_sums_schedule():
            compute_col_sums_nest = Nest(shape=(K, N))
            k, j = compute_col_sums_nest.get_indices()

            @compute_col_sums_nest.iteration_logic
            def _():
                b = _unsigned_cast(B[k, j], ScalarType.int32)
                col_sums[j] += b

            return compute_col_sums_nest.create_schedule()

        def get_scale_col_sums_schedule():
            scale_col_sums_nest = Nest(shape=(N, ))
            j = scale_col_sums_nest.get_indices()

            @scale_col_sums_nest.iteration_logic
            def _():
                col_sums[j] *= (-zero_point_A)

            return scale_col_sums_nest.create_schedule()

        def get_compute_row_sums_schedule():
            compute_row_sums_nest = Nest(shape=(M, K))
            i, k = compute_row_sums_nest.get_indices()

            @compute_row_sums_nest.iteration_logic
            def _():
                a = _cast(A[i, k], ScalarType.int32)
                row_sums[i] += a

            return compute_row_sums_nest.create_schedule()

        def get_offset_row_sums_schedule():
            offset_row_sums_nest = Nest(shape=(M, ))
            i = offset_row_sums_nest.get_indices()

            @offset_row_sums_nest.iteration_logic
            def _():
                row_sums[i] -= zero_point_A

            return offset_row_sums_nest.create_schedule()

        def get_scale_row_sums_schedule():
            scale_row_sums_nest = Nest(shape=(M, ))
            i = scale_row_sums_nest.get_indices()

            @scale_row_sums_nest.iteration_logic
            def _():
                row_sums[i] *= (-zero_point_B)

            return scale_row_sums_nest.create_schedule()

        def get_init_c_array_schedule():
            init_c_array_nest = Nest(shape=(M, N))
            i, j = init_c_array_nest.get_indices()

            @init_c_array_nest.iteration_logic
            def _():
                C[i, j] = row_sums[i] + col_sums[j]

            return init_c_array_nest.create_schedule()

        def get_matmul_schedule():
            matmul_nest = Nest(shape=(M, N, K))
            i, j, k = matmul_nest.get_indices()

            @matmul_nest.iteration_logic
            def _():
                a = _cast(A[i, k], ScalarType.int32)
                b = _unsigned_cast(B[k, j], ScalarType.int32)
                C[i, j] += a * b

            return matmul_nest.create_schedule()

        # Create and computeColSums
        compute_col_sums_schedule = get_compute_col_sums_schedule()
        scale_col_sums_schedule = get_scale_col_sums_schedule()

        # Create and compute RowSums
        compute_row_sums_schedule = get_compute_row_sums_schedule()
        offset_row_sums_schedule = get_offset_row_sums_schedule()
        scale_row_sums_schedule = get_scale_row_sums_schedule()

        # Init C
        init_c_array_schedule = get_init_c_array_schedule()

        # Do the matrix multiplication
        matmul_schedule = get_matmul_schedule()

        subschedules = (
            compute_col_sums_schedule, scale_col_sums_schedule, compute_row_sums_schedule, offset_row_sums_schedule,
            scale_row_sums_schedule, init_c_array_schedule, matmul_schedule
        )

        # Create a package and add our function to it
        package_name = "int8_matmul"
        package = Package()
        #package.add(fused_schedule, args=(A, B, C, zero_point_A, zero_point_B), base_name=package_name)
        subfunctions = [
            package.add(
                sched, args=(A, B, C, zero_point_A, zero_point_B), base_name=package_name + "_sched_" + str(index)
            ) for index, sched in enumerate(subschedules)
        ]

        # Now create a main function that invokes each of the subfunctions
        def main():
            for function in subfunctions:
                function(A, B, C, zero_point_A, zero_point_B)

        main_function = package.add(main, args=(A, B, C, zero_point_A, zero_point_B), base_name=package_name)

        # Build the HAT package
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=TEST_PACKAGE_DIR)

    def _verify_three_matrix_multiplication_function(
        self, function: "accera.Function", package: Package, package_name: str
    ) -> None:
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            A_test, B_test, D_test, E_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)
            E_ref = E_test + A_test @ B_test @ D_test

            v.check_correctness(
                function.name, before=(A_test, B_test, D_test, E_test), after=(A_test, B_test, D_test, E_ref)
            )

    def test_three_matrix_multiplication_case_study_part1(self) -> None:
        import accera as acc
        A = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(32, 256))
        C = acc.Array(role=acc.Array.Role.TEMP, shape=(256, 256))
        D = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
        E = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(256, 32))

        # Create nest0 and schedule0 for C += A @ B
        nest0 = acc.Nest(shape=(256, 256, 32))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 E += C @ D
        nest1 = acc.Nest(shape=(256, 32, 256))
        i1, j1, k1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            E[i1, j1] += C[i1, k1] * D[k1, j1]

        schedule1 = nest1.create_schedule()

        # redundant operation, included for clarity
        schedule0.reorder(i0, j0, k0)
        schedule1.reorder(i1, k1, j1)
        schedule = acc.fuse((schedule0, schedule1), partial=2)
        f, i, j, k0, j1 = schedule.get_indices()

        # TODO: support parameters
        # m, n = acc.create_parameters(2)
        # ii, jj = schedule.tile((i,j), (m,n))
        ii, jj = schedule.tile((i, j), (16, 32))
        schedule.reorder(i, j, f, jj, k0, j1, ii)

        plan = schedule.create_plan()

        # BUGBUG: caching + fusion requires proper access_indices introspection
        plan.cache(A, index=f, layout=acc.Array.Layout.LAST_MAJOR)
        plan.cache(B, index=f)
        plan.cache(C, index=f, layout=acc.Array.Layout.LAST_MAJOR)
        plan.cache(D, index=f)
        plan.cache(E, index=f, layout=acc.Array.Layout.LAST_MAJOR)

        plan.vectorize(ii)

        package = Package()
        function = package.add(plan, args=(A, B, D, E), base_name="fused_three_matrix_multiplication")
        self._verify_three_matrix_multiplication_function(
            function, package, "three_matrix_multiplication_case_study_part1"
        )

    def test_three_matrix_multiplication_case_study_part2(self) -> None:
        import accera as acc
        A = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(32, 256))
        C = acc.Array(role=acc.Array.Role.TEMP, shape=(256, 256))
        D = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
        E = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(256, 32))

        # Create nest0 and schedule0 for C += A @ B
        nest0 = acc.Nest(shape=(256, 256, 32))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 E += C @ D
        nest1 = acc.Nest(shape=(256, 32, 256))
        i1, j1, k1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            E[i1, j1] += C[i1, k1] * D[k1, j1]

        schedule1 = nest1.create_schedule()

        # TODO: support parameters
        # n = acc.create_parameters(1)
        # jj0 = schedule0.split(j0, n)
        # kk1 = schedule1.split(k1, n)
        jj0 = schedule0.split(j0, 8)
        kk1 = schedule1.split(k1, 8)

        # redundant operation, included for clarity
        schedule0.reorder(i0, j0, jj0, k0)
        schedule1.reorder(i1, k1, j1, kk1)
        schedule = acc.fuse((schedule0, schedule1), partial=2)
        f, i, j, jj0, k0, j1, kk1 = schedule.get_indices()

        # TODO: support parameters
        # m, s = acc.create_parameters(2)
        # ii, jj1 = schedule.tile((i, j1), (m, s))
        ii, jj1 = schedule.tile((i, j1), (64, 8))
        schedule.reorder(i, j, f, j1, ii, k0, jj0, kk1, jj1)

        plan = schedule.create_plan()

        # BUGBUG: caching + fusion requires proper access_indices introspection
        plan.cache(A, index=f)
        plan.cache(B, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
        plan.cache(C, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
        plan.cache(D, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
        plan.cache(E, index=f, layout=acc.Array.Layout.FIRST_MAJOR)

        plan.vectorize(jj0)    # unfused index
        plan.vectorize(jj1)    # child of an unfused index

        package = Package()
        function = package.add(plan, args=(A, B, D, E), base_name="fused_three_matrix_multiplication_part2")
        self._verify_three_matrix_multiplication_function(
            function, package, "three_matrix_multiplication_case_study_part2"
        )

    def test_three_matrix_multiplication_case_study_part3(self) -> None:
        import accera as acc
        A = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(32, 256))
        C = acc.Array(role=acc.Array.Role.TEMP, shape=(256, 256))
        D = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 32))
        E = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(256, 32))

        # Create nest0 and schedule0 for C += A @ B
        nest0 = acc.Nest(shape=(256, 256, 32))
        i0, j0, k0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, k0] * B[k0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 E += C @ D
        nest1 = acc.Nest(shape=(256, 32, 256))
        i1, j1, k1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            E[i1, j1] += C[i1, k1] * D[k1, j1]

        schedule1 = nest1.create_schedule()

        # TODO: support parameters
        # n = acc.create_parameters(1)
        # jj0 = schedule0.split(j0, n)
        # kk1 = schedule1.split(k1, n)
        jj0 = schedule0.split(j0, 8)
        kk1 = schedule1.split(k1, 8)

        # redundant operation, included for clarity
        schedule0.reorder(i0, j0, jj0, k0)
        schedule1.reorder(i1, k1, j1, kk1)
        schedule = acc.fuse((schedule0, schedule1), partial=2)

        f, i, j, jj0, k0, j1, kk1 = schedule.get_indices()

        # TODO: support parameters
        # m, t = acc.create_parameters(2)
        # ii, kk0 = schedule.tile((i, k0), (m, t))
        ii, kk0 = schedule.tile((i, k0), (64, 8))
        schedule.reorder(i, j, f, ii, jj0, k0, kk0, j1, kk1)

        plan = schedule.create_plan()

        # BUGBUG: caching + fusion requires proper access_indices introspection
        plan.cache(A, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
        plan.cache(B, index=f, layout=acc.Array.Layout.LAST_MAJOR)
        plan.cache(C, index=f, layout=acc.Array.Layout.FIRST_MAJOR)
        plan.cache(D, index=f, layout=acc.Array.Layout.LAST_MAJOR)
        plan.cache(E, index=f)

        # plan.vectorize(kk0)  # child of an unfused index, BUGBUG: failing correctness check (E is being stomped on with the same value due to broadcast)
        # plan.vectorize(kk1)  # unfused index, BUGBUG: failing correctness check (E is being stomped on with the same value due to broadcast)

        package = Package()
        function = package.add(plan, args=(A, B, D, E), base_name="fused_three_matrix_multiplication_part3")

        self._verify_three_matrix_multiplication_function(
            function, package, "three_matrix_multiplication_case_study_part3"
        )

    def test_naive_conv2d_skew(self) -> None:
        N = 224    # input
        Fi = 3    # input filters
        Fo = 5    # output filters
        K = 2    # kernel size
        M = N - K + 1    # output (no padding, stride 1)

        Input = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, N, Fi))
        Kernel = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, K, Fi, Fo))
        Output = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, M, Fo))

        nest = Nest(shape=(Fo, M, M, Fi, K, K))
        out_ch, out_r, out_c, in_ch, k_r, k_c = nest.get_indices()

        @nest.iteration_logic
        def _():
            Output[out_r, out_c, out_ch] += Input[out_r + k_r, out_c + k_c, in_ch] * Kernel[k_r, k_c, in_ch, out_ch]

        schedule = nest.create_schedule()

        # skew output row dim w.r.t. to kernel row dim (note: stride must be 1, padding not yet supported)
        schedule.skew(out_r, k_r)
        # skew output col dim w.r.t. to kernel col dim (note: stride must be 1, padding not yet supported)
        schedule.skew(out_c, k_c, unroll_loops_smaller_than=3)
        schedule.reorder(out_ch, in_ch, out_r, out_c, k_r, k_c)

        package = Package()
        function = package.add(schedule, args=(Input, Kernel, Output), base_name="naive_conv2d_skew")

        package_name = "test_naive_conv2d_skew"
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            # correctness check
            # cf. accera-community-samples/expr/Convolution/helper.py
            def naive_convolution_ref(
                input, kernel, output, row_stride=1, column_stride=1, row_padding=0, column_padding=0
            ):
                input_rows, input_columns, input_channels = input.shape
                output_rows, output_columns, out_filters = output.shape
                kernel_rows, kernel_columns, _, _ = kernel.shape
                output_ref = output.copy()
                for out_f in range(out_filters):
                    for out_r in range(output_rows):
                        for out_c in range(output_columns):
                            for in_ch in range(input_channels):
                                for k_r in range(kernel_rows):
                                    for k_c in range(kernel_columns):
                                        in_r = out_r * row_stride + k_r - row_padding
                                        in_c = out_c * column_stride + k_c - column_padding
                                        if in_r >= 0 and in_r < input_rows and in_c >= 0 and in_c < input_columns:
                                            output_ref[out_r, out_c,
                                                       out_f] += input[in_r, in_c, in_ch] * kernel[k_r, k_c, in_ch,
                                                                                                   out_f]
                return output_ref

            Input_test, Kernel_test, Output_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)
            Output_ref = naive_convolution_ref(Input_test, Kernel_test, Output_test)

            v.check_correctness(
                function.name,
                before=(Input_test, Kernel_test, Output_test),
                after=(Input_test, Kernel_test, Output_ref)
            )

    def test_strided_sub_array(self) -> None:
        N = 5
        subArrayNumRows = 2
        subArrayNumCols = 3

        Input = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(N, N))

        # Zero out a sub array of size [2, 3]:
        # xxxxx
        # x000x
        # xxxxx
        # x000x
        # xxxxx

        out_nest = Nest(shape=(subArrayNumRows, subArrayNumCols))
        i, j = out_nest.get_indices()

        @out_nest.iteration_logic
        def _():
            Output = Input.sub_array([1, 1], [subArrayNumRows, subArrayNumCols], [2, 1])
            Output[i, j] = 0.0

        schedule = out_nest.create_schedule()

        package = Package()
        function = package.add(schedule, args=(Input, ), base_name="strided_sub_array")

        package_name = "test_strided_sub_array"
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            # correctness check
            Data = np.random.random([N, N]).astype(np.float32)
            DataStrided = Data.copy()
            DataStrided[1, 1] = 0.0
            DataStrided[1, 2] = 0.0
            DataStrided[1, 3] = 0.0
            DataStrided[3, 1] = 0.0
            DataStrided[3, 2] = 0.0
            DataStrided[3, 3] = 0.0
            v.check_correctness(function.name, before=(Data, ), after=(DataStrided, ))

    def test_padded_nchwc_conv2d_manual_cache(self) -> None:
        input_channels = 64
        base_input_shape = (input_channels, 28, 28)    # CHW order
        buffer_padding = (0, 1, 1)
        conv_padding = (0, 1, 1)
        stride = (2, 2)
        kernel_shape = (3, 3)
        output_filters = 64

        import math
        unpadded_output_rows = math.floor(((base_input_shape[1] + (2 * conv_padding[1]) -
                                            (kernel_shape[0] - 1) - 1) / stride[0]) + 1)
        unpadded_output_columns = math.floor(((base_input_shape[2] + (2 * conv_padding[2]) -
                                               (kernel_shape[1] - 1) - 1) / stride[1]) + 1)
        base_output_shape = (output_filters, unpadded_output_rows, unpadded_output_columns)

        padded_input_shape = [base_input_shape[i] + 2 * buffer_padding[i] for i in range(len(base_input_shape))]
        padded_output_shape = [base_output_shape[i] + 2 * buffer_padding[i] for i in range(len(base_output_shape))]

        channels_per_block = 8
        input_channel_blocks = padded_input_shape[0] // channels_per_block
        output_filter_blocks = padded_output_shape[0] // channels_per_block
        nchwc_padded_input_shape = (
            input_channel_blocks, padded_input_shape[1], padded_input_shape[2], channels_per_block
        )
        nchwc_weights_shape = (kernel_shape[0], kernel_shape[1], input_channels, output_filters)
        nchwc_padded_output_shape = (
            output_filter_blocks, padded_output_shape[1], padded_output_shape[2], channels_per_block
        )

        Input = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=nchwc_padded_input_shape)
        Kernel = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=nchwc_weights_shape)
        Output = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=nchwc_padded_output_shape)

        nest = Nest(
            shape=(
                output_filter_blocks, input_channel_blocks, unpadded_output_rows, kernel_shape[0], kernel_shape[1],
                channels_per_block, unpadded_output_columns, channels_per_block
            )
        )

        out_f, in_ch, out_r, k_r, k_c, in_ch_b, out_c, out_f_b = nest.get_indices()

        row_stride, column_stride = stride
        channel_padding, row_padding, column_padding = conv_padding
        channel_buffer_padding, row_buffer_padding, column_buffer_padding = buffer_padding
        # Define the iteration logic

        @nest.iteration_logic
        def _():
            in_r = out_r * row_stride - row_padding + k_r
            in_c = out_c * column_stride - column_padding + k_c
            Output[out_f, out_r + row_buffer_padding, out_c + column_buffer_padding, out_f_b] += \
                Input[in_ch, in_r + row_buffer_padding, in_c + column_buffer_padding, in_ch_b] * \
                Kernel[k_r, k_c, in_ch * channels_per_block +
                       in_ch_b, out_f * channels_per_block + out_f_b]

        schedule = nest.create_schedule()

        out_c2 = schedule.split(out_c, 3)
        out_f2 = schedule.split(out_f, 4)

        # Apply re-ordering
        schedule.reorder(out_f, in_ch, out_r, out_c, k_r, k_c, in_ch_b, out_f2, out_c2, out_f_b)

        plan = schedule.create_plan()

        # Cache input and output arrays
        plan.cache(Input, index=in_ch_b, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(Output, index=out_f2, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(Kernel, index=in_ch_b, layout=Array.Layout.FIRST_MAJOR)

        # Kernelize the last 4 indices in the kernel loop
        plan.kernelize(unroll_indices=(in_ch_b, out_f2, out_c2), vectorize_indices=out_f_b)

        package = Package()
        function = package.add(plan, args=(Input, Kernel, Output), base_name="nchwc_conv2d_manual_cache")

        package_name = "test_nchwc_conv2d_manual_cache"
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            # correctness check
            def naive_nchwc_convolution_ref(input, kernel, output):
                input_channel_blocks, input_rows, input_columns, input_channels = input.shape
                out_filter_blocks, output_rows, output_columns, out_filters = output.shape
                kernel_rows, kernel_columns, _, _ = kernel.shape
                output_ref = output.copy()
                for out_f_block in range(out_filter_blocks):
                    for out_f in range(out_filters):
                        for out_r in range(output_rows - 2 * row_buffer_padding):
                            for out_c in range(output_columns - 2 * column_buffer_padding):
                                for in_ch_block in range(input_channel_blocks):
                                    for in_ch in range(input_channels):
                                        for k_r in range(kernel_rows):
                                            for k_c in range(kernel_columns):
                                                in_r = out_r * row_stride + k_r - row_padding
                                                in_c = out_c * column_stride + k_c - column_padding
                                                output_ref[out_f_block, out_r + row_buffer_padding, out_c + column_buffer_padding, out_f] += \
                                                    input[in_ch_block, in_r + row_buffer_padding, in_c + column_buffer_padding, in_ch] * \
                                                    kernel[k_r, k_c, in_ch_block * channels_per_block +
                                                        in_ch, out_f_block * channels_per_block + out_f]
                return output_ref

            # unpadded_Input_test, unpadded_Kernel_test, unpadded_Output_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)
            Input_test, Kernel_test, Output_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)
            Output_ref = naive_nchwc_convolution_ref(Input_test, Kernel_test, Output_test)

            v.check_correctness(
                function.name,
                before=(Input_test, Kernel_test, Output_test),
                after=(Input_test, Kernel_test, Output_ref)
            )

    def test_large_cache_vectorized(self) -> None:
        M = 1024
        N = 1024
        S = 1024

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, S))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(S, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, S))

        i, j, k = nest.get_indices()

        # Define the iteration logic
        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii = schedule.split(i, 512)
        jj = schedule.split(j, 512)
        kk = schedule.split(k, 512)

        # Apply re-ordering
        schedule.reorder(i, j, k, ii, jj, kk)

        plan = schedule.create_plan()

        # Cache input and output arrays
        plan.cache(B, index=ii, layout=Array.Layout.FIRST_MAJOR)

        # Vectorize
        plan.vectorize(kk)

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name="test_large_cache_vectorized")

        package_name = "test_large_cache_vectorized"
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir):
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

    def test_cross_compile(self) -> None:
        from accera import Target
        M = 128
        N = 256
        K = 256

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()

        # Split the k loop into blocks of 4
        kk = sched.split(k, 4)

        # Create a plan, specify the target to be PI3
        pi3 = Target(Target.Model.RASPBERRY_PI_3B, category=Target.Category.CPU)
        plan = sched.create_plan(pi3)

        # Then unroll kk
        plan.unroll(kk)

        # Create a package and add a function to the package based on the plan
        package = Package()
        package.add(plan, args=(A, B, C), base_name="hello_matmul_pi3_py")

        # Build the HAT package
        with verifiers.VerifyPackage(self, "hello_matmul_pi3", TEST_PACKAGE_DIR):
            package.build(
                name="hello_matmul_pi3",
                format=Package.Format.HAT_STATIC,
                mode=self.PACKAGE_MODE,
                output_dir=TEST_PACKAGE_DIR,
                platform=Package.Platform.RASPBIAN
            )

    def test_parameter_grid_no_regression(self) -> None:

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(16, 16))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(16, 16))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(16, 16))

        nest = Nest(shape=[16, 16, 16])
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        # Create a parameterized schedule
        schedule = nest.create_schedule()
        ii = schedule.split(i, size=4)

        plan = schedule.create_plan()

        # Create a package and add our grid of functions to it
        package = Package()

        package.add(plan, args=(A, B, C), base_name="alternative_matmul_16_16_16")

        # Build the HAT package
        package.build("hello_matmul_parameter_grid1", output_dir=TEST_PACKAGE_DIR)

        # Check that the package dir exists
        self.assertTrue(os.path.isdir(TEST_PACKAGE_DIR))

    def test_parameter_grid(self) -> None:

        P0, P1, P2, P3, P4, P5 = create_parameters(6)

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(P0, P1))

        nest = Nest(shape=[P0, P1, P2])
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        # Create a parameterized schedule
        schedule = nest.create_schedule()
        ii = schedule.split(i, size=P3)
        schedule.reorder(order=P5)

        plan = schedule.create_plan()
        plan.cache(A, level=P4)

        # Create a package and add our grid of functions to it
        package = Package()

        package.add(
            plan,
            args=(A, B, C),
            parameters={
                P0: 16,
                P1: 16,
                P2: 16,
                P3: 4,
                P4: 2,
                P5: (j, k, i, ii)
            },
            base_name="alternative_matmul_16_16_16"
        )

        # Build the HAT package
        package.build("hello_matmul_parameter_grid", output_dir=TEST_PACKAGE_DIR)

        # Check that the package dir exists
        self.assertTrue(os.path.isdir(TEST_PACKAGE_DIR))

    def _verify_matrix_multiplication_function(
        self, function: "accera.Function", package: Package, package_name: str
    ) -> None:
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            A_test, B_test, C_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)
            C_ref = C_test + A_test @ B_test

            v.check_correctness(function.name, before=(A_test, B_test, C_test), after=(A_test, B_test, C_ref))

    def _verify_convolution_function(
        self, function: "accera.Function", package: Package, package_name: str, buffer_padding: List[int],
        conv_padding: List[int], stride: List[int]
    ) -> None:
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        row_stride, column_stride = stride
        channel_padding, row_padding, column_padding = conv_padding
        channel_buffer_padding, row_buffer_padding, column_buffer_padding = buffer_padding

        # correctness check
        def naive_convolution_ref(input, kernel, output):
            input_channels, input_rows, input_columns = input.shape
            out_filters, output_rows, output_columns = output.shape
            _, _, kernel_rows, kernel_columns = kernel.shape
            output_ref = output.copy()
            for out_f in range(out_filters):
                for out_r in range(output_rows - 2 * row_buffer_padding):
                    for out_c in range(output_columns - 2 * column_buffer_padding):
                        for in_ch in range(input_channels):
                            for k_r in range(kernel_rows):
                                for k_c in range(kernel_columns):
                                    in_r = out_r * row_stride + k_r - row_padding
                                    in_c = out_c * column_stride + k_c - column_padding
                                    output_ref[out_f, out_r + row_buffer_padding, out_c + column_buffer_padding] += \
                                        input[in_ch, in_r + row_buffer_padding, in_c + column_buffer_padding] * \
                                        kernel[out_f, in_ch, k_r, k_c]
            return output_ref

        # unpadded_Input_test, unpadded_Kernel_test, unpadded_Output_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            Input_test, Kernel_test, Output_test = (np.random.random(p.shape).astype(np.float32) for p in function.args)
            Output_ref = naive_convolution_ref(Input_test, Kernel_test, Output_test)

            v.check_correctness(
                function.name,
                before=(Input_test, Kernel_test, Output_test),
                after=(Input_test, Kernel_test, Output_ref)
            )

    def _multicache_matmul_common(self, M, N, K, name_suffix, jjj_split=16) -> None:
        import accera as acc

        A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N))
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N))

        nest = acc.Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        jj = schedule.split(j, 128)
        kk = schedule.split(k, 256)
        kkk = schedule.split(kk, 4)
        jjj = schedule.split(jj, jjj_split)
        jjjj = schedule.split(jjj, 8)
        ii = schedule.split(i, 6)

        schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

        plan = schedule.create_plan()

        # multicache B by setting up a row-major submatrix cache at the
        # kk level with a trigger index at the jj level
        plan.cache(B, index=kk, trigger_index=jj)
        plan.cache(C, index=ii)

        plan.unroll(jjj)
        plan.unroll(ii)

        plan.vectorize(jjjj)

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=f"multicache_matmul_{name_suffix}")

        self._verify_matrix_multiplication_function(function, package, f"multicache_matmul_{name_suffix}")

    def test_multicache_matmul_no_boundary(self) -> None:
        self._multicache_matmul_common(1020, 1024, 1024, "no_boundary")

    def test_multicache_matmul_all_boundary(self) -> None:
        self._multicache_matmul_common(1023, 1023, 1023, "all_boundary")

    def test_multicache_matmul_internal_boundary(self) -> None:
        # This case forces multiple multicache buffers of different
        # shapes for a single cache due to unswitched boundary cases
        # existing between the trigger and cache levels
        # Because a jj split of 128 will evenly divide the N dim size of 1024
        # but an inner split of 24 won't evenly divide 128 though 8 will evenly divide 24

        self._multicache_matmul_common(1020, 1024, 1024, "internal_boundary", jjj_split=24)

    def _hierarchical_cache_matmul_common(
        self,
        M,
        N,
        K,
        A_cache_infos=[],
        B_cache_infos=[],
        C_cache_infos=[],
        force_boundary_conditions=False,
        cache_by_level=True
    ) -> None:
        import accera as acc

        A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N))
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N))

        nest = acc.Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        if force_boundary_conditions:
            # Split by prime numbers to force many boundary conditions
            jj = schedule.split(j, 127)
            kk = schedule.split(k, 251)
            kkk = schedule.split(kk, 11)
            jjj = schedule.split(jj, 17)
            jjjj = schedule.split(jjj, 7)
            ii = schedule.split(i, 5)
        else:
            jj = schedule.split(j, 128)
            kk = schedule.split(k, 256)
            kkk = schedule.split(kk, 4)
            jjj = schedule.split(jj, 16)
            jjjj = schedule.split(jjj, 8)
            ii = schedule.split(i, 6)

        order = [j, k, i, jj, kk, kkk, ii, jjj, jjjj]
        schedule.reorder(order)

        plan = schedule.create_plan()

        current_A_caches = [A]
        current_B_caches = [B]
        current_C_caches = [C]
        for (cache_level, cache_trigger_level, layout) in A_cache_infos:
            prev_cache = current_A_caches[len(current_A_caches) - 1]

            if cache_by_level:
                new_cache = plan.cache(prev_cache, level=cache_level, trigger_level=cache_trigger_level, layout=layout)
            else:
                cache_index = order[-cache_level]
                trigger_index = order[-cache_trigger_level]
                new_cache = plan.cache(prev_cache, index=cache_index, trigger_index=trigger_index, layout=layout)

            current_A_caches.append(new_cache)

        for (cache_level, cache_trigger_level, layout) in B_cache_infos:
            prev_cache = current_B_caches[len(current_B_caches) - 1]

            if cache_by_level:
                new_cache = plan.cache(prev_cache, level=cache_level, trigger_level=cache_trigger_level, layout=layout)
            else:
                cache_index = order[-cache_level]
                trigger_index = order[-cache_trigger_level]
                new_cache = plan.cache(prev_cache, index=cache_index, trigger_index=trigger_index, layout=layout)

            current_B_caches.append(new_cache)

        for (cache_level, layout) in C_cache_infos:
            prev_cache = current_C_caches[len(current_C_caches) - 1]

            if cache_by_level:
                new_cache = plan.cache(prev_cache, level=cache_level, layout=layout)
            else:
                cache_index = order[-cache_level]
                new_cache = plan.cache(prev_cache, index=cache_index, layout=layout)

            current_C_caches.append(new_cache)

        return plan, (A, B, C)

    def test_hierarchical_cache_matmul_2_caches_simple_boundaries_cache_by_index(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(5, 7, acc.Array.Layout.FIRST_MAJOR), (3, 5, acc.Array.Layout.LAST_MAJOR)]

        b_caches = [(4, 6, acc.Array.Layout.FIRST_MAJOR), (2, 4, acc.Array.Layout.LAST_MAJOR)]

        c_caches = [(8, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._hierarchical_cache_matmul_common(
            M,
            N,
            K,
            A_cache_infos=a_caches,
            B_cache_infos=b_caches,
            C_cache_infos=c_caches,
            force_boundary_conditions=False,
            cache_by_level=False
        )
        function = package.add(plan, args=args, base_name=f"hierarchical_cache_matmul_2_caches_simple_index")

        self._verify_matrix_multiplication_function(
            function, package, f"hierarchical_cache_matmul_2_caches_simple_index"
        )

    def test_hierarchical_cache_matmul_2_caches_simple_boundaries(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(5, 7, acc.Array.Layout.FIRST_MAJOR), (3, 5, acc.Array.Layout.LAST_MAJOR)]

        b_caches = [(4, 6, acc.Array.Layout.FIRST_MAJOR), (2, 4, acc.Array.Layout.LAST_MAJOR)]

        c_caches = [(8, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._hierarchical_cache_matmul_common(
            M,
            N,
            K,
            A_cache_infos=a_caches,
            B_cache_infos=b_caches,
            C_cache_infos=c_caches,
            force_boundary_conditions=False
        )
        function = package.add(plan, args=args, base_name=f"hierarchical_cache_matmul_2_caches_simple")

        self._verify_matrix_multiplication_function(function, package, f"hierarchical_cache_matmul_2_caches_simple")

    def test_hierarchical_cache_matmul_2_caches_complicated_boundaries(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(5, 7, acc.Array.Layout.FIRST_MAJOR), (3, 5, acc.Array.Layout.LAST_MAJOR)]

        b_caches = [(4, 6, acc.Array.Layout.FIRST_MAJOR), (2, 4, acc.Array.Layout.LAST_MAJOR)]

        c_caches = [(8, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._hierarchical_cache_matmul_common(
            M,
            N,
            K,
            A_cache_infos=a_caches,
            B_cache_infos=b_caches,
            C_cache_infos=c_caches,
            force_boundary_conditions=True
        )
        function = package.add(plan, args=args, base_name=f"hierarchical_cache_matmul_2_caches_complicated")

        self._verify_matrix_multiplication_function(
            function, package, f"hierarchical_cache_matmul_2_caches_complicated"
        )

    def test_hierarchical_cache_matmul_3_caches_simple_boundaries(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(5, 7, acc.Array.Layout.FIRST_MAJOR), (4, 5, acc.Array.Layout.LAST_MAJOR),
                    (3, 3, acc.Array.Layout.FIRST_MAJOR)]

        b_caches = [(5, 6, acc.Array.Layout.FIRST_MAJOR), (3, 5, acc.Array.Layout.LAST_MAJOR),
                    (2, 3, acc.Array.Layout.FIRST_MAJOR)]

        c_caches = [(8, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR),
                    (3, acc.Array.Layout.LAST_MAJOR)]

        package = Package()
        plan, args = self._hierarchical_cache_matmul_common(
            M,
            N,
            K,
            A_cache_infos=a_caches,
            B_cache_infos=b_caches,
            C_cache_infos=c_caches,
            force_boundary_conditions=False
        )
        function = package.add(plan, args=args, base_name=f"hierarchical_cache_matmul_3_caches_simple")

        self._verify_matrix_multiplication_function(function, package, f"hierarchical_cache_matmul_3_caches_simple")

    def test_hierarchical_cache_matmul_3_caches_complicated_boundaries(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(5, 7, acc.Array.Layout.FIRST_MAJOR), (4, 5, acc.Array.Layout.LAST_MAJOR),
                    (3, 3, acc.Array.Layout.FIRST_MAJOR)]

        b_caches = [(5, 6, acc.Array.Layout.FIRST_MAJOR), (3, 5, acc.Array.Layout.LAST_MAJOR),
                    (2, 3, acc.Array.Layout.FIRST_MAJOR)]

        c_caches = [(8, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR),
                    (3, acc.Array.Layout.LAST_MAJOR)]

        package = Package()
        plan, args = self._hierarchical_cache_matmul_common(
            M,
            N,
            K,
            A_cache_infos=a_caches,
            B_cache_infos=b_caches,
            C_cache_infos=c_caches,
            force_boundary_conditions=True
        )
        function = package.add(plan, args=args, base_name=f"hierarchical_cache_matmul_3_caches_complicated")

        self._verify_matrix_multiplication_function(
            function, package, f"hierarchical_cache_matmul_3_caches_complicated"
        )

    def test_hierarchical_cache_parameterized(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_level_1, a_level_2, a_trigger_1, a_trigger_2 = acc.create_parameters(4)
        b_level_1, b_level_2, b_trigger_1, b_trigger_2 = acc.create_parameters(4)
        c_level_1, c_level_2 = acc.create_parameters(2)

        a_caches = [(a_level_1, a_trigger_1, acc.Array.Layout.FIRST_MAJOR),
                    (a_level_2, a_trigger_2, acc.Array.Layout.LAST_MAJOR)]

        b_caches = [(b_level_1, b_trigger_1, acc.Array.Layout.FIRST_MAJOR),
                    (b_level_2, b_trigger_2, acc.Array.Layout.LAST_MAJOR)]

        c_caches = [(c_level_1, acc.Array.Layout.FIRST_MAJOR), (c_level_2, acc.Array.Layout.LAST_MAJOR)]

        package = Package()
        plan, args = self._hierarchical_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(
            plan,
            args=args,
            parameters={
                a_level_1: 5,
                a_trigger_1: 7,
                a_level_2: 3,
                a_trigger_2: 5,
                b_level_1: 5,
                b_trigger_1: 6,
                b_level_2: 3,
                b_trigger_2: 5,
                c_level_1: 8,
                c_level_2: 6,
            },
            base_name=f"parameterized_hierarchical_cache_matmul"
        )

        self._verify_matrix_multiplication_function(function, package, f"parameterized_hierarchical_cache_matmul")

    def _max_element_cache_matmul_common(
        self, M, N, K, A_cache_infos=[], B_cache_infos=[], C_cache_infos=[], force_boundary_conditions=False
    ) -> None:
        import accera as acc

        A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N))
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N))

        nest = acc.Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        if force_boundary_conditions:
            # Split by prime numbers to force many boundary conditions
            jj = schedule.split(j, 127)
            kk = schedule.split(k, 251)
            kkk = schedule.split(kk, 11)
            jjj = schedule.split(jj, 17)
            jjjj = schedule.split(jjj, 7)
            ii = schedule.split(i, 5)
        else:
            jj = schedule.split(j, 128)
            kk = schedule.split(k, 256)
            kkk = schedule.split(kk, 4)
            jjj = schedule.split(jj, 16)
            jjjj = schedule.split(jjj, 8)
            ii = schedule.split(i, 6)

        order = [j, k, i, jj, kk, kkk, ii, jjj, jjjj]
        schedule.reorder(order)

        plan = schedule.create_plan()

        current_A_caches = [A]
        current_B_caches = [B]
        current_C_caches = [C]
        for (cache_budget, layout) in A_cache_infos:
            prev_cache = current_A_caches[len(current_A_caches) - 1]
            new_cache = plan.cache(prev_cache, max_elements=cache_budget, layout=layout)
            current_A_caches.append(new_cache)

        for (cache_budget, layout) in B_cache_infos:
            prev_cache = current_B_caches[len(current_B_caches) - 1]
            new_cache = plan.cache(prev_cache, max_elements=cache_budget, layout=layout)
            current_B_caches.append(new_cache)

        for (cache_budget, layout) in C_cache_infos:
            prev_cache = current_C_caches[len(current_C_caches) - 1]
            new_cache = plan.cache(prev_cache, max_elements=cache_budget, layout=layout)
            current_C_caches.append(new_cache)

        return plan, (A, B, C)

    def test_simple_single_budget_cache(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        # This should set the cache level to the kkk index (footprint should be 6 * 4 = 24 elements at that point, whereas the next level out is 6 * 256 = 1536 elements)
        a_caches = [(128, acc.Array.Layout.LAST_MAJOR)]
        b_caches = []
        c_caches = []

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_simple_single_budget_cache")

        self._verify_matrix_multiplication_function(function, package, f"test_simple_single_budget_cache")

    def test_multiple_single_budget_caches(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        # This should set the cache level to the kkk index (footprint should be 6 * 4 = 24 elements at that point, whereas the next level out is 6 * 256 = 1536 elements)
        a_caches = [(128, acc.Array.Layout.LAST_MAJOR)]
        # This should set the cache level to the k index
        b_caches = [(K * N // 128, acc.Array.Layout.FIRST_MAJOR)]
        # This should set the cache level to the ii index
        c_caches = [(6 * 16, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_multiple_single_budget_caches")

        self._verify_matrix_multiplication_function(function, package, f"test_multiple_single_budget_caches")

    def test_hierarchical_budget_caches(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(128, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR)]
        b_caches = [(K * N // 128, acc.Array.Layout.FIRST_MAJOR), (K * N // (128 * 256), acc.Array.Layout.LAST_MAJOR)]
        c_caches = [(6 * 128, acc.Array.Layout.FIRST_MAJOR), (6 * 16, acc.Array.Layout.LAST_MAJOR)]

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_hierarchical_budget_caches")

        self._verify_matrix_multiplication_function(function, package, f"test_hierarchical_budget_caches")

    def test_hierarchical_budget_caches_boundary_conditions(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(128, acc.Array.Layout.LAST_MAJOR), (6, acc.Array.Layout.FIRST_MAJOR)]
        b_caches = [(K * N // 128, acc.Array.Layout.FIRST_MAJOR), (K * N // (128 * 256), acc.Array.Layout.LAST_MAJOR)]
        c_caches = [(6 * 128, acc.Array.Layout.FIRST_MAJOR), (6 * 16, acc.Array.Layout.LAST_MAJOR)]

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M,
            N,
            K,
            A_cache_infos=a_caches,
            B_cache_infos=b_caches,
            C_cache_infos=c_caches,
            force_boundary_conditions=True
        )
        function = package.add(plan, args=args, base_name=f"test_hierarchical_budget_caches_boundary_conditions")

        self._verify_matrix_multiplication_function(
            function, package, f"test_hierarchical_budget_caches_boundary_conditions"
        )

    def test_small_budget_cache(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(1, acc.Array.Layout.LAST_MAJOR)]
        b_caches = [(1, acc.Array.Layout.LAST_MAJOR)]
        c_caches = [(1, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_small_budget_cache")

        self._verify_matrix_multiplication_function(function, package, f"test_small_budget_cache")

    def test_max_budget_cache(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(M * K, acc.Array.Layout.LAST_MAJOR)]
        b_caches = [(K * N, acc.Array.Layout.FIRST_MAJOR)]
        c_caches = [(M * N, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_max_budget_cache")

        self._verify_matrix_multiplication_function(function, package, f"test_max_budget_cache")

    def test_overmax_budget_cache(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        a_caches = [(2 * M * K, acc.Array.Layout.LAST_MAJOR)]
        b_caches = [(2 * K * N, acc.Array.Layout.FIRST_MAJOR)]
        c_caches = [(2 * M * N, acc.Array.Layout.FIRST_MAJOR)]

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_overmax_budget_cache")

        self._verify_matrix_multiplication_function(function, package, f"test_overmax_budget_cache")

    def test_boundary_differently_shaped_budget_cache(self) -> None:
        import accera as acc

        M = 1024
        N = 1024
        K = 1024

        # Create max element caches that will have a different cache level in different boundary condition sections

        # with the default scheduling, there will be a boundary condition on the ii loop, where M = 0...1020 will be taken in steps of 6
        # then M = 1020...1024 will be taken in steps of 4
        # So if we create a budget that will be larger than the footprint in the boundary loop but smaller than the footprint in the base loop
        # we should wind up with two different cache buffers at two different levels in separate regions

        # 5 to be between 4 and 6, 256 because that is the K dimension tile size
        a_caches = [(5 * 256, acc.Array.Layout.LAST_MAJOR)]
        b_caches = []
        c_caches = []

        package = Package()
        plan, args = self._max_element_cache_matmul_common(
            M, N, K, A_cache_infos=a_caches, B_cache_infos=b_caches, C_cache_infos=c_caches
        )
        function = package.add(plan, args=args, base_name=f"test_boundary_differently_shaped_budget_cache")

        self._verify_matrix_multiplication_function(function, package, f"test_boundary_differently_shaped_budget_cache")

    def test_gpu_vec_add(self):
        from accera import Array, Nest, Package, ScalarType, Target

        # Define our vector sizes
        N = 2**16
        block_x = 256

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, ))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, ))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(N, ))

        nest = Nest(shape=(N, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i] = A[i] + B[i]

        schedule = nest.create_schedule()
        ii = schedule.split(i, block_x)
        schedule.reorder(i, ii)

        target = Target(category=Target.Category.GPU, runtime=Target.Runtime.VULKAN)
        plan = schedule.create_plan(target)
        plan.bind((i, ii), grid=(target.GridUnit.BLOCK_X, target.GridUnit.THREAD_X))

        test_name = "test_gpu_vec_add"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

    def test_gpu_vec_add(self):
        from accera import Array, Nest, Package, ScalarType, Target

        # Define our vector sizes
        N = 2**16
        block_x = 256

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, ))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, ))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(N, ))

        nest = Nest(shape=(N, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i] = A[i] + B[i]

        schedule = nest.create_schedule()
        ii = schedule.split(i, block_x)
        schedule.reorder(i, ii)

        target = Target(category=Target.Category.GPU, runtime=Target.Runtime.CUDA)
        plan = schedule.create_plan(target)
        plan.bind((i, ii), grid=(target.GridUnit.BLOCK_X, target.GridUnit.THREAD_X))

        test_name = "test_gpu_vec_add"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

            if CUDA_AVAILABLE:
                before = [np.random.rand(*p.shape).astype(np.float32) for p in function.args]
                after = [before[0], before[1]] + [before[0] + before[1]]

                v.check_correctness(function.name, before=before, after=after)

    def test_cuda_module_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target
        from accera._lang_python._lang import _MemorySpace

        # Define our vector sizes
        N = 2048
        block_x = 16
        block_y = block_x

        In = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, N))
        Out = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(N, N))

        nest = Nest(shape=(N, N))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            Out[i, j] = In[i, j]

        schedule = nest.create_schedule()

        ii, jj = schedule.tile((i, j), (block_x, block_y))
        schedule.reorder(i, j, ii, jj)

        target = Target(Target.Model.NVIDIA_V100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.cache(In, index=ii, location=_MemorySpace.SHARED, layout=Array.Layout.LAST_MAJOR)

        test_name = "test_cuda_module_output"
        package = Package()
        function = package.add(plan, args=(In, Out), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

            if CUDA_AVAILABLE:
                Input_test, Output_test = (np.random.uniform(-1, 1, p.shape).astype(np.float32) for p in function.args)
                Input_ref = Output_ref = Input_test

                v.check_correctness(function.name, before=(Input_test, Output_test), after=(Input_ref, Output_ref))

    def test_rocm_module_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target
        from accera._lang_python._lang import _MemorySpace

        # Define our vector sizes
        N = 32
        block_x = 16
        block_y = block_x

        In = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, N))
        Out = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(N, N))

        nest = Nest(shape=(N, N))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            Out[i, j] = In[i, j]

        schedule = nest.create_schedule()

        ii, jj = schedule.tile((i, j), (block_x, block_y))
        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.cache(In, index=ii, location=_MemorySpace.SHARED, layout=Array.Layout.LAST_MAJOR)

        test_name = "test_rocm_module_output"
        package = Package()
        function = package.add(plan, args=(In, Out), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

    def test_rocm_tensorize_single_block_single_warp_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 16
        N = M
        K = M
        outer_tile_x = 16
        outer_tile_y = outer_tile_x

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj = schedule.tile((i, j), (outer_tile_x, outer_tile_y))
        iii, jjj, kk = schedule.tile((ii, jj, k), (2, 2, 16))

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_Y, target.GridUnit.BLOCK_X, target.GridUnit.THREAD_Y, target.GridUnit.THREAD_X)
        )
        plan.tensorize(indices=(iii, jjj, kk))

        test_name = "test_rocm_tensorize_single_block_single_warp_output"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

    def test_rocm_tensorize_single_block_multi_warp_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 64
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj = schedule.tile((i, j), (outer_tile_x, outer_tile_y))
        iii, jjj, kk = schedule.tile((ii, jj, k), (2, 2, 16))

        schedule.reorder((i, j, k, ii, jj, iii, jjj, kk))

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_Y, target.GridUnit.BLOCK_X, target.GridUnit.THREAD_Y, target.GridUnit.THREAD_X)
        )
        plan.tensorize(indices=(iii, jjj, kk))

        test_name = "test_rocm_tensorize_single_block_multi_warp_output"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

    def test_rocm_tensorize_multi_block_multi_warp_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj = schedule.tile((i, j), (outer_tile_x, outer_tile_y))
        iii, jjj, kk = schedule.tile((ii, jj, k), (2, 2, 16))

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_Y, target.GridUnit.BLOCK_X, target.GridUnit.THREAD_Y, target.GridUnit.THREAD_X)
        )
        plan.tensorize(indices=(iii, jjj, kk))

        test_name = "test_rocm_tensorize_multi_block_multi_warp_output"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

    @expectedFailure(FailedReason.INVALID, "the hardware does not support the requested tensorcore shape")
    def test_rocm_tensorize_invalid_shape_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 256
        N = M
        K = M
        block_x = 64
        block_y = block_x
        tile_size = 64

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj = schedule.tile((i, j), (block_x, block_y))
        iii, jjj, kk = schedule.tile((ii, jj, k), (tile_size, tile_size, tile_size))

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.tensorize(indices=(iii, jjj, kk))

        test_name = "test_rocm_tensorize_invalid_shape_output"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

    def test_gpu_cache_simple(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target
        from accera.lang import CacheIndexing, BLOCK_X, BLOCK_Y, THREAD_X, THREAD_Y
        from accera._lang_python._lang import _MemorySpace

        M = 1024
        N = 1024
        K = 1024
        block_x = 16
        block_y = block_x
        k_tile_size = 32

        m_tile_size = block_x
        n_tile_size = block_y

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, k_tile_size))
        schedule.reorder(i, j, k, ii, jj, kk)

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.cache(A, index=ii, location=_MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(B, index=ii, location=_MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR)

        test_name = "test_gpu_cache_simple"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR_VERBOSE | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir,
                _quiet=False
            )

    def test_gpu_cache_double_buffering(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target
        from accera.lang import CacheIndexing, BLOCK_X, BLOCK_Y, THREAD_X, THREAD_Y
        from accera._lang_python._lang import _MemorySpace

        M = 2560
        N = 1536
        K = 2048
        block_x = 16
        block_y = block_x
        k_tile_size = 32

        m_tile_size = block_x
        n_tile_size = block_y

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, k_tile_size))
        schedule.reorder(i, j, k, ii, jj, kk)

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.cache(A, index=ii, double_buffer=True, location=_MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(B, index=ii, double_buffer=True, location=_MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR)

        test_name = "test_gpu_cache_double_buffering"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR_VERBOSE | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir,
                _quiet=False
            )

    def test_gpu_cache_double_buffering_trigger_index(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target
        from accera.lang import CacheIndexing, BLOCK_X, BLOCK_Y, THREAD_X, THREAD_Y
        from accera._lang_python._lang import _MemorySpace

        M = 2560
        N = 1536
        K = 2048
        block_x = 16
        block_y = block_x
        k_outer_tile_size = 512
        k_inner_tile_size = 32

        m_tile_size = block_x
        n_tile_size = block_y

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, k_outer_tile_size))
        kkk = schedule.split(kk, k_inner_tile_size)
        schedule.reorder(i, j, k, kk, ii, jj, kkk)

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.cache(
            A,
            index=ii,
            trigger_index=kk,
            double_buffer=True,
            location=_MemorySpace.SHARED,
            layout=Array.Layout.FIRST_MAJOR
        )
        plan.cache(
            B,
            index=ii,
            trigger_index=kk,
            double_buffer=True,
            location=_MemorySpace.SHARED,
            layout=Array.Layout.FIRST_MAJOR
        )

        test_name = "test_gpu_cache_double_buffering_trigger_index"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR_VERBOSE | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir,
                _quiet=False
            )

    def test_gpu_cache_double_buffering_mem_space(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target
        from accera.lang import CacheIndexing, BLOCK_X, BLOCK_Y, THREAD_X, THREAD_Y
        from accera._lang_python._lang import _MemorySpace

        M = 2560
        N = 1536
        K = 2048
        block_x = 16
        block_y = block_x
        k_tile_size = 32

        m_tile_size = block_x
        n_tile_size = block_y

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, k_tile_size))
        schedule.reorder(i, j, k, ii, jj, kk)

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )
        plan.cache(
            A, index=ii, double_buffer=True, location=_MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR
        )    # Double buffer should be in private mem
        plan.cache(
            B,
            index=ii,
            double_buffer=True,
            location=_MemorySpace.SHARED,
            double_buffer_location=_MemorySpace.SHARED,
            layout=Array.Layout.FIRST_MAJOR
        )

        test_name = "test_gpu_cache_double_buffering_mem_space"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR_VERBOSE | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir,
                _quiet=False
            )

    def test_cpu_cache_double_buffering_trigger_index(self) -> None:
        from accera import Array, Nest, Package, ScalarType

        M = 1024
        N = 1024
        K = 1024
        m_tile_size = 16
        n_tile_size = 16
        k_outer_tile_size = 256
        k_inner_tile_size = 32

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (m_tile_size, n_tile_size, k_outer_tile_size))
        kkk = schedule.split(kk, k_inner_tile_size)
        schedule.reorder(i, j, k, kk, ii, jj, kkk)

        plan = schedule.create_plan()
        plan.cache(A, index=ii, trigger_index=kk, double_buffer=True, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(B, index=ii, trigger_index=kk, double_buffer=True, layout=Array.Layout.FIRST_MAJOR)

        test_name = "test_cpu_cache_double_buffering_trigger_index"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(function, package, test_name)

    def test_gpu_barrier_opt(self):
        from accera import Array, Nest, Package, ScalarType, Target
        from accera._lang_python._lang import Allocate, _MemorySpace, Array as NativeArray
        from accera._lang_python._lang._gpu import Barrier
        from accera._lang_python import _MemoryLayout

        N = 256
        block_x = 16

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, ))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(N, ))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(N, ))

        nest = Nest(shape=(N, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            # Performs excessive barriers.
            shA = NativeArray(
                Allocate(
                    type=ScalarType.float32, layout=_MemoryLayout([block_x]).set_memory_space(_MemorySpace.SHARED)
                )
            )
            shB = NativeArray(
                Allocate(
                    type=ScalarType.float32, layout=_MemoryLayout([block_x]).set_memory_space(_MemorySpace.SHARED)
                )
            )
            Barrier()
            shA[i] = A[i]
            Barrier()
            Barrier()
            shA[i] = B[i]
            Barrier()    # Only this is needed.
            C[i] = shA[i] + shB[i]
            Barrier()

        schedule = nest.create_schedule()
        ii = schedule.split(i, block_x)
        schedule.reorder(i, ii)

        target = Target(category=Target.Category.GPU, runtime=Target.Runtime.ROCM)
        plan = schedule.create_plan(target)
        plan.bind((i, ii), grid=(target.GridUnit.BLOCK_X, target.GridUnit.THREAD_X))

        test_name = "test_gpu_barrier_opt"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )

    def test_rocm_gemm_tiled_output(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 16
        N = M
        K = M
        block_x = 16
        block_y = block_x
        tile_size = 16

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()
        ii, jj = schedule.tile((i, j), (block_x, block_y))
        iii, jjj, kk = schedule.tile((ii, jj, k), (tile_size, tile_size, tile_size))

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_X, target.GridUnit.BLOCK_Y, target.GridUnit.THREAD_X, target.GridUnit.THREAD_Y)
        )

        test_name = "test_rocm_gemm_tiled_output"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        # We expect the output to have a block dim = [1,1,1] and grid dim = [1,1,1]
        # there will be an inner 16x16x16 loop that performs the actual computation.
        # i.e. the computation is performed by a single block and which contains a single thread
        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir
            )

    # Thrifty caching
    # Note: these tests will only verify that the thrify caching cases compile and compute the correct result,
    #       however they will not validate when a cache buffer is successfully elided due to the delayed lowering
    #       model we have. Currently the only way to verify this is manual IR inspection following the LoopNestToValuFunc
    #       lowering pass

    def test_thrifty_caching_simple_input_cache(self) -> None:
        import accera as acc

        package = Package()

        M = 32
        N = 32
        K = 32

        A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

        nest = acc.Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii = schedule.split(i, 4)
        jj = schedule.split(j, 16)
        kk = schedule.split(k, 32)

        order = [i, j, k, ii, jj, kk]
        schedule.reorder(order)

        plan = schedule.create_plan()

        # This cache should get elided because at ii the active block is of shape 4x32, which is a contiguous subarray of the 32x32 base array A
        AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

        # This cache should not get elided because at k the active block is of shape 32x16, which is a discontiguous subarray of the 32x32 base array B
        BB = plan.cache(B, index=k, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

        function = package.add(plan, args=(A, B, C), base_name=f"test_thrifty_caching_simple_input_cache")

        self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_simple_input_cache")

    def test_thrifty_caching_simple_output_cache_elide(self) -> None:
        import accera as acc

        package = Package()

        M = 32
        N = 32
        K = 32

        A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

        nest = acc.Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii = schedule.split(i, 4)
        jj = schedule.split(j, 32)
        kk = schedule.split(k, 8)

        order = [i, j, k, ii, jj, kk]
        schedule.reorder(order)

        plan = schedule.create_plan()

        # This cache should get elided because at ii the active block has the shape 4x32, which is a contiguous subarray of the 32x32 base array C
        CC = plan.cache(C, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

        function = package.add(plan, args=(A, B, C), base_name=f"test_thrifty_caching_simple_output_cache_elide")

        self._verify_matrix_multiplication_function(
            function, package, f"test_thrifty_caching_simple_output_cache_elide"
        )

    # Note: The following thrifty cache tests are commented out as they increase the runtime of the smoke_test by too much
    # TODO : move these to a new exhaustive test suite that isn't run as part of the buddy build

    # def test_thrifty_caching_simple_output_cache_no_elide(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K))
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N))
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N))

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     jj = schedule.split(j, 8)
    #     kk = schedule.split(k, 32)

    #     order = [i, j, k, ii, jj, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # This cache should not get elided because at ii the active block is of shape 4x8, which is a discontiguous subarray of the base array C
    #     CC = plan.cache(C, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_simple_output_cache_no_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_simple_output_cache_no_elide")

    # def test_thrifty_caching_transpose_input_no_elide(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     jj = schedule.split(j, 8)
    #     kk = schedule.split(k, 32)

    #     order = [i, j, k, ii, jj, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # Note that at index ii, the active block is of shape 4x32 and is a contiguous sub-buffer of the base array A,
    #     # however the cache stride is different between elements so it should not get elided
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.LAST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_transpose_input_no_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_transpose_input_no_elide")

    # def test_thrifty_caching_transpose_input_elide(self) -> None:
    #     # This case transposes the shape of the input cache, however the schedule and cache are constructed such that the active
    #     # block is a 1-D slice that is contiguous in the base array

    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 1)
    #     jj = schedule.split(j, 8)
    #     kk = schedule.split(k, 32)

    #     order = [i, j, k, ii, jj, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # Note that at index ii, the active block is of shape 1x32 and is a contiguous sub-buffer of the base array A,
    #     # and even though the cache transposes the active block, it still has a stride of 1 between the elements as
    #     # so it should get elided
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.LAST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_transpose_input_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_transpose_input_elide")

    # def test_thrifty_caching_with_trigger_index_elide(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 16)
    #     iii = schedule.split(ii, 4)
    #     jj = schedule.split(j, 8)
    #     kk = schedule.split(k, 32)

    #     order = [i, k, ii, j, iii, jj, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # This cache should get elided because at ii the active block is of shape 4x32, which is a contiguous subarray of the 32x32 base array A
    #     # and the successive 4x32 active blocks within the 16x32 region covered at index ii are sequential contiguous subarrays of the 32x32 base array A
    #     # Note: index j between ii and iii in the order should have no effect on this
    #     AA = plan.cache(A, index=iii, trigger_index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_with_trigger_index_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_with_trigger_index_elide")

    # def test_thrifty_caching_convolution_duplication_no_elide(self) -> None:
    #     # A cache with a trigger index can result in input element duplication if the successive active blocks overlap.
    #     # When the active blocks overlap, the resulting multi-cache with duplication is certainly not a strict
    #     # subbuffer of the original base array, and therefore the cache should not get elided

    #     # Caching the input array in a Conv2D operation produces duplication as long as the
    #     # kernel rows and/or kernel columns loops are inside of the cache region in the loopnest

    #     input_channels = 32
    #     base_input_shape = (input_channels, 14, 14)
    #     buffer_padding = (0, 1, 1)
    #     conv_padding = (0, 1, 1)
    #     stride = (2, 2)
    #     kernel_shape = (3, 3)
    #     output_filters = 32

    #     import math
    #     unpadded_output_rows = math.floor(
    #         ((base_input_shape[1] + (2 * conv_padding[1]) - (kernel_shape[0] - 1) - 1) / stride[0]) + 1)
    #     unpadded_output_columns = math.floor(
    #         ((base_input_shape[2] + (2 * conv_padding[2]) - (kernel_shape[1] - 1) - 1) / stride[1]) + 1)
    #     base_output_shape = (output_filters, unpadded_output_rows, unpadded_output_columns)

    #     # Pad the buffers so we don't need to deal with conditionals at the edges in this test
    #     padded_input_shape = [base_input_shape[i] + 2*buffer_padding[i]
    #                           for i in range(len(base_input_shape))]
    #     padded_output_shape = [base_output_shape[i] + 2*buffer_padding[i]
    #                            for i in range(len(base_output_shape))]

    #     weights_shape = (output_filters, input_channels, kernel_shape[0], kernel_shape[1])

    #     Input = Array(role=Array.Role.INPUT,
    #                   element_type=ScalarType.float32, shape=padded_input_shape)
    #     Kernel = Array(role=Array.Role.INPUT,
    #                    element_type=ScalarType.float32, shape=weights_shape)
    #     Output = Array(role=Array.Role.INPUT_OUTPUT,
    #                    element_type=ScalarType.float32, shape=padded_output_shape)

    #     nest = Nest(shape=(output_filters,
    #                        input_channels,
    #                        unpadded_output_rows,
    #                        unpadded_output_columns,
    #                        kernel_shape[0],
    #                        kernel_shape[1]))

    #     out_f, in_ch, out_r, out_c, k_r, k_c = nest.get_indices()

    #     row_stride, column_stride = stride
    #     channel_padding, row_padding, column_padding = conv_padding
    #     channel_buffer_padding, row_buffer_padding, column_buffer_padding = buffer_padding
    #     # Define the iteration logic

    #     @nest.iteration_logic
    #     def _():
    #         in_r = out_r * row_stride - row_padding + k_r
    #         in_c = out_c * column_stride - column_padding + k_c
    #         Output[out_f, out_r + row_buffer_padding, out_c + column_buffer_padding] += \
    #             Input[in_ch, in_r + row_buffer_padding, in_c + column_buffer_padding] * \
    #             Kernel[out_f, in_ch, k_r, k_c]

    #     schedule = nest.create_schedule()

    #     # We don't need to reorder as the kernel row and kernel column loops are already inside
    #     # of the tensor shape loops in the nest

    #     plan = schedule.create_plan()

    #     # Cache input array at a level that will produce duplication
    #     # This thrifty cache should not be elided as it is duplicating input elements
    #     # so it has an inconsistent stride between the base input and the cache
    #     # The active cache here should be a 3x3 subarray in the rows/columns dimension of the input
    #     # and the trigger level should cause duplication between the column sections as there is overlap
    #     plan.cache(Input, trigger_index=out_c, index=k_r, thrifty=True, layout=Array.Layout.FIRST_MAJOR)

    #     package = Package()
    #     function = package.add(plan, args=(
    #         Input, Kernel, Output), base_name="test_thrifty_caching_convolution_duplication_no_elide")

    #     package_name = "test_thrifty_caching_convolution_duplication_no_elide"

    #     self._verify_convolution_function(function, package, package_name, buffer_padding, conv_padding, stride)

    # def test_thrifty_caching_convolution_no_duplication_elide(self) -> None:
    #     # This test creates a convolution loopnest but with the kernel row and kernel column loops
    #     # outside of the input cache region, so there is no element duplication and the cache is a subbuffer
    #     # of the input, so the thrifty cache can be elided

    #     input_channels = 32
    #     base_input_shape = (input_channels, 14, 14)
    #     buffer_padding = (0, 1, 1)
    #     conv_padding = (0, 1, 1)
    #     stride = (2, 2)
    #     kernel_shape = (3, 3)
    #     output_filters = 32

    #     import math
    #     unpadded_output_rows = math.floor(
    #         ((base_input_shape[1] + (2 * conv_padding[1]) - (kernel_shape[0] - 1) - 1) / stride[0]) + 1)
    #     unpadded_output_columns = math.floor(
    #         ((base_input_shape[2] + (2 * conv_padding[2]) - (kernel_shape[1] - 1) - 1) / stride[1]) + 1)
    #     base_output_shape = (
    #         output_filters, unpadded_output_rows, unpadded_output_columns)

    #     # Pad the buffers so we don't need to deal with conditionals at the edges in this test
    #     padded_input_shape = [base_input_shape[i] + 2*buffer_padding[i]
    #                           for i in range(len(base_input_shape))]
    #     padded_output_shape = [base_output_shape[i] + 2*buffer_padding[i]
    #                            for i in range(len(base_output_shape))]

    #     weights_shape = (output_filters, input_channels, kernel_shape[0], kernel_shape[1])

    #     Input = Array(role=Array.Role.INPUT,
    #                   element_type=ScalarType.float32, shape=padded_input_shape)
    #     Kernel = Array(role=Array.Role.INPUT,
    #                    element_type=ScalarType.float32, shape=weights_shape)
    #     Output = Array(role=Array.Role.INPUT_OUTPUT,
    #                    element_type=ScalarType.float32, shape=padded_output_shape)

    #     nest = Nest(shape=(output_filters,
    #                        input_channels,
    #                        unpadded_output_rows,
    #                        unpadded_output_columns,
    #                        kernel_shape[0],
    #                        kernel_shape[1]))

    #     out_f, in_ch, out_r, out_c, k_r, k_c = nest.get_indices()

    #     row_stride, column_stride = stride
    #     channel_padding, row_padding, column_padding = conv_padding
    #     channel_buffer_padding, row_buffer_padding, column_buffer_padding = buffer_padding
    #     # Define the iteration logic

    #     @nest.iteration_logic
    #     def _():
    #         in_r = out_r * row_stride - row_padding + k_r
    #         in_c = out_c * column_stride - column_padding + k_c
    #         Output[out_f, out_r + row_buffer_padding, out_c + column_buffer_padding] += \
    #             Input[in_ch, in_r + row_buffer_padding, in_c + column_buffer_padding] * \
    #             Kernel[out_f, in_ch, k_r, k_c]

    #     schedule = nest.create_schedule()

    #     # Reorder the schedule to put the kernel row and kernel column loops outside the
    #     # row, and column loops
    #     schedule.reorder(out_f, in_ch, k_r, k_c, out_r, out_c)

    #     plan = schedule.create_plan()

    #     # This thrifty cache should be elided as it is a strict subbuffer of the original input array
    #     plan.cache(Input, index=out_c, thrifty=True, layout=Array.Layout.FIRST_MAJOR)

    #     package = Package()
    #     function = package.add(plan, args=(
    #         Input, Kernel, Output), base_name="test_thrifty_caching_convolution_no_duplication_elide")

    #     package_name = "test_thrifty_caching_convolution_no_duplication_elide"

    #     self._verify_convolution_function(function, package, package_name, buffer_padding, conv_padding, stride)

    # def test_thrifty_caching_convolution_no_duplication_no_elide_padding(self) -> None:
    #     # This test creates a convolution loopnest but with the kernel row and kernel column loops
    #     # outside of the input cache region, so there is no element duplication and the cache is a subbuffer
    #     # of the input, but the cached region doesn't include the padding in the input buffer, so the cache
    #     # should not be elided

    #     input_channels = 32
    #     base_input_shape = (input_channels, 14, 14)
    #     buffer_padding = (0, 1, 1)
    #     conv_padding = (0, 1, 1)
    #     stride = (2, 2)
    #     kernel_shape = (3, 3)
    #     output_filters = 32

    #     import math
    #     unpadded_output_rows = math.floor(
    #         ((base_input_shape[1] + (2 * conv_padding[1]) - (kernel_shape[0] - 1) - 1) / stride[0]) + 1)
    #     unpadded_output_columns = math.floor(
    #         ((base_input_shape[2] + (2 * conv_padding[2]) - (kernel_shape[1] - 1) - 1) / stride[1]) + 1)
    #     base_output_shape = (
    #         output_filters, unpadded_output_rows, unpadded_output_columns)

    #     # Pad the buffers so we don't need to deal with conditionals at the edges in this test
    #     padded_input_shape = [base_input_shape[i] + 2*buffer_padding[i]
    #                           for i in range(len(base_input_shape))]
    #     padded_output_shape = [base_output_shape[i] + 2*buffer_padding[i]
    #                            for i in range(len(base_output_shape))]

    #     weights_shape = (output_filters, input_channels, kernel_shape[0], kernel_shape[1])

    #     Input = Array(role=Array.Role.INPUT,
    #                   element_type=ScalarType.float32, shape=padded_input_shape)
    #     Kernel = Array(role=Array.Role.INPUT,
    #                    element_type=ScalarType.float32, shape=weights_shape)
    #     Output = Array(role=Array.Role.INPUT_OUTPUT,
    #                    element_type=ScalarType.float32, shape=padded_output_shape)

    #     nest = Nest(shape=(output_filters,
    #                        input_channels,
    #                        unpadded_output_rows,
    #                        unpadded_output_columns,
    #                        kernel_shape[0],
    #                        kernel_shape[1]))

    #     out_f, in_ch, out_r, out_c, k_r, k_c = nest.get_indices()

    #     row_stride, column_stride = stride
    #     channel_padding, row_padding, column_padding = conv_padding
    #     channel_buffer_padding, row_buffer_padding, column_buffer_padding = buffer_padding
    #     # Define the iteration logic

    #     @nest.iteration_logic
    #     def _():
    #         in_r = out_r * row_stride - row_padding + k_r
    #         in_c = out_c * column_stride - column_padding + k_c
    #         Output[out_f, out_r + row_buffer_padding, out_c + column_buffer_padding] += \
    #             Input[in_ch, in_r + row_buffer_padding, in_c + column_buffer_padding] * \
    #             Kernel[out_f, in_ch, k_r, k_c]

    #     schedule = nest.create_schedule()

    #     # Reorder the schedule to put the kernel row and kernel column loops outside the
    #     # row, and column loops
    #     schedule.reorder(out_f, in_ch, k_r, k_c, out_r, out_c)

    #     plan = schedule.create_plan()

    #     # This thrifty cache should be elided as it is a strict subbuffer of the original input array
    #     plan.cache(Input, index=out_r, thrifty=True, layout=Array.Layout.FIRST_MAJOR)

    #     package = Package()
    #     function = package.add(plan, args=(
    #         Input, Kernel, Output), base_name="test_thrifty_caching_convolution_no_duplication_no_elide_padding")

    #     package_name = "test_thrifty_caching_convolution_no_duplication_no_elide_padding"

    #     self._verify_convolution_function(function, package, package_name, buffer_padding, conv_padding, stride)

    # def test_thrifty_caching_max_elements_elide(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 32)
    #     kkk = schedule.split(kk, 8)

    #     order = [i, k, j, kk, ii, kkk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # This cache should get elided because with a budget of 4*32 the cache level will be
    #     # at index kk, where the active block is of shape 4x32, which is a contiguous subarray of the 32x32 base array A
    #     AA = plan.cache(A, max_elements=4*32, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_max_elements_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_max_elements_elide")

    # def test_thrifty_caching_max_elements_no_elide(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 32)
    #     kkk = schedule.split(kk, 8)

    #     order = [i, k, j, kk, ii, kkk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # This cache should not get elided because with a budget of (4*32 - 1) the cache level will be
    #     # at index ii, where the active block is of shape 4x32, which is not a contiguous subarray of the 32x32 base array A
    #     AA = plan.cache(A, max_elements=(4*32 - 1), thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_max_elements_no_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_max_elements_no_elide")

    # def test_thrifty_caching_coefficient_layout_elide(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 32)
    #     kkk = schedule.split(kk, 8)

    #     order = [i, k, j, kk, ii, kkk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # This cache should get elided because with a cache level of kk, the active block is of shape 4x32, which is a contiguous subarray of the 32x32 base array A
    #     # and the cache layout is a coefficient-specified layout which is equivalent to FIRST_MAJOR for this case
    #     AA = plan.cache(A, index=kk, thrifty=True, layout=(32, 1))

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_coefficient_layout_elide")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_coefficient_layout_elide")

    # @expectedFailure(FailedReason.BUG, "Coefficient caches with gaps don't create sufficiently large buffers")
    # def test_thrifty_caching_coefficient_layout_no_elide_gaps(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 32)
    #     kkk = schedule.split(kk, 8)

    #     order = [i, k, j, kk, ii, kkk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # This cache should get not elided because with a cache level of kk, the active block is of shape 4x32, which is a contiguous subarray of the 32x32 base array A
    #     # but the cache layout is a coefficient-specified layout which is almost equivalent to FIRST_MAJOR for this case but has
    #     # 2 additional empty elements between rows
    #     AA = plan.cache(A, index=kk, thrifty=True, layout=(32 + 2, 1))

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_coefficient_layout_no_elide_gaps")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_coefficient_layout_no_elide_gaps")

    # def test_thrifty_caching_different_memory_space_no_elide(self) -> None:
    #     import accera as acc

    #     # TODO : update once MemorySpace is better surfaced
    #     from accera._lang_python._lang import _MemorySpace

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 32)

    #     order = [i, j, k, ii, kk]
    #     schedule.reorder(order)

    #     target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    #     plan = schedule.create_plan(target)

    #     # With a cache level of ii, the active block is 4x32 and is a contiguous subarray of the base array A, however
    #     # the cache should not get elided because it resides in a different memory space from the base array
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR, location=_MemorySpace.SHARED)

    #     # Shared -> PRIVATE move so this should not get elided even though with a cache index of kk it is a sinlge contiguous row
    #     # copy from the outer cache
    #     AAA = plan.cache(AA, index=kk, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR, location=_MemorySpace.PRIVATE)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_different_memory_space_no_elide")

    #     package_name = f"test_thrifty_caching_different_memory_space_no_elide"
    #     output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
    #     shutil.rmtree(output_dir, ignore_errors=True)

    #     gpu_package_format = Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE
    #     with verifiers.VerifyPackage(self, package_name, output_dir) as v:
    #         package.build(name=package_name, format=gpu_package_format, mode=self.PACKAGE_MODE, output_dir=output_dir)

    # def test_thrifty_caching_multiple_memory_spaces_elide(self) -> None:
    #     import accera as acc

    #     # TODO : update once MemorySpace is better surfaced
    #     from accera._lang_python._lang import _MemorySpace

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 32)

    #     order = [i, j, k, ii, kk]
    #     schedule.reorder(order)

    #     target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    #     plan = schedule.create_plan(target)

    #     # With a cache level of ii, the active block is 4x32 and is a contiguous subarray of the base array A, however
    #     # the cache should not get elided because it resides in a different memory space from the base array
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR, location=_MemorySpace.SHARED)

    #     # This cache is a contigous subarray of the outer cache and it is in the same memory space, so it should get elided
    #     AAA = plan.cache(AA, index=kk, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR, location=_MemorySpace.SHARED)

    #     function = package.add(plan,
    #                            args=(A,B,C),
    #                            base_name=f"test_thrifty_caching_multiple_memory_spaces_elide")

    #     package_name = f"test_thrifty_caching_multiple_memory_spaces_elide"
    #     output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
    #     shutil.rmtree(output_dir, ignore_errors=True)

    #     gpu_package_format = Package.Format.MLIR | Package.Format.CUDA | Package.Format.HAT_PACKAGE
    #     with verifiers.VerifyPackage(self, package_name, output_dir) as v:
    #         package.build(name=package_name, format=gpu_package_format, mode=self.PACKAGE_MODE, output_dir=output_dir, _quiet=False)

    # def test_thrifty_caching_hierarchical_elide_outer(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 8)

    #     order = [i, j, k, ii, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # With a cache level of k, the active block is 4x32 and is a contiguous subarray of the base array A,
    #     # so it should get elided
    #     AA = plan.cache(A, index=k, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     # With a cache level of ii, the active block is 4x8 inside a cache of size 4x32, so the cache should not get elided
    #     AAA = plan.cache(AA, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A, B, C),
    #                            base_name=f"test_thrifty_caching_hierarchical_elide_outer")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_hierarchical_elide_outer")

    # def test_thrifty_caching_hierarchical_elide_inner(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 8)

    #     order = [i, j, k, ii, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # With a cache level of ii, the active block is 4x8 and is not a contiguous subarray of the base array A,
    #     # so it should not get elided
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     # With a cache level of kk, the active block is 1x8 inside a cache of size 4x8, so it is a contiguous subarray
    #     # of the cache AA so this inner cache should get elided
    #     AAA = plan.cache(AA, index=kk, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A, B, C),
    #                            base_name=f"test_thrifty_caching_hierarchical_elide_inner")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_hierarchical_elide_inner")

    # def test_thrifty_caching_hierarchical_elide_middle(self) -> None:
    #     import accera as acc

    #     package = Package()

    #     M = 32
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 8)
    #     iii = schedule.split(ii, 2)
    #     kk = schedule.split(k, 16)
    #     kkk = schedule.split(kk, 4)

    #     order = [i, j, k, ii, kk, iii, kkk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # With a cache level of ii, the active block is 8x16 and is not a contiguous subarray of the base array A,
    #     # so it should not get elided
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     # With a cache level of kk, the active block is 2x16 inside a cache of size 8x16, so it is a contiguous subarray
    #     # of the cache array AA so this cache should get elided
    #     AAA = plan.cache(AA, index=kk, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     # With a cache level of iii, the active block is 2x4 inside a cache of size 2x16, or really 8x16 after the middle cache is elided.
    #     # In either case this is a discontiguous subarray so this cache should not get elided
    #     AAAA = plan.cache(AAA, index=iii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A, B, C),
    #                            base_name=f"test_thrifty_caching_hierarchical_elide_middle")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_hierarchical_elide_middle")

    # def test_thrifty_caching_elide_boundary_no_elide_main(self) -> None:
    #     # This case creates a loopnest where in a boundary condition the cached segment is a strict subarray of the base array
    #     # but the main section of the loop is not, so the boundary section of the cache should is elided, but the main
    #     # section is not

    #     import accera as acc

    #     package = Package()

    #     M = 33 # 33 so that M % i_split_size = 1 and a boundary condition is created
    #     N = 32
    #     K = 32

    #     A = acc.Array(role=acc.Array.Role.INPUT, shape=(M, K), layout=acc.Array.Layout.FIRST_MAJOR)
    #     B = acc.Array(role=acc.Array.Role.INPUT, shape=(K, N), layout=acc.Array.Layout.FIRST_MAJOR)
    #     C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(M, N), layout=acc.Array.Layout.FIRST_MAJOR)

    #     nest = acc.Nest(shape=(M, N, K))
    #     i, j, k = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         C[i, j] += A[i, k] * B[k, j]

    #     schedule = nest.create_schedule()

    #     ii = schedule.split(i, 4)
    #     kk = schedule.split(k, 8)

    #     order = [i, j, k, ii, kk]
    #     schedule.reorder(order)

    #     plan = schedule.create_plan()

    #     # With a cache level of k, the active block is 4x8 in the main part of the loopnest and is a discontiguous subarray of the base array A,
    #     # so it should not get elided,
    #     # However in the boundary condition on ii created because (M % i_split_size) = (33 % 4) = 1, the active block is 1x8, which is a contiguous
    #     # subarray of the base array A, so the boundary cache should get elided
    #     AA = plan.cache(A, index=ii, thrifty=True, layout=acc.Array.Layout.FIRST_MAJOR)

    #     function = package.add(plan,
    #                            args=(A, B, C),
    #                            base_name=f"test_thrifty_caching_elide_boundary_no_elide_main")

    #     self._verify_matrix_multiplication_function(function, package, f"test_thrifty_caching_elide_boundary_no_elide_main")

    def test_rocm_cache_tensorize(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = 1024
        K = 1024
        outer_tile_x = 64
        outer_tile_y = outer_tile_x
        outer_tile_k = 64

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (outer_tile_x, outer_tile_y, outer_tile_k))
        iii, jjj, kkk = schedule.tile((ii, jj, kk), (2, 2, 16))

        schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk)

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_Y, target.GridUnit.BLOCK_X, target.GridUnit.THREAD_Y, target.GridUnit.THREAD_X)
        )
        plan.tensorize(indices=(iii, jjj, kkk))
        plan.cache(
            A, index=ii, double_buffer=False, location=target.MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR
        )
        plan.cache(
            B, index=ii, double_buffer=False, location=target.MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR
        )

        test_name = "test_rocm_cache_tensorize"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR_VERBOSE | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir,
                _quiet=False
            )

    def test_rocm_cache_double_buffering_tensorize(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        M = 1024
        N = 1024
        K = 1024
        outer_tile_x = 64
        outer_tile_y = outer_tile_x
        outer_tile_k = 64

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.FIRST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=Array.Layout.FIRST_MAJOR)
        C = Array(
            role=Array.Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(M, N),
            layout=Array.Layout.FIRST_MAJOR
        )

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile((i, j, k), (outer_tile_x, outer_tile_y, outer_tile_k))
        iii, jjj, kkk = schedule.tile((ii, jj, kk), (2, 2, 16))

        schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk)

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            (i, j, ii, jj),
            grid=(target.GridUnit.BLOCK_Y, target.GridUnit.BLOCK_X, target.GridUnit.THREAD_Y, target.GridUnit.THREAD_X)
        )
        plan.tensorize(indices=(iii, jjj, kkk))
        plan.cache(A, index=ii, double_buffer=True, location=target.MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(B, index=ii, double_buffer=True, location=target.MemorySpace.SHARED, layout=Array.Layout.FIRST_MAJOR)

        test_name = "test_rocm_cache_double_buffering_tensorize"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR_VERBOSE | Package.Format.CUDA | Package.Format.HAT_PACKAGE,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG,
                output_dir=output_dir,
                _quiet=False
            )

    def test_fill_fp16(self):
        from accera import Array, Nest, Package, ScalarType
        from accera import _cast

        # Define our vector sizes
        N = 2**16

        Out = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float16, shape=(N, ))

        nest = Nest(shape=(N, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            Out[i] = _cast(2, ScalarType.float16)

        schedule = nest.create_schedule()
        plan = schedule.create_plan()

        package = Package()
        package_name = "test_fill_fp16"
        function = package.add(plan, args=(Out, ), base_name=package_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            def fill_fp16():
                return 2 * np.ones(N).astype(np.float16)

            Output_test = np.random.random(N).astype(np.float16)
            Output_ref = fill_fp16()

            v.check_correctness(function.name, before=(Output_test, ), after=(Output_ref, ))

    def test_abs_fp16(self):
        from accera import Array, Nest, Package, ScalarType, Target
        from accera import abs

        # Define our vector sizes
        N = 16

        In = Array(role=Array.Role.INPUT, element_type=ScalarType.float16, shape=(N, ))
        Out = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float16, shape=(N, ))

        nest = Nest(shape=(N, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            Out[i] = abs(In[i])

        schedule = nest.create_schedule()
        plan = schedule.create_plan()

        package = Package()
        package_name = "test_add_scalar_fp16"
        function = package.add(plan, args=(In, Out), base_name=package_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            def abs_fp16(a):
                return np.abs(a)

            Input_test, Output_test = (np.random.uniform(-1, 1, p.shape).astype(np.float16) for p in function.args)
            Output_ref = abs_fp16(Input_test)

            v.check_correctness(function.name, before=(Input_test, Output_test), after=(Input_test, Output_ref))

    def test_vec_add_fp16(self):
        from accera import Array, Nest, Package, ScalarType

        # Define our vector sizes
        N = 2**16

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float16, shape=(N, ))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float16, shape=(N, ))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float16, shape=(N, ))

        nest = Nest(shape=(N, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i] = A[i] + B[i]

        schedule = nest.create_schedule()
        plan = schedule.create_plan()

        package = Package()
        package_name = "test_vec_add_fp16"
        function = package.add(plan, args=(A, B, C), base_name=package_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(output_dir, ignore_errors=True)

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(name=package_name, format=self.PACKAGE_FORMAT, mode=self.PACKAGE_MODE, output_dir=output_dir)

            def vecadd_ref(a, b):
                return a + b

            Input0_test, Input1_test, Output_test = (
                np.random.random(p.shape).astype(np.float16) for p in function.args
            )
            Output_ref = vecadd_ref(Input0_test, Input1_test)

            v.check_correctness(
                function.name,
                before=(Input0_test, Input1_test, Output_test),
                after=(Input0_test, Input1_test, Output_ref)
            )


if __name__ == '__main__':
    unittest.main(verbosity=10)
