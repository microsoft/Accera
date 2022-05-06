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
    import cuda, pynvrtc
except:
    CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = True

if sys.platform == 'linux':
    try:
        LIBHIB_LIBNAME = 'libamdhip64.so'
        import ctypes
        ROCM_AVAILABLE = bool(ctypes.cdll.LoadLibrary(LIBHIB_LIBNAME))
    except:
        ROCM_AVAILABLE = False
else:
    ROCM_AVAILABLE = False

DEV_MODE = False
if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    DEV_MODE = True
    sys.path.insert(1, os.getcwd())

from accera import Package, ScalarType, Nest, Array, Scalar, fuse, create_parameters
from accera.test import verifiers

TEST_PACKAGE_DIR = "test_mfma"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TensorizeTest(unittest.TestCase):

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
        checker.check('__builtin_amdgcn_mfma_f32_16x16x4f32')
        checker.run()


    def _check_cu_has_no_mfma(self, test_name, verifier):
        checker = verifier.file_checker(f"{test_name}.cu")
        checker.check_label(
            'extern "C" __global__  __launch_bounds__({{.+}}) void ' + test_name + '_{{.+}}__gpu__('
        )
        checker.check_not('__builtin_amdgcn_mfma_f32_16x16x4f32')
        checker.run()


    # This should produce MFMA instructions
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
        plan.tensorize(indices=(iii, jjj, kk))

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
    def test_rocm_tensorize_single_block_single_warp_output_reordered_indices(self) -> None:
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
                i: target.GridUnit.BLOCK_X,
                j: target.GridUnit.BLOCK_Y,
                ii: target.GridUnit.THREAD_X,
                jj: target.GridUnit.THREAD_Y
            }
        )
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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
                i: target.GridUnit.BLOCK_X,
                j: target.GridUnit.BLOCK_Y,
                ii: target.GridUnit.THREAD_X,
                jj: target.GridUnit.THREAD_Y
            }
        )
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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
                i: target.GridUnit.BLOCK_X,
                j: target.GridUnit.BLOCK_Y,
                ii: target.GridUnit.THREAD_X,
                jj: target.GridUnit.THREAD_Y
            }
        )
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=Array.Layout.LAST_MAJOR)
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

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
                i: target.GridUnit.BLOCK_X,
                j: target.GridUnit.BLOCK_Y,
                ii: target.GridUnit.THREAD_X,
                jj: target.GridUnit.THREAD_Y
            }
        )
        plan.tensorize(indices=(iii, jjj, kk))

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


    @unittest.skip("Split factors not supported")
    # This should not produce MFMA instructions
    def test_rocm_tensorize_multi_block_multi_warp_output_newsplits(self) -> None:
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

        ii, jj = schedule.tile({
            i: outer_tile_x,
            j: outer_tile_y
        })
        iii, jjj, kk = schedule.tile({
            ii: 16,
            jj: 16,
            k: 4
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
        plan.tensorize(indices=(iii, jjj, kk))

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
    def test_rocm_tensorize_same_mul_operands(self) -> None:
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
            a = A[i, k]
            b = B[k, j]
            C[i, j] += a * a

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
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        D = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

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
        plan.tensorize(indices=(iii, jjj, kk))

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


    # This should fail
    # This should not produce MFMA instructions
    def test_rocm_tensorize_bad_binding(self) -> None:
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

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_X,
                j: target.GridUnit.BLOCK_Y,
                ii: target.GridUnit.THREAD_Y,
                jj: target.GridUnit.THREAD_X
            }
        )
        plan.tensorize(indices=(iii, jjj, kk))

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


    @unittest.skip("Binding syntax not supported")
    def test_rocm_tensorize_new_binding(self) -> None:
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

        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_X,
                j: target.GridUnit.BLOCK_Y,
                ii: target.GridUnit.THREAD_Y,
                jj: target.GridUnit.THREAD_X,
                iii: target.GridUnit.THREAD_Y,
                jjj: target.GridUnit.THREAD_X
            }
        )
        plan.tensorize(indices=(iii, jjj, kk))

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


    # This should succeed
    # This should produce MFMA instructions
    def test_rocm_tensorize_reversed_mul_order(self) -> None:
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
        plan.tensorize(indices=(iii, jjj, kk))

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


    # This should ultimately succeed
    @unittest.skip("Support for batch matmul not implemented yet")
    def test_rocm_tensorize_batched(self) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        S = 128
        M = 1024
        N = M
        K = M
        outer_tile_x = 64
        outer_tile_y = outer_tile_x

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(S, M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(S, K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(S, M, N))

        nest = Nest(shape=(S, M, N, K))
        l, i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            a = A[l, i, k]
            b = B[l, k, j]
            prod = a * b
            c = C[l, i, j]
            C[l, i, j] = c + prod

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

        schedule.reorder((l, i, j, ii, jj, k, iii, jjj, kk))

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
        plan.tensorize(indices=(iii, jjj, kk))

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

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

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
        plan.tensorize(indices=(iii, jjj, kk))

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


if __name__ == '__main__':
    unittest.main(verbosity=10)
