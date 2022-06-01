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

from accera._lang_python._lang import _MMAShape, _MMASchedulingPolicy, _MemorySpace
from accera.test import verifiers
from accera import Array, Nest, Package, ScalarType, Target, Constants
from accera.Targets import GridUnits

TEST_PACKAGE_DIR = "test_mfma"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
        checker.check('__builtin_amdgcn_mfma_')
        checker.run()


    def _check_cu_has_no_mfma(self, test_name, verifier):
        checker = verifier.file_checker(f"{test_name}.cu")
        checker.check_label(
            'extern "C" __global__  __launch_bounds__({{.+}}) void ' + test_name + '_{{.+}}__gpu__('
        )
        checker.check_not('__builtin_amdgcn_mfma_')
        checker.run()

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
                print("Verifying...")

                # Create the arrays with the appropriate layout
                A_test, B_test, C_test = (np.ndarray(p.shape, dtype=np.dtype(p.element_type.name), order=p.requested_layout.to_numpy_order()) for p in function.requested_args)

                # Create all the random input data
                A_test_data, B_test_data, C_test_data = (np.random.random(p.shape).astype(np.dtype(p.element_type.name)) for p in function.requested_args)

                # Assign the default-ordered input data to the appropriately-ordered arrays
                A_test[:] = A_test_data
                B_test[:] = B_test_data
                C_test[:] = C_test_data

                C_ref = C_test + A_test @ B_test

                v.check_correctness(function.name, before=(A_test, B_test, C_test), after=(A_test, B_test, C_ref), tolerance=tolerance)

            # apply optional file checks
            if file_check_fn:
                file_check_fn(v)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
            j: outer_tile_y,
        })

        iii, jjj, kk = schedule.tile({
            ii: 4,
            jj: 16,
            k: 64
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

        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M64xN64xK1_B4, num_total_passes=64)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4)

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


    def _cuda_tensorize(self, M, N, K, outer_tile_x, outer_tile_y, mfma_tile, mma_shape, num_total_passes, tolerance=1e-5, intype=ScalarType.float16, outtype=ScalarType.float16, num_fused_passes=None, scheduling_policy=_MMASchedulingPolicy.PASS_ORDER, verify=True) -> None:
        from accera import Target
        A = Array(role=Array.Role.INPUT, element_type=intype, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=intype, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=outtype, shape=(M, N))

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
            ii: mfma_tile[0],
            jj: mfma_tile[1],
            k: mfma_tile[2]
        })

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target=Target(Target.Model.NVIDIA_RTX_A6000)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_Y,
                j: target.GridUnit.BLOCK_X,
                ii: target.GridUnit.THREAD_Y,
                jj: target.GridUnit.THREAD_X,
            }
        )
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=mma_shape, num_total_passes=num_total_passes, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy)

        package = Package()
        test_name = "test_cuda_tensorize"
        test_name += f"_{M}x{N}x{K}"
        test_name += "_fp32" if intype == ScalarType.float32 else "_fp16"
        test_name += "_fp32_t" if outtype == ScalarType.float32 else "_fp16_t"
        test_name += str(mfma_tile[2]) + "_w"
        test_name += str(mfma_tile[1])
        if num_fused_passes is not None:
            test_name += "_p" + str(num_fused_passes)
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

    def test_cuda_tensorize_16x16x16_fp16_fp16_t16_w2(self) -> None:
        self._cuda_tensorize(16, 16, 16, 16, 16, (4, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3)

    def test_cuda_tensorize_16x16x32_fp16_fp16_t16_w2(self) -> None:
        self._cuda_tensorize(16, 16, 32, 16, 16, (4, 2, 32), _MMAShape.M16xN16xK16_B1, 2, 1e-2)

    def test_cuda_tensorize_16x16x384_fp16_fp32_t16_w2_p4(self) -> None:
        self._cuda_tensorize(16, 16, 384, 16, 16, (4, 2, 192), _MMAShape.M16xN16xK16_B1, 12, 1e-2, ScalarType.float16, ScalarType.float32, 4)

    def test_cuda_tensorize_64x64x64_fp16_fp16_t16_w2(self) -> None:
        self._cuda_tensorize(64, 64, 64, 64, 64, (4, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-2)

    def test_cuda_tensorize_1024x1024x1024_fp16_fp16_t16_w2(self) -> None:
        self._cuda_tensorize(1024, 1024, 1024, 64, 64, (4, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-2)

    def test_cuda_tensorize_1024x1024x2048_fp16_fp32_t16_w2_p8(self) -> None:
        self._cuda_tensorize(1024, 1024, 2048, 64, 64, (4, 2, 512), _MMAShape.M16xN16xK16_B1, 32, 1e-2, ScalarType.float16, ScalarType.float32, 8, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_cuda_tensorize_16x16x16_fp16_fp32_t16_w2(self) -> None:
        self._cuda_tensorize(16, 16, 16, 16, 16, (4, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3, ScalarType.float16, ScalarType.float32)


    def _cuda_cache_tensorize(self, M, N, K, outer_tile_m, outer_tile_n, outer_tile_k, test_name,
                              tensorize=True, tensor_splits=[4, 2, 16], mma_shape=_MMAShape.M16xN16xK16_B1, num_total_passes=1, cache=False, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              double_buffer=False, double_buffer_location=Constants.AUTO, vectorize=False,
                              scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                              bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                              array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              element_type=ScalarType.float16) -> None:
        from accera import Array, Nest, Package, Target

        A = Array(role=Array.Role.INPUT, element_type=element_type, shape=(M, K), layout=array_layouts[0])
        B = Array(role=Array.Role.INPUT, element_type=element_type, shape=(K, N), layout=array_layouts[1])
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=element_type, shape=(M, N), layout=array_layouts[2])

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile({
            i: outer_tile_m,
            j: outer_tile_n,
            k: outer_tile_k
        })

        if tensor_splits:
            iii, jjj, kkk = schedule.tile({
                ii: tensor_splits[0],
                jj: tensor_splits[1],
                kk: tensor_splits[2]
            })
            schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk)
        else:
            schedule.reorder(i, j, k, ii, jj, kk)


        target = Target(Target.Model.NVIDIA_RTX_A6000)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: bind_order[0],
                j: bind_order[1],
                ii: bind_order[2],
                jj: bind_order[3]
            }
        )

        if tensorize:
            plan.tensorize(indices=(iii, jjj, kkk), mma_shape=mma_shape, num_total_passes=num_total_passes, scheduling_policy=scheduling_policy)

        if cache:
            plan.cache(
                A, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[0]
            )
            plan.cache(
                B, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[1]
            )

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=CUDA_AVAILABLE,
            tolerance=1e-5 if element_type == ScalarType.float32 else 1e-2,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR_VERBOSE | Package.Format.DEFAULT
        )

    def test_cuda_cache_tensorize(self) -> None:
        self._cuda_cache_tensorize(M=1024, N=1024, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_cuda_cache_tensorize")

    # These are unexpected successes since it produces code without caching and succeeds, hence commenting. Enable tests once caching is working.
    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_cache_double_buffering_tensorize(self) -> None:
    #     self._cuda_cache_tensorize(M=1024, N=1024, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
    #                                 test_name="test_cuda_cache_double_buffering_tensorize", tensorize=True, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_cuda_non_square_simple(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_non_square_simple", tensorize=False, cache=False, element_type=ScalarType.float32)

    def test_cuda_non_square_last_major_inputs(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_inputs", tensorize=False, tensor_splits=None,
                                   cache=False, vectorize=False, element_type=ScalarType.float32,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_cuda_non_square_last_major_output(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_output", tensorize=False, tensor_splits=None,
                                   cache=False, vectorize=False, element_type=ScalarType.float32,
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_non_square_last_major_inputs_output(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_non_square_last_major_inputs_output", tensorize=False, tensor_splits=None,
                                   cache=False, vectorize=False, element_type=ScalarType.float32, 
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_inputs(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_tensorize_non_square_last_major_inputs", tensorize=True,
                                   cache=False, vectorize=False,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_output(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_tensorize_non_square_last_major_output", tensorize=True,
                                   cache=False, vectorize=False,
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square_last_major_inputs_output(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_cuda_tensorize_non_square_last_major_inputs_output", tensorize=True,
                                   cache=False, vectorize=False,
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_cuda_tensorize_non_square(self) -> None:
        self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_cuda_tensorize_non_square", tensorize=True, cache=False)

    # These are unexpected successes since it produces code without caching and succeeds, hence commenting. Enable tests once caching is working.
    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_cache_non_square(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
    #                                 test_name="test_cuda_cache_non_square", tensorize=False)

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_cache_double_buffering_non_square(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
    #                                 test_name="test_cuda_cache_double_buffering_non_square", tensorize=False, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_cache_double_buffering_tensorize_non_square(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
    #                                 test_name="test_cuda_cache_double_buffering_tensorize_non_square", tensorize=True, cache=True,
    #                                 double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)


    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_double_buffering_tensorize_non_square(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
    #                                 test_name="test_cuda_vectorized_cache_double_buffering_tensorize_non_square", tensorize=True, cache=True,
    #                                 double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True)

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_double_buffering_tensorize_non_square_blockorder(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
    #                                 test_name="test_cuda_vectorized_cache_double_buffering_tensorize_non_square_blockorder", tensorize=True, cache=True,
    #                                 double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square", tensorize=False, tensor_splits=None,
    #                                cache=True, vectorize=True,
    #                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_last_major_inputs(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square_last_major_inputs", tensorize=False, tensor_splits=None,
    #                                cache=True, vectorize=True,
    #                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
    #                                array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_last_major_output(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square_last_major_output", tensorize=False, tensor_splits=None,
    #                                cache=True, vectorize=True,
    #                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
    #                                array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_last_major_inputs_output(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square_last_major_inputs_output", tensorize=False, tensor_splits=None,
    #                                cache=True, vectorize=True,
    #                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
    #                                array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_small_tiles(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=32,
    #                                test_name="test_cuda_vectorized_cache_non_square_small_tiles", tensorize=False, tensor_splits=None,
    #                                cache=True, vectorize=True,
    #                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_transpose(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square_transpose", tensorize=False, tensor_splits=None,
    #                                cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR],
    #                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_double_buffer(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square_double_buffer", tensorize=False, tensor_splits=None,
    #                                cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
    #                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_double_buffer_small_tiles(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=32,
    #                                test_name="test_cuda_vectorized_cache_non_square_double_buffer_small_tiles", tensorize=False, tensor_splits=None,
    #                                cache=True, vectorize=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
    #                                bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    # @unittest.skip("Caching not yet implemented for CUDA")
    # def test_cuda_vectorized_cache_non_square_double_buffer_transpose(self) -> None:
    #     self._cuda_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
    #                                test_name="test_cuda_vectorized_cache_non_square_double_buffer_transpose", tensorize=False, tensor_splits=None,
    #                                cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR],
    #                                double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
    #                                vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])


    def _rocm_tensorize(self, M, N, K, outer_tile_x, outer_tile_y, mfma_tile, mma_shape, num_total_passes, tolerance=1e-5, intype=ScalarType.float32, outtype=ScalarType.float32, use_static_offsets=False, num_fused_passes=None, scheduling_policy=_MMASchedulingPolicy.PASS_ORDER, verify=True) -> None:
        from accera import Target
        A = Array(role=Array.Role.INPUT, element_type=intype, shape=(M, K))
        B = Array(role=Array.Role.INPUT, element_type=intype, shape=(K, N))
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=outtype, shape=(M, N))

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
            ii: mfma_tile[0],
            jj: mfma_tile[1],
            k: mfma_tile[2]
        })

        schedule.reorder((i, j, ii, jj, k, iii, jjj, kk))

        target=Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: target.GridUnit.BLOCK_Y,
                j: target.GridUnit.BLOCK_X,
                ii: target.GridUnit.THREAD_Y,
                jj: target.GridUnit.THREAD_X,
            }
        )
        plan.tensorize(indices=(iii, jjj, kk), mma_shape=mma_shape, num_total_passes=num_total_passes, use_static_offsets=use_static_offsets, num_fused_passes=num_fused_passes, scheduling_policy=scheduling_policy)

        package = Package()
        test_name = "test_rocm_tensorize"
        test_name += f"_{M}x{N}x{K}"
        test_name += "_fp32" if intype == ScalarType.float32 else "_fp16"
        test_name += "_fp32_t" if outtype == ScalarType.float32 else "_fp16_t"
        test_name += str(mfma_tile[2]) + "_w"
        test_name += str(mfma_tile[1])
        if use_static_offsets:
            test_name += "_tensormap"
        if num_fused_passes is not None:
            test_name += "_p" + str(num_fused_passes)
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        def file_check_fn(v):
            checker = v.file_checker(f"{test_name}.cu")
            checker.check("constexpr int8_t threadOffsetsMFMA")
            checker.run()

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            file_check_fn=file_check_fn if use_static_offsets else None,
            check_correctness=ROCM_AVAILABLE and verify,
            tolerance=tolerance,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.DEFAULT | Package.Format.MLIR
        )

    def test_rocm_tensorize_16x16x16_fp32_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4)

    def test_rocm_tensorize_16x16x16_fp32_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4)

    def test_rocm_tensorize_32x32x32_fp32_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK2_B1, 16)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w32(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(960, 1024, 1024, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 64)
        # self._rocm_tensorize(64, 64, 64, 64, 64, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 64)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 4, 32), _MMAShape.M32xN32xK2_B1, 16)

    def test_rocm_tensorize_1024x1024x1024_fp32_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4)

    def test_rocm_tensorize_2048x2048x2048_fp32_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 4, 32), _MMAShape.M32xN32xK2_B1, 16)

    def test_rocm_tensorize_2048x2048x2048_fp32_fp32_t64_w32(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64)

    def test_rocm_tensorize_2048x2048x2048_fp32_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 64)

    def test_rocm_tensorize_16x16x16_fp16_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_32x32x32_fp16_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t64_w32(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp32_t64_w32(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_16x16x16_fp16_fp16_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_32x32x32_fp16_fp16_t32_w4(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-3, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp16_t64_w32(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-3, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp16_t64_w16(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-3, ScalarType.float16, ScalarType.float16)

    # TODO: This requires tolerance to be set higher than the other tests (verify discrepancies)
    def test_rocm_tensorize_64x64x64_fp16_fp16_t16_w2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-2, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_64x64x64_fp16_fp16_t32_w4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-3, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_t16_w2(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-2, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t32_w4(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-2, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w32(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-2, ScalarType.float16, ScalarType.float16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w16(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-2, ScalarType.float16, ScalarType.float16)


    # Testing precomputed index map optimization
    def test_rocm_tensorize_16x16x16_fp32_fp32_t16_w2_tensormap(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4, 1e-2, ScalarType.float32, ScalarType.float32, True)

    def test_rocm_tensorize_32x32x32_fp32_fp32_t32_w4_tensormap(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK2_B1, 16, 1e-2, ScalarType.float32, ScalarType.float32, True)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w32_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64, 1e-2, ScalarType.float32, ScalarType.float32, True)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w16_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 64, 1e-2, ScalarType.float32, ScalarType.float32, True)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t16_w2_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-3, ScalarType.float16, ScalarType.float32, True)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t32_w4_tensormap(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-3, ScalarType.float16, ScalarType.float32, True)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_t16_w2_tensormap(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-2, ScalarType.float16, ScalarType.float16, True)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t32_w4_tensormap(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-2, ScalarType.float16, ScalarType.float16, True)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp32_t64_w32_tensormap(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-3, ScalarType.float16, ScalarType.float32, True)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w16_tensormap(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-2, ScalarType.float16, ScalarType.float16, True)

    # Testing tunable register usage
    def test_rocm_tensorize_16x16x16_fp32_fp32_t16_w2_tensormap_p1(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4, 1e-2, ScalarType.float32, ScalarType.float32, True, 1)

    def test_rocm_tensorize_16x16x16_fp32_fp32_t16_w2_p2(self) -> None:
        self._rocm_tensorize(16, 16, 16, 16, 16, (2, 2, 16), _MMAShape.M16xN16xK4_B1, 4, 1e-2, ScalarType.float32, ScalarType.float32, False, 2, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_1024x1024x1024_fp16_fp16_t16_w2_p1(self) -> None:
        self._rocm_tensorize(1024, 1024, 1024, 64, 64, (2, 2, 16), _MMAShape.M16xN16xK16_B1, 1, 1e-2, ScalarType.float16, ScalarType.float16, False, 1)

    def test_rocm_tensorize_32x32x32_fp32_fp32_t32_w4_p1(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK2_B1, 16, 1e-2, ScalarType.float32, ScalarType.float32, False, 1)

    def test_rocm_tensorize_32x32x32_fp32_fp32_t32_w4_tensormap_p16(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK2_B1, 16, 1e-2, ScalarType.float32, ScalarType.float32, True, 16, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_32x32x32_fp16_fp32_t32_w4_p2(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-2, ScalarType.float16, ScalarType.float32, False, 2)

    def test_rocm_tensorize_32x32x32_fp16_fp16_t32_w4_tensormap_p4(self) -> None:
        self._rocm_tensorize(32, 32, 32, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 4, 1e-2, ScalarType.float16, ScalarType.float16, True, 4)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w32_tensormap_p1(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64, 1e-2, ScalarType.float32, ScalarType.float32, True, 1, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w32_p4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64, 1e-2, ScalarType.float32, ScalarType.float32, False, 4)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w32_tensormap_p16(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64, 1e-2, ScalarType.float32, ScalarType.float32, True, 16, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp32_fp32_t64_w32_p64(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 64, 1e-2, ScalarType.float32, ScalarType.float32, False, 64)

    def test_rocm_tensorize_64x64x64_fp16_fp16_t64_w32_tensormap_p2(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-2, ScalarType.float16, ScalarType.float16, True, 2)

    def test_rocm_tensorize_64x64x64_fp16_fp16_t64_w32_p4(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-2, ScalarType.float16, ScalarType.float16, False, 4, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t64_w32_tensormap_p8(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-2, ScalarType.float16, ScalarType.float32, True, 8, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_64x64x64_fp16_fp32_t64_w32_p16(self) -> None:
        self._rocm_tensorize(64, 64, 64, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 16, 1e-2, ScalarType.float16, ScalarType.float32, False, 16)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w16_tensormap_p1(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-2, ScalarType.float16, ScalarType.float16, True, 1)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w16_p2(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-2, ScalarType.float16, ScalarType.float16, False, 2, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w16_tensormap_p4(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-2, ScalarType.float16, ScalarType.float16, True, 4, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_2048x2048x2048_fp16_fp16_t64_w16_p8(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 16, 1e-2, ScalarType.float16, ScalarType.float16, False, 8)

    def test_rocm_tensorize_2048x2048x2048_fp32_fp32_t64_w16_tensormap_p32(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 64, 1e-2, ScalarType.float32, ScalarType.float32, True, 32, _MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_tensorize_2048x2048x2048_fp32_fp32_t64_w16_p64(self) -> None:
        self._rocm_tensorize(2048, 2048, 2048, 128, 128, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 64, 1e-2, ScalarType.float32, ScalarType.float32, False, 64, _MMASchedulingPolicy.BLOCK_ORDER)

    # Test arbitrary K
    def test_rocm_tensorize_16x16x4_fp32_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 4, 16, 16, (2, 2, 4), _MMAShape.M16xN16xK4_B1, 1)

    def test_rocm_tensorize_16x16x20_fp32_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 20, 16, 16, (2, 2, 20), _MMAShape.M16xN16xK4_B1, 5)

    def test_rocm_tensorize_16x16x1264_fp16_fp32_t16_w2(self) -> None:
        self._rocm_tensorize(16, 16, 1264, 16, 16, (2, 2, 1264), _MMAShape.M16xN16xK16_B1, 79, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_32x32x48_fp32_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(32, 32, 48, 32, 32, (4, 4, 48), _MMAShape.M32xN32xK2_B1, 24)

    def test_rocm_tensorize_32x32x24_fp16_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(32, 32, 24, 32, 32, (4, 4, 32), _MMAShape.M32xN32xK8_B1, 3, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_32x32x136_fp16_fp32_t32_w4(self) -> None:
        self._rocm_tensorize(32, 32, 136, 32, 32, (4, 4, 136), _MMAShape.M32xN32xK8_B1, 17, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_64x64x1_fp32_fp32_t64_w32(self) -> None:
        self._rocm_tensorize(64, 64, 1, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK1_B2, 1)

    def test_rocm_tensorize_64x64x44_fp16_fp32_t64_w32(self) -> None:
        self._rocm_tensorize(64, 64, 44, 64, 64, (2, 32, 64), _MMAShape.M64xN64xK4_B2, 11, 1e-3, ScalarType.float16)

    def test_rocm_tensorize_64x64x3_fp32_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(64, 64, 3, 64, 64, (4, 16, 64), _MMAShape.M64xN64xK1_B4, 3)

    def test_rocm_tensorize_64x64x93_fp32_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(64, 64, 93, 64, 64, (4, 16, 93), _MMAShape.M64xN64xK1_B4, 93)

    def test_rocm_tensorize_64x64x20_fp16_fp32_t64_w16(self) -> None:
        self._rocm_tensorize(64, 64, 20, 64, 64, (4, 16, 64), _MMAShape.M64xN64xK4_B4, 5, 1e-3, ScalarType.float16)

    @unittest.skip("the hardware does not support the requested tensorcore shape")
    def test_rocm_tensorize_invalid_shape_output(self) -> None:
        self._rocm_tensorize(256, 256, 256, 64, 64, (64, 64, 64),
                             "test_rocm_tensorize_invalid_shape_output", False)


    def _rocm_cache_tensorize(self, M, N, K, outer_tile_m, outer_tile_n, outer_tile_k, test_name,
                              tensorize=True, tensor_splits=[2, 2, 16], mma_shape=_MMAShape.M16xN16xK4_B1, num_total_passes=4, cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR],
                              double_buffer=False, double_buffer_location=Constants.AUTO, vectorize=False, use_static_offsets=False,
                              scheduling_policy=_MMASchedulingPolicy.PASS_ORDER,
                              bind_order=[GridUnits.BLOCK_Y, GridUnits.BLOCK_X, GridUnits.THREAD_Y, GridUnits.THREAD_X],
                              array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR]) -> None:
        from accera import Array, Nest, Package, ScalarType, Target

        A = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(M, K), layout=array_layouts[0])
        B = Array(role=Array.Role.INPUT, element_type=ScalarType.float32, shape=(K, N), layout=array_layouts[1])
        C = Array(role=Array.Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N), layout=array_layouts[2])

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii, jj, kk = schedule.tile({
            i: outer_tile_m,
            j: outer_tile_n,
            k: outer_tile_k
        })

        if tensor_splits:
            iii, jjj, kkk = schedule.tile({
                ii: tensor_splits[0],
                jj: tensor_splits[1],
                kk: tensor_splits[2]
            })
            schedule.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk)
        else:
            schedule.reorder(i, j, k, ii, jj, kk)


        target = Target(Target.Model.AMD_MI100)
        plan = schedule.create_plan(target=target)
        plan.bind(
            mapping={
                i: bind_order[0],
                j: bind_order[1],
                ii: bind_order[2],
                jj: bind_order[3]
            }
        )

        if tensorize:
            plan.tensorize(indices=(iii, jjj, kkk), mma_shape=mma_shape, num_total_passes=num_total_passes, use_static_offsets=use_static_offsets, scheduling_policy=scheduling_policy)

        if cache:
            plan.cache(
                A, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[0]
            )
            plan.cache(
                B, index=ii, double_buffer=double_buffer, double_buffer_location=double_buffer_location, vectorize=vectorize, location=target.MemorySpace.SHARED, layout=cache_layouts[1]
            )

        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        self._verify_matrix_multiplication_function(
            function,
            package,
            test_name,
            check_correctness=ROCM_AVAILABLE,
            file_list=[f"{test_name}.cu", f"{test_name}.hat"],
            package_format=Package.Format.MLIR_VERBOSE | Package.Format.DEFAULT
        )

    def test_rocm_cache_tensorize(self) -> None:
        self._rocm_cache_tensorize(M=1024, N=1024, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_cache_tensorize")

    def test_rocm_cache_double_buffering_tensorize(self) -> None:
        self._rocm_cache_tensorize(M=1024, N=1024, K=1024, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_cache_double_buffering_tensorize", tensorize=True, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_rocm_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_non_square", tensorize=False, cache=False)

    def test_rocm_non_square_last_major_inputs(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_non_square_last_major_inputs", tensorize=False, tensor_splits=None,
                                   cache=False, vectorize=False,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_rocm_non_square_last_major_output(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_non_square_last_major_output", tensorize=False, tensor_splits=None,
                                   cache=False, vectorize=False,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_non_square_last_major_inputs_output(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_non_square_last_major_inputs_output", tensorize=False, tensor_splits=None,
                                   cache=False, vectorize=False,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_tensorize_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_tensorize_non_square", tensorize=True, cache=False)

    def test_rocm_cache_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_cache_non_square", tensorize=False)

    def test_rocm_cache_double_buffering_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_cache_double_buffering_non_square", tensorize=False, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_rocm_cache_double_buffering_tensorize_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_cache_double_buffering_tensorize_non_square", tensorize=True, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square", tensorize=True, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True)

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64, test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_blockorder", tensorize=True, cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, scheduling_policy=_MMASchedulingPolicy.BLOCK_ORDER)

    def test_rocm_vectorized_cache_non_square(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square", tensorize=False, tensor_splits=None,
                                   cache=True, vectorize=True,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_last_major_inputs(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_last_major_inputs", tensorize=False, tensor_splits=None,
                                   cache=True, vectorize=True,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.FIRST_MAJOR])

    def test_rocm_vectorized_cache_non_square_last_major_output(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_last_major_output", tensorize=False, tensor_splits=None,
                                   cache=True, vectorize=True,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_vectorized_cache_non_square_last_major_inputs_output(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_last_major_inputs_output", tensorize=False, tensor_splits=None,
                                   cache=True, vectorize=True,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y],
                                   array_layouts=[Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR, Array.Layout.LAST_MAJOR])

    def test_rocm_vectorized_cache_non_square_small_tiles(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=32,
                                   test_name="test_rocm_vectorized_cache_non_square_small_tiles", tensorize=False, tensor_splits=None,
                                   cache=True, vectorize=True,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_transpose(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_transpose", tensorize=False, tensor_splits=None,
                                   cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR],
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_double_buffer(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_double_buffer", tensorize=False, tensor_splits=None,
                                   cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_double_buffer_small_tiles(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=32,
                                   test_name="test_rocm_vectorized_cache_non_square_double_buffer_small_tiles", tensorize=False, tensor_splits=None,
                                   cache=True, vectorize=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_non_square_double_buffer_transpose(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=16, outer_tile_n=16, outer_tile_k=128,
                                   test_name="test_rocm_vectorized_cache_non_square_double_buffer_transpose", tensorize=False, tensor_splits=None,
                                   cache=True, cache_layouts=[Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR],
                                   double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE,
                                   vectorize=True, bind_order=[GridUnits.BLOCK_X, GridUnits.BLOCK_Y, GridUnits.THREAD_X, GridUnits.THREAD_Y])

    def test_rocm_vectorized_cache_double_buffering_tensorize_non_square_tensormap(self) -> None:
        self._rocm_cache_tensorize(M=2560, N=1536, K=2048, outer_tile_m=64, outer_tile_n=64, outer_tile_k=64,
                                    test_name="test_rocm_vectorized_cache_double_buffering_tensorize_non_square_tensormap", tensorize=True,
                                    cache=True, double_buffer=True, double_buffer_location=_MemorySpace.PRIVATE, vectorize=True, use_static_offsets=True)

if __name__ == '__main__':
    unittest.main(verbosity=10)
