#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# Tip: to run a particular test / set of tests:
# python -m unittest discover -k "test_input_array" path_to_accera/test dsl_tests.py
# python -m unittest discover -k "DSLTest_01" path_to_accera/test dsl_tests.py

import logging
import sys
import unittest
import os
import pathlib
import shutil
import numpy as np
from typing import Tuple

DEV_MODE = False
if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    DEV_MODE = True
    sys.path.insert(1, os.getcwd())

from accera import ScalarType, Array, Function, Nest, Target, Package, algorithms, cast, AllocateFlags, Role
from accera.test import verifiers
from accera.test.test_utils import expectedFailure, FailedReason
from accera._lang_python._lang import Dimension

INTERNAL_FUNCTION_OPTS = {
    "no_inline_into": True,
    "public": False
}

TEST_MODE = Package.Mode.DEBUG if DEV_MODE else Package.Mode.RELEASE
TEST_FORMAT = Package.Format.MLIR_DYNAMIC if DEV_MODE else Package.Format.HAT_DYNAMIC
TEST_PACKAGE_DIR = "test_acccgen"

# Groups of types commonly used for tests
INT_TYPES = [
    ScalarType.int8,
    ScalarType.int16,
    ScalarType.int32,
    ScalarType.int64,
    ScalarType.uint8,
    ScalarType.uint16,
    ScalarType.uint32,
    ScalarType.uint64,
]
FLOAT_TYPES = [ScalarType.float16, ScalarType.float32, ScalarType.float64]

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
os.environ["OMP_DISPLAY_AFFINITY"] = "TRUE"

# TODO: Remove all @expectedFailure decorators as implementation converges with spec


class DSLTest_01Arrays(unittest.TestCase):
    def _verify_nest(self, nest, args: Tuple[Array], package_name, correctness_check_values=None) -> None:

        # create a HAT package and add the function to it
        package = Package()
        function = package.add(nest, args, base_name=package_name)
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        # build the HAT package
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_input_array(self) -> None:
        A = Array(shape=(10, 20), role=Role.INPUT, element_type=ScalarType.float32)
        self.assertIsNotNone(A)

    def test_input_array_standard_layout(self) -> None:
        A = Array(shape=(10, 20), role=Role.INPUT, layout=Array.Layout.LAST_MAJOR)
        # A = Array(shape=(10, 20), layout=Array.Layout.LAST_MAJOR, role=Role.INPUT, element_type=ScalarType.float32)
        self.assertIsNotNone(A)

    def test_input_array_dimension_layout(self) -> None:
        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, 20),
            layout=(1, 10),
        )
        self.assertIsNotNone(A)

        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, 20),
            layout=(10, 1),
        )
        self.assertIsNotNone(A)

        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, ),
            layout=(1, ),
        )
        self.assertIsNotNone(A)

        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, 20, 50),
            layout=(1, 10, 200),
        )
        self.assertIsNotNone(A)

        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, 20, 50),
            layout=(200, 10, 1),
        )
        self.assertIsNotNone(A)

        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, 20, 50),
            layout=(1, 200, 10),
        )
        self.assertIsNotNone(A)

        A = Array(
            role=Role.INPUT,
            element_type=ScalarType.float32,
            shape=(10, 20, 50),
            layout=(10, 200, 1),
        )
        self.assertIsNotNone(A)

    def test_input_array_infinite_major_dimension(self) -> None:
        from accera import inf

        with self.assertRaises(ValueError):
            Array(
                role=Role.INPUT_OUTPUT,
                element_type=ScalarType.float32,
                shape=(inf, inf),
            )

        A = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(10, inf),
        )
        self.assertIsNotNone(A)
        self.assertEqual(A.shape[1], inf)

        nest = Nest(shape=(10, 16))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] += A[i, j]

        package = Package()
        package.add(nest, (A, ), base_name="inf_test")
        self.assertEqual(A.shape[1], 16)

        package_name = "input_array_inf_test"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_input_output_array(self) -> None:
        A = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(10, 20),
        )
        self.assertIsNotNone(A)

    def test_const_array(self) -> None:
        for dt in [
                bool,    # np.bool is deprecated in favor of bool
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
        ]:
            D = np.ones((128, 256), dtype=dt)
            A = Array(role=Role.CONST, data=D)
            self.assertIsNotNone(A)

    def test_const_array_type_layout(self) -> None:
        D = np.ones((128, 256), dtype=np.float64)
        for t in [ScalarType.bool] + INT_TYPES + FLOAT_TYPES:
            A = Array(
                role=Role.CONST,
                element_type=t,
                layout=Array.Layout.LAST_MAJOR,
                data=D,
            )
            self.assertIsNotNone(A)

    def test_temp_array(self) -> None:
        A = Array(
            role=Role.TEMP,
            element_type=ScalarType.float32,
            layout=Array.Layout.LAST_MAJOR,
            shape=(10, 20),
        )
        self.assertIsNotNone(A)
        B = Array(
            role=Role.TEMP,
            element_type=ScalarType.float32,
            layout=Array.Layout.FIRST_MAJOR,
            shape=(10, 20),
        )
        self.assertIsNotNone(B)

    def test_temp_array_materialization_1(self) -> None:
        # Materializes (allocates) a TEMP array externally to an added function

        def make_test_fn(package, A, B, C):
            T = Array(role=Role.TEMP, element_type=A.element_type, shape=A.shape)

            nest = Nest(A.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                T[i, j] = A[i, j] + B[i, j]
                C[i, j] += T[i, j]**2.0

            return package.add(nest, args=(A, B, C))

        A = Array(shape=(256, 32), role=Role.INPUT)
        B = Array(shape=(256, 32), role=Role.INPUT)
        C = Array(shape=(256, 32), role=Role.INPUT_OUTPUT)

        package = Package()
        make_test_fn(package, A, B, C)
        package_name = "test_temp_array_materialization_1"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_temp_array_materialization_2(self) -> None:
        # Materializes (allocates) a TEMP array within an added function

        package = Package()
        A = Array(shape=(256, 32), role=Role.INPUT)
        B = Array(shape=(256, 32), role=Role.INPUT_OUTPUT)

        def make_init_function(package, A):
            nest = Nest(A.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                A[i, j] = 3.14

            return package.add(nest, args=(A, ))

        init_fn = make_init_function(package, B)

        def make_helper_function2(package, A, B):

            nest = Nest(A.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                B[i, j] += A[i, j] * 2.0

            return package.add(nest, args=(A, B))

        helper_fn2 = make_helper_function2(package, A, B)

        def test_fn(A, B):
            T = Array(role=Role.TEMP, element_type=A.element_type, shape=A.shape)
            init_fn(T)
            helper_fn2(T, B)
            helper_fn2(A, B)

        package.add(test_fn, args=(A, B))

        package_name = "test_temp_array_materialization_2"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

        def test_fn_wrong_role(A, B):
            T = Array(role=Role.INPUT_OUTPUT, element_type=A.element_type, shape=A.shape)
            init_fn(T)
            helper_fn2(T, B)
            helper_fn2(A, B)

        package.add(test_fn_wrong_role, args=(A, B))

        package_name = "test_temp_array_materialization_2_wrong_role"
        with self.assertRaises(ValueError):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
                fail_on_error=True,
            )

    def test_temp_array_materialization_3(self) -> None:
        # Materializes (allocates) a TEMP array within some nest iteration logic
        # *without* passing the array as a function argument

        package = Package()
        A = Array(shape=(256, 32), role=Role.INPUT_OUTPUT)
        B = Array(shape=(256, 32), role=Role.INPUT_OUTPUT)

        nest = Nest(A.shape)
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            T = Array(role=Role.TEMP, element_type=A.element_type, shape=(1, ))

            # TODO: inject via introspection if we need to support this scenario
            T._allocate()
            T = T._get_native_array()

            T[0] = B[i, j]
            B[i, j] += A[i, j] * 2.0
            A[i, j] = T[0]

        package.add(nest, args=(A, B))
        package_name = "test_temp_array_materialization_3"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_dynamic_temp_array(self) -> None:
        def make_test_fn(package, A, B, C, N):
            T = Array(role=Role.TEMP, element_type=A.element_type, shape=A.shape)

            nest = Nest(B.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                T[i, j] = A[i, j] + B[i, j]
                C[i, j] += T[i, j]**2.0

            return package.add(nest, args=(A, B, C, N))

        N = Dimension()
        A = Array(shape=(256, N), role=Role.INPUT)
        B = Array(shape=(256, 32), role=Role.INPUT)
        C = Array(shape=(256, 32), role=Role.INPUT_OUTPUT)

        package = Package()
        make_test_fn(package, A, B, C, N)
        package_name = "test_dynamic_temp_array"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=Package.Mode.RELEASE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_first_major_array_access(self) -> None:
        A = Array(shape=(256, 32), role=Role.INPUT, layout=Array.Layout.FIRST_MAJOR)

        nest = Nest(shape=(256, 32))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] = 5.0

        A_test = np.random.random((256, 32)).astype(np.float32)
        A_expected = np.ndarray((256, 32)).astype(np.float32)
        A_expected.fill(5.0)
        correctness_check_values = {
            "pre": (A_test, ),
            "post": (A_expected, )
        }
        self._verify_nest(
            nest,
            (A, ),
            "test_first_major_array_access",
            correctness_check_values=correctness_check_values,
        )

    def test_last_major_array_access(self) -> None:
        A = Array(shape=(256, 32), role=Role.INPUT, layout=Array.Layout.LAST_MAJOR)

        nest = Nest(shape=(256, 32))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] = 5.0

        A_test = np.random.random((256, 32)).astype(np.float32, order="F")
        A_expected = np.ndarray((256, 32)).astype(np.float32, order="F")
        A_expected.fill(5.0)
        correctness_check_values = {
            "pre": (A_test, ),
            "post": (A_expected, )
        }
        self._verify_nest(
            nest,
            (A, ),
            "test_last_major_array_access",
            correctness_check_values=correctness_check_values,
        )

    def test_array_value_type_cast(self) -> None:
        A = Array(
            shape=(256, 32),
            role=Role.INPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.float32,
        )
        B = Array(
            shape=(256, 32),
            role=Role.INPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.int32,
        )

        nest = Nest(shape=(256, 32))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] = 5    # implicit cast from int8 to float32
            B[i, j] = 10    # implicit cast from int8 to int32

        A_test = np.random.random((256, 32)).astype(np.float32)
        A_expected = np.ndarray((256, 32)).astype(np.float32)
        A_expected.fill(5.0)

        B_test = np.random.random((256, 32)).astype(np.int32)
        B_expected = np.ndarray((256, 32)).astype(np.int32)
        B_expected.fill(10)

        correctness_check_values = {
            "pre": (A_test, B_test),
            "post": (A_expected, B_expected),
        }
        self._verify_nest(
            nest,
            (A, B),
            "test_array_value_type_cast",
            correctness_check_values=correctness_check_values,
        )

    def test_array_vectorize_cast(self) -> None:
        A = Array(
            shape=(256, 32),
            role=Role.INPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.uint8,
        )
        B = Array(
            shape=(256, 32),
            role=Role.INPUT_OUTPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.int16,
        )

        nest = Nest(shape=(256, 32))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i, j] = A[i, j]

        sched = nest.create_schedule()
        ii = sched.split(i, 4)
        jj = sched.split(j, 16)
        sched.reorder(i, j, ii, jj)
        plan = sched.create_plan()
        plan.vectorize(ii)    # ii to in-place-unroll ii and vectorize jj

        A_test = np.random.random((256, 32)).astype(np.uint8)
        B_test = np.random.random((256, 32)).astype(np.int16)
        B_expected = np.ndarray((256, 32)).astype(np.int16)
        B_expected[:, :] = A_test[:, :]

        correctness_check_values = {
            "pre": (A_test, B_test),
            "post": (A_test, B_expected),
        }
        self._verify_nest(plan, (A, B), "test_array_vectorize_cast", correctness_check_values=correctness_check_values)

    def test_interleaved_vectorize_cast(self) -> None:
        shape = (64, 32, 8, 2)
        A = Array(
            shape=shape,
            role=Role.INPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.uint8,
        )
        B = Array(
            shape=shape,
            role=Role.INPUT_OUTPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.int16,
        )

        nest = Nest(shape=shape)
        i, j, k, l = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i, j, k, l] = A[i, j, k, l]

        sched = nest.create_schedule()
        plan = sched.create_plan()
        plan.vectorize(k)

        A_test = np.random.random(shape).astype(np.uint8)
        B_test = np.random.random(shape).astype(np.int16)
        B_expected = np.ndarray(shape).astype(np.int16)
        B_expected[:, :, :, :] = A_test[:, :, :, :]

        correctness_check_values = {
            "pre": (A_test, B_test),
            "post": (A_test, B_expected),
        }
        self._verify_nest(
            plan, (A, B), "test_interleaved_vectorize_cast", correctness_check_values=correctness_check_values
        )

    def test_interleaved_vectorize_store(self) -> None:
        M = 32
        N = 48
        M_tile = 2
        N_tile = 16
        input_shape = (M, N)
        output_shape = (M // M_tile, N // N_tile, N_tile, M_tile)
        A = Array(
            shape=input_shape,
            role=Role.INPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.uint8,
        )
        B = Array(
            shape=output_shape,
            role=Role.INPUT_OUTPUT,
            layout=Array.Layout.FIRST_MAJOR,
            element_type=ScalarType.uint8,
        )

        nest = Nest(shape=output_shape)
        i_outer, j_outer, j_inner, i_inner = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i_outer, j_outer, j_inner, i_inner] = A[i_outer * M_tile + i_inner, j_outer * N_tile + j_inner]

        sched = nest.create_schedule()
        plan = sched.create_plan()
        plan.vectorize(j_inner)

        A_test = np.random.random(input_shape).astype(np.uint8)
        B_test = np.random.random(output_shape).astype(np.uint8)
        B_expected = np.ndarray(output_shape).astype(np.uint8)
        for i_outer in range(0, M, M_tile):
            i_outer_idx = i_outer // M_tile
            for j_outer in range(0, N, N_tile):
                j_outer_idx = j_outer // N_tile
                for j_inner in range(0, N_tile):
                    full_j = j_outer + j_inner
                    for i_inner in range(0, M_tile):
                        full_i = i_outer + i_inner
                        B_expected[i_outer_idx, j_outer_idx, j_inner, i_inner] = A_test[full_i, full_j]

        correctness_check_values = {
            "pre": (A_test, B_test),
            "post": (A_test, B_expected),
        }
        self._verify_nest(
            plan, (A, B), "test_interleaved_vectorize_store", correctness_check_values=correctness_check_values
        )

    def test_subarray(self) -> None:
        package = Package()

        arr = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(256, 256),
        )
        arr0 = arr.sub_array(offsets=(0, 0), shape=(128, 128))
        self.assertEqual(arr0.shape, [128, 128])
        self.assertEqual(arr0.element_type, arr.element_type)
        print(arr0.layout)

        # add a function that utilizes a subarray layout
        def make_subarray_fn(arr0):
            nest = Nest(shape=arr0.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                arr0[i, j] += 1.0

            return package.add(nest, args=(arr0, ))

        subarray_fn = make_subarray_fn(arr0)

        # add a function that instantiates a subarray of the input array and calls the function above
        def main(arr):
            arr1 = arr.sub_array(offsets=(0, 0), shape=(128, 128))
            print(arr1.layout)
            self.assertEqual(arr0.layout, arr1.layout)
            subarray_fn(arr1)

        package.add(main, args=(arr, ))

        package_name = "test_subarray"

        # BUGBUG: starting from LLVM 14, sub_array in Package.Mode.DEBUG needs to link
        # against libmlir_c_runner_utils.so for the memrefCopy function
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=Package.Mode.RELEASE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_subarray_l2(self) -> None:
        package = Package()

        arr = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(256, 256),
        )
        arr0 = arr.sub_array(offsets=(0, 0), shape=(128, 128))
        self.assertEqual(arr0.shape, [128, 128])
        self.assertEqual(arr0.element_type, arr.element_type)
        arr00 = arr0.sub_array(offsets=(64, 64), shape=(64, 64))
        self.assertEqual(arr00.shape, [64, 64])
        self.assertEqual(arr00.element_type, arr0.element_type)

        # add a function that utilizes a subarray layout
        def make_fn(A):
            nest = Nest(shape=A.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                A[i, j] += 1.0

            return package.add(nest, args=(A, ))

        subarray_fn = make_fn(arr0)
        subarray_fn1 = make_fn(arr00)

        # add a function that instantiates a subarray of the input array and calls the function above
        def main(arr):
            arr1 = arr.sub_array(offsets=(0, 0), shape=(128, 128))
            arr11 = arr1.sub_array(offsets=(64, 64), shape=(64, 64))
            print(f"{arr1.layout}\n{arr11.layout}")
            self.assertEqual(arr0.layout, arr1.layout)
            self.assertEqual(arr00.layout, arr11.layout)
            subarray_fn(arr1)
            subarray_fn1(arr11)

        package.add(main, args=(arr, ))

        # BUGBUG: starting from LLVM 14, sub_array in Package.Mode.DEBUG needs to link
        # against libmlir_c_runner_utils.so for the memrefCopy function
        package_name = "test_subarray_l2"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=Package.Mode.RELEASE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def _verify_helper(self, package, test_name, function_name=None, correctness_check_values=None) -> None:
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name
        with verifiers.VerifyPackage(self, test_name, output_dir) as v:
            shutil.rmtree(output_dir, ignore_errors=True)
            package.build(test_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if function_name and correctness_check_values:
                v.check_correctness(
                    function_name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_runtimesizes_vector_add(self) -> None:
        N = Dimension()

        A = Array(shape=(N, ), element_type=ScalarType.float32, role=Role.INPUT)
        B = Array(shape=(N, ), element_type=ScalarType.float32, role=Role.INPUT_OUTPUT)

        nest = Nest((N, ))

        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i] += A[i]

        package = Package()

        N_test = np.int64(128)
        A_test = np.random.random((N_test, )).astype(np.float32)
        B_test = np.random.random((N_test, )).astype(np.float32)
        correctness_check_values = {
            "pre": [N_test, A_test, B_test],
            "post": [N_test, A_test, B_test + A_test],
        }

        test_name = "test_runtimesizes_vector_add"
        function = package.add(nest, args=(N, A, B), base_name="test_runtimesizes_vector_add")
        self._verify_helper(package, test_name, function.name, correctness_check_values)

    def _simple_runtimesize_loopnest_common(self, name, splits=[]) -> None:
        M = Dimension()

        A = Array(shape=(M, ), element_type=ScalarType.float32, role=Role.INPUT)

        nest = Nest((M, ))

        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i] = 0.0

        sched = nest.create_schedule()
        current_inner_index = i
        for split in splits:
            current_inner_index = sched.split(current_inner_index, split)

        package = Package()
        package.add(sched, args=(A, M), base_name=name)
        self._verify_helper(package, name)

    def test_runtimesizes_simple(self) -> None:
        self._simple_runtimesize_loopnest_common("test_runtimesizes_simple")

    def test_runtimesizes_static_split_simple(self) -> None:
        self._simple_runtimesize_loopnest_common("test_runtimesizes_static_split_simple", splits=[8])

    def test_runtimesizes_two_static_split_simple(self) -> None:
        self._simple_runtimesize_loopnest_common("test_runtimesizes_two_static_split_simple", splits=[128, 8])

    def test_runtimesizes_two_static_split_boundary(self) -> None:
        self._simple_runtimesize_loopnest_common("test_runtimesizes_two_static_split_boundary", splits=[60, 8])

    def _test_runtimesizes_matmul_common(
        self, name, M, N, K, sizes_first=True, splits=None, level_caches=None, max_element_caches=None
    ) -> None:

        A = Array(shape=(M, K), element_type=ScalarType.float32, role=Role.INPUT)
        B = Array(shape=(K, N), element_type=ScalarType.float32, role=Role.INPUT)
        C = Array(shape=(M, N), element_type=ScalarType.float32, role=Role.INPUT_OUTPUT)

        nest = Nest((M, N, K))

        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()
        if splits:
            if len(splits) != 3:
                raise ValueError("Must have one split list per loopnest dimension")

            split_indices = [[i], [j], [k]]
            for idx, dimSplits in enumerate(splits):
                for split in dimSplits:
                    split_indices[idx].append(sched.split(split_indices[idx][-1], split))

            # Collate the splits into a schedule order by taking one index from each dimension
            #   in turn as long as that dimension has more split indices
            # E.g suppose we have split indices:
            # [
            #   [i, ii, iii],
            #   [j, jj, jjj, jjjj],
            #   [k, kk]
            # ]
            # Then produce the order:
            # [ i, j, k, ii, jj, kk, iii, jjj, jjjj]

            # First, pad the dimensions with `None` until they all have the same number of entries
            max_splits = max([len(dim_split_indices) for dim_split_indices in split_indices])
            for idx in range(len(split_indices)):
                num_padding_entries = max_splits - len(split_indices[idx])
                split_indices[idx] = split_indices[idx] + [None] * num_padding_entries

            # Now collate the entries and skip the `None` padding entries
            order = [
                split_index for split_level in zip(*split_indices) for split_index in split_level
                if split_index is not None
            ]
            sched.reorder(order)

        plan = sched.create_plan()
        if level_caches is not None and max_element_caches is not None:
            raise ValueError(
                "Test code only supports either level caches or max element caches but not both at this time"
            )

        arrays_and_caches = [[A], [B], [C]]
        if level_caches:
            if len(level_caches) != 3:
                raise ValueError("Must have one level cache entry per array (even if it is empty)")
            for idx, cache_levels in enumerate(level_caches):
                for cache_level in cache_levels:
                    arrays_and_caches[idx].append(plan.cache(arrays_and_caches[idx][-1], level=cache_level))

        if max_element_caches:
            if len(max_element_caches) != 3:
                raise ValueError("Must have one max element cache entry per array (even if it is empty)")
            for idx, max_elements_per_array in enumerate(max_element_caches):
                for element_budget in max_elements_per_array:
                    arrays_and_caches[idx].append(plan.cache(arrays_and_caches[idx][-1], max_elements=element_budget))

        package = Package()

        size_test_args = []
        size_args = []
        if isinstance(M, Dimension):
            M_test = np.int64(123)
            size_test_args.append(M_test)
            size_args.append(M)
        else:
            M_test = M

        if isinstance(N, Dimension):
            N_test = np.int64(234)
            size_test_args.append(N_test)
            size_args.append(N)
        else:
            N_test = N

        if isinstance(K, Dimension):
            K_test = np.int64(345)
            size_test_args.append(K_test)
            size_args.append(K)
        else:
            K_test = K

        A_test = np.random.random((M_test, K_test)).astype(np.float32)
        B_test = np.random.random((K_test, N_test)).astype(np.float32)
        C_test = np.random.random((M_test, N_test)).astype(np.float32)

        array_pre_args = [A_test, B_test, C_test]
        array_post_args = [A_test, B_test, C_test + A_test @ B_test]
        pre_args = (size_test_args + array_pre_args) if sizes_first else (array_pre_args + size_test_args)
        post_args = (size_test_args + array_post_args) if sizes_first else (array_post_args + size_test_args)

        correctness_check_values = {
            "pre": pre_args,
            "post": post_args,
        }

        array_args = [A, B, C]
        args = (size_args + array_args) if sizes_first else (array_args + size_args)

        function = package.add(plan, args=args, base_name=name)
        self._verify_helper(package, name, function.name, correctness_check_values)

    # 1/3 dynamic
    def test_matmul_partial_runtimesizes_M(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_partial_runtimesizes_M", Dimension(), 128, 32, sizes_first=True
        )

    def test_matmul_partial_runtimesizes_K(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_partial_runtimesizes_K", 64, 128, Dimension(), sizes_first=True
        )

    def test_matmul_partial_runtimesizes_N(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_partial_runtimesizes_N", 64, Dimension(), 32, sizes_first=True
        )

    # 2/3 dynamic
    def test_matmul_partial_runtimesizes_MN(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_partial_runtimesizes_MN", Dimension(), Dimension(), 32, sizes_first=True
        )

    def test_matmul_partial_runtimesizes_MK(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_partial_runtimesizes_MK", Dimension(), 128, Dimension(), sizes_first=True
        )

    def test_matmul_partial_runtimesizes_NK(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_partial_runtimesizes_NK", 64, Dimension(), Dimension(), sizes_first=True
        )

    # 3/3 dynamic
    def test_matmul_all_runtimesizes(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_matmul_all_runtimesizes", Dimension(), Dimension(), Dimension(), sizes_first=True
        )

    # Fails because debug mode expects all the arguments first
    # def test_matmul_partial_runtimesizes_MNK_size_last(self) -> None:
    #     self._test_runtimesizes_matmul_common(
    #         "test_matmul_partial_runtimesizes_MNK_size_last",
    #         Dimension(),
    #         Dimension(),
    #         Dimension(),
    #         sizes_first=False)

    def test_partial_runtimesizes_static_splits_matmul(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_partial_runtimesizes_static_splits_matmul", 256, Dimension(), 128, splits=[[], [8], []]
        )

    def test_all_runtimesizes_matmul_single_static_split(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_single_static_split",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [8], []]
        )

    def test_all_runtimesizes_matmul_single_dim_two_static_split(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_single_dim_two_static_split",
            Dimension("M"),
            Dimension("N"),
            Dimension("K"),
            splits=[[], [64, 8], []]
        )

    def test_all_runtimesizes_matmul_single_dim_two_static_split_boundary_1(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_single_dim_two_static_split_boundary_1",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [60, 8], []]
        )

    def test_all_runtimesizes_matmul_single_dim_two_static_split_boundary_2(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_single_dim_two_static_split_boundary_2",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [64, 6], []]
        )

    def test_all_runtimesizes_matmul_two_dim_single_static_split(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_two_dim_single_static_split",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [8], [16]]
        )

    def test_all_runtimesizes_matmul_two_dim_two_static_split(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_two_dim_two_static_split",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[32, 4], [64, 8], []]
        )

    def test_all_runtimesizes_matmul_three_dim_two_static_split_boundary(self) -> None:
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_three_dim_two_static_split_boundary",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[32, 3], [64, 7], [128, 15]]
        )

    def test_all_runtimesizes_matmul_two_dim_single_static_split_static_cache(self) -> None:
        # Creates a cache in the statically sized main loop, but no cache in the dynamically sized cleanup loops as dynamically-sized caches are not supported yet
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_two_dim_single_static_split_static_cache",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [8], [16]],
            level_caches=[[], [2], []]
        )    # Cache B at level 2, which will be the jj index in a (i, j, k, jj, kk) schedule

    def test_all_runtimesizes_matmul_two_dim_two_static_split_static_cache(self) -> None:
        # Creates a cache in the statically sized main loop, but no cache in the dynamically sized cleanup loops as dynamically-sized caches are not supported yet
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_two_dim_two_static_split_static_cache",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [64, 8], [32, 4]],
            level_caches=[[], [4], []]
        )    # Cache B at level 4, which will be the jj index in a (i, j, k, jj, kk, jjj, kkk) schedule

    def test_all_runtimesizes_matmul_two_dim_two_static_split_static_max_element_cache(self) -> None:
        j_outer_split = 64
        j_inner_split = 8
        k_outer_split = 32
        k_inner_split = 4
        self._test_runtimesizes_matmul_common(
            "test_all_runtimesizes_matmul_two_dim_two_static_split_static_max_element_cache",
            Dimension(),
            Dimension(),
            Dimension(),
            splits=[[], [j_outer_split, j_inner_split], [k_outer_split, k_inner_split]],
            max_element_caches=[[], [k_outer_split * j_inner_split], []]
        )

    def test_partial_dynamic_sized_uint8_matmul(self) -> None:

        test_name = "test_partial_dynamic_sized_uint8_matmul"

        M = 256
        N = 256
        K = Dimension()

        A = Array(shape=(M, K), element_type=ScalarType.uint8, role=Role.INPUT)
        B = Array(shape=(K, N), element_type=ScalarType.uint8, role=Role.INPUT)
        C = Array(shape=(M, N), element_type=ScalarType.int32, role=Role.INPUT_OUTPUT)

        nest = Nest((M, N, K))

        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()

        static_tile_shape = [60, 64, 64]
        compute_kernel_shape = [6, 16, 4]
        vector_kernel_shape = [1, 8, 2]

        ii, jj, kk = sched.tile(dict(zip([i, j, k], static_tile_shape)))
        iii, jjj, kkk = sched.tile(dict(zip([ii, jj, kk], compute_kernel_shape)))
        iiii, jjjj, kkkk = sched.tile(dict(zip([iii, jjj, kkk], vector_kernel_shape)))

        sched.reorder(i, j, k, ii, jj, kk, iii, jjj, kkk, iiii, jjjj, kkkk)

        plan = sched.create_plan()
        plan.cache(A, index=kkkk, trigger_index=iii, element_type=ScalarType.int16, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(B, index=jjjj, trigger_index=jj, element_type=ScalarType.int16, layout=Array.Layout.LAST_MAJOR)
        plan.cache(C, index=iii, layout=Array.Layout.FIRST_MAJOR)
        # TODO: Re-enable vectorization when vectorization pattern rewrite is available
        # plan.vectorize(jjjj)

        package = Package()
        # BUGBUG: dim args ordered first due to issue with Debug mode
        function = package.add(plan, args=(K, A, B, C), base_name=test_name)

        M_test = M
        N_test = N
        K_test = np.int64(128)
        A_test = np.random.random((M_test, K_test)).astype(np.uint8)
        B_test = np.random.random((K_test, N_test)).astype(np.uint8)
        C_test = np.random.random((M_test, N_test)).astype(np.int32)
        correctness_check_values = {
            "pre": [K_test, A_test, B_test, C_test],
            "post": [K_test, A_test, B_test, C_test + A_test @ B_test],
        }

        self._verify_helper(package, test_name, function.name, correctness_check_values)

    def test_all_dynamic_sizes_static_unroll_matmul(self) -> None:

        test_name = "test_all_dynamic_sizes_static_unroll_matmul"

        M = Dimension()
        N = Dimension()
        K = Dimension()

        A = Array(shape=(M, K), element_type=ScalarType.float32, role=Role.INPUT)
        B = Array(shape=(K, N), element_type=ScalarType.float32, role=Role.INPUT)
        C = Array(shape=(M, N), element_type=ScalarType.float32, role=Role.INPUT_OUTPUT)

        nest = Nest((M, N, K))

        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()

        jj = sched.split(j, 4)
        sched.reorder(i, j, k, jj)
        plan = sched.create_plan()
        plan.unroll(jj)

        package = Package()
        # BUGBUG: dim args ordered first due to issue with Debug mode
        function = package.add(plan, args=(M, N, K, A, B, C), base_name=test_name)

        M_test = np.int64(123)
        N_test = np.int64(234)
        K_test = np.int64(345)
        A_test = np.random.random((M_test, K_test)).astype(np.float32)
        B_test = np.random.random((K_test, N_test)).astype(np.float32)
        C_test = np.random.random((M_test, N_test)).astype(np.float32)
        correctness_check_values = {
            "pre": [M_test, N_test, K_test, A_test, B_test, C_test],
            "post": [M_test, N_test, K_test, A_test, B_test, C_test + A_test @ B_test],
        }

        self._verify_helper(package, test_name, function.name, correctness_check_values)

    def test_all_dynamic_sizes_static_vectorize_matmul(self) -> None:

        test_name = "test_all_dynamic_sizes_static_vectorize_matmul"

        M = Dimension()
        N = Dimension()
        K = Dimension()

        A = Array(shape=(M, K), element_type=ScalarType.float32, role=Role.INPUT)
        B = Array(shape=(K, N), element_type=ScalarType.float32, role=Role.INPUT)
        C = Array(shape=(M, N), element_type=ScalarType.float32, role=Role.INPUT_OUTPUT)

        nest = Nest((M, N, K))

        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()

        jj = sched.split(j, 8)
        sched.reorder(i, j, k, jj)
        plan = sched.create_plan()
        plan.vectorize(jj)

        package = Package()
        # BUGBUG: dim args ordered first due to issue with Debug mode
        function = package.add(plan, args=(M, N, K, A, B, C), base_name=test_name)

        M_test = np.int64(123)
        N_test = np.int64(234)
        K_test = np.int64(345)
        A_test = np.random.random((M_test, K_test)).astype(np.float32)
        B_test = np.random.random((K_test, N_test)).astype(np.float32)
        C_test = np.random.random((M_test, N_test)).astype(np.float32)
        correctness_check_values = {
            "pre": [M_test, N_test, K_test, A_test, B_test, C_test],
            "post": [M_test, N_test, K_test, A_test, B_test, C_test + A_test @ B_test],
        }

        self._verify_helper(package, test_name, function.name, correctness_check_values)

    def test_all_dynamic_sized_fp32_mlas_matmul(self) -> None:

        test_name = "test_all_dynamic_sized_fp32_mlas_matmul"

        M = Dimension(name="M")
        N = Dimension(name="N")
        K = Dimension(name="K")

        A = Array(shape=(M, K), element_type=ScalarType.float32, role=Role.INPUT)
        B = Array(shape=(K, N), element_type=ScalarType.float32, role=Role.INPUT)
        C = Array(shape=(M, N), element_type=ScalarType.float32, role=Role.INPUT_OUTPUT)

        assert A._size_str == "M*K"
        assert B._size_str == "K*N"
        assert C._size_str == "M*N"

        nest = Nest((M, N, K))

        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        sched = nest.create_schedule()

        NK_static_tile_shape = [128, 128]
        MNK_compute_kernel_shape = [6, 16, 4]
        N_vector_size = 8

        jj, kk = sched.tile(dict(zip([j, k], NK_static_tile_shape)))
        ii, jjj, kkk = sched.tile(dict(zip([i, jj, kk], MNK_compute_kernel_shape)))
        jjjj = sched.split(jjj, N_vector_size)

        sched.reorder(i, j, k, jj, kk, kkk, ii, jjj, jjjj)

        plan = sched.create_plan()
        plan.cache(B, index=jj, layout=Array.Layout.FIRST_MAJOR)
        plan.cache(C, index=ii, layout=Array.Layout.FIRST_MAJOR)
        # plan.unroll(kkk) # implicit
        plan.unroll(ii)
        plan.unroll(jjj)
        plan.vectorize(jjjj)

        package = Package()
        # BUGBUG: dim args ordered first due to issue with Debug mode
        function = package.add(plan, args=(M, N, K, A, B, C), base_name=test_name)

        M_test = np.int64(123)
        N_test = np.int64(234)
        K_test = np.int64(345)
        A_test = np.random.random((M_test, K_test)).astype(np.float32)
        B_test = np.random.random((K_test, N_test)).astype(np.float32)
        C_test = np.random.random((M_test, N_test)).astype(np.float32)
        correctness_check_values = {
            "pre": [M_test, N_test, K_test, A_test, B_test, C_test],
            "post": [M_test, N_test, K_test, A_test, B_test, C_test + A_test @ B_test],
        }

        self._verify_helper(package, test_name, function.name, correctness_check_values)


    #
    # This test is the implementation of range node that generates two functions
    # One is for getting the size of the output array, e.g. range_get_size()
    # another is for getting the output array, e.g. range_get_result()
    # The user may want to call the fucntions in the following way:
    # Step 1, call range_get_size to get the array size
    # Step 2, allocate the memory for the output array
    # Step 3, call range_get_result to fill in the result of output array
    #
    def test_output_array_range_node1(self) -> None:
        from accera import create_dimensions, floor, cast
        from accera._lang_python._lang import Scalar

        start = Scalar(ScalarType.float32)
        limit = Scalar(ScalarType.float32)
        delta = Scalar(ScalarType.float32)

        inputDim = create_dimensions()

        outputDim = Dimension(role=Role.OUTPUT)
        output = Array(shape=(inputDim, ), role=Role.INPUT_OUTPUT)
        output_start = Scalar(ScalarType.float32, Role.INPUT_OUTPUT)

        nest1 = Nest((1, ))

        @nest1.iteration_logic
        def _():
            outputDim.set(cast(floor((limit - start) / delta), ScalarType.int64))

        nest2 = Nest([inputDim])
        i = nest2.get_indices()

        @nest2.iteration_logic
        def _():
            output[i] = output_start
            output_start.set(output_start + delta)

        # Generate a function like:
        # range_get_size(float start, float limit, float delta, int64_t* output_dim);
        # range_get_result(int64_t input_dim, float* output, float* start, float delta);

        package = Package()
        get_size_fn_name = f"get_size"
        get_result_fn_name = f"get_result"
        get_size_fn = package.add(nest1, args=(start, limit, delta, outputDim), base_name=get_size_fn_name)
        get_result_fn = package.add(nest2, args=(inputDim, output, output_start, delta), base_name=get_result_fn_name)

        start_test = np.float32(1.0)
        limit_test = np.float32(5.0)
        delta_test = np.float32(0.5)
        size_test = np.floor((limit_test - start_test) / delta_test).astype(np.int64)
        x_ref = np.random.random((size_test,)).astype(np.float32)
        y_ref = np.arange(start_test, limit_test, delta_test, dtype=np.float32)

        outputDims_post_test = np.random.random((1,)).astype(np.int64)
        outputDims_post_test[0] = size_test
        outputDims_pre_test = np.random.random((1,)).astype(np.int64)
        start_array_pre_test = np.random.random((1,)).astype(np.float32)
        start_array_post_test = np.random.random((1,)).astype(np.float32)
        start_array_pre_test[0] = start_test
        start_array_post_test[0] = start_test

        for _ in range(0, size_test):
            start_array_post_test[0] += delta_test

        correctness_check_values = {
            "pre": [start_test, limit_test, delta_test, outputDims_pre_test],
            "post": [start_test, limit_test, delta_test, outputDims_post_test],
        }

        # TODO: Disabling this verification for now, re-enable it when undoing this change.
        # self._verify_helper(package, get_size_fn_name, get_size_fn.name, correctness_check_values)

        correctness_check_values = {
            "pre": [size_test, x_ref, start_array_pre_test, delta_test],
            "post": [size_test, y_ref, start_array_post_test, delta_test],
        }

        self._verify_helper(package, get_result_fn_name, get_result_fn.name, correctness_check_values)


    # This test is another implementation of range node using nested function calls
    def test_output_array_range_node2(self) -> None:
        from accera import create_dimensions, floor, cast
        from accera._lang_python._lang import Scalar, Dimension

        start = Scalar(ScalarType.float32)
        limit = Scalar(ScalarType.float32)
        delta = Scalar(ScalarType.float32)

        inputDim = create_dimensions()

        outputDim = Dimension(role=Role.OUTPUT)
        output = Array(shape=(inputDim, ), role=Role.INPUT_OUTPUT)
        output_start = Scalar(type=ScalarType.float32, role=Role.TEMP)

        nest1 = Nest((1, ))
        @nest1.iteration_logic
        def _():
            outputDim.set(cast(floor((limit - start) / delta), ScalarType.int64))

        nest2 = Nest((1, ))
        @nest2.iteration_logic
        def _():
            output_start.set(start)

        nest3 = Nest([inputDim])
        i = nest3.get_indices()

        @nest3.iteration_logic
        def _():
            output[i] = output_start
            output_start.set(output_start + delta)

        # Generate a function like:
        # range_get_size(float start, float limit, float delta, int64_t* output_dim);
        # ini_start(float* output_start, float start);
        # get_result(int64_t input_dim, float* output, float* start, float delta);
        # range_get_output_array(int64_t input_dim, float* output, float start, float delta);

        package = Package()
        # BUGBUG: dim args ordered first due to issue with Debug mode
        package.add(nest1, args=(start, limit, delta, outputDim), base_name=f"range_get_size")
        ini_start_fn = package.add(nest2, args=(start,), base_name=f"ini_start")
        get_result_fn = package.add(nest3, args=(inputDim, output, delta), base_name=f"get_result")

        nest4 = Nest((1, ))

        @nest4.iteration_logic
        def _():
            ini_start_fn(start)
            get_result_fn(inputDim, output, delta)

        # BUGBUG: dim args ordered first due to issue with Debug mode
        package.add(nest4, args=(inputDim, output, start, delta), base_name=f"range_get_output_array")

        package.build(
            "test_output_array_range_node2",
            format=TEST_FORMAT | Package.Format.MLIR_VERBOSE,
            mode=TEST_MODE,
            output_dir=TEST_PACKAGE_DIR
        )

    def _test_output_array_gather_node(self, axis: int) -> None:
        from accera import create_dimensions

        DataDim0, DataDim1, IndicesDim0, IndicesDim1 = create_dimensions()

        # rank(Output) = rank(Data) + rank(Indices) - 1 = 2 + 2 - 1 = 3
        OutputDim0, OutputDim1, OutputDim2 = create_dimensions(Role.OUTPUT)

        Data = Array(shape=(DataDim0, DataDim1), role=Role.INPUT)
        Indices = Array(shape=(IndicesDim0, IndicesDim1), role=Role.INPUT, element_type=ScalarType.index)

        # derive output dims from input dims
        # Note: negative indices are not supported
        if axis == 0:
            # represents a runtime output-only array (dynamically allocated)
            Output = Array(shape=(IndicesDim0, IndicesDim1, DataDim1), role=Role.INPUT_OUTPUT)

            nest_dims_0 = Nest((1, ))
            @nest_dims_0.iteration_logic
            def _():
                OutputDim0.set(IndicesDim0)
                OutputDim1.set(IndicesDim1)
                OutputDim2.set(DataDim1)

            nest_array_0 = Nest((IndicesDim0, IndicesDim1, DataDim1))
            i, j, k = nest_array_0.get_indices()

            @nest_array_0.iteration_logic
            def _():
                Output[i, j, k] = Data[Indices[i, j], k]

            # Generate a function like:
            #
            # Gather_rank_2_dim_axis_0(int64_t data_dim1, int64_t indices_dim0, int64_t indices_dim1,
            #   int64_t* output_dim0, int64_t* output_dim1, int64_t* output_dim2);
            #
            # Gather_rank_2_array_axis_0(int64_t output_array_dim0, int64_t output_array_dim1, int64_t output_array_dim2,
            #   float** output, float* data, int64_t* indices);

            package = Package()
            package.add(
                nest_dims_0,
                args=(DataDim1, IndicesDim0, IndicesDim1, OutputDim0, OutputDim1, OutputDim2),
                base_name=f"Gather_rank_2_dim_axis_{axis}"
            )
            package.add(
                nest_array_0,
                args=(IndicesDim0, IndicesDim1, DataDim0, DataDim1, Output, Data, Indices),
                base_name=f"Gather_rank_2_array_axis_{axis}"
            )

            package.build(
                "test_output_array_gather_node",
                format=TEST_FORMAT | Package.Format.MLIR_VERBOSE,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR
            )

        else:
            assert (axis == 1)
            # represents a runtime output-only array (dynamically allocated)
            Output = Array(shape=(DataDim0, IndicesDim0, IndicesDim1), role=Role.INPUT_OUTPUT)
            nest_dims_1 = Nest((1, ))
            @nest_dims_1.iteration_logic
            def _():
                OutputDim0.set(DataDim0)
                OutputDim1.set(IndicesDim0)
                OutputDim2.set(IndicesDim1)

            nest_array_1 = Nest((DataDim0, IndicesDim0, IndicesDim1))
            i, j, k = nest_array_1.get_indices()

            @nest_array_1.iteration_logic
            def _():
                Output[i, j, k] = Data[i, Indices[j, k]]

            # Generate a function like:
            #
            # Gather_rank_2_dim_axis_1(int64_t data_dim0, int64_t indices_dim0, int64_t indices_dim1,
            #   int64_t* output_dim0, int64_t* output_dim1, int64_t* output_dim2);
            #
            # Gather_rank_2_array_axis_1(int64_t output_array_dim0, int64_t output_array_dim1, int64_t output_array_dim2,
            #   float** output, float* data, int64_t* indices);
            #

            package = Package()
            package.add(
                nest_dims_1,
                args=(DataDim0, IndicesDim0, IndicesDim1, OutputDim0, OutputDim1, OutputDim2),
                base_name=f"Gather_rank_2_dim_axis_{axis}"
            )
            package.add(
                nest_array_1,
                args=(DataDim0, IndicesDim0, IndicesDim1, DataDim1, Output, Data, Indices),
                base_name=f"Gather_rank_2_array_axis_{axis}"
            )

            package.build(
                "test_output_array_gather_node",
                format=TEST_FORMAT | Package.Format.MLIR_VERBOSE,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR
            )

    def test_output_array_gather_node(self) -> None:
        self._test_output_array_gather_node(0)
        self._test_output_array_gather_node(1)

class DSLTest_02SimpleAffineLoopNests(unittest.TestCase):
    def _create_nest(self, shape: Tuple[int], type=ScalarType.float32) -> Tuple:
        # helper function to create a nest so that we can focus on the logic function
        M, N, S = shape

        A = Array(role=Role.INPUT, element_type=type, shape=(M, S))
        B = Array(role=Role.INPUT, element_type=type, shape=(S, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=type, shape=(M, N))

        return Nest(shape=(M, N, S)), A, B, C

    def _build_nest(self, nest, args: Tuple[Array], package_name, correctness_check_values=None, quiet=True) -> None:
        # helper function to build a nest so that we can focus on the logic function
        # create a HAT package and add the nest to it
        package = Package()
        function = package.add(nest, args, base_name=package_name)

        # build the HAT package
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir, _quiet=quiet)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_signed_types(self) -> None:
        for t in [ScalarType.int16, ScalarType.int32, ScalarType.int64] + FLOAT_TYPES:

            A = Array(role=Role.INPUT, element_type=t, shape=(16, 16))
            B = Array(role=Role.INPUT, element_type=t, shape=(16, 16))
            C = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(16, 16))

            nest = Nest(shape=(16, 16))
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += A[i, j] + B[i, j]
                C[i, j] += A[i, j] - B[i, j]
                C[i, j] += A[i, j] * B[i, j]
                C[i, j] += A[i, j] / B[i, j]

            dtype = np.dtype(t.name)

            A_test = np.random.random(A.shape).astype(dtype)
            B_test = np.ones((C.shape)).astype(dtype)    # avoid divide by zero
            C_test = np.random.random(C.shape).astype(dtype)

            C_ref = C_test + A_test + B_test
            C_ref = C_ref + A_test - B_test
            C_ref = C_ref + A_test * B_test
            C_ref = C_ref + A_test / B_test

            if (t == ScalarType.float16):    # TODO: verification issue with correctness check?
                correctness_check_values = None
            else:
                correctness_check_values = {
                    "pre": [A_test, B_test, C_test],
                    "post": [A_test, B_test, C_ref],
                }

            self._build_nest(nest, [A, B, C], f"test_types_{t.name}", correctness_check_values)

    def test_unsigned_types(self) -> None:
        for t in [
                ScalarType.uint8,
                ScalarType.uint16,
                ScalarType.uint32,
                ScalarType.uint64,
        ]:

            A = Array(role=Role.INPUT, element_type=t, shape=(16, 16))
            B = Array(role=Role.INPUT, element_type=t, shape=(16, 16))
            C = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(16, 16))

            nest = Nest(shape=(16, 16))
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += A[i, j] + B[i, j]
                C[i, j] += A[i, j] - B[i, j]
                C[i, j] += A[i, j] * B[i, j]
                C[i, j] += A[i, j] / B[i, j]

            dtype = np.dtype(t.name)
            A_test = np.random.random(A.shape).astype(dtype)
            B_test = np.ones((C.shape)).astype(dtype)    # avoid divide by zero
            C_test = np.random.random(C.shape).astype(dtype)

            C_ref = C_test + A_test + B_test
            C_ref = C_ref + A_test - B_test
            C_ref = C_ref + A_test * B_test
            C_ref = C_ref + A_test / B_test

            correctness_check_values = {
                "pre": [A_test, B_test, C_test],
                "post": [A_test, B_test, C_ref],
            }

            self._build_nest(nest, [A, B, C], f"test_types_{t.name}", correctness_check_values)

    def test_arithmetic_operations(self) -> None:
        for t in INT_TYPES + FLOAT_TYPES:
            nest, A, B, C = self._create_nest((16, 10, 11), type=t)
            i, j, k = nest.get_indices()

            int_val = 2
            float_val = 1.5

            @nest.iteration_logic
            def _():
                C[i, j] = A[i, k] + B[k, j]    # test assignment
                C[i, j] += A[i, k] - B[k, j]
                C[i, j] += A[i, k] * B[k, j]
                C[i, j] += A[i, k] / B[k, j]

                if t != ScalarType.float16:
                    C[i, j] += int_val + A[i, k]
                    C[i, j] += int_val - A[i, k]
                    C[i, j] += int_val * A[i, k]
                    C[i, j] += int_val / A[i, k]
                    C[i, j] += A[i, k] + int_val
                    C[i, j] += A[i, k] - int_val
                    C[i, j] += A[i, k] * int_val
                    C[i, j] += A[i, k] / int_val

                if t in FLOAT_TYPES:
                    C[i, j] += float_val + A[i, k]
                    C[i, j] += float_val - A[i, k]
                    C[i, j] += float_val * A[i, k]
                    C[i, j] += float_val / A[i, k]
                    C[i, j] += A[i, k] + float_val
                    C[i, j] += A[i, k] - float_val
                    C[i, j] += A[i, k] * float_val
                    C[i, j] += A[i, k] / float_val

                C[i, j] += -A[i, k]
                C[i, j] += A[i, k] // B[k, j]
                C[i, j] += A[i, k] % B[k, j]
                C[i, j] += A[i, k]**B[k, j]

            self._build_nest(nest, [A, B, C], f"test_arithmetic_operations_{t.name}")

    def test_relational_operations(self) -> None:
        from accera._lang_python._lang import _If

        for t in [ScalarType.bool] + INT_TYPES + FLOAT_TYPES:
            nest, A, B, C = self._create_nest((16, 10, 11))
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                def f1():
                    C[i, j] += A[i, k] + B[k, j]

                def f2():
                    C[i, j] -= A[i, k] + B[k, j]

                def f3():
                    C[i, j] *= A[i, k] + B[k, j]

                def f4():
                    C[i, j] /= A[i, k] + B[k, j]

                # BUGBUG: this syntax probably needs to change
                _If(A[i, k] == B[k, j], f1)
                _If(A[i, k] != B[k, j], f2)
                _If(A[i, k] < B[k, j], f3)
                _If(A[i, k] <= B[k, j], f4)
                _If(A[i, k] > B[k, j], f1)
                _If(A[i, k] >= B[k, j], f2)

            self._build_nest(nest, [A, B, C], f"test_relational_operations_{t.name}")

    def test_logical_operations(self) -> None:
        from accera import logical_and, logical_or, logical_not

        for t in [ScalarType.bool] + INT_TYPES:
            nest, A, B, C = self._create_nest((16, 10, 11), type=t)
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += logical_not(A[i, k])
                C[i, j] += logical_and(A[i, k], B[k, j])
                C[i, j] += logical_or(A[i, k], B[k, j])

            self._build_nest(nest, [A, B, C], f"test_logical_operations_{t.name}")

    def test_bitwise_operations(self) -> None:
        for t in INT_TYPES:
            nest, A, B, C = self._create_nest((16, 10, 11), type=t)
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += B[j, k] >> 1
                C[i, j] += A[i, j] << 2
                C[i, j] += A[i, j] & B[j, k]
                C[i, j] += A[i, j] | B[j, k]
                C[i, j] += A[i, j] ^ B[j, k]
                C[i, j] += ~A[i, j]

            self._build_nest(nest, [A, B, C], f"test_bitwise_operations_{t.name}")

    def test_intrinsics(self) -> None:
        from accera import max, min

        for t in INT_TYPES + FLOAT_TYPES:

            nest, A, B, C = self._create_nest((16, 10, 11), type=t)
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += max(A[i, j], B[j, k])
                C[i, j] += min(A[i, j], B[j, k])

            self._build_nest(nest, [A, B, C], f"test_intrinsics_{t.name}")

    def test_round_intrinsic(self) -> None:
        from accera import round as accround

        M = 16
        N = 8

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        B = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, N))

        nest = Nest((M, N))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i, j] = accround(A[i, j])

        A_test = np.random.uniform(low=-1000.0, high=1000.0, size=A.shape).astype(np.float32)
        # Ensure there's at least one element which tests the roundeven behavior in both directions
        A_test[0, 0] = 1.5    # Should round up to 2
        A_test[0, 1] = 2.5    # Should round down to 2
        B_test = np.zeros(B.shape).astype(np.int32)

        B_ref = A_test.round().astype(np.int32)
        self.assertEqual(B_ref[0, 0], 2)
        self.assertEqual(B_ref[0, 1], 2)

        correctness_check_values = {
            "pre": [A_test, B_test],
            "post": [A_test, B_ref]
        }

        self._build_nest(nest, [A, B], "test_round_intrinsic", correctness_check_values=correctness_check_values)

    @expectedFailure(FailedReason.INVALID, "x86 round intrinsic not supported on MacOS", sys.platform == "darwin")
    def test_round_intrinsic_vectorized(self) -> None:
        from accera import round as accround

        M = 256
        N = 128

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        B = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, N))

        nest = Nest((M, N))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i, j] = accround(A[i, j])

        sched = nest.create_schedule()
        ii, jj = sched.tile({
            i: 4,
            j: 8
        })
        sched.reorder(i, j, ii, jj)
        plan = sched.create_plan()
        plan.vectorize(ii)

        A_test = np.random.uniform(low=-1000.0, high=1000.0, size=A.shape).astype(np.float32)
        # Ensure there's at least one element which tests the roundeven behavior in both directions
        A_test[0, 0] = 1.5    # Should round up to 2
        A_test[0, 1] = 2.5    # Should round down to 2
        B_test = np.zeros(B.shape).astype(np.int32)

        B_ref = A_test.round().astype(np.int32)
        self.assertEqual(B_ref[0, 0], 2)
        self.assertEqual(B_ref[0, 1], 2)

        correctness_check_values = {
            "pre": [A_test, B_test],
            "post": [A_test, B_ref]
        }

        self._build_nest(
            plan, [A, B], "test_round_intrinsic_vectorized", correctness_check_values=correctness_check_values
        )

    # TODO : fix this test - it appears to abort on just the linux buddy build machine
    # def test_remainderf_intrinsic_rounding(self) -> None:
    #     from accera import remainderf, cast

    #     M = 16
    #     N = 8

    #     A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
    #     B = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, N))

    #     nest = Nest((M, N))
    #     i, j = nest.get_indices()

    #     @nest.iteration_logic
    #     def _():
    #         B[i, j] = cast(A[i, j] - remainderf(A[i, j], 1.0), ScalarType.int32)

    #     A_test = np.random.uniform(low=-1000.0, high=1000.0, size=A.shape).astype(np.float32)
    #     # Ensure there's at least one element which tests the roundeven behavior in both directions
    #     A_test[0, 0] = 1.5 # Should round up to 2
    #     A_test[0, 1] = 2.5 # Should round down to 2
    #     B_test = np.zeros(B.shape).astype(np.int32)

    #     B_ref = A_test.round().astype(np.int32)
    #     self.assertEqual(B_ref[0, 0], 2)
    #     self.assertEqual(B_ref[0, 1], 2)

    #     correctness_check_values = {
    #         "pre": [A_test, B_test],
    #         "post": [A_test, B_ref]
    #     }

    #     self._build_nest(nest, [A, B], "test_remainderf_intrinsic_rounding", correctness_check_values=correctness_check_values)

    @expectedFailure(FailedReason.INVALID, "x86 max min intrinsics not supported on MacOS", sys.platform == "darwin")
    def test_vectorized_max_min(self) -> None:
        from accera import max, min

        M = 128
        N = 256

        package = Package()
        func_names = []
        package_name = "test_vectorized_max_min"
        correctness_check_values = {}
        for t in [ScalarType.float32]:
            fn_name = f"test_vectorized_max_min_{t.name}"
            func_names.append(fn_name)

            nest = Nest((M, N))
            A = Array(role=Role.INPUT, element_type=t, shape=(M, N))
            B = Array(role=Role.INPUT, element_type=t, shape=(M, N))
            C_max = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(M, N))
            C_min = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(M, N))

            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                C_max[i, j] = max(A[i, j], B[i, j])
                C_min[i, j] = min(A[i, j], B[i, j])

            sched = nest.create_schedule()
            ii, jj = sched.tile({
                i: 4,
                j: 8
            })
            sched.reorder(i, j, ii, jj)
            plan = sched.create_plan()
            plan.vectorize(ii)
            function = package.add(plan, args=(A, B, C_max, C_min), base_name=fn_name)

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_max_test = np.random.random(C_max.shape).astype(np.float32)
            C_min_test = np.random.random(C_min.shape).astype(np.float32)

            C_max_ref = np.maximum(A_test, B_test)
            C_min_ref = np.minimum(A_test, B_test)

            correctness_check_values[fn_name] = {
                "pre": [A_test, B_test, C_max_test, C_min_test],
                "post": [A_test, B_test, C_max_ref, C_min_ref]
            }

        # build the HAT package
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT | Package.Format.MLIR_VERBOSE,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )
            for fn_name in func_names:
                if fn_name in correctness_check_values:
                    v.check_correctness(
                        function.name,
                        before=correctness_check_values[fn_name]["pre"],
                        after=correctness_check_values[fn_name]["post"],
                    )

    @expectedFailure(FailedReason.INVALID, "x86 max min intrinsics not supported on MacOS", sys.platform == "darwin")
    def test_vectorized_single_max_min_block(self) -> None:
        # In this test we're trying to find the single max and single min value of a 2-D array.
        # To vectorize this, we'll want to compute several maxs and mins in paralle and then reduce them
        # Note: This type of reduction can't be achieved with caching, so we manually construct a pattern similar to caching
        from accera import max, min

        M = 128
        N = 256

        M_outer_tile = 8
        M_tile = 4
        N_tile = 8

        package = Package()
        func_names = []
        package_name = "test_vectorized_single_max_min_block"
        correctness_check_values = {}
        for t in [ScalarType.float32]:
            fn_name = f"{package_name}_{t.name}"
            func_names.append(fn_name)

            A = Array(role=Role.INPUT, element_type=t, shape=(M, N))
            A_max = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(1, ))
            A_min = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(1, ))

            A_max_cache = Array(role=Role.TEMP, element_type=t, shape=(M_tile, N_tile), flags=AllocateFlags.STACK)
            A_min_cache = Array(role=Role.TEMP, element_type=t, shape=(M_tile, N_tile), flags=AllocateFlags.STACK)

            io_A_max_cache = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=A_max_cache.shape)
            io_A_min_cache = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=A_min_cache.shape)

            outer_i_dim = Dimension()
            outer_j_dim = Dimension()

            # inner compute nest

            inner_nest = Nest((M_tile, N_tile))
            inner_i, inner_j = inner_nest.get_indices()

            @inner_nest.iteration_logic
            def _():
                i = outer_i_dim + inner_i
                j = outer_j_dim + inner_j
                io_A_max_cache[inner_i, inner_j] = max(io_A_max_cache[inner_i, inner_j], A[i, j])
                io_A_min_cache[inner_i, inner_j] = min(io_A_min_cache[inner_i, inner_j], A[i, j])

            inner_sched = inner_nest.create_schedule()
            inner_plan = inner_sched.create_plan()
            inner_plan.vectorize(inner_i)
            inner_fn = package.add(
                inner_plan,
                args=(A, io_A_max_cache, io_A_min_cache, outer_i_dim, outer_j_dim),
                base_name=f"{fn_name}_inner",
                function_opts=INTERNAL_FUNCTION_OPTS
            )

            # Outer nest
            outer_nest = Nest((M, N))
            outer_i, outer_j = outer_nest.get_indices()

            @outer_nest.iteration_logic
            def _():
                inner_fn(A, io_A_max_cache, io_A_min_cache, outer_i, outer_j)

            outer_sched = outer_nest.create_schedule()
            outer_ii = outer_sched.split(outer_i, M_outer_tile)
            outer_iii, outer_jj = outer_sched.tile({
                outer_ii: M_tile,
                outer_j: N_tile
            })
            outer_sched.reorder(outer_i, outer_j, outer_ii, outer_iii, outer_jj)
            outer_plan = outer_sched.create_plan()
            outer_plan._erase_loops([outer_iii, outer_jj])
            outer_fn = package.add(
                outer_plan,
                args=(A, io_A_max_cache, io_A_min_cache),
                base_name=f"{fn_name}_outer",
                function_opts=INTERNAL_FUNCTION_OPTS
            )

            # Cache zeroing nests

            def _make_init_fn(package: Package, outer_arr: Array, arr: Array, base_name: str):
                zero_nest = Nest(arr.shape)
                indices = zero_nest.get_indices()

                @zero_nest.iteration_logic
                def _():
                    arr[indices] = outer_arr[indices]

                return package.add(
                    zero_nest, args=(outer_arr, arr), base_name=base_name, function_opts=INTERNAL_FUNCTION_OPTS
                )

            zero_max_cache_fn = _make_init_fn(package, A, io_A_max_cache, "max_cache_zeroing")
            zero_min_cache_fn = _make_init_fn(package, A, io_A_min_cache, "min_cache_zeroing")

            # Cache reducing nests

            def _make_cache_reduce_fn(package: Package, cache: Array, outer_arr: Array, base_name: str, use_max):
                reduce_nest = Nest(cache.shape)
                indices = reduce_nest.get_indices()
                if use_max:

                    @reduce_nest.iteration_logic
                    def _():
                        outer_arr[0] = max(outer_arr[0], cache[indices])
                else:

                    @reduce_nest.iteration_logic
                    def _():
                        outer_arr[0] = min(outer_arr[0], cache[indices])

                return package.add(
                    reduce_nest, args=(cache, outer_arr), base_name=base_name, function_opts=INTERNAL_FUNCTION_OPTS
                )

            reduce_max_cache_fn = _make_cache_reduce_fn(package, io_A_max_cache, A_max, "max_cache_reduce", True)
            reduce_min_cache_fn = _make_cache_reduce_fn(package, io_A_min_cache, A_min, "min_cache_reduce", False)

            # outer nest

            top_nest = Nest((1, ))

            @top_nest.iteration_logic
            def _():
                zero_max_cache_fn(A, A_max_cache)
                zero_min_cache_fn(A, A_min_cache)
                outer_fn(A, A_max_cache, A_min_cache)
                reduce_max_cache_fn(A_max_cache, A_max)
                reduce_min_cache_fn(A_min_cache, A_min)

            function = package.add(top_nest, args=(A, A_max, A_min), base_name=fn_name)

            A_test = np.random.random(A.shape).astype(np.float32)
            A_max_test = np.random.random(A_max.shape).astype(np.float32)
            A_min_test = np.random.random(A_min.shape).astype(np.float32)

            A_max_ref = np.max(A_test).reshape((1, ))
            A_min_ref = np.min(A_test).reshape((1, ))

            correctness_check_values[fn_name] = {
                "pre": [A_test, A_max_test, A_min_test],
                "post": [A_test, A_max_ref, A_min_ref]
            }

        # build the HAT package
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT | Package.Format.MLIR_VERBOSE,
                mode=Package.Mode.RELEASE,
                output_dir=output_dir
            )
            for fn_name in func_names:
                if fn_name in correctness_check_values:
                    v.check_correctness(
                        function.name,
                        before=correctness_check_values[fn_name]["pre"],
                        after=correctness_check_values[fn_name]["post"],
                    )

    def test_intrinsics_float(self) -> None:
        from accera import (
            abs,
            sqrt,
            exp,
            log,
            log10,
            log2,
            sin,
            cos,
            ceil,
            floor,
            tan,
            cosh,
            sinh,
            tanh,
        )

        # from accera._lang_python import fast_exp, fast_exp_mlas

        for t in FLOAT_TYPES:

            nest, A, B, C = self._create_nest((16, 10, 11), type=t)
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += abs(A[i, j])
                C[i, j] += exp(A[i, j])
                # C[i, j] += fast_exp(A[i, j])
                # C[i, j] += fast_exp_mlas(A[i, j])
                C[i, j] += log(B[j, k])
                C[i, j] += log2(B[j, k])
                C[i, j] += log10(A[i, j])
                C[i, j] += sin(A[i, j])
                C[i, j] += cos(B[j, k])
                C[i, j] += tan(A[i, j])
                C[i, j] += sqrt(B[j, k])
                C[i, j] += ceil(B[j, k])
                C[i, j] += floor(A[i, j])
                C[i, j] += sinh(A[i, j])
                C[i, j] += cosh(B[j, k])
                C[i, j] += tanh(A[i, j])

            self._build_nest(nest, [A, B, C], f"test_intrinsics_float_{t.name}")

    def test_convenience_syntax_1(self) -> None:
        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] + B[k, j]

        package = Package()
        package_name = "test_convenience_syntax_2"
        package.add(nest, args=(A, B, C), base_name="matmul")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_convenience_syntax_2(self) -> None:

        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        plan = nest.create_plan()

        package = Package()
        package_name = "test_convenience_syntax_2"
        package.add(plan, args=(A, B, C), base_name="matmul")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )


class DSLTest_03Schedules(unittest.TestCase):
    def _create_nest(self, shape: Tuple[int], type=ScalarType.float32) -> Tuple:
        M, N, S = shape

        A = Array(role=Role.INPUT, element_type=type, shape=(M, S))
        B = Array(role=Role.INPUT, element_type=type, shape=(S, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=type, shape=(M, N))

        return Nest(shape=(M, N, S)), A, B, C

    def _verify_schedule(self, schedule, args: Tuple[Array], package_name, correctness_check_values=None) -> None:

        # create a HAT package and add the function to it
        package = Package()
        function = package.add(schedule, args, base_name="schedule_test")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        # build the HAT package
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_schedule_reorder(self) -> None:
        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()
        schedule.reorder(k, i, j)
        self.assertEqual(schedule._indices, [k, i, j])

        schedule.reorder(order=(j, i, k))
        self.assertEqual(schedule._indices, [j, i, k])

        self._verify_schedule(schedule, [A, B, C], "test_schedule_reorder")

    def test_schedule_split(self) -> None:
        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        iii = schedule.split(ii, 2)
        iiii = schedule.split(iii, 2)
        for index in [ii, iii, iiii]:
            self.assertIsNotNone(index)
        self.assertEqual(schedule._indices, [i, ii, iii, iiii, j, k])
        self._verify_schedule(schedule, [A, B, C], "test_schedule_split1")

        # split size does not divide the dimension size
        schedule2 = nest.create_schedule()
        kk = schedule2.split(k, 4)    # original size of dimension k was 11
        self.assertIsNotNone(kk)
        self.assertEqual(schedule2._indices, [i, j, k, kk])
        self._verify_schedule(schedule2, [A, B, C], "test_schedule_split2")

        # split size == dimension size
        schedule3 = nest.create_schedule()
        kk = schedule3.split(k, 11)    # original size of dimension k was 11
        self.assertIsNotNone(kk)
        self.assertEqual(schedule3._indices, [i, j, k, kk])
        self._verify_schedule(schedule3, [A, B, C], "test_schedule_split3")

        # split size > dimension size
        schedule4 = nest.create_schedule()
        kk = schedule4.split(k, 13)    # original size of dimension k was 11
        self.assertIsNotNone(kk)
        self.assertEqual(schedule4._indices, [i, j, k, kk])
        self._verify_schedule(schedule4, [A, B, C], "test_schedule_split4")

    def test_schedule_set_invalid_order(self) -> None:
        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()
        ii = schedule.split(i, 2)
        iii = schedule.split(ii, 2)
        jj = schedule.split(j, 5)
        self.assertEqual(schedule._indices, [i, ii, iii, j, jj, k])

        with self.assertRaises(ValueError):
            schedule.reorder(k, i, jj, j)
        self.assertEqual(schedule._indices, [i, ii, iii, j, jj, k])

        with self.assertRaises(ValueError):
            schedule.reorder(k, ii, iii, j, jj, i)
        self.assertEqual(schedule._indices, [i, ii, iii, j, jj, k])

        schedule.reorder(i, j, ii, jj, iii, k)
        self.assertEqual(schedule._indices, [i, j, ii, jj, iii, k])

    def test_schedule_tile(self) -> None:
        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()
        ii, jj, kk = schedule.tile({
            i: 8,
            j: 2,
            k: 3
        })
        self.assertIsNotNone(ii)
        self.assertIsNotNone(jj)
        self.assertIsNotNone(kk)
        self.assertEqual(schedule._indices, [i, ii, j, jj, k, kk])
        self._verify_schedule(schedule, [A, B, C], "test_schedule_tile")

        # tile a subset of the iteration space
        schedule1 = nest.create_schedule()
        iii, kkk = schedule1.tile({
            i: 8,
            k: 3
        })
        self.assertIsNotNone(iii)
        self.assertIsNotNone(kkk)
        self.assertEqual(schedule1._indices, [i, iii, j, k, kkk])
        self._verify_schedule(schedule1, [A, B, C], "test_schedule_tile_subset")

    def test_schedule_skew(self) -> None:
        for N in [10, 224]:    # input sizes
            for K in [1, 3, 5]:    # filter sizes
                M = N - K + 1    # output size

                A = Array(role=Role.INPUT, shape=(N, ))
                B = Array(role=Role.INPUT, shape=(K, ))
                C = Array(role=Role.INPUT_OUTPUT, shape=(M, ))

                nest = Nest(shape=(M, K))
                i, j = nest.get_indices()

                @nest.iteration_logic
                def _():
                    C[i] += A[i + j] * B[j]

                schedule = nest.create_schedule()

                A_test = np.random.random(A.shape).astype(np.float32)
                B_test = np.random.random(B.shape).astype(np.float32)
                C_test = np.random.random(C.shape).astype(np.float32)
                correctness_check_values = {
                    "pre": [A_test, B_test, C_test],
                    "post": [
                        A_test,
                        B_test,
                        C_test + np.convolve(np.flip(B_test), A_test, "valid"),
                    ],
                }

                # Skew dimension i with respect to dimension j.
                schedule.skew(i, j)
                self._verify_schedule(
                    schedule,
                    [A, B, C],
                    f"test_schedule_skew_i_j_{N}_{K}",
                    correctness_check_values,
                )

                # Skew dimension j with respect to dimension i.
                schedule1 = nest.create_schedule()
                schedule1.skew(j, i)
                self._verify_schedule(
                    schedule1,
                    [A, B, C],
                    f"test_schedule_skew_j_i_{N}_{K}",
                    correctness_check_values,
                )

    def test_schedule_skew_unrolling(self) -> None:
        N = 10    # input size
        K = 3    # filter size
        M = N - K + 1    # output size = 8

        A = Array(role=Role.INPUT, shape=(N, ))
        B = Array(role=Role.INPUT, shape=(K, ))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, ))

        nest = Nest(shape=(M, K))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i] += A[i + j] * B[j]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [
                A_test,
                B_test,
                C_test + np.convolve(np.flip(B_test), A_test, "valid"),
            ],
        }

        # Skew dimension i with respect to dimension j, with unrolling.
        schedule = nest.create_schedule()
        schedule.skew(i, j, unroll_loops_smaller_than=3)
        self._verify_schedule(
            schedule,
            [A, B, C],
            "test_schedule_skew_i_j_with_unrolling",
            correctness_check_values,
        )

        # Skew dimension j with respect to dimension i, with unrolling.
        schedule1 = nest.create_schedule()
        schedule1.skew(j, i, unroll_loops_smaller_than=3)
        self._verify_schedule(
            schedule1,
            [A, B, C],
            f"test_schedule_skew_j_i_with_unrolling",
            correctness_check_values,
        )

    def test_schedule_pad(self) -> None:
        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        # Adds empty elements to the beginning of dimension i, j, k
        schedule.pad(i, 2)
        ii = schedule.split(i, 3)    # (2 + 16) // 3
        # should result in these loops for i, ii
        #  i: [2, 3:3), ii: [0, 1:1)  <-- partial (front padding)
        #  i: [3: 18:3), ii: [0, 3:1) <-- full

        schedule.pad(j, 3)
        jj = schedule.split(j, 3)    # (3 + 10) // 3
        # should result in these loops for j, jj
        #  j: [3, 12:3), jj: [0, 3:3)   <-- full (front padding == split size)
        #  j: [12, 13:3), jj: [0, 1:1)  <-- partial (automatic back padding)

        schedule.pad(k, 11)
        kk = schedule.split(k, 4)    # (11 + 11) // 4
        # should result in these loops for k, kk
        #  k: [11, 12:1), kk: [0, 1: 1) <-- partial
        #  k: [12, 20:4), kk: [0: 4: 1) <-- full
        #  k: [20, 22:4), kk: [0: 2: 1) <-- partial (automatic back padding)

        schedule.reorder(i, ii, k, j, jj, kk)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_test + A_test @ B_test],
        }
        self._verify_schedule(schedule, [A, B, C], "test_schedule_pad", correctness_check_values)

    def test_schedule_pad_inner_index_no_bc_1(self) -> None:
        I = 16
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, ))

        nest = Nest(shape=(I, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(ii, 2)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_inner_index_no_bc_1",
            correctness_check_values,
        )

    def test_schedule_pad_inner_index_no_bc_2(self) -> None:
        I = 16
        J = 8
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, J))

        nest = Nest(shape=(I, J))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(ii, 2)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_inner_index_no_bc_2",
            correctness_check_values,
        )

    def test_schedule_pad_inner_index_no_bc_3(self) -> None:
        I = 16
        J = 8
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, J))

        nest = Nest(shape=(I, J))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(ii, 2)
        schedule.reorder(i, j, ii)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_inner_index_no_bc_3",
            correctness_check_values,
        )

    @expectedFailure(FailedReason.BUG, "Padding with boundary conditions is broken")
    def test_schedule_pad_inner_index_bc_1(self) -> None:
        I = 17
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, ))

        nest = Nest(shape=(I, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(ii, 2)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_inner_index_bc_1",
            correctness_check_values,
        )

    @expectedFailure(FailedReason.BUG, "Padding with boundary conditions is broken")
    def test_schedule_pad_inner_index_bc_2(self) -> None:
        I = 17
        J = 8
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, J))

        nest = Nest(shape=(I, J))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(ii, 2)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_inner_index_bc_2",
            correctness_check_values,
        )

    @expectedFailure(FailedReason.BUG, "Padding with boundary conditions is broken")
    def test_schedule_pad_inner_index_bc_3(self) -> None:
        I = 17
        J = 8
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, J))

        nest = Nest(shape=(I, J))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(ii, 2)
        schedule.reorder(i, j, ii)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_inner_index_bc_3",
            correctness_check_values,
        )

    @expectedFailure(FailedReason.BUG, "Padding of outer indices is unsupported")
    def test_schedule_pad_outer_index_no_bc_1(self) -> None:
        I = 16
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, ))

        nest = Nest(shape=(I, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(i, 2)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_outer_index_no_bc_1",
            correctness_check_values,
        )

    @expectedFailure(FailedReason.BUG, "Padding of outer indices is unsupported")
    def test_schedule_pad_outer_index_no_bc_2(self) -> None:
        I = 16
        J = 8
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, J))

        nest = Nest(shape=(I, J))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(i, 2)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_outer_index_no_bc_2",
            correctness_check_values,
        )

    @expectedFailure(FailedReason.BUG, "Padding of outer indices is unsupported")
    def test_schedule_pad_outer_index_no_bc_3(self) -> None:
        I = 16
        J = 8
        A = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(I, J))

        nest = Nest(shape=(I, J))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        schedule = nest.create_schedule()
        ii = schedule.split(i, 4)
        schedule.pad(i, 2)
        schedule.reorder(i, j, ii)

        A_test = np.random.random(A.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test],
            "post": [A_test * 2]
        }
        self._verify_schedule(
            schedule,
            [A],
            "test_schedule_pad_outer_index_no_bc_3",
            correctness_check_values,
        )

    def test_convenience_syntax(self) -> None:

        nest, A, B, C = self._create_nest((16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        package = Package()
        package_name = "test_convenience_syntax"
        package.add(schedule, args=(A, B, C), base_name="plan_test")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )


class DSLTest_04Fusing(unittest.TestCase):
    def _verify_func(
        self, package, function, package_name, correctness_check_values, quiet=True, mode=TEST_MODE
    ) -> None:
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        # build the HAT package
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=mode, output_dir=output_dir, _quiet=quiet)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def _verify_schedule(
        self, schedule, args: Tuple[Array], package_name, correctness_check_values, quiet=True
    ) -> None:
        # create a HAT package and add the function to it
        package = Package()
        function = package.add(schedule, args, base_name="fusing_test")
        self._verify_func(package, function, package_name, correctness_check_values, quiet)

    def test_full_iteration_space_fusing(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT, shape=(16, 16))
        B = Array(role=Role.INPUT, shape=(16, 16))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 16))

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

        # Create a fused schedule
        schedule = fuse(schedule0, schedule1)
        f, i, j = schedule.get_indices()

        schedule.reorder(i, j, f)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, (C_test + A_test) * B_test],
        }
        self._verify_schedule(
            schedule,
            (A, B, C),
            "test_full_iteration_space_fusing1",
            correctness_check_values,
        )

        # computing the output block-by-block:
        #  first computing C[0:4, 0:4] += A[0:4, 0:4]
        #  then computing C[0:4, 0:4] *= B[0:4, 0:4]
        ii, jj = schedule.tile({
            i: 4,
            j: 4
        })
        schedule.reorder(i, j, f, ii, jj)

        self._verify_schedule(
            schedule,
            (A, B, C),
            "test_full_iteration_space_fusing2",
            correctness_check_values,
        )

    def test_partial_iteration_space_fusing_1(self) -> None:
        from accera import fuse, Nest, max
        from accera._lang_python._lang import Scalar

        A = Array(role=Role.INPUT, shape=(16, 11))
        B = Array(role=Role.INPUT, shape=(11, 10))
        C = Array(role=Role.INPUT, shape=(16, 10))

        # Fully-connected neural layer with activation: C = op(C + A @ B)
        # Create nest0 and schedule0
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
            # BUGBUG: should implicitly convert Scalar
            C[i1, j1] = max(C[i1, j1], Scalar(0.0))

        schedule1 = nest1.create_schedule()

        schedule = fuse((schedule0, schedule1), partial=2)
        f, i, j, k = schedule.get_indices()
        schedule.reorder(i, j, f, k)

        # unfused indices (k) must not precede the fusing index (f)
        with self.assertRaises(ValueError):
            schedule.reorder(i, j, k, f)
        self.assertEqual(schedule._indices, [i, j, f, k])

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, np.maximum(C_test + A_test @ B_test, 0.0)],
        }
        self._verify_schedule(
            schedule,
            (A, B, C),
            "test_partial_iteration_space_fusing_1",
            correctness_check_values,
        )

    def test_partial_iteration_space_fusing_2(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(16, ))
        B = Array(role=Role.INPUT_OUTPUT, shape=(4, ))

        n0 = Nest([16])
        i0 = n0.get_indices()

        @n0.iteration_logic
        def _():
            A[i0] *= A[i0]

        s0 = n0.create_schedule()

        n1 = Nest([16, 4])
        i1, j1 = n1.get_indices()

        @n1.iteration_logic
        def _():
            B[j1] += A[i1]

        s1 = n1.create_schedule()

        fs = fuse((s0, s1), partial=1)
        f, i, j = fs.get_indices()
        jj = fs.split(j, 2)
        fs.reorder(i, f, j, jj)

        A_test_pre = np.random.random(A.shape).astype(np.float32)
        B_test_pre = np.random.random(B.shape).astype(np.float32)
        A_test_post = A_test_pre * A_test_pre
        B_test_post = B_test_pre + np.sum(A_test_post)
        correctness_check_values = {
            "pre": [A_test_pre, B_test_pre],
            "post": [A_test_post, B_test_post],
        }

        self._verify_schedule(
            fs,
            (A, B),
            "test_partial_iteration_space_fusing_2",
            correctness_check_values,
        )

    def test_unequal_iteration_space_fusing_1(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT, shape=(16, 16))
        B = Array(role=Role.INPUT, shape=(16, 10))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 16))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 16))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 with a smaller iteration space size
        nest1 = Nest(shape=(16, 10))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        # Create a fused schedule: the smaller iteration space (nest1) should
        # be automatically end-padded with no-ops

        schedule = fuse(schedule0, schedule1)
        f, i, j = schedule.get_indices()
        schedule.reorder(i, j, f)

        # Emitted fused loop should look like:
        # for i in range(0, 16):
        #   for j in range(0, 10):
        #      for f in range(2):
        #         if f == 0:
        #           C[i, j] += A[i, j]
        #         if f == 1:
        #           C[i, j] *= B[i, j]
        #   for j in range(10, 16):
        #      for f in range(2):
        #         if f == 0:
        #           C[i, j] += A[i, j]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        C_ref = C_test + A_test    # nest0
        C_ref[:, :B.shape[1]] = C_ref[:, :B.shape[1]] * B_test    # nest1

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_ref],
        }
        self._verify_schedule(
            schedule,
            (A, B, C),
            "test_unequal_iteration_space_fusing_1",
            correctness_check_values,
        )

    def test_unequal_iteration_space_fusing_2(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT, shape=(16, 10))
        B = Array(role=Role.INPUT, shape=(16, 16))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 16))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 10))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 with a larger iteration space size
        nest1 = Nest(shape=(16, 16))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        # Create a fused schedule: the smaller iteration space (nest0) should
        # be automatically end-padded with no-ops

        schedule = fuse(schedule0, schedule1)
        f, i, j = schedule.get_indices()
        schedule.reorder(i, j, f)

        # Emitted fused loop should look like:
        # for i in range(0, 16):
        #   for j in range(0, 10):
        #      for f in range(2):
        #         if f == 0:
        #           C[i, j] += A[i, j]
        #         if f == 1:
        #           C[i, j] *= B[i, j]
        #   for j in range(10, 16):
        #      for f in range(2):
        #         if f == 1:
        #           C[i, j] *= B[i, j]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        C_ref = np.copy(C_test)

        C_ref[:, :A.shape[1]] = C_test[:, :A.shape[1]] + A_test    # nest0
        C_ref *= B_test    # nest1

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_ref],
        }
        self._verify_schedule(
            schedule,
            (A, B, C),
            "test_unequal_iteration_space_fusing_2",
            correctness_check_values,
        )

    def test_unequal_iteration_space_fusing_3(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT, shape=(16, 16))
        B = Array(role=Role.INPUT, shape=(16, 10))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 16))

        # Create nest0 and schedule
        nest0 = Nest(shape=(16, 16))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        # Create nest1 and schedule1 with a smaller iteration space size
        nest1 = Nest(shape=(16, 10))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        # Create a fused schedule: the smaller iteration space (nest1) should
        # be automatically end-padded with no-ops
        schedule = fuse(schedule0, schedule1)
        f, i, j = schedule.get_indices()

        # computing the output block-by-block:
        #  first computing C[0:4, 0:4] += A[0:4, 0:4]
        #  then computing C[0:4, 0:4] *= B[0:4, 0:4]
        ii, jj = schedule.tile({
            i: 4,
            j: 4
        })
        schedule.reorder(i, j, f, ii, jj)

        # Emitted fused loop should look like:
        # for i in range(0, 16, 4):
        #   # run both kernels in the smaller iteration spaces
        #   # (tiled block)
        #   for j in range(0, 8, 4):
        #       for f in range(2):
        #           if f == 0:
        #               for ii in range(0, 4):
        #                   for jj in range(0, 4):
        #                       C[i+ii, j+jj] += A[i+ii, j+jj]
        #           if f == 1:
        #               for ii in range(0, 4):
        #                   for jj in range(0, 4):
        #                       C[i+ii, j+jj] *= B[i+ii, j+jj]
        #
        #   # run both kernels in the smaller iteration space
        #   # (boundary block for split)
        #   for j in range(8, 10): # range < split size
        #       for f in range(2):
        #           if f == 0:
        #               for ii in range(0, 4):
        #                   C[i+ii, j] += A[i+ii, j]
        #           if f == 1:
        #               for ii in range(0, 4):
        #                   C[i+ii, j] *= B[i+ii, j]
        #
        #   # run kernel with the larger iteration space
        #   # (boundary block for split)
        #   for j in range(10, 16): # range < split size
        #       for f in range(2):
        #           if f == 0:
        #               for ii in range(0, 4):
        #                   C[i+ii, j] += A[i+ii, j]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        C_ref = C_test + A_test    # nest0
        C_ref[:, :B.shape[1]] = C_ref[:, :B.shape[1]] * B_test    # nest1

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_ref],
        }
        self._verify_schedule(
            schedule,
            (A, B, C),
            "test_unequal_iteration_space_fusing_3",
            correctness_check_values,
        )

    def test_concat_fusing_1(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(3, ))
        B = Array(role=Role.INPUT_OUTPUT, shape=(7, ))

        n1 = Nest(A.shape)
        n2 = Nest(B.shape)

        n1_i = n1.get_indices()

        @n1.iteration_logic
        def _():
            A[n1_i] /= A[n1_i]

        n2_i = n2.get_indices()

        @n2.iteration_logic
        def _():
            B[n2_i] *= B[n2_i]

        fused = fuse([n.create_schedule() for n in [n1, n2]], partial=0)

        # Emitted fused loop should look like:
        # for f in range(3):
        #     if f == 0:
        #         for i in range(3):
        #             A[i] /= A[i]
        #     if f == 1:
        #         for i in range(7):
        #             B[i] *= B[i]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)

        A_ref = A_test / A_test
        B_ref = B_test * B_test

        correctness_check_values = {
            "pre": [A_test, B_test],
            "post": [A_ref, B_ref]
        }
        self._verify_schedule(fused, (A, B), "test_concat_fusing_1", correctness_check_values)

    def test_concat_fusing_2(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(11, ))
        B = Array(role=Role.INPUT_OUTPUT, shape=(7, ))
        C = Array(role=Role.INPUT_OUTPUT, shape=(5, ))

        n1 = Nest(A.shape)
        n2 = Nest(B.shape)
        n3 = Nest(C.shape)

        n1_i = n1.get_indices()

        @n1.iteration_logic
        def _():
            A[n1_i] += A[n1_i]

        n2_i = n2.get_indices()

        @n2.iteration_logic
        def _():
            B[n2_i] *= B[n2_i]

        n3_i = n3.get_indices()

        @n3.iteration_logic
        def _():
            C[n3_i] /= C[n3_i]

        fused = fuse([n.create_schedule() for n in [n1, n2, n3]], partial=0)

        # Emitted fused loop should look like:
        # for f in range(3):
        #     if f == 0:
        #         for i in range(11):
        #           A[i}] += A[i}]
        #     if f == 1:
        #         for i in range(7):
        #           B[i}] *= B[i}]
        #     if f == 2:
        #         for i in range(5):
        #           C[i}] /= C[i}]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test + A_test
        B_ref = B_test * B_test
        C_ref = C_test / C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_schedule(fused, (A, B, C), "test_concat_fusing_2", correctness_check_values)

    def test_concat_fusing_3(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(3, 16))
        B = Array(role=Role.INPUT_OUTPUT, shape=(7, 16))

        n1 = Nest(A.shape)
        n2 = Nest(B.shape)

        n1_i, n1_j = n1.get_indices()

        @n1.iteration_logic
        def _():
            A[n1_i, n1_j] /= A[n1_i, n1_j]

        n2_i, n2_j = n2.get_indices()

        @n2.iteration_logic
        def _():
            B[n2_i, n2_j] *= B[n2_i, n2_j]

        fused = fuse([n.create_schedule() for n in [n1, n2]], partial=0)

        # Emitted fused loop should look like:
        # for f in range(3):
        #     if f == 0:
        #         for i in range(3):
        #             for j in range(16):
        #                 A[i,j] /= A[i,j]
        #     if f == 1:
        #         for i in range(7):
        #             for j in range(16):
        #                 B[i,j] *= B[i,j]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)

        A_ref = A_test / A_test
        B_ref = B_test * B_test

        correctness_check_values = {
            "pre": [A_test, B_test],
            "post": [A_ref, B_ref]
        }
        self._verify_schedule(fused, (A, B), "test_concat_fusing_3", correctness_check_values)

    def test_concat_fusing_4(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(11, 16))
        B = Array(role=Role.INPUT_OUTPUT, shape=(7, 16))
        C = Array(role=Role.INPUT_OUTPUT, shape=(5, 16))

        n1 = Nest(A.shape)
        n2 = Nest(B.shape)
        n3 = Nest(C.shape)

        n1_i, n1_j = n1.get_indices()

        @n1.iteration_logic
        def _():
            A[n1_i, n1_j] += A[n1_i, n1_j]

        n2_i, n2_j = n2.get_indices()

        @n2.iteration_logic
        def _():
            B[n2_i, n2_j] *= B[n2_i, n2_j]

        n3_i, n3_j = n3.get_indices()

        @n3.iteration_logic
        def _():
            C[n3_i, n3_j] /= C[n3_i, n3_j]

        fused = fuse([n.create_schedule() for n in [n1, n2, n3]], partial=0)

        # Emitted fused loop should look like:
        # for f in range(3):
        #     if f == 0:
        #         for i in range(11):
        #             for j in range(16):
        #                 A[i,j] += A[i,j]
        #     if f == 1:
        #         for i in range(7):
        #             for j in range(16):
        #                 B[i,j] *= B[i,j]
        #     if f == 2:
        #         for i in range(5):
        #             for j in range(16):
        #                 C[i,j] /= C[i,j]

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test + A_test
        B_ref = B_test * B_test
        C_ref = C_test / C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_schedule(fused, (A, B, C), "test_concat_fusing_4", correctness_check_values)

    def test_multi_concat_fusing_1(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(1024 + 13, ))
        B = Array(role=Role.INPUT_OUTPUT, shape=(1024 + 11, ))
        C = Array(role=Role.INPUT_OUTPUT, shape=(1024 + 7, ))
        D = Array(role=Role.INPUT_OUTPUT, shape=(1024 + 3, ))

        # Create nest0 and schedule
        nest0 = Nest(A.shape)
        i0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            A[i0] += A[i0]

        # Create nest1 and schedule1
        nest1 = Nest(B.shape)
        i1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            B[i1] *= B[i1]

        # Create a fused schedule
        s0, s1 = [n.create_schedule() for n in [nest0, nest1]]
        s0.split(i0, 11)
        s1.split(i1, 5)
        fused1 = fuse([s0, s1], partial=0)

        nest2 = Nest(C.shape)
        i2 = nest2.get_indices()

        @nest2.iteration_logic
        def _():
            C[i2] *= C[i2]

        s2 = nest2.create_schedule()
        s2.split(i2, 13)
        fused2 = fuse([fused1, s2], partial=0)

        nest3 = Nest(D.shape)
        i3 = nest3.get_indices()

        @nest3.iteration_logic
        def _():
            D[i3] *= D[i3]

        s3 = nest3.create_schedule()
        s3.split(i3, 7)
        fused3 = fuse([fused2, s3], partial=0)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        D_test = np.random.random(D.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test, D_test],
            "post": [
                A_test + A_test,
                B_test * B_test,
                C_test * C_test,
                D_test * D_test,
            ],
        }
        self._verify_schedule(fused3, (A, B, C, D), "test_multi_concat_fusing_1", correctness_check_values)

    def test_multi_partial_fusion_1(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(3, 11))
        B = Array(role=Role.INPUT_OUTPUT, shape=(3, 7))
        C = Array(role=Role.INPUT_OUTPUT, shape=(3, 5))

        n1 = Nest(A.shape)
        n2 = Nest(B.shape)
        n3 = Nest(C.shape)

        n1_i, n1_j = n1.get_indices()

        @n1.iteration_logic
        def _():
            A[n1_i, n1_j] += A[n1_i, n1_j]

        n2_i, n2_j = n2.get_indices()

        @n2.iteration_logic
        def _():
            B[n2_i, n2_j] *= B[n2_i, n2_j]

        n3_i, n3_j = n3.get_indices()

        @n3.iteration_logic
        def _():
            C[n3_i, n3_j] /= C[n3_i, n3_j]

        fused = fuse([n.create_schedule() for n in [n1, n2, n3]], partial=1)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test + A_test
        B_ref = B_test * B_test
        C_ref = C_test / C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_schedule(fused, (A, B, C), "test_multi_partial_fusion_1", correctness_check_values)

    def test_multi_partial_fusion_2(self) -> None:
        from accera import fuse, Nest

        A = Array(role=Role.INPUT_OUTPUT, shape=(3, 11, 4))
        B = Array(role=Role.INPUT_OUTPUT, shape=(3, 7))
        C = Array(role=Role.INPUT_OUTPUT, shape=(3, 5, 6, 8))

        n1 = Nest(A.shape)
        n2 = Nest(B.shape)
        n3 = Nest(C.shape)

        n1_i, n1_j, n1_k = n1.get_indices()

        @n1.iteration_logic
        def _():
            A[n1_i, n1_j, n1_k] += A[n1_i, n1_j, n1_k]

        n2_i, n2_j = n2.get_indices()

        @n2.iteration_logic
        def _():
            B[n2_i, n2_j] *= B[n2_i, n2_j]

        n3_i, n3_j, n3_k, n3_k2 = n3.get_indices()

        @n3.iteration_logic
        def _():
            C[n3_i, n3_j, n3_k, n3_k2] /= C[n3_i, n3_j, n3_k, n3_k2]

        fused = fuse([n.create_schedule() for n in [n1, n2, n3]], partial=1)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test + A_test
        B_ref = B_test * B_test
        C_ref = C_test / C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_schedule(fused, (A, B, C), "test_multi_partial_fusion_2", correctness_check_values)

    def test_hierarchical_partial_fuse(self) -> None:
        from accera import fuse

        M = 256
        N = 128
        M_tile = 32
        N_tile = 16
        A = Array(role=Role.INPUT, shape=(M, ))
        B = Array(role=Role.INPUT, shape=(N, ))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, N))

        # Create nest0 and schedule
        nest0 = Nest(shape=(M, N))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0] * B[j0]

        schedule0 = nest0.create_schedule()
        ii0, jj0 = schedule0.tile({
            i0: M_tile,
            j0: N_tile
        })
        schedule0.reorder(i0, j0, ii0, jj0)

        # Create nest1 and schedule1
        nest1 = Nest(shape=(M, N))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] = C[i1, j1] * 0.1

        schedule1 = nest1.create_schedule()
        ii1, jj1 = schedule1.tile({
            i1: M_tile,
            j1: N_tile
        })
        schedule1.reorder(i1, j1, ii1, jj1)

        schedule_01 = fuse((schedule0, schedule1), partial=2)
        f, i, j, ii0, jj0, ii1, jj1 = schedule_01.get_indices()
        schedule_01.reorder(i, j, f, ii0, jj0, ii1, jj1)

        # Create nest2 and schedule2
        nest2 = Nest(shape=(M, N))
        i2, j2 = nest2.get_indices()

        @nest2.iteration_logic
        def _():
            C[i2, j2] = C[i2, j2] + 0.2

        schedule2 = nest2.create_schedule()
        ii2, jj2 = schedule2.tile({
            i2: M_tile,
            j2: N_tile
        })
        schedule2.reorder(i2, j2, ii2, jj2)

        # Create nest3 and schedule3
        nest3 = Nest(shape=(M, N))
        i3, j3 = nest3.get_indices()

        @nest3.iteration_logic
        def _():
            C[i3, j3] = C[i3, j3] + 0.3

        schedule3 = nest3.create_schedule()
        ii3, jj3 = schedule3.tile({
            i3: M_tile,
            j3: N_tile
        })
        schedule3.reorder(i3, j3, ii3, jj3)

        schedule_23 = fuse((schedule2, schedule3), partial=2)
        f_23, i_23, j_23, ii2, jj2, ii3, jj3 = schedule_23.get_indices()
        schedule_23.reorder(i_23, j_23, f_23, ii2, jj2, ii3, jj3)

        schedule_0123 = fuse((schedule_01, schedule_23), partial=1)
        f_0123, i_0123, j_01, f_01, ii0, jj0, ii1, jj1, j_23, f_23, ii2, jj2, ii3, jj3 = schedule_0123.get_indices()
        schedule_0123.reorder(i_0123, f_0123, j_01, f_01, ii0, jj0, ii1, jj1, j_23, f_23, ii2, jj2, ii3, jj3)

        plan = schedule_0123.create_plan()

        # Create a package and add our function definition to it
        package_name = "test_hierarchical_partial_fuse"
        package = Package()
        package.add(plan, args=(A, B, C), base_name="test_hierarchical_partial_fuse")

        self._verify_schedule(plan, (A, B, C), "test_hierarchical_partial_fuse", None)

    def test_nested_nests_matmul(self):
        test_name = "test_nested_nests_matmul"

        M = 20
        N = 32
        K = 12
        M_tile = 4
        N_tile = 16
        K_tile = 3

        package = Package()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        B_temp = Array(role=Role.TEMP, element_type=ScalarType.float32, shape=(K_tile, N_tile))
        io_B_temp = Array(role=Role.INPUT_OUTPUT, element_type=B_temp.element_type, shape=B_temp.shape)

        i_tile_idx = Dimension()
        j_tile_idx = Dimension()
        k_tile_idx = Dimension()

        pack_b_nest = Nest([K_tile, N_tile])
        pb_k, pb_j = pack_b_nest.get_indices()

        @pack_b_nest.iteration_logic
        def _pack_b():
            full_k = pb_k + k_tile_idx
            full_j = pb_j + j_tile_idx
            io_B_temp[pb_k, pb_j] = B[full_k, full_j]

        pack_b_fn = package.add(pack_b_nest, args=(B, io_B_temp, j_tile_idx, k_tile_idx), base_name="pack_b_tile_fn")

        matmul_nest = Nest([M_tile, N_tile, K_tile])
        mm_i, mm_j, mm_k = matmul_nest.get_indices()

        @matmul_nest.iteration_logic
        def _matmul():
            full_i = mm_i + i_tile_idx
            full_j = mm_j + j_tile_idx
            full_k = mm_k + k_tile_idx
            C[full_i, full_j] += A[full_i, full_k] * io_B_temp[mm_k, mm_j]

        matmul_sched = matmul_nest.create_schedule()
        mm_jj = matmul_sched.split(mm_j, 8)
        matmul_sched.reorder(mm_k, mm_i, mm_j, mm_jj)
        matmul_plan = matmul_sched.create_plan()
        matmul_plan.vectorize(mm_jj)
        matmul_fn = package.add(
            matmul_plan, args=(A, B, C, io_B_temp, i_tile_idx, j_tile_idx, k_tile_idx), base_name="matmul_tile_fn"
        )

        tile_nest = Nest([M, N, K])
        i, j, k = tile_nest.get_indices()

        @tile_nest.iteration_logic
        def _tile_logic():
            pack_b_fn(B, B_temp, j, k)
            matmul_fn(A, B, C, B_temp, i, j, k)

        tile_sched = tile_nest.create_schedule()
        ii, jj, kk = tile_sched.tile(dict(zip([i, j, k], [M_tile, N_tile, K_tile])))
        tile_sched.reorder(i, j, k, ii, jj, kk)
        tile_plan = tile_sched.create_plan()
        tile_plan._erase_loops([ii, jj, kk])
        full_fn = package.add(tile_plan, args=(A, B, C), base_name="full_matmul_fn")

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test
        B_ref = B_test
        C_ref = A_test @ B_test + C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_func(package, full_fn, test_name, correctness_check_values, mode=Package.Mode.RELEASE)

    def test_nested_nests_matmul_boundary(self):
        test_name = "test_nested_nests_matmul_boundary"
        from accera import min

        M = 20
        N = 32
        K = 12
        M_tile = 4
        N_tile = 12    # 32 doesn't divide 12 so we should have an 8 element boundary in the N dimension
        K_tile = 3

        package = Package()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        B_temp = Array(role=Role.TEMP, element_type=ScalarType.float32, shape=(K_tile, N_tile))
        io_B_temp = Array(role=Role.INPUT_OUTPUT, element_type=B_temp.element_type, shape=B_temp.shape)

        i_tile_idx = Dimension()
        j_tile_idx = Dimension()
        k_tile_idx = Dimension()

        n_tile_dim = Dimension()

        pack_b_nest = Nest([K_tile, n_tile_dim])
        pb_k, pb_j = pack_b_nest.get_indices()

        @pack_b_nest.iteration_logic
        def _pack_b():
            full_k = pb_k + k_tile_idx
            full_j = pb_j + j_tile_idx
            io_B_temp[pb_k, pb_j] = B[full_k, full_j]

        pack_b_fn = package.add(
            pack_b_nest, args=(n_tile_dim, B, io_B_temp, j_tile_idx, k_tile_idx), base_name="pack_b_tile_fn"
        )

        matmul_nest = Nest([M_tile, n_tile_dim, K_tile])
        mm_i, mm_j, mm_k = matmul_nest.get_indices()

        @matmul_nest.iteration_logic
        def _matmul():
            full_i = mm_i + i_tile_idx
            full_j = mm_j + j_tile_idx
            full_k = mm_k + k_tile_idx
            C[full_i, full_j] += A[full_i, full_k] * io_B_temp[mm_k, mm_j]

        matmul_sched = matmul_nest.create_schedule()
        mm_jj = matmul_sched.split(mm_j, 8)
        matmul_sched.reorder(mm_k, mm_i, mm_j, mm_jj)
        matmul_plan = matmul_sched.create_plan()
        matmul_fn = package.add(
            matmul_plan,
            args=(n_tile_dim, A, B, C, io_B_temp, i_tile_idx, j_tile_idx, k_tile_idx),
            base_name="matmul_tile_fn"
        )

        tile_nest = Nest([M, N, K])
        i, j, k = tile_nest.get_indices()

        @tile_nest.iteration_logic
        def _tile_logic():
            n_tile_extent = min(cast(N_tile, ScalarType.index), cast(N, ScalarType.index) - j)
            pack_b_fn(n_tile_extent, B, B_temp, j, k)
            matmul_fn(n_tile_extent, A, B, C, B_temp, i, j, k)

        tile_sched = tile_nest.create_schedule()
        ii, jj, kk = tile_sched.tile(dict(zip([i, j, k], [M_tile, N_tile, K_tile])))
        tile_sched.reorder(i, j, k, ii, jj, kk)
        tile_plan = tile_sched.create_plan()
        tile_plan._erase_loops([ii, jj, kk])
        full_fn = package.add(tile_plan, args=(A, B, C), base_name="full_matmul_fn")

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test
        B_ref = B_test
        C_ref = A_test @ B_test + C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_func(package, full_fn, test_name, correctness_check_values, mode=Package.Mode.RELEASE)

    def test_double_nested_nests_matmul_boundary(self):
        test_name = "test_double_nested_nests_matmul_boundary"
        from accera import min

        M = 20
        N = 32
        K = 12
        M_tile = 4
        N_tile = 12    # 32 doesn't divide 12 so we should have an 8 element boundary in the N dimension
        N_kernel_tile = 8    # Doesn't divide N_tile so we should have a 4 element boundary in the N dimension in the outer main loop and no inner boundary in the outer boundary loop
        K_tile = 3

        package = Package()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        B_temp = Array(role=Role.TEMP, element_type=ScalarType.float32, shape=(K_tile, N_tile))
        io_B_temp = Array(role=Role.INPUT_OUTPUT, element_type=B_temp.element_type, shape=B_temp.shape)

        n_tile_dim = Dimension()
        n_kernel_dim = Dimension()

        i_tile_idx = Dimension()
        j_tile_idx = Dimension()
        k_tile_idx = Dimension()

        i_kernel_idx = Dimension()
        j_kernel_idx = Dimension()
        k_kernel_idx = Dimension()

        pack_b_nest = Nest([K_tile, n_tile_dim])
        pb_k, pb_j = pack_b_nest.get_indices()

        @pack_b_nest.iteration_logic
        def _pack_b():
            full_k = pb_k + k_tile_idx
            full_j = pb_j + i_tile_idx
            io_B_temp[pb_k, pb_j] = B[full_k, full_j]

        pack_b_fn = package.add(
            pack_b_nest,
            args=(n_tile_dim, B, io_B_temp, i_tile_idx, k_tile_idx),
            base_name="pack_b_tile_fn",
            function_opts=INTERNAL_FUNCTION_OPTS
        )

        matmul_kernel_nest = Nest((n_kernel_dim, ))
        mmk_j = matmul_kernel_nest.get_indices()

        @matmul_kernel_nest.iteration_logic
        def _matmul():
            tile_j = mmk_j + j_kernel_idx

            full_i = i_kernel_idx + i_tile_idx
            full_j = tile_j + j_tile_idx
            full_k = k_kernel_idx + k_tile_idx
            C[full_i, full_j] += A[full_i, full_k] * io_B_temp[k_kernel_idx, tile_j]

        matmul_kernel_sched = matmul_kernel_nest.create_schedule()
        mmk_jj = matmul_kernel_sched.split(mmk_j, N_kernel_tile)
        matmul_kernel_sched.reorder(mmk_j, mmk_jj)
        matmul_kernel_plan = matmul_kernel_sched.create_plan()
        matmul_kernel_fn = package.add(
            matmul_kernel_plan,
            args=(
                n_kernel_dim, A, B, C, io_B_temp, i_tile_idx, j_tile_idx, k_tile_idx, i_kernel_idx, j_kernel_idx,
                k_kernel_idx
            ),
            base_name="matmul_kernel_fn",
            function_opts=INTERNAL_FUNCTION_OPTS
        )

        matmul_tile_nest = Nest([M_tile, n_tile_dim, K_tile])
        mm_i, mm_j, mm_k = matmul_tile_nest.get_indices()

        @matmul_tile_nest.iteration_logic
        def _matmul():
            n_kernel_extent = min(cast(N_kernel_tile, ScalarType.index), n_tile_dim - mm_j)
            matmul_kernel_fn(n_kernel_extent, A, B, C, io_B_temp, i_tile_idx, j_tile_idx, k_tile_idx, mm_i, mm_j, mm_k)

        matmul_tile_sched = matmul_tile_nest.create_schedule()
        mm_jj = matmul_tile_sched.split(mm_j, N_tile)
        mm_jjj = matmul_tile_sched.split(mm_jj, N_kernel_tile)
        matmul_tile_sched.reorder(mm_k, mm_i, mm_j, mm_jj, mm_jjj)
        matmul_tile_plan = matmul_tile_sched.create_plan()
        matmul_tile_plan._erase_loops([mm_jjj])
        matmul_tile_fn = package.add(
            matmul_tile_plan,
            args=(n_tile_dim, A, B, C, io_B_temp, i_tile_idx, j_tile_idx, k_tile_idx),
            base_name="matmul_tile_fn",
            function_opts=INTERNAL_FUNCTION_OPTS
        )

        tile_nest = Nest([M, N, K])
        i, j, k = tile_nest.get_indices()

        @tile_nest.iteration_logic
        def _tile_logic():
            n_tile_extent = min(cast(N_tile, ScalarType.index), cast(N, ScalarType.index) - j)
            pack_b_fn(n_tile_extent, B, B_temp, j, k)
            matmul_tile_fn(n_tile_extent, A, B, C, B_temp, i, j, k)

        tile_sched = tile_nest.create_schedule()
        ii, jj, kk = tile_sched.tile(dict(zip([i, j, k], [M_tile, N_tile, K_tile])))
        tile_sched.reorder(i, j, k, ii, jj, kk)
        tile_plan = tile_sched.create_plan()
        tile_plan._erase_loops([ii, jj, kk])
        full_fn = package.add(tile_plan, args=(A, B, C), base_name="full_matmul_fn")

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)

        A_ref = A_test
        B_ref = B_test
        C_ref = A_test @ B_test + C_test

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_ref, B_ref, C_ref],
        }
        self._verify_func(package, full_fn, test_name, correctness_check_values, mode=Package.Mode.RELEASE)


class DSLTest_05Targets(unittest.TestCase):
    def test_known_targets(self) -> None:
        intel_name = "Intel 6400"
        intel = Target(known_name=intel_name, num_threads=44)
        self.assertEqual(intel.name, intel_name)
        self.assertEqual(intel.num_threads, 44)    # override
        self.assertEqual(intel.vector_bytes, 32)    # default
        self.assertEqual(intel.vector_registers, 16)    # default
        self.assertEqual(intel.category, Target.Category.CPU)    # default

        pi3_name = "Raspberry Pi 3B"
        pi3 = Target(
            Target.Model.RASPBERRY_PI_3B,
            category=Target.Category.CPU,
            frequency_GHz=1.2,
        )
        self.assertEqual(pi3.name, pi3_name)
        self.assertEqual(pi3.num_threads, 8)
        self.assertEqual(pi3.category, Target.Category.CPU)

    def test_custom_targets(self) -> None:
        my_target = Target(
            name="Custom processor",
            category=Target.Category.CPU,
            architecture="x86_64",
            family="Broadwell",
            extensions=[
                "MMX",
                "SSE",
                "SSE2",
                "SSE3",
                "SSSE3",
                "SSE4",
                "SSE4.1",
                "SSE4.2",
                "AVX",
                "AVX2",
                "FMA3",
            ],
            num_cores=22,
            num_threads=44,
            frequency_GHz=3.2,
            turbo_frequency_GHz=3.8,
            cache_sizes=[32, 256, 56320],
            cache_lines=[64, 64, 64],
        )
        self.assertEqual(my_target.name, "Custom processor")
        self.assertEqual(my_target.category, Target.Category.CPU)
        self.assertEqual(my_target.architecture, "x86_64")
        self.assertTrue("SSE3" in my_target.extensions)

    def test_gpu_targets(self) -> None:
        v100_name = "NVidia V100"
        v100 = Target(Target.Model.NVIDIA_V100, category=Target.Category.GPU)
        self.assertEqual(v100.name, v100_name)
        self.assertEqual(v100.category, Target.Category.GPU)
        self.assertEqual(v100.warp_size, 32)

        mi100 = Target(Target.Model.AMD_MI100)
        self.assertEqual(mi100.warp_size, 64)
        self.assertEqual(mi100.frequency_GHz, 1.502)

        a100 = Target(Target.Model.NVIDIA_RTX_A6000)
        self.assertEqual(a100.warp_size, 32)


class DSLTest_06PlansCaching(unittest.TestCase):
    def _create_plan(self, shape: Tuple[int], type=ScalarType.float32) -> Tuple:
        M, N, S = shape

        A = Array(role=Role.INPUT, element_type=type, shape=(M, S))
        B = Array(
            role=Role.INPUT,
            element_type=type,
            shape=(S, N),
            layout=Array.Layout.LAST_MAJOR,
        )    # use a different caching layout
        C = Array(role=Role.INPUT_OUTPUT, element_type=type, shape=(M, N))

        nest = Nest(shape=(M, N, S))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        plan = nest.create_plan()

        return plan, [A, B, C], [i, j, k]

    def _verify_plan(self, plan, args: Tuple[Array], package_name, correctness_check_values=None) -> None:
        # create a HAT package and add the function to it
        package = Package()
        function = package.add(plan, args, base_name="caching_test")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        # build the HAT package
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_caching_by_level(self) -> None:
        plan, args, indices = self._create_plan((16, 10, 11))
        A, B, C = args
        _, j, _ = indices

        AA = plan.cache(A, level=2)
        self.assertEqual(AA.index, j)

        # input, different layout
        BB = plan.cache(B, level=2, layout=Array.Layout.FIRST_MAJOR)
        self.assertEqual(BB.index, j)

        self._verify_plan(plan, [A, B, C], "test_caching_by_level")

    def test_caching_by_index(self) -> None:
        plan, args, indices = self._create_plan((16, 10, 11))
        A, B, C = args
        _, j, _ = indices

        with self.assertRaises(ValueError):
            AA = plan.cache(A, index=j, level=1)

        AA = plan.cache(A, index=j)    # input
        self.assertEqual(AA.index, j)

        # input, different layout
        BB = plan.cache(B, index=j, layout=Array.Layout.FIRST_MAJOR)
        self.assertEqual(BB.index, j)

        CC = plan.cache(C, index=j)    # input/output
        self.assertEqual(CC.index, j)

        self._verify_plan(plan, [A, B, C], "test_caching_by_index")

    def test_caching_by_element_budget(self) -> None:
        plan, args, _ = self._create_plan((256, 10, 11))
        A, B, C = args

        AA = plan.cache(A, max_elements=1024)
        self.assertEqual(AA.index, None)
        self.assertEqual(AA.max_elements, 1024)

        self._verify_plan(plan, [A, B, C], "test_caching_by_element_budget")

    def test_thrifty_caching(self) -> None:
        plan, args, indices = self._create_plan((16, 10, 11))
        A, B, C = args
        _, j, k = indices

        # A is row-major, thrifty mode should skip caching
        AA = plan.cache(A, thrifty=True, index=j)
        self.assertIsNotNone(AA)

        # B is column-major, thrifty mode should cache
        BB = plan.cache(B, thrifty=True, index=k)
        self.assertIsNotNone(BB)

        self._verify_plan(plan, [A, B, C], "test_thrifty_caching")

    @expectedFailure(FailedReason.NOT_IN_PY, "Various target memory identifiers")
    def test_cache_mapping(self) -> None:
        A = Array(role=Role.INPUT, shape=(1024, ))

        nest = Nest(shape=(64, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i] += 2

        v100 = Target(Target.Model.NVIDIA_V100, category=Target.Category.GPU, num_threads=16)
        plan = nest.create_plan(v100)

        plan.cache(i, type=v100.MemorySpace.SHARED)
        self._verify_plan(plan, [A], "test_cache_mapping")

    def test_cache_trigger_level(self) -> None:
        A = Array(role=Role.INPUT, shape=(1024, 1024))
        B = Array(role=Role.INPUT_OUTPUT, shape=(1024, 1024))

        nest = Nest(shape=(1024, 1024))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i, j] += A[i, j]

        schedule = nest.create_schedule()
        ii = schedule.split(i, 128)
        jj = schedule.split(j, 256)
        schedule.reorder(i, j, ii, jj)

        plan = schedule.create_plan()

        plan.cache(A, index=ii, trigger_index=j)

        self._verify_plan(plan, [A, B], "test_cache_trigger_level")

    def test_cache_trigger_level_matmul(self) -> None:
        M = 1024
        N = 1024
        S = 1024

        A = Array(role=Role.INPUT, shape=(M, S))
        B = Array(role=Role.INPUT, shape=(S, N))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, N))

        nest = Nest(shape=(M, N, S))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        jj = schedule.split(j, 128)
        kk = schedule.split(k, 256)
        kkk = schedule.split(kk, 4)
        jjj = schedule.split(jj, 16)
        jjjj = schedule.split(jjj, 8)
        ii = schedule.split(i, 6)

        schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)
        plan = schedule.create_plan()
        plan.cache(B, index=kkk, trigger_index=k, layout=Array.Layout.FIRST_MAJOR)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_test + A_test @ B_test],
        }

        self._verify_plan(
            plan,
            [A, B, C],
            "test_cache_trigger_level_matmul",
            correctness_check_values=correctness_check_values,
        )

    def test_hierachical_caching(self) -> None:
        M = 1024
        N = 1024
        S = 1024

        A = Array(role=Role.INPUT, shape=(M, S))
        B = Array(role=Role.INPUT, shape=(S, N))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, N))

        nest = Nest(shape=(M, N, S))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        jj = schedule.split(j, 128)
        kk = schedule.split(k, 256)
        kkk = schedule.split(kk, 4)
        jjj = schedule.split(jj, 16)
        jjjj = schedule.split(jjj, 8)
        ii = schedule.split(i, 6)

        schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)
        plan = schedule.create_plan()

        AA = plan.cache(A, level=5, trigger_level=7, layout=Array.Layout.FIRST_MAJOR)
        AAA = plan.cache(AA, level=3, trigger_level=5, layout=Array.Layout.LAST_MAJOR)
        BB = plan.cache(B, level=6, trigger_level=7, layout=Array.Layout.FIRST_MAJOR)
        BBB = plan.cache(BB, level=2, trigger_level=5, layout=Array.Layout.LAST_MAJOR)
        CC = plan.cache(C, level=8, layout=Array.Layout.FIRST_MAJOR)
        CCC = plan.cache(CC, level=6, layout=Array.Layout.LAST_MAJOR)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_test + A_test @ B_test],
        }

        self._verify_plan(
            plan,
            [A, B, C],
            "test_hierarchical_caching",
            correctness_check_values=correctness_check_values,
        )


class DSLTest_07PlansVectorizationParallelization(unittest.TestCase):
    def _verify_plan(self, plan, args: Tuple[int], package_name, correctness_check_values=None) -> None:
        package = Package()
        function = package.add(plan, args, base_name="vectorization_parallelization_test")

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_unroll(self) -> None:
        from accera import Target, Nest

        A = Array(role=Role.INPUT, shape=(3, 5))

        my_target = Target(category=Target.Category.CPU)

        nest = Nest(shape=(3, 5))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] *= 2.0

        plan1 = nest.create_plan(my_target)
        plan1.unroll(index=j)
        self._verify_plan(plan1, [A], "test_unroll1")

        plan2 = nest.create_plan(my_target)
        plan2.unroll(index=i)
        self._verify_plan(plan2, [A], "test_unroll2")

    def test_vectorize(self) -> None:
        from accera import Target, Nest

        A = Array(role=Role.INPUT, shape=(64, ))
        B = Array(role=Role.INPUT, shape=(64, ))
        C = Array(role=Role.INPUT_OUTPUT, shape=(64, ))

        my_target = Target(category=Target.Category.CPU, vector_bytes=16, vector_registers=2)

        nest = Nest(shape=(64, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i] = A[i] * B[i]

        plan = nest.create_plan(my_target)
        plan.vectorize(index=i)
        self._verify_plan(plan, [A, B, C], "test_vectorize")

    def test_kernelize(self) -> None:
        from accera import Target, Nest

        A = Array(role=Role.INPUT, shape=(16, 11))
        B = Array(role=Role.INPUT, shape=(11, 10))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 10))

        nest = Nest(shape=(16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        my_target = Target(category=Target.Category.CPU, vector_bytes=16, vector_registers=2)
        plan = nest.create_plan(my_target)

        # Shorthand for:
        # plan.unroll(i)
        # plan.unroll(j)
        # plan.vectorize(k)
        plan.kernelize(unroll_indices=(i, j), vectorize_indices=k)
        self._verify_plan(plan, [A, B, C], "test_kernelize")

    def test_kernelize_2(self) -> None:
        from accera import Target, Nest

        A = Array(role=Role.INPUT, shape=(16, 16))
        B = Array(role=Role.INPUT, shape=(16, 16))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 16))

        nest = Nest(shape=(16, 16, 16))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        my_target = Target(category=Target.Category.CPU, vector_bytes=16, vector_registers=2)
        plan = nest.create_plan(my_target)

        # Shorthand for:
        # plan.unroll(i)
        # plan.vectorize(j)
        # plan.vectorize(k)
        plan.kernelize(unroll_indices=(i, ), vectorize_indices=(j, k))
        self._verify_plan(plan, [A, B, C], "test_kernelize_2")

    @expectedFailure(FailedReason.NOT_IN_PY, "pinning parallelization to CPU cores")
    def test_cpu_bind(self) -> None:
        A = Array(role=Role.INPUT, shape=(16, 11))
        B = Array(role=Role.INPUT, shape=(11, 10))
        C = Array(role=Role.INPUT_OUTPUT, shape=(16, 10))

        nest = Nest(shape=(16, 10, 11))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        target = Target("HOST", num_threads=16)
        plan = nest.create_plan(target)

        plan.parallelize(indices=(i, j, k), pin=(target.cores[0], target.cores[1]))    # TODO: confirm syntax
        self._verify_plan(plan, [A, B, C], "test_cpu_bind")

    def test_gpu_bind(self) -> None:
        M = 128
        N = 256
        K = 256
        A = Array(role=Role.INPUT, shape=(M, K))
        B = Array(role=Role.INPUT, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        v100 = Target(Target.Model.NVIDIA_V100, category=Target.Category.GPU)
        plan = nest.create_plan(v100)

        plan.bind(mapping={
            i: v100.GridUnit.BLOCK_X,
            j: v100.GridUnit.THREAD_X,
            k: v100.GridUnit.THREAD_Y,
        })

        test_name = "test_gpu_bind"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name

        with verifiers.VerifyPackage(
                self,
                test_name,
                output_dir,
                file_list=[f"{test_name}.cu", f"{test_name}.hat"],
        ) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG not supported
                output_dir=output_dir,
            )

    def test_gpu_multi_index_bind(self) -> None:
        M = 128
        N = 256
        K = 256
        A = Array(role=Role.INPUT, shape=(M, K))
        B = Array(role=Role.INPUT, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()
        ii, jj = schedule.tile({
            i: 32,
            j: 32
        })

        iii, jjj = schedule.tile({
            ii: 4,
            jj: 16
        })

        schedule.reorder(i, j, ii, jj, iii, jjj, k)

        v100 = Target(Target.Model.NVIDIA_V100, category=Target.Category.GPU)
        plan = schedule.create_plan(v100)

        plan.bind(
            mapping={
                i: v100.GridUnit.BLOCK_X,
                j: v100.GridUnit.BLOCK_Y,
                (ii, jj): v100.GridUnit.THREAD_X,
                jjj: v100.GridUnit.THREAD_Y
            }
        )

        test_name = "test_gpu_multi_index_bind"
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG not supported
                output_dir=output_dir
            )

    def test_scheduling_strategies(self) -> None:
        A = Array(role=Role.INPUT, shape=(256, 1024))
        B = Array(role=Role.INPUT, shape=(1024, 512))
        C = Array(role=Role.INPUT_OUTPUT, shape=(256, 512))

        nest = Nest(shape=(256, 512, 1024))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        target = Target("HOST", num_threads=16)

        # disable correctness checking on windows because the
        # install location of libomp.dll is non-standard as of now
        if sys.platform.startswith("win"):
            correctness_check_values = None
        else:
            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)
            correctness_check_values = {
                "pre": [A_test, B_test, C_test],
                "post": [A_test, B_test, C_test + A_test @ B_test],
            }

        schedule = nest.create_schedule()
        ii = schedule.split(i, A.shape[0] // min(4, target.num_threads))
        # set the index (k) that cannot be parallelized as innermost
        schedule.reorder(i, ii, j, k)

        for policy in ["static", "dynamic"]:
            plan = schedule.create_plan(target)

            # wrong order
            with self.assertRaises(ValueError):
                plan.parallelize(indices=(k, ii), policy=policy)

            # non-contiguous
            with self.assertRaises(ValueError):
                plan.parallelize(indices=(i, j), policy=policy)

            # non-collapsed
            plan.parallelize(indices=i, policy=policy)
            self._verify_plan(
                plan,
                [A, B, C],
                f"test_parallelize_i_{policy}",
                correctness_check_values,
            )

            # parallelizing middle index
            plan_ii = schedule.create_plan(target)
            plan_ii.parallelize(indices=ii, policy=policy)
            self._verify_plan(
                plan_ii,
                [A, B, C],
                f"test_parallelize_ii_{policy}",
                correctness_check_values,
            )

            try:
                # partial collapsed
                plan_partial = schedule.create_plan(target)
                plan_partial.parallelize(indices=(i, ii, j), policy=policy)
                self._verify_plan(
                    plan_partial,
                    [A, B, C],
                    f"test_parallelize_i_ii_j_{policy}",
                    correctness_check_values,
                )

                # partial collapsed inner indices
                plan_partial_inner = schedule.create_plan(target)
                plan_partial_inner.parallelize(indices=(ii, j), policy=policy)
                self._verify_plan(
                    plan_partial_inner,
                    [A, B, C],
                    f"test_parallelize_ii_j_{policy}",
                    correctness_check_values,
                )
            except:
                # BUGBUG: partial collapsed + dynamic is broken in mlir-translate since LLVM 14
                # 3  libsystem_platform.dylib 0x00000001a4ce74a4 _sigtramp + 56
                # 4  mlir-translate           0x000000010259dea4 llvm::OpenMPIRBuilder::createParallel(
                #       llvm::OpenMPIRBuilder::LocationDescription const&, llvm::IRBuilderBase::InsertPoint,
                #       llvm::function_ref<void (llvm::IRBuilderBase::InsertPoint, llvm::IRBuilderBase::InsertPoint,
                #       llvm::BasicBlock&)>, llvm::function_ref<llvm::IRBuilderBase::InsertPoint (llvm::IRBuilderBase::InsertPoint,
                #       llvm::IRBuilderBase::InsertPoint, llvm::Value&, llvm::Value&, llvm::Value*&)>,
                #           std::__1::function<void(llvm::IRBuilderBase::InsertPoint)>, llvm::Value*, llvm::Value*,
                #           llvm::omp::ProcBindKind, bool) + 2972
                if policy == "dynamic":
                    pass

            # fully collapsed will result in correctness issues because parallelizing k can stomp on the C matrix
            # where multiple threads try to update C[i, j] for different values of k


class DSLTest_08DeferredLayout(unittest.TestCase):
    def _verify_package(self, plan, args, package_name, correctness_check_values) -> None:
        package = Package()
        function = package.add(plan, args, base_name="deferred_layout")

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_deferred_layout_predefined(self) -> None:
        matrix = np.random.rand(128, 128).astype(np.float32)
        B_test = np.random.random(matrix.shape).astype(np.float32)

        for layout in [Array.Layout.FIRST_MAJOR, Array.Layout.LAST_MAJOR]:
            A = Array(role=Role.CONST, data=matrix, layout=Array.Layout.DEFERRED)
            B = Array(
                role=Role.INPUT_OUTPUT,
                element_type=ScalarType.float32,
                shape=matrix.shape,
            )

            nest = Nest(shape=matrix.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                B[i, j] += A[i, j]

            # create a cache for the constant array
            plan1 = nest.create_plan()
            AA = plan1.cache(A, i, layout=layout)    # , thrifty=True) # TODO

            # create another cache, using a different plan, for testing purposes
            plan2 = nest.create_plan()
            BB = plan2.cache(B, i)

            with self.assertRaises(ValueError):
                B.deferred_layout(cache=BB)    # non-const array

            with self.assertRaises(ValueError):
                A.deferred_layout(cache=BB)    # wrong cache

            # update the constant array's layout based on the cache
            A.deferred_layout(cache=AA)
            self.assertEqual(A.layout, AA.layout)

            with self.assertRaises(ValueError):
                A.deferred_layout(cache=AA)    # duplicate

            package_name = f"test_deferred_layout_predefined_{layout}".replace(".", "_")    # sanitize path name

            self._verify_package(plan1, (B, ), package_name, {
                "pre": [B_test],
                "post": [B_test + matrix]
            })

    def test_deferred_layout_coefficients(self) -> None:
        matrix = np.random.rand(128, 128).astype(np.float32)
        B_test = np.random.random(matrix.shape).astype(np.float32)

        for layout in [(128, 1), (1, 128)]:
            A = Array(role=Role.CONST, data=matrix, layout=Array.Layout.DEFERRED)
            B = Array(
                role=Role.INPUT_OUTPUT,
                element_type=ScalarType.float32,
                shape=matrix.shape,
            )

            nest = Nest(shape=matrix.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                B[i, j] += A[i, j]

            plan = nest.create_plan()
            AA = plan.cache(A, i, layout=layout)    # , thrifty=True) # TODO

            A.deferred_layout(cache=AA)
            self.assertEqual(A.layout, AA.layout)

            package_name = (f"test_deferred_layout_coefficients_{'_'.join(map(str, layout))}")
            self._verify_package(plan, (B, ), package_name, {
                "pre": [B_test],
                "post": [B_test + matrix]
            })


class DSLTest_09Parameters(unittest.TestCase):
    def test_parameterization_1(self) -> None:
        from accera import create_parameters, Nest

        P0, P1, P2, P3 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        package = Package()
        package_name = "test_parameterization_1"

        # Use the templated nest to add two different functions to the package
        package.add(
            nest,
            args=(A, B, C),
            parameters={
                P0: 16,
                P1: 16,
                P2: 16,
                P3: 1.0
            },
            base_name="matmul_16_16_16_1",
        )
        package.add(
            nest,
            args=(A, B, C),
            parameters={
                P0: 32,
                P1: 32,
                P2: 32,
                P3: 2.0
            },
            base_name="matmul_32_32_32_2",
        )

        P4, P5 = create_parameters()

        # Create a parameterized schedule
        schedule = nest.create_schedule()
        ii = schedule.split(i, size=P4)

        P6 = create_parameters()
        schedule.reorder(order=P6)

        # Create a parameterized plan
        plan = schedule.create_plan()
        plan.cache(A, level=P5)

        # Add another function to the package
        package.add(
            plan,
            args=(A, B, C),
            parameters={
                P0: 16,
                P1: 16,
                P2: 16,
                P3: 1.0,
                P4: 4,
                P5: 2,
                P6: (j, k, i, ii),
            },
            base_name="alternative_matmul_16_16_16",
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_parameterization_2(self) -> None:
        from accera import create_parameters, Nest

        P0, P1, P2, P3 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        package = Package()
        package_name = "test_parameterization_2"

        P4, P5 = create_parameters()

        # Create a parameterized schedule
        schedule = nest.create_schedule()
        ii = schedule.split(i, size=P4)
        jj = schedule.split(j, size=P4)
        kk = schedule.split(k, size=P4)

        P6, P7, P8 = create_parameters()
        schedule.reorder(order=P6)

        # Create a parameterized plan
        plan = schedule.create_plan()
        plan.cache(A, level=P5)
        plan.kernelize(unroll_indices=P7, vectorize_indices=P8)

        # Add another function to the package
        package.add(
            plan,
            args=(A, B, C),
            parameters={
                P0: 256,
                P1: 256,
                P2: 256,
                P3: 1.0,
                P4: 4,
                P5: 2,
                P6: (j, k, i, ii, jj, kk),
                P7: (ii, jj),
                P8: kk,
            },
            base_name="matmul_256_256_256",
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_parameterization_3(self) -> None:
        from accera import create_parameters, Nest

        for N in [10, 224]:    # input sizes
            for K in [1, 3, 5]:    # filter sizes
                M = N - K + 1    # output size

                P = create_parameters()

                A = Array(role=Role.INPUT, shape=(N, ))
                B = Array(role=Role.INPUT, shape=(K, ))
                C = Array(role=Role.INPUT_OUTPUT, shape=(M, ))

                nest = Nest(shape=(M, K))
                i, j = nest.get_indices()

                @nest.iteration_logic
                def _():
                    C[i] += A[i + j] * B[j]

                schedule = nest.create_schedule()

                A_test = np.random.random(A.shape).astype(np.float32)
                B_test = np.random.random(B.shape).astype(np.float32)
                C_test = np.random.random(C.shape).astype(np.float32)
                correctness_check_values = {
                    "pre": [A_test, B_test, C_test],
                    "post": [
                        A_test,
                        B_test,
                        C_test + np.convolve(np.flip(B_test), A_test, "valid"),
                    ],
                }

                # Skew dimension i with respect to dimension j with unroll loop not smaller than P.
                schedule.skew(i, j, P)

                # create a HAT package and add the function to it
                package = Package()
                package_name = f"test_parameterization_3_skew_i_j_{N}_{K}"
                function = package.add(
                    schedule,
                    args=(A, B, C),
                    parameters={P: 0},
                    base_name=f"schedule_test_skew_i_j_{N}_{K}",
                )
                output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

                # build the HAT package
                with verifiers.VerifyPackage(self, package_name, output_dir) as v:
                    package.build(
                        package_name,
                        format=TEST_FORMAT,
                        mode=TEST_MODE,
                        output_dir=output_dir,
                    )
                    if correctness_check_values:
                        v.check_correctness(
                            function.name,
                            before=correctness_check_values["pre"],
                            after=correctness_check_values["post"],
                        )

    def test_parameterization_4(self) -> None:
        from accera import create_parameters, Nest

        M = 16
        N = 10
        S = 11
        type = ScalarType.float32
        A = Array(role=Role.INPUT, element_type=type, shape=(M, S))
        B = Array(role=Role.INPUT, element_type=type, shape=(S, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=type, shape=(M, N))

        nest = Nest(shape=(M, N, S))

        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        P1, P2, P3, P4, P5, P6 = create_parameters()

        # Adds empty elements to the beginning of dimension i, j, k
        schedule.pad(i, P1)
        ii = schedule.split(i, P2)    # (2 + 16) // 3
        # should result in these loops for i, ii
        #  i: [2, 3:3), ii: [0, 1:1)  <-- partial (front padding)
        #  i: [3: 18:3), ii: [0, 3:1) <-- full

        schedule.pad(j, P3)
        jj = schedule.split(j, P4)    # (3 + 10) // 3
        # should result in these loops for j, jj
        #  j: [3, 12:3), jj: [0, 3:3)   <-- full (front padding == split size)
        #  j: [12, 13:3), jj: [0, 1:1)  <-- partial (automatic back padding)

        schedule.pad(k, P5)
        kk = schedule.split(k, P6)    # (11 + 11) // 4
        # should result in these loops for k, kk
        #  k: [11, 12:1), kk: [0, 1: 1) <-- partial
        #  k: [12, 20:4), kk: [0: 4: 1) <-- full
        #  k: [20, 22:4), kk: [0: 2: 1) <-- partial (automatic back padding)

        schedule.reorder(i, ii, k, j, jj, kk)

        A_test = np.random.random(A.shape).astype(np.float32)
        B_test = np.random.random(B.shape).astype(np.float32)
        C_test = np.random.random(C.shape).astype(np.float32)
        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_test + A_test @ B_test],
        }

        # create a HAT package and add the function to it
        package = Package()
        package_name = "test_parameterization_4_pad"
        function = package.add(
            schedule,
            args=(A, B, C),
            parameters={
                P1: 2,
                P2: 3,
                P3: 3,
                P4: 3,
                P5: 11,
                P6: 4
            },
            base_name="schedule_test_pad_parameter",
        )
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        # build the HAT package
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=output_dir)
            if correctness_check_values:
                v.check_correctness(
                    function.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

    def test_parameterization_5(self) -> None:
        from accera import create_parameters

        A = Array(role=Role.INPUT, shape=(256, 1024))
        B = Array(role=Role.INPUT, shape=(1024, 512))
        C = Array(role=Role.INPUT_OUTPUT, shape=(256, 512))

        nest = Nest(shape=(256, 512, 1024))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        target = Target("HOST", num_threads=16)

        # disable correctness checking on windows because the
        # install location of libomp.dll is non-standard as of now
        if sys.platform.startswith("win"):
            correctness_check_values = None
        else:
            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)
            correctness_check_values = {
                "pre": [A_test, B_test, C_test],
                "post": [A_test, B_test, C_test + A_test @ B_test],
            }

        schedule = nest.create_schedule()
        ii = schedule.split(i, A.shape[0] // target.num_threads)
        # set the index (k) that cannot be parallelized as innermost
        schedule.reorder(i, ii, j, k)

        P1, P2, P3, P4, P5, P6, P7, P8 = create_parameters()

        for policy in ["static", "dynamic"]:
            plan = schedule.create_plan(target)

            # non-collapsed
            plan.parallelize(indices=P1, policy=P2)

            package_name = f"parameterized_test_parallelize_i_{policy}"
            package = Package()
            function = package.add(
                plan,
                args=[A, B, C],
                parameters={
                    P1: i,
                    P2: policy
                },
                base_name=f"parameterized_vectorization_parallelization_test_i_{policy}",
            )

            output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
            with verifiers.VerifyPackage(self, package_name, output_dir) as v:
                package.build(
                    package_name,
                    format=TEST_FORMAT,
                    mode=TEST_MODE,
                    output_dir=output_dir,
                )
                if correctness_check_values:
                    v.check_correctness(
                        function.name,
                        before=correctness_check_values["pre"],
                        after=correctness_check_values["post"],
                    )

            # parallelizing middle index
            plan_ii = schedule.create_plan(target)
            plan_ii.parallelize(indices=P3, policy=P4)

            package_name = f"parameterized_test_parallelize_ii_{policy}"
            package_ii = Package()
            function_ii = package_ii.add(
                plan_ii,
                args=[A, B, C],
                parameters={
                    P3: ii,
                    P4: policy
                },
                base_name=f"parameterized_vectorization_parallelization_test_ii_{policy}",
            )

            output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
            with verifiers.VerifyPackage(self, package_name, output_dir) as v:
                package_ii.build(
                    package_name,
                    format=TEST_FORMAT,
                    mode=TEST_MODE,
                    output_dir=output_dir,
                )
            if correctness_check_values:
                v.check_correctness(
                    function_ii.name,
                    before=correctness_check_values["pre"],
                    after=correctness_check_values["post"],
                )

            # partial collapsed
            plan_partial = schedule.create_plan(target)
            plan_partial.parallelize(indices=P5, policy=P6)

            package_name = f"parameterized_test_parallelize_i_ii_j_{policy}"
            package_partial = Package()
            function_partial = package_partial.add(
                plan_ii,
                args=[A, B, C],
                parameters={
                    P5: (i, ii, j),
                    P6: policy
                },
                base_name=f"parameterized_vectorization_parallelization_test_i_ii_j_{policy}",
            )

            output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
            with verifiers.VerifyPackage(self, package_name, output_dir) as v:
                package_partial.build(
                    package_name,
                    format=TEST_FORMAT,
                    mode=TEST_MODE,
                    output_dir=output_dir,
                )
                if correctness_check_values:
                    v.check_correctness(
                        function_partial.name,
                        before=correctness_check_values["pre"],
                        after=correctness_check_values["post"],
                    )

            # partial collapsed inner indices
            plan_partial_inner = schedule.create_plan(target)
            plan_partial_inner.parallelize(indices=P7, policy=P8)

            package_name = f"parameterized_test_parallelize_ii_j_{policy}"
            package_partial_inner = Package()
            function_partial_inner = package_partial_inner.add(
                plan,
                args=[A, B, C],
                parameters={
                    P7: (ii, j),
                    P8: policy
                },
                base_name=f"parameterized_vectorization_parallelization_test_ii_j_{policy}",
            )

            output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
            with verifiers.VerifyPackage(self, package_name, output_dir) as v:
                package_partial_inner.build(
                    package_name,
                    format=TEST_FORMAT,
                    mode=TEST_MODE,
                    output_dir=output_dir,
                )
                if correctness_check_values:
                    v.check_correctness(
                        function_partial_inner.name,
                        before=correctness_check_values["pre"],
                        after=correctness_check_values["post"],
                    )

    def test_parameterization_grid(self) -> None:
        from accera import create_parameters, create_parameter_grid, Nest, Schedule

        P0, P1, P2, P3, P4 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        sched: Schedule = nest.create_schedule()
        sched.split(j, P4)

        package = Package()
        package_name = "test_parameter_grid_generation"

        parameter_grid = {
            P0: [8, 16],
            P1: [16, 32],
            P2: [16],
            P3: [1.0, 2.0],
            P4: [3, 5, 7],
        }

        parameters = create_parameter_grid(parameter_grid)
        package.add(sched, args=(A, B, C), base_name="matmul", parameters=parameters)

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_parameterization_grid_with_random_seed(self) -> None:
        from accera import create_parameters, create_parameter_grid, Nest, Schedule

        P0, P1, P2, P3, P4 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        sched: Schedule = nest.create_schedule()
        sched.split(j, P4)

        package = Package()
        package_name = "test_parameterization_grid_with_random_seed"

        parameter_grid = {
            P0: [8, 16],
            P1: [16, 32],
            P2: [16],
            P3: [1.0, 2.0],
            P4: [3, 5, 7],
        }

        parameters = create_parameter_grid(parameter_grid, sample=5, seed=123)
        parameters_dup = create_parameter_grid(parameter_grid, sample=5, seed=123)

        self.assertListEqual(parameters, parameters_dup)

        package.add(sched, args=(A, B, C), base_name="matmul", parameters=parameters)

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_parameterization_loopIndex_order(self) -> None:
        from accera import create_parameter_grid, create_parameters, Nest

        P0, P1, P2, P3 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        package = Package()
        package_name = "test_parameterization_loopIndex_order"

        P4, P5 = create_parameters()

        # Create a parameterized schedule
        schedule = nest.create_schedule()
        ii = schedule.split(i, size=P4)
        jj = schedule.split(j, size=P4)
        kk = schedule.split(k, size=P4)

        schedule.reorder(order=P5)

        # Create a parameterized plan
        plan = schedule.create_plan()

        parameter_grid = {
            P0: [256],
            P1: [256, 512],
            P2: [16, 32],
            P3: [1.0],
            P4: [4, 8, 16],
            P5: (i, j, k, ii, jj, kk),
        }

        parameters = create_parameter_grid(
            parameter_grid,
            filter_func=lambda *p: schedule.is_valid_loop_order(p[0][5]),
            sample=5,
        )

        # Add another function to the package
        package.add(plan, args=(A, B, C), parameters=parameters, base_name="matmul_256_256_256")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_fusion_parameterization_1(self) -> None:
        from accera import create_parameters, Nest, fuse

        A = Array(role=Role.INPUT, element_type=float, shape=(32, ))
        B = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(32, ))
        C = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(1, ))

        n0 = Nest([32, 32])
        i0, j0 = n0.get_indices()

        @n0.iteration_logic
        def _():
            B[i0] += A[i0] * A[j0]

        s0 = n0.create_schedule()

        n0_up = Nest(n0.get_shape())
        i0_up, j0_up = n0_up.get_indices()

        @n0_up.iteration_logic
        def _():
            B[i0_up] += A[i0_up] * A[j0_up]

        s0_up = n0_up.create_schedule()

        n1 = Nest([32])
        i1 = n1.get_indices()

        @n1.iteration_logic
        def _():
            C[0] += B[i1]

        s1 = n1.create_schedule()

        P0 = create_parameters()
        jj0 = s0.split(j0, P0)

        jj0_up = s0_up.split(j0_up, 16)

        fs = fuse((s0, s1), partial=1)
        f, i, j, jj = fs.get_indices()
        fs.reorder(i, f, j, jj)

        fs_up = fuse((s0_up, s1), partial=1)
        f_up, i_up, j_up, jj_up = fs_up.get_indices()
        fs_up.reorder(i_up, f_up, j_up, jj_up)

        package = Package()
        package_name = "test_fusion_parameterization_1"

        package.add(fs_up, args=(A, B, C), base_name="fuse_unparameterized_1")

        package.add(
            fs,
            args=(A, B, C),
            parameters={
                P0: 16,
            },
            base_name="fuse_1",
        )
        package.add(
            fs,
            args=(A, B, C),
            parameters={
                P0: 3,
            },
            base_name="fuse_2",
        )
        package.add(
            fs, args=(A, B, C), parameters=[{
                P0: 5
            }, {
                P0: 7
            }], base_name="fuse_3"
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_fusion_parameterization_2(self) -> None:
        """
        Goes through a different codepath from the above tests because the
        schedules are emitted directly prior to the fused schedule, which
        matters because the fused schedule has references to the schedule
        """
        from accera import create_parameters, Nest, fuse

        A = Array(role=Role.INPUT, element_type=float, shape=(32, ))
        B = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(32, ))
        C = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(1, ))

        n0 = Nest([32, 32])
        i0, j0 = n0.get_indices()

        @n0.iteration_logic
        def _():
            B[i0] += A[i0] * A[j0]

        s0 = n0.create_schedule()

        n1 = Nest([32])
        i1 = n1.get_indices()

        @n1.iteration_logic
        def _():
            C[0] += B[i1]

        s1 = n1.create_schedule()

        P0 = create_parameters()
        jj0 = s0.split(j0, P0)

        fs = fuse((s0, s1), partial=1)

        package = Package()
        package_name = "test_fusion_parameterization_2"

        package.add(
            s0, args=(A, B), parameters={P0: 16}, base_name="s0_1"
        )
        package.add(
            s0, args=(A, B), parameters={P0: 32}, base_name="s0_2"
        )
        package.add(
            s1, args=(C, B), parameters={P0: 16}, base_name="s1_1"
        )
        package.add(
            fs,
            args=(A, B, C),
            parameters={
                P0: 16,
            },
            base_name="fuse_1",
        )
        package.add(
            fs,
            args=(A, B, C),
            parameters={
                P0: 32,
            },
            base_name="fuse_2",
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_fusion_parameterization_3(self) -> None:
        from accera import create_parameters, Nest, fuse

        A = Array(role=Role.INPUT, element_type=float, shape=(32, ))
        B = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(32, ))
        C = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(1, ))

        n0 = Nest([32, 32])
        i0, j0 = n0.get_indices()

        @n0.iteration_logic
        def _():
            B[i0] += A[i0] * A[j0]

        s0 = n0.create_schedule()

        n1 = Nest([32])
        i1 = n1.get_indices()

        @n1.iteration_logic
        def _():
            C[0] += B[i1]

        s1 = n1.create_schedule()

        P0, P1 = create_parameters()
        jj0 = s0.split(j0, P0)

        fs = fuse((s0, s1), partial=1)
        f, i, j, jj = fs.get_indices()
        ii = fs.split(i, P1)
        fs.reorder(f, i, j, ii, jj)

        package = Package()
        package_name = "test_fusion_parameterization_3"

        package.add(
            fs, args=(A, B, C), parameters={
                P0: 16,
                P1: 8
            }, base_name="fuse_1"
        )
        package.add(
            fs,
            args=(A, B, C),
            parameters={
                P0: 32,
                P1: 4,
            },
            base_name="fuse_2",
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_fusion_parameterization_4(self) -> None:
        from accera import create_parameters, Nest, fuse, create_parameter_grid

        A = Array(role=Role.INPUT, element_type=float, shape=(128, ))
        B = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(128, ))
        C = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=(1, ))

        n0 = Nest([128, 128])
        i0, j0 = n0.get_indices()

        @n0.iteration_logic
        def _():
            B[i0] += A[i0] * A[j0]

        s0 = n0.create_schedule()

        n1 = Nest([128])
        i1 = n1.get_indices()

        @n1.iteration_logic
        def _():
            C[0] += B[i1]

        s1 = n1.create_schedule()

        P0, P1, P2 = create_parameters()
        jj0 = s0.split(j0, P0)

        fs = fuse((s0, s1), partial=1)
        f, i, j, jj = fs.get_indices()
        ii = fs.split(i, P1)
        fs.reorder(i, f, j, ii, jj)
        jjj = fs.split(jj, P2)

        package = Package()
        package_name = "test_fusion_parameterization_4"

        # Expected loop structure
        # P0 = 16
        # P1 = 8
        # P2 = 4
        # for i in range(128, step=P1):
        #     for f in range(2):
        #         if f == 0:
        #             for j in range(128, step=P0):
        #                 for ii in range(P1):
        #                     for jj in range(P0, step=P2):
        #                         for jjj in range(P2):
        #                             ...
        #         if f == 1:
        #             for ii in range(P1):
        #                 ...
        package.add(
            fs, args=(A, B, C), parameters={
                P0: 16,
                P1: 8,
                P2: 4
            }, base_name="fuse_1"
        )

        # Expected loop structure
        # P0 = 32
        # P1 = 4
        # P2 = 8
        # for i in range(128, step=P1):
        #     for f in range(2):
        #         if f == 0:
        #             for j in range(128, step=P0):
        #                 for ii in range(P1):
        #                     for jj in range(P0, step=P2):
        #                         for jjj in range(P2):
        #                             ...
        #         if f == 1:
        #             for ii in range(P1):
        #                 ...
        package.add(
            fs, args=(A, B, C), parameters={
                P0: 32,
                P1: 4,
                P2: 8
            }, base_name="fuse_2"
        )
        package.add(
            fs,
            args=(A, B, C),
            parameters=create_parameter_grid({
                P0: [64, 8],
                P1: [12, 16, 20],
                P2: [2, 10]
            }),
            base_name="fuse_grid",
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_parameterization_auxiliary_data(self) -> None:
        from accera import create_parameters, create_parameter_grid, Nest, Schedule
        from hatlib import HATPackage

        P0, P1, P2, P3, P4 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        sched: Schedule = nest.create_schedule()
        sched.split(j, P4)

        package = Package()
        package_name = "test_parameterization_auxiliary_data"

        parameter_grid = {
            P0: [8, 16],
            P1: [16, 32],
            P2: [16],
            P3: [1.0, 2.0],
            P4: [3, 5, 7],
        }

        parameters = create_parameter_grid(parameter_grid)
        package.add(sched, args=(A, B, C), base_name="matmul", parameters=parameters)

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

        hat_package = HATPackage(pathlib.Path(TEST_PACKAGE_DIR) / f"{package_name}.hat")
        functions = [fn for fn in hat_package.get_functions()]
        for function in functions:
            data_point = function.auxiliary["accera"]["parameters"]
            if data_point:
                self.assertIn(int(data_point["P0"]), [8, 16])
                self.assertIn(int(data_point["P1"]), [16, 32])
                self.assertIn(int(data_point["P2"]), [16])
                self.assertIn(float(data_point["P3"]), [1.0, 2.0])
                self.assertIn(int(data_point["P4"]), [3, 5, 7])

    def test_parameterization_arithmetic_operation(self) -> None:
        from accera import create_parameters, Nest

        P0, P1, P2, P3 = create_parameters()

        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P0, P2))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(P2, P1))
        C = Array(
            role=Role.INPUT_OUTPUT,
            element_type=ScalarType.float32,
            shape=(P0, P1),
        )

        nest = Nest(shape=(P0, P1, P2))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += P3 * A[i, k] * B[k, j]

        package = Package()
        package_name = "test_parameterization_arithmetic_operation"

        fma_unit_count, vector_size, multiplier = create_parameters()

        # Create a parameterized schedule, the parameter arithmetic operation is just made up for testing purpose
        schedule = nest.create_schedule()
        schedule.split(i, size=fma_unit_count * vector_size * multiplier)
        schedule.split(j, size=vector_size + multiplier * fma_unit_count)
        schedule.split(k, size=(vector_size + multiplier) * fma_unit_count)

        # Create a parameterized plan
        plan = schedule.create_plan()

        # Add another function to the package
        package.add(
            plan,
            args=(A, B, C),
            parameters={
                P0: 256,
                P1: 256,
                P2: 256,
                P3: 1.0,
                fma_unit_count: 2,
                vector_size: 16,
                multiplier: 2,
            },
            base_name="matmul_256_256_256",
        )

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_parameterization_gpu_bind(self) -> None:
        from accera import create_parameters
        P0, P1, P2, P3, P4, P5 = create_parameters()

        M = 128
        N = 256
        K = 256
        A = Array(role=Role.INPUT, shape=(M, K))
        B = Array(role=Role.INPUT, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        v100 = Target(Target.Model.NVIDIA_V100, category=Target.Category.GPU)
        plan = nest.create_plan(v100)

        plan.bind(mapping={
            P0: P3,
            P1: P4,
            P2: P5
        })

        test_name = "test_parameterization_gpu_bind"
        package = Package()
        package.add(
            plan,
            args=(A, B, C),
            parameters={
                P0: i,
                P1: j,
                P2: k,
                P3: v100.GridUnit.BLOCK_X,
                P4: v100.GridUnit.THREAD_X,
                P5: v100.GridUnit.THREAD_Y,
            },
            base_name=test_name
        )

        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / test_name

        with verifiers.VerifyPackage(self, test_name, output_dir, file_list=[f"{test_name}.cu",
                                                                             f"{test_name}.hat"]) as v:
            package.build(
                name=test_name,
                format=Package.Format.MLIR | Package.Format.DEFAULT,
                mode=Package.Mode.RELEASE,    # Package.Mode.DEBUG not supported
                output_dir=output_dir
            )

    def test_parameterization_subarray(self) -> None:
        from accera import create_parameters

        package = Package()
        P0, P1 = create_parameters()
        arr = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(256, 256))
        arr0 = arr.sub_array(offsets=(0, 0), shape=(P0, P1))

        # add a function that utilizes a subarray layout
        def make_subarray_fn(arr0):
            nest = Nest(shape=arr0.shape)
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                arr0[i, j] += 1.

            return package.add(
                nest, args=(arr0, ), parameters={
                    P0: 128,
                    P1: 128
                }
            )

        subarray_fn = make_subarray_fn(arr0)

        # add a function that instantiates a subarray of the input array and calls the function above
        def main(arr):
            # emit code that creates a subarray view of another array, here the arr has been replaced by native object,
            # so arr.sub_array calls native function to create the view, the shape is not parameterizable in this case.
            arr1 = arr.sub_array(offsets=(0, 0), shape=(128, 128))
            print(arr1.layout)
            self.assertEqual(arr0.layout, arr1.layout)
            subarray_fn(arr1)

        package.add(main, args=(arr, ))

        package_name = "test_parameterization_subarray"

        # BUGBUG: starting from LLVM 14, sub_array in Package.Mode.DEBUG needs to link
        # against libmlir_c_runner_utils.so for the memrefCopy function
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=TEST_FORMAT, mode=Package.Mode.RELEASE, output_dir=TEST_PACKAGE_DIR)


class DSLTest_10Packages(unittest.TestCase):
    def _create_plan(self, target=Target.HOST) -> Function:
        A = Array(role=Role.INPUT_OUTPUT, shape=(64, ))

        nest = Nest(shape=(64, ))
        i = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i] += 2.0

        plan = nest.create_plan(target)
        return plan, A

    def test_HAT_packages(self) -> None:
        from accera import Target

        pi3 = Target(Target.Model.RASPBERRY_PI_3B, category=Target.Category.CPU)
        plan, A = self._create_plan(pi3)

        package = Package()
        package_name = "MyPackage"
        package.add(plan, args=(A, ), base_name="func1")
        package.add(plan, args=(A, ), base_name="func2")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=Package.Format.HAT_STATIC,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
                platform=Package.Platform.RASPBIAN,
            )

    def test_MLIR_packages(self) -> None:
        plan, A = self._create_plan()

        package = Package()
        package_name = "MyPackage"
        package.add(plan, args=(A, ), base_name="func1")
        package.add(plan, args=(A, ), base_name="func2")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=Package.Format.MLIR_STATIC,
                output_dir=TEST_PACKAGE_DIR,
            )

    def test_default_output_dir(self) -> None:
        plan, A = self._create_plan()

        package = Package()
        package_name = "MyPackage"
        package.add(plan, args=(A, ), base_name="func1")
        package.add(plan, args=(A, ), base_name="func2")

        with verifiers.VerifyPackage(self, package_name):
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE)

    def test_debug_mode_1(self) -> None:
        M = N = K = 16
        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii = schedule.split(i, 4)
        schedule.reorder(i, k, j, ii)
        plan = schedule.create_plan()
        plan.unroll(ii)

        package = Package()
        package_name = "MyDebugPackage"
        function = package.add(plan, args=(A, B, C), base_name="func1")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT,
                output_dir=output_dir,
                mode=Package.Mode.DEBUG,
                tolerance=1e-5,
            )

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)

            v.check_correctness(
                function.name,
                before=[A_test, B_test, C_test],
                after=[A_test, B_test, C_test + A_test @ B_test],
            )

    def test_debug_mode_2(self) -> None:
        M = N = K = 16
        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, K))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(K, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest = Nest(shape=(M, N, K))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        schedule = nest.create_schedule()

        ii = schedule.split(i, 4)
        schedule.reorder(i, k, j, ii)
        plan = schedule.create_plan()
        plan.unroll(ii)
        # deliberately introduce a correctness issue
        plan.parallelize(indices=k)

        package = Package()
        package_name = "MyDebugPackageIncorrect"
        function = package.add(plan, args=(A, B, C), base_name="func1")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT,
                output_dir=output_dir,
                mode=Package.Mode.DEBUG,
                tolerance=1e-5,
            )

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)

            try:
                v.check_correctness(
                    function.name,
                    before=[A_test, B_test, C_test],
                    after=[A_test, B_test, C_test + A_test @ B_test],
                )
            except Exception as e:
                print(e)

    def test_debug_mode_fusion_1(self) -> None:
        from accera import fuse

        M = N = 16
        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest0 = Nest(shape=(M, N))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        nest1 = Nest(shape=(M, N))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        schedule = fuse(schedule0, schedule1, partial=1)
        f, i, j0, j1 = schedule.get_indices()
        ii = schedule.split(i, 2)
        schedule.reorder(i, ii, f, j0, j1)

        package = Package()
        package_name = "MyFusionDebugPackage"
        function = package.add(schedule, args=(A, B, C), base_name="fusion_func1")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT,
                output_dir=output_dir,
                mode=Package.Mode.DEBUG,
                tolerance=1e-5,
            )

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)

            v.check_correctness(
                function.name,
                before=[A_test, B_test, C_test],
                after=[A_test, B_test, (C_test + A_test) * B_test],
            )

    def test_debug_mode_fusion_2(self) -> None:
        from accera import fuse

        M = N = 16
        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest0 = Nest(shape=(M, N))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        nest1 = Nest(shape=(M, N))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        # Reorder schedule1 before fusing
        schedule1.reorder(j1, i1)
        # Fuse schedule0 with the reordered schedule1
        schedule = fuse(schedule0, schedule1)
        f, a, b = schedule.get_indices()

        # Deliberately break logical equivalence
        # before: C[1,0] = C[1,0] * B[1,0] + A[1,0]
        # after: C[1,0] = (C[1,0] + A[1,0]) * B[1,0]
        schedule.reorder(a, b, f)

        package = Package()
        package_name = "MyFusionDebugPackageIncorrect"
        function = package.add(schedule, args=(A, B, C), base_name="fusion_func1")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT,
                output_dir=output_dir,
                mode=Package.Mode.DEBUG,
                tolerance=1e-5,
            )

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)

            try:
                v.check_correctness(
                    function.name,
                    before=[A_test, B_test, C_test],
                    after=[A_test, B_test, (C_test + A_test) * B_test],
                )
            except Exception as e:
                print(e)

    def test_debug_mode_fusion_cascading_1(self) -> None:
        from accera import fuse

        M = N = 16
        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest0 = Nest(shape=(M, N))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        nest1 = Nest(shape=(M, N))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        schedule_f1 = fuse(schedule0, schedule1)
        f, i, j = schedule_f1.get_indices()
        schedule_f1.reorder(i, j, f)

        nest2 = Nest(shape=(M, N))
        i2, j2 = nest2.get_indices()

        @nest2.iteration_logic
        def _():
            C[i2, j2] -= 1.0

        schedule2 = nest2.create_schedule()

        # set the fused schedule first in the fusing order
        schedule_f2 = fuse(schedule_f1, schedule2, partial=2)

        package = Package()
        package_name = "MyFusionDebugPackageCascade1"
        function = package.add(schedule_f2, args=(A, B, C), base_name="fusion_func1")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT,
                output_dir=output_dir,
                mode=Package.Mode.DEBUG,
                tolerance=1e-5,
            )

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)

            v.check_correctness(
                function.name,
                before=[A_test, B_test, C_test],
                after=[A_test, B_test, (C_test + A_test) * B_test - 1.0],
            )

    def test_debug_mode_fusion_cascading_2(self) -> None:
        from accera import fuse

        M = N = 16
        A = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(M, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))

        nest0 = Nest(shape=(M, N))
        i0, j0 = nest0.get_indices()

        @nest0.iteration_logic
        def _():
            C[i0, j0] += A[i0, j0]

        schedule0 = nest0.create_schedule()

        nest1 = Nest(shape=(M, N))
        i1, j1 = nest1.get_indices()

        @nest1.iteration_logic
        def _():
            C[i1, j1] *= B[i1, j1]

        schedule1 = nest1.create_schedule()

        schedule_f1 = fuse(schedule0, schedule1)
        f, i, j = schedule_f1.get_indices()
        schedule_f1.reorder(i, j, f)

        nest2 = Nest(shape=(M, N))
        i2, j2 = nest2.get_indices()

        @nest2.iteration_logic
        def _():
            C[i2, j2] -= 1.0

        schedule2 = nest2.create_schedule()

        # set an unfused schedule first in the fusing order
        schedule_f2 = fuse(schedule2, schedule_f1, partial=2)

        package = Package()
        package_name = "MyFusionDebugPackageCascade2"
        function = package.add(schedule_f2, args=(A, B, C), base_name="fusion_func1")
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name

        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(
                package_name,
                format=TEST_FORMAT,
                output_dir=output_dir,
                mode=Package.Mode.DEBUG,
                tolerance=1e-5,
            )

            A_test = np.random.random(A.shape).astype(np.float32)
            B_test = np.random.random(B.shape).astype(np.float32)
            C_test = np.random.random(C.shape).astype(np.float32)

            v.check_correctness(
                function.name,
                before=[A_test, B_test, C_test],
                after=[A_test, B_test, (C_test - 1.0 + A_test) * B_test],
            )

    def test_add_description(self) -> None:
        from hatlib import HATFile

        (
            plan,
            A,
        ) = self._create_plan()

        package = Package()
        package_name = "MyPackage"
        package.add(plan, args=(A, ), base_name="func1")
        package.add(plan, args=(A, ), base_name="func2")

        description1 = {
            "Dependencies": ["numpy", "onnx", "scipy"],
            "Documentation": "https://docs.readthedocs.io.",
            "SHA": "0bb913ce84afa28127ea3fd2a9995e219dad322a",
        }

        package.add_description(
            other=description1,
            version="1.0",
            author="Microsoft Research",
            license="https://mit-license.org",
        )

        description2 = {
            "Documentation": "",    # clearing a value
            "SHA": None,    # removing a value
            "Release Notes": "https://stackoverflow.com",    # adding an entry
        }

        package.add_description(other=description2)
        package.add_description(version="2.0")

        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(
                package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR,
            )

        hat_file = HATFile.Deserialize(pathlib.Path(TEST_PACKAGE_DIR) / f"{package_name}.hat")
        hat_description = hat_file.description.auxiliary
        self.assertEqual(hat_description["Dependencies"], description1["Dependencies"])
        self.assertEqual(hat_description["Documentation"], description2["Documentation"])
        self.assertNotIn("SHA", hat_description)
        self.assertEqual(hat_description["Release Notes"], description2["Release Notes"])
        self.assertEqual(hat_file.description.version, "2.0")
        self.assertEqual(hat_file.description.author, "Microsoft Research")
        self.assertEqual(hat_file.description.license_url, "https://mit-license.org")

    def test_logic_function_conditionals(self) -> None:
        def make_test_fn(package, A, B, C):
            T = Array(role=Role.TEMP, element_type=A.element_type, shape=A.shape)

            nest = Nest(A.shape)
            i, j = nest.get_indices()

            from accera._lang_python._lang import _If

            def if_func():
                T[i, j] = A[i, j] + B[i, j]
                C[i, j] += T[i, j]**2.

            def elseif_func():
                T[i, j] = A[i, j] - B[i, j]
                C[i, j] += T[i, j]**2.

            def else_func():
                C[i, j] = A[i, j] + B[i, j]

            @nest.iteration_logic
            def _():
                _If(j < 100, if_func).ElseIf(i > 100, elseif_func).Else(else_func)

            return package.add(nest, args=(A, B, C))

        A = Array(shape=(256, 32), role=Role.INPUT)
        B = Array(shape=(256, 32), role=Role.INPUT)
        C = Array(shape=(256, 32), role=Role.INPUT_OUTPUT)

        package = Package()
        make_test_fn(package, A, B, C)
        package_name = "test_logic_function_conditionals"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, format=TEST_FORMAT, mode=TEST_MODE, output_dir=TEST_PACKAGE_DIR)


class DSLTest_11AutoPlan(unittest.TestCase):
    def _create_plan(self, shape: Tuple[int], type=ScalarType.float32) -> Tuple:
        M, N, S = shape

        A = Array(role=Role.INPUT, element_type=type, shape=(M, S))
        B = Array(role=Role.INPUT, element_type=type, shape=(S, N))
        C = Array(role=Role.INPUT_OUTPUT, element_type=type, shape=(M, N))

        nest = Nest(shape=(M, N, S))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] = A[i, k] * B[k, j]

        sched = nest.create_schedule()
        ii = sched.split(i, 64)
        jj = sched.split(i, 64)
        kk = sched.split(k, 64)
        sched.reorder(i, j, k, ii, jj, kk)

        plan = sched.create_plan()

        return plan, [A, B, C], [i, j, k, ii, jj, kk]

    def _verify_autoplan(self, plan, args: Tuple[Array], package_name, functions_counter) -> None:
        # create a HAT package and add the function to it
        package = Package()
        functions = package.add(plan, args, base_name="autoplan_test")
        # Assert the total number of functions added to a package are equal to 2.
        # Each function is added for a unique value of layout as inferred by plan.auto()
        if functions and isinstance(functions, list):
            self.assertEqual(len(functions), functions_counter)

    def test_autoplan(self) -> None:
        plan, args, indices = self._create_plan((1024, 1024, 1024))
        A, B, C = args
        i, j, k, ii, jj, kk = indices

        plan.auto(algorithms.NoneCacheHeuristics(source=B, index=j))
        self._verify_autoplan(plan, [A, B, C], "test_autoplan_for_caching_1", 2)

        plan.auto(algorithms.NoneCacheHeuristics(source=B, layout=Array.Layout.FIRST_MAJOR))
        # Testing list of heuristics with a subsequent call to `plan.auto()`
        # This will create a product of prameters, (2*6 = 12) and hence, 12 functions
        # are added to a package
        self._verify_autoplan(plan, [A, B, C], "test_autoplan_for_caching_2", 12)

        # create a new plan
        plan_2, args_2, indices_2 = self._create_plan((1024, 1024, 1024))
        A, B, C = args_2
        i, j, k, ii, jj, kk = indices_2

        plan_2.auto(algorithms.NoneCacheHeuristics(source=B, layout=Array.Layout.FIRST_MAJOR))
        # A total of 6 functions will be created, one for each unique value of index
        self._verify_autoplan(plan_2, [A, B, C], "test_autoplan_for_caching_3", 6)

        # create a new plan
        plan_3, args_3, indices_3 = self._create_plan((1024, 1024, 1024))
        A, B, C = args_3
        i, j, k, ii, jj, kk = indices_3

        plan_3.auto(algorithms.NoneCacheHeuristics(source=B))
        # A total of 12 functions will be created, one for each unique combination of
        # index and layout value.
        self._verify_autoplan(plan_3, [A, B, C], "test_autoplan_for_caching_4", 12)


if __name__ == "__main__":
    unittest.main(verbosity=10)
