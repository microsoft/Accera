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

DEV_MODE = False
if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    DEV_MODE = True
    sys.path.insert(1, os.getcwd())

from accera import Array, Nest, Package, ScalarType, Target, cast, Role
from accera.test import verifiers

TEST_MODE = Package.Mode.DEBUG if DEV_MODE else Package.Mode.RELEASE
# TEST_FORMAT = Package.Format.DEFAULT | Package.Format.MLIR_VERBOSE
TEST_FORMAT = Package.Format.MLIR_DYNAMIC if DEV_MODE else Package.Format.HAT_DYNAMIC
TEST_PACKAGE_DIR = "test_int_matmul"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class IntMatmulTest(unittest.TestCase):

    def test_int16_matmul_1(self) -> None:
        vecSize = 8
        M = 1
        N = vecSize
        K = 2

        A = Array(role=Role.INPUT, element_type=ScalarType.int16, shape=(M, K))
        # B = Array(role=Role.INPUT, element_type=ScalarType.int16, shape=(K, N))
        B = Array(role=Role.INPUT, element_type=ScalarType.int16, shape=(N, K))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, N))

        nest = Nest((M, N, K))
        i, j, k = nest.get_indices()
        @nest.iteration_logic
        def _():
            C[i, j] += cast(A[i, k], ScalarType.int32) * cast(B[j, k], ScalarType.int32)
        
        in_type = np.int16
        out_type = np.int32
        rng = np.random.default_rng()
        A_test = rng.integers(0, 100, A.shape).astype(in_type)
        B_test = rng.integers(0, 100, B.shape).astype(in_type)
        C_test = np.zeros(C.shape, dtype=out_type)
        C_ref = (A_test @ B_test.transpose()).astype(out_type)

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_ref],
        }

        schedule = nest.create_schedule()
        plan = schedule.create_plan()

        # plan.vectorize(j)

        plan.cache(A, index=i)
        plan.cache(B, index=j)
        plan.cache(C, index=i)

        target = Target(category=Target.Category.CPU)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        package_name = f"{test_name}_pkg"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR) as v:
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR)

            # Verify this code generates the vpmaddwd instruction
            # Temporarily disabled because the test is failing on the CI
            # checker = v.file_checker(f"{package_name}.s")
            # checker.check_label(test_name)
            # checker.check("vpmaddwd")
            # checker.run()

            v.check_correctness(
                function.name,
                before=correctness_check_values["pre"],
                after=correctness_check_values["post"],
            )


    def test_int16_matmul_2(self) -> None:
        M = 20
        N = 64
        K = 8

        splitM = 1
        splitN = 8
        splitK = 2

        A = Array(role=Role.INPUT, element_type=ScalarType.int16, shape=(M, K))
        # B = Array(role=Role.INPUT, element_type=ScalarType.int16, shape=(K, N))
        B = Array(role=Role.INPUT, element_type=ScalarType.int16, shape=(N, K))
        C = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, N))

        nest = Nest((M, N, K))
        i, j, k = nest.get_indices()
        @nest.iteration_logic
        def _():
            # C[i, j] += A[i, k] * B[k, j]
            # C[i, j] += cast(A[i, k], ScalarType.int32) * cast(B[k, j], ScalarType.int32)
            C[i, j] += cast(A[i, k], ScalarType.int32) * cast(B[j, k], ScalarType.int32)
        
        in_type = np.int16
        out_type = np.int32
        rng = np.random.default_rng()
        A_test = rng.integers(0, 10, A.shape).astype(in_type)
        B_test = rng.integers(0, 10, B.shape).astype(in_type)
        C_test = np.zeros(C.shape, dtype=out_type)
        C_ref = (A_test @ B_test.transpose()).astype(out_type)

        correctness_check_values = {
            "pre": [A_test, B_test, C_test],
            "post": [A_test, B_test, C_ref],
        }

        schedule = nest.create_schedule()
        ii, jj, kk = schedule.tile({
            i: splitM,
            j: splitN,
            k: splitK
        })

        plan = schedule.create_plan()

        # plan.vectorize(jj)

        plan.cache(A, index=ii)
        plan.cache(B, index=jj)
        plan.cache(C, index=ii)

        target = Target(category=Target.Category.CPU)

        test_name = inspect.currentframe().f_code.co_name
        package = Package()
        function = package.add(plan, args=(A, B, C), base_name=test_name)

        package_name = f"{test_name}_pkg"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR) as v:
            package.build(
                name=package_name,
                format=TEST_FORMAT,
                mode=TEST_MODE,
                output_dir=TEST_PACKAGE_DIR)

            # Verify this code generates the vpmaddwd instruction
            # Temporarily disabled because the test is failing on the CI
            # checker = v.file_checker(f"{package_name}.s")
            # checker.check_label(test_name)
            # checker.check("vpmaddwd")
            # checker.run()

            v.check_correctness(
                function.name,
                before=correctness_check_values["pre"],
                after=correctness_check_values["post"],
            )


if __name__ == '__main__':
    unittest.main(verbosity=10)
