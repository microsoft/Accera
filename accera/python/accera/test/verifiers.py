#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import hatlib as hat
import numpy as np
import pathlib
import unittest
from copy import deepcopy
from typing import List


class CorrectnessChecker():
    def __init__(self, hat_path):
        self.package = hat.load(hat_path)

    def run(
        self, function_name: str, before: List["numpy.ndarray"], after: List["numpy.ndarray"], tolerance: float = 1e-5
    ):

        if function_name not in self.package.names:
            raise ValueError(f"{function_name} is not found")

        # use temporaries so that we don't have side effects
        input_outputs = deepcopy(before)
        self.package[function_name](*input_outputs)

        for actual, desired in zip(input_outputs, after):
            np.testing.assert_allclose(actual, desired, tolerance)


class VerifyPackage():
    def __init__(
        self, testcase: unittest.TestCase, package_name: str, output_dir: str = None, file_list: List[str] = None
    ):
        self.testcase = testcase
        hat_file = pathlib.Path(output_dir or pathlib.Path.cwd()) / f"{package_name}.hat"
        self.file_list = [pathlib.Path(output_dir or pathlib.Path.cwd()) / f
                          for f in file_list] if file_list else [hat_file]
        self.correctness_checker = None

    def __enter__(self):
        for f in self.file_list:
            if f.exists():
                f.unlink()    # the missing_ok flag is Python 3.8+ only
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self.file_list:
            self.testcase.assertTrue(f.is_file(), f"{f} was expected, but not found to be a file")

    def check_correctness(
        self, function_name: str, before: List["numpy.ndarray"], after: List["numpy.ndarray"], tolerance: float = 1e-5
    ):
        """Performs correctness-checking on a function
        Args:
            function_name
            before: values before calling the function
            after: desired values after calling the function
            tolerance: absolute tolerance for floating point comparison
        """
        hat_files = list(filter(lambda f: f.suffix == ".hat", self.file_list))
        if hat_files:
            assert len(hat_files) == 1
            hat_file = hat_files[0]
            if not self.correctness_checker:
                self.correctness_checker = CorrectnessChecker(hat_file)
            self.correctness_checker.run(function_name, before, after, tolerance)
