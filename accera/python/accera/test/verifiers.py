#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import hatlib as hat
import numpy as np
import glob
import os
import pathlib
import shutil
import unittest
from copy import deepcopy
from typing import List


class CorrectnessChecker():
    def __init__(self, hat_path):
        self.hat_package, self.func_map = hat.load(hat_path)

    def run(
        self, function_name: str, before: List["numpy.ndarray"], after: List["numpy.ndarray"], tolerance: float = 1e-5
    ):
        if function_name not in self.func_map.names:
            raise ValueError(f"{function_name} is not found")

        # use temporaries so that we don't have side effects
        input_outputs = deepcopy(before)
        self.func_map[function_name](*input_outputs)
        print("Verifying...")

        for actual, desired in zip(input_outputs, after):
            # Cast to double here since assert_allclose does not have support for bfloat 16
            # and we don't want to take dependency on the bfloat16 extension to numpy
            actual = actual.astype(np.double)
            desired = desired.astype(np.double)
            np.testing.assert_allclose(actual, desired, rtol=tolerance)


class FileChecker():
    """Wrapper around FileCheck that verifies the contents of an input file
    https://llvm.org/docs/CommandGuide/FileCheck.html
    """
    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path
        self.directives = []

    def check(self, pattern):
        self.directives.append(f"CHECK: {pattern}")

    def check_label(self, pattern):
        self.directives.append(f"CHECK-LABEL: {pattern}")

    def check_next(self, pattern):
        self.directives.append(f"CHECK-NEXT: {pattern}")

    def check_same(self, pattern):
        self.directives.append(f"CHECK-SAME: {pattern}")

    def check_count(self, pattern, n):
        self.directives.append(f"CHECK-COUNT-{n}: {pattern}")

    def check_dag(self, pattern):
        self.directives.append(f"CHECK-DAG: {pattern}")

    def check_not(self, pattern):
        self.directives.append(f"CHECK-NOT: {pattern}")

    def run(self, quiet=True):
        from ..utilities import run_command
        from ..build_config import BuildConfig

        match_filename = f"{self.file_path}.filecheck"
        with open(match_filename, "w") as f:
            for d in self.directives:
                print(d, file=f)

        def find_filecheck():
            # first try PATH
            filecheck_exe = shutil.which("FileCheck")

            # not found in PATH, try BuildConfig (Dev environments only)
            if not filecheck_exe:
                if pathlib.Path(BuildConfig.llvm_filecheck).exists():
                    filecheck_exe = BuildConfig.llvm_filecheck

            if not filecheck_exe:
                raise RuntimeError("ERROR: Could not find FileCheck, please ensure that FileCheck is in the PATH")
            return filecheck_exe

        run_command(
            f"{find_filecheck()} {match_filename} --input-file {self.file_path}",
            working_directory=self.file_path.parent,
            quiet=quiet
        )


class VerifyPackage():
    def __init__(
        self, testcase: unittest.TestCase, package_name: str, output_dir: str = None, file_list: List[str] = None
    ):
        self.testcase = testcase
        hat_file = pathlib.Path(output_dir or pathlib.Path.cwd()) / f"{package_name}.hat"
        self.file_list = [pathlib.Path(output_dir or pathlib.Path.cwd()) / f
                          for f in file_list] if file_list else [hat_file]
        self.correctness_checker = None
        self.output_dir = output_dir

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
            tolerance: relative tolerance for floating point comparison
        """
        hat_files = list(filter(lambda f: f.suffix == ".hat", self.file_list))
        if hat_files:
            assert len(hat_files) == 1
            hat_file = hat_files[0]
            if not self.correctness_checker:
                self.correctness_checker = CorrectnessChecker(hat_file)
            self.correctness_checker.run(function_name, before, after, tolerance)
        else:
            print("Warning: check_correctness was called but no hat file was generated. Correctness check skipped.")

    def file_checker(self, filename):
        """Returns a checker for applying FileCheck directives
        Args:
            filename: name or path to the file to apply the checks
                If a non-path is provided, searches the output directory for the first instance
                The non-path can be a glob-like regex, e.g. "*myfile.mlir"
        """
        filepath = pathlib.Path(filename)

        # Python 3.7 on Windows raises an OSError for is_file() for non-existent files,
        # use os.path.isfile() instead
        if not os.path.isfile(filepath.absolute()):
            files = glob.glob(f"{self.output_dir}/**/{filename}", recursive=True)
            if not files:
                raise ValueError(f"{filename} not found, did you set the correct Package.Format?")
            filepath = pathlib.Path(files[0])

        return FileChecker(filepath.resolve())