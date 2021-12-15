#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import pathlib
import unittest

class VerifyPackage():
    def __init__(self, testcase: unittest.TestCase, package_name: str, output_dir: str=None):
        self.testcase = testcase
        self.hat_file = pathlib.Path(output_dir or pathlib.Path.cwd()) / f"{package_name}.hat"

    def __enter__(self):
        if self.hat_file.exists():
            self.hat_file.unlink() # the missing_ok flag is Python 3.8+ only

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.testcase.assertTrue(self.hat_file.is_file())
