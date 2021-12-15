# -*- coding: utf-8 -*-

####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from setuptools import setup

import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(SCRIPT_DIR / "accera/python/setuputils"))
import setuputils as utils

setup(ext_modules=[utils.CMakeExtension("_lang_python")],
      cmdclass=dict(build_ext=utils.CMakeBuild),
      use_scm_version=utils.scm_version("accera/python/accera/_version.py"))
