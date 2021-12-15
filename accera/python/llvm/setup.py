# -*- coding: utf-8 -*-

####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# Note: must be run from this folder, because setuptools will look for setup.cfg in the
# current working directory.

from setuptools import setup

import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(SCRIPT_DIR / "../setuputils"))
import setuputils as utils


class ComponentBuild(utils.CMakeBuild):
    def initialize_options(self):
        super().initialize_options()
        self.component = "accera-llvm"


class ComponentInstall(utils.ComponentInstallBase):
    def initialize_options(self):
        super().initialize_options()
        self.component = "accera-llvm"
        self.script_dir = SCRIPT_DIR


# runs cmake in the repo root to install the component
setup(ext_modules=[utils.CMakeExtension("_lang_python", utils.repo_root())],
      cmdclass=dict(build_ext=ComponentBuild, install_lib=ComponentInstall),
      use_scm_version=utils.scm_version(str(SCRIPT_DIR / "_version.py")))
