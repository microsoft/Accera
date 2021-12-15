# -*- coding: utf-8 -*-

####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#
# Component setup utilities for building cmake-based python packages
####################################################################################################

# cf https://github.com/pybind/cmake_example/blob/master/setup.py

import os
import sys
import pathlib
import subprocess

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# Build for at least macOS 10.15 when compiling on a 10.15 system or above.
# This may be overridden by setting MACOSX_DEPLOYMENT_TARGET before calling setup.py
# cf. https://github.com/pandas-dev/pandas/blob/master/setup.py
if (sys.platform == "darwin") and ("MACOSX_DEPLOYMENT_TARGET" not in os.environ):
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.15"


def developer_mode():
    "True if we are in developer mode, else we are in release packaging mode"
    # TODO: any better way?
    return os.environ.get("ACCERA_PACKAGE_FOR_CI", "0") != "1"


def repo_root() -> pathlib.Path:
    "Returns the repository root relative to this file"
    # note: update if we move this file elsewhere
    return pathlib.Path(os.path.abspath(__file__)).parent / "../../../"


class ChdirRepoRoot():
    "Context manager that changes directory to the repository root while active"

    def __init__(self):
        self.repo_root = repo_root()
        self.cwd = os.getcwd()

    def __enter__(self):
        if self.repo_root.is_dir:
            os.chdir(str(self.repo_root))

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.component = None

    def get_component_build_lib(self):
        "Gets the component-specific staging directory"
        return f"{self.build_lib}-{self.component}" if self.component else self.build_lib

    def build_extension(self, ext):
        with ChdirRepoRoot():    # CMake must be run at the repository root
            cfg = "Debug" if self.debug else "RelWithDebInfo"

            # CMake lets you override the generator - we need to check this.
            # Can be set with Conda-Build, for example.
            cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
            llvm_setup_variant = os.environ.get("LLVM_SETUP_VARIANT", "Default")

            # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
            # ACCERA_VERSION_INFO shows you how to pass a value into the C++ code
            # from Python.
            cmake_args = [
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DACCERA_VERSION_INFO={self.distribution.get_version()}",
                f"-DCMAKE_BUILD_TYPE={cfg}",    # not used on MSVC, but no harm
                f"-DLLVM_SETUP_VARIANT={llvm_setup_variant}"
            ]
            build_args = []

            if self.compiler.compiler_type != "msvc":
                # Using Ninja-build since it a) is available as a wheel and b)
                # multithreads automatically. MSVC would require all variables be
                # exported for Ninja to pick it up, which is a little tricky to do.
                # Users can override the generator with CMAKE_GENERATOR in CMake
                # 3.15+.
                if not cmake_generator:
                    cmake_args += ["-GNinja"]

            else:

                # Single config generators are handled "normally"
                single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

                # CMake allows an arch-in-generator style for backward compatibility
                contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

                # Specify the arch if using MSVC generator, but only if it doesn't
                # contain a backward-compatibility arch spec already in the
                # generator name.
                if not single_config and not contains_arch:
                    cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

                # Multi-config generators have a different way to specify configs
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={self.get_component_build_lib()}"]
                build_args += ["--config", cfg]

            # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
            # across all generators.
            if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
                # self.parallel is a Python 3 only way to set parallel jobs by hand
                # using -j in the build_ext call, not supported by pip or PyPA-build.
                if hasattr(self, "parallel") and self.parallel:
                    # CMake 3.12+ only.
                    build_args += ["-j{}".format(self.parallel)]

            if self.compiler.compiler_type == "msvc":
                self.build_temp = self.build_temp.replace("Release", "RelWithDebInfo")
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            cwd = os.path.join(os.getcwd(), self.build_temp)
            subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=cwd)
            subprocess.check_call(["cmake", "--build", ".", "--target", ext.name] + build_args, cwd=cwd)

            install_dir = os.path.join(ext.sourcedir, f"{self.get_component_build_lib()}")
            if not os.path.exists(install_dir):
                os.makedirs(install_dir)
            self.announce(f"Attempting to install files to {install_dir}", level=1)
            component = self.component or "accera"
            subprocess.check_call(
                ["cmake", "--install", ".", "--prefix", install_dir, "--component", component, "--config", cfg],
                cwd=cwd)

            def is_top_level_package():
                return not self.component

            if developer_mode() and is_top_level_package():
                # install the other components for development purposes
                # in packaging mode, the setup.py's for each components will be called separately
                for component in ["accera-compilers", "accera-llvm", "accera-gpu"]:
                    subprocess.check_call(
                        ["cmake", "--install", ".", "--prefix", install_dir, "--component", component, "--config", cfg],
                        cwd=cwd)


class ComponentInstallBase(install_lib):
    """Customizes install_lib to copy files from a component-specific staging directory
    cf. https://github.com/TylerGubala/blenderpy/blob/master/setup.py
    """
    def initialize_options(self):
        super().initialize_options()
        self.component = None
        self.script_dir = None

    def get_component_build_dir(self):
        return f"{self.build_dir}-{self.component}" if self.component else self.build_dir

    def run(self):
        self.build_dir = str(repo_root() / self.get_component_build_dir())

        self.announce(f"Copying files from {self.build_dir}", level=2)
        super().run()


def scm_version(version_file):
    # pip install setuptools_scm
    from setuptools_scm import get_version
    from functools import partial

    with ChdirRepoRoot():    # git_version() can only run at the repository root
        # Dev commit version format (tag must be of the format n(.n)+):
        # no distance and clean:
        #   {tag}
        # distance and clean:
        #   {++tag}.dev{distance}+{scm letter}{revision hash}
        # no distance and not clean:
        #   {tag}+dYYYYMMDD
        # distance and not clean:
        #   {++tag}.dev{distance}+{scm letter}{revision hash}.dYYYYMMDD
        dev_commit_version = get_version(version_scheme="python-simplified-semver")
        version_str = get_version(version_scheme="python-simplified-semver", local_scheme="no-local-version")

        # Whl filename version format (tag must be of the format n(.n)+):
        # ACCERA_PACKAGE_FOR_CI=1:
        #   no distance:
        #     {tag}
        #   distance:
        #     {++tag}.dev{distance}
        # else:
        #   no distance:
        #     {tag}
        #   distance:
        #     {++tag}
        def whl_version(version_str, version):
            if developer_mode():
                version_str = version_str.partition(".dev")[0]
            return version_str

        return {
            "version_scheme": partial(whl_version, version_str),
            "local_scheme": "no-local-version",    #  disable local version for whl
            "fallback_version": version_str,
            "write_to": version_file,
            "write_to_template": f"__version__ = '{dev_commit_version}'"    #  provide commit version in __version__
        }
