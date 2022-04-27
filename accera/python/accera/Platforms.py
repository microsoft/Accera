####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import sys
import importlib.resources
from platform import machine
from enum import Enum, unique
from hatlib import LibraryReference as HATLibraryReference


@unique
class Platform(Enum):
    # TODO: should we reconcile with hatlib.OperatingSystem?
    HOST = "host"
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "mac"
    MACOS_ARM64 = "mac_arm64"
    ANDROID = "android"
    IOS = "ios"
    RASPBIAN = "raspbian"


@unique
class LibraryDependency(Enum):
    OPENMP = "openmp"
    VULKAN = "vulkan"


def find_vulkan_wrapper(file_name):
    try:
        from ._version import __version__
    except:
        # CMake-driven builds do not generate a _version.py
        __version__ = ""

    try:
        with importlib.resources.path("accera", file_name) as p:
            return {
                "target_file": str(p),
                "version": __version__
            }
    except FileNotFoundError:
        # not currently installed
        return None


# TODO: rename and export so that it is updatable
platform_libraries = {
    LibraryDependency.VULKAN: {
        Platform.LINUX: find_vulkan_wrapper("libacc-vulkan-runtime-wrappers.so"),
        Platform.MACOS: find_vulkan_wrapper("libacc-vulkan-runtime-wrappers.dylib"),
        Platform.WINDOWS: find_vulkan_wrapper("acc-vulkan-runtime-wrappers.lib")
    },
    LibraryDependency.OPENMP: {
        Platform.LINUX: {
            "target_file": "-lomp5",
            "version": ""
        },
        Platform.MACOS: {
            "target_file": "-lomp",
            "version": ""
        },
        Platform.MACOS_ARM64: {
            "target_file": "/opt/homebrew/lib/libomp.dylib",
            "version": ""
        },
    # best effort: assumes lib will be in the current working dir,
    # because the install location is non-deterministic
        Platform.WINDOWS: {
            "target_file": "libomp.lib",
            "version": ""
        }
    }
}


def get_library_reference(dependency: LibraryDependency, platform: Platform):

    if platform == Platform.HOST:
        if sys.platform.startswith("win"):
            platform = Platform.WINDOWS
        elif sys.platform.startswith("darwin"):
            if machine() == "arm64":
                platform = Platform.MACOS_ARM64
            else:
                platform = Platform.MACOS
        else:
            platform = Platform.LINUX

    if dependency in platform_libraries and platform in platform_libraries[dependency] and platform_libraries[
            dependency][platform]:
        return HATLibraryReference(name=dependency.value, **platform_libraries[dependency][platform])

    return None
