#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Authors: Mason Remy
# Requires: Python 3.7+
####################################################################################################

import os
import subprocess
import sys
import shutil
import shlex
import platform
import io

try:
    from .build_config import BuildConfig    # Package mode
except:
    from build_config import BuildConfig    # CLI mode

config = BuildConfig()


class OpenFile:
    def __init__(self, path, open_arg, pretend=False):
        self.path = path
        self.pretend = pretend
        self.open_arg = open_arg
        self.file = None

    def __enter__(self):
        if self.pretend:
            return None
        else:
            self.file = open(self.path, self.open_arg)
            return self.file

    def __exit__(self, exit_type, exit_value, exit_traceback):
        if self.file:
            self.file.__exit__(exit_type, exit_value, exit_traceback)
            self.file = None


def rmdir(path, pretend=False, quiet=True):
    if not quiet:
        print("rm -rf {}".format(path))

    if not pretend:
        shutil.rmtree(path)


def makedir(path, pretend=False, quiet=True):
    if not quiet:
        print("mkdir {}".format(path))

    if not pretend:
        os.makedirs(path, exist_ok=True)


def preprocess_command(command_to_run, shell, cmake_command=True):
    if cmake_command:
        command_to_run = f'cmake -E env LLVM_SYMBOLIZER_PATH="{config.llvm_symbolizer}" {command_to_run}'
    if platform.system() == "Windows":
        return command_to_run
    elif type(command_to_run) == str and not shell:
        return shlex.split(command_to_run)
    elif type(command_to_run) == list and shell:
        return subprocess.list2cmdline(command_to_run)
    else:
        return command_to_run


def dump_file_contents(iostream):
    if isinstance(iostream, io.TextIOBase):
        with open(iostream.name, "r") as f:
            print(f.name)
            print(f.read())


def run_command(
    command_to_run,
    working_directory=None,
    cmake_command=False,
    stdout=None,
    stderr=None,
    shell=False,
    pretend=False,
    quiet=True
):
    if not working_directory:
        working_directory = os.getcwd()

    if not quiet:
        print(f"\ncd {working_directory}")
        print(f"{command_to_run}\n")

    command_to_run = preprocess_command(command_to_run, shell, cmake_command=cmake_command)
    if not pretend:
        with subprocess.Popen(command_to_run, close_fds=(platform.system() != "Windows"), shell=shell, stdout=stdout,
                              stderr=stderr, cwd=working_directory) as proc:
            proc.wait()
            if proc.returncode:
                dump_file_contents(stderr)
                dump_file_contents(stdout)
                raise subprocess.CalledProcessError(proc.returncode, " ".join(command_to_run))


def rename_files_in_dir(root_dir, filename_replacements_dict):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            filepath = os.path.abspath(os.path.join(root, filename))
            if filename in filename_replacements_dict:
                new_filepath = os.path.abspath(os.path.join(root, filename_replacements_dict[filename]))
                if os.path.exists(new_filepath):
                    os.remove(new_filepath)
                os.rename(filepath, new_filepath)


def replace_file_text_in_dir(root_dir, text_replacements_dict, replace_file_extensions=[".cpp", ".h", ".txt"]):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            filepath = os.path.abspath(os.path.join(root, filename))
            if os.path.splitext(filepath)[1] in replace_file_extensions:
                with open(filepath, 'r') as unmodified_file:
                    file_text = unmodified_file.read()
                for find_text in text_replacements_dict:
                    replace_text = text_replacements_dict[find_text]
                    file_text = file_text.replace(find_text, replace_text)
                with open(filepath, 'w') as modified_file:
                    modified_file.write(file_text)


def get_built_target_path(build_dir, build_config, built_target_name):
    if config.config_in_build_path:
        return os.path.join(build_dir, build_config, built_target_name)
    else:
        return os.path.join(build_dir, built_target_name)


def get_cmake_initialization_cmd(build_config="Release"):
    return 'cmake .. -DCMAKE_C_COMPILER="{0}" -DCMAKE_CXX_COMPILER="{1}" -DCMAKE_BUILD_TYPE={2} -DLLVM_CUSTOM_PATH={4} -DUSE_LIBCXX={5} {3}'.format(
        config.c_compiler, config.cxx_compiler, build_config, config.additional_cmake_init_args,
        config.llvm_custom_path, config.use_libcxx
    )


def get_cmake_build_cmd(build_config="Release"):
    return 'cmake --build . --config {}'.format(build_config)


def set_high_performance_gpu(target_executable, pretend=False, quiet=True):
    if platform.system() == "Windows":
        command = "@PowerShell -Command Set-ItemProperty HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences -Name (Resolve-Path {}) -Value \"GpuPreference='2;'\"".format(
            target_executable
        )
        run_command(command, cmake_command=False, shell=True, pretend=pretend, quiet=quiet)


def is_windows():
    return sys.platform.startswith("win")
