#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

# Authors: Mason Remy
# Requires: Python 3.7+
####################################################################################################

import argparse
import os

from utilities import *
from rc_gpu_runner_config import RCGPURunnerConfig

def rc_gpu_runner(rc_mlir_input_path, mlir_runner_utils_path=None, vulkan_runtime_wrapper_path=None,
              verbose=False, printRCIR=False, printVulkanIR=False, warmupCount=0,
              runCount=10, stdout=None, stderr=None):
    rc_gpu_runner_path = os.path.abspath(RCGPURunnerConfig.rc_gpu_runner)
    rc_gpu_runner_args = [rc_mlir_input_path]
    if mlir_runner_utils_path:
        rc_gpu_runner_args += ["--mlir-runner-utils={}".format(mlir_runner_utils_path)]

    if vulkan_runtime_wrapper_path:
        rc_gpu_runner_args += ["--vulkan-runtime-wrapper={}".format(vulkan_runtime_wrapper_path)]

    if verbose:
        rc_gpu_runner_args += ["--verbose"]

    if printRCIR:
        rc_gpu_runner_args += ["--printRCIR"]

    if printVulkanIR:
        rc_gpu_runner_args += ["--printVulkanIR"]

    if warmupCount:
        rc_gpu_runner_args += ["--warmupCount={}".format(str(int(warmupCount)))]

    if runCount:
        rc_gpu_runner_args += ["--runCount={}".format(str(int(runCount)))]

    command = " ".join([rc_gpu_runner_path] + rc_gpu_runner_args)
    run_command(command, stdout=stdout, stderr=stderr)

def add_rc_gpu_runner_args(parser):
    parser.add_argument("input", metavar="path/to/module.mlir", type=str, help="Path to the MLIR file to JIT run on the GPU")
    parser.add_argument("--mlir-runner-utils", type=str, help="Alternative path to mlir_runner_utils shared library to use")
    parser.add_argument("--vulkan-runtime-wrapper", type=str, help="Alternative path to vulkan-runtime-wrapper shared library to use")
    parser.add_argument("--verbose", action="store_true", help="Print more verbose logging from acc-gpu-runner")
    parser.add_argument("--printRCIR", action="store_true", help="Print module after running Accera passes on it to console")
    parser.add_argument("--printVulkanIR", action="store_true", help="Print module after running GPU and Vulkan passes on it to console")
    parser.add_argument("--warmupCount", type=int, default=0, help="Number of warmup runs to perform")
    parser.add_argument("--runCount", type=int, default=10, help="Number of timed runs to perform")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_rc_gpu_runner_args(parser)
    args = parser.parse_args()

    rc_mlir_input_path = os.path.abspath(args.input)
    mlir_runner_utils = None
    if args.mlir_runner_utils:
        mlir_runner_utils = os.path.abspath(args.mlir_runner_utils)

    vulkan_runtime_wrapper = None
    if args.vulkan_runtime_wrapper:
        vulkan_runtime_wrapper = os.path.abspath(args.vulkan_runtime_wrapper)

    rc_gpu_runner(rc_mlir_input_path=rc_mlir_input_path,
                  mlir_runner_utils_path=mlir_runner_utils,
                  vulkan_runtime_wrapper_path=vulkan_runtime_wrapper,
                  verbose=args.verbose,
                  printRCIR=args.printRCIR,
                  printVulkanIR=args.printVulkanIR,
                  warmupCount=args.warmupCount,
                  runCount=args.runCount)
