####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import os
import ctypes
import logging
import numpy as np
import pathlib
from dataclasses import dataclass
from functools import reduce
from typing import Any, List
import time
from ..hat import HATPackage
from ..accc import AcceraSharedLibraryBuildProject
from ..utilities import is_windows

# Set the performance counter scale
if hasattr(time, 'perf_counter_ns'):
    perf_counter = time.perf_counter_ns
    perf_counter_scale = 1000000000
else:
    perf_counter = time.perf_counter
    perf_counter_scale = 1


# TODO: migrate to use HAT tools
@dataclass
class _Parameter:
    declared_type: Any = None
    buffer: Any = None
    size: int = 0
    element_type: type = np.float32
    element_size_bytes: int = 4

    def __init__(self, param_def):

        self.size = reduce(lambda x, y: x * y, param_def.shape)
        # TODO: make this more generic
        if param_def.declared_type == "float*":
            self.element_type = np.float32
            self.element_size_bytes = 4
            buffer_type = ctypes.c_float * self.size
            self.buffer = buffer_type(*([0.] * self.size))  # allocates the buffer
            self.declared_type = ctypes.POINTER(ctypes.c_float)
        elif param_def.declared_type == "double*":
            self.element_type = np.float64
            self.element_size_bytes = 8
            buffer_type = ctypes.c_double * self.size
            self.buffer = buffer_type(*([0.] * self.size))  # allocates the buffer
            self.declared_type = ctypes.POINTER(ctypes.c_double)
        elif param_def.declared_type == "int64_t*":
            self.element_type = np.int64
            self.element_size_bytes = 8
            buffer_type = ctypes.c_int64 * self.size
            self.buffer = buffer_type(*([0] * self.size))  # allocates the buffer
            self.declared_type = ctypes.POINTER(ctypes.c_int64)
        elif param_def.declared_type == "int32_t*":
            self.element_type = np.int32
            self.element_size_bytes = 4
            buffer_type = ctypes.c_int32 * self.size
            self.buffer = buffer_type(*([0] * self.size))  # allocates the buffer
            self.declared_type = ctypes.POINTER(ctypes.c_int32)
        elif param_def.declared_type == "int16_t*":
            self.element_type = np.int16
            self.element_size_bytes = 2
            buffer_type = ctypes.c_int16 * self.size
            self.buffer = buffer_type(*([0] * self.size))  # allocates the buffer
            self.declared_type = ctypes.POINTER(ctypes.c_int16)
        elif param_def.declared_type == "void*":
            self.element_type = np.int8
            self.element_size_bytes = 1
            buffer_type = ctypes.c_int8 * self.size
            self.buffer = buffer_type(*([0] * self.size))  # allocates the buffer
            self.declared_type = ctypes.POINTER(ctypes.c_int8)

        else:
            raise NotImplementedError(f"Unsupported declared_type: {param_def.declared_type}")

def _check_type_match(elements, parameters):
    ''' This function checks if the types of the input element and the expected parameter match
    '''
    for value, param in zip(elements, parameters):
        if value.dtype != param.element_type:
            raise ValueError(f"Input elements of type {value.dtype} doesn't match the expected function parameter of type {param.element_type}.")

def _numpy_to_c_type(array, parameters):
    # for each parameter, use the known pre_values instead of its buffer, so that we have a basis to compare the post_values
    # BUGBUG: ndpointer does not resolve correctly, using raw ctypes.POINTER's instead
    return [arr.flatten().ctypes.data_as(p.declared_type) for p, arr in zip(parameters, array)]

def _generate_input_sets(parameters: List[_Parameter], input_sets_minimum_size_MB: int, num_additional: int = 10):
    from copy import deepcopy
    set_size = reduce(lambda x, y: x + y, [p.size * p.element_size_bytes for p in parameters])
    num_input_sets = (input_sets_minimum_size_MB * 1024 * 1024 // set_size) + 1 + num_additional

    logging.debug(f"[Benchmarking] Using {num_input_sets} input sets, each {set_size} bytes")
    if len(parameters):    # sanity check
        assert (deepcopy(parameters)[0].buffer != parameters[0].buffer)
    return [[ctypes.cast(p.buffer, p.declared_type) for p in deepcopy(parameters)] for _ in range(num_input_sets)]


class HatPackageLoader:
    def __init__(self, package_dirpath: str):
        """Builds a shared library for the HAT package
        Args:
            package_dirpath: path to a HAT package
        """
        # TODO: migrate to use HAT tools
        hat_package = HATPackage(package_dirpath)

        # TODO: for now, ignore target information when getting functions
        self.functions = {fn.name: fn for fn in hat_package.get_functions()}
        self.hat_functions = hat_package.get_functions()

        main_cpp = self._generate_main_cpp(hat_package)
        build_project = AcceraSharedLibraryBuildProject(hat_package.path,
                                                        hat_package.name,
                                                        exports=list(self.functions),
                                                        link_target_paths=hat_package.link_targets,
                                                        additional_srcs=[main_cpp])
        self.built_so = build_project.build()

    def _generate_main_cpp(self, hat_package: HATPackage):
        """ctypes-specific logic:
        Generate a cpp that includes the HAT file, so that any inline functions
        can be resolved when ctypes loads the shared object. This simulates a user
        adding a main.cpp that includes the HAT file (which ctypes does not support).
        """
        cpp_path = hat_package.path / "main.cpp"
        with open(cpp_path, "w") as f:
            for hat_file in hat_package.path.glob("*.hat"):
                print(f"#include <{hat_file.name}>", file=f)
        return str(cpp_path.absolute())

    def _load_library(self):
        "Loads the dynamic library"
        # add Accera dependencies needed by the shared object
        # these cases will still require setup:
        #  - windows (python <= 3.7): caller needs to set PATH before invoking python
        #       e.g. PATH=<path_to_accera>;%PATH%
        #  - macOS: caller needs to set DYLD_LIBRARY_PATH before invoking python
        #       e.g. DYLD_LIBRARY_PATH=<path_to_accera>
        # Note: assumes the vulkan loader is already part of the DLL search paths via the Vulkan SDK
        if hasattr(os, "add_dll_directory"):
            for d in self.built_so.dependencies:
                os.add_dll_directory(d.parent)
        # else:
            # os.add_dll_directory is not supported on this system.
            # You may need to append the path to this package to one of these environment variables: PATH (windows),
            # LD_LIBRARY_PATH (linux), or DYLD_LIBRARY_PATH (macOS) to help us locate the shared object dependencies

        # strip off the suffix for windows
        lib_path = pathlib.Path(
            self.built_so.library_path).with_suffix("") if is_windows() else self.built_so.library_path

        return ctypes.cdll.LoadLibrary(str(lib_path))


class CorrectnessCheck(HatPackageLoader):
    """A simplified interface for correctness-checking without benchmarking

    Use AutoBenchmark if you need to perform grid search with correctness checking
    """

    def run(self, function_name: str, before: List["numpy.ndarray"], after: List["numpy.ndarray"], tolerance: float = 1e-5):
        """Performs correctness checking on a function
        Args:
            function_name: name of the function
            before: inputs and outputs to pass as parameters to the function
            after: desired inputs and outputs after the function is called, will be compared against the actual values
            correctness_check_values: pre-invocation and post-invocation parameter values
            tolerance: tolerance
        """
        if function_name not in self.functions:
            raise ValueError(f"{function_name} is not found")

        lib = self._load_library()

        self._check_correctness(function_name, lib, {"pre": before, "post": after}, tolerance)


    def _check_correctness(self, function_name, lib, correctness_check_values: dict, tolerance: float):
        function_sym = lib[function_name]
        function_def = self.functions[function_name]
        parameters = [_Parameter(param_def) for param_def in function_def.arguments]
        function_sym.argtypes = [p.declared_type for p in parameters]

        pre_values = correctness_check_values["pre"]
        _check_type_match(pre_values, parameters)

        post_values = correctness_check_values["post"]
        _check_type_match(post_values, parameters)

        args = _numpy_to_c_type(pre_values, parameters)

        # Pre-run
        if "packing_functions" in function_def.auxiliary:
            self._pack_args(function_def, lib, correctness_check_values, args)

        # run
        function_sym(*args)

        # Post-run
        if "unpacking_functions" in function_def.auxiliary:
            self._unpack_args(function_def, lib, correctness_check_values, args, parameters)

        for p, expected, actual in zip(parameters, post_values, args):
            assert expected.size == p.size, f"post_value size is {expected.size}, while parameter size is {p.size}"
            np.testing.assert_allclose(expected.flatten(), np.ctypeslib.as_array(actual, shape=(p.size, )), tolerance)

    def _pack_args(self, function_def, lib, correctness_check_values, args):
        packing_functions = function_def.auxiliary["packing_functions"]
        pack_inter_values = correctness_check_values["pack_inter"]

        pre_values = correctness_check_values["pre"]

        # Pack the arguments that requires packing
        for i, packing_function_name in enumerate(packing_functions):
            arg_idx = packing_functions[packing_function_name]
            func_def = self.functions[packing_function_name]

            # Assume packing functions pack one value only
            parameters = [_Parameter(param_def) for param_def in func_def.arguments]
            intermediate_args = [pre_values[arg_idx], pack_inter_values[i]]
            _check_type_match(intermediate_args, parameters)
            intermediate_args = [arg.flatten().ctypes.data_as(param.declared_type) for arg, param in zip(intermediate_args, parameters)]

            packing_function = lib[packing_function_name]
            packing_function(*intermediate_args)

            # Replace the main function argument with the packed version
            args[arg_idx] = intermediate_args[1]

    def _unpack_args(self, function_def, lib, correctness_check_values, args, parameters):
        unpacking_functions = function_def.auxiliary["unpacking_functions"]
        unpack_inter_values = correctness_check_values["unpack_inter"]

        for i, unpacking_function_name in enumerate(unpacking_functions):
            arg_idx = unpacking_functions[unpacking_function_name]
            func_def = self.functions[unpacking_function_name]

            out_param = _Parameter(func_def.arguments[1])
            unpacking_out = unpack_inter_values[i]
            # TODO: check if input parameter types to the function match the pre-calculated argument
            _check_type_match([unpacking_out], [out_param])
            unpacking_out = unpacking_out.flatten().ctypes.data_as(out_param.declared_type)

            unpacking_function = lib[unpacking_function_name]
            unpacking_function(*[args[arg_idx], unpacking_out])

            # Replace the main function output with the unpacked version
            args[arg_idx] = unpacking_out

            # Replace the output parameter with the unpacked parameter
            # NOTE: The parameters are changed in case of unpacking functions only and not with packing functions
            #       because the output parameters of a compound function is the output parameters of the unpacking function (if any)
            parameters[arg_idx] = out_param


class AutoBenchmark(CorrectnessCheck):
    """Automatically benchmarks functions in a HAT package on the current system

    Requirements:
        A compilation toolchain in your PATH: cl.exe & link.exe (Windows), gcc (Linux), or clang (macOS)
    """
    def run(self,
            function_name: str,
            warmup_iterations: int = 10,
            min_timing_iterations: int = 100,
            min_time_in_sec: int = 10,
            correctness_check_values: dict = None,
            tolerance: float = 1e-5,
            input_sets_minimum_size_MB=50) -> float:
        """Runs benchmarking for a function.
           Multiple inputs are run through the function until both minimum time and minimum iterations have been reached.
           The mean duration is then calculated as mean_duration = total_time_elapsed / total_iterations_performed.
        Args:
            function_name: name of the function
            warmup_iterations: number of warmup iterations
            min_timing_iterations: minimum number of timing iterations
            min_time_in_sec: minimum amount of time to run the benchmark
            correctness_check_values: optional pre-invocation and post-invocation parameter values for correctness checking
            tolerance: tolerance for correctness checking
            input_sets_minimum_size_MB: generate enough input sets to exceed this size to avoid cache hits
        Returns:
            Mean duration in seconds,
            Vector of timings in seconds for each batch that was run
        """
        if function_name not in self.functions:
            raise ValueError(f"{function_name} is not found")

        lib = self._load_library()

        # construct the arguments
        # TODO: function_def.return is ignored
        function_def = self.functions[function_name]

        require_packing = "packing_functions" in function_def.auxiliary
        require_unpacking = "unpacking_functions" in function_def.auxiliary

        if correctness_check_values:
            self._check_correctness(function_name, lib, correctness_check_values, tolerance)

        # Profile the function
        mean_elapsed_time, batch_timings = self._profile_function(function_name, lib, warmup_iterations, min_timing_iterations, min_time_in_sec, input_sets_minimum_size_MB)
        print(f"[Benchmarking] Mean duration per iteration: {mean_elapsed_time:.8f}s")

        # Profile the packing and unpacking overhead (if any)
        if require_packing:
            packing_overhead = self._profile_auxiliary_functions(function_def, "packing_functions", lib, warmup_iterations, min_timing_iterations, min_time_in_sec, input_sets_minimum_size_MB)
            print(f"[Benchmarking] Packing overhead = {packing_overhead:.8f}s")

        if require_unpacking:
            unpacking_overhead = self._profile_auxiliary_functions(function_def, "unpacking_functions", lib, warmup_iterations, min_timing_iterations, min_time_in_sec, input_sets_minimum_size_MB)
            print(f"[Benchmarking] Unpacking overhead = {unpacking_overhead:.8f}s")

        return mean_elapsed_time, batch_timings

    def _profile_function(self, function_name, lib, warmup_iterations, min_timing_iterations, min_time_in_sec, input_sets_minimum_size_MB):
        function_def = self.functions[function_name]
        parameters = [_Parameter(param_def) for param_def in function_def.arguments]

        # generate sufficient input sets to overflow the L3 cache, since we don't know the size of the model
        # we'll make a guess based on the minimum input set size
        input_sets = _generate_input_sets(parameters, input_sets_minimum_size_MB)
        print(f"Using {len(input_sets)} input sets")

        function_sym = lib[function_name]
        function_sym.argtypes = [p.declared_type for p in parameters]

        print(f"[Benchmarking] Warming up for {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            for calling_args in input_sets:
                function_sym(*calling_args)

        # TODO: profile this portion to measure the overhead of calling the function
        print(f"[Benchmarking] Timing for at least {min_time_in_sec}s and at least {min_timing_iterations} iterations...")
        start_time = perf_counter()
        end_time = perf_counter()

        i = 0
        i_max = len(input_sets)
        iterations = 0
        batch_timings = []
        while ((end_time - start_time) / perf_counter_scale) < min_time_in_sec:
            batch_start_time = perf_counter()
            for _ in range(min_timing_iterations):
                iterations += 1
                function_sym(*input_sets[i])
                i = iterations % i_max
            end_time = perf_counter()
            batch_timings.append((end_time - batch_start_time) / perf_counter_scale)

        elapsed_time = ((end_time - start_time) / perf_counter_scale)
        mean_elapsed_time = elapsed_time / iterations
        return mean_elapsed_time, batch_timings

    def _profile_auxiliary_functions(self, function_def, aux_function, lib, warmup_iterations, min_timing_iterations, min_time_in_sec, input_sets_minimum_size_MB):
        aux_functions = function_def.auxiliary[aux_function]
        total_overhead = sum([mean for mean, _ in [self._profile_function(function_name, lib, warmup_iterations, min_timing_iterations, min_time_in_sec, input_sets_minimum_size_MB) for function_name in aux_functions]])
        return total_overhead
