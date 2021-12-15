#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import gc
import os
import sys
import platform
sys.path = ['.'] + sys.path

import argparse
import onnxruntime as onnxrt

import numpy as np
import subprocess
import pathlib
import onnx_hat_package
import json
import pandas as pd
import time

if hasattr(time, 'perf_counter_ns'):
    perf_counter = time.perf_counter_ns
    perf_counter_scale = 1000000000
else:
    perf_counter = time.perf_counter
    perf_counter_scale = 1

VERIFY_TESTING_MODE = False

model_base_name_list =  ["gpt2_small"] # "gpt2_medium", "gpt2_large", "gpt2_xlarge"

model_batch_sep_list = [(1, 10)] # (1, 128), (1,256), (1,1024)

def _get_os_name():
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform.startswith('darwin'):
        return 'macos'
    else:
        return 'linux'

def _is_windows():
    return _get_os_name() == 'windows'

def _get_architecture_name():
    if platform.machine().startswith('arm'):
        return 'arm'
    else:
        return platform.machine()

def _create_dllmain(filename: str):
    f = pathlib.Path(filename).with_suffix(".cc")
    with open(os.path.join(f), 'w') as dllmain_cc:
        print(
            """
            #include <windows.h>
            BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }
            """,
            file=dllmain_cc
        )

    obj_file = f.with_suffix(".obj")
    subprocess.run(['cl', '/Fo' + str(obj_file.absolute()), '/c', str(f.absolute())], check=True)
    return str(obj_file.absolute())

def accera_provider_settings(hat_package_dir):

    # TODO: HAT packages should have an entry for the base
    hat_dir = pathlib.Path(hat_package_dir)
    package_name = hat_dir.name
    print(hat_dir)
    from onnx_hat_package import ONNXHATPackage
    hat = ONNXHATPackage(hat_dir)

    found_functions = hat.get_functions_for_target(
        os=_get_os_name(), arch=_get_architecture_name())

    print("Found the following functions:")
    for fn in found_functions:
        print(f"\t{fn.name}")

    objs = set([str(pathlib.Path(func.link_file).absolute()) for func in
                found_functions if hasattr(func, 'link_file')])

    so_file = (
        hat_dir / package_name).with_suffix(".dll" if _is_windows() else ".so")
    so_file = str(so_file.absolute())

    if _is_windows():
        objs.add(_create_dllmain(
            (hat_dir / package_name).with_name(package_name + "_dllmain.cc")))

        subprocess.run(
            ['link', '-dll', '-FORCE:MULTIPLE'] +
            [f'-EXPORT:{fn.name}' for fn in found_functions] +
            [f'-out:{so_file}'] + list(objs), check=True)

    else:
        subprocess.run(
            ['g++',
             '-shared',
             '-fPIC',
             '-o', so_file
             ] + list(objs), check=True)
    settings = {}
    settings['custom_library'] = so_file

    node_to_func = {}
    for func in found_functions:
        if not func.onnx: continue

        node_funcs = node_to_func.setdefault(func.onnx[ONNXHATPackage.NodeTypeKey], [])
        node_funcs.append(func.onnx)
    settings['node_to_func'] = node_to_func

    return settings


def create_accera_settings_from_package(hat_package):
    settings = accera_provider_settings(hat_package)
    return {'accera_settings': json.dumps(settings)}


def get_min_input_sets(input_shapes, MB_to_exceed=50):
    "cf ELL/ONNXBenchmarks"
    sizes = []
    for shape in input_shapes:
        size = 1
        for s in shape:
            size *= s
        sizes.append(size)
    set_size = np.sum(sizes) * 4  # Size of float tensors in bytes
    return ((MB_to_exceed * 1 * 1024) // set_size) + 1


def get_input_sets(ort_session, num_additional=10):
    input_shapes = []
    for inp in ort_session.get_inputs():
        input_shapes.append(inp.shape)

    generator = np.random.default_rng(seed=2021)

    # Create inputs
    input_list = ort_session.get_inputs()
    num_input_sets = get_min_input_sets(input_shapes, 1) + num_additional
    print("\t\tUsing {} input sets".format(num_input_sets))
    input_sets = []
    for i in range(num_input_sets):
        ort_inputs = {}
        for i, inp in enumerate(input_list):
            if 'int64' in inp.type:
                ort_inputs[inp.name] = generator.integers(0, 255, size=input_shapes[i], dtype=np.int64)
            else:
                ort_inputs[inp.name] = generator.random(input_shapes[i]).astype(dtype=np.float32)
        input_sets.append(ort_inputs)
    return input_sets


def get_optimized_onnx_model_expected_name(base_name, batch, seq):
    opt_suffix = "_opt"
    specifier = f"_b{batch}_s{seq}"
    expected_name = f"{base_name}{specifier}{opt_suffix}.onnx"
    return expected_name


def test_transformer_model_on_target_per_node(model_name, output_dir):
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    return test_model_on_target(model_name, output_dir, options=options, num_results=100)


def test_transformer_model_on_target(output_dir):
    import onnxruntime as onnxrt

    results = {}
    for base_name in model_base_name_list:
        for batch, seq in model_batch_sep_list:

            options = onnxrt.SessionOptions()
            options.add_free_dimension_override_by_name("batch", batch)
            options.add_free_dimension_override_by_name("sequence", seq)
            options.add_free_dimension_override_by_name("seq", seq)

            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = False

            num_results = 100
            if seq > 128:
                num_results = 10

            optimized_onnx_model = get_optimized_onnx_model_expected_name(base_name, batch, seq)

            cpu_time = test_model_on_target_cpu(optimized_onnx_model, output_dir, options=options, num_results=num_results)

            rc_time = test_model_on_target_accera(optimized_onnx_model, output_dir, options=options, num_results=num_results)

            name = f"{base_name} batch={batch} seq={seq}"
            results[name] = {
                "ort": cpu_time, "hat_ep": rc_time,
                "ort/hat_ep": cpu_time / rc_time,
            }
        temp_df = pd.DataFrame.from_dict(results, orient='index')
        temp_df.to_csv(f"test_transformer_model_partial.csv")

    df = pd.DataFrame.from_dict(results, orient='index')

    print(df)
    df.to_csv("test_transformer_model.csv")


def test_model_on_target_cpu(model, output_dir, outputs=None, options=None, num_results=10000, syms=None, large_model=True, skip_correctness=False, min_time_in_sec=15, num_batches=10):
    os.environ["OMP_NUM_THREADS"] = "1"

    model_name = pathlib.Path(model).name
    output_dir = pathlib.Path(output_dir) / model_name

    providers = [
        ('CPUExecutionProvider', {}),
    ]

    cpu_sess = onnxrt.InferenceSession(
        model, sess_options=options, providers=providers)

    # Create inputs
    input_sets = get_input_sets(cpu_sess, num_additional=num_batches)
    num_input_sets = len(input_sets)

    print(f"Running {num_batches} warm-up iterations")
    for i in range(num_batches):
        cpu_res = cpu_sess.run(outputs, input_sets[i % num_input_sets])

    print("*" * 20)
    print(f"Running CPU session for {min_time_in_sec}s")
    iterations = 0
    wall_clock_start = perf_counter()
    wall_clock_end = perf_counter()
    batch_times = []
    while (wall_clock_end - wall_clock_start) < (min_time_in_sec * perf_counter_scale):
        batch_start = perf_counter()
        for b in range(num_batches):
            iterations += 1
            index = (iterations + num_batches) % num_input_sets
            cpu_res = cpu_sess.run(None, input_sets[index])
        batch_end = perf_counter()
        batch_times.append(batch_end - batch_start)
        wall_clock_end = perf_counter()

    total_time_sec = (wall_clock_end - wall_clock_start) / perf_counter_scale

    cpu_time = (total_time_sec * 1000) / iterations # mean duration in ms
    # Run garbage collection
    gc.collect()

    print(f"\n\n{'*' * 10}\nTesting {model_name} for {min_time_in_sec}s\nCPU Time: {cpu_time}\n{'*' * 10}\n")
    os.environ.pop("OMP_NUM_THREADS")

    return cpu_time

def test_model_on_target_accera(model, output_dir, outputs=None, options=None, num_results=10000, syms=None, large_model=True, skip_correctness=False, min_time_in_sec=15, num_batches=10):
    os.environ["OMP_NUM_THREADS"] = "1"

    model_name = pathlib.Path(model).name
    output_dir = pathlib.Path(output_dir) / model_name

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    accera_settings = create_accera_settings_from_package(output_dir)
    providers = [
        ('AcceraExecutionProvider', accera_settings),
        ('CPUExecutionProvider', {}),
    ]

    try:
        rc_sess = onnxrt.InferenceSession(
            model,  sess_options=options, providers=providers)
    except:
        print("expection",sys.exc_info()[0],"occurred.")

    input_sets = get_input_sets(rc_sess, num_additional=num_batches)
    num_input_sets = len(input_sets)

    print(f"Running {num_batches} warm-up iterations")
    for i in range(num_batches):
        rc_results = rc_sess.run(outputs, input_sets[i % num_input_sets])


    print(f"Running Accera EP session for {min_time_in_sec}s")
    iterations = 0
    wall_clock_start = perf_counter()
    wall_clock_end = perf_counter()
    batch_times = []
    while (wall_clock_end - wall_clock_start) < (min_time_in_sec * perf_counter_scale):
        batch_start = perf_counter()
        for b in range(num_batches):
            iterations += 1
            index = (iterations + num_batches) % num_input_sets
            rc_results = rc_sess.run(None, input_sets[index])
        batch_end = perf_counter()
        batch_times.append(batch_end - batch_start)
        wall_clock_end = perf_counter()

    total_time_sec = (wall_clock_end - wall_clock_start) / perf_counter_scale

    rc_time = (total_time_sec * 1000) / iterations # mean duration in ms
    # Run garbage collection
    gc.collect()

    print(f"\n\n{'*' * 10}\nTesting {model_name} for {min_time_in_sec}s\nHAT Time: {rc_time}\n{'*' * 10}\n")
    os.environ.pop("OMP_NUM_THREADS")

    return rc_time


def test_model_on_target(model, output_dir, outputs=None, options=None, num_results=10000, syms=None, large_model=True, skip_correctness=False, min_time_in_sec=15, num_batches=20):
    os.environ["OMP_NUM_THREADS"] = "1"
    model_name = pathlib.Path(model).name
    output_dir = pathlib.Path(output_dir) / model_name

    providers = [
        ('CPUExecutionProvider', {}),
    ]

    cpu_sess = onnxrt.InferenceSession(
        model, sess_options=options, providers=providers)

    # Create inputs
    input_sets = get_input_sets(cpu_sess, num_additional=num_batches)
    num_input_sets = len(input_sets)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    accera_settings = create_accera_settings_from_package(output_dir)
    providers = [
        ('AcceraExecutionProvider', accera_settings),
        ('CPUExecutionProvider', {}),
    ]

    rc_sess = onnxrt.InferenceSession(
        model,  sess_options=options, providers=providers)

    if not skip_correctness:
        cpu_res = cpu_sess.run(outputs, input_sets[0])
        rc_results = rc_sess.run(outputs, input_sets[0])

        try:
            np.testing.assert_allclose(
                cpu_res, rc_results, rtol=1e-02, atol=1e-04, verbose=True)
        except AssertionError as e:
            print("!" * 20)
            print("Value mismatch detected")

            if VERIFY_TESTING_MODE:
                raise e
            else:
                print(e)
                print("!" * 20)

        # Do warm-up
        print(f"Running {num_batches} warm-up iterations")
        for i in range(num_batches):
            cpu_res = cpu_sess.run(outputs, input_sets[i % num_input_sets])
            rc_results = rc_sess.run(outputs, input_sets[i % num_input_sets])
            if VERIFY_TESTING_MODE:
                np.testing.assert_allclose(
                    cpu_res, rc_results, rtol=1e-02, atol=1e-04, verbose=True)


    print("*" * 20)
    print(f"Running CPU session for {min_time_in_sec}s")
    iterations = 0
    wall_clock_start = perf_counter()
    wall_clock_end = perf_counter()
    batch_times = []
    while (wall_clock_end - wall_clock_start) < (min_time_in_sec * perf_counter_scale):
        batch_start = perf_counter()
        for b in range(num_batches):
            iterations += 1
            index = (iterations + num_batches) % num_input_sets
            cpu_res = cpu_sess.run(None, input_sets[index])
        batch_end = perf_counter()
        batch_times.append(batch_end - batch_start)
        wall_clock_end = perf_counter()

    total_time_sec = (wall_clock_end - wall_clock_start) / perf_counter_scale

    cpu_time = (total_time_sec * 1000) / iterations # mean duration in ms
    # Run garbage collection
    gc.collect()

    print(f"Running Accera EP session for {min_time_in_sec}s")
    iterations = 0
    wall_clock_start = perf_counter()
    wall_clock_end = perf_counter()
    batch_times = []
    while (wall_clock_end - wall_clock_start) < (min_time_in_sec * perf_counter_scale):
        batch_start = perf_counter()
        for b in range(num_batches):
            iterations += 1
            index = (iterations + num_batches) % num_input_sets
            rc_results = rc_sess.run(None, input_sets[index])
        batch_end = perf_counter()
        batch_times.append(batch_end - batch_start)
        wall_clock_end = perf_counter()

    total_time_sec = (wall_clock_end - wall_clock_start) / perf_counter_scale

    rc_time = (total_time_sec * 1000) / iterations # mean duration in ms
    # Run garbage collection
    gc.collect()

    print(f"\n\n{'*' * 10}\nTesting {model_name} for {min_time_in_sec}s\nCPU Time: {cpu_time}\tHAT Time: {rc_time}\n{'*' * 10}\n")
    os.environ.pop("OMP_NUM_THREADS")

    return (cpu_time, rc_time)

def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='hat_packages', help='by default output dir name is "hat_packages"')
    parser.add_argument('-pn', '--per_node', default=False, help='is the test per node test?')
    parser.add_argument('-mn', '--model_name', default='', help='the name of the model for per node test')
    parser.add_argument('-md', '--model_dir', default='', help='the directory of the models for per node test')
    args = parser.parse_args(args)

    if args.per_node == False:
        test_transformer_model_on_target(args.output_dir)
    else:
        if args.model_dir != '':
            model_files = os.listdir(args.model_dir)
            for model_name in model_files:
                model_file = os.path.join(args.model_dir, model_name)
                test_transformer_model_on_target_per_node(model_file, args.output_dir)
                gc.collect()
        else:
            test_transformer_model_on_target_per_node(args.model_name, args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
