#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import os
import sys
sys.path = ['.'] + sys.path
import numpy as np
import argparse
import onnxruntime as onnxrt
import onnx
from helper import get_name
import pathlib
from pathlib import Path

model_base_name_list = ["gpt2_small"] # "gpt2_medium", "gpt2_large", "gpt2_xlarge"

model_batch_sep_list = [(1, 10)] # (1, 128), (1,256), (1,1024)

size_list_per_node_test = [
        # GPT2 small
        ([], (10,768,768), 1.0),
        ([], (10,3702,768), 1.0),
        ([], (10,768,3702), 1.0),
        ([], (10,768,2304), 1.0),
        ([], (128,768,768), 1.0),
        ([], (128,3702,768), 1.0),
        ([], (128,768,3702), 1.0),
        ([], (128,768,2304), 1.0),
        ([], (256,768,768), 1.0),
        ([], (256,3702,768), 1.0),
        ([], (256,768,3702), 1.0),
        ([], (256,768,2304), 1.0),
        ([], (1024,768,768), 1.0),
        ([], (1024,3702,768), 1.0),
        ([], (1024,768,3702), 1.0),
        ([], (1024,768,2304), 1.0),
        # GPT2 medium
        ([], (10, 1024, 1024), 1.0),
        ([], (10, 3072, 1024), 1.0),
        ([], (10, 4096, 1024), 1.0),
        ([], (10, 1024, 4096), 1.0),
        ([], (128, 1024, 1024), 1.0),
        ([], (128, 3072, 1024), 1.0),
        ([], (128, 4096, 1024), 1.0),
        ([], (128, 1024, 4096), 1.0),
        ([], (256, 1024, 1024), 1.0),
        ([], (256, 3072, 1024), 1.0),
        ([], (256, 4096, 1024), 1.0),
        ([], (256, 1024, 4096), 1.0),
        ([], (1024, 1024, 1024), 1.0),
        ([], (1024, 3072, 1024), 1.0),
        ([], (1024, 4096, 1024), 1.0),
        ([], (1024, 1024, 4096), 1.0),
        # GPT2 large
        ([], (10, 1280, 1280), 1.0),
        ([], (10, 3840, 1280), 1.0),
        ([], (10, 5120, 1280), 1.0),
        ([], (10, 1280, 5120), 1.0),
        ([], (128, 1280, 1280), 1.0),
        ([], (128, 3840, 1280), 1.0),
        ([], (128, 5120, 1280), 1.0),
        ([], (128, 1280, 5120), 1.0),
        ([], (256, 1280, 1280), 1.0),
        ([], (256, 3840, 1280), 1.0),
        ([], (256, 5120, 1280), 1.0),
        ([], (256, 1280, 5120), 1.0),
        ([], (1024, 1280, 1280), 1.0),
        ([], (1024, 3840, 1280), 1.0),
        ([], (1024, 5120, 1280), 1.0),
        ([], (1024, 1280, 5120), 1.0),
        # Common to all GPT2
        ([1,12], (10,64,10), 1.0),
        ([1,16], (10,64,10), 1.0),
        ([1,20], (10,64,10), 1.0),
        ([1,12], (128,64,128), 1.0),
        ([1,16], (128,64,128), 1.0),
        ([1,20], (128,64,128), 1.0),
        ([1,12], (256,64,256), 1.0),
        ([1,16], (256,64,256), 1.0),
        ([1,20], (256,64,256), 1.0),
        ([1,12], (1024,64,1024), 1.0),
        ([1,16], (1024,64,1024), 1.0),
        ([1,20], (1024,64,1024), 1.0),
    ]

def get_name(name):
    if os.path.exists(name):
        return name
    rel = os.path.join("testdata", name)
    if os.path.exists(rel):
        return rel
    this = os.path.dirname(__file__)
    data = os.path.join(this, "..", "testdata")
    res = os.path.join(data, name)
    if os.path.exists(res):
        return res
    raise FileNotFoundError("Unable to find '{0}' or '{1}' or '{2}'".format(name, rel, res))

def make_matmul_model(M, N, K, batch=[], filename=None, overwrite=False, use_constant_weights=False, model_dir='testdata'):
    specifier = 'x'.join(map(str, batch)) + '_'.join(map(str, ['', M, N, K]))
    expected_name = filename or f'matmul{specifier}.onnx'

    def make_model():
        from onnx import helper, TensorProto

        initializers = []
        if use_constant_weights:
            b_tensor = helper.make_tensor("B", TensorProto.FLOAT, batch + [K, N], np.random.random(batch + [K, N]).astype(dtype=np.float32).flatten())
            initializers.append(b_tensor)

            inputs = [
                helper.make_tensor_value_info('A', TensorProto.FLOAT, batch + [M, K]),
            ]
        else:
            inputs = [
                helper.make_tensor_value_info('A', TensorProto.FLOAT, batch + [M, K]),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, batch + [K, N]),
            ]


        graph = helper.make_graph(
            [  # nodes
                helper.make_node("MatMul", ["A", "B"], [
                                 "C"], f"MatMul_{specifier}"),
            ],
            f"matmul_{specifier}",  # name
            inputs,
            [  # outputs
                helper.make_tensor_value_info('C', TensorProto.FLOAT, batch + [M, N]),
            ],
            initializers)
        model = helper.make_model(graph)
        model_path = os.path.join(model_dir, expected_name)
        onnx.save(model, model_path)
        return get_name(model_path)

    if overwrite:
        return make_model()

    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()


def make_gemm_model(M, N, K, alpha=1.0, beta=1.0, transA=0, transB=0, filename=None, overwrite=False, use_constant_weights=True, model_dir='testdata'):
    suffix = '_'.join([str(x) for x in ['', M, N, K, alpha, beta, transA, transB]])
    expected_name = filename or f'gemm{suffix}.onnx'

    def make_model():
        from onnx import helper, TensorProto

        def trans_no_trans(val: int, trans: bool):
            trans_val = -1 if val == -2 else -2
            return trans_val if trans else val

        A_shape = [M, K]
        B_shape = [K, N]
        real_A_shape = [A_shape[trans_no_trans(-2, transA)], A_shape[trans_no_trans(-1, transA)]]
        real_B_shape = [B_shape[trans_no_trans(-2, transB)], B_shape[trans_no_trans(-1, transB)]]
        real_C_shape = [N]
        real_Y_shape = [M, N]

        initializers = []
        if use_constant_weights:
            b_tensor = helper.make_tensor(
                "B", TensorProto.FLOAT, real_B_shape, np.random.random(real_B_shape).astype(dtype=np.float32).flatten())
            initializers.append(b_tensor)

            inputs = [
                helper.make_tensor_value_info('A', TensorProto.FLOAT, real_A_shape),

                helper.make_tensor_value_info('C', TensorProto.FLOAT, real_C_shape),
            ]
        else:
            inputs = [
                helper.make_tensor_value_info('A', TensorProto.FLOAT, real_A_shape),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, real_B_shape),
                helper.make_tensor_value_info('C', TensorProto.FLOAT, real_C_shape),
            ]

        graph = helper.make_graph(
            [  # nodes
                helper.make_node("Gemm", ["A", "B", "C"], [
                                 "Y"], f"Gemm{suffix}",
                                 alpha=alpha, beta=beta, transA=transA,
                                 transB=transB),
            ],
            f"gemm{suffix}",  # name
            inputs,
            [  # outputs
                helper.make_tensor_value_info('Y', TensorProto.FLOAT, real_Y_shape),
            ],
            initializers
            )
        model = helper.make_model(graph)
        model_path = os.path.join(model_dir, expected_name)
        onnx.save(model, model_path)
        return get_name(model_path)

    if overwrite:
        return make_model()

    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()

def _emit_hat_package_for_model(model, package_name, target, output_dir, large_model=True):
    from accera import onnx_emitter, Target

    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)

    if target == "pi4":
        target_device = Target("Raspberry Pi 4B", category=Target.Category.CPU)
    elif target == "pi3":
        target_device = Target("Raspberry Pi 3B", category=Target.Category.CPU)
    elif target == "host":
         target_device = Target.HOST

    inferred_model, optimized_lib = onnx_emitter.emit_package_for_model(model, str(pathlib.Path(output_dir).absolute()), target=target_device, large_model=large_model)

    return inferred_model, optimized_lib

def emit_hat_package_for_target_per_node(target, node_type, output_dir, model_dir):

    for batch, layout, alpha in size_list_per_node_test:
        M, N, K = layout
        model = make_matmul_model(M=M, N=N, K=K, overwrite=True, model_dir=model_dir) if node_type == 'matmul' else make_gemm_model(M=M, N=N, K=K, overwrite=True) if node_type == 'gemm' else None
        model_name = pathlib.Path(model).name

        full_output_dir = f"{output_dir}/matmul_{M}_{N}_{K}.onnx" if node_type == 'matmul' else f"{output_dir}/{model_name}" if node_type == 'gemm' else None

        package_name = f"matmul_{M}_{N}_{K}" if node_type == 'matmul' else model_name if node_type == 'gemm' else None

        model_name, emitted_lib = _emit_hat_package_for_model(
            model, package_name=package_name, target=target, output_dir=full_output_dir)

        print('\033[92m' + f"\n\nSuccessfully done!\nThe following libs have been emitted for the '{target}' target for model '{model}' under this directory '{full_output_dir}':")
        print(f"\n{emitted_lib}" + '\033[0m')

def emit_hat_package_for_target(target, output_dir, model_dir):

    for base_name in model_base_name_list:
        for batch, seq in model_batch_sep_list:

            def make_optimized_gpt_model(base_name, batch, seq):
                opt_suffix = "_opt"
                specifier = f"_b{batch}_s{seq}"
                expected_name = f"{base_name}{specifier}{opt_suffix}.onnx"
                try:
                    return get_name(expected_name)
                except FileNotFoundError:
                    pass

                gpt = get_name(f"{model_dir}/{base_name}/{base_name}.onnx")
                print("Loading ", gpt)
                model = onnx.load(gpt)

                model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch
                model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = seq

                temp_name = f"{base_name}{specifier}.onnx"
                onnx.external_data_helper.convert_model_to_external_data(model, location=f"{base_name}.weights")
                onnx.save_model(model, temp_name)

                sess_option = onnxrt.SessionOptions()
                sess_option.intra_op_num_threads = 1
                sess_option.inter_op_num_threads = 1
                sess_option.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
                sess_option.optimized_model_filepath = expected_name
                sess_option.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
                _ = onnxrt.InferenceSession(temp_name, sess_option, providers=['CPUExecutionProvider'])

                return get_name(sess_option.optimized_model_filepath)

            from onnxruntime.transformers import shape_infer_helper
            global optimized_onnx_model
            optimized_onnx_model = make_optimized_gpt_model(base_name, batch, seq)

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

            os.environ["OMP_NUM_THREADS"] = "1"
            model_name = pathlib.Path(optimized_onnx_model).name
            out_dir = pathlib.Path(output_dir) / model_name
            model_name, emitted_lib = _emit_hat_package_for_model(
                optimized_onnx_model, package_name=model_name, target=target, output_dir=out_dir)

            print('\033[92m' + f"\n\nSuccessfully done!\nThe following libs have been emitted for the '{target}' target under this directory '{out_dir}':")
            print(f"\n{emitted_lib}" + '\033[0m')


def main(args=[]):

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', default='host', choices=['pi3', 'host'], help='by default target is the host machine')
    parser.add_argument('-o', '--output_dir', default='hat_packages', help='by default output dir name is "hat_packages"')
    parser.add_argument('-pn', '--per_node', default=False, help='is the test per node test?')
    parser.add_argument('-nt', '--node_type', default='', choices=['gemm', 'matmul'], help='the type of the node?')
    parser.add_argument('-md', '--model_dir', default='testdata', help='the directory of the model?')
    args = parser.parse_args(args)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.per_node == False:
        emit_hat_package_for_target(args.target, args.output_dir, model_dir)
    else:
        emit_hat_package_for_target_per_node(args.target, args.node_type, args.output_dir, model_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
