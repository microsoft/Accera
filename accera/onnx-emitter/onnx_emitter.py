#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import sys
import argparse
import onnx
import os

from pathlib import Path

from accera import Package, Target, Array
from accera.hat import ONNXHATPackage
from accera.samples import MLAS, MLASOptions

from onnx import helper
from onnxruntime.tools.symbolic_shape_infer import (SymbolicShapeInference, get_shape_from_type_proto)


# TODO: When ORT has FusedMatMul shape inference upstream, remove this
# Should be replaced with
# model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
def _infer_shapes(model):
    # cf onnxruntime/tools/symbolic_shape_infer.py
    ssi = SymbolicShapeInference(int_max=2**31 - 1, auto_merge=True, guess_output_rank=False, verbose=0)

    def compute_matmul_shape(node, output_dtype=None, transA=False, transB=False):
        lhs_shape = ssi._get_shape(node, 0)
        rhs_shape = ssi._get_shape(node, 1)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        lhs_reduce_dim = 0
        rhs_reduce_dim = 0
        assert lhs_rank > 0 and rhs_rank > 0

        if lhs_rank == 1 and rhs_rank == 1:
            new_shape = []
        elif lhs_rank == 1:
            new_shape = rhs_shape[:-2]
            if transB:
                rhs_reduce_dim = -1
                new_shape += [rhs_shape[-2]]
            else:
                rhs_reduce_dim = -2
                new_shape += [rhs_shape[-1]]
        elif rhs_rank == 1:
            new_shape = lhs_shape[:-2]
            if transA:
                lhs_reduce_dim = -2
                new_shape += [lhs_shape[-1]]
            else:
                lhs_reduce_dim = -1
                new_shape += [lhs_shape[-2]]
        else:
            lhs_reduce_dim = -2 if transA else -1
            rhs_reduce_dim = -1 if transB else -2
            new_shape = ssi._broadcast_shapes(
                lhs_shape[:-2], rhs_shape[:-2]) + [lhs_shape[-1 if transA else -2], rhs_shape[-2 if transB else -1]]

        # merge reduce dim
        ssi._check_merged_dims([lhs_shape[lhs_reduce_dim], rhs_shape[rhs_reduce_dim]], allow_broadcast=False)

        if output_dtype is None:
            # infer output_dtype from input type when not specified
            output_dtype = ssi.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = ssi.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def infer_FusedMatMul(node):
        transA = get_attribute(node, 'transA', 0) == 1
        transB = get_attribute(node, 'transB', 0) == 1
        ssi._compute_matmul_shape(node, transA=transA, transB=transB)

    # patch SymbolicShapeInference
    ssi._compute_matmul_shape = compute_matmul_shape
    ssi._infer_FusedMatMul = infer_FusedMatMul

    ssi.dispatcher_.update({'FusedMatMul': ssi._infer_FusedMatMul})

    # cf onnxruntime/tools/symbolic_shape_infer.py
    all_shapes_inferred = False
    ssi._preprocess(model)
    while ssi.run_:
        all_shapes_inferred = ssi._infer_impl()
    ssi._update_output_from_vi()
    if not all_shapes_inferred:
        raise Exception("Incomplete symbolic shape inference")
    return ssi.out_mp_


def get_attribute(node, attr_name, default_value=None):
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value


def get_initializer(model, name):
    graph_ = model.graph
    result = [init for init in graph_.initializer if init.name == name]
    if result:
        return result[0]
    else:
        return None


def get_value_info(shape_inferred_model, name):
    graph_ = shape_inferred_model.graph
    found = [vi for vi in list(graph_.value_info) + list(graph_.input) if vi.name == name]
    if found:
        return found[0]
    return None


def get_shape(shape_inferred_model, name):
    graph_ = shape_inferred_model.graph

    found = [i for i in list(graph_.initializer) if i.name == name]
    if found:
        return found[0].dims

    found = [vi for vi in list(graph_.value_info) + list(graph_.input) if vi.name == name]
    if found:
        return get_shape_from_type_proto(found[0].type)

    return None


def load_model(model_file):
    return onnx.load_model(model_file)


def get_target(target_name):
    if target_name == 'pi4':
        return Target(Target.Model.RASPBERRY_PI_4B, category=Target.Category.CPU)
    elif target_name == 'pi3':
        return Target("Raspberry Pi 3B", category=Target.Category.CPU)
    else:
        return Target.HOST


def get_target_options(target):
    if "Raspberry Pi" in target.name:
        # TODO: Make use of the different attributes between the Pi devices
        return MLASOptions(KUnroll=2,
                           BCacheSizeThreshold=64**1,
                           NumRowsInKernel=2,
                           NumColumnsInKernelScaleFactor=4,
                           BMatrixTileSize=[128, 128])
    else:
        return MLASOptions()


def handle_gemm_node(node, model, package, target=Target.HOST):
    A_input_name = node.input[0]
    B_input_name = node.input[1]
    C_input_name = node.input[2]
    Y_output_name = node.output[0]

    A_shape = get_shape(model, A_input_name)
    B_shape = get_shape(model, B_input_name)
    C_shape = get_shape(model, C_input_name)
    Y_shape = get_shape(model, Y_output_name)

    alpha = get_attribute(node, 'alpha', 1.0)
    beta = get_attribute(node, 'beta', 1.0)
    transA = get_attribute(node, 'transA', 0) == 1
    transB = get_attribute(node, 'transB', 0) == 1

    print(f"[accera] handle_gemm_node called for \nA = [{', '.join(map(str, A_shape))}]"
          f"\nB = [{', '.join(map(str, B_shape))}]"
          f"\nC = [{', '.join(map(str, C_shape))}]"
          f"\nY = [{', '.join(map(str, Y_shape))}]"
          f"\nalpha = {alpha} beta = {beta} transA = {transA} transB = {transB}")

    if transA or transB:
        print("\n\ntransA and transB attributes not supported at this time")
        return {}

    # accera BUG: adding name to ctor throws
    # RuntimeError: EmitterContext is not set!
    A = Array(role=Role.INPUT, element_type=float, shape=A_shape)    # , name=A_input_name)
    B = Array(role=Role.INPUT, element_type=float, shape=B_shape)    # , name=B_input_name)
    C = Array(role=Role.INPUT, element_type=float, shape=C_shape)    # , name=node.input([2])
    Y = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=Y_shape)    # , name=Y_output_name)

    emitted_info = {}
    opts = get_target_options(target)
    B_init_data = get_initializer(model, B_input_name)
    if B_init_data:
        print(f"B Initializer detected for {node.name} for input {B_input_name}")
        opts = opts._replace(PackBFuncName=f"{node.name}_reshape_B",
                             PackBBufferSizeFuncName=f"{node.name}_reshape_B_size")
        emitted_info['node_packing_functions'] = {B_input_name: [opts.PackBFuncName, opts.PackBBufferSizeFuncName]}

    plan, args = MLAS(A,
                      B,
                      Y,
                      alpha=alpha,
                      transA=transA,
                      transB=transB,
                      beta=beta,
                      bias=C,
                      zero_C=True,
                      opts=opts,
                      target=target)
    assert (args == (A, B, C, Y))

    emitted_info.update({
        ONNXHATPackage.NodeNameKey: node.name,
        ONNXHATPackage.NodeTypeKey: node.op_type,
        ONNXHATPackage.NodeDomainKey: node.domain,
        ONNXHATPackage.NodeArgsKey: node.input[0:3] + [Y_output_name],

        # TODO: This should be a lot easier
        ONNXHATPackage.NodeArgShapesKey: [
            list(arg.shape) for arg in [A, B, C, Y]]
    })

    return package.add(plan,
                       args=[A, B, C, Y],
                       base_name=node.name,
                       auxiliary={ONNXHATPackage.AuxTableName: emitted_info})


def handle_matmul_node(node, model, package, target=Target.HOST):
    A_input_name = node.input[0]
    B_input_name = node.input[1]
    C_output_name = node.output[0]

    A_shape = get_shape(model, A_input_name)
    B_shape = get_shape(model, node.input[1])
    C_shape = get_shape(model, C_output_name)

    alpha = get_attribute(node, 'alpha', 1.0)
    transA = get_attribute(node, 'transA', 0) == 1
    transB = get_attribute(node, 'transB', 0) == 1

    print(f"[accera] handle_matmul_node called for \nA = [{', '.join(map(str, A_shape))}]"
          f"\nB = [{', '.join(map(str, B_shape))}]"
          f"\nC = [{', '.join(map(str, C_shape))}]"
          f"\nalpha = {alpha} transA = {transA} transB = {transB}")

    A = Array(role=Role.INPUT, element_type=float, shape=A_shape)    # , name=A_input_name)
    B = Array(role=Role.INPUT, element_type=float, shape=B_shape)    # , name=B_input_name)
    C = Array(role=Role.INPUT_OUTPUT, element_type=float, shape=C_shape)    # , name=C_output_name)

    emitted_info = {}

    B_init_data = get_initializer(model, B_input_name)
    opts = get_target_options(target)

    if B_init_data:
        print(f"B Initializer detected for {node.name} for input {B_input_name}")
        opts = opts._replace(PackBFuncName=f"{node.name}_reshape_B",
                             PackBBufferSizeFuncName=f"{node.name}_reshape_B_size")
        emitted_info['node_packing_functions'] = {B_input_name: [opts.PackBFuncName, opts.PackBBufferSizeFuncName]}

    emitted_info.update({
        ONNXHATPackage.NodeNameKey: node.name,
        ONNXHATPackage.NodeTypeKey: node.op_type,
        ONNXHATPackage.NodeDomainKey: node.domain,
        ONNXHATPackage.NodeArgsKey: [A_input_name, B_input_name, C_output_name],

        # TODO: This should be a lot easier
        ONNXHATPackage.NodeArgShapesKey: list(arg.shape for arg in [A, B, C])
    })

    plan, args = MLAS(A, B, C, alpha=alpha, transA=transA, transB=transB, zero_C=True, opts=opts, target=target)
    return package.add(plan, args, base_name=node.name, auxiliary={ONNXHATPackage.AuxTableName: emitted_info})


ONNX_NODE_HANDLERS = {
    'FusedMatMul': handle_matmul_node,
    'Gemm': handle_gemm_node,
    'MatMul': handle_matmul_node,
}


def emit_package_for_model(model,
                           output_dir,
                           large_model=False,
                           target=Target.HOST,
                           format=Package.Format.STATIC_LIBRARY | Package.Format.HAT_PACKAGE,
                           mode=Package.Mode.RELEASE):
    model = _infer_shapes(model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    inferred_model = str((output_path / "inferred.onnx").absolute())

    # To support big models like GPT2-large, we need to save weights as external data
    if large_model:
        onnx.external_data_helper.convert_model_to_external_data(model)
    onnx.save(model, inferred_model)

    # Create a package and add our function definition to it
    package = Package()

    for node in filter(lambda node: node.op_type in ONNX_NODE_HANDLERS, model.graph.node):
        ONNX_NODE_HANDLERS[node.op_type](node, model, package, target)

    # Build the HAT package
    return inferred_model, package.build(model.graph.name, format=format, mode=mode, output_dir=output_path)


def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='The input model file', default=None)
    parser.add_argument('-t', '--target', help='The target the emitter is emitting HAT package for',
                        default='host', choices=['pi3', 'host'])
    parser.add_argument(
        '-o', '--output', help='The output model file', default=None)
    args = parser.parse_args(args)

    model = load_model(args.input)
    output_dir = args.output or os.getcwd()

    emit_package_for_model(model, output_dir, target=get_target(args.target))


if __name__ == "__main__":
    main(sys.argv[1:])
