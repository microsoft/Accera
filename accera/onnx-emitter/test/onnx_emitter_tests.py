#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import os
import numpy as np
import pathlib
import unittest
import sys

# onnx is not automatically present on an accera system
# fail gracefully if the dependencies are not installed
try:
    import onnx
except ImportError:
    print("""[WARNING] onnx not installed, skipping tests.
You can fix this by running pip install -r accera/onnx-emitter/test/requirements.txt""")
    exit(0)

DEV_MODE = False
if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    DEV_MODE = True
    sys.path.insert(1, os.getcwd())

PACKAGE_DIR = pathlib.Path("hat_packages") / "onnx_emitter_tests"

from accera import onnx_emitter
from accera.test import verifiers


def get_name(name):
    # cf. onnxruntime/test/python/helper/py
    if os.path.exists(name):
        return name
    rel = pathlib.Path("testdata") / name
    if os.path.exists(rel):
        return str(rel)
    this = pathlib.Path(__file__).parent
    data = this / ".." / "testdata"
    res = data / name
    if os.path.exists(res):
        return str(res)
    raise FileNotFoundError(f"Unable to find '{name}' or '{str(rel)}' or '{str(res)}'")


def make_fused_matmul_model(M, N, K, batch=[], alpha=1.0, transA=0, transB=0, filename=None, overwrite=False):
    specifier = '_'.join([str(x) for x in ['', M, N, K, alpha, transA, transB]])
    expected_name = filename or f'fused_matmul_{specifier}.onnx'

    def make_model():
        from onnx import helper, TensorProto

        def trans_no_trans(val: int, trans: bool):
            trans_val = -1 if val == -2 else -2
            return trans_val if trans else val

        A_shape = [M, K]
        B_shape = [K, N]
        real_A_shape = [A_shape[trans_no_trans(-2, transA)], A_shape[trans_no_trans(-1, transA)]]
        real_B_shape = [B_shape[trans_no_trans(-2, transB)], B_shape[trans_no_trans(-1, transB)]]
        graph = helper.make_graph(
            [    # nodes
                helper.make_node("FusedMatMul", ["A", "B"], ["C"],
                                 f"MatMul_{specifier}",
                                 domain="com.microsoft",
                                 alpha=alpha,
                                 transA=transA,
                                 transB=transB),
            ],
            f"fused_matmul_{specifier}",    # name
            [    # inputs
                helper.make_tensor_value_info('A', TensorProto.FLOAT, batch + real_A_shape),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, batch + real_B_shape),
            ],
            [    # outputs
                helper.make_tensor_value_info('C', TensorProto.FLOAT, batch + [M, N]),
            ],
            [    # initializers
            ])
        model = helper.make_model(graph)
        onnx.save(model, 'testdata/' + expected_name)
        return get_name(expected_name)

    if overwrite:
        return make_model()
    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()


def make_matmul_model(M, N, K, batch=[], filename=None, overwrite=False, use_constant_weights=False):
    specifier = 'x'.join(map(str, batch)) + '_'.join(map(str, ['', M, N, K]))
    expected_name = filename or f'matmul_{specifier}.onnx'

    def make_model():
        from onnx import helper, TensorProto
        initializers = []
        if use_constant_weights:
            b_tensor = helper.make_tensor("B", TensorProto.FLOAT, batch + [K, N],
                                          np.random.random(batch + [K, N]).astype(dtype=np.float32).flatten())
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
            [    # nodes
                helper.make_node("MatMul", ["A", "B"], ["C"], f"MatMul_{specifier}"),
            ],
            f"matmul_{specifier}",    # name
            inputs,
            [    # outputs
                helper.make_tensor_value_info('C', TensorProto.FLOAT, batch + [M, N]),
            ],
            initializers)
        model = helper.make_model(graph)
        onnx.save(model, 'testdata/' + expected_name)
        return get_name(expected_name)

    if overwrite:
        return make_model()
    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()


def make_gemm_model(M,
                    N,
                    K,
                    alpha=1.0,
                    beta=1.0,
                    transA=0,
                    transB=0,
                    filename=None,
                    overwrite=False,
                    use_constant_weights=True):
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
            b_tensor = helper.make_tensor("B", TensorProto.FLOAT, real_B_shape,
                                          np.random.random(real_B_shape).astype(dtype=np.float32).flatten())
            initializers.append(b_tensor)
            # c_tensor = helper.make_tensor(
            #     "C", TensorProto.FLOAT, real_C_shape, np.random.random(real_C_shape).astype(dtype=np.float32).flatten())
            # initializers.append(c_tensor)
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
            [    # nodes
                helper.make_node("Gemm", ["A", "B", "C"], ["Y"],
                                 f"Gemm{suffix}",
                                 alpha=alpha,
                                 beta=beta,
                                 transA=transA,
                                 transB=transB),
            ],
            f"gemm{suffix}",    # name
            inputs,
            [    # outputs
                helper.make_tensor_value_info('Y', TensorProto.FLOAT, real_Y_shape),
            ],
            initializers)
        model = helper.make_model(graph)
        onnx.save(model, 'testdata/' + expected_name)
        return get_name(expected_name)

    if overwrite:
        return make_model()
    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()


class ONNXEmitterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs("testdata", exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if not DEV_MODE:
            import shutil
            shutil.rmtree("testdata", ignore_errors=True)
            shutil.rmtree(PACKAGE_DIR, ignore_errors=True)

    def test_fused_matmul_model(self) -> None:
        M = 128
        N = 64
        K = 128
        batch = [1, 20]

        model = onnx.load(make_fused_matmul_model(M=M, N=N, K=K, batch=batch, alpha=1.0, overwrite=True))

        output_dir = str((PACKAGE_DIR / "fused_matmul").absolute())
        with verifiers.VerifyPackage(self, model.graph.name, output_dir):
            onnx_emitter.emit_package_for_model(model, output_dir)

    def test_matmul_model(self) -> None:
        M = 128
        N = 64
        K = 128
        batch = [1, 20]
        model = onnx.load(make_matmul_model(M=M, N=N, K=K, batch=batch, overwrite=True, use_constant_weights=False))

        output_dir = str((PACKAGE_DIR / "matmul").absolute())
        with verifiers.VerifyPackage(self, model.graph.name, output_dir):
            onnx_emitter.emit_package_for_model(model, output_dir)

    def test_gemm_model(self) -> None:
        M = 128
        N = 64
        K = 128
        alpha = 1.0
        beta = 1.0
        transA = 0
        transB = 0
        model = onnx.load(
            make_gemm_model(M=M,
                            N=N,
                            K=K,
                            alpha=alpha,
                            beta=beta,
                            transA=transA,
                            transB=transB,
                            overwrite=True,
                            use_constant_weights=True))

        output_dir = str((PACKAGE_DIR / "gemm").absolute())
        with verifiers.VerifyPackage(self, model.graph.name, output_dir):
            onnx_emitter.emit_package_for_model(model, output_dir)


if __name__ == '__main__':
    unittest.main(verbosity=10)
