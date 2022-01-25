#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Requires: Python 3.7+
#
# Utility to parse and validate a HAT package for use with the ONNX Runtime
####################################################################################################

from dataclasses import dataclass, field
from hatlib import HATPackage, Function
from pathlib import Path

@dataclass
class ONNXHATFunction:
    name: str = ""
    include_file: Path = field(default_factory=Path)
    link_file: Path = field(default_factory=Path)
    function: Function = field(default_factory=Function)
    onnx: dict = field(default_factory=dict)


class ONNXHATPackage(HATPackage):
    AuxTableName = "onnx"
    NodeNameKey = "node_name"
    NodeTypeKey = "node_type"
    NodeDomainKey = "node_domain"
    NodeArgsKey = "node_args"
    NodeArgShapesKey = "node_arg_shapes"
    NodePackingFunctionsKey = "node_packing_functions"
    RequiredAuxKeys = [NodeNameKey, NodeTypeKey, NodeDomainKey, NodeArgsKey,
                       NodeArgShapesKey]

    def __init__(self, dirpath):
        super().__init__(dirpath)

    def get_functions_for_target(self, os: str, arch: str, required_extensions: list = []):
        def has_onnx_auxiliary_table(function):
            return function.auxiliary and self.AuxTableName in function.auxiliary

        # Get functions for the target with an ONNX auxiliary table
        all_functions_for_target = super().get_functions_for_target(os=os, arch=arch, required_extensions=required_extensions)

        # TODO: This is broken until pack functions are added to the HAT file
        # filtered_hat_funcs = list(filter(has_onnx_auxiliary_table, all_functions_for_target))
        filtered_hat_funcs = all_functions_for_target
        onnx_funcs = []

        for func in filtered_hat_funcs:
            onnx_aux_table = func.auxiliary[self.AuxTableName] if func.auxiliary else {}
            # for key_name in self.RequiredAuxKeys:
            #     if key_name not in onnx_aux_table:
            #         raise ValueError(
            #             f"ONNX HAT file {func.hat_file.name} function {func.name} has ONNX auxiliary table without a required '{key_name}' key")
            fn_name = func.name
            fn_include_path = Path(func.hat_file.path)
            fn_link_path = Path(func.link_target)
            onnx_funcs.append(ONNXHATFunction(name=fn_name,
                                              include_file=fn_include_path,
                                              link_file=fn_link_path,
                                              function=func,
                                              onnx=onnx_aux_table))

        return onnx_funcs

    @staticmethod
    def from_hat_package(pkg: HATPackage):
        return ONNXHATPackage(pkg.path)
