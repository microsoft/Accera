#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Requires: Python 3.7+
####################################################################################################

from scripts import ONNXHATPackage

import argparse
import os
import sys

def main(cl_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to HAT package directory to parse")
    parser.add_argument("-n", "--node_type", type=str, help="The ONNX node type to filter the HAT package for function implementations")
    parser.add_argument("-s", "--os", required=True, type=str, help="Operating System to require for built functions")
    parser.add_argument("-a", "--architecture", required=True, type=str, help="CPU Architecture to filter on")
    parser.add_argument("-e", "--extensions", type=str, nargs="*", default=[], help="Required architecture extensions to include")
    args = parser.parse_args(cl_args)

    onnx_hat_package = ONNXHATPackage(args.input)

    print(f"Evaluating directory {onnx_hat_package.path} as an ONNX HAT package")

    onnx_funcs = onnx_hat_package.get_functions_for_target(os=args.os, arch=args.architecture, required_extensions=args.extensions)
    if args.node_type:
        print(f"Searching for {args.node_type} functions...")

        def node_type_filter(onnx_func):
            return onnx_func.onnx["node_type"] == args.node_type
        filtered_funcs = list(filter(node_type_filter, onnx_funcs))

        print(f"Found {len(filtered_funcs)} functions with node_type = {args.node_type}:")
        for func in filtered_funcs:
            print(f"\t{func.name}, declared in {func.include_file}, link target {func.link_file}")

        matching_link_targets = set([func.link_file for func in filtered_funcs])
        print(f"link target files: {matching_link_targets}")
    else:
        print(f"Searching for all ONNX functions...")
        print(f"Found {len(onnx_funcs)} functions:")

        for func in onnx_funcs:
            print(f"\t{func.name}, op_type = {func.onnx['node_type']} declared in in {func.include_file}, link target {func.link_file}")

        all_link_targets = set([func.link_file for func in onnx_funcs])
        print(f"link target files: {all_link_targets}")

if __name__ == "__main__":
    main(cl_args=sys.argv[1:])
