#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import csv
from io import StringIO
from itertools import islice
import os
from typing import Any, Dict, List
import argparse
import sys
import subprocess

from accera_gemm import benchmark_gemm
from gemm_opts import GemmOpts

CONFIG_HEADERS = ["type", "m", "n", "k", "transA", "transB", "alpha", "beta", "lda", "ldb", "ldc"]
RESULTS_HEADERS = CONFIG_HEADERS + ['time_ms', 'gflops']


def benchmark_gemm_shapes(data: List[GemmOpts], target: str, output_prefix: str, ext_benchmarker: str = None):

    ext_benchmarker_output = ''
    if ext_benchmarker:
        proc = subprocess.run([ext_benchmarker, '--headers'], capture_output=True, text=True, check=True)
        ext_benchmarker_headers = proc.stdout.strip().split(',')
        if ext_benchmarker_headers != RESULTS_HEADERS:
            print(f"Unexpected headers from external benchmarker tool: [{', '.join(ext_benchmarker_headers)}]")
            exit(1)

        ext_benchmarker_output += proc.stdout

    def exec_ext_benchmarker(gemm: GemmOpts):
        if ext_benchmarker:
            proc = subprocess.run([ext_benchmarker] + list(
                map(
                    str, [
                        gemm.type, gemm.m, gemm.n, gemm.k,
                        '1' if gemm.transA else '0',
                        '1' if gemm.transB else '0', gemm.alpha, gemm.beta, gemm.lda, gemm.ldb, gemm.ldc
                    ]
                )
            ),
                                  capture_output=True,
                                  text=True,
                                  check=True)
            return proc.stdout
        else:
            return ''

    accera_results:Dict[GemmOpts,List[Dict[str,Any]]] = {}
    for gemm in data:
        results = benchmark_gemm(gemm, target, os.path.split(output_prefix)[0] or '.')
        accera_results[gemm] = results

        ext_benchmarker_output += exec_ext_benchmarker(gemm)

    with open(f'{output_prefix}_accera.csv', 'w', newline='') as accera_results_file:
        accera_results_values = list(accera_results.values())
        accera_headers = list(set(accera_results_values[0][0].keys()) - set(RESULTS_HEADERS))
        writer = csv.DictWriter(accera_results_file, RESULTS_HEADERS + accera_headers)
        writer.writeheader()
        writer.writerows([line for result in accera_results_values for line in result])

    if ext_benchmarker:
        # rocblas_gemm -> rocblas, cublas_gemm -> cublas
        ext_benchmarker_name = os.path.splitext(os.path.basename(ext_benchmarker))[0].split('_')[0]
        
        ext_benchmarker_results_filename = f'{output_prefix}_{ext_benchmarker_name}.csv'
        with open(ext_benchmarker_results_filename, 'w', newline='') as ext_benchmarker_results_file:
            ext_benchmarker_results_file.write(ext_benchmarker_output)

        with open(ext_benchmarker_results_filename) as ext_benchmarker_results_file:
            ext_benchmarker_reader = csv.DictReader(ext_benchmarker_results_file, ext_benchmarker_headers)
            ext_benchmarker_results = list(islice(ext_benchmarker_reader, 1, None))

        comparison_file = f'{output_prefix}_comparison.csv'
        comparison_headers = CONFIG_HEADERS + [ext_benchmarker_name + '_time_ms', ext_benchmarker_name + '_gflops', 'accera_time_ms', 'accera_gflops']
        with open(comparison_file, 'w', newline='') as comparison_results_file:
            writer = csv.DictWriter(comparison_results_file, comparison_headers)
            writer.writeheader()

            for ext_benchmarker_result in ext_benchmarker_results:
                gemm = GemmOpts(**{k: ext_benchmarker_result[k]
                                   for k in GemmOpts.__dataclass_fields__.keys()})

                accera_result = accera_results.get(gemm)
                assert accera_result

                accera_result = min(accera_result, key=lambda r: r['time_ms'])

                comparison_result = ext_benchmarker_result.copy()
                del comparison_result['time_ms']
                del comparison_result['gflops']
                comparison_result[ext_benchmarker_name + '_time_ms'] = ext_benchmarker_result['time_ms']
                comparison_result[ext_benchmarker_name + '_gflops'] = ext_benchmarker_result['gflops']
                comparison_result['accera_time_ms'] = accera_result['time_ms']
                comparison_result['accera_gflops'] = accera_result['gflops']

                writer.writerow(comparison_result)


def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The input config file (csv)', required=False)
    parser.add_argument('-z', '--string', help='input config string (csv, semi-colon per row)', required=False)
    parser.add_argument('-t', '--target', help='The target the emitter is emitting HAT package for')
    parser.add_argument('-o', '--output', help='The output prefix', default="results")
    parser.add_argument('-r', '--ext_benchmarker', help="The path to the external benchmarker tool (e.g. ../../build/temp.linux-x86_64-3.8/tools/benchmarkers/rocblas/rocblas_gemm')", required=False)

    args = parser.parse_args(args)

    if args.string and args.input:
        raise RuntimeError("input and string options are mutually exclusive")

    f = None
    if args.string:
        args.string = ','.join(CONFIG_HEADERS) + '\n' + '\n'.join(args.string.split(';'))
        f = StringIO(args.string)

    try:
        if f is None:
            f = open(args.input)

        reader = csv.DictReader(f, CONFIG_HEADERS)
        gemm_opts = [GemmOpts(**data) for data in islice(reader, 1, None)]

    finally:
        f.close()

    benchmark_gemm_shapes(gemm_opts, args.target, args.output, args.ext_benchmarker)


if __name__ == "__main__":
    main(sys.argv[1:])
