#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import csv
from io import StringIO
from itertools import islice
import os
from typing import List
import argparse
import sys
import subprocess
import accera_gemm
import cosmosdb
import re
import gemm_opts
import git
import shutil
from datetime import datetime
import torch

def get_current_commit_id():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

def get_current_commit_datetime():
    repo = git.Repo(search_parent_directories=True)
    return str(repo.head.object.committed_datetime)

def get_current_branch():
    repo = git.Repo(search_parent_directories=True)
    return repo.active_branch.name

def exec_ext_benchmarker(gpu_id: int, gemm: gemm_opts.GemmOpts, datatype, benchmark_tool):
    proc = subprocess.run([benchmark_tool] + list(
        map(str, [datatype, gemm.m, gemm.n, gemm.k, int(gemm.transA), int(gemm.transB), gemm.alpha, gemm.beta, gemm.lda, gemm.ldb, gemm.ldc, gpu_id])),
        capture_output=True,
        text=True)
    proc.check_returncode()
    return proc.stdout

def run_pytorch_matmul(gemm: gemm_opts.GemmOpts, dtype, gpu_id: int):
    cuda = torch.device('cuda')
    type = torch.float32 if dtype == "s" else torch.float16
    with torch.cuda.device(gpu_id):
        a = torch.randn(gemm.m, gemm.k, dtype=type, device=cuda)
        b = torch.randn(gemm.k, gemm.n, dtype=type, device=cuda)
        c = torch.randn(gemm.m, gemm.n, dtype=type, device=cuda)
        if gemm.transA:
            a = a.t().contiguous().t()

        if gemm.transB:
            b = b.t().contiguous().t()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        c = (gemm.alpha * torch.matmul(a, b)) + (gemm.beta * c)

        # Under timer
        start.record()
        c = (gemm.alpha * torch.matmul(a, b)) + (gemm.beta * c)
        end.record()

        torch.cuda.synchronize()
        time_taken_ms = start.elapsed_time(end)
        throughput_tflops = 2 * gemm.m * gemm.n * gemm.k * 1.0e-9 / time_taken_ms
    return time_taken_ms, throughput_tflops, f"Total time taken: {time_taken_ms} ms, {throughput_tflops} TFlops"

def benchmark_gemm_shapes(data: List[gemm_opts.GemmOpts], dtype, batch_size: int, git_branch: str, target_name: str, output_prefix: str, category: str, rocblas: str, composable_kernel: str, cublas: str, cutlass: str, pytorch: str, available_gpus, container_name, verbose, check, compiler_ver, deviceProperties):
    result_dir = os.path.split(output_prefix)[0] or '.'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    commit_id = get_current_commit_id()
    commit_datetime = get_current_commit_datetime()
    commit_branch = git_branch.replace("refs/heads/", "") if git_branch else get_current_branch()
    result_rows = []

    if rocblas or cublas or pytorch:
        benchmark_tool_name = 'pytorch' if pytorch else ('rocblas' if rocblas else 'cublas')
        benchmark_tool = rocblas if rocblas else cublas
        print(f'Running {benchmark_tool_name} baseline benchmarks')

        for gemm in data:
            best_time = 0.0
            best_throughput = 0.0
            best_gpu = 0
            prog_out = ''
            for gpu_id in range(len(available_gpus)):
                if available_gpus[gpu_id]:
                    print(f"Processing input: {gemm} on GPU {gpu_id}")
                    if pytorch:
                        time_taken_ms, throughput, output = run_pytorch_matmul(gemm, dtype, gpu_id)
                    else:
                        output = exec_ext_benchmarker(gpu_id, gemm, dtype, benchmark_tool)
                        tokens = output.split(",")
                        throughput = float(tokens[len(tokens) - 1].rstrip())
                        time_taken_ms = tokens[len(tokens) - 2]

                    print(output)
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_time = time_taken_ms
                        best_gpu = gpu_id
                        prog_out = output
            else:
                benchmarkResult = accera_gemm.BenchmarkResult(opts=gemm, dtype=dtype, gpu_id=best_gpu, commit_id=commit_id, commit_datetime=commit_datetime, commit_branch=commit_branch, target_name=target_name, deviceProperties=deviceProperties[best_gpu])
                benchmarkResult.compiler_version = compiler_ver
                benchmarkResult.target_rt = pytorch if pytorch else ('ROCM' if rocblas else 'CUDA')
                benchmarkResult.compilable = True
                benchmarkResult.executable = True
                benchmarkResult.time_ms = best_time
                benchmarkResult.category = category
                benchmarkResult.TFlops = str(best_throughput)
                benchmarkResult.prog_out = prog_out
                result_rows.append(benchmarkResult.get_result_row())
        else:
            cosmosdb.upsert_benchmark_results(result_rows, benchmark_tool_name, verbose)
            cosmosdb.show_benchmark_summary(benchmark_tool_name)
    elif composable_kernel:
        print('Running composable_kernel baseline benchmarks')
        for gemm in data:
            print(f"Processing input: {gemm} on GPU 0")
            datatype = '0' if dtype == 's' else '1'
            if gemm.transA and gemm.transB:
                layout = '3'
            elif gemm.transA and not gemm.transB:
                layout = '2'
            elif not gemm.transA and gemm.transB:
                layout = '1'
            else:
                layout = '0'
            lda = gemm.m if not gemm.transA else gemm.k
            ldb = gemm.k if not gemm.transB else gemm.n
            ldc = gemm.m
            benchmarkResult = accera_gemm.BenchmarkResult(opts=gemm, dtype=dtype, gpu_id=0, commit_id=commit_id, commit_datetime=commit_datetime, commit_branch=commit_branch, target_name=target_name, deviceProperties=deviceProperties[0])
            #                                          op     datatype  layout  verify  init  log  repeat  M___         N___         K___         StrideA   StrideB   StrideC
            proc = subprocess.run([composable_kernel, 'gemm', datatype, layout, '1',    '0',  '0', '2',    str(gemm.m), str(gemm.n), str(gemm.k), str(lda), str(ldb), str(ldc)], capture_output=True, text=True)
            print(proc.stdout)
            benchmarkResult.compiler_version = compiler_ver
            benchmarkResult.target_rt = 'ROCM'
            benchmarkResult.compilable = True
            benchmarkResult.executable = True
            benchmarkResult.category = category
            benchmarkResult.prog_out = proc.stdout
            matches = re.search('Best Perf.*: (.+) ms, (.+) TFlops', proc.stdout)
            if matches:
                benchmarkResult.time_ms = matches.group(1)
                benchmarkResult.TFlops = matches.group(2)
                print(matches.group(0))
                result_rows.append(benchmarkResult.get_result_row())
            else:
                raise Exception("Did not find a match for the result.")
        else:
            cosmosdb.upsert_benchmark_results(result_rows, "composable_kernel", verbose)
            cosmosdb.show_benchmark_summary("composable_kernel")
    elif cutlass:
        print('Running CUTLASS baseline benchmarks')
        for gemm in data:
            print(f"Processing input: {gemm} on GPU 0")
            datatype = 'f32' if dtype == 's' else 'f16'
            layoutA = 't' if gemm.transA else 'n'
            layoutB = 't' if gemm.transB else 'n'
            result_filename = f'{output_prefix}_cutlass'
            benchmarkResult = accera_gemm.BenchmarkResult(opts=gemm, dtype=dtype, gpu_id=0, commit_id=commit_id, commit_datetime=commit_datetime, commit_branch=commit_branch, target_name=target_name, deviceProperties=deviceProperties[0])
            proc = subprocess.run([cutlass, '--operation=Gemm', f'--A={datatype}:{layoutA}', f'--B={datatype}:{layoutB}', f'--C={datatype}:*', f'--m={gemm.m}', f'--n={gemm.n}', f'--k={gemm.k}',
                                   f'--alpha={gemm.alpha}', f'--beta={gemm.beta}', '--op_class=tensorop', f'--output={result_filename}'], capture_output=True, text=True)
            print(proc.stdout)
            benchmarkResult.compiler_version = compiler_ver
            benchmarkResult.target_rt = 'CUDA'
            benchmarkResult.compilable = True
            benchmarkResult.executable = True
            benchmarkResult.category = category
            benchmarkResult.prog_out = proc.stdout

            # Read the cutlass result file and find max throughput
            result_filename += '.gemm.csv'
            result_file = open(result_filename, 'r')
            lines = result_file.readlines()
            result_file.close()
            maxThroughput = 0.0
            min_time = 0.0
            for i in range(1, len(lines)): # skip the first line of headers
                tokens = lines[i].split(',')
                maxThroughput = max(maxThroughput, float(tokens[len(tokens) - 1]) / 1000) # last item is throughput
                min_time = min(min_time, float(tokens[len(tokens) - 3]))
            benchmarkResult.time_ms = min_time
            benchmarkResult.TFlops = maxThroughput
            result_rows.append(benchmarkResult.get_result_row())
            print(f'Max throughput: {maxThroughput} TFlops')
        else:
            cosmosdb.upsert_benchmark_results(result_rows, "cutlass", verbose)
            cosmosdb.show_benchmark_summary("cutlass")
    else:
        for gemm in data:
            print(f"\nProcessing input: {gemm}")
            accera_gemm.benchmark_gemm(gemm, dtype, batch_size, output_prefix, category, available_gpus, container_name, verbose, compiler_ver, commit_id, commit_datetime, commit_branch, target_name, check, deviceProperties)
        # else:
        #     if container_name:
        #         cosmosdb.show_benchmark_summary(container_name)

def prepare_system_for_benchmark(target, available_gpus):
    deviceProperties = []
    compiler_ver = ''
    if target == 'AMD MI100':
        # fix shader clock speeds
        proc = subprocess.run(["rocm-smi", '--setsclk', '15'], capture_output=True, text=True)
        print(proc.stdout)

        proc = subprocess.run(["rocm-smi", '-g'], capture_output=True, text=True)
        print(proc.stdout)

        proc = subprocess.run(["hipcc", '--version'], capture_output=True, text=True)
        compiler_ver = proc.stdout
        print(compiler_ver)

        for deviceId in range(len(available_gpus)):
            proc = subprocess.run(["rocm-smi", '-a', '-d', str(deviceId), '--json'], capture_output=True, text=True)
            deviceProperties.append(proc.stdout)
    elif target == 'NVidia RTX A6000':
        # https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
        # enable persistence mode
        proc = subprocess.run(["nvidia-smi", '--persistence-mode=1'], capture_output=True, text=True)
        print(proc.stdout)

        # Run in exclusive process mode (1 process per GPU)
        proc = subprocess.run(["nvidia-smi", '--compute-mode=3'], capture_output=True, text=True)
        print(proc.stdout)

        # Set application clocks
        proc = subprocess.run(["nvidia-smi", '--applications-clocks=8001,2100'], capture_output=True, text=True)
        print(proc.stdout)

        proc = subprocess.run(["nvcc", '--version'], capture_output=True, text=True)
        compiler_ver = proc.stdout
        print(compiler_ver)

        for deviceId in range(len(available_gpus)):
            proc = subprocess.run(["nvidia-smi", '-q', '-i', str(deviceId)], capture_output=True, text=True)
            deviceProperties.append(proc.stdout)

    return deviceProperties, compiler_ver

def get_gemm_input_from_stream(stream):
    reader = csv.DictReader(stream, gemm_opts.CONFIG_HEADERS)
    return [gemm_opts.GemmOpts(**data) for data in islice(reader, 1, None)]

def get_gemm_input_from_file(file):
    result = []
    with open(file) as f:
        result = get_gemm_input_from_stream(f)
    return result

def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--devices', help='The devices to use for the benchmark', required=False, default="0,1,2,3")
    parser.add_argument('-i', '--input', nargs="+", help='Comma-separated list of input config files (csv)', required=False)
    parser.add_argument('-y', '--type', help='The data type for the input set, h or fp16, s for fp32', required=False)
    parser.add_argument('-s', '--batch_size', type=int, help='No. of iterations for the benchmark run', required=False, default=2)
    parser.add_argument('-b', '--branch', help='The git branch to use to tag the results to', required=False)
    parser.add_argument('-z', '--string', help='input config string (csv, semi-colon per row)', required=False)
    parser.add_argument('-t', '--target', help='The target the emitter is emitting HAT package for')
    parser.add_argument('-o', '--output', help='The output prefix', default="results")
    parser.add_argument('-ct', '--category', help='The category of gemm inputs (used for classification), e.g. bert, cube etc.', default="")
    parser.add_argument('-roc', '--rocblas', help="The path to the rocblas_gemm tool", required=False)
    parser.add_argument('-cu', '--cublas', help="The path to the cublas_gemm tool", required=False)
    parser.add_argument('-pt', '--pytorch', help="The platform to use for PyTorch, e.g. ROCM or CUDA", required=False)
    parser.add_argument('-ck', '--composable_kernel', help="The path to the composable-kernel tool", required=False)
    parser.add_argument('-cl', '--cutlass', help="The path to the cutlass tool", required=False)
    parser.add_argument('-u', '--upload', help="Specify the CosmosDB container name to upload the results to", required=False)
    parser.add_argument('-v', '--verbose', action="store_true", help="Enable verbose logging", required=False)
    parser.add_argument('-c', '--check', action="store_true", help="Verify correctness of the generated kernels", required=False)
    parser.add_argument('-j', '--no_janitor', action="store_false", help="Don't cleanup the output dir after running benchmark", required=False)

    args = parser.parse_args(args)

    if args.string and args.input:
        raise RuntimeError("input and string options are mutually exclusive")

    if args.rocblas and args.composable_kernel:
        raise RuntimeError("rocblas and composable_kernel options are mutually exclusive")

    if not args.type:
        raise RuntimeError("No type argument passed")

    f = None
    if args.string:
        args.string = ','.join(gemm_opts.CONFIG_HEADERS) + '\n' + '\n'.join(args.string.split(';'))
        f = StringIO(args.string)

    gemm_inputs = []
    if f is None:
        for file in args.input:
            gemm_inputs += get_gemm_input_from_file(file)
    else:
        gemm_inputs = get_gemm_input_from_stream(f)

    available_gpus = []
    devices = args.devices.split(",")
    for dev in devices:
        dev_id = int(dev)
        while len(available_gpus) <= dev_id:
            available_gpus.append(False)
        else:
            available_gpus[dev_id] = True

    print(f"Running on devices: {args.devices}")

    print("Clean the output directory...")
    output_dir = os.path.split(args.output)[0] or '.'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    print(datetime.now())

    deviceProperties, compiler_ver = prepare_system_for_benchmark(args.target, available_gpus)

    benchmark_gemm_shapes(gemm_inputs, args.type, args.batch_size, args.branch, args.target, args.output, args.category,
                            args.rocblas, args.composable_kernel, args.cublas, args.cutlass, args.pytorch,
                            available_gpus, args.upload, args.verbose, args.check, compiler_ver, deviceProperties)

    print("Cleaning up output directory after benchmark")
    if args.no_janitor: # This is the correct logic since this option is a store_false
        shutil.rmtree(output_dir)

    print(datetime.now())


if __name__ == "__main__":
    main(sys.argv[1:])
