#!/usr/bin/env python3

####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Requires: Python 3.7+
####################################################################################################

import argparse
import sys
import pandas as pd
import numpy as np
import traceback

import accera as acc
from accera.tuning import AutoBenchmark
from accera.hat import HATFile, HATPackage

def write_back_to_hat(hat_file_path, function_name, mean_time_secs):
    # Write back the runtime to the HAT file
    hat_file = HATFile.Deserialize(hat_file_path)
    hat_func = hat_file.function_map.get(function_name)
    hat_func.auxiliary["mean_duration_in_sec"] = mean_time_secs

    hat_file.Serialize(hat_file_path)

    # Workaround to remove extra empty lines
    with open(hat_file_path, "r") as f:
        lines = f.readlines()
        lines = [lines[i] for i in range(len(lines)) if not(lines[i] == "\n" \
                                    and i < len(lines)-1 and lines[i+1] == "\n")]
    with open(hat_file_path, "w") as f:
        f.writelines(lines)

def autobenchmark_package(package_directory, store_in_hat=False, batch_size=10, min_time_in_sec=10, input_sets_minimum_size_MB=50):
    benchmark = AutoBenchmark(package_directory)

    results = []
    functions = benchmark.hat_functions
    for hat_function in functions:
        function_name = hat_function.name
        print(f"\nBenchmarking function: {function_name}")
        if "Initialize" in function_name: # Skip init functions
            continue

        try:
            _, batch_timings = benchmark.run(function_name,
                                           warmup_iterations=batch_size,
                                           min_timing_iterations=batch_size,
                                           min_time_in_sec=min_time_in_sec,
                                           input_sets_minimum_size_MB=input_sets_minimum_size_MB
                                           )

            sorted_batch_means = np.array(sorted(batch_timings)) / batch_size
            num_batches = len(batch_timings)

            mean_of_means = sorted_batch_means.mean()
            median_of_means = sorted_batch_means[num_batches//2]
            mean_of_small_means = sorted_batch_means[0 : num_batches//2].mean()
            robust_mean_of_means = sorted_batch_means[num_batches//5 : -num_batches//5].mean()
            min_of_means = sorted_batch_means[0]

            if store_in_hat:
                # Write back the runtime to the HAT file
                hat_file_path = hat_function.hat_file.path
                write_back_to_hat(hat_file_path, function_name, mean_of_means)
            results.append({"function_name": function_name,
                            "mean": mean_of_means,
                            "median_of_means": median_of_means,
                            "mean_of_small_means": mean_of_small_means,
                            "robust_mean": robust_mean_of_means,
                            "min_of_means": min_of_means,
                            })
        except Exception as e:
            exc_type, exc_val, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr)
            print("\nException message: ", e)
            print(f"WARNING: Failed to run function {function_name}, skipping this benchmark.")
    return results

def main(argv):
    arg_parser = argparse.ArgumentParser(
        description="Benchmarks each function in a HAT package and estimates its duration.\n"
        "Example:\n"
        "    accera.benchmark_hat <path_to_HAT_package>\n")

    arg_parser.add_argument("path_to_hat_package_dir",
        help="Path to the HAT pakcage directory",
        default=None)
    arg_parser.add_argument("--store_in_hat",
        help="If set, will write the duration as meta-data back into the hat file",
        action='store_true')
    arg_parser.add_argument("--results_file",
        help="Full path where the results will be written",
        default="results.csv")
    arg_parser.add_argument("--batch_size",
        help="The number of function calls in each batch (at least one full batch is executed)",
        default=10)
    arg_parser.add_argument("--min_time_in_sec",
        help="Minimum number of seconds to run the benchmark for",
        default=30)
    arg_parser.add_argument("--input_sets_minimum_size_MB",
        help="Minimum size in MB of the input sets. Typically this is large enough to ensure eviction of the biggest cache on the target (e.g. L3 on an desktop CPU)",
        default=50)


    args = vars(arg_parser.parse_args(argv))

    results = autobenchmark_package(args["path_to_hat_package_dir"], args["store_in_hat"], batch_size=int(args["batch_size"]), min_time_in_sec=int(args["min_time_in_sec"]), input_sets_minimum_size_MB=int(args["input_sets_minimum_size_MB"]))
    df = pd.DataFrame(results)
    df.to_csv(args["results_file"], index=False)
    pd.options.display.float_format = '{:8.8f}'.format
    print(df)

    print(f"Results saved to {args['results_file']}")

def main_command():
    main(sys.argv[1:]) # drop the first argument (program name)

if __name__ == "__main__":
    main_command()
