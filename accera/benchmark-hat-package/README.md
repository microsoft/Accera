# benchmark_hat_package
Tool used to benchmark functions in a HAT package.

It is common to produce a HAT package with Accera that includes multiple functions that have the same logic but have different schedules. This tool can be used to find the best performing function on a given target.

## Description
This tool will search a given HAT package and perform the following actions:
- Introspect the function data to find input and output arguments
- Pre-allocate a set of input and output buffers. The set will be large enough to ensure that data is not kept in any caches (e.g. L1, L2 or L3 of a CPU)
- Generate random input data
- Call the function in a loop running through input sets until a minimum amount of time and minimum number of iterations has passed
- Calculate the mean duration for the function
- Store the results, either in a __.csv__ file or write it back to the function's __.hat__ file in the package as meta-data

NOTE: The results should only be used to compare relative performance of functions measured using this tool. It is not accurate to compare duration measurents from this tool with duration measured from another tool.

## Requirements
- Python 3.7+
- Pandas installed:
    - `pip install pandas`
- Numpy installed:
    - `pip install numpy`
- The `accera` package installed. If you don't have it installed, you can find the instructions for how to build your own or install a pre-built version [here](../../docs/Install/README.md)

## Usage
```
> python benchmark_hat_package.py --help
usage: benchmark_hat_package.py [-h] [--store_in_hat]
                                [--results_file RESULTS_FILE]
                                [--min_iterations MIN_ITERATIONS]
                                [--min_time_in_sec MIN_TIME_IN_SEC]
                                [--input_sets_minimum_size_MB INPUT_SETS_MINIMUM_SIZE_MB]
                                path_to_hat_package

Runs a simple benchmark for the HAT package to get a mean duration of each
function. Example: benchmark_hat_package.py <path_to_HAT_package>

positional arguments:
  path_to_hat_package_dir   Path to the HAT package directory

optional arguments:
  -h, --help            show this help message and exit
  --store_in_hat        If set, will write the duration as meta-data back into
                        the hat file
  --results_file RESULTS_FILE
                        Full path where the results will be written
  --min_iterations MIN_ITERATIONS
                        Minimum number of iterations to run
  --min_time_in_sec MIN_TIME_IN_SEC
                        Minimum number of seconds to run the benchmark for
  --input_sets_minimum_size_MB INPUT_SETS_MINIMUM_SIZE_MB
                        Minimum size in MB of the input sets. Typically this
                        is large enough to ensure eviction of the biggest
                        cache on the target (e.g. L3 on an desktop CPU)
```

For example:
```
python benchmark_hat_package.py C:\myProject\HAT_package_dir --min_time_in_sec=15
```

### --store_in_hat
When using `--store_in_hat` flag, the .hat file will be updated with an `auxiliary` data section like:
```
[functions.myfunction_py_c3723b5f.auxiliary]
duration_in_sec = 1.5953456437541567e-06
```
