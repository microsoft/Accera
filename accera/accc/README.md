# ACCC
End-to-end eAccera Compiler toolchain.

- [ACCC](#accc)
  - [Description](#description)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Parameters](#parameters)
  - [Examples](#examples)
    - [CPU](#cpu)
    - [GPU](#gpu)

## Description

Given:
- Accera DSL C++ file
- csv file of domain sizes
- Library name
- Runner main file

It will:
- Create a generator for the Accera DSL
- Run the generator and lower the result, emitting an object file and header for the current host machine
- Create a runner project for the runner main file
- Build and run the runner program

[Back to top](#ACCC)

## Requirements
- Python 3.7+
- Pandas installed:
    - `pip install pandas`
- Accera sample uses generalized emitting infrastructure, following the pattern in `samples\GEMM\MLAS_value\Accera_Sample.cpp`

[Back to top](#ACCC)

## Usage
- Build Accera repo install target
    ```
    mkdir build
    cd build
    cmake .. <<platform-specific cmake initialization>>
    cmake --build . --config Release --target install
- Run `accc.py` from the install directory with desired Accera DSL file, config files, name, and output directory
    ```
    cd build
    python install/bin/accc.py path/to/accera_sample.cpp
                              --domain path/to/config.csv
                              --library_name MyDesiredLibraryName
                              --output path/to/outputdir
                              --generator_custom_args path/to/default/custom/args/generator_file.yml
                              --main path/to/runner_main.cpp
                              --main_custom_args path/to/default/custom/args/main_file.yml
                              --run
    ```

[Back to top](#ACCC)

## Parameters
```
>  python <build_dir>\install\bin\accc.py --help
usage: accc.py [-h] [-c {Release,Debug,RelWithDebInfo}] -d DOMAIN -m LIBRARY_NAME -o OUTPUT [--generator_custom_args GENERATOR_CUSTOM_ARGS] [--debug_ir_output_file DEBUG_IR_OUTPUT_FILE] [--dump_all_passes] [-t {CPU,GPU}] [--main MAIN] [--main_custom_args MAIN_CUSTOM_ARGS] [--deploy_shared_libraries [DEPLOY_SHARED_LIBRARIES [DEPLOY_SHARED_LIBRARIES ...]]] [--run] [-n] path/to/generator_sample.cpp

positional arguments:
  path/to/generator_sample.cpp
                        Path to the generator main C++ file containing Accera DSL to emit

optional arguments:
  -h, --help            show this help message and exit
  -c {Release,Debug,RelWithDebInfo}, --build_config {Release,Debug,RelWithDebInfo}
                        Config to build generator and (optional) main program with
  -d DOMAIN, --domain DOMAIN
                        Path to the domain sizes configuration file to emit for.
  -m LIBRARY_NAME, --library_name LIBRARY_NAME
                        Name to give the generated library, as well as the name prefix for the generator, emitted files, and main
  -o OUTPUT, --output OUTPUT
                        Output directory, also used as a working directory
  --generator_custom_args GENERATOR_CUSTOM_ARGS
                        Path to the yaml file giving the generator-specific custom parameters
  --debug_ir_output_file DEBUG_IR_OUTPUT_FILE
                        File to pipe IR after each stage to. Warning: including this option will slow down compilation.
  --dump_all_passes     Dump the result of each pass to a separate file in the all_passes directory. Warning: including this option will slow down compilation.
  -t {CPU,GPU}, --target {CPU,GPU}
                        The target type of hardware to lower to
  --main MAIN           Path to main C++ to deploy and build against emitted binary. Optional - if a path is not provided no main program will be produced.
  --main_custom_args MAIN_CUSTOM_ARGS
                        Path to the yaml file giving the main-program-specific custom parameters
  --deploy_shared_libraries [DEPLOY_SHARED_LIBRARIES [DEPLOY_SHARED_LIBRARIES ...]]
                        Paths to shared libraries to copy into the main build directory. By default, nothing is deployed for a CPU target and the Vulkan runtime wrapper is deployed for a GPU target
  --run                 Run the main program after building it
  -n, --pretend         Print out the commands that would be run, but don't actually invoke any of them.
```

[Back to top](#ACCC)

## Examples

### CPU
Generating for the sample in `samples/GEMM/MLAS_value/Accera_Sample.cpp`:

```
> mkdir build
> cd build
> cmake .. <<platform-specific cmake initialization>>
...
> cmake --build . --config Release --target install
...
> python install\bin\accc.py ..\samples\GEMM\MLAS_value\Accera_Sample.cpp --domain ..\experimental\configs\gemm\smoke_test.csv --library_name mlas_value_sample_lib --output .\mlas_value_sample --generator_custom_args ..\samples\GEMM\MLAS_value\default_generator_args.yml --main ..\samples\GEMM\timing_main.cpp --run
...
```

The above invocation will:
1. Create a directory `mlas_value_sample`
1. Create a subdirectory `mlas_value_sample/generator` and make an Accera generator CMake project there with the given Accera DSL file.
1. Build the generator
1. Run the generator with the given domain csv and custom argument values from the given config file.
1. Run `acc-opt.exe`, `mlir-translate.exe`, `llc.exe`, and `opt.exe` lowering the emitted code to a header and object file.
1. Create a subdirectory `mlas_value_sample/mlas_value_sample_lib_intermediate` and put intermediate IR files there that are the result of running the generator, `acc-opt.exe`, `mlir-translate.exe`, `llc.exe`, and `opt.exe`, which include the final header for the Accera sample.
1. Create a subdirectory `mlas_value_sample/lib` containing the project for the static library for the Accera sample.
1. Create a subdirectory `mlas_value_sample/logs` and put the `stdout` and `stderr` logs for each phase there.
1. (Because the `--main` argument was provided) Create a subdirectory `mlas_value_sample/main` and make an Accera main CMake project there with the given Accera main file and build the project.
1. (Because the `--run` argument was provided) Run the build main project.

Note: the intermediate files and the generator and runner projects will be named based on the `--library_name` parameter

```
> dir mlas_value_sample\
...
10/27/2020  02:44 PM    <DIR>          .
10/27/2020  02:44 PM    <DIR>          ..
10/27/2020  02:44 PM    <DIR>          generator
10/27/2020  02:44 PM    <DIR>          lib
10/27/2020  02:45 PM    <DIR>          logs
10/27/2020  02:45 PM    <DIR>          main
10/27/2020  02:44 PM    <DIR>          mlas_value_sample_lib_intermediate

> dir mlas_value_sample\generator\
...
10/27/2020  02:44 PM    <DIR>          .
10/27/2020  02:44 PM    <DIR>          ..
10/27/2020  02:44 PM    <DIR>          build
10/27/2020  02:44 PM             3,893 CMakeLists.txt
10/27/2020  02:44 PM    <DIR>          include
10/27/2020  02:44 PM    <DIR>          src

> dir mlas_value_sample\generator\build\Release\
...
10/27/2020  02:44 PM    <DIR>          .
10/27/2020  02:44 PM    <DIR>          ..
10/27/2020  02:44 PM        36,547,584 mlas_value_sample_lib_generator.exe

> dir mlas_value_sample\lib\
...
10/27/2020  02:44 PM    <DIR>          .
10/27/2020  02:44 PM    <DIR>          ..
10/27/2020  02:45 PM    <DIR>          build
10/27/2020  02:44 PM               794 CMakeLists.txt
10/27/2020  02:44 PM    <DIR>          include
10/27/2020  02:44 PM    <DIR>          src

> dir mlas_value_sample\lib\build\Release\
...
10/27/2020  02:45 PM    <DIR>          .
10/27/2020  02:45 PM    <DIR>          ..
10/27/2020  02:45 PM            34,786 mlas_value_sample_lib.lib

> dir mlas_value_sample\mlas_value_sample_lib_intermediate\
...
10/27/2020  02:44 PM    <DIR>          .
10/27/2020  02:44 PM    <DIR>          ..
10/27/2020  02:44 PM             8,164 BenchmarkingUtilities.bc
10/27/2020  02:44 PM            46,426 BenchmarkingUtilities.ll
10/27/2020  02:44 PM            26,864 BenchmarkingUtilities.mlir
10/27/2020  02:44 PM             5,653 BenchmarkingUtilities.obj
10/27/2020  02:44 PM            45,889 BenchmarkingUtilities_llvm.mlir
10/27/2020  02:44 PM            19,384 MLASValueGEMM_256_256_256_module.bc
10/27/2020  02:44 PM         3,328,284 MLASValueGEMM_256_256_256_module.ll
10/27/2020  02:44 PM            12,091 MLASValueGEMM_256_256_256_module.mlir
10/27/2020  02:44 PM             9,399 MLASValueGEMM_256_256_256_module.obj
10/27/2020  02:44 PM         2,030,591 MLASValueGEMM_256_256_256_module_llvm.mlir
10/27/2020  02:44 PM            15,040 MLASValueGEMM_32_32_32_module.bc
10/27/2020  02:44 PM         1,980,990 MLASValueGEMM_32_32_32_module.ll
10/27/2020  02:44 PM             9,059 MLASValueGEMM_32_32_32_module.mlir
10/27/2020  02:44 PM             9,179 MLASValueGEMM_32_32_32_module.obj
10/27/2020  02:44 PM         1,262,946 MLASValueGEMM_32_32_32_module_llvm.mlir
10/27/2020  02:44 PM            12,676 MLASValueGEMM_49_128_256_module.bc
10/27/2020  02:44 PM         1,777,183 MLASValueGEMM_49_128_256_module.ll
10/27/2020  02:44 PM             9,184 MLASValueGEMM_49_128_256_module.mlir
10/27/2020  02:44 PM             7,121 MLASValueGEMM_49_128_256_module.obj
10/27/2020  02:44 PM         1,134,335 MLASValueGEMM_49_128_256_module_llvm.mlir
10/27/2020  02:44 PM             2,656 mlas_value_sample_lib.h

> dir mlas_value_sample\main\
...
10/27/2020  02:45 PM    <DIR>          .
10/27/2020  02:45 PM    <DIR>          ..
10/27/2020  02:45 PM    <DIR>          build
10/27/2020  02:45 PM             4,310 CMakeLists.txt
10/27/2020  02:45 PM    <DIR>          deploy
10/27/2020  02:45 PM    <DIR>          include
10/27/2020  02:45 PM    <DIR>          src

> dir mlas_value_sample\main\build\Release\
...
10/27/2020  02:45 PM    <DIR>          .
10/27/2020  02:45 PM    <DIR>          ..
10/27/2020  02:45 PM           478,720 mlas_value_sample_lib_main.exe

> dir mlas_value_sample\logs\
...
10/27/2020  02:45 PM    <DIR>          .
10/27/2020  02:45 PM    <DIR>          ..
10/27/2020  02:45 PM               100 emitted_library_stderr.txt
10/27/2020  02:45 PM             1,827 emitted_library_stdout.txt
10/27/2020  02:44 PM                 0 emit_stderr.txt
10/27/2020  02:44 PM                 0 emit_stdout.txt
10/27/2020  02:44 PM               241 generator_stderr.txt
10/27/2020  02:44 PM           106,018 generator_stdout.txt
10/27/2020  02:44 PM                 0 llc_stderr.txt
10/27/2020  02:44 PM                 0 llc_stdout.txt
10/27/2020  02:45 PM               241 main_stderr.txt
10/27/2020  02:45 PM            76,403 main_stdout.txt
10/27/2020  02:44 PM                 0 mlir_lowering_stderr.txt
10/27/2020  02:44 PM                 0 opt_stderr.txt
10/27/2020  02:44 PM                 0 opt_stdout.txt
10/27/2020  02:44 PM                 0 translate_mlir_stderr.txt
10/27/2020  02:44 PM                 0 translate_mlir_stdout.txt
```

Splitting the `accc.py` invocation above for better readability:
```
python install\bin\accc.py
    ..\samples\GEMM\MLAS_value\Accera_Sample.cpp
    --domain ..\experimental\configs\gemm\smoke_test.csv
    --library_name mlas_value_sample_lib
    --output .\mlas_value_sample
    --generator_custom_args ..\samples\GEMM\MLAS_value\default_generator_args.yml
    --main ..\samples\GEMM\timing_main.cpp
    --run
```

[Back to top](#ACCC)

### GPU

Generating for the sample in `samples/VectorAddition/BasicLoopNest/GPUVectorAddition.cpp`:

```
> mkdir build
> cd build
> cmake .. <<platform-specific cmake initialization>>
...
> cmake --build . --config Release --target install
...
> python install\bin\accc.py ..\samples\VectorAddition\BasicLoopNest\GPUVectorAddition.cpp --domain ..\experimental\configs\VectorAddition\smoke_test.csv --library_name vec_add_gpu_lib --output .\vec_add_gpu --main ..\samples\VectorAddition\timing_main.cpp --gpu --run
...
```

Splitting the `accc.py` invocation above for better readability:
```
python install\bin\accc.py
    ..\samples\VectorAddition\BasicLoopNest\GPUVectorAddition.cpp
    --domain ..\experimental\configs\VectorAddition\smoke_test.csv
    --library_name vec_add_gpu_lib
    --output .\vec_add_gpu
    --main ..\samples\VectorAddition\timing_main.cpp
    --gpu
    --run
```

Note, many of the same files that the CPU samples create are also created for GPU samples, but the Vulkan Runtime library will also be deployed alongside the runner main program:
```
dir vec_add_gpu\main\build\Release\
...
10/27/2020  02:59 PM    <DIR>          .
10/27/2020  02:59 PM    <DIR>          ..
10/27/2020  02:59 PM            57,856 acc-vulkan-runtime-wrappers.dll
10/27/2020  02:59 PM           480,256 vec_add_gpu_lib_main.exe
```

[Back to top](#ACCC)
