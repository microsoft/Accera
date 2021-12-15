# acc-gpu-runner

The `acc-gpu-runner` tool is functionally a wrapper around `acc-opt` and `mlir-vulkan-runner`.
It takes in a Accera-emitted MLIR file produced by a Accera generator and does the following:
- Runs the Accera lowering passes like `acc-opt` does
- Runs the GPU and Vulkan passes that `mlir-vulkan-runner` does
- Runs the lowered MLIR code in a GPU JIT engine

## Requirements
- Vulkan SDK installed. See https://vulkan.lunarg.com/
- If you have multiple GPUs on your machine, you may need to modify your system settings to use your desired GPU when running `acc-gpu-runner`
    - On Windows:
        1. Press the Windows key
        1. Search for "Graphics Settings" and run
        1. Select "Desktop App" from the drop-down menu and click browse
        1. Navigate to and select your built `acc-gpu-runner.exe`
        1. Once that is added to the list, click on its "Options" button and select the GPU you want it to run with

## Usage
```
> cmake --build . --target acc-gpu-runner
...
> bin\acc-gpu-runner.exe <input.mlir> [--mlir-bin-path=path/to/MLIR/install/bin] [--verbose]
```
Note: by default, will use the path to the MLIR `bin` directory from the MLIR installed via CMake to load the following dynamic libraries:
- `vulkan-runtime-wrappers.dll`
- `mlir_runner_utils.dll`

The `--mlir-bin-path` optional argument can be provided to load these libraries from a different directory.
