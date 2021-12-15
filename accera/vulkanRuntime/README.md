# Vulkan Runtime Wrapper Library

This library wraps calls to the Vulkan runtime and the shared libs produced can be dynamically loaded and used with the acc-gpu-runner for JIT execution.

This is a port of the `VulkanRuntime` and `mlir-vulkan-wrappers` lib from the llvm-project codebase at `<llvm-project>/mlir/tools/mlir-vulkan-runner/`

## Requirements

The Vulkan SDK is required to build this library - see https://vulkan.lunarg.com/
