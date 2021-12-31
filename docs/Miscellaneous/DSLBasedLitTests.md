# Getting started with DSL lit tests

## Summary
This guide will walk you through the process of adding a unit test for the Accera eDSLs. This unit test will make use of the [Catch2](https://github.com/catchorg/Catch2) testing library. Additionally, it will also make use of the [LLVM Integrated Tester (lit)](https://www.llvm.org/docs/CommandGuide/lit.html) to verify the emitted IR output.

This allows the authoring of tests that can be used to verify not only the API but also the IR that gets emitted and all relevant transforms on this IR.

## Requirements
Must have `lit` installed and discoverable on your PATH. Easiest way is to use `pip install lit`.

## The steps
1. Open `tools/acc-opt/test/value_mlir_test.cpp`
1. Add the Catch2 part of the skeleton of your new test
    ```cpp
    TEST_CASE("function_decl1")
    {
    }
    ```
1. Add the lit part of the skeleton above the function you just created
    ```cpp
    // CHECK-LABEL: module @function_decl1 {

    // CHECK: }
    TEST_CASE("function_decl1")
    ```
1. Fill in the Catch2 part of the test function
    ```cpp
    TEST_CASE("function_decl1")
    {
        auto f1 =
            DeclareFunction("f1")
                .Define([] {});
        CHECK(f1);
        auto f2 =
            DeclareFunction("f2")
                .Parameters(Value{ ValueType::Int32, ScalarLayout })
                .Define([](Scalar) {});
        CHECK(f2);
    }
    ```
1. Verify the emitted output of the newly created test
    1. Build the `value_mlir_test` target
    1. Execute `value_mlir_test` with the newly added test name as a command-line argument
        ```shell
        $ bin/value_mlir_test function_decl1
        Filters: function_decl1
        #map0 = affine_map<() -> (0)>
        #map1 = affine_map<(d0, d1) -> (d0 * 4 + d1)>


        module @function_decl1 {
            func @f1_0() {
                return
            }
            func @f2_92176250623052430(%arg0: memref<i32, #map0> {llvm.noalias = true}) {
                return
            }
        }


        ===============================================================================
        test cases: 1 | 1 passed
        assertions: - none -
        ```

    1. Update the tests until all tests pass with the correct IR being output
1. Use the verified output to author the lit portion of the test
    ```cpp
    // CHECK-LABEL: module @function_decl1 {
    // CHECK-NEXT: func @f1_
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    // CHECK-NEXT: func @f2_{{[0-9]+}}(%arg0: memref<i32, [[MAP:#map[0-9]+]]> {llvm.noalias = true}) {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    // CHECK-NEXT: }
    TEST_CASE("function_decl1")
    ```
1. Verify that the entire test passes by building the `check-all` target. This will build the `value_mlir_target`, execute all tests, and verify their output.
    ```shell
    $ cmake --build . --target check-all
    [2/3] Running the lit regression tests
    -- Testing: 2 tests, 2 threads --
    PASS: Accera :: value_mlir_test.cpp (1 of 2)
    PASS: Accera :: commandline.mlir (2 of 2)
    Testing Time: 0.31s
    Expected Passes    : 2
    ```
