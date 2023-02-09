#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import unittest
import os
import time
import sys

if "@CMAKE_INSTALL_PREFIX@"[1:-1] != "CMAKE_INSTALL_PREFIX":
    sys.path.insert(1, "@CMAKE_INSTALL_PREFIX@")
else:
    sys.path.insert(1, os.getcwd())

from accera import Package, Role
from accera.test import verifiers

TEST_PACKAGE_DIR = "test_acccgen"


class ModuleScope:
    """Ensures that the global Package module is restored when using
    private APIs to set and clear the active module
    """
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        from accera._lang_python import _SetActiveModule, _ClearActiveModule
        _ClearActiveModule()
        _SetActiveModule(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        from accera._lang_python import _SetActiveModule, _ClearActiveModule
        _ClearActiveModule()
        _SetActiveModule(Package._default_module)


class ContainerTypesTests(unittest.TestCase):
    def test_valor(self) -> None:
        from accera import ScalarType
        from accera._lang_python import _MemoryLayout
        from accera._lang_python._lang import _Valor, ViewAdapter

        layout = _MemoryLayout([9999, 1])
        x = _Valor(ScalarType.int32, layout)
        self.assertEqual(repr(x.layout), repr(layout))

        x_view = ViewAdapter(x)
        self.assertIsNotNone(x_view)

    def test_scalar(self) -> None:
        from accera import ScalarType, Scalar, as_index
        from accera._lang_python import _MemoryLayout
        from accera._lang_python._lang import Array, ViewAdapter, _Valor

        s = Scalar(ScalarType.int64)
        self.assertEqual(s.type, ScalarType.int64)

        s = Scalar(42)
        self.assertEqual(s, 42)
        other = Scalar(10)
        s > other

        s = Scalar(ScalarType.index)
        self.assertEqual(s.type, ScalarType.index)

        s = as_index(42)
        other = as_index(10)
        s >= other    # comparison with non-null types

        arr = Array(ScalarType.bool, _MemoryLayout([1]))
        s = Scalar(arr)
        self.assertIsNotNone(s)

        s_view = ViewAdapter(s)
        self.assertIsNotNone(s_view)

        for val in [True, False]:
            s = val
            self.assertEqual(s, val)

        # test scalar creation from value with no layout
        for t in [ScalarType.int8, ScalarType.int16, ScalarType.int32, ScalarType.uint8, ScalarType.uint16,
                  ScalarType.float16, ScalarType.float32, ScalarType.float64]:
            x = _Valor(t, _MemoryLayout())
            s = Scalar(x)
            self.assertIsNotNone(s)

    def test_scalar_conditionals(self) -> None:
        from accera import Scalar

        s = Scalar(42)
        self.assertIsInstance(s < 10, Scalar)
        self.assertIsInstance(s > 10, Scalar)
        self.assertIsInstance(s <= 10, Scalar)
        self.assertIsInstance(s >= 10, Scalar)
        self.assertIsInstance(s == 10, Scalar)
        self.assertIsInstance(s != 10, Scalar)

    def test_cast(self) -> None:
        from accera import cast, Array, Nest, ScalarType
        for t in [ScalarType.int8, ScalarType.int16, ScalarType.int32, ScalarType.int64, ScalarType.float16,
                  ScalarType.float32, ScalarType.float64]:
            M, S, N = 16, 11, 10
            A = Array(role=Role.INPUT, element_type=t, shape=(M, S))
            B = Array(role=Role.INPUT, element_type=ScalarType.float32, shape=(S, N))
            C = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(M, N))

            nest = Nest(shape=(M, N, S))
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += A[i, k] + cast(B[k, j], t)
                C[i, j] += A[i, k] - cast(B[k, j], t)
                C[i, j] += A[i, k] * cast(B[k, j], t)
                C[i, j] += A[i, k] / cast(B[k, j], t)

            package = Package()
            package.add(nest, args=(A, B, C), base_name=f"test_cast_{t.name}")
            package_name = f"test_cast_{t.name}"
            with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
                package.build(package_name, output_dir=TEST_PACKAGE_DIR)

    def test_unsigned_cast(self) -> None:
        from accera import cast, Array, Nest, ScalarType
        for t in [ScalarType.uint8, ScalarType.uint16, ScalarType.uint32, ScalarType.uint64]:
            M, S, N = 16, 11, 10
            A = Array(role=Role.INPUT, element_type=t, shape=(M, S))
            B = Array(role=Role.INPUT, element_type=ScalarType.int32, shape=(S, N))
            C = Array(role=Role.INPUT_OUTPUT, element_type=t, shape=(M, N))

            nest = Nest(shape=(M, N, S))
            i, j, k = nest.get_indices()

            @nest.iteration_logic
            def _():
                C[i, j] += A[i, k] + cast(B[k, j], t)
                C[i, j] += A[i, k] - cast(B[k, j], t)
                C[i, j] += A[i, k] * cast(B[k, j], t)
                C[i, j] += A[i, k] / cast(B[k, j], t)

            package = Package()
            package.add(nest, args=(A, B, C), base_name=f"test_unsigned_cast_{t.name}")
            package_name = f"test_unsigned_cast_{t.name}"
            with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
                package.build(package_name, output_dir=TEST_PACKAGE_DIR)


class PackagingTypesTests(unittest.TestCase):
    def test_compiler_options(self) -> None:
        from accera import CompilerOptions, _GetTargetDeviceFromName

        target = _GetTargetDeviceFromName("host")
        self.assertEqual(target.device_name, "host")

        options = CompilerOptions()
        self.assertIsNotNone(options)

        # check some default values
        self.assertEqual(options.vector_width, 4)
        self.assertEqual(options.target_device.device_name, target.device_name)

    def test_module(self) -> None:
        from accera import CompilerOptions
        from accera._lang_python import _Module

        module = _Module(name="test", options=CompilerOptions())
        self.assertIsNotNone(module)
        with ModuleScope(module):
            module.Print()
            module.Verify()

            header_filename = f"test_module_{time.time()}.hat"
            module.WriteHeader(header_filename)
            with open(header_filename, 'r') as f:
                print(f)

            self.assertTrue(os.path.isfile(header_filename))
            os.remove(header_filename)
            module = None

    def test_conditional_function(self) -> None:
        from accera import ScalarType, Package, Nest, Array, Scalar, as_index
        from accera._lang_python._lang import _If

        M = 10
        N = 20
        A = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        nest = Nest(shape=(M * 2, N * 2))
        i, j = nest.get_indices()

        def logic_fn():
            self.assertIsInstance(i, Scalar)
            self.assertIsInstance(j, Scalar)

            def if_block():
                A[i, j] += 1.

            _If(i < as_index(M) and j < as_index(N), if_block)

        nest.iteration_logic(logic_fn)
        schedule = nest.create_schedule()
        plan = schedule.create_plan()
        plan.cache(A, i)

        package = Package()
        package.add(plan, args=(A, ))

        # test introspection
        package_name = "test_conditional_function"
        with verifiers.VerifyPackage(self, package_name, TEST_PACKAGE_DIR):
            package.build(package_name, output_dir=TEST_PACKAGE_DIR)

    def test_package_single_target_enforcement(self) -> None:
        from accera import ScalarType, Package, Target, Nest, Array

        M = 10
        N = 20
        A = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        nest = Nest(shape=(M * 2, N * 2))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] += 1.

        pi3 = Target(Target.Model.RASPBERRY_PI_3B, category=Target.Category.CPU)
        plan1 = nest.create_plan(pi3)
        plan2 = nest.create_plan()

        package = Package()
        for i in range(10):
            package.add(plan1, args=(A, ))

        with self.assertRaises(RuntimeError):
            package.add(plan2, args=(A, ))

    def test_default_module_specialization(self) -> None:
        from accera import ScalarType, Target, Nest, Array

        M = 10
        N = 20
        A = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(M, N))
        nest = Nest(shape=(M * 2, N * 2))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            A[i, j] += 1.

        pi3 = Target(Target.Model.RASPBERRY_PI_3B, category=Target.Category.CPU)
        plan1 = nest.create_plan(pi3)
        plan2 = nest.create_plan()

        for i, (plan, platform) in enumerate(zip([plan1, plan2], [Package.Platform.RASPBIAN, Package.Platform.HOST])):
            p = Package()
            p.add(plan, args=(A, ))
            name = f"specialization{i}"
            with verifiers.VerifyPackage(self, name, TEST_PACKAGE_DIR):
                p.build(name, output_dir=TEST_PACKAGE_DIR, platform=platform)

    def test_cross_module_references(self) -> None:
        from accera import CompilerOptions, Array
        from accera._lang_python import _Module, _ResolveConstantDataReference
        import numpy as np

        M = 5
        N = 16
        data = [float(x) for x in range(M * N)]
        np_arr = np.array(data).reshape(M, N)

        module1 = _Module(name="const_module", options=CompilerOptions())
        with ModuleScope(module1):
            array = Array(role=Role.CONST, data=np_arr)
            self.assertIsNotNone(array)
            module1.Print()

            module2 = _Module(name="ref_module", options=CompilerOptions())
            with ModuleScope(module2):
                _ResolveConstantDataReference(array._value)
                module2.Print()

                # Explicitly delete any existing active module before restoring the active module
                # because all MLIRContext instances wind up sharing a global MLIR ScopedContext
                # and the MLIRContext destructor clears this ScopedContext.
                # When assigning over an existing module, however, the new module is created
                # before the old one is cleaned up, so the destructor of the old one clears out
                # the ScopedContext created by the new one.
                # TODO : Have MLIRContexts behave better with switching between multiple modules
                module2 = None
            module1 = None

    def test_emit_unpacked_buffer(self) -> None:
        from accera import Array
        import numpy as np
        import pathlib
        import shutil

        def embed_buffer(input_matrix: Array, output_matrix: Array):
            from accera import Nest

            _M_input, _N_input = input_matrix.shape
            _M_output, _N_output = output_matrix.shape

            if _M_input != _M_output or _N_input != _N_output:
                raise RuntimeError("Incompatible shapes for arguments")

            M = _M_input
            N = _N_input

            nest = Nest(shape=[M, N])
            i, j = nest.get_indices()

            @nest.iteration_logic
            def _():
                output_matrix[i, j] += input_matrix[i, j]

            schedule = nest.create_schedule()
            return schedule.create_plan()

        domains = [
            {
                "domain": (32, 64)
            },
        # {"domain": (128, 128)}
        ]

        package = Package()

        for domain in domains:
            M, N = domain["domain"]
            data = [float(x) for x in range(M * N)]
            np_arr = np.array(data, dtype=np.float32)
            np_arr = np_arr.reshape(M, N)
            # input_matrix = Array(role=Role.INPUT, shape=(M, N))
            input_matrix = Array(role=Role.CONST, shape=(M, N), data=np_arr)
            output_matrix = Array(role=Role.INPUT_OUTPUT, shape=(M, N))
            domain["function"] = package.add(
                embed_buffer(input_matrix, output_matrix), args=(output_matrix, ), base_name=f"ew_accumulate_{M}_{N}"
            )
            # args=(input_matrix,output_matrix,), base_name=f"ew_accumulate_{M}_{N}")

            output_ref = np.ones(output_matrix.shape, dtype=np.float32)
            domain["correctness_check"] = {
                "before": (output_ref, ),
                "after": (output_ref + np_arr, )
            }

        package_name = "embedded_unpacked_buffer"
        output_dir = pathlib.Path(TEST_PACKAGE_DIR) / package_name
        shutil.rmtree(package_name, ignore_errors=True)
        with verifiers.VerifyPackage(self, package_name, output_dir) as v:
            package.build(package_name, format=Package.Format.MLIR_DYNAMIC, output_dir=output_dir)

            # make sure we can compile the module and resolve the cross-module references
            for domain in domains:
                v.check_correctness(domain["function"].name, **domain["correctness_check"])

    def test_add_callable_function(self) -> None:
        from accera import Array, ScalarType, Nest

        package = Package()
        arr = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(256, 256))
        arr2_placeholder = Array(
            role=Role.INPUT_OUTPUT, element_type=ScalarType.float32, shape=(arr.shape[0] // 8, arr.shape[1] // 8)
        )

        # create a nest
        zero_out_nest = Nest(arr2_placeholder.shape)
        i, j = zero_out_nest.get_indices()

        @zero_out_nest.iteration_logic
        def _():
            arr2_placeholder[i, j] = 0.

        zero_out_fn = package.add(zero_out_nest, args=(arr2_placeholder, ), base_name="zero_out")

        # create our main function that calls the nest function
        def main_test(arr):
            m, n = arr.shape
            arr0 = arr.sub_array(offsets=(0, 0), shape=(m // 2, n // 2))
            arr1 = arr0.sub_array(offsets=(0, 0), shape=(m // 4, n // 4))
            arr2 = arr1.sub_array(offsets=(0, 0), shape=(m // 8, n // 8))
            zero_out_fn(arr2)
            # do something with arr2
            arr2[0, 0] = 3.14
            arr2[-1, -1] = 3.14

        package.add(main_test, args=(arr, ), base_name="main")
        with verifiers.VerifyPackage(self, "test_add_function", TEST_PACKAGE_DIR):
            package.build("test_add_function", output_dir=TEST_PACKAGE_DIR)


class ExecutionPlanTypesTests(unittest.TestCase):
    def test_gpu_config(self) -> None:
        from accera._lang_python._lang import _GPU, _Dim3
        gpu_config = _GPU(grid=_Dim3(x=8, y=16, z=1), block=_Dim3(16, 32, 2), dynamic_shared_memory_size=0, blocks_per_SM=0)

        self.assertEqual(gpu_config.grid.x, 8)
        self.assertEqual(gpu_config.grid.y, 16)
        self.assertEqual(gpu_config.grid.z, 1)

        self.assertEqual(gpu_config.block.x, 16)
        self.assertEqual(gpu_config.block.y, 32)
        self.assertEqual(gpu_config.block.z, 2)


class LogicTypesTests(unittest.TestCase):
    def test_if_context(self) -> None:
        from accera._lang_python._lang import _If, Scalar

        def if_block():
            pass

        def elseif_block():
            pass

        def else_block():
            pass

        s1 = Scalar(10)
        s2 = Scalar(20)

        if_ctx = _If(s1 < s2, if_block)

        if_ctx.ElseIf(s1 > s2, elseif_block)

        if_ctx.Else(else_block)

    def test_logic(self) -> None:
        from accera import logic_function

        def test_fn():
            return 42

        logic = logic_function(test_fn)
        self.assertIsNotNone(logic)

    def test_conditional_logic(self) -> None:
        from accera import ScalarType, Nest, Array, as_index
        from accera._lang_python._lang import _If

        M = 100
        A = Array(role=Role.INPUT_OUTPUT, element_type=ScalarType.int32, shape=(M, M))
        nest = Nest(shape=(M, M))
        i, j = nest.get_indices()

        def test_fn():
            def if_block():
                A[i, j] = 42

            _If(i < as_index(10) and j < as_index(20), if_block)

        nest.iteration_logic(test_fn)
        sched = nest.create_schedule()
        plan = sched.create_plan()
        implementation = plan._create_function(args=(A, ))
        self.assertIsNotNone(implementation)


class TargetsTest(unittest.TestCase):
    def test_equivalence_check(self) -> None:
        from accera import Target
        t1 = Target()
        t2 = Target()
        self.assertEqual(t1, t2)

        t2.num_threads = 64
        self.assertNotEqual(t1, t2)

        t3 = Target(Target.Model.RASPBERRY_PI_3B, category=Target.Category.CPU)
        self.assertNotEqual(t1, t3)


if __name__ == '__main__':
    unittest.main(verbosity=10)
