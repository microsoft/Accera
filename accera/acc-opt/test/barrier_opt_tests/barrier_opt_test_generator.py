#!/usr/bin/env python3

import inspect
import sys

import accera as acc
# from accera._lang_python._lang import _If, as_index


def build_package(plan, args, name):
    package = acc.Package()
    package.add(plan, args=args, base_name=name)
    package.build(name, format=acc.Package.Format.MLIR_VERBOSE | acc.Package.Format.DEFAULT, output_dir="build")


def barrier():
    acc._lang_python._lang._gpu.Barrier()


def barrier_trivial_test_1():
    '''Simple sequential code test with no barriers'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        shA[i] = A[i]
        A[i] *= 2.0
        B[i] = shA[i]

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_single_warp_test_1():
    '''Simple sequential code test with a single warp of threads'''
    N = 16
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))

    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([N]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        barrier()
        shA[i] = A[i]
        barrier()
        A[i] *= 2.0
        barrier()
        A[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A,), name=inspect.currentframe().f_code.co_name)


def barrier_single_warp_test_2():
    '''Simple sequential code test with a single warp of threads'''
    N = 4096
    blocksize = 16

    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))

    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([N]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        barrier()
        shA[i] = A[i]
        barrier()
        A[i] *= 2.0
        barrier()
        A[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A,), name=inspect.currentframe().f_code.co_name)


def barrier_single_warp_test_3():
    '''Simple sequential code test with a single warp of threads'''
    N = 4096
    blocksize = 32
    subblocksize = 32

    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))

    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([N]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        barrier()
        shA[i] = A[i]
        barrier()
        A[i] *= 2.0
        barrier()
        A[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    iii = schedule.split(ii, subblocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_Y, iii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A,), name=inspect.currentframe().f_code.co_name)


def barrier_multi_warp_test_1():
    '''Simple sequential code test with multiple warps of threads'''
    N = 4096
    blocksize = 256
    subblocksize = 32

    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))

    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([N]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        barrier()
        shA[i] = A[i]
        barrier()
        A[i] *= 2.0
        barrier()
        A[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    iii = schedule.split(ii, subblocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_Y, iii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A,), name=inspect.currentframe().f_code.co_name)


def barrier_seq_test_1():
    '''Simple sequential code test'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        # Performs excessive barriers.
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        barrier()
        shA[i] = A[i]
        barrier()
        A[i] *= 2.0
        barrier() # Only this is needed.
        B[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_seq_test_2():
    '''More complex sequential code test'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        # Performs excessive barriers.
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        barrier()
        shA[i] = A[i]
        barrier()
        shB[i] = B[i]
        barrier()
        C[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_seq_test_3():
    '''More complex sequential code test'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        # Performs excessive barriers.
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        shB[i] = A[i]
        barrier()
        shA[i] = A[i]
        barrier()
        shB[i] = B[i]
        barrier()
        C[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_if_test_1():
    '''Test with an if block'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        def if_block():
            barrier()
            A[i] += 2.0
            barrier()

        barrier()
        shA[i] = A[i]
        barrier()
        acc._lang_python._lang._If(i < acc._lang_python._lang.as_index(N), if_block)
        barrier() # Only this is needed.
        A[i] = shA[i]
        barrier()


    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, C), name=inspect.currentframe().f_code.co_name)


def barrier_if_test_2():
    '''Test with an if block'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        def if_block():
            barrier()
            shB[i] = A[i]
            barrier()
            A[i] = shB[i]
            barrier()

        barrier()
        shA[i] = A[i]
        barrier()

        acc._lang_python._lang._If(i < acc._lang_python._lang.as_index(N), if_block)

        barrier()
        A[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, C), name=inspect.currentframe().f_code.co_name)


def barrier_if_test_3():
    '''Test with an if/else construct'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    # Define the loop nest logic
    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        def if_block():
            barrier()
            shA[i] = A[i]
            barrier()

        def else_block():
            shA[i] = B[i]
            barrier()

        barrier()
        C[i] = shA[i]
        barrier() # This barrier is needed
        acc._lang_python._lang._If(i < acc._lang_python._lang.as_index(N), if_block).Else(else_block)
        barrier() # This barrier is needed
        C[i] = shA[i]
        barrier()

    schedule = nest.create_schedule()
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_if_test_4():
    '''Test with an if/else construct'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    # Define the loop nest logic
    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        def if_block():
            barrier()
            B[i] = shA[i]
            barrier()

        def else_block():
            barrier()
            A[i] = shB[i]
            barrier()

        barrier()
        shA[i] = A[i]
        barrier()
        shB[i] = B[i]
        barrier()

        acc._lang_python._lang._If(i < acc._lang_python._lang.as_index(N), if_block).Else(else_block)

        barrier()
        shA[i] = A[i]
        barrier()
        shB[i] = A[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_loop_test_1():
    '''Test with a loop'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        start = acc.Scalar(0)
        stop = acc.Scalar(32)
        step = acc.Scalar(1)

        def loop_block(ii):
            barrier()
            A[i] = shA[i]
            barrier()

        barrier()
        shA[i] = A[i]
        barrier()
        acc.ForRange(start, stop, step, loop_block)
        barrier()
        shA[i] = B[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_loop_test_2():
    '''Test with a loop'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        start = acc.Scalar(0)
        stop = acc.Scalar(32)
        step = acc.Scalar(1)

        def loop_block(ii):
            shA[i] = B[i]
            barrier()
            A[i] = shA[i]
            barrier()

        A[i] = shA[i]
        barrier()
        acc.ForRange(start, stop, step, loop_block)
        barrier() # unnecessary
        shA[i] = A[i]


    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


## TODO: examples for each of the simple cases inside a loop
def barrier_loop_test_3():
    '''Test with sibling loops'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        start = acc.Scalar(0)
        stop = acc.Scalar(32)
        step = acc.Scalar(1)

        def loop_block_1(ii):
            barrier()
            shA[i] = A[i]
            barrier()
            barrier()
            shA[i] = B[i]
            barrier() # Only this is needed.
            C[i] = shA[i] + shB[i]
            barrier()

        def loop_block_2(ii):
            barrier()
            shA[i] = A[i]
            barrier()
            barrier()
            shA[i] = B[i]
            barrier() # Only this is needed.
            C[i] = shA[i] + shB[i]
            barrier()

        barrier()
        shA[i] = A[0]
        barrier()
        acc.ForRange(start, stop, step, loop_block_1)
        barrier()
        C[0] = B[i]
        barrier()
        acc.ForRange(start, stop, step, loop_block_2)
        barrier()
        C[1] = A[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_loop_test_4():
    '''Test with doubly-nested loops'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    B = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))
        shB = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        start = acc.Scalar(0)
        stop = acc.Scalar(32)
        step = acc.Scalar(1)

        def loop_block_1(ii):
            barrier()
            shA[i] = A[i]
            barrier()
            acc.ForRange(start, stop, step, loop_block_2)
            barrier()
            shA[i] = B[i]
            barrier() # Only this is needed.
            C[i] = shA[i] + shB[i]
            barrier()

        def loop_block_2(ii):
            barrier()
            shA[i] = A[i]
            barrier()
            barrier()
            shA[i] = B[i]
            barrier() # Only this is needed.
            C[i] = shA[i] + shB[i]
            barrier()

        barrier()
        shA[i] = A[0]
        barrier()
        acc.ForRange(start, stop, step, loop_block_1)
        barrier()
        C[0] = B[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, B, C), name=inspect.currentframe().f_code.co_name)


def barrier_loop_test_5():
    '''Test with inter-iteration dependencies in a loop'''
    N = 4096
    A = acc.Array(role=acc.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,))
    C = acc.Array(role=acc.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,))

    blocksize = 16
    nest = acc.Nest(shape=(N,))
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        shA = acc.NativeArray(acc.Allocate(type=acc.ScalarType.float32, layout=acc._lang_python._MemoryLayout([blocksize]).set_memory_space(acc._lang_python._lang._MemorySpace.SHARED)))

        start = acc.Scalar(0)
        stop = acc.Scalar(32)
        step = acc.Scalar(1)

        def loop_block(ii):
            barrier()
            shA[i] = A[i]
            barrier()
            C[i] = shA[i]
            barrier()

        barrier()
        A[i] = shA[i]
        barrier()
        acc.ForRange(start, stop, step, loop_block)
        barrier()
        C[i] = A[i]
        barrier()

    schedule = nest.create_schedule()
    ii = schedule.split(i, blocksize)
    target = acc.Target(category=acc.Target.Category.GPU, runtime=acc.Target.Runtime.ROCM)
    plan = schedule.create_plan(target)
    plan.bind(mapping={i: target.GridUnit.BLOCK_X, ii: target.GridUnit.THREAD_X})

    build_package(plan, args=(A, C), name=inspect.currentframe().f_code.co_name)


def testfunctions():
    return [(name, fn) for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction) if name.startswith('barrier_')]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        for name, fn in testfunctions():
            print(name, end=" ")
        print()
    else:
        for name, fn in testfunctions():
            fn()
