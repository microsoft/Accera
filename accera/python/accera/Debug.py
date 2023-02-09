####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import Dict, List, Tuple, Any

from .Targets import Target
from .Package import Package
from .lang import Array, Nest, Function
from ._lang_python import Role
from ._lang_python._lang import Dimension


def get_args_to_debug(func: Function) -> List[Array]:
    """Gets the arguments of interest to debugging
    For example, INPUT_OUTPUT Arrays
    """
    args_to_check = [
        arg for arg in func.requested_args if isinstance(arg, Array) and arg.role == Role.INPUT_OUTPUT
    ]
    return args_to_check


def add_check_allclose(package: Package, array: Array, atol: float = 1e-5, target: Target = Target.HOST) -> Function:
    """Adds a function to check whether two arrays are equal up to the specified tolerance
    Inspired by numpy.testing.assert_allclose

    Args:
        package: the package to add the function
        array: the array specification
        atol: the absolute tolerance
    """
    from ._lang_python._lang import CheckAllClose

    shape = array.shape
    element_type = array.element_type
    layout = array._requested_layout
    resolved_shape = [0 if isinstance(s, Dimension) else s for s in shape]
    shape_str = '_'.join(map(str, resolved_shape))

    # placeholders
    actual = Array(role=Role.INPUT, element_type=element_type, shape=shape, layout=layout)
    desired = Array(role=Role.INPUT, element_type=element_type, shape=shape, layout=layout)
    dims = [x for x in shape if isinstance(x, Dimension)]

    # so that we can unwrap the native arrays
    nest = Nest((1, )) 

    @nest.iteration_logic
    def _():
        CheckAllClose(actual, desired, atol, dims)

    plan = nest.create_plan(target)
    args = dims + [actual, desired]
    return package.add(plan, args=args, base_name=f"_debug_check_allclose_{shape_str}")


def add_debugging_functions(
    package: Package, functions_to_args: Dict[str, Tuple[List, Function]], atol: float = 1e-5
) -> Dict:
    """Adds debugging functions to check whether INPUT_OUTPUT arrays are
    equal up to the specified tolerance.

    Args:
        package: the package to add the function
        functions_to_args: A dictionary that maps function name to arguments
            of that function to be debugged
        atol: the absolute tolerance
    Returns:
        A dictionary that maps function name to debugging function names
        for arguments to be debugged
    """
    available_fns = dict()    # reuse functions if the argument signatures match
    result = dict()

    def get_signature(arg: Array):
        return hash(f"{arg.element_type.value}:{str(arg.layout)}")

    for name in functions_to_args:
        function, args = functions_to_args[name]
        result[name] = []
        for arg in args:
            sig = get_signature(arg)
            if sig not in available_fns:
                available_fns[sig] = add_check_allclose(package, arg, atol, function.target)
            result[name].append(available_fns[sig].name)

    return result
