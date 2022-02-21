####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import ast
import inspect
import textwrap

from .LogicFunction import LogicFunction


class FunctionCallAssignVisitor(ast.NodeVisitor):
    '''
    Visitor pattern class that searches an AST for variables assigned
    to the return value of a given function
    '''
    def __init__(self, func_name):
        self.assigned_variables = []
        self.target_func_name = func_name

    def visit_Assign(self, node):
        '''
        Invoked for every Assign node
        AST Assign from a Call return example:
        Assign(
            targets=[Tuple(elts=[Name(id='i', ctx=Store()),
                                 Name(id='j', ctx=Store()),
                                 Name(id='k', ctx=Store())],
                    ctx=Store())],
            value=Call(func=Attribute(value=Name(id='f', ctx=Load()), attr='make_things_func', ctx=Load()), args=[], keywords=[]),
            type_comment=None)
        '''
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            called_fn_name = node.value.func.attr
            if called_fn_name == self.target_func_name:
                for elt in node.targets[0].elts:
                    self.assigned_variables.append(elt.id)


class NamedNodeVisitor(ast.NodeVisitor):
    '''
    Visitor pattern class that searches an AST for occurrences
    of nodes with the given names
    '''
    def __init__(self, names):
        self.names = names
        self.found = set()

    def visit_Name(self, node):
        if node.id in self.names:
            self.found.add(node.id)


class TrackAssignmentsVisitor(ast.NodeVisitor):
    '''
    Visitor pattern class that traverses an AST and tracks variables that are assigned based on an initial set of base variables.
    This class maintains a distinction between basic assignments to the original base variables and derived variables
    that are the result of some operation on the original variables.
    E.g.:
    given ['i', 'j', 'k'] as base variables,
    foo = i         # foo is a basic assignment to i
    foo = i + 1     # foo is a derived variable from i
    bar = foo       # bar is a derived variable because foo is a derived variable
    i = i + 1       # i is now a derived variable, and so is anything else that referenced it
    '''
    def __init__(self, base_vars):
        self.basic_assignments = {key: key
                                  for key in base_vars}
        self.derived_vars = set()

    def visit_Assign(self, node):
        '''
        Invoked for every Assign node
        AST Assign examples:
        Assign(targets=[Name(id='bar', ctx=Store())], value=BinOp(left=Name(id='i', ctx=Load()), op=Add(), right=Constant(value=1, kind=None)), type_comment=None),
        Assign(targets=[Name(id='baz', ctx=Store())], value=Name(id='j', ctx=Load()), type_comment=None),
        '''
        # Any simple assignments of the base variables should go into the self.basic_assignments map
        # Any expression-based assignment should go into the derived_vars list
        # Any simple assignment using a variable in the derived_vars list should go in the derived_vars list

        # Only bother examining assignments where the value includes something in self.basic_assignments or self.derived_vars
        source_node_names = set(list(self.basic_assignments.keys()) + list(self.derived_vars))
        source_node_visitor = NamedNodeVisitor(source_node_names)
        source_node_visitor.visit(node.value)
        if len(source_node_visitor.found) > 0:
            # Only support single assignment statements currently
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if isinstance(node.value, ast.Name):
                    # Simple assignment, so examine where the named value comes from
                    if node.value.id in self.basic_assignments:
                        self.basic_assignments[target] = node.value.id
                    elif node.value.id in self.derived_vars:
                        self.derived_vars.add(target)
                    else:
                        raise ValueError("Unrecognized assignment value: {}".format(node.value.id))
                else:
                    # It is something other than a simple assignment so it is a derived variable
                    if target in self.basic_assignments:
                        # Remove this as a basic_assignment and anything that was referencing this
                        # as a basic_assignment now must be a derived variable since we aren't tracking
                        # the order of this assignment relative to the logic function evaluation
                        del self.basic_assignments[target]
                        keys = list(self.basic_assignments.keys())
                        for other_elt in keys:
                            if self.basic_assignments[other_elt] == target:
                                del self.basic_assignments[other_elt]
                    self.derived_vars.add(target)


class ArrayAccessVisitor(ast.NodeVisitor):
    '''
    Visitor pattern class that traverses an AST and finds array subscript operations
    that use named variables.
    Note: expressions in the indexing are not supported. E.g. A[i, k] is supported, but A[i, k + 1] is not
    '''
    def __init__(self):
        self.accesses = {}

    def visit_Subscript(self, node):
        '''
        Invoked for every Subscript node

        AST Subscript example (Python 3.8):
        >>> ast.dump(node)
        "Subscript(
            value=Name(id='B', ctx=Load()),
            slice=Index(
                value=Tuple(
                    elts=[Name(id='k', ctx=Load()),
                          Name(id='j', ctx=Load())],
                ctx=Load())),
            ctx=Load())"

        From Python 3.9 onwards, a slice is no longer wrapped in Index:

        >>> ast.dump(node)
        "Subscript(
            value=Name(id='B', ctx=Load()),
            slice=Tuple(
                elts=[Name(id='k', ctx=Load()),
                      Name(id='j', ctx=Load())],
                ctx=Load()),
            ctx=Load())"
        '''
        try:
            #       Python < 3.9                                     Python >= 3.9
            slice = node.slice.value if hasattr(node.slice, "value") else node.slice
            access_indices = [elt.id for elt in slice.elts]
            if node.value.id not in self.accesses:
                self.accesses[node.value.id] = []
            self.accesses[node.value.id].append(access_indices)
        except AttributeError:
            raise ValueError("Only base indices are currently supported when indexing into an array that is cached")


def get_array_accesses(func):
    "returns named variables used to subscript into arrays in the given function"
    code = inspect.getsource(func)
    dedented_code = textwrap.dedent(code)
    tree = ast.parse(dedented_code)
    visitor = ArrayAccessVisitor()
    visitor.visit(tree)
    return visitor.accesses


def get_globals_and_nonlocals(func):
    "gets global and non-local variables that are visible in the given function"
    closure_vars = inspect.getclosurevars(func)
    vars = closure_vars.globals
    vars.update(closure_vars.nonlocals)
    return vars


def get_array_access_indices(arr, func: LogicFunction):
    "get the indices used to access the given global array in the given function using variables defined in the given stack frame"

    # Find array accesses in the logic function
    wrapped_func = func.func
    array_accesses = get_array_accesses(wrapped_func)
    func_arrays = func.get_captures(type(arr))
    func_indices = func.get_indices()
    access_elt_names = []
    for array_name in array_accesses:
        # skip over the arrays until we find the one that matches arr
        if array_name not in func_arrays or func_arrays[array_name] is not arr:
            continue
        for access in array_accesses[array_name]:
            current_access_elt_names = access

            if not access_elt_names:
                access_elt_names = current_access_elt_names

            else:
                if access_elt_names != current_access_elt_names:
                    raise NotImplementedError("Currently only supports one indexing pattern per array per kernel")

    return [func_indices[elt_name] for elt_name in access_elt_names]
