####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from typing import Callable
from inspect import Parameter, signature
from dataclasses import dataclass, field
from functools import wraps, singledispatch

from ..Targets import Target
from ..lang.Array import Array
from ..lang.Dimension import Dimension
from .._lang_python._lang import Array as NativeArray


@singledispatch
def _unpack_arg(arg: NativeArray):
    return arg    # already unpacked


# Sometimes arrays can be passed directly into Functions
# When this happens, we resolve to the native array, and optionally materialize
# them if the value is empty
@_unpack_arg.register(Array)
def _(arg: Array):
    if arg._value.is_empty:
        if arg.role == Array.Role.TEMP:
            # BUGBUG: this has a side effect, but we also don't want to
            # repeat allocations if the same temp array gets passed into
            # multiple Function's
            # Alternatives are to have the caller explicitly _allocate() (perhaps a better
            # option, but then we would have to explain why these are "temp" arrays),
            # or introspect the arbitrary function implementation (not a good idea)
            arg._allocate()
        else:
            # Restrict the allocate-on-use behavior to Temp arrays only, where these two
            # statements are true:
            # - Temporary arrays are not function arguments, and not constant arrays.
            # - Both function arguments and constant arrays should have non-empty values.
            raise ValueError(
                """A non-temporary array is being used like a temporary array.
Did you specify role=Array.Role.TEMP?"""
            )
    return arg._get_native_array()    # unpack


def role_to_usage(role):
    from .._lang_python import _FunctionParameterUsage

    if role == Array.Role.INPUT or role == Dimension.Role.INPUT:
        return _FunctionParameterUsage.INPUT
    else:
        return _FunctionParameterUsage.INPUT_OUTPUT


@dataclass
class Function:
    name: str = ""    # base_name + _ + generated unique_id
    base_name: str = ""
    public: bool = False
    external: bool = False
    decorated: bool = True    # do we want to expose this?
    requested_args: tuple = ()    # args as provided into Package.add
    args: tuple = ()    # unpacked versions of the args (as native arrays)
    arg_size_references: tuple = () # references from array args to dimension arg positions for dynamically sized arrays
    param_overrides: dict = field(default_factory=dict)    # overrides for constants
    definition: Callable = None
    no_inline: bool = False
    auxiliary: dict = field(default_factory=dict)
    target: Target = Target.HOST
    output_verifiers: list = field(default_factory=list)

    def __post_init__(self):
        # automatically fill if not specified
        if self.args and not self.requested_args:
            self.requested_args = self.args

    def _emit(self):
        from .._lang_python import _DeclareFunction

        if hasattr(self, "_native_fn") and self._native_fn.is_defined:
            return

        self._native_fn = _DeclareFunction(self.name + "_impl")
        for delayed_param, value in self.param_overrides.items():
            delayed_param.set_value(value)

        if self.args:
            usages = [role_to_usage(arg.role) for arg in self.requested_args]
            self._native_fn.parameters(self.args, usages, self.arg_size_references)

            if self.output_verifiers:
                self._native_fn.outputVerifiers(self.output_verifiers)

        self._native_fn.inlinable(not self.no_inline)

        sig = signature(self.definition)

        @wraps(self.definition)
        def wrapper_fn(args):
            if len(args) == len(sig.parameters):
                self.definition(args)
            else:
                # we only have one argument in the definition, so look
                # to see if it a variadic positional argument (*arg)
                if list(sig.parameters.values())[0].kind == Parameter.VAR_POSITIONAL:
                    self.definition(*args)
                else:
                    self.definition(args)

        self._native_fn.define(wrapper_fn)

        if self.public:
            api_decl = _DeclareFunction(self.name)
            if self.args:
                api_decl.parameters(self.args, usages, self.arg_size_references)
            if self.base_name:
                api_decl.baseName(self.base_name)
            api_decl.public(True).decorated(False).headerDecl(True).rawPointerAPI(True).define(self._native_fn)

    def __call__(self, *args):
        self._emit()
        self._native_fn.__call__(list(map(_unpack_arg, args)))
