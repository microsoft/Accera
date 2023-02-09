####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import hatlib as hat
import json
import logging
import os
import re
import shutil
from collections import OrderedDict
from enum import Enum, Flag, auto
from functools import wraps, singledispatch
from hashlib import md5
from secrets import token_hex
from typing import *

from . import _lang_python, lang, algorithms
from .Targets import Target, Runtime
from .Parameter import *
from .Constants import inf
from .Platforms import Platform, get_library_reference

_R_INT = r"(\d+)"
_R_DIM3 = f"dim3\({_R_INT},\s*{_R_INT},\s*{_R_INT}\)"
_R_GPU_LAUNCH = f"<<<{_R_DIM3},\s*{_R_DIM3},\s*{_R_INT}>>>"
del _R_DIM3


@singledispatch
def _convert_arg(arg: _lang_python._lang._Valor):
    if isinstance(arg, _lang_python._lang.Dimension):
        return _lang_python._lang.Scalar(arg, role=arg.role)
    if isinstance(arg, _lang_python._lang.Scalar):
        return arg
    if arg.layout == _lang_python._MemoryLayout():
        return _lang_python._lang.Scalar(arg)
    else:
        return _lang_python._lang.Array(arg)


@_convert_arg.register(lang.Array)
def _(arg: lang.Array):
    return arg._get_native_array()


@singledispatch
def _resolve_array_shape(source, arr: lang.Array):
    is_infinite_value = (
        arr.shape[-1].get_value() == inf
        if isinstance(arr.shape[-1], DelayedParameter)
        else arr.shape[-1] == inf
    )
    if is_infinite_value:
        # TODO: support shape inference for lang.Function, Callable if needed
        raise NotImplementedError(f"Array shape cannot be resolved for {type(source)}")
    return


@_resolve_array_shape.register(lang.Nest)
def _(source, arr: lang.Array):
    from .lang.IntrospectionUtilities import get_array_access_indices

    is_infinite_value = (
        arr.shape[-1].get_value() == inf
        if isinstance(arr.shape[-1], DelayedParameter)
        else arr.shape[-1] == inf
    )
    if is_infinite_value:
        # introspect array access index to determine dimensions of the array
        logic_fns = source.get_logic()
        # TODO: support multiple logic fns if needed
        assert len(logic_fns) == 1, "Only one logic function is supported"
        access_indices = get_array_access_indices(arr, logic_fns[0])
        assert len(access_indices) == len(
            arr.shape
        ), "Access indices and shape must have the same dimensions"
        idx = source.get_indices().index(access_indices[-1])

        # initialize the array with the new shape
        inferred_shape = tuple(arr.shape[:-1] + [source.get_shape()[idx]])
        arr._init_delayed(inferred_shape)


@_resolve_array_shape.register(lang.Schedule)
def _(source, arr: lang.Array):
    _resolve_array_shape(source._nest, arr)


@_resolve_array_shape.register(lang.Plan)
def _(source, arr: lang.Array):
    _resolve_array_shape(source._sched._nest, arr)


def _emit_module(module_to_emit, target, mode, output_dir, name):
    from . import accc

    assert target._device_name, "Target is unknown"
    working_dir = os.path.join(output_dir, "_tmp")

    proj = accc.AcceraProject(output_dir=working_dir, library_name=name)
    proj.module_file_sets = [
        accc.ModuleFileSet(name=name, common_module_dir=working_dir)
    ]
    module_to_emit.Save(proj.module_file_sets[0].generated_mlir_filepath)

    proj.generate_and_emit(
        build_config=mode.value,
        system_target=target._device_name,
        runtime=target.runtime.name,
    )

    # Create initial HAT files containing shape and type metadata that the C++ layer has access to
    header_path = os.path.join(output_dir, name + ".hat")
    module_to_emit.WriteHeader(header_path)

    # Complete the HAT file with information we have stored at this layer
    hat_file = hat.HATFile.Deserialize(header_path)
    hat_file.dependencies.link_target = os.path.basename(
        proj.module_file_sets[0].object_filepath
    )
    hat_file.Serialize(header_path)

    # copy HAT package files into output directory
    shutil.copy(proj.module_file_sets[0].object_filepath, output_dir)
    return header_path


class SetActiveModule:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        _lang_python._ClearActiveModule()
        _lang_python._SetActiveModule(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _lang_python._ClearActiveModule()
        _lang_python._SetActiveModule(Package._default_module)


class Package:
    "A package of functions that can be built and linked with client code."

    class Format(Flag):
        DYNAMIC_LIBRARY = auto()
        STATIC_LIBRARY = auto()
        HAT_PACKAGE = auto()
        MLIR = auto()
        MLIR_VERBOSE = auto()
        SOURCE = auto()
        DEFAULT = auto()  # HAT_DYNAMIC on HOST target, HAT_STATIC otherwise
        HAT_DYNAMIC = (
            HAT_PACKAGE | DYNAMIC_LIBRARY
        )  #: HAT package format, dynamically linked.
        HAT_STATIC = (
            HAT_PACKAGE | STATIC_LIBRARY
        )  #: HAT package format, statically linked
        HAT_SOURCE = HAT_PACKAGE | SOURCE
        MLIR_DYNAMIC = (
            HAT_DYNAMIC | MLIR
        )  #: MLIR (debugging) package format, dynamically linked.
        MLIR_STATIC = (
            HAT_STATIC | MLIR
        )  #: MLIR (debugging) package format, statically linked.
        MLIR_SOURCE = HAT_SOURCE | MLIR

    class Mode(Enum):
        RELEASE = "Release"  #: Release (maximally optimized).
        DEBUG = "Debug"  #: Debug mode (automatically tests logical equivalence).

    class _Options(Flag):
        NONE = auto() # (enable auto unroll | low precision fp ops)
        DISABLE_AUTO_UNROLL = auto()
        HIGH_PRECISION_FLOATING_POINT_OPS = auto()

    Platform = Platform

    # class attribute to track the default module
    _default_module = None

    def __init__(self):
        self._fns: OrderedDict[str, Any] = OrderedDict()
        self._description = {}
        self._dynamic_dependencies = set()

    def _create_gpu_utility_module(
        self, compiler_options, target, mode, output_dir, name="AcceraGPUUtilities"
    ):
        gpu_utility_module = _lang_python._Module(name=name, options=compiler_options)

        with SetActiveModule(gpu_utility_module):
            gpu_init_fn = _lang_python._DeclareFunction("AcceraGPUInitialize")
            gpu_deinit_fn = _lang_python._DeclareFunction("AcceraGPUDeInitialize")

            gpu_init_fn.public(True).decorated(False).headerDecl(True).rawPointerAPI(
                True
            ).addTag("rc_gpu_init")
            gpu_deinit_fn.public(True).decorated(False).headerDecl(True).rawPointerAPI(
                True
            ).addTag("rc_gpu_deinit")

            # No common initialization / de-initialization at this layer, however lowering passes may add steps
            def empty_func(args):
                pass

            gpu_init_fn.define(empty_func)
            gpu_deinit_fn.define(empty_func)

        return _emit_module(gpu_utility_module, target, mode, output_dir, name)

    def _create_mapping_of_heuristic_parameters_with_possible_values(
        self,
        source: Union["accera.Nest", "accera.Schedule", "accera.Plan", "accera.Function", Callable]
    ):
        parameter_dict = {}
        heuristic_parameters = source._get_heuristic_parameters()
        # heuristic_parameters is a list of list of params
        if heuristic_parameters:
            for heuristic_parameter_list in heuristic_parameters:
                for heuristic_parameter in heuristic_parameter_list:
                    possible_values = heuristic_parameter.get_possible_values()
                    parameter_dict[heuristic_parameter] = possible_values
            return parameter_dict

    def add(
        self,
        source: Union[
            "accera.Nest", "accera.Schedule", "accera.Plan", "accera.Function", Callable
        ],
        args: List[Union["accera.Dimension", "accera.Array"]] = None,
        base_name: str = "",
        parameters: Union[dict, List[dict]] = {},
        function_opts: dict = {},
        auxiliary: dict = {}
    ) -> Union["accera.Function", List["accera.Function"]]:
        """Adds a function to the package. If multiple parameters are provided,
        generates and adds them according to the parameter grid.

        Returns a list of functions added if multiple parameters are provided, otherwise the function added.

        Args:
            source: The source which defines the function's implementation.
            args: The order of external-scope arrays used in the function signature.
            base_name: A base name for the function. The full name for the function will be the
                base name followed by an automatically-generated unique identifier.
            parameters: A mapping of parameter to values for each parameter used by the function implementation (if any).
                        Optionally, can be a list of mappings, which will result in multiple functions.
            function_opts: A dictionary of advanced options to set on the function, e.g. {"no_inline" : True}
            auxiliary: A dictionary of auxiliary metadata to include in the HAT package.
        """

        # TEMP arrays in the args list are a programming error because they are meant to be internally defined in a function
        # Note: this does not prevent TEMP arrays from being passed as an argument to a function, but they cannot be the
        #       api-defining arguments for the function
        for idx, arg in enumerate(args):
            if isinstance(arg, (lang.Array, _lang_python._lang.Scalar, _lang_python._lang.Dimension)) and arg.role == _lang_python.Role.TEMP:
                raise ValueError(f"Error in package.add() for function {base_name}: args includes TEMP array at positions {idx}")

        heuristic_parameters_dict = {}
        if isinstance(source, lang.Plan):
            heuristic_parameters_dict = self._create_mapping_of_heuristic_parameters_with_possible_values(source)

        # TODO: Get product of user-defined parameters and heuristic parameters
        # TODO: product_parameter_grid = get_product_of_parameters(parameters, heuristic_parameters_dict)
        product_parameter_grid = []

        # Create a list of delayed parameter for each possible value separately using `get_parameters_from_grid`
        if heuristic_parameters_dict: 
            product_parameter_grid = get_parameters_from_grid(heuristic_parameters_dict)

        # TODO: Add functions for product parameter grid in next PR instead of adding fns separately for
        # user-defined and heuristic parameters.
        if parameters and not isinstance(parameters, dict):
            return [self._add_function(source, args, base_name, p, function_opts, auxiliary) for p in parameters]
        elif product_parameter_grid and not isinstance(product_parameter_grid, dict):
            return [self._add_function(source, args, base_name, p, function_opts, auxiliary) for p in product_parameter_grid]
        else:
            return self._add_function(
                source, args, base_name, parameters, function_opts, auxiliary
            )

    def _add_function(
        self,
        source: Union[
            "accera.Nest", "accera.Schedule", "accera.Plan", "accera.Function", Callable
        ],
        args: List[Union["accera.Dimension", "accera.Array"]] = None,
        base_name: str = "",
        parameters: dict = {},
        function_opts: dict = {},
        auxiliary: dict = {}
    ) -> "accera.Function":
        """Adds a function to the package.

        Args:
            source: The source which defines the function's implementation.
            args: The order of external-scope arrays to use in the function signature.
            base_name: A base name for the function. The full name for the function will be the
                base name followed by an automatically-generated unique identifier.
            parameters: A value for each parameter if the function's implementation is parameterized.
            function_opts: A dictionary of advanced options to set on the function, e.g. {"no_inline" : True}
            auxiliary: A dictionary of auxiliary metadata to include in the HAT package.
        """
        from .lang import LoopIndex

        # Auxiliary data should be one copy per function
        auxiliary_metadata = auxiliary.copy()
        param_value_dict = {}
        for delayed_param, value in parameters.items():
            delayed_param.set_value(value)
            if isinstance(value, int):
                param_value_dict[delayed_param._name] = value
            else:
                if isinstance(value, tuple) or isinstance(value, list):
                    if all(isinstance(v, LoopIndex) for v in value):
                        param_value_dict[delayed_param._name] = str(
                            [x._name for x in value]
                    )
                    else:
                        raise ValueError("Invalid value of parameters")
                else:
                    param_value_dict[delayed_param._name] = str(value)
        auxiliary_metadata["accera"] = {"parameters": param_value_dict}

        def validate_target(target: Target):
            # can't use set because targets are mutable (therefore unhashable)
            for f in self._fns.values():
                if not target.is_compatible_with(f.target):
                    raise NotImplementedError(
                        "Function target being added is currently incompatible with existing functions in package"
                    )

        def get_function_name(target: Target):
            # Get a function name using a stable hash of [base_name, signature, target, and parameters]
            # If no base_name is provided, use a unique identifier to avoid collisions (assume user
            # does not care about the function name in this case)
            # ref: https://death.andgravity.com/stable-hashing
            suffix = (
                md5(
                    json.dumps(
                        tuple(
                            map(
                                lambda x: str(x),
                                [
                                    base_name or token_hex(4),
                                    target,
                                    auxiliary_metadata["accera"],
                                ]
                                + [
                                    (a.role, a.element_type, a.shape, a.layout)
                                    if isinstance(a, lang.Array) else (a.name, a.type)
                                    if isinstance(a, _lang_python._lang.Dimension) else None
                                    for a in args 
                                ],
                            )
                        )
                    ).encode("utf-8")
                )
                .digest()
                .hex()[:16]
            )  # truncate

            # Function names must begin with an _ or alphabetical character
            return f"{base_name}_{suffix}" if base_name else f"_{suffix}"

        def compute_arg_size_references(args, SENTINEL_VALUE=-1):
            """Computes the arg position references for arrays with Dimension arg-sized shapes
            Uses -1 as a sentinel value indicating a statically-sized dimension of an Array or a Dimension in the arg list
            Given args containing Arrays and Dimensions, detects which dimensions are associated with which array
            E.g. given dynamic dimension N, static dimension M = 256, and arrays A { M x N }, B { N x N }, and C { M x M }
                  with args = (A, B, C, N), will compute the references:
                      ((-1, 3), (3, 3), (-1, -1), -1)
                  Because:
                   - A is {M x N}, and N is at arg position 3 in the args list and M is statically sized
                   - B is {N x N}, and N is at arg position 3 in the args list
                   - C is {M x M}, and M is statically sized
                   - N is not an array
            """
            arg_size_refs = []
            for arg in args:
                if isinstance(arg, lang.Array):
                    arr_dim_mappings = [args.index(dim) if isinstance(dim, _lang_python._lang.Dimension) else SENTINEL_VALUE for dim in arg.shape]
                    arg_size_refs.append(arr_dim_mappings)
                else:
                    arg_size_refs.append([SENTINEL_VALUE])
            return arg_size_refs


        # Resolve any undefined argument shapes based on the source usage pattern
        for arr in args:
            if isinstance(arr, lang.Array):
                _resolve_array_shape(source, arr)

        if isinstance(source, lang.Nest) or isinstance(source, lang.Schedule):
            # assumption: convenience functions are for host targets only
            source = source.create_plan(Target.HOST)
            # fall-through

        if isinstance(source, lang.Plan):
            self._dynamic_dependencies.update(source._dynamic_dependencies)
            source = source._create_function(
                args, **function_opts
            )
            # fall-through

        arg_names = [arg.name if isinstance(arg, lang.Array) or isinstance(arg, _lang_python._lang.Dimension) else "" for arg in args]
        arg_sizes = [arg._size_str if isinstance(arg, lang.Array) else "" for arg in args]

        if isinstance(source, lang.Function):
            source: lang.Function

            # due to the fall-through, we only need to validate here
            validate_target(source.target)

            native_array_dim_args = [arg._get_native_array() if isinstance(arg, lang.Array) else arg for arg in args ]

            source.name = get_function_name(source.target)
            source.base_name = base_name
            source.auxiliary = auxiliary_metadata
            source.param_overrides = parameters
            source.args = tuple(native_array_dim_args)
            source.arg_size_references = compute_arg_size_references(args)
            source.arg_names = arg_names
            source.arg_sizes = arg_sizes
            source.requested_args = args
            self._fns[source.name] = source
            return source  # for composability

        elif isinstance(source, Callable):

            # due to the fall-through, we only need to validate here
            validate_target(Target.HOST)

            @wraps(source)
            def wrapper_fn(args):
                source(*map(_convert_arg, args))

            name = get_function_name(Target.HOST)

            wrapped_func = lang.Function(
                name=name,
                base_name=base_name,
                args=tuple(map(_convert_arg, args)),
                arg_size_references=compute_arg_size_references(args),
                arg_names=arg_names,
                arg_sizes=arg_sizes,
                requested_args=args,
                definition=wrapper_fn,
                auxiliary=auxiliary_metadata,
                target=Target.HOST,
                **function_opts
            )

            self._fns[name] = wrapped_func
            return wrapped_func  # for composability

        else:
            raise ValueError("Invalid type for source")

    def _add_functions_to_module(self, module, fail_on_error=False):
        with SetActiveModule(module):
            to_pop = []
            for name, wrapped_func in self._fns.items():
                try:
                    wrapped_func._emit()
                except Exception as e:
                    to_pop.append(name)
                    logging.error(
                        f"Compiler error when trying to build function {name}"
                    )
                    logging.error(e)
                    if fail_on_error:
                        raise
                    else:
                        continue
            for name in to_pop:
                del self._fns[name]

    def _add_debug_utilities(self, tolerance):
        from .Debug import get_args_to_debug, add_debugging_functions

        # add_check_all_close will modify the self._fns dictionary (because
        # it is adding debug functions), to avoid this, we first gather information
        # about the functions to add
        fns_to_add = {
            name: (wrapped_func, get_args_to_debug(wrapped_func))
            for name, wrapped_func in self._fns.items()
        }

        # only add if there are actually arguments to debug
        return add_debugging_functions(
            self,
            {
                name: fn_and_args
                for name, fn_and_args in fns_to_add.items()
                if fn_and_args[1]
            },
            atol=tolerance,
        )

    def _generate_target_options(self, platform: Platform, mode: Mode = Mode.RELEASE):
        from .build_config import BuildConfig

        if len(self._fns) == 0:
            raise RuntimeError("No functions have been added")

        # target consistency is enforced during _add_function()
        target = list(self._fns.values())[0].target
        host_target_device = _lang_python._GetTargetDeviceFromName("host")

        if platform in [
            Package.Platform.HOST,
            Package.Platform.LINUX,
            Package.Platform.MACOS,
            Package.Platform.WINDOWS,
        ]:
            target_device = _lang_python._GetTargetDeviceFromName(platform.value)
        else:
            target_device = _lang_python.TargetDevice()

        # Architecture
        if target.architecture == Target.Architecture.HOST:
            target_device.architecture = host_target_device.architecture
            target_device.cpu = host_target_device.cpu
            target_device.features = host_target_device.features

        elif target.architecture == Target.Architecture.ARM:
            # All known targets that are ARM are supported completely
            target_device = _lang_python._GetTargetDeviceFromName(target._device_name)
            target_device.architecture = "arm"
            if "fpu" in target.extensions:
                target._device_name += "F"

        elif target.architecture == Target.Architecture.X86_64:
            target_device.architecture = "x86_64"

            if "AVX512" in target.extensions:
                target_device.device_name = "avx512"
                target_device.cpu = "skylake-avx512"
                # TODO: make this functionality less hidden
                avx512_feat_str = ",".join(
                    [f"+{feature.lower()}" for feature in target.extensions]
                )

                target_device.features = avx512_feat_str

        elif target.architecture == Target.Architecture.X86:
            target_device.architecture = "x86"

        _lang_python._CompleteTargetDevice(target_device)

        compiler_options = _lang_python.CompilerOptions()
        compiler_options.target_device = target_device
        compiler_options.debug = mode == Package.Mode.DEBUG
        compiler_options.gpu_only = (
            target.category == Target.Category.GPU and target.runtime != Runtime.VULKAN
        )

        BuildConfig.obj_extension = ".obj" if target_device.is_windows() else ".o"

        libs = list(
            filter(
                None,
                [
                    get_library_reference(dep, platform)
                    for dep in self._dynamic_dependencies
                ],
            )
        )
        return target, target_device, compiler_options, libs

    def _make_accc_options(self, options: _Options):
        from . import accc
        accc_opts = accc.Options.NONE
        if options & Package._Options.DISABLE_AUTO_UNROLL:
            accc_opts |= accc.Options.DISABLE_AUTO_UNROLL
        if options & Package._Options.HIGH_PRECISION_FLOATING_POINT_OPS:
            accc_opts |= accc.Options.HIGH_PRECISION_FLOATING_POINT_OPS
        return accc_opts


    def build(
        self,
        name: str,
        format: Format = Format.DEFAULT,
        mode: Mode = Mode.RELEASE,
        platform: Platform = Platform.HOST,
        tolerance: float = 1e-5,
        output_dir: str = None,
        fail_on_error: bool = False,
        _opts: _Options = _Options.NONE,
        _quiet=True,
    ):
        """Builds a HAT package.

        Args:
            name: The package name.
            format: The format of the package.
            mode: The package mode, such as whether it is optimized or used for debugging.
            platform: The platform where the package runs.
            tolerance: The tolerance for correctness checking when `mode = Package.Mode.DEBUG`.
            output_dir: The path to an output directory. Defaults to the current directory if unspecified.
        """

        from . import accc

        (
            target,
            target_device,
            compiler_options,
            dynamic_dependencies,
        ) = self._generate_target_options(platform, mode)

        if target.category == Target.Category.GPU:
            if target.runtime == Target.Runtime.NONE:
                raise ValueError("GPU targets must specify a runtime")
            if mode == Package.Mode.DEBUG:
                raise ValueError("GPU targets do not support Package.Mode.DEBUG")

        cross_compile = platform != Platform.HOST

        format_is_default = bool(
            format & Package.Format.DEFAULT
        )  # store it as a boolean because we're going to turn off the actual flag
        if format_is_default:
            format &= ~Package.Format.DEFAULT  # Turn off "DEFAULT"

            if target.runtime in [Target.Runtime.CUDA, Target.Runtime.ROCM]:
                format |= Package.Format.HAT_SOURCE
            else:
                format |= Package.Format.HAT_STATIC
                if not cross_compile:
                    format |= Package.Format.HAT_DYNAMIC

        dynamic_link = bool(format & Package.Format.DYNAMIC_LIBRARY)
        if cross_compile and dynamic_link:
            raise ValueError(
                "Package.Format.DYNAMIC_LIBRARY is not supported when cross-compiling"
            )

        output_dir = output_dir or os.getcwd()
        working_dir = os.path.join(output_dir, "_tmp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(working_dir, exist_ok=True)

        # Debug mode: add utility functions for checking results and mark target functions
        if mode == Package.Mode.DEBUG:
            debug_utilities = self._add_debug_utilities(tolerance)
            for fn_name, utilities in debug_utilities.items():
                self._fns[fn_name].output_verifiers = utilities

        # Create the package module
        package_module = _lang_python._Module(name=name, options=compiler_options)
        self._add_functions_to_module(package_module, fail_on_error)

        # Emit the package module
        if format & Package.Format.SOURCE:
            output_type = (
                accc.ModuleOutputType.CUDA
                if compiler_options.gpu_only
                else accc.ModuleOutputType.CPP
            )
        else:
            output_type = accc.ModuleOutputType.OBJECT

        # Emit the supporting modules
        supporting_hats = []
        if (
            not compiler_options.gpu_only
            and output_type == accc.ModuleOutputType.OBJECT
        ):
            supporting_hats.append(
                Package._emit_default_module(
                    compiler_options, target, mode, output_dir, f"{name}_Globals"
                )
            )
            if any(
                fn.target.category == Target.Category.GPU
                and fn.target.runtime == Target.Runtime.VULKAN
                for fn in self._fns.values()
            ):
                supporting_hats.append(
                    self._create_gpu_utility_module(
                        compiler_options, target, mode, output_dir
                    )
                )

        proj = accc.AcceraProject(
            output_dir=working_dir, library_name=name, output_type=output_type
        )
        proj.module_file_sets = [
            accc.ModuleFileSet(
                name=name, common_module_dir=working_dir, output_type=output_type
            )
        ]
        package_module.Save(proj.module_file_sets[0].generated_mlir_filepath)

        # Enable dumping of IR passes based on build format
        dump_ir = bool(format & (Package.Format.MLIR | Package.Format.MLIR_VERBOSE))
        dump_ir_verbose = bool(format & Package.Format.MLIR_VERBOSE)

        accc_options = self._make_accc_options(_opts)

        proj.generate_and_emit(
            build_config=mode.value,
            system_target=target_device.device_name,
            runtime=target.runtime.name,
            dump_all_passes=dump_ir,
            dump_intrapass_ir=dump_ir_verbose,
            gpu_only=compiler_options.gpu_only,
            quiet=_quiet,
            _options=accc_options
        )

        path_root = os.path.join(output_dir, name)
        extension = ".hat"

        if format & Package.Format.SOURCE:
            shutil.copy(proj.module_file_sets[0].translated_source_filepath, output_dir)

        if format & (Package.Format.DYNAMIC_LIBRARY | Package.Format.STATIC_LIBRARY):
            shutil.copy(proj.module_file_sets[0].object_filepath, output_dir)

        if format & Package.Format.HAT_PACKAGE:
            # Create initial HAT file containing shape and type metadata that the C++ layer has access to
            header_path = path_root + extension
            package_module.WriteHeader(header_path)

            # Complete the HAT file with information we have stored at this layer
            hat_file: hat.HATFile = hat.HATFile.Deserialize(header_path)

            if format & (
                Package.Format.DYNAMIC_LIBRARY | Package.Format.STATIC_LIBRARY
            ):
                hat_file.dependencies.link_target = os.path.basename(
                    proj.module_file_sets[0].object_filepath
                )

            supporting_hats = map(hat.HATFile.Deserialize, supporting_hats)
            supporting_objs = []
            supporting_decls = []
            for support in supporting_hats:
                path = os.path
                dependency_path = path.abspath(
                    path.join(output_dir, support.dependencies.link_target)
                )

                # Collect the supporting modules as dependencies
                supporting_objs.append(
                    hat.LibraryReference(target_file=dependency_path)
                )

                # Collecting the supporting code decls
                supporting_decls.append(support.declaration.code)

                # Merge the function maps
                hat_file._function_table.function_map.update(
                    support._function_table.function_map
                )

            decl_code = hat_file.declaration.code
            hat_file.dependencies.dynamic = dynamic_dependencies + supporting_objs
            hat_file.declaration.code = decl_code._new(
                "\n".join(map(str, ["", decl_code] + supporting_decls))
            )

            for fn_name in self._fns:
                fn: lang.Function = self._fns[fn_name]

                if fn.public:
                    hat_func = hat_file.function_map.get(fn_name)

                    if hat_func is None:
                        raise ValueError(
                            f"Couldn't find header-declared function {fn_name} in emitted HAT file"
                        )

                    hat_func.auxiliary = fn.auxiliary

                    if (
                        fn.target.category == Target.Category.GPU
                        and fn.target.runtime != Target.Runtime.VULKAN
                    ):
                        # TODO: Remove this when the header is emitted as part of the compilation
                        gpu_source = proj.module_file_sets[0].translated_source_filepath
                        gpu_device_func = fn_name + "__gpu__"
                        with open(gpu_source) as gpu_source_f:
                            s = re.search(
                                gpu_device_func + _R_GPU_LAUNCH, gpu_source_f.read()
                            )
                            if not s:
                                raise RuntimeError("Couldn't parse emitted source code")
                            launch_parameters = list(
                                map(int, [s[n] for n in range(1, 7)])
                            )
                            dynamic_shared_mem_bytes = int(s[7])
                        gpu_source = os.path.split(gpu_source)[1]

                        hat_target: hat.Target = hat_file.target
                        hat_target.required.gpu.runtime = fn.target.runtime.name
                        hat_target.required.gpu.model = fn.target.name

                        hat_func.runtime = fn.target.runtime.name
                        hat_func.launches = gpu_device_func

                        hat_file.device_function_map[gpu_device_func] = hat.Function(
                            name=gpu_device_func,
                            description=f"Device function launched by {fn_name}",
                            calling_convention=hat.CallingConventionType.Device,
                            arguments=hat_func.arguments,
                            return_info=hat_func.return_info,
                            launch_parameters=launch_parameters,
                            dynamic_shared_mem_bytes=dynamic_shared_mem_bytes,
                            provider=gpu_source,
                            runtime=fn.target.runtime.name,
                        )

            if target_device.is_windows():
                hat_os = hat.OperatingSystem.Windows
            elif target_device.is_macOS():
                hat_os = hat.OperatingSystem.MacOS
            elif target_device.is_linux():
                hat_os = hat.OperatingSystem.Linux
            hat_file.target.required.os = hat_os
            hat_file.target.required.cpu.architecture = target_device.architecture

            # Not all of these features are necessarily used in this module, however we don't currently have a way
            # of determining which are and are not used so to be safe we require all of them
            hat_file.target.required.cpu.extensions = target_device.features.split(",")

            hat_file.description.author = self._description.get("author", "")
            hat_file.description.version = self._description.get("version", "")
            hat_file.description.license_url = self._description.get("license", "")
            if "auxiliary" in self._description:
                hat_file.description.auxiliary = self._description["auxiliary"]

            hat_file.Serialize(header_path)

            if not cross_compile and (format & Package.Format.STATIC_LIBRARY):
                lib_hat_path = f"{path_root}_lib{extension}"
                hat.create_static_package(header_path, lib_hat_path)
                
                lib_hat_file = hat_file.Deserialize(lib_hat_path)
                lib_hat_file.dependencies.auxiliary["static"] = lib_hat_file.dependencies.link_target
                lib_hat_file.Serialize()
                
                shutil.move(lib_hat_path, header_path)

            if dynamic_link:
                dyn_hat_path = f"{path_root}_dyn{extension}"
                hat.create_dynamic_package(header_path, dyn_hat_path)

                dyn_hat_file = hat_file.Deserialize(dyn_hat_path)
                dyn_hat_file.dependencies.auxiliary["dynamic"] = dyn_hat_file.dependencies.link_target
                dyn_hat_file.Serialize()

                shutil.move(dyn_hat_path, header_path)
            
            # TODO: plumb cross-compilation of static libs

        return proj.module_file_sets

    def add_description(
        self,
        author: str = None,
        license: str = None,
        other: dict = {},
        version: str = None,
    ):
        """Adds descriptive metadata to the HAT package.

        Args:
            author: Name of the individual or group that authored the package.
            license: The internet URL of the license used to release the package.
            other: User-specific descriptive metadata.
                If the key already exists, the value will be overwritten
                To remove a key, set its value to None
            version: The package version.
        """
        if other:
            if "auxiliary" not in self._description:
                self._description["auxiliary"] = other
            else:
                self._description["auxiliary"].update(other)

            # remove any keys marked None
            keys_to_remove = [
                k for k, v in self._description["auxiliary"].items() if v is None
            ]
            for k in keys_to_remove:
                del self._description["auxiliary"][k]

        if version is not None:
            self._description["version"] = version
        if author is not None:
            self._description["author"] = author
        if license is not None:
            self._description["license"] = license

    @classmethod
    def _init_default_module(cls):
        # Creates a default module that is initialized once per import
        # This module will hold global data (such as constant Arrays), and will be
        # included in every built package
        cls._default_module = _lang_python._Module(name="PackageGlobals")
        _lang_python._SetActiveModule(cls._default_module)

    @classmethod
    def _emit_default_module(cls, compiler_options, target, mode, output_dir, name):
        # Specializes and then emits the default module
        cls._default_module.SetDataLayout(compiler_options)
        return _emit_module(cls._default_module, target, mode, output_dir, name)
