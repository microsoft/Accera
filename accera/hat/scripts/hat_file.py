#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Requires: Python 3.7+
#
# Utility to parse the TOML metadata from HAT files
####################################################################################################

import argparse
import os
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

# Requires tomlkit: pip install tomlkit
import tomlkit

# TODO : type-checking on leaf node values

def _read_toml_file(filepath):
    path = os.path.abspath(filepath)
    toml_doc = None
    with open(path, "r") as f:
        file_contents = f.read()
        toml_doc = tomlkit.parse(file_contents)
    return toml_doc

def _check_required_table_entry(table, key):
    if key not in table:
        # TODO : add more context to this error message
        raise ValueError(f"Invalid HAT file: missing required key {key}")

def _check_required_table_entries(table, keys):
    for key in keys:
        _check_required_table_entry(table, key)


class ParameterType(Enum):
    AffineArray = "affine_array"
    RuntimeArray = "runtime_array"
    Element = "element"
    Void = "void"

class UsageType(Enum):
    Input = "input"
    Output = "output"
    InputOutput = "input_output"

class CallingConventionType(Enum):
    StdCall = "stdcall"
    CDecl = "cdecl"
    FastCall = "fastcall"
    VectorCall = "vectorcall"

class TargetType(Enum):
    CPU = "CPU"
    GPU = "GPU"

class OperatingSystem(Enum):
    Windows = "windows"
    MacOS = "macos"
    Linux = "linux"

@dataclass
class AuxiliarySupportedTable:
    AuxiliaryKey = "auxiliary"
    auxiliary: dict = field(default_factory=dict)

    def add_auxiliary_table(self, table):
        if len(self.auxiliary) > 0:
            table.add(self.AuxiliaryKey, self.auxiliary)

    @staticmethod
    def parse_auxiliary(table):
        if AuxiliarySupportedTable.AuxiliaryKey in table:
            return table[AuxiliarySupportedTable.AuxiliaryKey]
        else:
            return {}

@dataclass
class Description(AuxiliarySupportedTable):
    TableName: str = "description"
    comment: str = ""
    author: str = ""
    version: str = ""
    license_url: str = ""

    def to_table(self):
        description_table = tomlkit.table()
        description_table.add("author", self.author)
        description_table.add("version", self.version)
        description_table.add("license_url", self.license_url)

        self.add_auxiliary_table(description_table)

        return description_table

    @staticmethod
    def parse_from_table(table):
        return Description(author=table["author"],
                           version=table["version"],
                           license_url=table["license_url"],
                           auxiliary=AuxiliarySupportedTable.parse_auxiliary(table))

@dataclass
class Parameter:
    # All parameter keys
    name: str = ""
    description: str = ""
    logical_type: ParameterType = None
    declared_type: str = ""
    element_type: str = ""
    usage: UsageType = None

    # Affine array parameter keys
    shape: str = ""
    affine_map: list = field(default_factory=list)
    affine_offset: int = -1

    # Runtime array parameter keys
    size: str = ""

    def to_table(self):
        table = tomlkit.inline_table()
        table.append("name", self.name)
        table.append("description", self.description)
        table.append("logical_type", self.logical_type.value)
        table.append("declared_type", self.declared_type)
        table.append("element_type", self.element_type)
        table.append("usage", self.usage.value)

        if self.logical_type == ParameterType.AffineArray:
            table.append("shape", self.shape)
            table.append("affine_map", self.affine_map)
            table.append("affine_offset", self.affine_offset)
        elif self.logical_type == ParameterType.RuntimeArray:
            table.append("size", self.size)

        return table

    # TODO : change "usage" to "role" in schema
    @staticmethod
    def parse_from_table(param_table):
        required_table_entries = ["name", "description", "logical_type", "declared_type", "element_type", "usage"]
        _check_required_table_entries(param_table, required_table_entries)
        affine_array_required_table_entries = ["shape", "affine_map", "affine_offset"]
        runtime_array_required_table_entries = ["size"]

        name = param_table["name"]
        description = param_table["description"]
        logical_type = ParameterType(param_table["logical_type"])
        declared_type = param_table["declared_type"]
        element_type = param_table["element_type"]
        usage = UsageType(param_table["usage"])

        param = Parameter(name=name, description=description, logical_type=logical_type, declared_type=declared_type, element_type=element_type, usage=usage)
        if logical_type == ParameterType.AffineArray:
            _check_required_table_entries(param_table, affine_array_required_table_entries)
            param.shape = param_table["shape"]
            param.affine_map = param_table["affine_map"]
            param.affine_offset = param_table["affine_offset"]
        elif logical_type == ParameterType.RuntimeArray:
            _check_required_table_entries(param_table, runtime_array_required_table_entries)
            param.size = param_table["size"]

        return param


@dataclass
class Function(AuxiliarySupportedTable):
    name: str = ""
    description: str = ""
    calling_convention: CallingConventionType = None
    arguments: list = field(default_factory=list)
    return_info: Parameter = None
    hat_file: any = None
    link_target: Path = None

    def to_table(self):
        table = tomlkit.table()
        table.add("name", self.name)
        table.add("description", self.description)
        table.add("calling_convention", self.calling_convention.value)
        arg_tables = [arg.to_table() for arg in self.arguments]
        arg_array = tomlkit.array()
        for arg_table in arg_tables:
            arg_array.append(arg_table)
        table.add("arguments", arg_array) # TODO : figure out why this isn't indenting after serialization in some cases
        table.add("return", self.return_info.to_table())

        self.add_auxiliary_table(table)

        return table

    @staticmethod
    def parse_from_table(function_table):
        required_table_entries = ["name", "description", "calling_convention", "arguments", "return"]
        _check_required_table_entries(function_table, required_table_entries)
        arguments = [Parameter.parse_from_table(param_table) for param_table in function_table["arguments"]]
        return_info = Parameter.parse_from_table(function_table["return"])
        return Function(name=function_table["name"],
                        description=function_table["description"],
                        calling_convention=CallingConventionType(function_table["calling_convention"]),
                        arguments=arguments,
                        return_info=return_info,
                        auxiliary=AuxiliarySupportedTable.parse_auxiliary(function_table))


class FunctionTable:
    TableName = "functions"
    def __init__(self, function_map):
        self.function_map = function_map
        self.functions = self.function_map.values()

    def to_table(self):
        serialized_map = { function_key : self.function_map[function_key].to_table() for function_key in self.function_map }
        func_table = tomlkit.table()
        for function_key in self.function_map:
            func_table.add(function_key, self.function_map[function_key].to_table())
        return func_table

    @staticmethod
    def parse_from_table(all_functions_table):
        function_map = {function_key: Function.parse_from_table(all_functions_table[function_key]) for function_key in all_functions_table}
        return FunctionTable(function_map)


@dataclass
class Target:

    @dataclass
    class Required:

        @dataclass
        class CPU:
            TableName = TargetType.CPU.value
            architecture: str = ""
            extensions: list = field(default_factory=list)

            def to_table(self):
                table = tomlkit.table()
                table.add("architecture", self.architecture)
                table.add("extensions", self.extensions)
                return table

            @staticmethod
            def parse_from_table(table):
                required_table_entries = ["architecture", "extensions"]
                _check_required_table_entries(table, required_table_entries)
                return Target.Required.CPU(architecture=table["architecture"], extensions=table["extensions"])

        # TODO : support GPU
        class GPU:
            TableName = TargetType.CPU.value

            def to_table(self):
                return tomlkit.table()

            @staticmethod
            def parse_from_table(table):
                pass


        TableName = "required"
        os: OperatingSystem = None
        cpu: CPU = None
        gpu: GPU = None

        def to_table(self):
            table = tomlkit.table()
            table.add("os", self.os.value)
            table.add(Target.Required.CPU.TableName, self.cpu.to_table())
            if self.gpu is not None:
                table.add(Target.Required.GPU.TableName, self.gpu.to_table())
            return table

        @staticmethod
        def parse_from_table(table):
            required_table_entries = ["os", Target.Required.CPU.TableName]
            _check_required_table_entries(table, required_table_entries)
            cpu_info = Target.Required.CPU.parse_from_table(table[Target.Required.CPU.TableName])
            if Target.Required.GPU.TableName in table:
                gpu_info = Target.Required.GPU.parse_from_table(table[Target.Required.GPU.TableName])
            else:
                gpu_info = Target.Required.GPU()
            return Target.Required(os=table["os"], cpu=cpu_info, gpu=gpu_info)

    # TODO : support optimized_for table
    class OptimizedFor:
        TableName = "optimized_for"

        def to_table(self):
            return tomlkit.table()

        @staticmethod
        def parse_from_table(table):
            pass

    TableName = "target"
    required: Required = None
    optimized_for: OptimizedFor = None

    def to_table(self):
        table = tomlkit.table()
        table.add(Target.Required.TableName, self.required.to_table())
        if self.optimized_for is not None:
            table.add(Target.OptimizedFor.TableName, self.optimized_for.to_table())
        return table

    @staticmethod
    def parse_from_table(target_table):
        required_table_entries = [Target.Required.TableName]
        _check_required_table_entries(target_table, required_table_entries)
        required_data = Target.Required.parse_from_table(target_table[Target.Required.TableName])
        if Target.OptimizedFor.TableName in target_table:
            optimized_for_data = Target.OptimizedFor.parse_from_table(target_table[Target.OptimizedFor.TableName])
        else:
            optimized_for_data = Target.OptimizedFor()
        return Target(required=required_data, optimized_for=optimized_for_data)

@dataclass
class LibraryReference:
    name: str = ""
    version: str = ""
    target_file: str = ""

    def to_table(self):
        table = tomlkit.inline_table()
        table.append("name", self.name)
        table.append("version", self.version)
        table.append("target_file", self.target_file)
        return table

    @staticmethod
    def parse_from_table(table):
        return LibraryReference(name=table["name"],
                                   version=table["version"],
                                   target_file=table["target_file"])


@dataclass
class Dependencies(AuxiliarySupportedTable):
    TableName = "dependencies"
    link_target: str = ""
    deploy_files: list = field(default_factory=list)
    dynamic: list = field(default_factory=list)

    def to_table(self):
        table = tomlkit.table()
        table.add("link_target", self.link_target)
        table.add("deploy_files", self.deploy_files)

        dynamic_arr = tomlkit.array()
        for elt in self.dynamic:
            dynamic_arr.append(elt.to_table())
        table.add("dynamic", dynamic_arr)

        self.add_auxiliary_table(table)
        return table

    @staticmethod
    def parse_from_table(dependencies_table):
        required_table_entries = ["link_target", "deploy_files", "dynamic"]
        _check_required_table_entries(dependencies_table, required_table_entries)
        dynamic = [LibraryReference.parse_from_table(lib_ref_table) for lib_ref_table in dependencies_table["dynamic"]]
        return Dependencies(link_target=dependencies_table["link_target"],
                            deploy_files=dependencies_table["deploy_files"],
                            dynamic=dynamic,
                            auxiliary=AuxiliarySupportedTable.parse_auxiliary(dependencies_table))

@dataclass
class CompiledWith:
    TableName = "compiled_with"
    compiler: str = ""
    flags: str = ""
    crt: str = ""
    libraries: list = field(default_factory=list)

    def to_table(self):
        table = tomlkit.table()
        table.add("compiler", self.compiler)
        table.add("flags", self.flags)
        table.add("crt", self.crt)

        libraries_arr = tomlkit.array()
        for elt in self.libraries:
            libraries_arr.append(elt.to_table())
        table.add("libraries", libraries_arr)

        return table

    @staticmethod
    def parse_from_table(compiled_with_table):
        required_table_entries = ["compiler", "flags", "crt", "libraries"]
        _check_required_table_entries(compiled_with_table, required_table_entries)
        libraries = [LibraryReference.parse_from_table(lib_ref_table) for lib_ref_table in compiled_with_table["libraries"]]
        return CompiledWith(compiler=compiled_with_table["compiler"],
                            flags=compiled_with_table["flags"],
                            crt=compiled_with_table["crt"],
                            libraries=libraries)

@dataclass
class Declaration:
    TableName = "declaration"
    code: str = ""

    def to_table(self):
        table = tomlkit.table()
        table.add("code", self.code)
        return table

    @staticmethod
    def parse_from_table(declaration_table):
        required_table_entries = ["code"]
        _check_required_table_entries(declaration_table, required_table_entries)
        return Declaration(code=declaration_table["code"])

@dataclass
class HATFile:
    name: str = ""
    description: Description = None
    _function_table: FunctionTable = None
    functions: list = field(default_factory=list)
    function_map: dict = field(default_factory=dict)
    target: Target = None
    dependencies: Dependencies = None
    compiled_with: CompiledWith = None
    declaration: Declaration = None
    path: Path = None

    HATPrologue = "\n#ifndef __{0}__\n#define __{0}__\n\n#ifdef TOML\n"
    HATEpilogue = "\n#endif // TOML\n\n#endif // __{0}__"

    def __post_init__(self):
        self.functions = self._function_table.functions
        self.function_map = self._function_table.function_map
        for func in self.functions:
            func.hat_file = self
            func.link_target = self.path.parent / self.dependencies.link_target

    def Serialize(self, filepath=None):
        if filepath is None:
            filepath = self.path
        root_table = tomlkit.table()
        root_table.add(Description.TableName, self.description.to_table())
        root_table.add(FunctionTable.TableName, self._function_table.to_table())
        root_table.add(Target.TableName, self.target.to_table())
        root_table.add(Dependencies.TableName, self.dependencies.to_table())
        root_table.add(CompiledWith.TableName, self.compiled_with.to_table())
        root_table.add(Declaration.TableName, self.declaration.to_table())
        with open(filepath, "w") as out_file:
            out_file.write(self.HATPrologue.format(self.name))
            out_file.write(tomlkit.dumps(root_table))
            out_file.write(self.HATEpilogue.format(self.name))

    @staticmethod
    def Deserialize(filepath):
        hat_toml = _read_toml_file(filepath)
        name = os.path.splitext(os.path.basename(filepath))[0]
        required_entries = [Description.TableName,
                            FunctionTable.TableName,
                            Target.TableName,
                            Dependencies.TableName,
                            CompiledWith.TableName,
                            Declaration.TableName]
        _check_required_table_entries(hat_toml, required_entries)
        hat_file = HATFile(name=name,
                           description=Description.parse_from_table(hat_toml[Description.TableName]),
                           _function_table=FunctionTable.parse_from_table(hat_toml[FunctionTable.TableName]),
                           target=Target.parse_from_table(hat_toml[Target.TableName]),
                           dependencies=Dependencies.parse_from_table(hat_toml[Dependencies.TableName]),
                           compiled_with=CompiledWith.parse_from_table(hat_toml[CompiledWith.TableName]),
                           declaration=Declaration.parse_from_table(hat_toml[Declaration.TableName]),
                           path=Path(filepath).resolve())
        return hat_file
