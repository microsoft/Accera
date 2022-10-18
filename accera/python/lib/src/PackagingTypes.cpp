////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

namespace py = pybind11;
namespace value = accera::value;
namespace util = accera::utilities;

using namespace pybind11::literals;

namespace accera::python::lang
{
namespace
{
    void DefinePackagingEnums(py::module& module)
    {
        py::enum_<value::FunctionParameterUsage>(module, "_FunctionParameterUsage", "An enumeration of function parameter usages")
            .value("INPUT_OUTPUT", value::FunctionParameterUsage::inputOutput)
            .value("INPUT", value::FunctionParameterUsage::input)
            .value("OUTPUT", value::FunctionParameterUsage::output);
    }

    void DefinePackagingStructs(py::module& module)
    {
        py::class_<value::TargetDevice>(module, "TargetDevice", "Properties of a target device")
            .def(py::init<>())
            .def_readwrite("device_name", &value::TargetDevice::deviceName)
            .def_readwrite("triple", &value::TargetDevice::triple)
            .def_readwrite("architecture", &value::TargetDevice::architecture)
            .def_readwrite("data_layout", &value::TargetDevice::dataLayout)
            .def_readwrite("cpu", &value::TargetDevice::cpu)
            .def_readwrite("features", &value::TargetDevice::features)
            .def_readwrite("num_bits", &value::TargetDevice::numBits)
            .def("has_feature", &value::TargetDevice::HasFeature, "feature"_a, R"pbdoc(
Helper function to test whether the TargetDevice has a particular feature.

If this is filled in by LLVM for the host target, the possible features are target dependent
and include, but are not limited to, the following:

X86: cx8, cmov, mmx, fxsr, sse, sse2, sse3, pclmul, ssse3, cx16, sse4.1, sse4.2, movbe, popcnt, aes, rdrnd,
avx, fma, xsave, f16c, sahf, lzcnt, sse4a, prfchw, xop, lwp, fma4, tbm, mwaitx, 64bit, clzero, wbnoinvd,
fsgsbase, sgx, bmi, avx2, bmi2, invpcid, rtm, avx512f, avx512dq, rdseed, adx, avx512ifma, clflushopt,
clwb, avx512pf, avx512er, avx512cd, sha, avx512bw, avx512vl, prefetchwt1, avx512vbmi, pku, waitpkg,
avx512vbmi2, shstk, gfni, vaes, vpclmulqdq, avx512vnni, avx512bitalg, avx512vpopcntdq, rdpid, cldemote,
movdiri, movdir64b, enqcmd, pconfig, avx512bf16, xsaveopt, xsavec, xsaves, ptwrite

AArch64: neon, fp-armv8, crc, crypto

ARM: fp16, neon, vfp3, d16, vfp4, hwdiv-arm, hwdiv
)pbdoc")
            .def("is_windows", &value::TargetDevice::IsWindows)
            .def("is_linux", &value::TargetDevice::IsLinux)
            .def("is_macOS", &value::TargetDevice::IsMacOS);

        py::class_<value::CompilerOptions>(module, "CompilerOptions", "Standard compiler switches.")
            .def(py::init<>())
            .def_readwrite("optimize", &value::CompilerOptions::optimize, "Optimize output code using LLVM. Defaults to True.")
            .def_readwrite("position_independent_code", &value::CompilerOptions::positionIndependentCode, "Generate position independent code (equivalent to -fPIC).")
            .def_readwrite("profile", &value::CompilerOptions::profile, "Emit profiling code.")
            .def_readwrite("use_fast_math", &value::CompilerOptions::useFastMath, "Allow emitting more efficient code that isn't necessarily IEEE-754 compatible. Defaults to True")
            .def_readwrite("include_diagnostic_info", &value::CompilerOptions::includeDiagnosticInfo, "Allow printing of diagnostic messages from the compiled model.")
            .def_readwrite("target_device", &value::CompilerOptions::targetDevice, "Name of the target device. Defaults to 'host'.")
            .def_readwrite("execution_runtime", &value::CompilerOptions::executionRuntime, "Target the specified runtime for execution.")
            .def_readwrite("use_blas", &value::CompilerOptions::useBlas, "Emit code that calls an external BLAS library. Defaults to True.")
            .def_readwrite("unroll_loops", &value::CompilerOptions::unrollLoops, "Explicitly unroll loops in certain cases.")
            .def_readwrite("inline_operators", &value::CompilerOptions::inlineOperators, "Emit inline code for common operations. Defaults to True.")
            .def_readwrite("allow_vector_instructions", &value::CompilerOptions::allowVectorInstructions, "Enable ELL's vectorization.")
            .def_readwrite("vector_width", &value::CompilerOptions::vectorWidth, "Size of vector units. Defaults to 4.")
            .def_readwrite("debug", &value::CompilerOptions::debug, "Emit debug code.")
            .def_readwrite("gpu_only", &value::CompilerOptions::gpu_only, "Emit only the GPU device code and do not emit the GPU host code.")
            // .def_readwrite("modelFile", &value::CompilerOptions::modelFile) // doesn't apply to accera
            .def_readwrite("global_value_alignment", &value::CompilerOptions::globalValueAlignment, "The byte alignment to use for global values. Defaults to 32.")
            .def_readwrite("use_bare_ptr_call_conv", &value::CompilerOptions::useBarePtrCallConv, "Whether to bare pointer style declarations for defined functions.")
            .def_readwrite("emit_c_wrapper_decls", &value::CompilerOptions::emitCWrapperDecls, "Whether to emit C wrapper declarations for defined functions. Defaults to True.")
            .def_readwrite("c_wrapper_prefix", &value::CompilerOptions::cWrapperPrefix, "The function name prefix to give to C wrapper declarations. Defaults to '_mlir_ciface_'");
    }

    void DefinePackagingFunctions(py::module& module, py::module& subModule)
    {
        module.def("_GetTargetDeviceFromName", &value::GetTargetDevice, "device_name"_a);
        module.def("_CompleteTargetDevice", &value::CompleteTargetDevice, "partial_device_info"_a);
        module.def("_GetKnownDeviceNames", &value::GetKnownDeviceNames);

        module.def("_DeclareFunction", &value::DeclareFunction, "name"_a);
        module.def("_ResolveConstantDataReference", &value::ResolveConstantDataReference);
        module.def(
            "_SetActiveModule", [](value::MLIRContext& ctx) {
                value::SetContext(ctx);
            },
            "module"_a);
        module.def(
            "_ClearActiveModule", [] {
                value::ClearContext();
            });

        subModule.def("GetTargetDevice", &value::GetContextTargetDevice);
        subModule.def("GetCompilerOptions", &value::GetContextCompilerOptions);
    }

    void DefineModuleClass(py::module& module)
    {
        py::class_<value::MLIRContext, std::unique_ptr<value::MLIRContext>>(module, "_Module", "A specialization of EmitterContext that emits MLIR IR.")
            .def(py::init<const std::string&, const value::CompilerOptions&>(), "name"_a, "options"_a = value::CompilerOptions{})
            .def(
                "Allocate",
                [](value::MLIRContext& c, value::ValueType type, const util::MemoryLayout& layout, size_t alignment) {
                    return c.Allocate(type, layout, alignment);
                },
                "type"_a,
                "layout"_a,
                "alignment"_a = 0)
            .def("Print", &value::MLIRContext::print, "Prints the module")
            .def("Save", &value::MLIRContext::save, "filename"_a)
            .def("Verify", &value::MLIRContext::verify)
            .def("WriteHeader", &value::MLIRContext::writeHeader, "filename"_a = std::nullopt)
            .def("SetMetadata", &value::MLIRContext::setMetadata)
            .def("GetFullMetadata", &value::MLIRContext::getFullMetadata)
            .def("SetDataLayout", &value::MLIRContext::setDataLayout);
    }

    void DefineFunctionClass(py::module& module)
    {
        py::class_<value::FunctionDeclaration>(module, "Function", "Describes a function that can be emitted")
            .def(py::init<std::string>(),
                 "name"_a,
                 R"pbdoc(
Constructor

Args:
    name: The name of the function
)pbdoc")
            .def("returns", &value::FunctionDeclaration::Returns, "return_type"_a, py::return_value_policy::reference_internal,
                 R"pbdoc(
Sets the return type for this function declaration

Args:
    return_type: A _Valor instance describing type of the value that is expected and
        its memory layout to be returned by the function
)pbdoc")
            .def("decorated", &value::FunctionDeclaration::Decorated, "should_decorate"_a, py::return_value_policy::reference_internal,
                 R"pbdoc(
Sets whether this function should be decorated (mangled)

Args:
    should_decorate: A bool value specifying whether this function should be decorated
)pbdoc")
            .def("public", &value::FunctionDeclaration::Public, "public"_a, py::return_value_policy::reference_internal, "If `public` is true, set the function to appear in the public header, otherwise the function is internal.")
            .def("external", &value::FunctionDeclaration::External, "external"_a, py::return_value_policy::reference_internal, "Sets whether the function declaration is an external declaration.")
            .def("cWrapper", &value::FunctionDeclaration::CWrapper, "cWrapper"_a, py::return_value_policy::reference_internal, "Sets whether an MLIR C wrapper function should be emitted for this function")
            .def("headerDecl", &value::FunctionDeclaration::HeaderDecl, "headerDecl"_a, py::return_value_policy::reference_internal, "Sets whether the function should be part of the generated header file.")
            .def("rawPointerAPI", &value::FunctionDeclaration::RawPointerAPI, "rawPointerAPI"_a, py::return_value_policy::reference_internal, "Sets whether the function should provide a raw pointer API.")
            .def(
                "inlinable", [](value::FunctionDeclaration& fn, bool inlinable) {
                    (void)fn.Inlined(inlinable ? value::FunctionInlining::always : value::FunctionInlining::never);
                    return fn;
                },
                "inlinable"_a,
                py::return_value_policy::reference_internal,
                "Sets whether the function is allowed to be inlined.")
            .def("addTag", &value::FunctionDeclaration::AddTag, "addTag"_a, py::return_value_policy::reference_internal, "A tag to add to a function as an attribute.")
            .def("baseName", &value::FunctionDeclaration::BaseName, "baseName"_a, py::return_value_policy::reference_internal, "Sets the base name for this function to use as an alias in the generated header file.")
            .def("outputVerifiers", &value::FunctionDeclaration::OutputVerifiers, "outputVerifiers"_a, py::return_value_policy::reference_internal, "Sets the verification functions for output checking, one per output argument.")
            .def(
                "define", [](value::FunctionDeclaration& fn, std::function<std::optional<value::Value>(std::vector<value::Value>)> defFn) -> value::FunctionDeclaration& {
                    (void)fn.Define(defFn);
                    return fn;
                },
                "func"_a,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Specifies a function definition for this declaration

Args:
    func: A callable that takes zero or more _Valor library observer types and returns void or a
        _Valor library observer type. This callable defines this function.

Remarks:
    If this function or `Imported` is not called, this function declaration is treated as an external function.
    Not all contexts may support an external function.
)pbdoc")
            .def(
                "parameters", [](value::FunctionDeclaration& fn, std::vector<value::ViewAdapter> types, std::optional<std::vector<value::FunctionParameterUsage>> usages, std::optional<std::vector<std::vector<int64_t>>> argSizeReferences) -> value::FunctionDeclaration& {
                    return fn.Parameters(types, usages, argSizeReferences);
                },
                "param_types"_a,
                "param_usages"_a = std::nullopt,
                "param_arg_size_references"_a = std::nullopt,
                py::return_value_policy::reference_internal,
                R"pbdoc(
Sets the parameters this function requires

Args:
    param_types: Zero or more _Valor instances or view types with a GetValue() member function describing
        the types of the arguments and their memory layout expected by the function
)pbdoc")
            .def(
                "__call__", [](const value::FunctionDeclaration& fn, std::vector<value::ViewAdapter> args) {
                    return fn.Call(args);
                },
                "arguments"_a)
            .def_property_readonly("is_defined", &value::FunctionDeclaration::IsDefined)
            .def_property_readonly("is_public", &value::FunctionDeclaration::IsPublic)
            .def_property_readonly("emits_header_decl", &value::FunctionDeclaration::EmitsHeaderDecl);
    }
} // namespace

void DefinePackagingTypes(py::module& module, py::module& subModule)
{
    DefinePackagingEnums(module);
    DefinePackagingStructs(module);
    DefinePackagingFunctions(module, subModule);
    DefineModuleClass(module);
    DefineFunctionClass(subModule);
}
} // namespace accera::python::lang
