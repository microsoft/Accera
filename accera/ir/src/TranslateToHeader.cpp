////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ir/include/TranslateToHeader.h"
#include "ir/include/IRUtil.h"
#include "ir/include/Metadata.h"
#include "ir/include/value/ValueDialect.h"

#include <llvm/IR/Type.h>
#include <value/include/FunctionDeclaration.h>
#include <value/include/LLVMUtilities.h>
#include <value/include/MLIREmitterContext.h>
#include <value/include/Debugging.h>

#include <utilities/include/Boolean.h>
#include <utilities/include/Exception.h>
#include <utilities/include/TypeTraits.h>

#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Target/LLVMIR/TypeToLLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Translation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>

// The TOML include needs to occur after LLVM includes due to some unfortunate #defines
#include "hat/include/HATEmitter.h"

using namespace accera::utilities;

namespace accera
{
namespace ir
{
    namespace
    {
        class LLVMTypeConverterDynMem: public mlir::LLVMTypeConverter
        {
        public:
            LLVMTypeConverterDynMem(MLIRContext *ctx, const mlir::LowerToLLVMOptions &options):
                mlir::LLVMTypeConverter(ctx, options)
            {}

            Type convertMemRefToBarePtr(mlir::BaseMemRefType type) {
                if (type.isa<mlir::UnrankedMemRefType>())
                    // Unranked memref is not supported in the bare pointer calling convention.
                    return {};

                Type elementType = convertType(type.getElementType());
                if (!elementType)
                    return {};
                return mlir::LLVM::LLVMPointerType::get(elementType, type.getMemorySpaceAsInt());
            }

            Type convertCallingConventionType(Type type) {
                if (getOptions().useBarePtrCallConv)
                    if (auto memrefTy = type.dyn_cast<mlir::BaseMemRefType>())
                        return convertMemRefToBarePtr(memrefTy);

                return convertType(type);
            }

            LogicalResult barePtrFuncArgTypeConverterDynMem(Type type,
                                                SmallVectorImpl<Type> &result) {
                auto llvmTy = convertCallingConventionType(type);
                if (!llvmTy)
                    return mlir::failure();

                result.push_back(llvmTy);
                return mlir::success();
            }

            Type convertFunctionSignature(
                FunctionType funcTy, bool isVariadic,
                LLVMTypeConverter::SignatureConversion &result) 
            {
                // Select the argument converter depending on the calling convention.
                if (getOptions().useBarePtrCallConv) {
                    for (auto &en : llvm::enumerate(funcTy.getInputs())) {
                        Type type = en.value();
                        llvm::SmallVector<Type, 8> converted;
                        if (failed(barePtrFuncArgTypeConverterDynMem(type, converted)))
                            return {};
                        result.addInputs(en.index(), converted);
                    }
                }
                else {
                    for (auto &en : llvm::enumerate(funcTy.getInputs())) {
                        Type type = en.value();
                        llvm::SmallVector<Type, 8> converted;
                        if (failed(mlir::structFuncArgTypeConverter(*this, type, converted)))
                            return {};
                        result.addInputs(en.index(), converted);
                    }
                }

                llvm::SmallVector<Type, 8> argTypes;
                argTypes.reserve(llvm::size(result.getConvertedTypes()));
                for (Type type : result.getConvertedTypes())
                    argTypes.push_back(type);

                // If function does not return anything, create the void result type,
                // if it returns on element, convert it, otherwise pack the result types into
                // a struct.
                Type resultType = funcTy.getNumResults() == 0
                                        ? mlir::LLVM::LLVMVoidType::get(&getContext())
                                        : packFunctionResults(funcTy.getResults());
                if (!resultType)
                    return {};
                return mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, isVariadic);
            }
        };

        value::ValueModuleOp GetValueModule(mlir::ModuleOp& module)
        {
            value::ValueModuleOp valueModuleOp;
            bool found = false;
            module.walk([&](value::ValueModuleOp op) {
                valueModuleOp = op;
                found = true;
            });
            if (!found)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Given a ModuleOp that doesn't contain a ValueModuleOp");
            }
            return valueModuleOp;
        }

        std::vector<value::ValueModuleOp> GetValueModules(std::vector<mlir::ModuleOp>& modules)
        {
            std::vector<value::ValueModuleOp> valueModuleOps;
            valueModuleOps.reserve(modules.size());
            for (auto& module : modules)
            {
                valueModuleOps.push_back(GetValueModule(module));
            }
            return valueModuleOps;
        }

        std::string GetHeaderPrologue(const std::string& libraryName)
        {
            std::ostringstream os;
            os << "//\n";
            os << "// Header for Accera library " << libraryName << "\n";
            os << "//\n\n";
            os << "#if defined(__cplusplus)\n";
            os << "#include <cstdint>\n";
            os << "#else\n";
            os << "#include <stdint.h>\n";
            os << "#include <stdbool.h>\n";
            os << "#endif // defined(__cplusplus)\n";

            // for float16_t
            os << "#if !defined(ACCERA_FLOAT)\n";
            os << "#define ACCERA_FLOAT 1\n";
            os << "typedef uint16_t float16_t;\n";
            os << "typedef uint16_t bfloat16_t;\n";
            os << "#endif // !defined(ACCERA_FLOAT)\n";

            os << "#if defined(__cplusplus)\n";
            os << "extern \"C\"\n";
            os << "{\n";
            os << "#endif // defined(__cplusplus)\n\n";

            os << "//\n// Functions\n//\n\n";
            return os.str();
        }

        std::string GetHeaderEpilogue()
        {
            std::ostringstream os;
            os << "#if defined(__cplusplus)\n";
            os << "} // extern \"C\"\n";
            os << "#endif // defined(__cplusplus)\n";
            return os.str();
        }

        std::string GetDebugPrologue()
        {
            std::ostringstream os;

            os << "#include <stdio.h>\n";
            os << "#include <stdarg.h>\n\n";

            os << "#ifndef _ACCERA_SYMBOL_EXPORT\n";
            os << "#ifdef _WIN32\n";
            os << "#define _ACCERA_SYMBOL_EXPORT __declspec(dllexport)\n";
            os << "#else\n";
            os << "#define _ACCERA_SYMBOL_EXPORT __attribute__((visibility(\"default\")))\n";
            os << "#endif // _WIN32\n";
            os << "#endif // _ACCERA_SYMBOL_EXPORT\n\n";

            return os.str();
        }

        std::string GetDebugCode()
        {
            std::ostringstream os;

            // FILE* stderr is platform-specific, define and export a function
            auto fnName = GetPrintErrorFunctionName();

            os << "#ifndef " << fnName << "_DEFINED_\n";
            os << "#define " << fnName << "_DEFINED_\n";
            os << "_ACCERA_SYMBOL_EXPORT int " << fnName << "(const char* fmt, ...) {\n";
            os << "    va_list args;\n";
            os << "    va_start(args, fmt);\n";
            os << "    int ret = vfprintf(stderr, fmt, args);\n";
            os << "    va_end(args);\n";
            os << "    return ret;\n";
            os << "}\n";
            os << "#endif // " << fnName << "_DEFINED_\n\n";

            return os.str();
        }

        bool DebugMode(std::vector<value::ValueModuleOp> valueModuleOps)
        {
            return std::any_of(valueModuleOps.begin(), valueModuleOps.end(), [](auto m) {
                return m->getAttr(accera::ir::GetDebugModeAttrName());
            });
        }

        struct LLVMType
        {
            mlir::Type type;

            // Optional source MLIR type that "type" was derived from
            // This contains additional info not available in the LLVM dialect, 
            // such as whether the type was unsigned (LLVM only supports signless types)
            std::optional<mlir::Type> source;
        };

        template <typename StreamType>
        void WriteLLVMType(StreamType& os, LLVMType t);

        template <typename StreamType>
        void WriteLLVMType(StreamType& os, mlir::Type t)
        {
            WriteLLVMType(os, { t, std::nullopt });
        }

        template <typename StreamType>
        void WriteStructType(StreamType& os, LLVMType t)
        {
            // ignore source MLIR type (there is no real equivalent)
            assert(t.type.isa<mlir::LLVM::LLVMStructType>());
            auto llvmTy = t.type.dyn_cast<mlir::LLVM::LLVMStructType>();
            os << "{ ";
            for (auto& it : llvm::enumerate(llvmTy.getBody()))
            {
                if (it.index() != 0)
                {
                    os << ", ";
                }
                WriteLLVMType(os, it.value());
            }
            os << " }";
        }

        template <typename StreamType>
        void WriteArrayType(StreamType& os, LLVMType t)
        {
            assert(t.type.isa<mlir::LLVM::LLVMArrayType>());
            auto llvmTy = t.type.dyn_cast<mlir::LLVM::LLVMArrayType>();
            auto size = llvmTy.getNumElements();
            auto elemType = llvmTy.getElementType();

            // use additional MLIR type information, if available
            if (t.source && t.source->isa<mlir::ShapedType>())
            {
                auto mlirTy = t.source->dyn_cast<mlir::ShapedType>();
                WriteLLVMType(os, { elemType, mlirTy.getElementType() });
            }
            else
            {
                WriteLLVMType(os, elemType);
            }

            os << "[" << size << "]";
        }

        template <typename StreamType>
        void WritePointerType(StreamType& os, LLVMType t)
        {
            assert(t.type.isa<mlir::LLVM::LLVMPointerType>());
            auto llvmTy = t.type.dyn_cast<mlir::LLVM::LLVMPointerType>();
            auto elemType = llvmTy.getElementType();

            // use additional MLIR type information, if available
            if (t.source && t.source->isa<mlir::ShapedType>())
            {
                auto mlirTy = t.source->dyn_cast<mlir::ShapedType>();
                WriteLLVMType(os, { elemType, mlirTy.getElementType() });
            }
            else
            {
                WriteLLVMType(os, elemType);
            }
            os << "*";
        }

        template <typename StreamType>
        void WriteIndexType(StreamType& os, LLVMType t)
        {
            auto type = t.type;
            auto size = type.getIntOrFloatBitWidth();
            if (type.isUnsignedInteger(size))
            {
                assert(size > 1); // booleans are signless currently
                os << "uint" << size << "_t";
                return;
            }
            else if (type.isInteger(size))
            {
                if (size > 1)
                {
                    os << "int" << size << "_t";
                }
                else
                {
                    os << "bool";
                }
                return;
            }
            assert(false && "Error: unsupported bit width");
        }

        template <typename StreamType>
        void WriteIntegerType(StreamType& os, LLVMType t)
        {
            // use additional MLIR type information, if available
            auto type = t.source.has_value() ? *t.source : t.type;
            auto size = type.getIntOrFloatBitWidth();
            if (type.isUnsignedInteger(size))
            {
                assert(size > 1); // booleans are signless currently
                os << "uint" << size << "_t";
                return;
            }
            else if (type.isInteger(size))
            {
                if (size > 1)
                {
                    os << "int" << size << "_t";
                }
                else
                {
                    os << "bool";
                }
                return;
            }
            assert(false && "Error: unsupported bit width");
        }

        template <typename StreamType>
        void WriteFloatType(StreamType& os, LLVMType t)
        {
            if (t.type.isF16())
            {
                os << "float16_t";
            }
            else if(t.type.isBF16())
            {
                os << "bfloat16_t";
            }
            else if (t.type.isF32())
            {
                os << "float";
            }
            else if (t.type.isF64())
            {
                os << "double";
            }
        }

        template <typename StreamType>
        void WriteFunctionType(StreamType& os, LLVMType t, std::optional<std::string> name = std::nullopt)
        {
            // returnType name(paramType, ...);

            auto fnTy = t.type.dyn_cast<mlir::LLVM::LLVMFunctionType>();
            assert(fnTy);

            std::optional<mlir::Type> sourceType;
            if (t.source) // use additional MLIR type information, if available
            {
                auto sourceFnTy = t.source->dyn_cast<mlir::FunctionType>();
                assert(sourceFnTy.getNumResults() <= 1);
                if (sourceFnTy.getNumResults() == 1)
                {
                    sourceType = sourceFnTy.getResult(0);
                }
            }
            LLVMType returnType = { fnTy.getReturnType(), sourceType };
            WriteLLVMType(os, returnType);

            os << ' ';
            if (name)
            {
                os << *name;
            }
            os << "(";
            auto numParams = fnTy.getNumParams();
            for (unsigned i = 0; i < numParams; ++i)
            {
                if (i != 0)
                {
                    os << ", ";
                }

                sourceType = std::nullopt;
                if (t.source) // use additional MLIR type information, if available
                {
                    auto sourceFnTy = t.source->dyn_cast<mlir::FunctionType>();
                    sourceType = sourceFnTy.getInput(i);
                }
                LLVMType paramType = { fnTy.getParamType(i), sourceType };
                WriteLLVMType(os, paramType);
            }
            os << ");";
        }

        template <typename StreamType>
        void WriteFunctionTypeAlias(StreamType& os, LLVMType t, std::string name, std::string baseName)
        {
            // returnType (*baseName)(paramType, ...) = name;

            auto fnTy = t.type.dyn_cast<mlir::LLVM::LLVMFunctionType>();
            assert(fnTy);

            os << "#ifndef __" << baseName << "_DEFINED__\n";
            os << "#define __" << baseName << "_DEFINED__\n";

            std::optional<mlir::Type> sourceType;
            if (t.source) // use additional MLIR type information, if available
            {
                auto sourceFnTy = t.source->dyn_cast<mlir::FunctionType>();
                assert(sourceFnTy.getNumResults() <= 1);
                if (sourceFnTy.getNumResults() == 1)
                {
                    sourceType = sourceFnTy.getResult(0);
                }
            }

            LLVMType returnType = { fnTy.getReturnType(), sourceType };
            WriteLLVMType(os, returnType);

            os << " (*" << baseName << ")(";
            auto numParams = fnTy.getNumParams();
            for (unsigned i = 0; i < numParams; ++i)
            {
                if (i != 0)
                {
                    os << ", ";
                }

                sourceType = std::nullopt;
                if (t.source) // use additional MLIR type information, if available
                {
                    auto sourceFnTy = t.source->dyn_cast<mlir::FunctionType>();
                    sourceType = sourceFnTy.getInput(i);
                }
 
                LLVMType paramType = { fnTy.getParamType(i), sourceType };
                WriteLLVMType(os, paramType);
           }
            os << ") = " << name << ";\n";
            os << "#endif\n";
        }

        template <typename StreamType>
        void WriteLLVMType(StreamType& os, LLVMType t)
        {
            using namespace mlir::LLVM;

            mlir::TypeSwitch<mlir::Type>(t.type)
                .Case([&](LLVMPointerType ptrTy) {
                    auto elementType = ptrTy.getElementType();

                    // std-to-llvm getVoidPtrType uses getInt8PtrTy, so follow that pattern
                    // (unless we are looking at a shaped MLIR type containing ui8/i8)
                    if ((
                            (!t.source ||
                             !t.source->isa<mlir::ShapedType>()) &&
                            elementType.isInteger(8)) ||
                        elementType.isa<LLVMStructType>())
                    {
                        os << "void*";
                    }
                    else
                    {
                        WritePointerType(os, t);
                    }
                })
                .Case([&](LLVMStructType) {
                    WriteStructType(os, t);
                })
                .Case([&](LLVMArrayType) {
                    WriteArrayType(os, t);
                })
                .Case([&](mlir::IntegerType) {
                    if (t.source && t.source->isa<mlir::IndexType>()) {
                        WriteIndexType(os, t);
                    }
                    else {
                        WriteIntegerType(os, t);
                    }
                })
                .Case([&](mlir::IndexType) {
                    WriteIntegerType(os, t);
                })
                .Case([&](mlir::FloatType) {
                    WriteFloatType(os, t);
                })
                .Case([&](LLVMVoidType) {
                    os << "void";
                })
                .Case([&](LLVMFunctionType) {
                    WriteFunctionType(os, t);
                })
                .Default([&](Type) {
                    os << "[[UNKNOWN]]";
                });
        }

        std::string GetLLVMTypeString(LLVMType t)
        {
            std::ostringstream os;
            WriteLLVMType(os, t);
            return os.str();
        }

        std::string GetLLVMElementTypeString(LLVMType t)
        {
            return mlir::TypeSwitch<mlir::Type, std::string>(t.type)
                .Case([&](mlir::LLVM::LLVMPointerType ptrTy) { 
                    assert(t.source && "MLIR type is required");
                    auto mlirType = mlir::TypeSwitch<mlir::Type, mlir::Type>(*t.source)
                        .Case([&](mlir::ShapedType shapedTy) {
                            return shapedTy.getElementType();
                        })
                        .Default([&](mlir::Type type) {
                            assert(false && "Error: unsupported MLIR type");
                            return type;                       
                        });

                    return GetLLVMElementTypeString({ ptrTy.getElementType(), mlirType });
                })
                .Default([&](mlir::Type) {
                    return GetLLVMTypeString(t);
                });
        }

        template <typename StreamType>
        mlir::LogicalResult WriteFunctionDeclaration(StreamType& os, value::ValueFuncOp fn, bool useBarePtrCallConv)
        {
            if (fn->hasAttr("external"))
            {
                // Don't write function declarations for extern decls
                return mlir::success();
            }
            auto context = fn.getContext();

            auto name = fn.getName().str();
            if (name.find(accera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix(), 0) == 0)
            {
                // Don't emit temporary functions in the header, these are used to sequence function pointers
                // created down the line
                return mlir::success();
            }

            auto fnType = fn.getType().dyn_cast<mlir::FunctionType>();
            assert(fnType.getNumResults() <= 1);

            mlir::LowerToLLVMOptions options(context);
            options.useBarePtrCallConv = useBarePtrCallConv;
            mlir::LLVMTypeConverter llvmTypeConverter(context, options);

            mlir::TypeConverter::SignatureConversion conversion(fnType.getNumInputs());
            auto llvmType = llvmTypeConverter.convertFunctionSignature(fnType, false, conversion);
            if (!llvmType)
            {
                LLVMTypeConverterDynMem llvmTypeConverterDynMem(context, options);
                mlir::TypeConverter::SignatureConversion conversionDynMem(fnType.getNumInputs());
                llvmType = llvmTypeConverterDynMem.convertFunctionSignature(fnType, false, conversionDynMem);
            }

            WriteFunctionType(os, { llvmType, fnType }, name);

            os << "\n\n";

            // if a base name is set, emit an alias for this function using the base name
            if (auto baseName = fn->getAttrOfType<mlir::StringAttr>(ir::BaseNameAttrName))
            {
                WriteFunctionTypeAlias(os, { llvmType, fnType }, name, baseName.getValue().str());
                os << "\n\n";
            }

            return mlir::success();
        }

        std::string MakeFunctionDeclaration(value::ValueFuncOp fn, bool useBarePtrCallConv)
        {
            std::ostringstream os;
            [[maybe_unused]] auto ok = WriteFunctionDeclaration(os, fn, useBarePtrCallConv);
            return os.str();
        }

        template <typename StreamType>
        mlir::LogicalResult WriteModuleHeader(StreamType& os,
                                              value::ValueModuleOp& module,
                                              bool useBarePtrCallConv)
        {
            // Write out function signatures
            module.walk([&os, useBarePtrCallConv](value::ValueFuncOp fn) {
                WriteFunctionDeclaration(os, fn, useBarePtrCallConv);
            });
            return mlir::success();
        }

        template <typename StreamType>
        mlir::LogicalResult WriteMultiModuleHeader(StreamType& os,
                                                   const std::string& name,
                                                   std::vector<mlir::ModuleOp>& modules,
                                                   bool useBarePtrCallConv)
        {
            auto valueModuleOps = GetValueModules(modules);
            auto debugMode = DebugMode(valueModuleOps);

            os << GetHeaderPrologue(name);

            if (debugMode)
            {
                os << GetDebugPrologue();
            }

            for (auto& module : valueModuleOps)
            {
                WriteModuleHeader(os, module, useBarePtrCallConv);
            }

            if (debugMode)
            {
                os << GetDebugCode();
            }

            os << GetHeaderEpilogue();
            return mlir::success();
        }

        std::unique_ptr<hat::Parameter> ConvertToIncompleteHATParameter(mlir::Type type, const std::string& runtimeSizeStr = "")
        {
            std::unique_ptr<hat::Parameter> param;
            if (type.isa<mlir::ShapedType>())
            {
                // mlir::ShapedType is either a mlir::MemRefType, a mlir::TensorType, or a VectorType
                // It's either an AffineArray or a RuntimeArray
                auto shapedType = type.cast<mlir::ShapedType>();
                if (shapedType.hasStaticShape())
                {
                    // It has a static shape (it is ranked and all dimensions have a known size) so it is an AffineArray
                    auto shape = shapedType.getShape();

                    std::vector<size_t> affineMap(shape.size());
                    size_t affineOffset = 0;
                    if (type.isa<mlir::MemRefType>())
                    {
                        auto memRefType = type.cast<mlir::MemRefType>();
                        llvm::SmallVector<int64_t, 4> strides;
                        int64_t offset = 0;
                        [[maybe_unused]] auto ok = mlir::getStridesAndOffset(memRefType, strides, offset);
                        std::transform(strides.begin(), strides.end(), affineMap.begin(), [](int64_t val) { return static_cast<size_t>(val); });
                        affineOffset = static_cast<size_t>(offset);
                    }
                    else
                    {
                        // MLIR doesn't encode stride information for tensors and vectors, so assume
                        // that they are dense and canonically ordered
                        // e.g. if it's a tensor<16 x 8 x 4 x f32>, assume the strides are [32, 4, 1]
                        size_t currentShardSize = 1;
                        for (size_t i = 0; i < shape.size(); ++i)
                        {
                            // Loop from back to front
                            size_t idx = shape.size() - i - 1;
                            affineMap[idx] = currentShardSize;
                            currentShardSize *= shape[idx];
                        }
                    }
                    auto affineArray = std::make_unique<hat::AffineArrayParameter>();
                    std::vector<size_t> shapeVec;
                    std::transform(shape.begin(), shape.end(), std::back_inserter(shapeVec), [](int64_t val) { return static_cast<size_t>(val); });
                    affineArray->Shape(shapeVec);
                    affineArray->AffineMap(affineMap);
                    affineArray->AffineOffset(affineOffset);
                    param = std::move(affineArray);
                }
                else
                {
                    // It has at least one unknown dimension so it is a RuntimeArray
                    auto runtimeArray = std::make_unique<hat::RuntimeArrayParameter>();
                    runtimeArray->Size(runtimeSizeStr);
                    param = std::move(runtimeArray);
                }
            }
            else
            {
                // ElementType is the only other option now
                param = std::make_unique<hat::ElementParameter>();
            }

            return param;
        }

        std::vector<hat::UsageType> GetHATParameterUsages(const value::ValueFuncOp& funcOp)
        {
            std::vector<hat::UsageType> hatUsages;
            if (auto usagesAttr = funcOp->getAttrOfType<mlir::ArrayAttr>(UsagesAttrName))
            {
                auto usagesVec = accera::ir::util::ConvertArrayAttrToIntVector(usagesAttr);
                std::transform(usagesVec.cbegin(), usagesVec.cend(), std::back_inserter(hatUsages), [](int usage) {
                    switch (static_cast<accera::value::FunctionParameterUsage>(usage))
                    {
                    case accera::value::FunctionParameterUsage::input:
                        return hat::UsageType::Input;
                    case accera::value::FunctionParameterUsage::output:
                        return hat::UsageType::Output;
                    case accera::value::FunctionParameterUsage::inputOutput:
                    default:
                        return hat::UsageType::InputOutput;
                    }
                });
            }
            return hatUsages;
        }

        mlir::LogicalResult WriteMultiModuleHATFile(llvm::raw_ostream& os,
                                                    const std::string& name,
                                                    std::vector<mlir::ModuleOp>& modules)
        {
            hat::Package package(name);

            // Set package description
            // TODO : plumb this info through
            package.Description.Comment(name);
            package.Description.Author("");
            package.Description.Version("");
            package.Description.LicenseURL("");

            // Add function data
            auto valueModuleOps = GetValueModules(modules);
            for (auto& module : valueModuleOps)
            {
                module.walk([&package](value::ValueFuncOp fn) {
                    // Only write function declarations for functions marked for header emission
                    if (fn->hasAttr(ir::HeaderDeclAttrName))
                    {
                        auto context = fn.getContext();
                        auto fnName = fn.getName().str();

                        auto fnType = fn.getType().dyn_cast<mlir::FunctionType>();
                        assert(fnType.getNumResults() <= 1);

                        bool useBarePtrCallConv = fn->hasAttr(ir::RawPointerAPIAttrName);

                        mlir::LowerToLLVMOptions options(context);
                        options.useBarePtrCallConv = useBarePtrCallConv;
                        options.emitCWrappers = false;
                        mlir::LLVMTypeConverter llvmTypeConverter(context, options);
                        
                        mlir::TypeConverter::SignatureConversion conversion(fnType.getNumInputs());
                        auto convertedType = llvmTypeConverter.convertFunctionSignature(fnType, /*isVariadic=*/false, conversion);
                        mlir::LLVM::LLVMFunctionType llvmType;

                        if (convertedType)
                            llvmType = convertedType.dyn_cast<mlir::LLVM::LLVMFunctionType>();
                        else 
                        {
                            LLVMTypeConverterDynMem llvmTypeConverterDynMem(context, options);
                            mlir::TypeConverter::SignatureConversion conversionDynMem(fnType.getNumInputs());
                            llvmType = llvmTypeConverterDynMem.convertFunctionSignature(fnType, /*isVariadic=*/false, conversionDynMem).dyn_cast<mlir::LLVM::LLVMFunctionType>();
                        }

                        auto function = std::make_unique<hat::Function>();
                        function->Name(fnName);
                        function->Description("");
                        function->CallingConvention(hat::CallingConventionType::CDecl); // TODO : plumb this through

                        auto usages = GetHATParameterUsages(fn);
                        auto numInputs = fnType.getNumInputs();
                        for (unsigned i = 0; i < numInputs; ++i)
                        {
                            // TODO : plumb name / description / etc through

                            // Get the logical type information from the MLIR standard dialect version of the function signature
                            // as the LLVM converted version will lose shape and signness information
                            const auto llvmArgType = llvmTypeConverter.convertType(llvmType.getParamType(i));
                            const auto mlirArgType = fnType.getInput(i);
                            std::unique_ptr<hat::Parameter> arg = ConvertToIncompleteHATParameter(mlirArgType); // TODO : plumb through size string
                            arg->Name(""); // TODO : plumb parameter name through
                            arg->Description(""); // TODO : plumb parameter description
                            arg->Usage(usages.size() == numInputs ? usages[i] : hat::UsageType::InputOutput);

                            auto declaredType = GetLLVMTypeString({ llvmArgType, mlirArgType }); // TODO : support for const
                            arg->DeclaredType(declaredType);

                            auto elementType = GetLLVMElementTypeString({ llvmArgType, mlirArgType });
                            arg->ElementType(elementType);

                            function->AddArgument(std::move(arg));
                        }

                        if (fnType.getNumResults() == 0)
                        {
                            // Void return
                            auto returnParam = std::make_unique<hat::VoidParameter>();
                            returnParam->Usage(hat::UsageType::Output);
                            function->Return(std::move(returnParam));
                        }
                        else
                        {
                            // TODO : plumb through size string
                            auto returnParam = ConvertToIncompleteHATParameter(fnType.getResult(0));
                            returnParam->Usage(hat::UsageType::Output);
                            function->Return(std::move(returnParam));
                        }

                        auto codeDecl = MakeFunctionDeclaration(fn, useBarePtrCallConv);
                        function->CodeDeclaration(codeDecl);

                        package.AddFunction(std::move(function));
                    }
                });
            }

            // TODO : plumb through target information or provide via accc
            package.Target.Required.OperatingSystem(hat::OperatingSystemType::Windows);
            package.Target.Required.CPU.Architecture("");
            package.Target.Required.CPU.Extensions({ "" });

            package.Target.OptimizedFor.CPU.Name("");
            package.Target.OptimizedFor.CPU.Family("");
            package.Target.OptimizedFor.CPU.ClockFrequency(0.0);
            package.Target.OptimizedFor.CPU.Cores(0);
            package.Target.OptimizedFor.CPU.Threads(0);
            package.Target.OptimizedFor.CPU.Cache.InstructionKB(0);
            package.Target.OptimizedFor.CPU.Cache.SizesKB({ 0 });
            package.Target.OptimizedFor.CPU.Cache.LineSizes({ 0 });

            // TODO : GPU

            // TODO : plumb through dependencies or provide via accc
            package.Dependencies.LinkTarget(""); // Link target needs to be filled in by outer layer that knows both how far the lowering goes (object files vs fully built library) to get the right file extension
            package.Dependencies.DeployFiles({});

            std::vector<hat::ExternalLibraryReference> dynamicDependencies;
            package.Dependencies.Dynamic(dynamicDependencies);

            // TODO : plumb through compiled with information or provide via accc
            package.CompiledWith.Compiler("");
            package.CompiledWith.Flags("");
            package.CompiledWith.CRuntime("");

            std::vector<hat::ExternalLibraryReference> staticLibsLinked;
            package.CompiledWith.Libraries(staticLibsLinked);

            package.CodePrologue(GetHeaderPrologue(name));
            package.CodeEpilogue(GetHeaderEpilogue());

            if (DebugMode(valueModuleOps))
            {
                package.CodePrologue(package.CodePrologue() + GetDebugPrologue()); // append to the prologue
                package.DebugCode(GetDebugCode());
            }

            os << package.Serialize();

            return mlir::success();
        }

        } // namespace

    mlir::LogicalResult TranslateToHeader(std::vector<mlir::ModuleOp>& modules, const std::string& libraryName, llvm::raw_ostream& os)
    {
        return WriteMultiModuleHATFile(os, libraryName, modules);
    }

    mlir::LogicalResult TranslateToHeader(mlir::ModuleOp module, llvm::raw_ostream& os)
    {
        std::string name = (module.getName()) ? module.getName()->str() : "MODULE";
        std::vector<mlir::ModuleOp> moduleVec{ module };
        return TranslateToHeader(moduleVec, name, os);
    }

    mlir::LogicalResult TranslateToHeader(mlir::ModuleOp module, std::ostream& os)
    {
        llvm::raw_os_ostream out(os);
        return TranslateToHeader(module, out);
    }

} // namespace ir
} // namespace accera

static mlir::TranslateFromMLIRRegistration
    registration("mlir-to-header", [](mlir::ModuleOp module, llvm::raw_ostream& output) {
        return accera::ir::TranslateToHeader(module, output);
    });
