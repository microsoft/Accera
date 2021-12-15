////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#include <toml++/toml.h>

// TODO : move to HAT repo

namespace hat
{

template <typename StreamType>
void EnableTOML(StreamType& os)
{
    os << "\n";
    os << "#ifdef __TOML__";
    os << "\n";
}

template <typename StreamType>
void DisableTOML(StreamType& os)
{
    os << "\n";
    os << "#endif // __TOML__";
    os << "\n";
}

// Guard for a region where the output stream will be used to print TOML code
template <typename StreamType>
struct TOMLEnableGuard
{
    TOMLEnableGuard(StreamType& os) :
        _os(os)
    {
        EnableTOML(_os);
    }

    ~TOMLEnableGuard()
    {
        DisableTOML(_os);
    }

    StreamType& _os;
};

// C-style header include guard
template <typename StreamType>
struct HeaderIncludeGuard
{
    HeaderIncludeGuard(StreamType& os, const std::string& name) :
        _os(os), _name(name)
    {
        _os << "\n";
        _os << "#ifndef __" << _name << "__\n";
        _os << "#define __" << _name << "__\n";
    }

    ~HeaderIncludeGuard()
    {
        _os << "\n";
        _os << "#endif // __" << _name << "__\n";
    }

    StreamType& _os;
    std::string _name;
};

// Guard for a region where the output stream will be used to print C++ code
template <typename StreamType>
struct TOMLDisableGuard
{
    TOMLDisableGuard(StreamType& os) :
        _os(os)
    {
        DisableTOML(_os);
    }

    ~TOMLDisableGuard()
    {
        EnableTOML(_os);
    }

    StreamType& _os;
};

enum class LogicalParamType
{
    Void,
    Element,
    AffineArray,
    RuntimeArray
};

inline std::string Serialize(const LogicalParamType& logicalType)
{
    switch (logicalType)
    {
    case LogicalParamType::Void:
        return "void";
    case LogicalParamType::Element:
        return "element";
    case LogicalParamType::AffineArray:
        return "affine_array";
    case LogicalParamType::RuntimeArray:
        return "runtime_array";
    default:
        return "[[UNKNOWN]]";
    }
}

enum class UsageType
{
    Input,
    Output,
    InputOutput
};

inline std::string Serialize(const UsageType& usageType)
{
    switch (usageType)
    {
    case UsageType::Input:
        return "input";
    case UsageType::Output:
        return "output";
    case UsageType::InputOutput:
        return "input_output";
    default:
        return "[[UNKNOWN]]";
    }
}

enum class CallingConventionType
{
    StdCall,
    CDecl,
    FastCall,
    VectorCall
};

inline std::string Serialize(const CallingConventionType& callingConvention)
{
    switch (callingConvention)
    {
    case CallingConventionType::StdCall:
        return "stdcall";
    case CallingConventionType::CDecl:
        return "cdecl";
    case CallingConventionType::FastCall:
        return "fastcall";
    case CallingConventionType::VectorCall:
        return "vectorcall";
    default:
        return "[[UNKNOWN]]";
    }
}

enum class OperatingSystemType
{
    Windows,
    MacOS,
    Linux
};

inline std::string Serialize(const OperatingSystemType& os)
{
    switch (os)
    {
    case OperatingSystemType::Windows:
        return "windows";
    case OperatingSystemType::MacOS:
        return "macos";
    case OperatingSystemType::Linux:
        return "linux";
    default:
        return "[[UNKNOWN]]";
    }
}

enum class GPURuntime
{
    CUDA,
    ROCM,
    Vulkan
};

inline std::string Serialize(const GPURuntime& os)
{
    switch (os)
    {
    case GPURuntime::CUDA:
        return "cuda";
    case GPURuntime::ROCM:
        return "rocm";
    case GPURuntime::Vulkan:
        return "vulkan";
    default:
        return "[[UNKNOWN]]";
    }
}

template <typename ElementType>
toml::array ConvertVectorToTOMLArray(const std::vector<ElementType>& vector)
{
    toml::array arr;
    for (const auto& element : vector)
    {
        arr.push_back(element);
    }
    return arr;
}

class TOMLSerializable
{
public:
    virtual toml::table Serialize() const = 0;
};

class OptionalTable
{
public:
    OptionalTable() :
        _isSet(false) {}
    void Set() { _isSet = true; }
    bool IsSet() const { return _isSet; }

private:
    bool _isSet;
};

// TODO : add auxiliary to more elements in HAT schema
template <typename Subclass>
class AuxiliaryExtensible
{
public:
    AuxiliaryExtensible() :
        _auxiliaryIsSet(false) {}

    Subclass& Auxiliary(const toml::table& auxiliary)
    {
        _auxiliaryIsSet = true;
        _auxiliary = auxiliary;
        return static_cast<Subclass>(*this);
    }

    toml::table Auxiliary() const
    {
        return _auxiliary;
    }

protected:
    void SerializeAuxiliary(toml::table& parentTable) const
    {
        if (_auxiliaryIsSet)
        {
            parentTable.insert_or_assign("auxiliary", _auxiliary);
        }
    }

private:
    bool _auxiliaryIsSet; // TODO : leverage OptionalTable inside AuxiliaryTable
    toml::table _auxiliary;
};

class Parameter : public TOMLSerializable
    , public AuxiliaryExtensible<Parameter>
{
public:
    Parameter() = delete; // Construct via subclasses

    Parameter& Name(const std::string& name)
    {
        _name = name;
        return *this;
    }
    std::string Name() const { return _name; }

    Parameter& Description(const std::string& description)
    {
        _description = description;
        return *this;
    }
    std::string Description() const { return _description; }

    LogicalParamType LogicalType() const { return _logicalType; }

    Parameter& DeclaredType(const std::string& declaredType)
    {
        _declaredType = declaredType;
        return *this;
    }
    std::string DeclaredType() const { return _declaredType; }

    Parameter& ElementType(const std::string& elementType)
    {
        _elementType = elementType;
        return *this;
    }
    std::string ElementType() const { return _elementType; }

    Parameter& Usage(const UsageType& usage)
    {
        _usage = usage;
        return *this;
    }
    UsageType Usage() const { return _usage; }

    virtual toml::table Serialize() const
    {
        return SerializeCommonParameters();
    }

protected:
    Parameter(const LogicalParamType& logicalType) :
        _logicalType(logicalType) {}

    toml::table SerializeCommonParameters() const
    {
        toml::table paramTable;
        paramTable.insert_or_assign("name", _name);
        paramTable.insert_or_assign("description", _description);
        paramTable.insert_or_assign("logical_type", hat::Serialize(_logicalType));
        paramTable.insert_or_assign("declared_type", _declaredType);
        paramTable.insert_or_assign("element_type", _elementType);
        paramTable.insert_or_assign("usage", hat::Serialize(_usage));
        paramTable.is_inline(true);
        SerializeAuxiliary(paramTable);
        return paramTable;
    }

private:
    std::string _name;
    std::string _description;
    LogicalParamType _logicalType;
    std::string _declaredType;
    std::string _elementType;
    UsageType _usage;
};

class AffineArrayParameter : public Parameter
{
public:
    AffineArrayParameter() :
        Parameter(LogicalParamType::AffineArray) {}

    AffineArrayParameter& Shape(const std::vector<size_t>& shape)
    {
        _shape = shape;
        return *this;
    }
    std::vector<size_t> Shape() const { return _shape; }

    AffineArrayParameter& AffineMap(const std::vector<size_t>& affineMap)
    {
        _affineMap = affineMap;
        return *this;
    }
    std::vector<size_t> AffineMap() const { return _affineMap; }

    AffineArrayParameter& AffineOffset(size_t affineOffset)
    {
        _affineOffset = affineOffset;
        return *this;
    }
    size_t AffineOffset() const { return _affineOffset; }

    toml::table Serialize() const
    {
        toml::table paramTable = SerializeCommonParameters();

        std::vector<int64_t> castShape;
        std::transform(_shape.begin(), _shape.end(), std::back_inserter(castShape), [](size_t val) { return static_cast<int64_t>(val); });
        toml::array shapeArr = ConvertVectorToTOMLArray(castShape);
        paramTable.insert_or_assign("shape", shapeArr);

        std::vector<int64_t> castMap;
        std::transform(_affineMap.begin(), _affineMap.end(), std::back_inserter(castMap), [](size_t val) { return static_cast<int64_t>(val); });
        toml::array affineMapArr = ConvertVectorToTOMLArray(castMap);
        paramTable.insert_or_assign("affine_map", affineMapArr);

        paramTable.insert_or_assign("affine_offset", static_cast<int64_t>(_affineOffset));

        return paramTable;
    }

private:
    std::vector<size_t> _shape;
    std::vector<size_t> _affineMap;
    size_t _affineOffset;
};

class RuntimeArrayParameter : public Parameter
{
public:
    RuntimeArrayParameter() :
        Parameter(LogicalParamType::RuntimeArray) {}

    RuntimeArrayParameter& Size(const std::string& size)
    {
        _size = size;
        return *this;
    }
    std::string Size() const { return _size; }

    toml::table Serialize() const
    {
        toml::table paramTable = SerializeCommonParameters();
        paramTable.insert_or_assign("size", _size);
        return paramTable;
    }

private:
    std::string _size;
};

class ElementParameter : public Parameter
{
public:
    ElementParameter() :
        Parameter(LogicalParamType::Element) {}
};

class VoidParameter : public Parameter
{
public:
    VoidParameter() :
        Parameter(LogicalParamType::Void)
    {
        DeclaredType("void");
        ElementType("void");
    }
};

class Function : public TOMLSerializable
    , public AuxiliaryExtensible<Function>
{
public:
    Function& Name(const std::string& name)
    {
        _name = name;
        return *this;
    }
    std::string Name() const { return _name; }

    Function& Description(const std::string& description)
    {
        _description = description;
        return *this;
    }
    std::string Description() const { return _description; }

    Function& CallingConvention(const CallingConventionType& callingConvention)
    {
        _callingConvention = callingConvention;
        return *this;
    }
    CallingConventionType CallingConvention() const { return _callingConvention; }

    Function& AddArgument(std::unique_ptr<Parameter>&& argument)
    {
        _arguments.push_back(std::move(argument));
        return *this;
    }
    Function& Arguments(std::vector<std::unique_ptr<Parameter>>&& arguments)
    {
        _arguments = std::move(arguments);
        return *this;
    }
    std::vector<Parameter*> Arguments() const
    {
        std::vector<Parameter*> argPtrs;
        std::transform(_arguments.begin(), _arguments.end(), std::back_inserter(argPtrs), [](const std::unique_ptr<Parameter>& argPtr) { return argPtr.get(); });
        return argPtrs;
    }

    Function& Return(std::unique_ptr<Parameter>&& returnInfo)
    {
        _return = std::move(returnInfo);
        return *this;
    }
    Parameter* Return() const { return _return.get(); }

    Function& CodeDeclaration(const std::string& codeDeclaration)
    {
        _codeDeclaration = codeDeclaration;
        return *this;
    }
    std::string CodeDeclaration() const { return _codeDeclaration; };

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("name", _name);
        table.insert_or_assign("description", _description);
        table.insert_or_assign("calling_convention", hat::Serialize(_callingConvention));

        toml::array argArr;
        auto orderedArgs = Arguments();
        for (const auto& arg : orderedArgs)
        {
            argArr.push_back(arg->Serialize());
        }
        table.insert_or_assign("arguments", argArr);

        table.insert_or_assign("return", _return->Serialize());

        SerializeAuxiliary(table);

        return table;
    }

private:
    std::string _name;
    std::string _description;
    CallingConventionType _callingConvention;
    std::vector<std::unique_ptr<Parameter>> _arguments;
    std::unique_ptr<Parameter> _return;
    std::string _codeDeclaration;
};

class PackageDescription : public TOMLSerializable
    , public AuxiliaryExtensible<PackageDescription>
{
public:
    PackageDescription& Comment(const std::string& comment)
    {
        _comment = comment;
        return *this;
    }
    std::string Comment() const { return _comment; }

    PackageDescription& Author(const std::string& author)
    {
        _author = author;
        return *this;
    }
    std::string Author() const { return _author; }

    PackageDescription& Version(const std::string& version)
    {
        _version = version;
        return *this;
    }
    std::string Version() const { return _version; }

    PackageDescription& LicenseURL(const std::string& licenseURL)
    {
        _licenseURL = licenseURL;
        return *this;
    }
    std::string LicenseURL() const { return _licenseURL; }

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("comment", _comment);
        table.insert_or_assign("author", _author);
        table.insert_or_assign("version", _version);
        table.insert_or_assign("license_url", _licenseURL);

        SerializeAuxiliary(table);

        return table;
    }

private:
    std::string _comment;
    std::string _author;
    std::string _version;
    std::string _licenseURL;
};

class RequiredCPUInfo : public TOMLSerializable
    , public AuxiliaryExtensible<RequiredCPUInfo>
{
public:
    RequiredCPUInfo& Architecture(const std::string& architecture)
    {
        _architecture = architecture;
        return *this;
    }
    std::string Architecture() const { return _architecture; }

    RequiredCPUInfo& Extensions(const std::vector<std::string>& extensions)
    {
        _extensions = extensions;
        return *this;
    }
    std::vector<std::string> Extensions() const { return _extensions; }

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("architecture", _architecture);
        table.insert_or_assign("extensions", ConvertVectorToTOMLArray(_extensions));

        SerializeAuxiliary(table);

        return table;
    }

private:
    std::string _architecture;
    std::vector<std::string> _extensions;
};

class RequiredGPUInfo : public TOMLSerializable
    , public AuxiliaryExtensible<RequiredCPUInfo>
    , public OptionalTable
{
public:
    RequiredGPUInfo& Runtime(const GPURuntime& runtime)
    {
        Set();
        _runtime = runtime;
        return *this;
    }
    GPURuntime Runtime() const { return _runtime; }

    RequiredGPUInfo& InstructionSetVersion(const std::string& instructionSetVersion)
    {
        Set();
        _instructionSetVersion = instructionSetVersion;
        return *this;
    }
    std::string InstructionSetVersion() const { return _instructionSetVersion; }

    RequiredGPUInfo& MinimumThreads(size_t minimumThreads)
    {
        Set();
        _minimumThreads = minimumThreads;
        return *this;
    }
    size_t MinimumThreads() const { return _minimumThreads; }

    RequiredGPUInfo& MinimumGlobalMemoryKB(size_t minimumGlobalMemoryKB)
    {
        Set();
        _minimumGlobalMemoryKB = minimumGlobalMemoryKB;
        return *this;
    }
    size_t MinimumGlobalMemoryKB() const { return _minimumGlobalMemoryKB; }

    RequiredGPUInfo& MinimumSharedMemoryKB(size_t minimumSharedMemoryKB)
    {
        Set();
        _minimumSharedMemoryKB = minimumSharedMemoryKB;
        return *this;
    }
    size_t MinimumSharedMemoryKB() const { return _minimumSharedMemoryKB; }

    RequiredGPUInfo& MinimumTextureMemoryKB(size_t minimumTextureMemoryKB)
    {
        Set();
        _minimumTextureMemoryKB = minimumTextureMemoryKB;
        return *this;
    }
    size_t MinimumTextureMemoryKB() const { return _minimumTextureMemoryKB; }

    toml::table Serialize() const
    {
        toml::table table;

        if (IsSet())
        {
            table.insert_or_assign("runtime", hat::Serialize(_runtime));
            table.insert_or_assign("instruction_set_version", _instructionSetVersion);
            table.insert_or_assign("min_threads", static_cast<int64_t>(_minimumThreads));
            table.insert_or_assign("min_global_memory_KB", static_cast<int64_t>(_minimumGlobalMemoryKB));
            table.insert_or_assign("min_shared_memory_KB", static_cast<int64_t>(_minimumSharedMemoryKB));
            table.insert_or_assign("min_texture_memory_KB", static_cast<int64_t>(_minimumTextureMemoryKB));

            SerializeAuxiliary(table);
        }

        return table;
    }

private:
    GPURuntime _runtime;
    std::string _instructionSetVersion;
    size_t _minimumThreads;
    size_t _minimumGlobalMemoryKB;
    size_t _minimumSharedMemoryKB;
    size_t _minimumTextureMemoryKB;
};

// TODO : finalize HW ID vs characteristics
class TargetRequiredInfo : public TOMLSerializable
{
public:
    TargetRequiredInfo& OperatingSystem(const OperatingSystemType& operatingSystem)
    {
        _operatingSystem = operatingSystem;
        return *this;
    }
    OperatingSystemType OperatingSystem() const { return _operatingSystem; }

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("CPU", CPU.Serialize());
        if (GPU.IsSet())
        {
            table.insert_or_assign("GPU", GPU.Serialize());
        }
        table.insert_or_assign("os", hat::Serialize(_operatingSystem));

        return table;
    }

    RequiredCPUInfo CPU;
    RequiredGPUInfo GPU;

private:
    OperatingSystemType _operatingSystem;
};

class CPUCacheInfo : public TOMLSerializable
    , public AuxiliaryExtensible<RequiredCPUInfo>
    , public OptionalTable
{
public:
    CPUCacheInfo& InstructionKB(size_t instructionKB)
    {
        Set();
        _instructionKB = instructionKB;
        return *this;
    }
    size_t InstructionKB() const { return _instructionKB; }

    CPUCacheInfo& SizesKB(const std::vector<size_t>& sizesKB)
    {
        Set();
        _sizesKB = sizesKB;
        return *this;
    }
    std::vector<size_t> SizesKB() const { return _sizesKB; }

    CPUCacheInfo& LineSizes(const std::vector<size_t>& lineSizes)
    {
        Set();
        _lineSizes = lineSizes;
        return *this;
    }
    std::vector<size_t> LineSizes() const { return _lineSizes; }

    toml::table Serialize() const
    {
        toml::table table;

        if (IsSet())
        {
            table.insert_or_assign("instruction_KB", static_cast<int64_t>(_instructionKB));

            std::vector<int64_t> castSizesKB;
            std::transform(_sizesKB.begin(), _sizesKB.end(), std::back_inserter(castSizesKB), [](size_t val) { return static_cast<int64_t>(val); });
            toml::array sizesKBArr = ConvertVectorToTOMLArray(castSizesKB);
            table.insert_or_assign("sizes_KB", sizesKBArr);

            std::vector<int64_t> castLineSizesKB;
            std::transform(_lineSizes.begin(), _lineSizes.end(), std::back_inserter(castLineSizesKB), [](size_t val) { return static_cast<int64_t>(val); });
            toml::array lineSizesArr = ConvertVectorToTOMLArray(castLineSizesKB);
            table.insert_or_assign("line_sizes", lineSizesArr);

            SerializeAuxiliary(table);
        }

        return table;
    }

private:
    size_t _instructionKB;
    std::vector<size_t> _sizesKB;
    std::vector<size_t> _lineSizes;
};

// TODO : finalize HW ID vs characteristics
class OptimizedForCPUInfo : public TOMLSerializable
    , public AuxiliaryExtensible<OptimizedForCPUInfo>
    , public OptionalTable
{
public:
    CPUCacheInfo Cache;

    OptimizedForCPUInfo& Name(const std::string& name)
    {
        Set();
        _name = name;
        return *this;
    }
    std::string Name() const { return _name; }

    OptimizedForCPUInfo& Family(const std::string& family)
    {
        Set();
        _family = family;
        return *this;
    }
    std::string Family() const { return _family; }

    OptimizedForCPUInfo& ClockFrequency(double clockFrequency)
    {
        Set();
        _clockFrequency = clockFrequency;
        return *this;
    }
    double ClockFrequency() const { return _clockFrequency; }

    OptimizedForCPUInfo& Cores(size_t cores)
    {
        Set();
        _cores = cores;
        return *this;
    }
    size_t Cores() const { return _cores; }

    OptimizedForCPUInfo& Threads(size_t threads)
    {
        Set();
        _threads = threads;
        return *this;
    }
    size_t Threads() const { return _threads; }

    toml::table Serialize() const
    {
        toml::table table;

        if (IsSet())
        {
            table.insert_or_assign("name", _name);
            table.insert_or_assign("family", _family);
            table.insert_or_assign("clock_frequency", _clockFrequency);
            table.insert_or_assign("cores", static_cast<int64_t>(_cores));
            table.insert_or_assign("threads", static_cast<int64_t>(_threads));

            if (Cache.IsSet())
            {
                table.insert_or_assign("cache", Cache.Serialize());
            }

            SerializeAuxiliary(table);
        }

        return table;
    }

private:
    std::string _name;
    std::string _family;
    double _clockFrequency;
    size_t _cores;
    size_t _threads;
};

class OptimizedForGPUInfo : public TOMLSerializable
    , public AuxiliaryExtensible<OptimizedForCPUInfo>
    , public OptionalTable
{
public:
    OptimizedForGPUInfo& Name(const std::string& name)
    {
        Set();
        _name = name;
        return *this;
    }
    std::string Name() const { return _name; }

    OptimizedForGPUInfo& Family(const std::string& family)
    {
        Set();
        _family = family;
        return *this;
    }
    std::string Family() const { return _family; }

    OptimizedForGPUInfo& ClockFrequency(double clockFrequency)
    {
        Set();
        _clockFrequency = clockFrequency;
        return *this;
    }
    double ClockFrequency() const { return _clockFrequency; }

    OptimizedForGPUInfo& Cores(size_t cores)
    {
        Set();
        _cores = cores;
        return *this;
    }
    size_t Cores() const { return _cores; }

    OptimizedForGPUInfo& Threads(size_t threads)
    {
        Set();
        _threads = threads;
        return *this;
    }
    size_t Threads() const { return _threads; }

    OptimizedForGPUInfo& InstructionSetVersion(const std::string& instructionSetVersion)
    {
        Set();
        _instructionSetVersion = instructionSetVersion;
        return *this;
    }
    std::string InstructionSetVersion() const { return _instructionSetVersion; }

    OptimizedForGPUInfo& GlobalMemoryKB(size_t globalMemoryKB)
    {
        Set();
        _globalMemoryKB = globalMemoryKB;
        return *this;
    }
    size_t GlobalMemoryKB() const { return _globalMemoryKB; }

    OptimizedForGPUInfo& SharedMemoryKB(size_t sharedMemoryKB)
    {
        Set();
        _sharedMemoryKB = sharedMemoryKB;
        return *this;
    }
    size_t SharedMemoryKB() const { return _sharedMemoryKB; }

    OptimizedForGPUInfo& TextureMemoryKB(size_t textureMemoryKB)
    {
        Set();
        _textureMemoryKB = textureMemoryKB;
        return *this;
    }
    size_t TextureMemoryKB() const { return _textureMemoryKB; }

    OptimizedForGPUInfo& SharedMemoryLineSize(size_t sharedMemoryLineSize)
    {
        Set();
        _sharedMemoryLineSize = sharedMemoryLineSize;
        return *this;
    }
    size_t SharedMemoryLineSize() const { return _sharedMemoryLineSize; }

    toml::table Serialize() const
    {
        toml::table table;

        if (IsSet())
        {
            table.insert_or_assign("name", _name);
            table.insert_or_assign("family", _family);
            table.insert_or_assign("clock_frequency", _clockFrequency);
            table.insert_or_assign("cores", static_cast<int64_t>(_cores));
            table.insert_or_assign("threads", static_cast<int64_t>(_threads));
            table.insert_or_assign("instruction_set_version", _instructionSetVersion);
            table.insert_or_assign("global_memory_KB", static_cast<int64_t>(_globalMemoryKB));
            table.insert_or_assign("shared_memory_KB", static_cast<int64_t>(_sharedMemoryKB));
            table.insert_or_assign("shared_memory_line_size", static_cast<int64_t>(_sharedMemoryLineSize));
            table.insert_or_assign("texture_memory_KB", static_cast<int64_t>(_textureMemoryKB));

            SerializeAuxiliary(table);
        }

        return table;
    }

private:
    std::string _name;
    std::string _family;
    double _clockFrequency;
    size_t _cores;
    size_t _threads;
    std::string _instructionSetVersion;
    size_t _globalMemoryKB;
    size_t _sharedMemoryKB;
    size_t _textureMemoryKB;
    size_t _sharedMemoryLineSize;
};

class TargetOptimizedForInfo : public TOMLSerializable
    , public AuxiliaryExtensible<TargetOptimizedForInfo>
{
public:
    OptimizedForCPUInfo CPU;
    OptimizedForGPUInfo GPU;

    toml::table Serialize() const
    {
        toml::table table;

        if (CPU.IsSet())
        {
            table.insert_or_assign("CPU", CPU.Serialize());
        }
        if (GPU.IsSet())
        {
            table.insert_or_assign("GPU", GPU.Serialize());
        }

        return table;
    }
};

class TargetInfo : public TOMLSerializable
{
public:
    TargetRequiredInfo Required;
    TargetOptimizedForInfo OptimizedFor;

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("required", Required.Serialize());
        table.insert_or_assign("optimized_for", OptimizedFor.Serialize());

        return table;
    }
};

class ExternalLibraryReference : public TOMLSerializable
{
public:
    ExternalLibraryReference& Name(const std::string& name)
    {
        _name = name;
        return *this;
    }
    std::string Name() const { return _name; }

    ExternalLibraryReference& Version(const std::string& version)
    {
        _version = version;
        return *this;
    }
    std::string Version() const { return _version; }

    ExternalLibraryReference& TargetFile(const std::string& targetFile)
    {
        _targetFile = targetFile;
        return *this;
    }
    std::string TargetFile() const { return _targetFile; }

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("name", _name);
        table.insert_or_assign("version", _version);
        table.insert_or_assign("target_file", _targetFile);
        table.is_inline(true);

        return table;
    }

private:
    std::string _name;
    std::string _version;
    std::string _targetFile;
};

class DependenciesInfo : public TOMLSerializable
    , public AuxiliaryExtensible<DependenciesInfo>
{
public:
    DependenciesInfo& LinkTarget(const std::string& linkTarget)
    {
        _linkTarget = linkTarget;
        return *this;
    }
    std::string LinkTarget() const { return _linkTarget; }

    DependenciesInfo& DeployFiles(const std::vector<std::string>& deployFiles)
    {
        _deployFiles = deployFiles;
        return *this;
    }
    std::vector<std::string> DeployFiles() const { return _deployFiles; }

    DependenciesInfo& Dynamic(const std::vector<ExternalLibraryReference>& dynamic)
    {
        _dynamic = dynamic;
        return *this;
    }
    std::vector<ExternalLibraryReference> Dynamic() const { return _dynamic; }

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("link_target", _linkTarget);

        toml::array deployFilesArr = ConvertVectorToTOMLArray(_deployFiles);
        table.insert_or_assign("deploy_files", deployFilesArr);

        toml::array dynamicArr;
        for (const auto& dynamicRef : _dynamic)
        {
            dynamicArr.push_back(dynamicRef.Serialize());
        }
        table.insert_or_assign("dynamic", dynamicArr);

        SerializeAuxiliary(table);

        return table;
    }

private:
    std::string _linkTarget;
    std::vector<std::string> _deployFiles;
    std::vector<ExternalLibraryReference> _dynamic;
};

class CompiledWithInfo : public TOMLSerializable
    , public AuxiliaryExtensible<CompiledWithInfo>
{
public:
    CompiledWithInfo Compiler(const std::string& compiler)
    {
        _compiler = compiler;
        return *this;
    }
    std::string Compiler() const { return _compiler; }

    CompiledWithInfo Flags(const std::string& flags)
    {
        _flags = flags;
        return *this;
    }
    std::string Flags() const { return _flags; }

    CompiledWithInfo CRuntime(const std::string& cRuntime)
    {
        _cRuntime = cRuntime;
        return *this;
    }
    std::string CRuntime() const { return _cRuntime; }

    CompiledWithInfo Libraries(const std::vector<ExternalLibraryReference>& libraries)
    {
        _libraries = libraries;
        return *this;
    }
    std::vector<ExternalLibraryReference> Libraries() const { return _libraries; }

    toml::table Serialize() const
    {
        toml::table table;

        table.insert_or_assign("compiler", _compiler);
        table.insert_or_assign("flags", _flags);
        table.insert_or_assign("crt", _cRuntime);

        toml::array libArr;
        for (const auto& lib : _libraries)
        {
            libArr.push_back(lib.Serialize());
        }
        table.insert_or_assign("libraries", libArr);

        SerializeAuxiliary(table);

        return table;
    }

private:
    std::string _compiler;
    std::string _flags;
    std::string _cRuntime;
    std::vector<ExternalLibraryReference> _libraries;
};

class Package : public AuxiliaryExtensible<Package>
{
public:
    PackageDescription Description;
    TargetInfo Target;
    DependenciesInfo Dependencies;
    CompiledWithInfo CompiledWith;

    Package(const std::string& name) :
        _name(name) {}

    Package& AddFunction(std::unique_ptr<Function>&& function)
    {
        _functions.push_back(std::move(function));
        return *this;
    }

    Package& Functions(std::vector<std::unique_ptr<Function>>&& functions)
    {
        _functions = std::move(functions);
        return *this;
    }
    std::vector<Function*> Functions() const
    {
        std::vector<Function*> functionPtrs;
        std::transform(_functions.begin(), _functions.end(), std::back_inserter(functionPtrs), [](const std::unique_ptr<Function>& funcPtr) { return funcPtr.get(); });
        return functionPtrs;
    }

    Package& CodePrologue(const std::string& prologue)
    {
        _codePrologue = prologue;
        return *this;
    }
    std::string CodePrologue() const { return _codePrologue; }

    Package& CodeEpilogue(const std::string& epilogue)
    {
        _codeEpilogue = epilogue;
        return *this;
    }
    std::string CodeEpilogue() const { return _codeEpilogue; }

    Package& DebugCode(const std::string& info)
    {
        _debugCode = info;
        return *this;
    }
    std::string DebugCode() const { return _debugCode; }

    std::string Serialize() const
    {
        // By default, the toml utility library will serialize a table in lexical order,
        // so to have a particular user-friendly order, serialize different subtables into
        // an output stream individually

        // The toml utility library only supports outputing to std::ostringstream
        std::ostringstream serializationOs;
        {
            HeaderIncludeGuard includeGuard(serializationOs, _name);
            {
                TOMLEnableGuard enableTOML(serializationOs);

                toml::table descriptionTable = Description.Serialize();
                toml::table descriptionTableWrapper;
                descriptionTableWrapper.insert_or_assign("description", descriptionTable);
                serializationOs << descriptionTableWrapper << "\n\n";

                toml::table functionTable;
                std::vector<std::string> functionDeclStrings;
                for (const auto& func : _functions)
                {
                    functionTable.insert_or_assign(func->Name(), func->Serialize());
                    functionDeclStrings.push_back(func->CodeDeclaration());
                }
                toml::table functionTableWrapper;
                functionTableWrapper.insert_or_assign("functions", functionTable);
                serializationOs << functionTableWrapper << "\n\n";

                toml::table targetTable = Target.Serialize();
                toml::table targetTableWrapper;
                targetTableWrapper.insert_or_assign("target", targetTable);
                serializationOs << targetTableWrapper << "\n\n";

                toml::table dependenciesTable = Dependencies.Serialize();
                toml::table dependenciesTableWrapper;
                dependenciesTableWrapper.insert_or_assign("dependencies", dependenciesTable);
                serializationOs << dependenciesTableWrapper << "\n\n";

                toml::table compiledWithTable = CompiledWith.Serialize();
                toml::table compiledWithTableWrapper;
                compiledWithTableWrapper.insert_or_assign("compiled_with", compiledWithTable);
                serializationOs << compiledWithTableWrapper << "\n\n";

                toml::table declarationTable;

                std::ostringstream codeOs;
                {
                    TOMLDisableGuard disableTOML(codeOs);
                    codeOs << _codePrologue;
                    for (const auto& funcDecl : functionDeclStrings)
                    {
                        codeOs << funcDecl << "\n";
                    }
                    codeOs << _debugCode;
                    codeOs << _codeEpilogue;
                }

                declarationTable.insert_or_assign("code", codeOs.str());
                toml::table declarationTableWrapper;
                declarationTableWrapper.insert_or_assign("declaration", declarationTable);

                // Newlines aren't added here since they're part of the string already when serialized
                serializationOs << declarationTableWrapper;
            }
        }

        return serializationOs.str();
    }

private:
    std::vector<std::unique_ptr<Function>> _functions;
    std::string _codePrologue;
    std::string _codeEpilogue;
    std::string _debugCode;
    std::string _name;
};
} // namespace hat
