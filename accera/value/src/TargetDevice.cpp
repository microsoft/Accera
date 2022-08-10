////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Lisa Ong, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include "value/include/TargetDevice.h"
#include "value/include/EmitterException.h"
#include "value/include/LLVMUtilities.h"

#include <llvm/ADT/Triple.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Target/TargetMachine.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

#include <map>

namespace accera
{
namespace value
{
    namespace
    {
        /// <summary> An enum containing the relocation model of the LLVM machine code output {Static, PIC_, DynamicNoPIC, ROPI, RWPI, ROPI_RWPI} </summary>
        using OutputRelocationModel = llvm::Reloc::Model;

        static const size_t c_defaultNumBits = 64;

        // ref: https://interrupt.memfault.com/blog/arm-cortexm-with-llvm-clang
        // Clang Target Triple Internals
        // The makeup of a --target value is actually:

        // <arch><sub>-<vendor>-<sys>-<abi>

        // A rough breakdown of what is expected for ARM targets is:

        // arch: One of the “registered targets” that was output from llc --version so for ARM it is arm.
        // sub : When left blank the value is inferred from other flags specified as part of the compilation. For ARM Cortex-M, v6m, v7m, v7em, v8m, etc are all be legal values.
        // vendor: For ARM this can be left blank or you can specify unknown explicitly.
        // sys: For embedded targets this will be none. If you are compiling code targeting an OS, the name of the OS will be used. For example, linux or darwin.
        // abi: For Embedded ARM this will always be eabi (“embedded-application binary interface”)

        // Triples
        std::string c_macTriple = "x86_64-apple-macosx10.12.0"; // alternate: "x86_64-apple-darwin16.0.0"
        std::string c_macArm64Triple = "arm64-apple-darwin21.4.0";
        std::string c_linuxTriple = "x86_64-pc-linux-gnu";
        std::string c_windowsTriple = "x86_64-pc-win32";
        std::string c_armv6Triple = "armv6--linux-gnueabihf"; // raspberry pi 0
        std::string c_armv7Triple = "armv7--linux-gnueabihf"; // raspberry pi 3 and orangepi0
        std::string c_arm64Triple = "aarch64-unknown-linux-gnu"; // DragonBoard
        std::string c_iosTriple = "aarch64-apple-ios"; // alternates: "arm64-apple-ios7.0.0", "thumbv7-apple-ios7.0"

        // CPUs
        std::string c_armCortexM4 = "cortex-m4";
        std::string c_pi0Cpu = "arm1136jf-s";
        std::string c_pi3Cpu = "cortex-a53";
        std::string c_orangePi0Cpu = "cortex-a7";

        // clang settings:
        // target=armv7-apple-darwin

        std::string c_macDataLayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128";
        std::string c_linuxDataLayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128";
        std::string c_windowsDataLayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128";
        std::string c_armDataLayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64";
        std::string c_arm64DataLayout = "e-m:e-i64:64-i128:128-n32:64-S128"; // DragonBoard
        std::string c_iosDataLayout = "e-m:o-i64:64-i128:128-n32:64-S128";

        const std::map<std::string, std::function<void(TargetDevice&)>> KnownTargetDeviceNameMap = {
            { "mac", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_macTriple;
                 targetDevice.dataLayout = c_macDataLayout;
             } },
            { "mac_arm64", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_macArm64Triple;
                 targetDevice.dataLayout = c_macDataLayout;
             } },
            { "linux", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_linuxTriple;
                 targetDevice.dataLayout = c_linuxDataLayout;
             } },
            { "windows", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_windowsTriple;
                 targetDevice.dataLayout = c_windowsDataLayout;
             } },
            { "avx512", [](TargetDevice& targetDevice) {
                 targetDevice.architecture = "x86_64";
                 targetDevice.cpu = "skylake-avx512";
                 targetDevice.numBits = 64;
                 targetDevice.features = "+avx512f";
             } },
            { "pi0", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_armv6Triple;
                 targetDevice.dataLayout = c_armDataLayout;
                 targetDevice.numBits = 32;
                 targetDevice.cpu = c_pi0Cpu; // maybe not necessary
             } },
            { "pi3", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_armv7Triple;
                 targetDevice.dataLayout = c_armDataLayout;
                 targetDevice.numBits = 32;
                 targetDevice.cpu = c_pi3Cpu; // maybe not necessary
             } },
            { "orangepi0" /* orangepi (Raspbian) */, [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_armv7Triple;
                 targetDevice.dataLayout = c_armDataLayout;
                 targetDevice.numBits = 32;
                 targetDevice.cpu = c_orangePi0Cpu; // maybe not necessary
             } },
            { "pi3_64" /* pi3 (openSUSE) */, [](TargetDevice& targetDevice) {
                 // need to set arch to aarch64?
                 targetDevice.triple = c_arm64Triple;
                 targetDevice.dataLayout = c_arm64DataLayout;
                 targetDevice.numBits = 64;
                 targetDevice.cpu = c_pi3Cpu;
             } },
            { "aarch64" /* arm64 linux (DragonBoard) */, [](TargetDevice& targetDevice) {
                 // need to set arch to aarch64?
                 targetDevice.triple = c_arm64Triple;
                 targetDevice.dataLayout = c_arm64DataLayout;
                 targetDevice.numBits = 64;
             } },
            { "ios", [](TargetDevice& targetDevice) {
                 targetDevice.triple = c_iosTriple;
                 targetDevice.dataLayout = c_iosDataLayout;
             } },
            { "cortex-m0", [](TargetDevice& targetDevice) {
                 targetDevice.triple = "armv6m-unknown-none-eabi";
                 targetDevice.features = "+armv6-m,+v6m";
                 targetDevice.architecture = "thumb";
             } },
            { "cortex-m4", [](TargetDevice& targetDevice) {
                 targetDevice.triple = "thumbv7em-arm-none-eabi";
                 targetDevice.architecture = "thumb";
                 targetDevice.cpu = c_armCortexM4;
                 targetDevice.dataLayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64";
             } }
        };

        llvm::Triple GetNormalizedTriple(std::string tripleString)
        {
            auto normalizedTriple = llvm::Triple::normalize(tripleString.empty() ? llvm::sys::getDefaultTargetTriple() : tripleString);
            return llvm::Triple(normalizedTriple);
        }

        // Function prototypes used internally
        void SetHostTargetProperties(TargetDevice& targetDevice);
        bool HasKnownDeviceName(TargetDevice& targetDevice);
        void SetTargetPropertiesFromName(TargetDevice& targetDevice);
        void VerifyCustomTargetProperties(TargetDevice& targetDevice);
        void SetTargetDataLayout(TargetDevice& targetDevice);
    } // namespace

    bool TargetDevice::IsWindows() const
    {
        auto tripleObj = GetNormalizedTriple(triple);
        return tripleObj.getOS() == llvm::Triple::Win32;
    }

    bool TargetDevice::IsLinux() const
    {
        auto tripleObj = GetNormalizedTriple(triple);
        return tripleObj.getOS() == llvm::Triple::Linux;
    }

    bool TargetDevice::IsMacOS() const
    {
        auto tripleObj = GetNormalizedTriple(triple);
        return tripleObj.getOS() == llvm::Triple::MacOSX || tripleObj.getOS() == llvm::Triple::Darwin;
    }

    TargetDevice GetTargetDevice(std::string deviceName)
    {
        TargetDevice target;
        target.deviceName = deviceName;
        CompleteTargetDevice(target);
        return target;
    }

    std::vector<std::string> GetKnownDeviceNames()
    {
        std::vector<std::string> names;
        for (const auto& [name, _] : KnownTargetDeviceNameMap)
        {
            names.push_back(name);
        }
        return names;
    }

    void CompleteTargetDevice(TargetDevice& targetDevice)
    {
        auto deviceName = targetDevice.deviceName;

        if (targetDevice.numBits == 0)
        {
            targetDevice.numBits = c_defaultNumBits;
        }

        // Set low-level args based on target name (if present)
        if (deviceName == "host")
        {
            SetHostTargetProperties(targetDevice);
        }
        else if (HasKnownDeviceName(targetDevice))
        {
            SetTargetPropertiesFromName(targetDevice);
        }
        else if (deviceName == "custom")
        {
            SetTargetDataLayout(targetDevice);
            VerifyCustomTargetProperties(targetDevice);
        }
        else
        {
            throw EmitterException(EmitterError::targetNotSupported, "Unknown target device name: " + deviceName);
        }
    }

    namespace
    {
        void SetTargetDataLayout(TargetDevice& targetDevice)
        {
            std::string error;
            const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetDevice.triple, error);
            if (target == nullptr)
            {
                throw EmitterException(EmitterError::targetNotSupported, "Couldn't create target " + error);
            }
            const OutputRelocationModel relocModel = OutputRelocationModel::Static;

            // Aarch64 only supports Tiny, Small, Large
            const llvm::CodeModel::Model codeModel = (targetDevice.architecture == "aarch64") ? llvm::CodeModel::Small : llvm::CodeModel::Medium;

            const llvm::TargetOptions options;
            std::unique_ptr<llvm::TargetMachine> targetMachine(target->createTargetMachine(targetDevice.triple,
                                                                                           targetDevice.cpu,
                                                                                           targetDevice.features,
                                                                                           options,
                                                                                           relocModel,
                                                                                           codeModel,
                                                                                           llvm::CodeGenOpt::Level::Default));

            if (!targetMachine)
            {
                throw EmitterException(EmitterError::targetNotSupported, "Unable to allocate target machine");
            }

            llvm::DataLayout dataLayout(targetMachine->createDataLayout());
            targetDevice.dataLayout = dataLayout.getStringRepresentation();
        }

        void SetHostTargetProperties(TargetDevice& targetDevice)
        {
            auto hostTripleString = llvm::sys::getProcessTriple();
            llvm::Triple hostTriple(hostTripleString);

            targetDevice.triple = hostTriple.normalize();
            targetDevice.architecture = llvm::Triple::getArchTypeName(hostTriple.getArch()).str();
            targetDevice.cpu = llvm::sys::getHostCPUName().str();

            llvm::SubtargetFeatures targetDeviceFeatures;
            llvm::StringMap<bool> hostFeatures;
            llvm::sys::getHostCPUFeatures(hostFeatures);
            for (const auto& feature : hostFeatures)
            {
                targetDeviceFeatures.AddFeature(feature.first(), feature.second);
            }
            targetDevice.features = targetDeviceFeatures.getString();
            if (!targetDevice.features.empty())
            {
                targetDevice.features.pop_back();
            }

            SetTargetDataLayout(targetDevice);
        }

        bool HasKnownDeviceName(TargetDevice& targetDevice)
        {
            auto deviceName = targetDevice.deviceName;
            return (KnownTargetDeviceNameMap.find(deviceName) != KnownTargetDeviceNameMap.end());
        }

        void SetTargetPropertiesFromName(TargetDevice& targetDevice)
        {
            auto deviceName = targetDevice.deviceName;
            auto it = KnownTargetDeviceNameMap.find(deviceName);
            if (it != KnownTargetDeviceNameMap.end())
            {
                (it->second)(targetDevice);
            }
        }

        void VerifyCustomTargetProperties(TargetDevice& targetDevice)
        {
            if (targetDevice.triple == "")
            {
                throw EmitterException(EmitterError::badFunctionArguments, "Missing 'triple' information");
            }
            if (targetDevice.cpu == "")
            {
                throw EmitterException(EmitterError::badFunctionArguments, "Missing 'cpu' information");
            }
        }
    } // namespace
} // namespace value
} // namespace accera
