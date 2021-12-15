////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LoopNest_convolution_test.h"
#include "TestUtil.h"

#include <value/include/CachingStrategies.h>
#include <value/include/ComputeContext.h>
#include <value/include/EmitterContext.h>
#include <value/include/FunctionDeclaration.h>
#include <value/include/LLVMContext.h>
#include <value/include/LoopNests.h>
#include <value/include/Matrix.h>
#include <value/include/Scalar.h>
#include <value/include/Tensor.h>
#include <value/include/Value.h>
#include <value/include/Vector.h>

#include <value/include/loopnests/CodeGenerator.h>
#include <value/include/loopnests/Kernel.h>
#include <value/include/loopnests/LoopNest.h>
#include <value/include/loopnests/LoopNestPrinter.h>

#include <emitters/include/IRFunctionEmitter.h>

#include <utilities/include/FunctionUtils.h>
#include <utilities/include/Logger.h>

#include <testing/include/testing.h>

#include <optional>
#include <tuple>
#include <vector>

using namespace accera::emitters;
using namespace accera::utilities;
using namespace accera::logging;
using namespace accera::value;
using namespace accera::value::loopnests;

using namespace archived;

namespace accera
{
// Tests of convolution via LoopNests

int GetOutputDimensionSize(int inputSize, int receptiveFieldSize, int stride, int paddingSize)
{
    return (inputSize + 2 * paddingSize - receptiveFieldSize) / stride + 1;
}

struct ConvolutionConfig
{
    ConvolutionConfig(const std::vector<int>& inputSizes,
                      int outputFilters,
                      const std::vector<int>& receptiveFieldSize,
                      const std::vector<int>& strideSize,
                      const std::vector<int>& paddingSize,
                      const std::vector<int>& inputBlockSizes,
                      const std::vector<int>& outputBlockSizes)
    {
        outputSize[2] = outputFilters;
        for (int dim = 0; dim < 3; dim++)
        {
            inputSize[dim] = inputSizes[dim];
            inputBlockSize[dim] = inputBlockSizes[dim];
            outputBlockSize[dim] = outputBlockSizes[dim];

            // Value that are only computed in the row/column dimensions
            if (dim < 2)
            {
                receptiveField[dim] = receptiveFieldSize[dim];
                stride[dim] = strideSize[dim];
                padding[dim] = paddingSize[dim];
                outputSize[dim] = GetOutputDimensionSize(inputSize[dim], receptiveFieldSize[dim], strideSize[dim], paddingSize[dim]);
            }

            if (inputBlockSize[dim] > 0)
            {
                inputBlockCount[dim] = inputSize[dim] / inputBlockSize[dim];
                if (inputSize[dim] % inputBlockSize[dim] != 0)
                {
                    inputBlockCount[dim]++;
                }
            }

            if (outputBlockSize[dim] > 0)
            {
                outputBlockCount[dim] = outputSize[dim] / outputBlockSize[dim];
                if (outputSize[dim] % outputBlockSize[dim] != 0)
                {
                    outputBlockCount[dim]++;
                }
            }
        }

        weightSize[0] = outputSize[2];
        weightSize[1] = inputSize[2];
        weightSize[2] = receptiveField[0];
        weightSize[3] = receptiveField[1];

        MemoryShape inputPackedShape = { inputBlockCount[2], inputSize[0], inputSize[1], inputBlockSize[2] };
        MemoryShape inputPackedPadding = { 0, padding[0], padding[1], 0 };
        inputPackedPaddedLayout = { inputPackedShape, inputPackedPadding };
        MemoryShape inputLogicalPadding = { padding[0], padding[1], 0 };
        inputLogicalPaddedLayout = { MemoryShape{ inputSize[0], inputSize[1], inputSize[2] }, inputLogicalPadding };

        outputPackedLayout = { MemoryShape{ outputBlockCount[2], outputSize[0], outputSize[1], outputBlockSize[2] } };
        outputLogicalLayout = { MemoryShape{ outputSize[0], outputSize[1], outputSize[2] } };

        weightPackedLayout = { MemoryShape{
            outputBlockCount[2],
            inputBlockCount[2],
            weightSize[2],
            weightSize[3],
            inputBlockSize[2],
            outputBlockSize[2] } };
    }

    int inputSize[3];
    int outputSize[3];
    int weightSize[4];
    int receptiveField[2];
    int stride[2];
    int padding[2];

    int inputBlockSize[3];
    int outputBlockSize[3];

    int inputBlockCount[3];
    int outputBlockCount[3];

    MemoryLayout inputPackedPaddedLayout;
    MemoryLayout inputLogicalPaddedLayout;

    MemoryLayout outputPackedLayout;
    MemoryLayout outputLogicalLayout;

    MemoryLayout weightPackedLayout;
};

Tensor NaiveForLoopConvolution(const ConvolutionConfig& config, Tensor input, Array weights)
{
    auto output = MakeTensor<int>(config.outputSize[0], config.outputSize[1], config.outputSize[2], "expectedOutput");
    ForRange(config.outputSize[2], [&](Scalar outputChannel) {
        ForRange(config.inputSize[2], [&](Scalar inputChannel) {
            ForRange(config.outputSize[0], [&](Scalar outputRow) {
                ForRange(config.outputSize[1], [&](Scalar outputColumn) {
                    ForRange(config.receptiveField[0], [&](Scalar weightRow) {
                        ForRange(config.receptiveField[1], [&](Scalar weightColumn) {
                            Scalar inputRow = outputRow * config.stride[0] + weightRow - config.padding[0];
                            Scalar inputColumn = outputColumn * config.stride[1] + weightColumn - config.padding[1];
                            If(inputRow >= 0, [&] {
                                If(inputRow < Scalar{ config.inputSize[0] }, [&] {
                                    If(inputColumn >= 0, [&] {
                                        If(inputColumn < Scalar{ config.inputSize[1] }, [&] {
                                            output(outputRow, outputColumn, outputChannel) +=
                                                input(inputRow, inputColumn, inputChannel) *
                                                weights({ outputChannel, inputChannel, weightRow, weightColumn });
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
    return output;
}

Tensor UnpackOutputTensor(const ConvolutionConfig& config, Value packedOutput)
{
    auto unpackedOutput = MakeTensor<int>(config.outputSize[0], config.outputSize[1], config.outputSize[2], "unpackedOutput");
    packedOutput.SetLayout(config.outputPackedLayout);
    auto packedOutputArray = Array(packedOutput);

    ForRange(config.outputSize[2], [&](Scalar channelIdx) {
        ForRange(config.outputSize[0], [&](Scalar rowIdx) {
            ForRange(config.outputSize[1], [&](Scalar columnIdx) {
                unpackedOutput(rowIdx, columnIdx, channelIdx) = packedOutputArray({ channelIdx / config.outputBlockSize[2],
                                                                                    rowIdx,
                                                                                    columnIdx,
                                                                                    channelIdx % config.outputBlockSize[2] });
            });
        });
    });
    return unpackedOutput;
}

// Manually pad the input tensor
Scalar EfficientDirectConvolution_Test1()
{
    int inputRows = 4;
    int inputColumns = 4;
    int inputChannels = 4;
    int inputChannelBlockSize = 2;
    int outputFilters = 4;
    int outputColumnBlockSize = 2;
    int outputChannelBlockSize = 2;
    int weightRowsCols = 3;
    int stride = 1;
    int padding = 1;

    ConvolutionConfig config({ inputRows, inputColumns, inputChannels },
                             outputFilters,
                             { weightRowsCols, weightRowsCols },
                             { stride, stride },
                             { padding, padding },
                             { 0, 0, inputChannelBlockSize },
                             { 0, outputColumnBlockSize, outputChannelBlockSize });

    // clang-format off
    // Ordered in { ChannelBlock, Row, Column, Channel } memory layout
    // With padding in the row and column dimensions, but not in the channel dimension
    Vector paddedPackedInputData = {
        // Columns:
        // Pad      Col 0       Col 1       Col 2       Col 3       Pad
        // Channel block 0
        0,  0,      0,  0,      0,  0,      0,  0,      0,  0,      0,  0,  // Padding row
        0,  0,      0, 16,      1, 17,      2, 18,      3, 19,      0,  0,  // Row 0

        0,  0,      4, 20,      5, 21,      6, 22,      7, 23,      0,  0,  // Row 1
        0,  0,      8, 24,      9, 25,      10, 26,     11, 27,     0,  0,  // Row 2

        0,  0,      12, 28,     13, 29,     14, 30,     15, 31,     0,  0,  // Row 3
        0,  0,      0,  0,      0,  0,      0,  0,      0,  0,      0,  0,  // Padding row

        // Channel block 1
        0,  0,      0,  0,      0,  0,      0,  0,      0,  0,      0,  0,  // Padding row
        0,  0,      32, 48,     33, 49,     34, 50,     35, 51,     0,  0,  // Row 0

        0,  0,      36, 52,     37, 53,     38, 54,     39, 55,     0,  0,  // Row 1
        0,  0,      40, 56,     41, 57,     42, 58,     43, 59,     0,  0,  // Row 2

        0,  0,      44, 60,     45, 61,     46, 62,     47, 63,     0,  0,  // Row 3
        0,  0,      0,  0,      0,  0,      0,  0,      0,  0,      0,  0   // Padding row
    };
    // clang-format on

    auto packedPaddedInputValue = paddedPackedInputData.GetValue();
    packedPaddedInputValue.SetLayout(config.inputLogicalPaddedLayout);
    auto paddedInput = Tensor(packedPaddedInputValue);
    auto unpaddedInput = MakeIncrementingTensor<int>(config.inputSize[0],
                                                     config.inputSize[1],
                                                     config.inputSize[2],
                                                     "input");
    auto output = MakeTensor<int>(config.outputSize[0],
                                  config.outputSize[1],
                                  config.outputSize[2],
                                  "output");
    auto weights = MakeIncrementingArray<int>({ config.outputSize[2],
                                                config.inputSize[2],
                                                config.receptiveField[0],
                                                config.receptiveField[1] },
                                              "weights");

    // build up expected with naive for-loop implementation
    auto expectedOutput = NaiveForLoopConvolution(config, unpaddedInput, weights);

    loopnests::Index i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

    // Define LoopNest
    auto nest = Using({ paddedInput, weights }, ArgumentType::Input)
                    .Using({ output }, ArgumentType::Output)
                    .ForAll(j, 0, config.outputSize[2])
                    .ForAll(i, 0, config.inputSize[2])
                    .ForAll(l, 0, config.outputSize[0])
                    .ForAll(k, 0, config.outputSize[1])
                    .ForAll(n, 0, config.receptiveField[0])
                    .ForAll(m, 0, config.receptiveField[1])
                    .Do([=](Tensor input_, Array weights_, Tensor output_, Scalar j_, Scalar i_, Scalar l_, Scalar k_, Scalar n_, Scalar m_) {
                        Scalar inputRow = l_ * config.stride[0] + n_ - config.padding[0];
                        Scalar inputColumn = k_ * config.stride[1] + m_ - config.padding[1];
                        output_(l_, k_, j_) += input_(inputRow, inputColumn, i_) * weights_({ j_, i_, n_, m_ });
                    });

    auto& schedule = nest.GetSchedule();

    auto iTopLevel = i;
    auto jTopLevel = j;
    auto kTopLevel = k;

    auto jBlock = schedule.Split(j, config.outputBlockSize[2]);
    auto iBlock = schedule.Split(i, config.inputBlockSize[2]);
    auto kBlock = schedule.Split(k, config.outputBlockSize[1]);

    schedule.SetOrder({ jBlock, iBlock, l, kBlock, n, m, i, k, j });

    schedule.Unroll(i);
    schedule.Unroll(j);
    schedule.Unroll(k);

    auto extraCacheInputParams = std::make_tuple(false, config.inputBlockSize[2], config.padding[0], config.padding[1]);
    schedule.Cache<ConvolutionInputCachingStrategy>(paddedInput,
                                                    {},
                                                    { config.inputSize[2], config.inputSize[0], config.inputSize[1] },
                                                    { iBlock },
                                                    std::nullopt,
                                                    extraCacheInputParams);

    auto extraCacheOutputParams = std::make_tuple(config.outputBlockSize[2]);
    schedule.Cache<ConvolutionOutputCachingStrategy>(output,
                                                     { l, kTopLevel, jTopLevel },
                                                     { config.outputBlockSize[1], config.outputBlockSize[2] },
                                                     { jBlock, kBlock },
                                                     std::nullopt,
                                                     extraCacheOutputParams);

    auto extraCacheWeightParams = std::make_tuple(true, config.outputBlockSize[2], config.inputBlockSize[2]);
    schedule.Cache<ConvolutionWeightCachingStrategy>(weights,
                                                     {},
                                                     { config.outputSize[2], config.inputSize[2], config.receptiveField[0], config.receptiveField[1] },
                                                     { jBlock, iBlock },
                                                     std::nullopt,
                                                     extraCacheWeightParams);

#if 0 // DEBUGGING
    auto loop = nest.GetUnderlyingLoopNest();
    DebugDump(loop);
#endif
    // Run the generator
    nest.Run();

    // Unpack the output
    auto unpackedOutput = UnpackOutputTensor(config, output.GetValue());

    return VerifySame(unpackedOutput, expectedOutput);
}

Scalar EfficientDirectConvolution_AutomaticInputPad(const ConvolutionConfig& config)
{
    auto unpaddedInput = MakeIncrementingTensor<int>(config.inputSize[0], config.inputSize[1], config.inputSize[2], "input");

    // Allocate based on the padded packed layout, which may have additional space if there are incomplete blocks
    auto paddedInputValue = StaticAllocate("packedInputValue", unpaddedInput.GetType(), config.inputPackedPaddedLayout);

    // View with the logical padded layout, allowing there to be excess space at the end of the buffer
    paddedInputValue.SetLayout(config.inputLogicalPaddedLayout);

    Tensor paddedInput = paddedInputValue;
    For(unpaddedInput, [&](Scalar row, Scalar column, Scalar channel) {
        paddedInput(row, column, channel) = unpaddedInput(row, column, channel);
    });
    //auto output = MakeTensor<int>(config.outputSize[0], config.outputSize[1], config.outputSize[2], "output");
    auto outputValue = StaticAllocate("packedOutputValue", unpaddedInput.GetType(), config.outputPackedLayout);
    outputValue.SetLayout(config.outputLogicalLayout);
    Tensor output = outputValue;

    auto baseWeights = MakeIncrementingArray<int>({ config.weightSize[0], config.weightSize[1], config.weightSize[2], config.weightSize[3] }, "weights");

    // Allocate based on the packed weight layout, adjusted for incomplete blocks
    auto packedWeightValue = StaticAllocate("packedWeightValue", baseWeights.GetType(), config.weightPackedLayout);
    packedWeightValue.SetLayout(baseWeights.GetValue().GetLayout());
    Array weights = packedWeightValue;
    For(baseWeights, [&](const std::vector<Scalar>& indices) {
        weights(indices) = baseWeights(indices);
    });

    // build up expected with naive for-loop implementation
    auto expectedOutput = NaiveForLoopConvolution(config, unpaddedInput, weights);

    loopnests::Index i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

    // Define LoopNest
    auto nest = Using({ paddedInput, weights }, ArgumentType::Input)
                    .Using({ output }, ArgumentType::Output)
                    .ForAll(j, 0, config.outputSize[2])
                    .ForAll(i, 0, config.inputSize[2])
                    .ForAll(l, 0, config.outputSize[0])
                    .ForAll(k, 0, config.outputSize[1])
                    .ForAll(n, 0, config.receptiveField[0])
                    .ForAll(m, 0, config.receptiveField[1])
                    .Do([=](Tensor input_, Array weights_, Tensor output_, Scalar j_, Scalar i_, Scalar l_, Scalar k_, Scalar n_, Scalar m_) {
                        Scalar inputRow = l_ * config.stride[0] + n_ - config.padding[0];
                        Scalar inputColumn = k_ * config.stride[1] + m_ - config.padding[1];
                        output_(l_, k_, j_) += input_(inputRow, inputColumn, i_) * weights_({ j_, i_, n_, m_ });
                    });

    auto& schedule = nest.GetSchedule();

    auto iTopLevel = i;
    auto jTopLevel = j;
    auto kTopLevel = k;

    auto jBlock = schedule.Split(j, config.outputBlockSize[2]);
    auto iBlock = schedule.Split(i, config.inputBlockSize[2]);
    auto kBlock = schedule.Split(k, config.outputBlockSize[1]);

    schedule.SetOrder({ jBlock, iBlock, l, kBlock, n, m, i, k, j });

    schedule.Unroll(i);
    schedule.Unroll(j);
    schedule.Unroll(k);

    auto extraCacheInputParams = std::make_tuple(true, config.inputBlockSize[2], config.padding[0], config.padding[1]);
    schedule.Cache<ConvolutionInputCachingStrategy>(paddedInput,
                                                    {},
                                                    { config.inputSize[2], config.inputSize[0], config.inputSize[1] },
                                                    { iBlock },
                                                    std::nullopt,
                                                    extraCacheInputParams);

    auto extraCacheOutputParams = std::make_tuple(config.outputBlockSize[2]);
    schedule.Cache<ConvolutionOutputCachingStrategy>(output,
                                                     { l, kTopLevel, jTopLevel },
                                                     { config.outputBlockSize[1], config.outputBlockSize[2] },
                                                     { jBlock, kBlock },
                                                     std::nullopt,
                                                     extraCacheOutputParams);

    auto extraCacheWeightParams = std::make_tuple(true, config.outputBlockSize[2], config.inputBlockSize[2]);
    schedule.Cache<ConvolutionWeightCachingStrategy>(weights,
                                                     {},
                                                     { config.weightSize[0], config.weightSize[1], config.weightSize[2], config.weightSize[3] },
                                                     { jBlock, iBlock },
                                                     std::nullopt,
                                                     extraCacheWeightParams);

#if 0 // DEBUGGING
    auto loop = nest.GetUnderlyingLoopNest();
    DebugDump(loop);
#endif
    // Run the generator
    nest.Run();

    // Unpack the output
    auto unpackedOutput = UnpackOutputTensor(config, output.GetValue());

    return VerifySame(unpackedOutput, expectedOutput);
}

Scalar EfficientDirectConvolution_Test2()
{
    int inputRows = 4;
    int inputColumns = 4;
    int inputChannels = 4;
    int inputChannelBlockSize = 2;

    int outputFilters = 4;
    int outputColumnBlockSize = 2;
    int outputChannelBlockSize = 2;

    int weightRowsCols = 3;
    int stride = 1;
    int padding = 1;

    ConvolutionConfig config({ inputRows, inputColumns, inputChannels },
                             outputFilters,
                             { weightRowsCols, weightRowsCols },
                             { stride, stride },
                             { padding, padding },
                             { 0, 0, inputChannelBlockSize },
                             { 0, outputColumnBlockSize, outputChannelBlockSize });

    return EfficientDirectConvolution_AutomaticInputPad(config);
}

// Only difference from Test2 is that the output column block size doesn't evenly divide the output columns
Scalar EfficientDirectConvolution_Test3()
{
    int inputRows = 4;
    int inputColumns = 4;
    int inputChannels = 4;
    int inputChannelBlockSize = 2;

    int outputFilters = 4;
    int outputColumnBlockSize = 3;
    int outputChannelBlockSize = 2;

    int weightRowsCols = 3;
    int stride = 1;
    int padding = 1;

    ConvolutionConfig config({ inputRows, inputColumns, inputChannels },
                             outputFilters,
                             { weightRowsCols, weightRowsCols },
                             { stride, stride },
                             { padding, padding },
                             { 0, 0, inputChannelBlockSize },
                             { 0, outputColumnBlockSize, outputChannelBlockSize });

    return EfficientDirectConvolution_AutomaticInputPad(config);
}

// Difference from Test1 is that the input channel block size doesn't evenly divide the input channels
Scalar EfficientDirectConvolution_Test4()
{
    int inputRows = 4;
    int inputColumns = 4;
    int inputChannels = 4;
    int inputChannelBlockSize = 3;

    int outputFilters = 4;
    int outputColumnBlockSize = 2;
    int outputChannelBlockSize = 2;

    int weightRowsCols = 3;
    int stride = 1;
    int padding = 1;

    ConvolutionConfig config({ inputRows, inputColumns, inputChannels },
                             outputFilters,
                             { weightRowsCols, weightRowsCols },
                             { stride, stride },
                             { padding, padding },
                             { 0, 0, inputChannelBlockSize },
                             { 0, outputColumnBlockSize, outputChannelBlockSize });

    return EfficientDirectConvolution_AutomaticInputPad(config);
}

// Difference from Test1 is that the output channel block size doesn't evenly divide the output filters
Scalar EfficientDirectConvolution_Test5()
{
    int inputRows = 4;
    int inputColumns = 4;
    int inputChannels = 4;
    int inputChannelBlockSize = 2;

    int outputFilters = 4;
    int outputColumnBlockSize = 2;
    int outputChannelBlockSize = 3;

    int weightRowsCols = 3;
    int stride = 1;
    int padding = 1;

    ConvolutionConfig config({ inputRows, inputColumns, inputChannels },
                             outputFilters,
                             { weightRowsCols, weightRowsCols },
                             { stride, stride },
                             { padding, padding },
                             { 0, 0, inputChannelBlockSize },
                             { 0, outputColumnBlockSize, outputChannelBlockSize });

    return EfficientDirectConvolution_AutomaticInputPad(config);
}

// Difference from Test1 is that the input channel block size, output column block size,
// and output channel block size don't evenly divide the input channels, output columns,
// and output channels, respectively
Scalar EfficientDirectConvolution_Test6()
{
    int inputRows = 4;
    int inputColumns = 4;
    int inputChannels = 4;
    int inputChannelBlockSize = 3;

    int outputFilters = 4;
    int outputColumnBlockSize = 3;
    int outputChannelBlockSize = 3;

    int weightRowsCols = 3;
    int stride = 1;
    int padding = 1;

    ConvolutionConfig config({ inputRows, inputColumns, inputChannels },
                             outputFilters,
                             { weightRowsCols, weightRowsCols },
                             { stride, stride },
                             { padding, padding },
                             { 0, 0, inputChannelBlockSize },
                             { 0, outputColumnBlockSize, outputChannelBlockSize });

    return EfficientDirectConvolution_AutomaticInputPad(config);
}
} // namespace accera
