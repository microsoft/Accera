////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
//
//  Library for runtime utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Random.h"

#include <random>

static std::default_random_engine RandomEngine;

void GetNextRandomValue(float* val)
{
    std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
    *val = distribution(RandomEngine);
}

void GetNextRandomIntValue(int* val, int lo, int hi)
{
    std::uniform_int_distribution<> distribution(lo, hi);
    *val = distribution(RandomEngine);
}

void GetNextNRandomValues(float* val, unsigned int N)
{
    std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
    for (unsigned int idx = 0; idx < N; ++idx)
    {
        val[idx] = distribution(RandomEngine);
    }
}

void GetNextNRandomIntValues(int* val, int lo, int hi, unsigned int N)
{
    std::uniform_int_distribution<> distribution(lo, hi);
    for (unsigned int idx = 0; idx < N; ++idx)
    {
        val[idx] = distribution(RandomEngine);
    }
}

void ResetRandomEngine(unsigned int seed)
{
    RandomEngine.seed(seed);
}
