////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Mason Remy
//
//  Library for runtime utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

void ResetRandomEngine(unsigned int seed);

void GetNextRandomValue(float*);
void GetNextRandomIntValue(int*, int lo, int hi);
void GetNextNRandomValues(float* buffer, unsigned int N);
void GetNextNRandomIntValues(int* buffer, int lo, int hi, unsigned int N);

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)
