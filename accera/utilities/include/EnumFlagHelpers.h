////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#define ACCERA_DEFINE_ENUM_FLAG_OPERATORS(ENUMTYPE)                                                             \
    inline ENUMTYPE operator|(ENUMTYPE a, ENUMTYPE b)                                                        \
    {                                                                                                        \
        return ENUMTYPE(((std::underlying_type_t<ENUMTYPE>)a) | ((std::underlying_type_t<ENUMTYPE>)b));      \
    }                                                                                                        \
    inline ENUMTYPE& operator|=(ENUMTYPE& a, ENUMTYPE b)                                                     \
    {                                                                                                        \
        return (ENUMTYPE&)(((std::underlying_type_t<ENUMTYPE>&)a) |= ((std::underlying_type_t<ENUMTYPE>)b)); \
    }                                                                                                        \
    inline ENUMTYPE operator&(ENUMTYPE a, ENUMTYPE b)                                                        \
    {                                                                                                        \
        return ENUMTYPE(((std::underlying_type_t<ENUMTYPE>)a) & ((std::underlying_type_t<ENUMTYPE>)b));      \
    }                                                                                                        \
    inline ENUMTYPE& operator&=(ENUMTYPE& a, ENUMTYPE b)                                                     \
    {                                                                                                        \
        return (ENUMTYPE&)(((std::underlying_type_t<ENUMTYPE>&)a) &= ((std::underlying_type_t<ENUMTYPE>)b)); \
    }                                                                                                        \
    inline ENUMTYPE operator~(ENUMTYPE a)                                                                    \
    {                                                                                                        \
        return ENUMTYPE(~((std::underlying_type_t<ENUMTYPE>)a));                                             \
    }                                                                                                        \
    inline ENUMTYPE operator^(ENUMTYPE a, ENUMTYPE b)                                                        \
    {                                                                                                        \
        return ENUMTYPE(((std::underlying_type_t<ENUMTYPE>)a) ^ ((std::underlying_type_t<ENUMTYPE>)b));      \
    }                                                                                                        \
    inline ENUMTYPE& operator^=(ENUMTYPE& a, ENUMTYPE b)                                                     \
    {                                                                                                        \
        return (ENUMTYPE&)(((std::underlying_type_t<ENUMTYPE>&)a) ^= ((std::underlying_type_t<ENUMTYPE>)b)); \
    }
