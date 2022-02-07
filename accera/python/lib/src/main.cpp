////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

#include <ir/include/InitializeAccera.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace lang = accera::python::lang;

PYBIND11_MODULE(_lang_python, m)
{
    accera::ir::InitializeAccera();

    m.doc() = R"pbdoc(
        .. currentmodule:: accera

        .. autosummary::
            :toctree: _generate

        )pbdoc";

    auto lang_mod = m.def_submodule("_lang");

    lang::DefineContainerTypes(m, lang_mod);
    lang::DefineNestTypes(lang_mod);
    lang::DefineExecutionPlanTypes(lang_mod);
    lang::DefineSchedulingTypes(lang_mod);
    lang::DefinePackagingTypes(m, lang_mod);
    lang::DefineOperations(lang_mod);

#ifdef ACCERA_VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(ACCERA_VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
