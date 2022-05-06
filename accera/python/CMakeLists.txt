####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

if(NOT BUILD_PYTHON_BINDINGS)
  return()
endif(NOT BUILD_PYTHON_BINDINGS)

include(AddPyBind11)

set(library_name _lang_python)

set(src lib/src/ContainerTypes.cpp
  lib/src/ExecutionPlanTypes.cpp
  lib/src/NestTypes.cpp
  lib/src/Operations.cpp
  lib/src/PackagingTypes.cpp
  lib/src/SchedulingTypes.cpp
  lib/src/main.cpp)

include_directories(lib/include)

pybind11_add_module(${library_name} ${src} ${include})
add_dependencies(${library_name} acc-opt)
add_dependencies(${library_name} acc-translate)
if(Vulkan_FOUND)
  add_dependencies(${library_name} acc-vulkan-runtime-wrappers)
endif()
target_link_libraries(${library_name} PRIVATE value utilities)
target_compile_definitions(
  ${library_name} PRIVATE ACCERA_VERSION_INFO="${ACCERA_VERSION_INFO}"
)
target_precompile_headers(
  ${library_name} PRIVATE lib/include/AcceraTypes.h
)

InstallAcceraPyLibrary(${library_name} accera "accera")
InstallAcceraDirectory(accera accera "accera")

# samples
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
  install(DIRECTORY samples/ # trailing slash is important!
          COMPONENT accera
          CONFIGURATIONS ${CONFIG}
          DESTINATION "accera/samples"
  )
endforeach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
InstallAcceraDirectory(samples accera "accera/samples")

# tests
configure_file("accera/test/smoke_test.py" "test/smoke_test.py" @ONLY)
add_test(NAME ${library_name}_smoketest COMMAND python -m pytest -v --junitxml=test/test-smoke_test.xml test/smoke_test.py)

configure_file("accera/test/unit_tests.py" "test/unit_tests.py" @ONLY)
add_test(NAME ${library_name}_unittests COMMAND python -m pytest -v --junitxml=test/test-unit_tests.xml test/unit_tests.py)

configure_file("accera/test/dsl_tests.py" "test/dsl_tests.py" @ONLY)
add_test(NAME ${library_name}_dsltests COMMAND python -m pytest -v --junitxml=test/test-dsl_tests.xml test/dsl_tests.py)

configure_file("accera/test/mfma_tests.py" "test/mfma_tests.py" @ONLY)
add_test(NAME ${library_name}_mfmatests COMMAND python -m pytest -v --junitxml=test/test-mfma_tests.xml test/mfma_tests.py)

# sub-packages
add_subdirectory(compilers)
add_subdirectory(gpu)
add_subdirectory(llvm)
