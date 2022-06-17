####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

add_executable(nest_dialect_test ir_test_main.cpp LowLevelIRTests.cpp NestUtilTests.cpp NestIRTests.cpp IRTestVerification.cpp)
target_link_libraries(nest_dialect_test PRIVATE ir mlirHelpers transforms Catch2::Catch2WithMain)

catch_discover_tests(nest_dialect_test)

# set_target_properties(nest_dialect_test check-all PROPERTIES FOLDER "test")
set_target_properties(nest_dialect_test PROPERTIES FOLDER "test")
