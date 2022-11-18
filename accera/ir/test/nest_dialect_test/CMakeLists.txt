####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

add_executable(nest_dialect_test  LowLevelIRTests.cpp NestUtilTests.cpp NestIRTests.cpp IRTestVerification.cpp DynamicRangeTests.cpp)
target_link_libraries(nest_dialect_test PRIVATE ir mlirHelpers transforms CatchWrapper)
catch_discover_tests(nest_dialect_test)
catch_discover_tests(nest_dialect_test TEST_SPEC 'exclude:[demo]' TEST_PREFIX 'nest_dialect_tests_')
set_target_properties(nest_dialect_test PROPERTIES FOLDER "tests")
