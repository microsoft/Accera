####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

add_executable(ir_tests ir_tests.cpp)
target_link_libraries(ir_tests PRIVATE ir transforms value mlirHelpers CatchWrapper)
catch_discover_tests(ir_tests TEST_SPEC 'exclude:[demo]' TEST_PREFIX 'ir_tests_')
set_target_properties(ir_tests PROPERTIES FOLDER "tests")
