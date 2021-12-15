#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set -e

if [ "$1" = "" ]; then
    echo "Usage:"
    echo "  $0 <TEST_BINARY>"
    exit 0
fi

TEST_BINARY=$1

TEST_TYPE=$2
if [ "$2" = "" ]; then
    TEST_TYPE="lit"
fi

ARGS=$3
if [ "$3" != "" ]; then
    shift 2
    ARGS=$@
fi

build()
{
    TARGET=$1
    # TODO: detect if we need to specify a config (e.g., because the generator is VS or XCode or something that needs it)
    # and do so (maybe by parsing the output of `cmake -LA -N ..` and looking at `CMAKE_MAKE_PROGRAM`?)

    cmake --build . --target ${TARGET}
}

build ${TEST_BINARY}

if [ "${TEST_TYPE}" == "lit" ]; then
    bin/${TEST_BINARY} ${ARGS} \
        | python3 tools/acc-opt/test/process_tests.py -p "jit_" \
        | FileCheck  --check-prefix=JIT -v ../tools/acc-opt/test/${TEST_BINARY}.cpp
elif [ "${TEST_TYPE}" == "jit" ]; then
    build acc-opt
    bin/${TEST_BINARY} ${ARGS} \
        | python3 tools/acc-opt/test/process_tests.py -p "jit_"
elif [ "${TEST_TYPE}" == "asm" ]; then
    build acc-opt
    bin/${TEST_BINARY} ${ARGS} \
        | python3 tools/acc-opt/test/process_tests.py -p "asm_" -t "asm"
fi
