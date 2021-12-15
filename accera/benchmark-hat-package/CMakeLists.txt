####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(ROOT_BIN_DIR ${CMAKE_BINARY_DIR}/bin)

set(copy_files
    benchmark_hat_package.py
)

set(deployed_files )
foreach(file ${copy_files})
    configure_file(${file} ${ROOT_BIN_DIR}/${file} COPYONLY)
    set(deployed_files ${deployed_files} ${ROOT_BIN_DIR}/${file})
endforeach()

#
# Install files
#
install(FILES ${deployed_files} TYPE BIN)
install(
        FILES ${deployed_files}
        COMPONENT accera
        DESTINATION "accera/tools/benchmark"
)
