####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(ROOT_BIN_DIR ${CMAKE_BINARY_DIR}/bin)

set(copy_scripts
    parse_toml_header.py
)

set(deployed_scripts )
foreach(script ${copy_scripts})
    configure_file(${script} ${ROOT_BIN_DIR}/${script} COPYONLY)
    set(deployed_scripts ${deployed_scripts} ${ROOT_BIN_DIR}/${script})
endforeach()

#
# Install toml scripts
#
install(PROGRAMS ${deployed_scripts} TYPE BIN)
