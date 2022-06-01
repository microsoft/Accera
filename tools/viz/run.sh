#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -e

pip install -r requirements.txt
cp -f ../../docs/assets/logos/Accera_darktext.svg static/.

# python viz_tool.py [--port <port>]
python viz_tool.py