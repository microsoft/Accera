#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -e

# Run this from the repo root
ACCERA_ROOT=`pwd`
pip install mkdocs-material mkdocs-git-revision-date-plugin
cp README.md docs/README.md
python docs/set_version.py $ACCERA_ROOT $ACCERA_ROOT/docs
mkdocs serve