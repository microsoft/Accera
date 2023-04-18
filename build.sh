#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
set -e

# Run this from the repo root
pip install mkdocs-material mkdocs-git-revision-date-plugin
cp README.md docs/README.md

# To use a different port: mkdocs serve -a localhost:8765
mkdocs serve