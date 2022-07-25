@echo off
REM ####################################################################################################
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License. See LICENSE in the project root for license information.
REM Build script for the Accera Python package
REM ####################################################################################################

REM Run this from the repo root
pip install mkdocs-material mkdocs-git-revision-date-plugin
copy README.md docs\README.md
mkdocs serve
