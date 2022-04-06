#!/bin/sh
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

pip install bump2version

# Usage: bump2version <part>, where <part> is major, minor, or patch
# Note: This will override .bumpversion.cfg
# TODO: tie the version with git tags

bump2version patch