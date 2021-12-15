####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# use a directory so that we can include generated files (such as _version.py)
set(deployed_dir "src")

InstallAcceraDirectory(${deployed_dir} accera-gpu "accera/gpu")