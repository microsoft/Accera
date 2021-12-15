#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Requires: Python 3.7+
####################################################################################################

from scripts import HATFile

import argparse
import os
import sys

def main(cl_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to HAT file to read/write")
    args = parser.parse_args(cl_args)

    hat_file = HATFile.Deserialize(args.input)
    hat_file.Serialize()

if __name__ == "__main__":
    main(cl_args=sys.argv[1:])
