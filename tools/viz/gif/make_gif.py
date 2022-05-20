#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import argparse
import glob
from PIL import Image

DEFAULTS = {
    "output": "output.gif",
    "duration_ms": 500
}

parser = argparse.ArgumentParser(
    description="""A tool to help create animated gifs from images. 
Example: python make_gif.py "/path/to/fuse2b*" --duration 750
"""
)
parser.add_argument(
    'pattern', type=str, help="The unix glob-style pattern to match (in quotes). E.g. '/path/to/fuse2b*"
)
parser.add_argument(
    '--output', type=str, default=DEFAULTS["output"], help=f"The output filename. Defaults to `{DEFAULTS['output']}`"
)
parser.add_argument(
    '--duration',
    type=int,
    default=DEFAULTS["duration_ms"],
    help=f"The duration between frames (in milliseconds). Defaults to {DEFAULTS['duration_ms']}"
)

args = parser.parse_args()

FILE_PATTERN = args.pattern
OUTPUT = args.output or DEFAULTS["output"]
FRAME_DURATION = args.duration or DEFAULTS["duration_ms"]


def make_gif(pattern, output):
    files = sorted(list(glob.iglob(f"{pattern}", recursive=True)))
    frames = [Image.open(f) for f in files]
    frames[0].save(output, append_images=frames, save_all=True, duration=FRAME_DURATION, loop=0)


make_gif(args.pattern, args.output)