#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Updates doc files with the latest git tag version
####################################################################################################

import os
import sys
import glob
from git import Repo

VERSION_PATTERN = "<<VERSION>>"


def get_latest_git_tag(repo_rootdir: str) -> str:
    repo = Repo(os.path.abspath(repo_rootdir))
    if not repo.tags:
        sys.exit(f"Repository {repo} has no tags")
    return str(repo.tags[-1])


def inplace_search_replace(filepath: str, search: str, replace: str):
    with open(filepath, 'r') as f:
        s = f.read()
        if search in s:
            print(f"Updating {filepath}")

    with open(filepath, 'w') as f:
        s = s.replace(search, replace)
        f.write(s)


def update_versions(doc_rootdir: str, version: str):
    doc_files = glob.glob(f"{os.path.abspath(doc_rootdir)}/**/*.md", recursive=True)
    for doc_file in doc_files:
        inplace_search_replace(doc_file, VERSION_PATTERN, version)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} repo_root doc_root")

    version = get_latest_git_tag(sys.argv[1])
    update_versions(sys.argv[2], version)
