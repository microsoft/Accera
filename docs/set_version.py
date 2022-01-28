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
from typing import List


def get_git_tags(repo_rootdir: str) -> List[str]:
    repo = Repo(os.path.abspath(repo_rootdir))
    if not repo.tags or len(repo.tags) < 2:
        sys.exit(f"Repository {repo} has less than 2 tags")
    return str(repo.tags[-2]), str(repo.tags[-1])


def inplace_search_replace(filepath: str, search: str, replace: str):
    with open(filepath, 'r') as f:
        s = f.read()
        if search in s:
            print(f"Updating {filepath}")

    with open(filepath, 'w') as f:
        s = s.replace(search, replace)
        f.write(s)


def update_versions(doc_rootdir: str, old_version: str, new_version: str):
    doc_files = glob.glob(f"{os.path.abspath(doc_rootdir)}/**/*.md", recursive=True)
    for doc_file in doc_files:
        inplace_search_replace(doc_file, old_version, new_version)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} repo_root doc_root")

    old_version, new_version = get_git_tags(sys.argv[1])
    update_versions(sys.argv[2], old_version, new_version)
