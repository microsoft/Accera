####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################


class LoopIndex:
    def __init__(self, nest=None, name=None):
        self._nest = nest
        self.base_index = id(self)
        self._name = name

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    def create_child_index(self):
        child = LoopIndex(self._nest)
        child.base_index = self.base_index
        return child

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
