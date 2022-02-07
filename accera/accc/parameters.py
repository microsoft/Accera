#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Authors: Mason Remy
# Requires: Python 3.5+
####################################################################################################

import os
import yaml

# TODO : need a better system for arbitrarily-nested lists
list_delimiters = [",", ":"]


# Parameters represent a single instance of a parameter key-value
class BaseParameter:
    cmdline_arg_prefix = "--"
    cmdline_arg_keyval_combiner = "="

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_cmd_arg(self):
        return self.cmdline_arg_prefix + self.name_to_cmd_arg(
        ) + self.cmdline_arg_keyval_combiner + self.value_to_cmd_arg()

    def value_to_cmd_arg(self):
        return str(self.value)

    def name_to_cmd_arg(self):
        return self.name


class ListParameter(BaseParameter):
    def __init__(self, name, value, serialization_delimiter):
        BaseParameter.__init__(self, name, value)
        self.serialization_delimiter = serialization_delimiter

    def value_to_cmd_arg(self):
        if len(self.value) == 0:
            return ""
        value_strs = []
        for elt in self.value:
            if isinstance(elt, BaseParameter):
                value_strs += [elt.value_to_cmd_arg()]
            else:
                value_strs += [str(elt)]
        return self.serialization_delimiter.join(value_strs)


class ParameterCollection:
    def __init__(self, param_list):
        self.param_list = param_list

    def to_cmd_arglist(self):
        return [param.to_cmd_arg() for param in self.param_list]

    def to_cmd_argstring(self):
        return " ".join(self.to_cmd_arglist())

    def merge(self, other):
        if not isinstance(other, ParameterCollection):
            raise ValueError("Can only merge objects of type ParameterCollection")
        self.param_list += other.param_list

    def add(self, param):
        if not isinstance(param, BaseParameter):
            raise ValueError("Can only add objects of type BaseParameter")
        self.param_list += [param]


class DomainParameter(ListParameter):
    domain_key = "domain"

    def __init__(self, domain, delimiter=list_delimiters[0]):
        ListParameter.__init__(self, self.domain_key, domain, delimiter)


class DomainParameterList(ListParameter):
    domain_key = "domains"

    def __init__(self, domain_list, delimiter=list_delimiters[1]):
        ListParameter.__init__(self, self.domain_key, domain_list, delimiter)


class LibraryNameParameter(BaseParameter):
    library_name_key = "library-name"

    def __init__(self, library_name):
        BaseParameter.__init__(self, self.library_name_key, library_name)


def parse_parameter_type(name, value):
    if isinstance(value, str) or isinstance(value, int) or isinstance(value, bool):
        return BaseParameter(name, value)
    elif isinstance(value, list):
        inner_params = [parse_parameter_type(name + "_" + str(idx), value[idx]) for idx in range(len(value))]
        inner_list_last_delimiter_idx = -1
        for param in inner_params:
            if isinstance(param, ListParameter):
                delimiter_idx = list_delimiters.index(param.serialization_delimiter)
                if delimiter_idx > inner_list_last_delimiter_idx:
                    inner_list_last_delimiter_idx = delimiter_idx
        return ListParameter(
            name, inner_params, serialization_delimiter=list_delimiters[inner_list_last_delimiter_idx + 1]
        )


def parse_parameters_from_yaml_file(yaml_filepath, parameter_key="Generator"):
    if not os.path.exists(yaml_filepath):
        raise ValueError("Provided yaml filepath does not exist: " + yaml_filepath)
    with open(yaml_filepath, "r") as f:
        yaml_data = yaml.safe_load(f)
    param_dict = yaml_data[parameter_key]
    param_list = []
    for key in param_dict:
        param_list += [parse_parameter_type(key, param_dict[key])]
    return ParameterCollection(param_list)


def parse_domain_list_from_csv(filepath, comment="#"):
    import pandas as pd
    csv_df = pd.read_csv(filepath, skipinitialspace=True, comment=comment)
    domain_list = [DomainParameter(list(csv_df.iloc[x])) for x in range(len(csv_df))]
    return DomainParameterList(domain_list)
