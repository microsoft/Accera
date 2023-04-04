#!/usr/bin/env python3
import accera as acc

def add_function_build_pkg(plan: acc.Plan, A, B, C, func_name: str):
    package = acc.Package()
    package.add(plan, args=(A, B, C), base_name=func_name)
    package.build(func_name, format=acc.Package.Format.HAT_SOURCE)