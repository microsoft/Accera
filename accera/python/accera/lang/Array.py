####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

import logging
from typing import *
from enum import Enum, auto
from functools import partial

from .._lang_python import ScalarType, _MemoryLayout, AllocateFlags, Role
from .._lang_python._lang import Array as NativeArray, Dimension
from .Layout import Layout, MemoryMapLayout
from ..Parameter import DelayedParameter
from ..Constants import inf, k_dynamic_size
from .NativeLoopNestContext import NativeLoopNestContext


class Array:
    "A multi-dimensional array"

    Layout = Layout

    def __init__(
        self,
        role: "accera.Role",
        name: str = '',
        data: Union["numpy.ndarray"] = None,
        element_type: Union["accera.ScalarType", type] = None,
        layout: Union["accera.Array.Layout", Tuple[int]] = Layout.FIRST_MAJOR,
        offset: int = 0,
        shape: Tuple[Union[int, DelayedParameter, Dimension]] = None,
        flags: "accera.AllocateFlags" = AllocateFlags.NONE
    ):
        """Creates an Array

        Args:
            role: The role of the array determines if the array scope is internal or external and if the array is mutable or immutable.
            data: The contents of a constant array. Required for `Array,Role.CONST` arrays but should not be specified for other roles.
            element_type: The array element type, default: ScalarType.float32 or inferred from data
            layout: The affine memory map, default: Array.Layout.FIRST_MAJOR
                A tuple may also be used to specify an explicit affine memory map. An affine memory map computes a
                scalar index into flattened memory (s_index) by performing a vector dot product of the shape vector (v_shape)
                with the affine memory map (v_memory_map). The general formula is:

                    s_index = v_shape.dot(v_memory_map) + s_offset

                where s_offset=0 in the current implementation.

               Currently, only affine memory maps for the first-major and last-major layouts are supported. For example,
               given a 4-dimensional shape vector v_shape = (s0, s1, s2, s3):

                    First-major: (s0xs1xs2, s0xs1, s2, 1)

                    Last-major: (1, s0, s0xs1, s0xs1xs2)

              In both cases, the last dimension (s3) is not used in computing the affine memory map.
            offset: The offset of the affine memory map | integer (positive, zero, or negative), default: 0
            shape: The array shape. Required for roles other than `Role.CONST`, should not be specified for `Role.CONST`
        """

        self._role = role
        self._name = name
        self._data = data
        self._element_type = element_type
        self._layout = layout
        self._requested_layout = layout    # TODO : is there a better name for this? This is the layout as specified via the DSL, not the MemoryLayout object that gets produced in the C++ code
        self._offset = offset
        self._shape = shape
        self._native_array = None
        self._delayed_calls = {}
        self._flags = flags
        self._size_str = ""

        if self._role == Role.CONST:
            if self._data is None:
                raise ValueError("data is required for Role.CONST")
            shape = self._data.shape    # infer shape from data
            self._shape = shape

            # For some reason ScalarType.__entries does not resolve correctly at this point
            # so we need this mapping instead of using ScalarType.__entries[str(numpy.dtype)]
            type_map = {
                ScalarType.bool: "bool",
                ScalarType.int8: "int8",
                ScalarType.int16: "int16",
                ScalarType.int32: "int32",
                ScalarType.int64: "int64",
                ScalarType.uint8: "uint8",
                ScalarType.uint16: "uint16",
                ScalarType.uint32: "uint32",
                ScalarType.uint64: "uint64",
                ScalarType.float16: "float16",
                ScalarType.float32: "float32",
                ScalarType.float64: "float64",
            }
            dtype_map = {y: x
                         for x, y in type_map.items()}
            if self._element_type:    # override the data.dtype
                dtype = type_map.get(self._element_type, None)
                if dtype:
                    if str(self._data.dtype) != dtype:
                        logging.debug(f"[API] Converted from {self._data.dtype} to {dtype}")
                        self._data = self._data.astype(dtype)
                    # else no conversion needed
                else:
                    raise NotImplementedError(f"Unsupported element type {self._element_type} for Role.CONST")
            else:    # infer element_type from data
                self._element_type = dtype_map.get(str(self._data.dtype), None)
                if self._element_type:
                    logging.debug(f"[API] Inferred {self._data.dtype} as {self._element_type}")
                else:
                    raise NotImplementedError(f"Unsupported dtype {self._data.dtype} for Role.CONST")

        if not self._element_type:
            self._element_type = ScalarType.float32
        elif not isinstance(self._element_type, ScalarType):
            if self._element_type is int:
                self._element_type = ScalarType.int64
            elif self._element_type is float:
                self._element_type = ScalarType.float32
            else:
                raise ValueError("Unknown element type on Array instance")

        if shape:
            if any([isinstance(s, DelayedParameter) for s in shape]):
                self._delayed_calls[partial(self._init_delayed)] = tuple([s for s in shape])
                return
            elif shape[-1] == inf:
                if (len(shape) > 1 and any([s == inf for s in shape[:-1]])):
                    raise ValueError("Only the last dimension can be inf")
                return    # shape will be resolved in Package.add based on access index

        self._create_native_array()

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return self._name

    @property
    def requested_layout(self):
        return self._requested_layout

    @property
    def shape(self):
        return list(self._shape)

    @property
    def role(self):
        return self._role

    @property
    def element_type(self):
        return self._element_type

    @property
    def flags(self):
        return self._flags

    @property
    def _value(self):
        if self._native_array:
            return self._native_array._value
        else:
            return None

    def _get_native_array(self):
        return self._native_array

    def _init_delayed(self, shape: Tuple[int]):
        self._shape = shape
        self._create_native_array()

    def deferred_layout(self, cache: "accera.Cache"):
        """Specifies the layout for a Role.CONST array
        Args:
            cache: The layout to set
        """
        if cache.target != self:
            raise ValueError("The cache is not created for this array")
        if self._layout != Array.Layout.DEFERRED:
            raise ValueError("Array layout is already set")
        if self._role != Role.CONST:
            raise ValueError("Array role must be Role.CONST")

        self._layout = cache.layout
        self._create_native_array()

    def sub_array(self, offsets: Tuple[int], shape: Tuple[int], strides: Tuple[int] = ()):
        """Gets a sub-view of the Array.

        Similar to numpy.ndarray.view, this does not make a copy of the Array,
        but returns a view into the Array. Therefore, updates to the contents
        of the sub-view will apply to the original Array (and vice-versa).

        Args:
            offsets: The offsets into the Array where the sub-view begins (required only for signature parity)
            shape: The shape of the sub-view
            strides: The stride values for each rank used for creating the sub-view (required only for signature parity)
        """
        return SubArray(self, shape)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        # This likely needs to be context dependent
        return id(self) == id(other)

    def _create_native_array(self):
        self._shape = [x.get_value() if isinstance(x, DelayedParameter) else x for x in self._shape]
        self._size_str = '*'.join([x.name if isinstance(x, Dimension) else str(x) for x in self._shape])
        self._offset = self._offset.get_value() if isinstance(self._offset, DelayedParameter) else self._offset
        self._layout = self._layout.get_value() if isinstance(self._layout, DelayedParameter) else self._layout

        runtime_shape = [k_dynamic_size if isinstance(x, Dimension) else x for x in self._shape]

        mm_layout = MemoryMapLayout(self._layout, runtime_shape, self._offset)
        memory_layout = _MemoryLayout(runtime_shape, order=mm_layout.order)

        if self._role == Role.CONST:
            if self._layout != Array.Layout.DEFERRED:
                if self._element_type == ScalarType.float32:
                    # pass directly as a python buffer because float32's are not native to python
                    self._native_array = NativeArray(buffer=self._data, memory_layout=memory_layout)
                else:
                    data = list(self._data.flatten())
                    self._native_array = NativeArray(data=data, memory_layout=memory_layout)

            # else defer creating native array until layout is set

        else:
            self._native_array = NativeArray(self._element_type, memory_layout)
            self._layout = self._native_array.layout

    def _build_native_context(self, context: NativeLoopNestContext):
        context.function_args.append(self._native_array)

    def _replay_delayed_calls(self):
        '''
        This method is called once per adding function, so it can be called multiple times when  
        multiple functions get added. In order for the functions to be added correctly, we need to make sure all 
        the residual states are cleared between different method calls.

        For example, in Schedule class, we identify that Schedule._index_map can have residual states, so we need to reset self._index_map
        before we replay the delayed methods.

        If there is no residual state between different method calls, no need to reset.
        '''
        for delayed_call in self._delayed_calls:
            params = self._delayed_calls[delayed_call]
            if isinstance(params, Tuple):
                resolved_param_list = []
                for p in params:
                    if isinstance(p, DelayedParameter):
                        resolved_param_list.append(p.get_value())
                    else:
                        resolved_param_list.append(p)
                delayed_call(tuple(resolved_param_list))
            else:
                delayed_call(params.get_value())

    def _allocate(self):
        from .._lang_python._lang import Allocate

        if not self._value.is_empty:
            return    # already contains data

        # Note: we are blowing away the original Value and replacing with a new allocated Value
        self._native_array = NativeArray(Allocate(type=self._element_type, layout=self._layout, flags=self._flags))
        assert (not self._value.is_empty)


class SubArray(Array):
    """Placeholders for subarrays used in function definitions
    In general, there are two use cases for subarrays:
    1. As placeholders in function definitions (handled by this class)

            # define a nest that uses a subarray of a specific layout
            # in this case, the first quadrant of a (32, 32) array
            A = acc.Array(role=acc.Role.INPUT_OUTPUT, shape=(32, 32))
            A0 = A.sub_array(offsets=(0, 0), shape=(A.shape[0]//2, A.shape[1]//2))
            nest = Nest(A0.shape)

            @nest.iteration_logic:
            def _():
                A0[i, j] = 42.0

            # add a function that takes the subarray (A0) as argument
            package.add(nest, args=(A0,), base_name="my_subarray_fn")

    2. For emitting code that creates a subarray view of another array
       (handled by _lang_python._lang.Array.sub_array())

            # emit the function defined above
            my_subarray_fn = package.add(nest, args=(A0,), base_name="my_subarray_fn")

            def main(A):
                # take the subarray view of input A
                A0 = A.sub_array(offsets=(0, 0), shape=(A.shape[0]//2, A.shape[1]//2))
                my_subarray_fn(A0) # call function defined above, passing in the subarray view

            # add a function that receives the full array (A) as argument
            package.add(main, args=(A,), base_name="main")

    """
    def __init__(self, source: Array, shape: Tuple[Union[int, DelayedParameter]], name: str = None):
        self._source = source
        self._role = source.role
        self._name = name if name is not None else (source._name + '_sub')
        self._shape = shape or source.shape
        self._element_type = source.element_type
        self._layout = source._layout
        self._requested_layout = source._requested_layout
        self._offset = 0
        self._delayed_calls = {}
        
        if self._shape and any([isinstance(s, DelayedParameter) for s in self._shape]):
            self._delayed_calls[partial(self._init_delayed)] = tuple([s for s in self._shape])
            return

        self._create_native_array()

    @property
    def name(self):
        return self._name

    def _create_native_array(self):
        # Creates a placeholder NativeArray of the expected layout
        # Note that this is *not* the actual subarray view, but is intended only for
        # defining functions. To get the actual subarray view, call
        # NativeArray.sub_array() on the source NativeArray after it is materialized.
        self._shape = [x.get_value() if isinstance(x, DelayedParameter) else x for x in self._shape]
        self._size_str = '*'.join([x.name if isinstance(x, Dimension) else str(x) for x in self._shape])

        runtime_shape = [k_dynamic_size if isinstance(x, Dimension) else x for x in self._shape]

        self._layout = _MemoryLayout.get_subarray_layout(self._source._layout, runtime_shape)
        self._native_array = NativeArray(self._element_type, self._layout)

