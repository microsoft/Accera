[//]: # (Project: Accera)
[//]: # (Version: v1.2.1)

# Section 1: Arrays
Accera stores data in multi-dimensional arrays of scalar elements where all the elements of the array share the same basic data type (e.g., float32, int8). An array has a fixed dimension *d* that is known at compile-time (e.g., a matrix is a 2-dimensional array). Each dimension has a positive size and the sequence of *d* sizes is called the *shape* of the array. An element of an array is referred to by a *d*-coordinate zero-based *index vector*.

## Affine memory layout
Arrays are multi-dimensional while computer memories have a linear (one-dimensional) address space. There are many strategies to map a multi-dimensional array in one-dimensional computer memory. Accera arrays are required to have an affine memory layout where each array has an *affine memory map* that is a *d*-dimensional vector denoted by *a*, and a *memory offset* value denoted by *o*. The array element that corresponds to the index vector *i* is stored at memory address *i&middot; a+o* (where *i&middot; a* denotes a vector dot product).

Affine memory maps are rich enough to represent many standard array layouts. For example, 2-dimensional arrays (matrices) can be represented as *row-major*, *column-major*, *triangular*, *banded*, and *Toeplitz* matrices in affine maps. However, affine maps cannot represent z-ordering or other striped or blocked layouts.

[comment]: # (MISSING: add a mechanism that would support z-order, blocked, and striped arrays. Basically, this is equivalent to adding mod and floordiv operations to the memory map. Alternatively, this could be achieved by somehow adding split and reorder operations to arrays.)

## Array shape
In an affine memory map, each dimension corresponds to an element where the dimension having the largest absolute value of element is called the *major dimension*. The shape of an array is defined at compile-time, with the possible exception of the major dimension size that can remain undefined.

The major dimension is the only one whose size is not necessarily defined at compile time and in this case, Accera assumes that the size is arbitrary (or infinite). In other words, the iterations of the loops determine how much of the array is visited along this dimension. For example, a row-major matrix must have a compile-time-constant number of columns. However, the number of rows can be left undefined and the sizes of the loops control how many rows are processed.

## Default and inferred memory layout
Although the user can explicitly specify the memory map, Accera does offer shortcuts. The user can set the layout to be `FIRST_MAJOR` (e.g., for two-dimensional arrays, first-major is equivalent to row-major) or `LAST_MAJOR`. In both cases, the affine map is inferred from the array shape. Specifically, if the layout is `LAST_MAJOR` and the shape is denoted by the vector *s*, then the map *a* is set to *[1, s0, s0&times;s1, s0&times;s1&times;s2, ...]*. If the layout is `FIRST_MAJOR` and the dimension equals 4, then *a* is set to *[s0&times;s1&times;s2, s1&times;s2, s2, 1]*. Note that, in both cases, the size of the major dimension is not used in the definition of *a*. This indicates why the major dimension size does not need to be defined at compile time. If no layout is specified, the default layout is `FIRST_MAJOR`.

## Array properties
Accera arrays are defined with either *internal scope* or *external scope*. An internal array is a private array that exists inside a specific Accera function only and cannot be accessed outside of that function. An external array is defined outside of an Accera function and passed in as an argument. The memory layout of an external array is specified as a part of the Accera function signature. Moreover, external arrays are assumed to be disjoint, i.e., they do not share any memory with each other.

Accera arrays are either mutable or immutable. The elements of a mutable array can be set by an Accera function, while an immutable array is read-only.

Array properties are not explicitly set by the programmer but are implied by the *role* of the array (see below).

## Array roles
Accera supports the following four array roles where each role is treated differently. 
-	Input
-	Input/Output
-	Constant
-	Temporary 


### Input arrays
Input arrays are *immutable external* arrays whose element type, shape, and affine layout are known at compile time. However, their contents are only available at runtime. If the Accera function is emitted as a function in C, each input array is passed in as a *const* pointer argument. For example, we can construct a 10&times;20 input array of 32-bit floating-point numbers by writing
```python
import accera as acc

A = acc.Array(shape=(10, 20), role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32)
```
The layout of this array would be the default layout, which is `acc.Array.Layout.FIRST_MAJOR`.

### Input/output arrays
Input/Output arrays are similar to the input arrays except for the fact that they are *mutable external* arrays, i.e., their values can be changed. This type of array is used to output the results of the loop-nest computation. If the Accera function is emitted as a function in C, each input array is passed in as a non-const pointer argument.

### Constant arrays
These are the only type of Accera arrays whose contents are known at compile-time. Constant arrays are *immutable internal* arrays whose memory layout can be chosen automatically without any external constraints since they are internally scoped. For example, a constant array can be automatically laid out according to loop-nest’s memory access pattern. The layout of a constant array could even depend on its contents (e.g., its sparsity pattern). 

We must provide the constant array data (the element values) when we construct it. This data can be any Python buffer or a *numpy* array:
```python
import accera as acc
import numpy as np

matrix = np.random.rand(16, 16)
B = acc.Array(role=acc.Array.Role.CONST, data=matrix)
```

### Temporary arrays
Temporary arrays are *mutable internal* arrays that are used when two Accera schedules are fused into one (more on fusing in [Section 4](<04%20Fusing.md>)). The elements of a temporary array are initialized to zeros and used to store intermediate values. Similar to constant arrays, temporary arrays can be laid out arbitrarily. In fact, the Accera compiler can even choose to not store them in physical memory at all. 


<div style="page-break-after: always;"></div>
