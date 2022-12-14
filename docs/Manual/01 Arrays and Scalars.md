[//]: # (Project: Accera)
[//]: # (Version: v1.2.13)

# Section 1: Arrays and Scalars

## Arrays
Accera stores data in multi-dimensional arrays of scalar elements where all the array elements share the same primary data type (e.g., float32, int8). An array has a constant number of dimensions *d* known at compile-time (e.g., a matrix is a 2-dimensional array). Each dimension has a positive size, and the sequence of *d* sizes is called the *shape* of the array. An element of an array is referred to by a *d*-coordinate zero-based *index vector*.

### Affine memory layout
Arrays are multi-dimensional, while computer memories have a linear (one-dimensional) address space. There are many strategies to represent a multi-dimensional array in one-dimensional computer memory. Accera arrays must have an affine memory layout, where each array has an *affine memory map* that is a *d*-dimensional vector denoted by *a* and a *memory offset* value denoted by *o*. The array element that corresponds to the index vector *i* is stored at memory address *i&middot; a+o* (where *i&middot; a* denotes a vector dot product).

Affine memory maps are rich enough to represent many standard array layouts. For example, in affine maps, 2-dimensional arrays (matrices) can be represented as *row-major*, *column-major*, *triangular*, *banded*, and *Toeplitz* matrices. However, affine maps cannot represent z-ordering or striped or blocked layouts.

[comment]: # (MISSING: add a mechanism that would support z-order, blocked, and striped arrays. Basically, this is equivalent to adding mod and floordiv operations to the memory map. Alternatively, this could be achieved by somehow adding split and reorder operations to arrays, or an optional functor that does index vector calculations and returns a scalar index)

[comment]: # (Consider adding examples to illustrate affine memory maps and concepts introduced above)

### Array shape
In an affine memory map, each dimension corresponds to an element, where the dimension having the largest absolute value of the element is called the *major dimension*. The user must specify all dimension sizes except for the major dimension when constructing an Array. Accera assumes that the size is arbitrary (or infinite) if the major dimension is not specified. In other words, the iterations of the loops determine how much of the array is visited along this dimension. 

For example, a row-major matrix must have a compile-time-constant number of columns. However, the number of rows can be left undefined, and the loops' sizes control how many rows are processed.

### Compile-time and runtime dimension sizes
The number of dimensions of Accera arrays are known at compile-time. However, the user can choose to specify the sizes of each dimension at compile-time or at runtime. Runtime dimension sizes are only resolved at runtime, typically as inputs to an Accera function.

For example, a function that implements generalized matrix multiply can receive the `M`, `N`, `K` dimension sizes as inputs along with the `M` &times; `N`, `M` &times; `K`, and `N` &times; `K` Arrays.

Furthermore, an Array can have a mixture of compile-time and runtime dimension sizes.

### Default and inferred memory layout
Although the user can explicitly specify the memory map, Accera offers some conveniences. The user can set the layout as `FIRST_MAJOR` (e.g., for two-dimensional arrays, first-major is equivalent to row-major) or `LAST_MAJOR`. In both cases, the affine map is inferred from the array shape. Specifically, if the layout is `LAST_MAJOR` and the shape is denoted by the vector *s*, then the map *a* is set to *[1, s0, s0&times;s1, s0&times;s1&times;s2, ...]*. If the layout is `FIRST_MAJOR` and the dimension equals 4, then *a* is set to *[s0&times;s1&times;s2, s1&times;s2, s2, 1]*. In both cases, the size of the major dimension is not used in the definition of *a*. This indicates that the major dimension size is not needed. If no layout is specified, the default layout is `FIRST_MAJOR`.

### Array properties
Accera arrays are defined with either *internal scope* or *external scope*. An internal array is a private array that exists inside a specific Accera function only and cannot be accessed outside of that function. An external array is defined outside of an Accera function and passed in as an argument. The memory layout of an external array is specified as a part of the Accera function signature. Moreover, external arrays are assumed to be disjoint, i.e., they do not share any memory.

Accera arrays are either mutable or immutable. The elements of a mutable array can be set by an Accera function, while an immutable array is read-only.

Array properties are not explicitly set by the user but are implied by the *role* of the array (see below).

### Array roles
Accera supports the following four array roles where each role is treated differently. 

-   Input
-   Input/Output
-   Output
-   Constant
-   Temporary 

#### Input arrays
Input arrays are *immutable external* arrays whose element type, shape, and affine layout can be known at compile-time. However, their contents are only available at runtime. If the Accera function is emitted as a function in C, each input array is passed as a *const* pointer argument. For example, we can construct a 10&times;20 input array of 32-bit floating-point numbers by writing
```Python
import accera as acc

A = acc.Array(shape=(10, 20), role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32)
```
The layout of this array would be the default layout, which is `acc.Array.Layout.FIRST_MAJOR`.

The shape (and similarly, the layout) of Input arrays can also be set at runtime:

```Python
N = acc.create_dimensions()
A = acc.Array(shape=(N, 20), role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32)
```

#### Input/output arrays
Input/Output arrays are similar to the input arrays except that they are *mutable external* arrays, i.e., their values can be changed. This type of array is used to output the results of the loop-nest computation. If the Accera function is emitted as a function in C, each input array is passed as a non-const pointer argument.

#### Output arrays
Output arrays are *variable-shaped mutable external* arrays whose shapes and affine layout are known at runtime. The key differences with Input/Output arrays are:

* Output arrays are dynamically allocated at runtime. The caller of an Accera function that uses Output arrays will need to implement the `__accera_allocate` function to allocate memory (and also perform the subsequent deallocation).
* Output arrays are uninitialized by default. Accera will produce an error if operators such as `+=` are used on an Output array without prior initialization through assignment.
* For simplicity, output dimensions (`acc.Dimension.Role.OUTPUT`) must be used for specifying an Output array shape or layout (this limitation may be lifted in the future).

Output arrays are useful for operations that adjust the array shape depending on the input values. For example, the Range operation generates variable output sizes based on the start, end, and step inputs:

```Python
import accera as acc

# inputs
Start = acc.Scalar()
End = acc.Scalar()
Step = acc.Scalar()

# compute the variable output size
N = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)
N.value = acc.floor((End - Start) / Step)

# create an Output array with the variable output size
A = acc.Array(shape=(N, ), role=acc.Array.Role.OUTPUT, element_type=acc.ScalarType.float32)
```

The layout of this array is the default layout, which is `acc.Array.Layout.FIRST_MAJOR`.


#### Constant arrays
These are the only Accera arrays whose contents are known at compile-time. Constant arrays are *immutable internal* arrays whose memory layout can be chosen automatically without any external constraints since they are internally scoped. For example, a constant array can be automatically laid out according to the loop nest's memory access pattern. The layout of a constant array could even depend on its contents (e.g., its sparsity pattern). The dimension sizes of a constant array must be known at compile-time.

We must provide the constant array data (the element values) when constructing it. This data can be any Python buffer or a *numpy* array:
```Python
import accera as acc
import numpy as np

matrix = np.random.rand(16, 16)
B = acc.Array(role=acc.Array.Role.CONST, data=matrix)
```

#### Temporary arrays
Temporary arrays are *mutable internal* arrays that are used when two Accera schedules are fused into one (more on fusing in [Section 4](<04%20Fusing.md>)). The elements of a temporary array are initialized to zeros and used to store intermediate values. Similar to constant arrays, temporary arrays can be laid out arbitrarily. In fact, the Accera compiler can even choose not to store them in physical memory at all. 

## Scalars
A scalar represents a single number whose value is mutable and set at runtime. Scalars are useful as input arguments to functions or when computing a single-valued numeric result.

[Section 2](<02%20Simple%20Affine%20Loop%20Nests.md>) lists the operations can be performed on scalars.


<div style="page-break-after: always;"></div>


