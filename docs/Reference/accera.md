[//]: # (Project: Accera)
[//]: # (Version: v1.2.14)

# Accera v1.2.14 Reference

# Module functions
* [`accera.cast`](functions/cast.md) `(value, type)`
* [`accera.create_dimensions`](functions/create_dimensions.md) `([role])`
* [`accera.create_parameters`](functions/create_parameters.md) `()`
* [`accera.create_parameter_grid`](functions/create_parameter_grid.md) `(parameter_choices[, filter_func, sample, seed])`
* [`accera.fuse`](functions/fuse.md) `(schedules[, partial])`

# Top level enumerations
* [`accera.CacheStrategy](<enumerations/CacheStrategy.md>)
* [`accera.ScalarType`](<enumerations/ScalarType.md>)
* [`accera.MMASchedulingPolicy`](<enumerations/MMASchedulingPolicy.md>)
* [`accera.MMAShape`](<enumerations/MMAShape.md>)

# Classes

## `class accera.Array`
A multidimensional array of scalar elements.

### Constructors
* [`Array`](<classes/Array/Array.md>) `(role[, data, element_type, layout, offset, shape])`

### Enumerations
* [`accera.Array.Layout`](<classes/Array/Layout.md>)
* [`accera.Array.Role`](<classes/Array/Role.md>)

### Methods
* [`deferred_layout`](<classes/Array/deferred_layout.md>) `(layout)`
* [`sub_array`](<classes/Array/sub_array.md>) `(offsets, shape[, strides])`

---

## `class accera.Cache`

A local copy of an `Array` block.

---

## `class accera.Dimension`
A runtime dimension.

### Constructors
* [`Dimension`](<classes/Dimension/Dimension.md>) `([role, value])`

### Enumerations
* [`accera.Dimension.Role`](<classes/Dimension/Role.md>)

---

## `class accera.Index`

An index representing one of the loops in a `Nest` or one of the iteration-space dimensions of a `Schedule` or a `Plan`.

---

## `class accera.Nest`

The logic of a loop nest.

### Constructors
* [`Nest`](<classes/Nest/Nest.md>) `(shape)`

### Methods
* [`iteration_logic`](<classes/Nest/iteration_logic.md>) `(logic)`
* [`create_plan`](<classes/Nest/create_plan.md>) `([target])`
* [`create_schedule`](<classes/Nest/create_schedule.md>) `()`
* [`get_indices`](<classes/Nest/get_indices.md>) `()`

---

## `class accera.Package`

Represents a collection of functions that can be built and emitted for use in client code.

### Constructors
* [`Package`](<classes/Package/Package.md>) `()`

### Enumerations
* [`accera.Package.Format`](<classes/Package/Format.md>)
* [`accera.Package.Mode`](<classes/Package/Mode.md>)
* [`accera.Package.Platform`](<classes/Package/Platform.md>)

### Methods
* [`add_description`](<classes/Package/add_description.md>) `([author, license, other, version])`
* [`add`](<classes/Package/add.md>) `(args, source[, base_name, parameters])`
* [`build`](<classes/Package/build.md>) `(name[, error_path, format, mode, os, tolerance])`

---

## `class accera.Parameter`

A placeholder that can be used instead of concrete values when constructing or calling the methods of a `Nest`, `Schedule`, or `Plan`.

---


## `class accera.Plan`
A scheduled (ordered) loop nest with target-specific implementation details.

### Methods
* [`cache`](<classes/Plan/cache.md>) `(source[, index, trigger_index, layout, level, trigger_level, max_elements,  thrifty, location, double_buffer, double_buffer_location, vectorize])`
* [`bind`](<classes/Plan/bind.md>) `(indices, grid)`
* [`kernelize`](<classes/Plan/kernelize.md>) `(unroll_indices[, vectorize_indices])`
* [`parallelize`](<classes/Plan/parallelize.md>) `(indices[, pin, policy, max_threads])`
* [`tensorize`](<classes/Plan/tensorize.md>) `(indices, mma_shape[, use_static_offsets, num_total_passes, num_fused_passes, scheduling_policy])`
* [`unroll`](<classes/Plan/unroll.md>) `(index)`
* [`vectorize`](<classes/Plan/vectorize.md>) `(index)`

---


## `class accera.Scalar`

A scalar element.

### Constructors
* [`Scalar`](<classes/Scalar/Scalar.md>) `([element_type, value])`


---


## `class accera.Schedule`

A scheduled (ordered) loop nest with no target-specific implementation details.

### Methods
* [`create_plan`](<classes/Schedule/create_plan.md>) `([target])`
* [`pad`](<classes/Schedule/pad.md>) `(index, size)`
* [`reorder`](<classes/Schedule/reorder.md>) `(indices)`
* [`skew`](<classes/Schedule/skew.md>) `(index, reference_index)`
* [`split`](<classes/Schedule/split.md>) `(index, size)`
* [`tile`](<classes/Schedule/tile.md>) `(indices, sizes)`

---

## `class accera.Target`

A target platform for the cross-compiler.

### Constructors
* [`Target`](<classes/Target/Target.md>) `([architecture, cache_lines, cache_sizes, category, extensions, family, frequency_GHz, model, name, num_cores, num_threads, turbo_frequency_GHz])`

### Enumerations
* [`accera.Target.Architecture`](<classes/Target/Architecture.md>)
* [`accera.Target.Category`](<classes/Target/Category.md>)
* [`accera.Target.Models`](<classes/Target/Model.md>)

<div style="page-break-after: always;"></div>


