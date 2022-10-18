[//]: # (Project: Accera)
[//]: # (Version: v1.2.11)

# Accera v1.2.11 Reference

## `accera.cast(value, type)`
The `cast` operation converts a value from one `acc.ScalarType` to another.

Accera performs implicit casting between most types. Therefore, this operation should only be used to override the implicit casting behavior documented in [Section 2](<../../Manual/02%20Simple%20Affine%20Loop%20Nests.md>).

Limitation: casting constants may result in truncation.

[comment]: # (MISSING: examples for constant casting that cause unexpected truncation)


## Arguments

argument | description | type/default
--- | --- | ---
`value` | The value to cast |
`type` | The destination type | `acc.ScalarType`

## Returns
The result after casting

## Examples

Casting from float32 to int16:

```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(10, 20))
B = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.int16, shape=(10, 20))

nest = acc.Nest(10, 20)
i, j = nest.get_indices()

@nest.iteration_logic:
def _():
    B[i, j] = acc.cast(A[i, j], acc.ScalarType.int16) # explicit cast to int16
    ...
```

In comparison, casting from int16 to float32 is implicit, which means the `cast` operation can be omitted:

```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.int16, shape=(10, 20))
B = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(10, 20))

nest = acc.Nest(10, 20)
i, j = nest.get_indices()

@nest.iteration_logic:
def _():
    B[i, j] = A[i, j] # implicit cast to float32
    ...
```

Casting a constant to int8:

```python

A = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.int8, shape=(10, 20))

nest = acc.Nest(10, 20)
i, j = nest.get_indices()

@nest.iteration_logic:
def _():
    A[i, j] = acc.cast(10, acc.ScalarType.int8)
    ...

```


<div style="page-break-after: always;"></div>


