[//]: # (Project: Accera)
[//]: # (Version: v1.2.10)

# Accera v1.2.10 Reference

## `accera.Schedule.is_valid_loop_order(*order)`
The `is_valid_loop_order` function determines if an order of indices is valid. For a description of valid schedule orders, refer to [reorder](reorder.md).

## Arguments

argument | description | type/default
--- | --- | ---
`*order` | The order of indices to check for validity | variable `Index` arguments

## Examples

Checks if an order is valid:

```python
print(schedule.is_valid_loop_order(k, i, j))
```

Uses this function as part of a parameter filter to determine which permutations of loop order parameters are valid:

```python

P1, P2, P2, P4, P5, loop_order = acc.create_parameters()
schedule.reorder(order=loop_order)

def my_filter(parameters_choice):
    P1, P2, P3, P4, P5, loop_order = parameters_choice

    return P1 > P2 \
        and P3 > P4 \
        and P1 * P5 < P3 \
        and P2 * P5 < P4 \
        and schedule.is_valid_loop_order(loop_order)

 parameters = acc.create_parameter_grid({
        P1: [64, 128, 256],
        P2: [32, 128], 
        P3: [16, 32, 128],
        P4: [8, 64],
        P5: [4],
        loop_order: (i, j, k, ii, jj, kk)
    }, my_filter)
```


<div style="page-break-after: always;"></div>


