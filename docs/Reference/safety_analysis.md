[//]: # (Project: Accera)
[//]: # (Version: v1.2.8)

# Accera v1.2.8 Reference

# Safety Analysis

One of the most important features in Accera is to provide safety guarantees to preserve the underlying logic no matter how we transform the schedule. Not all Accera schedules are safe, but those that are safe are much easier to work with.

## Order-invariant Schedules
Order-invariant schedules are always safe because Accera transformations never remove any iterations. They only change the order of the loop-nest iterations, or add empty iterations in the form of padding when necessary. Recall that a `Nest` represents a simple nest. A simple nest is assumed to be order-invariant, and therefore any schedule created by a call to `create_schedule()` is safe.

## Safety and Fusing
Fusing is another way to create a schedule (see [Section 4 of the Accera manual](<../Manual/04%20Fusing.md>)). Say that we have a sequence of *n* schedules: `schedule0`, `schedule1`, ... and we partially fuse their first *m* dimensions. Namely:
```python
schedule = acc.fuse((schedule0, schedule1, ...), partial=m)
```
At this point, `schedule` is equivalent to sequentially executing the individual schedules. However, is the fused `schedule` safe? In other words, does `schedule` guarantee the preservation of underlying logic, regardless of the applied transformation?

The dimensions of `schedule` fall into three categories:

* *Fusing dimensions*: at first, this category contains a single dimension, the first dimension of `schedule`. However, if this dimension is split, its derived dimensions are added to this category.
* *Fused dimensions*: at first, this category contains the next *m* dimensions of `schedule`. If any of these dimensions are split, the derived dimensions are also added to this category.
* *Unfused dimensions*: all the remaining dimensions.

Note that the individual schedules being fused may have been created by previous fusing operations. The categories above relate to the role of each dimension in the *current* fusing operation.

### Theorem
Imagine that we apply a sequence of transformations to `schedule`, which may derive new dimensions. Derived dimensions belong to the same category as the dimension from which they were derived. Suppose the fusing dimension (and its derived dimensions) precedes all the unfused dimensions. In that case, for any value of the fused dimensions, all the corresponding work from `schedule0` is executed before any of the corresponding work from `schedule1`. Similarly, all the corresponding work from `schedule1` is executed before any of the corresponding work from `schedule2`; and so on.

#### Proof
For simplicity, assume that there is only one fusing dimension, `f`. Also, assume that we've only fused two schedules, `schedule0` and `schedule1`. Note that these simplifying assumptions can easily be relaxed.

Assume that `f` precedes all of the unfused dimensions. Therefore, dimensions that precede `f` are necessarily fused dimensions. Let `U` be a sequence of concrete values for all the fused dimensions, and let `V` denote only those values that correspond to dimensions that precede `f`. The work from `schedule0` that corresponds to the concrete values in `U` is contained in the slice (V, 0, \*, ..., \*). Similarly, the work form `schedule1` that corresponds to the values in `U` is contained in (V, 1, \*, ..., \*). Finally, note that the former slice lexicographically precedes the latter, concluding the proof.

#### An example
To make the theorem less abstract, we demonstrate how it applies to a simple example. Assume that we start with two three-dimensional schedules, `schedule0` and `schedule1`, and we fuse their first two dimensions:
```python
i0, j0, k0 = schedule0.get_indices() # redundant operation, included for clarity
i1, j1, k1 = schedule1.get_indices() # redundant operation, included for clarity
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k0, k1 = schedule.get_indices()
```
Next, say that we transform `schedule` by tiling dimensions `j` and `k0` to reorder the dimensions as follows:
```python
jj, kk0 = schedule.tile({
    j: 4,
    k0: 4
})
schedule.reorder(j, i, f, k0, k1, kk0, jj)
```
Dimensions `i`, `j`, and `jj` are fused dimensions, while `k0`, `kk0`, and `k1` are unfused dimensions. Note that the fusing dimension `f` precedes all of the unfused dimensions, satisfying the theorem's condition. Next, choose concrete values for the fused dimensions, say, `i=4`, `j=3`, and `jj=2`. The work from `schedule0` that corresponds to these values is contained in the slice (3, 4, 0, \*, \*, \*, \*). Similarly, the work from `schedule1` that corresponds to these values is contained in the slice (3, 4, 1, \*, \*, \*, \*). The former slice lexicographically precedes the latter and is therefore executed first.

### Safety
The theorem holds for any schedule, but it does not imply that every schedule is safe. Additional effort is required to prove whether a specific schedule is safe. When performing a `fuse` operation, we must examine the specific circumstances and consider whether the theorem provides a sufficient condition for safety.

<div style="page-break-after: always;"></div>


