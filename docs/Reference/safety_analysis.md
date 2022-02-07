[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Safety Analysis

One of the most important features of the Accera language is that it can provide safety guarantees, which make the programmer's job easier. We say that an Accera schedule is *safe* if its underlying logic is guaranteed not to change, regardless of how we transform it. Not all Accera schedules are safe, but those that are safe are much easier to work with.

First, note that order-invariant schedules are obviously safe. This is because Accera transformations only change the order of the loop-nest iterations, never remove any iterations, and possibly add empty iterations in the form of padding. Recall that a `Nest` represents a simple nest, which is assumed to be order-invariant, and therefore any schedule that was created by a call to `create_schedule()` is safe.

## Safety and Fusing
Another way to create a schedule is via fusing (see [Section 4 of the Accera manual](<../Manual/04%20Fusing.md>)). Say that we have a sequence of *n* schedules: `schedule0`, `schedule1`, ... and we partially fuse their first *m* dimensions. Namely,
```python
schedule = acc.fuse((schedule0, schedule1, ...), partial=m)
```
At this point, `schedule` is equivalent to executing the individual schedules one-by-one. However, is `schedule` safe in the sense defined above? In other words, does `schedule` guarantee that its underlying logic is preserved, regardless of how it is transformed?

The dimensions of `schedule` fall into three categories:

* *Fusing dimensions*: at first, this category contains a single dimension, which is the first dimension of `schedule`. However, if this dimension is split, then its derived dimensions are added to the category.
* *Fused dimensions*: at first, this category contains the next *m* dimensions in `schedule`. If any of these dimensions are split, the derived dimensions are also added to the category.
* *Unfused dimensions*: includes all the remaining dimensions.

Note that the individual schedules being fused may themselves be the result of a previous fusing operation. The categories noted above only relate to the role of each dimensions in the current fusing operation.

### A Theorem
Imagine that we apply a sequence of transformations to `schedule`, which may derive new dimensions. Derived dimensions belong to the same category as the dimension from which they were derived. If the fusing dimension (and all dimensions derived from it) precedes all the unfused dimensions, then for any value of the fused dimensions, all the corresponding work from `schedule0` is executed before any of the corresponding work from `schedule1`; all the corresponding work from `schedule1` is executed before any of the corresponding work from `schedule2`; etc.

#### Proof
For simplicity, assume that there is only one fusing dimension and that its name is `f`. Also for simplicity, assume that we only fused two schedules, `schedule0` and `schedule1`. Note that these simplifying assumptions can easily be relaxed.

Assume that `f` precedes all of the unfused dimensions. Therefore, dimensions that precede `f` are necessarily fused dimensions. Let U be a sequence of concrete values for all the fused dimensions and let V denote only those values that correspond to dimensions that precede `f`. The work from `schedule0` that corresponds to the concrete values in U is contained in the slice (V, 0, \*, ..., \*). Similarly, the work form `schedule1` that corresponds to the values in U is contained in (V, 1, \*, ..., \*). Finally, note that the former slice lexicographically precedes the latter. This concludes the proof.

#### An example
To make the theorem less abstract, we demonstrate how it applies to a simple example. Assume that we start with two schedules, `schedule0` and `schedule1`, both three-dimensional, and we fuse their first two dimensions:
```python
i0, j0, k0 = schedule0.get_indices() # redundant operation, included for clarity
i1, j1, k1 = schedule1.get_indices() # redundant operation, included for clarity
schedule = acc.fuse((schedule0, schedule1), partial=2)
f, i, j, k0, k1 = schedule.get_indices()
```
Next, say that we transform `schedule` by tiling dimensions `j` and `k0` and reordering the dimensions as follows:
```python
jj, kk0 = schedule.tile((j, k0), (4, 4))
schedule.reorder(j, i, f, k0, k1, kk0, jj)
```
Dimensions `i`, `j`, and `jj` are fused dimensions, while `k0`, `kk0`, and `k1` are unfused dimensions. Note that the fusing dimension `f` precedes all of the unfused dimensions, so the condition of the theorem is satisfied. Next, choose concrete values for the fused dimensions, say, `i=4`, `j=3`, and `jj=2`. The work from `schedule0` that corresponds to these values is contained in the slice (3, 4, 0, *, *, *, *), and the work from `schedule1` that corresponds to these values is contained in the slice (3, 4, 1, *, *, *, *). The former slice lexicographically precedes the latter and is therefore executed first.

### Safety
The theorem holds for any schedule, but it does not imply that every schedule is safe. Additional effort is required to prove whether a specific schedule is safe or not. When we perform a `fuse` operation, we must examine the specific circumstances and consider whether the theorem provides a sufficient condition for safety.


<div style="page-break-after: always;"></div>
