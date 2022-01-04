[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Case Study - Unrolled 2D Convolution Grid Search (part 2)

In this part of the case study, we explain an alternative way of using Accera to implement Unrolled Convolution. In this method, we split the unrolled convolution into different steps, and create a separate schedule for each step and finally fuse them together.

As explained earlier, we can divide the unrolled convolution operation into 4 steps:

1. Pack the input
2. Pack the weights
3. Perform Matrix Multiplication
4. Unpack the output

We will explain how to write each one in Accera, and then we will fuse them together, and apply some optimizations to make the implementation more performant

First, we define the input and output tensors.

```python
# Input and Output Arrays
Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                 shape=(input_rows, input_columns, input_channels))
Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                  shape=(kernel_rows, kernel_columns, input_channels, output_filters))
Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
                  shape=(output_rows, output_columns, output_filters))
```

We also define some temporary tensors that would be used in the intermediate steps.
```python
# Temp Intermediate Arrays
packed_input_rows = output_rows*output_columns
packed_input_columns = kernel_rows*kernel_columns*input_channels

packed_weights_rows = kernel_rows*kernel_columns*input_channels
packed_weights_columns = out_filters

PackedInput = acc.Array(role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, \
                shape=(packed_input_rows, packed_input_columns))
PackedWeights = acc.Array(role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, \
                shape=(packed_weights_rows, packed_weights_columns))
PackedOutput = acc.Array(role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, \
                shape=(packed_weights_rows, packed_input_columns))
```

Then, we define some parameters that would be used later while creating the schedule and the action plan
```python
p_j_split_size, p_j_split2_size, p_j_split3_size, p_input_channels_split_size = acc.create_parameters(4)
```

To **Pack the input**, we can express the input packing (im2col) operation in python as:
```python
PackedInput = np.zeros((packed_input_rows, packed_input_columns)).astype(np.float32)

for out_r in range(output_rows):
    for out_c in range(output_columns):
        for in_ch in range(input_channels):
            for k_r in range(kernel_rows):
                for k_c in range(kernel_columns):
                    in_r = in_ch*kernel_rows*kernel_columns+k_r*kernel_rows+k_c
                    in_c = out_r*output_rows+out_c
                    PackedInput[in_r, in_c] = Input[out_r*row_stride+k_r, out_c*column_stride+k_c, in_ch]
```

We can express this in Accera as:
```python
pack_input_nest = acc.Nest(shape=(output_rows, output_columns, input_channels, kernel_rows, kernel_columns))

out_r1, out_c1, in_ch1, k_r1, k_c1 = pack_input_nest.get_indices()

@pack_input_nest.iteration_logic
def _():
    in_r = in_ch1*kernel_rows*kernel_columns+k_r1*kernel_rows+k_c1
    in_c = out_r1*output_rows+out_c1
    PackedInput[in_r, in_c] = Input[out_r1*row_stride+k_r1, out_c1*column_stride+k_c1, in_ch1]

pack_input_schedule = pack_input_nest.create_schedule()

# Add a split along the input channels dimension
in_ch21 = pack_input_schedule.split(in_ch1, p_input_channels_split_size)

# Move the input channels block as the outer-most dimension to enable accessing the input for packing block-by-block.
# Also, move shared dimensions with the Weights (i.e kernel rows and kernel columns) after the input channels block dimension to enable easier fusing later.
pack_input_schedule.reorder(in_ch1, k_r1, k_c1, in_ch21, out_r1, out_c1)
```

To **Pack the weights**, we can express the weights packing (im2col) operation in python as:
```python
PackedWeights = np.zeros((packed_weights_rows, packed_weights_columns)).astype(np.float32)

for out_f in range(out_filters):
    for k_r in range(kernel_rows):
        for k_c in range(kernel_columns):
            for in_ch in range(input_channels):
                PackedWeights[out_f,in_ch*kernel_rows*kernel_columns+k_r*kernel_rows+k_c] = Weights[k_r, k_c, in_ch, out_f]
```

We can express this in Accera as:
```python
pack_weights_nest = acc.Nest(shape=(kernel_rows, kernel_columns, input_channels, output_filters))

k_r2, k_c2, in_ch2, out_f2 = pack_weights_nest.get_indices()

@pack_weights_nest.iteration_logic
def _():
    PackedWeights[out_f2, in_ch2*kernel_rows*kernel_columns+k_r2*kernel_rows+k_c2] = Weights[k_r2, k_c2, in_ch2, out_f2]

pack_weights_schedule = pack_weights_nest.create_schedule()

# Add a split along the input channels dimension
in_ch22 = pack_weights_schedule.split(in_ch2, p_input_channels_split_size)

# Move the input channels block as the outer-most dimension to enable accessing the input for packing block-by-block.
# Also, move shared dimensions with the input (i.e kernel rows and kernel columns) after the input channels block dimension to enable easier fusing later.
pack_weights_schedule.reorder(in_ch2, k_r2, k_c2, in_ch22, out_f2)
```

To **Perform Matrix Multiplication**, we define the matrix multiplication schedule as we did in the *Matrix Multiplication case study*. We can write the matrix multiplication in python as:
```python
for out_f in range(out_filters):
    for out_r in range(output_rows):
        for out_c in range(output_columns):
            Output[out_r, out_c, out_f] = PackedOutput[out_f, out_r*output_rows+out_c]
```

We can express this in Accera as:
```python
matmul_nest = acc.Nest(shape=(packed_weights_rows, packed_input_columns, packed_weights_columns))

i3, j3, k3 = matmul_nest.get_indices()

@matmul_nest.iteration_logic
def _():
    PackedOutput[i3,j3] += PackedWeights[i3,k3] * PackedInput[k3,j3]

matmul_schedule = matmul_nest.create_schedule()
```

Next, we add some tiles and splits to make the implementation more performant. We should remember that:
- The `i` dimension represents `output_rows`&times;`output_columns`.
- The `j` dimension represents `output_filters`.
- The `k` dimension represents `kernel_rows`&times;`kernel_columns`&times;`input_channels`.

First, we want to be able to do the matrix multiplication for a block of the input and a block of the weights sequentially. The common dimension between the two is the `input_channels`. This means that we want to split along the `input_channels` dimension, and set this `input_channels` block dimension as the outer most dimension.

Moreover, if we want to enable partial unpacking (along the input channels dimension) followed by partial matrix multiplication. Then, the value of the `input_channels` block should match the value of the `input_channels` block chosen during the previous steps by splitting on `p_input_channels_split_size`. To achieve that, we will split the `k` dimension by `p_input_channels_split_size`&times;`kernel_rows`&times;`kernel_columns`, so the remaining block value would be equal to splitting the `input_channels` by `p_input_channels_split_size`.

Similarly, to get a whole number from the split along the `i` dimension, we would split it by `output_columns`. Since the `j` dimension represents the `output_filters`, we can split by any value `p_j_split_size`.

```python
# Tile splits to place some blocks of the input and output matrices in the L2 cache
ii3 = matmul_schedule.split(i3, output_columns)
jj3 = matmul_schedule.split(j3, p_j_split_size)
kk3 = matmul_schedule.split(k3, p_input_channels_split_size*kernel_rows*kernel_columns)
```

We can further split the `j` dimension twice for unrolling and vectorization. We can also split the `k` dimension again for unrolling by a value of `kernel_rows`&times;`kernel_columns`.

```python
# Kernel splits
jjj3 = matmul_schedule.split(jj3, p_j_split2_size)
jjjj3 = matmul_schedule.split(jjj3, p_j_split3_size)
kkk3 = matmul_schedule.split(kk3, kernel_rows*kernel_columns)

matmul_schedule.reorder(k3, j3, i3, jj3, kk3, kkk3, ii3, jjj3, jjjj3)
```

Finally to **Unpack the output**, we define the output unpacking (col2im) operation in python as:
```python
for out_f in range(out_filters):
    for out_r in range(output_rows):
        for out_c in range(output_columns):
            Output[out_r, out_c, out_f] = PackedOutput[out_f, out_r*output_rows+out_c]
```
We can express this in Accera as:
```python
output_unpack_nest = acc.Nest(shape=(out_filters, output_rows, output_columns))
out_f4, out_r4, out_c4 = output_unpack_nest.get_indices()

@output_unpack_nest.iteration_logic
def _():
    Output[out_r4, out_c4, out_f4] = PackedOutput[out_f4, out_r4*output_rows+out_c4]

output_unpack_schedule = output_unpack_nest.create_schedule()
```

Now, we can then fuse the four created schedules into one schedule
```python
# Fuse the input packing and the weights packing schedules across the first four dimensions in_ch, k_r, k_c, in_ch2
pack_schedule = acc.fuse((pack_input_schedule, pack_weights_schedule), partial=4)

# Fuse the packing schedule with the matrix multiplication schedule along the first dimension in_ch block
# NOTE: Currently we can not do this because the range doesn't match but it should be possible because the values match
matmul_pack_schedule = acc.fuse((pack_schedule, matmul_schedule), partial=1)

# Fuse the matmul and packing schedule with the output unpacking schedule
unrolled_conv_schedule = acc.fuse((matmul_pack_schedule, output_unpack_schedule), partial=0)

f3,\
in_ch_f2,
f2,\
k_rf1, k_cf1,\
f1,\
out_r1, out_c1,\
out_f2,\
j3, i3, jj3, kk3, kkk3, ii3, jjj3, jjjj3,\
out_f4, out_r4, out_c4 = unrolled_conv_schedule.get_indices()
```

Then, we define an action plan, and we use caching, unrolling, and vectorization to make the implementation more performant.

```python
plan = conv_schedule.create_action_plan()

# Cache input and output arrays
plan.cache(PackedInput, index=ii3)
plan.cache(PackedOutput, index=jj3)
plan.cache(PackedWeights, index=kk3)

# Unroll the non-vectorized kernel loops
plan.unroll(ii3)
plan.unroll(jjj3)

# Vectorize the innermost kernel loop
plan.vectorize(jjjj3)
```

To create a package with this matrix multiplication function, we need to know the values of the parameters `p_j_split_size`, `p_j_split2_size`, `p_j_split3_size`, and `p_input_channels_split_size`. Let's assume for now that we are given those constants, then we could set those parameters, and add the function to the Accera package and build it. We will explain in future tutorials how to get the right values for them.

```python
auxiliary_data = {"j_split_size": p_j_split_size,
                  "j_split2_size": p_j_split2_size,
                  "j_split3_size": p_j_split3_size,
                  "input_channels_split_size": p_input_channels_split_size
                 }

name = f"unrolled_conv2d_{input_rows}_{input_columns}_{input_channels}_{output_filters}"
package = acc.Package()
function = package.add_function(plan,
                    args=(Input, Weights, Output),
                    parameters={
                        p_j_split_size: j_split_size,,
                        p_j_split2_size: j_split2_size,
                        p_j_split3_size: j_split3_size,
                        p_input_channels_split_size: input_channels_split_size
                    },
                    base_name=name,
                    auxiliary=auxiliary_data)

package.build(name, format=acc.Package.Format.HAT, output_dir=name)
```
