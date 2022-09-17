# Runtime-sized implementation of Gather
# Prototype only - meant to illustrate future versions of Accera

import accera as acc
import numpy as np

# https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
op_type = "Gather"


def generate_rank_2(axis: int):

    DataDim0, DataDim1, IndicesDim0, IndicesDim1 = acc.create_dimensions()

    # rank(Output) = rank(Data) + rank(Indices) - 1 = 2 + 2 - 1 = 3
    OutputDim0, OutputDim1, OutputDim2 = acc.create_dimensions(role=acc.Dimension.Role.OUTPUT)

    Data = acc.Array(shape=(DataDim0, DataDim1), role=acc.Array.Role.INPUT)
    Indices = acc.Array(shape=(IndicesDim0, IndicesDim1), role=acc.Array.Role.INPUT, type=acc.ScalarType.index)

    # represents a runtime output-only array (dynamically allocated)
    Output = acc.Array(shape=(OutputDim0, OutputDim1, OutputDim2), role=acc.Array.Role.OUTPUT)

    # derive output dims from input dims
    # Note: negative indices are not supported
    if axis == 0:
        OutputDim0.value = IndicesDim0
        OutputDim1.value = IndicesDim1
        OutputDim2.value = DataDim1

        nest = acc.Nest(OutputDim0, OutputDim1, OutputDim2)
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            Output[i, j, k] = Data[Indices[i, j], k]

    else:
        assert (axis == 1)
        OutputDim0.value = DataDim0
        OutputDim1.value = IndicesDim0
        OutputDim2.value = IndicesDim1

        nest = acc.Nest(OutputDim0, OutputDim1, OutputDim2)
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            Output[i, j, k] = Data[i, Indices[j, k]]

    # Generate a function like:
    #
    # Gather_rank_2_axis_0(float* data, int64_t data_dim0, int64_t data_dim1,
    #    int64_t* indices, int64_t indices_dim0, int64_t indices_dim1,
    #    float** output, int64_t* output_dim0, int64_t* output_dim1, int64_t* output_dim2);
    #
    # This function dynamically allocates memory for the output pointer, which must
    # be released by the caller. The caller must also provide an implementation of:
    #     void _accera_allocate(void** memory, size_t bytes);
    #
    package = acc.Package()
    package.add(
        nest,
        args=(Data, DataDim0, DataDim1, Indices, IndicesDim0, IndicesDim1, Output, OutputDim0, OutputDim1, OutputDim2),
        base_name=f"Gather_rank_2_axis_{axis}"
    )

    # TODO: package.build, etc to return a hat package
    return package


def generate_numpy_rank_2(axis: int):
    if axis == 0:

        def fn(data, indices):
            output_dim0 = indices.shape[0]
            output_dim1 = indices.shape[1]
            output_dim2 = data.shape[1]

            output = np.zeros(shape=(output_dim0, output_dim1, output_dim2), dtype=data.dtype)
            for i in range(output_dim0):
                for j in range(output_dim1):
                    for k in range(output_dim2):
                        output[i, j, k] = data[indices[i, j], k]
            return output

        return fn
    else:
        assert (axis == 1)

        def fn(data, indices):
            output_dim0 = data.shape[0]
            output_dim1 = indices.shape[0]
            output_dim2 = indices.shape[1]

            output = np.zeros(shape=(output_dim0, output_dim1, output_dim2), dtype=data.dtype)
            for i in range(output_dim0):
                for j in range(output_dim1):
                    for k in range(output_dim2):
                        output[i, j, k] = data[i, indices[j, k]]
            return output

        return fn


def verify(axis: int):
    # For now we just use numpy loops to verify correctness of the iteration logic
    # without running Accera (until Accera implements runtime sizes)
    # In the future, this may also verify against an ONNX graph

    data = np.random.randn(5, 4).astype(np.float32)
    indices = np.array([[0, 1], [1, 2]])
    y_ref = np.take(data, indices, axis)

    # TODO: replace with generate_rank_2(axis)
    gather = generate_numpy_rank_2(axis)
    y = gather(data, indices)

    np.testing.assert_allclose(y_ref, y, atol=1e-5)


def main():
    for axis in [0, 1]:
        verify(axis)


if __name__ == "__main__":
    main()