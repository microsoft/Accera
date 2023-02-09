# Runtime-sized implementation of Range
# Prototype only - meant to illustrate future versions of Accera

import accera as acc
import numpy as np

# https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
op_type = "Range"


def generate():

    Start = acc.Scalar()
    Limit = acc.Scalar()
    Delta = acc.Scalar()

    OutputDim = acc.create_dimensions(role=acc.Role.OUTPUT)
    Output = acc.Array(shape=(OutputDim, ), role=acc.Role.OUTPUT)

    OutputDim.value = acc.floor((Limit - Start) / Delta)
    nest = acc.Nest(OutputDim)
    i = nest.get_indices()

    @nest.iteration_logic
    def _():
        Output[i] = Start
        Start += Delta

    # Generate a function like:
    #
    # Range(float start, float limit, float delta, float** output, int64_t* output_dim);
    #
    # This function dynamically allocates memory for the output pointer, which must
    # be released by the caller. The caller must also provide an implementation of:
    #     void _accera_allocate(void** memory, size_t bytes);
    #
    package = acc.Package()
    package.add(nest, args=(Start, Limit, Delta, Output, OutputDim), base_name=f"Range")

    # TODO: package.build, etc to return a hat package
    return package


def generate_numpy():

    def fn(start, limit, delta):
        output_dim = int((limit - start) // delta)
        output = np.zeros(shape=(output_dim, ), dtype=np.float32)

        for i in range(output_dim):
            output[i] = start
            start += delta

        return output

    return fn


def verify():
    # For now we just use numpy loops to verify correctness of the iteration logic
    # without running Accera (until Accera implements runtime sizes)
    # In the future, this may also verify against an ONNX graph

    start = np.float32(1)
    limit = np.float32(5)
    delta = np.float32(2)
    y_ref = np.arange(start, limit, delta, dtype=np.float32)

    # TODO: replace with generate()
    range_fn = generate_numpy()
    y = range_fn(start, limit, delta)

    np.testing.assert_allclose(y_ref, y, atol=1e-5)


def main():
    verify()


if __name__ == "__main__":
    main()