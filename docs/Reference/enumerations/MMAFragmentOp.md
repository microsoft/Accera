[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Accera v1.2 Reference
## `accera.MMAFragmentOp`

type | description
--- | ---
`accera.MMAFragmentOp.NONE` | No-op which does not modify the fragment data, i.e. `f(x) = x`.
`accera.MMAFragmentOp.ReLU` | Rectified linear unit activation function ([details](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))), i.e. `f(x) = max(0, x)`.
`accera.MMAFragmentOp.ReLU_NoConditional` | Rectified linear unit activation function which does not generate divergent code, i.e. `f(x) = x * bool(x > 0)`.
`accera.MMAFragmentOp.CLEAR` | Sets the data to constant 0, i.e. `f(x) = 0`.

<div style="page-break-after: always;"></div>
