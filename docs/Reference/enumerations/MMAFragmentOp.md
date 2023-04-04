[//]: # (Project: Accera)
[//]: # (Version: v1.2)

# Accera v1.2 Reference
## `accera.MMAFragmentOp`

type | description | Mathematical formula
--- | --- | ---
`accera.MMAFragmentOp.NONE` | No-op which does not modify the fragment data. | `f(x) = x`
`accera.MMAFragmentOp.ReLU` | Rectified linear unit activation function ([details](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))). | `f(x) = max(0, x)`
`accera.MMAFragmentOp.ReLU_NoConditional` | Rectified linear unit activation function which does not generate divergent code. | `f(x) = x * bool(x > 0)`
`accera.MMAFragmentOp.SET` | Sets the data to scalar constant, C. | `f(x) = C`
`accera.MMAFragmentOp.SCALE` | Multiplies the data by a scalar constant, C. | `f(x) = C.f(x)`

<div style="page-break-after: always;"></div>
