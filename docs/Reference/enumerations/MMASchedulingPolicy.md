[//]: # (Project: Accera)
[//]: # (Version: v1.2.9)

# Accera v1.2.9 Reference
## `accera.MMASchedulingPolicy`

type | description
--- | ---
`accera.MMASchedulingPolicy.PASS_ORDER` | Process pass groups (fused passed) sequentially, within each pass group compute all the MFMA blocks. This allocates Accmulator registers required for all the blocks, however it only allocates input (A, B) registers which are only required for the current pass group.
`accera.MMASchedulingPolicy.BLOCK_ORDER` | Process MFMA blocks sequentially, for each block iterate over all the passes. This allocates Accumulator registers required for only 1 block and input (A, B) registers required for the entire pass group currently being processed. In this mode, input data for the same pass group is loaded into registers multiple times, once per block.

<div style="page-break-after: always;"></div>
